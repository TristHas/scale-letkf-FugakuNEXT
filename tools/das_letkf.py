"""High-level reproduction of the SCALE ``das_letkf`` pipeline using global dumps.

The helpers in this module stitch together the already-validated ``obs_local``,
``letkf_core``, and ``postproc`` Python ports so that an entire LETKF tile can be
re-analysed directly from the global dump tree (``letkf_dump``).  The entrypoint
``run_rank_analysis`` processes call IDs sequentially, updates the background
state in-place, and exposes the resulting per-member analysis arrays so that
they can be compared against the reference ``anal3d`` dumps.

Running the full domain requires ~13 million calls per PE for the SC23 setup and
is therefore expensive.  The function accepts optional ``call_ids``/``max_calls``
arguments so smaller subsets can be validated quickly (as used in the unit test).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping
from tqdm import tqdm
import numpy as np

from .das_replay import ObsLocalReplay, _call_id_to_indices
from .letkf_core import LetkfCoreIdentifier, LetkfCoreInputs, letkf_core
from .postproc import (
    PostprocIdentifier,
    PostprocInputs,
    _load_background_cube,
    _relax_beta,
    _sample_background,
    LETKF_CONSTANTS,
    postproc,
)
from .params import GRID_CONSTANTS

MEMBER = int(LETKF_CONSTANTS["MEMBER"])
NV3D = int(GRID_CONSTANTS["nv3d"])
NV2D = int(GRID_CONSTANTS.get("nv2d", 0))


@dataclass(frozen=True)
class DasLetkfIdentifier:
    dump_dir: Path | str
    pe_tag: str = "pe000000"
    member: str = "mem0001"


@dataclass
class RankAnalysis:
    pe_tag: str
    analysis: np.ndarray  # (nij1, nlev, MEMBER, NV3D)
    member_tags: tuple[str, ...]
    updated_mask: np.ndarray  # (nij1, nlev, NV3D)
    processed_call_count: int
    total_call_count: int

    def member_array(self, tag: str) -> np.ndarray:
        """Return ``analysis`` slice for the requested member tag."""

        try:
            idx = self.member_tags.index(tag)
        except ValueError as exc:  # pragma: no cover - defensive
            raise KeyError(tag) from exc
        return self.analysis[:, :, idx, :]


def run_rank_analysis(
    identifier: DasLetkfIdentifier | Mapping[str, object],
    *,
    call_ids: Iterable[int] | None = None,
    max_calls: int | None = None,
) -> RankAnalysis:
    """Re-run ``das_letkf`` for a single rank using global dumps.

    Parameters
    ----------
    identifier
        Configuration describing the dump directory, PE tag, and member group
        used to build the observation/state metadata.
    call_ids
        Optional iterable of explicit call IDs to evaluate.  When ``None``,
        every call for the rank is processed (this is very expensive).
    max_calls
        Optional hard cap on the number of calls processed from ``call_ids``.
        This is mainly useful for tests where only a prefix needs to be checked.
    """

    ident = _normalize_identifier(identifier)
    dump_dir = Path(ident.dump_dir)
    replay = ObsLocalReplay(dump_dir, ident.pe_tag, ident.member)
    var_count = max(1, replay.var_count)
    total_calls = replay.nij1 * replay.nlev * var_count

    analysis = np.array(_load_background_cube(dump_dir, ident.pe_tag), copy=True)
    if analysis.shape[2] < MEMBER:
        raise ValueError("Background cube does not contain all ensemble members")
    analysis = analysis[:, :, :MEMBER, :NV3D]
    updated_mask = np.zeros((replay.nij1, replay.nlev, NV3D), dtype=bool)

    processed = 0
    call_iter = _iter_call_ids(call_ids, total_calls, max_calls)
    member_tags = tuple(f"mem{idx + 1:04d}" for idx in range(MEMBER))

    for call_id in tqdm(list(call_iter)):
        outputs, meta = replay.run_call(call_id)
        if meta.nvar > NV3D:
            raise NotImplementedError("2-D variable localisation is not supported for SC23")

        lc_inputs = _make_letkf_inputs(ident, replay, meta, outputs)
        core_outputs = letkf_core(lc_inputs)
        pp_inputs = _make_postproc_inputs(dump_dir, ident, meta, lc_inputs, core_outputs)
        post_outputs = postproc(pp_inputs)

        ij = meta.ij - 1
        ilev = meta.ilev - 1
        nvar_idx = meta.nvar - 1
        analysis[ij, ilev, :, nvar_idx] = np.asarray(post_outputs["anal_members"], dtype=np.float64)
        updated_mask[ij, ilev, nvar_idx] = True
        processed += 1

    if processed == 0:
        raise ValueError("No calls were processed; increase max_calls or supply call_ids")

    return RankAnalysis(
        pe_tag=ident.pe_tag,
        analysis=analysis,
        member_tags=member_tags,
        updated_mask=updated_mask,
        processed_call_count=processed,
        total_call_count=total_calls,
    )


def call_id_to_indices(call_id: int, nij1: int, nlev: int, *, var_count: int | None = None) -> tuple[int, int, int]:
    """Helper that exposes the replay grid mapping (1-based)."""

    target_var_count = var_count if var_count is not None else max(1, NV3D + NV2D)
    return _call_id_to_indices(call_id, nij1, nlev, target_var_count)


def _iter_call_ids(
    call_ids: Iterable[int] | None,
    total_calls: int,
    max_calls: int | None,
) -> Iterable[int]:
    if call_ids is None:
        iterator: Iterable[int] = range(1, total_calls + 1)
    else:
        normalized = sorted({int(cid) for cid in call_ids})
        iterator = normalized
    if max_calls is not None:
        iterator = itertools.islice(iterator, max_calls)
    for call_id in iterator:
        if call_id < 1 or call_id > total_calls:
            raise ValueError(f"call_id {call_id} outside valid range 1..{total_calls}")
        yield call_id


def _make_letkf_inputs(
    ident: DasLetkfIdentifier,
    replay: ObsLocalReplay,
    meta,
    outputs: Mapping[str, np.ndarray],
) -> LetkfCoreInputs:
    nobsl = int(outputs["hdxf"].shape[0])
    before_meta = {
        "stage": "letkf_core",
        "phase": "before",
        "kind": meta.kind,
        "call_id": meta.call_id,
        "ij": meta.ij,
        "ilev": meta.ilev,
        "nvar": meta.nvar,
        "n2nc": meta.n2nc,
        "n2n": meta.n2n,
        "nobsl": nobsl,
        "nobstotal": replay.nobstotal,
        "parm_infl": 1.0,
        "rdiag_wloc": "true",
        "infl_update": "false",
        "ri": meta.ri,
        "rj": meta.rj,
        "rlev": meta.rlev,
        "rz": meta.rz,
    }
    before_arrays = {
        "hdxf": np.asarray(outputs["hdxf"], dtype=np.float64),
        "rdiag": np.asarray(outputs["rdiag"], dtype=np.float64),
        "rloc": np.asarray(outputs["rloc"], dtype=np.float64),
        "dep": np.asarray(outputs["dep"], dtype=np.float64),
    }
    identifier = LetkfCoreIdentifier(
        dump_dir=ident.dump_dir,
        call_id=meta.call_id,
        pe_tag=ident.pe_tag,
        member=ident.member,
    )
    return LetkfCoreInputs(
        identifier=identifier,
        before_meta=before_meta,
        before_arrays=before_arrays,
        after_meta={},
        after_arrays={},
    )


def _make_postproc_inputs(
    dump_dir: Path,
    ident: DasLetkfIdentifier,
    meta,
    lc_inputs: LetkfCoreInputs,
    core_outputs: Mapping[str, np.ndarray | float | None],
) -> PostprocInputs:
    ij_idx = meta.ij - 1
    ilev_idx = meta.ilev - 1
    nvar_idx = meta.nvar - 1
    beta = _relax_beta(meta.ri, meta.rj, meta.rz)
    g_mean, g_pert = _sample_background(
        dump_dir,
        ident.pe_tag,
        ij_idx,
        ilev_idx,
        nvar_idx,
    )

    before_meta = {
        "stage": "postproc",
        "phase": "before",
        "kind": meta.kind,
        "call_id": meta.call_id,
        "ij": meta.ij,
        "ilev": meta.ilev,
        "nvar": meta.nvar,
        "n2nc": meta.n2nc,
        "n2n": meta.n2n,
        "beta": beta,
        "parm": float(core_outputs["parm_infl"]),
        "relax_alpha": LETKF_CONSTANTS["RELAX_ALPHA"],
        "relax_alpha_spread": LETKF_CONSTANTS["RELAX_ALPHA_SPREAD"],
        "relax_spread_out": str(LETKF_CONSTANTS["RELAX_SPREAD_OUT"]).lower(),
        "relax_to_inflated_prior": str(LETKF_CONSTANTS["RELAX_ALPHA"] > 0.0).lower(),
        "det_run": str(LETKF_CONSTANTS.get("DET_RUN", False)).lower(),
        "gues_mean": g_mean,
    }

    before_arrays = {
        "trans": np.asarray(core_outputs["trans"], dtype=np.float64),
        "transm": np.asarray(core_outputs["transm"], dtype=np.float64),
        "gues_members": np.asarray(g_pert, dtype=np.float64),
    }

    identifier = PostprocIdentifier(
        dump_dir=ident.dump_dir,
        call_id=meta.call_id,
        pe_tag=ident.pe_tag,
        member=ident.member,
    )
    return PostprocInputs(
        identifier=identifier,
        before_meta=before_meta,
        before_arrays=before_arrays,
        after_meta={},
        after_arrays={},
    )


def _normalize_identifier(identifier: DasLetkfIdentifier | Mapping[str, object]) -> DasLetkfIdentifier:
    if isinstance(identifier, DasLetkfIdentifier):
        return identifier
    dump_dir = identifier.get("dump_dir")
    pe_tag = identifier.get("pe_tag", "pe000000")
    member = identifier.get("member", "mem0001")
    if dump_dir is None:
        raise ValueError("identifier must include dump_dir")
    return DasLetkfIdentifier(dump_dir=dump_dir, pe_tag=str(pe_tag), member=str(member))


__all__ = ["DasLetkfIdentifier", "RankAnalysis", "run_rank_analysis", "call_id_to_indices"]
