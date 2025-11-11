"""Replay das_letkf stages using only the global dumps (grid, state, obs)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

from .load_das_letkf import load_das_obs_local_before
from .load_dumps import (
    _normalize_member,
    _normalize_pe_tag,
    _read_binary_array,
    load_grid_info,
)
from .obs_local import (
    ObsLocalIdentifier,
    ObsLocalInputs,
    TargetGrid,
    _load_ctype_metadata,
    _load_obsda_member,
    obs_local,
)
from .params import DA_CONSTANTS, GRID_CONSTANTS, LETKF_CONSTANTS

VAR_NAMES = ["U", "V", "W", "T", "P", "QV", "QC", "QR", "QI", "QS", "QG"]
VAR_INDEX = {name: idx for idx, name in enumerate(VAR_NAMES)}
MEMBER = int(LETKF_CONSTANTS["MEMBER"])


def _call_id_to_indices(call_id: int, nij1: int, nlev: int, var_count: int) -> Tuple[int, int, int]:
    call0 = call_id - 1
    grid_idx, nvar0 = divmod(call0, var_count)
    ilev_idx, ij0 = divmod(grid_idx, nij1)
    return ij0 + 1, ilev_idx + 1, nvar0 + 1


def _load_state_cube(dump_dir: Path, pe_tag: str, member: str) -> np.ndarray:
    path = dump_dir / "gues3d" / f"gues3d_{pe_tag}.{member}.bin"
    if not path.exists():
        raise FileNotFoundError(path)
    return _read_binary_array(path, ">f8")


@dataclass
class ObsLocalBeforeState:
    call_id: int
    ij: int
    ilev: int
    nvar: int
    kind: str
    n2nc: int
    n2n: int
    ri: float
    rj: float
    rlev: float
    rz: float
    search_q0: np.ndarray


@dataclass
class _CallMeta:
    call_id: int
    ij: int
    ilev: int
    nvar: int
    kind: str
    slot_index: int
    n2nc: int
    n2n: int
    ri: float
    rj: float
    rlev: float
    rz: float


class ObsLocalReplay:
    """Stateful helper that replays obs_local calls using global dumps."""

    def __init__(self, dump_dir: Path | str, pe_tag: str, member: str) -> None:
        self.dump_dir = Path(dump_dir)
        self.pe_tag = _normalize_pe_tag(pe_tag)
        self.member = _normalize_member(member)
        rig1, rjg1, _, hgt1 = load_grid_info(self.dump_dir, self.pe_tag, self.member)
        self.rig1 = rig1
        self.rjg1 = rjg1
        self.hgt1 = hgt1
        self.nij1 = rig1.size
        self.nlev = int(GRID_CONSTANTS["nlev"])
        self.nv3d = int(GRID_CONSTANTS["nv3d"])
        self.nv2d = int(GRID_CONSTANTS.get("nv2d", 0))
        self.var_count = self.nv3d + self.nv2d
        self.var_slots = self.nv3d + 1  # reserve an extra slot for 2D fields
        self._state_cube = _load_state_cube(self.dump_dir, self.pe_tag, self.member)
        if self._state_cube.shape[2] < MEMBER:
            raise ValueError("State cube does not contain all ensemble members")
        self.obs_dataset = _load_obsda_member(self.dump_dir, self.pe_tag, self.member)
        (
            self.hori_loc,
            self.vert_loc,
            self.ctype_slices,
            self.ctype_groups,
            self.ctype_meta,
        ) = _load_ctype_metadata(
            self.dump_dir,
            self.pe_tag,
            self.member,
            self.obs_dataset.elm,
            self.obs_dataset.typ,
            self.obs_dataset.count,
        )
        self.nctype = len(self.hori_loc)
        self.search_q0 = np.ones(
            (self.nctype, self.var_slots, self.nij1, self.nlev),
            dtype=np.int16,
        )
        self._next_call = 1
        self._nobstotal = self.obs_dataset.count
        self._last_call_id = None
        self._last_outputs = None
        self._last_meta = None

    @property
    def nobstotal(self) -> int:
        return self._nobstotal

    def peek_before(self, call_id: int) -> ObsLocalBeforeState:
        """Return the obs_local.before variables for ``call_id``."""

        self._advance_to(call_id)
        meta = self._build_call_meta(call_id)
        search_view = self._search_view(meta)
        return ObsLocalBeforeState(
            call_id=call_id,
            ij=meta.ij,
            ilev=meta.ilev,
            nvar=meta.nvar,
            kind=meta.kind,
            n2nc=meta.n2nc,
            n2n=meta.n2n,
            ri=meta.ri,
            rj=meta.rj,
            rlev=meta.rlev,
            rz=meta.rz,
            search_q0=np.array(search_view, copy=True),
        )

    def build_inputs_for(self, call_id: int, *, copy_search: bool = True) -> ObsLocalInputs:
        """Construct ObsLocalInputs (from global dumps only) for ``call_id``."""

        self._advance_to(call_id)
        meta = self._build_call_meta(call_id)
        return self._build_inputs(meta, include_search_copy=copy_search)

    def run_call(self, call_id: int) -> tuple[Dict[str, np.ndarray], _CallMeta]:
        """Execute ``obs_local`` for ``call_id`` and update the replay state."""

        self._advance_to(call_id)
        if self._next_call != call_id:
            raise ValueError(f"Call order mismatch: expected {self._next_call}, got {call_id}")
        meta = self._build_call_meta(call_id)
        inputs = self._build_inputs(meta, include_search_copy=False)
        search_view = self._search_view(meta)
        outputs = obs_local(inputs, search_view)
        self._next_call += 1
        self._last_call_id = call_id
        self._last_outputs = outputs
        self._last_meta = meta
        return outputs, meta

    def last_result(self, call_id: int) -> tuple[Dict[str, np.ndarray], _CallMeta]:
        if self._last_call_id != call_id or self._last_outputs is None:
            raise ValueError(f"No cached result for call {call_id}")
        return self._last_outputs, self._last_meta

    def _advance_to(self, call_id: int) -> None:
        while self._next_call < call_id:
            self.run_call(self._next_call)

    def _build_call_meta(self, call_id: int) -> _CallMeta:
        ij, ilev, nvar_global = _call_id_to_indices(
            call_id,
            self.nij1,
            self.nlev,
            max(1, self.var_count),
        )
        slot_index = min(nvar_global - 1, self.var_slots - 1)
        ri = float(self.rig1[ij - 1])
        rj = float(self.rjg1[ij - 1])
        rlev = self._pressure_mean(ij - 1, ilev - 1)
        rz = float(self.hgt1[ij - 1, ilev - 1])
        kind = "3d" if nvar_global <= self.nv3d else "2d"
        n2nc, n2n = 1, 1  # SC23 uses uniform variable localization
        return _CallMeta(
            call_id=call_id,
            ij=ij,
            ilev=ilev,
            nvar=nvar_global,
            kind=kind,
            slot_index=slot_index,
            n2nc=n2nc,
            n2n=n2n,
            ri=ri,
            rj=rj,
            rlev=rlev,
            rz=rz,
        )

    def _build_inputs(self, meta: _CallMeta, *, include_search_copy: bool) -> ObsLocalInputs:
        target = TargetGrid(
            ri=meta.ri,
            rj=meta.rj,
            rlev=meta.rlev,
            rz=meta.rz,
            nvar=meta.nvar,
        )
        before_meta = {
            "stage": "obs_local",
            "phase": "before",
            "call_id": meta.call_id,
            "ij": meta.ij,
            "ilev": meta.ilev,
            "nvar": meta.nvar,
            "n2nc": meta.n2nc,
            "n2n": meta.n2n,
            "kind": meta.kind,
            "ri": meta.ri,
            "rj": meta.rj,
            "rlev": meta.rlev,
            "rz": meta.rz,
        }
        before_arrays = {}
        if include_search_copy:
            before_arrays["search_q0"] = np.array(self._search_view(meta), copy=True)
        identifier = ObsLocalIdentifier(
            dump_dir=self.dump_dir,
            call_id=meta.call_id,
            pe_tag=self.pe_tag,
            member=self.member,
        )
        return ObsLocalInputs(
            request=identifier,
            target=target,
            obs=self.obs_dataset,
            hori_loc=self.hori_loc,
            vert_loc=self.vert_loc,
            ctype_slices=self.ctype_slices,
            ctype_groups=self.ctype_groups,
            ctype_meta=self.ctype_meta,
            before_meta=before_meta,
            before_arrays=before_arrays,
            after_arrays={},
        )

    def _pressure_mean(self, ij_idx: int, ilev_idx: int) -> float:
        values = self._state_cube[ij_idx, ilev_idx, :MEMBER, VAR_INDEX["P"]]
        return float(np.mean(values, dtype=np.float64))

    def _search_view(self, meta: _CallMeta) -> np.ndarray:
        return self.search_q0[:, meta.slot_index, meta.ij - 1, meta.ilev - 1]


def test_obs_local_before_from_global(
    dump_dir: Path | str,
    pe_tag: str,
    member: str,
    call_ids: Iterable[int],
) -> Dict[int, float]:
    """Compare reconstructed obs_local.before values with the Fortran dumps."""

    replay = ObsLocalReplay(dump_dir, pe_tag, member)
    errors: Dict[int, float] = {}
    for call_id in call_ids:
        derived = replay.peek_before(call_id)
        stage = load_das_obs_local_before(dump_dir, call_id, pe_tag=pe_tag, member=member)
        meta = stage["meta"]
        arrays = stage["data"]
        fields = ["ri", "rj", "rlev", "rz"]
        max_err = 0.0
        for key in fields:
            diff = abs(getattr(derived, key) - float(meta[key]))
            max_err = max(max_err, diff)
        search_match = np.array_equal(derived.search_q0, arrays.get("search_q0"))
        if not search_match:
            raise AssertionError(f"search_q0 mismatch for call {call_id}")
        errors[call_id] = max_err
        replay.run_call(call_id)
    return errors


__all__ = ["ObsLocalReplay", "ObsLocalBeforeState", "test_obs_local_before_from_global"]
