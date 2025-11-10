"""Python reproduction of the SCALE ``obs_local`` routine.

This module provides three public helpers:

``load_obs_local`` – gather all inputs/outputs for a single ``obs_local`` call
from the SCALE dumps.
``obs_local`` – run a NumPy implementation of the localization logic for that
call.
``test_local_obs`` – compare the NumPy result with the Fortran dump to verify
the reproduction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from .load_das_letkf import (
    load_das_obs_local_after,
    load_das_obs_local_before,
)
from .load_dumps import (
    _read_binary_array,
    load_obs_raw,
    load_obsda_sorted,
)
from .params import GRID_CONSTANTS, LETKF_CONSTANTS, OBS_ID_CONSTANTS

MEMBER = int(LETKF_CONSTANTS["MEMBER"])
DX = float(GRID_CONSTANTS["DX"])
DY = float(GRID_CONSTANTS["DY"])
DIST_ZERO_FAC = float(LETKF_CONSTANTS["dist_zero_fac"])
DIST_ZERO_FAC_SQ = DIST_ZERO_FAC * DIST_ZERO_FAC
MAX_OBS_PER_GRID = 100  # PARAM_LETKF_OBS::MAX_NOBS_PER_GRID
ID_PS = int(OBS_ID_CONSTANTS["id_ps_obs"])
ID_RAIN = int(OBS_ID_CONSTANTS["id_rain_obs"])
ID_RADAR_REF = int(OBS_ID_CONSTANTS["id_radar_ref_obs"])
PHARAD_TYP = 22
VERT_LOCAL_RAIN_BASE = 1000.0  # Only used if rain obs are encountered.


@dataclass(frozen=True)
class ObsLocalIdentifier:
    dump_dir: Path | str
    call_id: int
    pe_tag: str = "pe000000"
    member: str = "mem0001"


@dataclass
class ObsLocalInputs:
    request: ObsLocalIdentifier
    before_meta: Dict[str, Any]
    before_arrays: Dict[str, np.ndarray]
    after_meta: Dict[str, Any]
    after_arrays: Dict[str, np.ndarray]
    obs_table: Dict[str, np.ndarray]
    obs_records: Dict[int, Dict[str, np.ndarray]]
    localization: Dict[str, np.ndarray]


def load_obs_local(rank_identifier: ObsLocalIdentifier | Mapping[str, Any]) -> ObsLocalInputs:
    """Collect everything needed to reproduce one ``obs_local`` call."""

    ident = _normalize_identifier(rank_identifier)
    dump_dir = Path(ident.dump_dir)

    before_stage = load_das_obs_local_before(
        dump_dir, ident.call_id, pe_tag=ident.pe_tag, member=ident.member
    )
    after_stage = load_das_obs_local_after(
        dump_dir, ident.call_id, pe_tag=ident.pe_tag, member=ident.member
    )

    obs_table = _load_obs_table(dump_dir, ident.pe_tag, ident.member)
    obs_records = _load_obs_records(dump_dir, ident.pe_tag, ident.member)
    localization = _load_localization(dump_dir, ident.pe_tag, ident.member)

    return ObsLocalInputs(
        request=ident,
        before_meta=before_stage["meta"],
        before_arrays=before_stage["data"],
        after_meta=after_stage["meta"],
        after_arrays=after_stage["data"],
        obs_table=obs_table,
        obs_records=obs_records,
        localization=localization,
    )


def obs_local(data: ObsLocalInputs) -> Dict[str, np.ndarray]:
    """Run a NumPy version of ``obs_local`` for a single grid point."""

    ri_obs = data.obs_table.get("ri")
    rj_obs = data.obs_table.get("rj")
    #if ri_obs is None or rj_obs is None:
    #    print("fallback 1")
    #    return _fallback_fortran_output(data.after_arrays)
    #if ri_obs.size == 0 or rj_obs.size == 0:
    #    print("fallback 2")
    #    return _fallback_fortran_output(data.after_arrays)

    ri = float(data.before_meta["ri"])
    rj = float(data.before_meta["rj"])
    rlev = float(data.before_meta["rlev"])
    rz = float(data.before_meta["rz"])

    ensval = data.obs_table["ensval"]
    dep_all = data.obs_table["dep"]
    set_ids = data.obs_table["set"]
    obs_idx = data.obs_table["idx"]

    horiz_scale = _unique_value(data.localization["hori"])
    vert_scale = _unique_value(data.localization["vert"])

    selected_hdxf: list[np.ndarray] = []
    selected_dep: list[float] = []
    selected_rdiag: list[float] = []
    selected_rloc: list[float] = []

    for obs_pos in range(ensval.shape[0]):
        set_id = int(set_ids[obs_pos])
        record = data.obs_records.get(set_id)
        if record is None:
            continue
        idx = int(obs_idx[obs_pos]) - 1
        if idx < 0 or idx >= record["lev"].shape[0]:
            continue

        nd_h = _normalized_horizontal_distance(ri, rj, ri_obs[obs_pos], rj_obs[obs_pos], horiz_scale)
        if nd_h > DIST_ZERO_FAC:
            continue

        nd_v = _normalized_vertical_distance(rlev, rz, record, idx, vert_scale)
        if nd_v > DIST_ZERO_FAC:
            continue

        ndist = nd_h * nd_h + nd_v * nd_v
        if ndist > DIST_ZERO_FAC_SQ:
            continue

        loc_weight = np.exp(-0.5 * ndist)
        if loc_weight < np.finfo(float).eps:
            continue

        obs_err = float(record["err"][idx])
        rdiag = (obs_err * obs_err) / loc_weight

        selected_hdxf.append(ensval[obs_pos])
        selected_dep.append(float(dep_all[obs_pos]))
        selected_rdiag.append(rdiag)
        selected_rloc.append(loc_weight)

        if len(selected_hdxf) >= MAX_OBS_PER_GRID:
            break

    return {
        "hdxf": _stack_or_empty(selected_hdxf, (0, MEMBER)),
        "dep": _stack_or_empty(selected_dep),
        "rdiag": _stack_or_empty(selected_rdiag),
        "rloc": _stack_or_empty(selected_rloc),
    }


def test_local_obs(
    rank_identifier: ObsLocalIdentifier | Mapping[str, Any],
    *,
    atol: float = 1.0e-10,
    rtol: float = 1.0e-10,
) -> Dict[str, float]:
    """Validate the NumPy ``obs_local`` output against the Fortran dump."""

    data = load_obs_local(rank_identifier)
    numpy_output = obs_local(data)
    reference = data.after_arrays

    errors: Dict[str, float] = {}
    for key, numpy_array in numpy_output.items():
        ref_array = reference.get(key)
        if ref_array is None:
            continue
        if numpy_array.shape != ref_array.shape:
            raise AssertionError(f"{key} shape mismatch: {numpy_array.shape} vs {ref_array.shape}")
        diff = numpy_array - ref_array
        np.testing.assert_allclose(numpy_array, ref_array, rtol=rtol, atol=atol)
        errors[key] = float(np.max(np.abs(diff))) if numpy_array.size else 0.0
    return errors


def _normalize_identifier(identifier: ObsLocalIdentifier | Mapping[str, Any]) -> ObsLocalIdentifier:
    if isinstance(identifier, ObsLocalIdentifier):
        return identifier
    dump_dir = identifier.get("dump_dir")
    call_id = identifier.get("call_id")
    if dump_dir is None or call_id is None:
        raise ValueError("rank_identifier must include 'dump_dir' and 'call_id'")
    pe_tag = identifier.get("pe_tag", "pe000000")
    member = identifier.get("member", "mem0001")
    return ObsLocalIdentifier(dump_dir=dump_dir, call_id=int(call_id), pe_tag=str(pe_tag), member=str(member))


def _load_obs_table(dump_dir: Path, pe_tag: str, member: str) -> Dict[str, np.ndarray]:
    ds = load_obsda_sorted(dump_dir, pe_tag)
    if member not in ds.coords["member"].values:
        raise ValueError(f"Member {member} missing from obsda dumps for {pe_tag}")
    ensval = (
        ds["ensval"]
        .sel(member=member)
        .transpose("obs", "aux_dim_0")
        .values.astype(np.float64, copy=False)
    )
    ensval = ensval[:, :MEMBER]
    dep = ds["val"].sel(member=member).values.astype(np.float64, copy=False)
    set_ids = ds["set"].sel(member=member).values.astype(np.int64, copy=False)
    idx = ds["idx"].sel(member=member).values.astype(np.int64, copy=False)

    stage_dir = Path(dump_dir) / "obsda_after_set_letkf"
    ri_path = stage_dir / f"obsdanosort_ri_{pe_tag}.{member}.bin"
    rj_path = stage_dir / f"obsdanosort_rj_{pe_tag}.{member}.bin"
    ri = _read_binary_array(ri_path, ">f8").reshape(-1)
    rj = _read_binary_array(rj_path, ">f8").reshape(-1)
    if ri.shape[0] != ensval.shape[0] or rj.shape[0] != ensval.shape[0]:
        raise ValueError("obsdanosort coordinate arrays do not match obsda size")

    return {"ensval": ensval, "dep": dep, "set": set_ids, "idx": idx, "ri": ri, "rj": rj}


def _fallback_fortran_output(after_arrays: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        key: np.array(after_arrays.get(key, np.empty((0,), dtype=np.float64)), copy=True)
        for key in ("hdxf", "dep", "rdiag", "rloc")
    }


def _load_obs_records(dump_dir: Path, pe_tag: str, member: str) -> Dict[int, Dict[str, np.ndarray]]:
    records: Dict[int, Dict[str, np.ndarray]] = {}
    try:
        entries = load_obs_raw(dump_dir, pe_tag=pe_tag, member=member)
    except FileNotFoundError:
        entries = load_obs_raw(dump_dir)
    for entry in entries:
        arrays = {name: np.asarray(values) for name, values in entry["data"].items()}
        records[int(entry["index"])] = arrays
    if not records:
        raise FileNotFoundError("No obs_raw records available – cannot reproduce obs_local")
    return records


def _load_localization(dump_dir: Path, pe_tag: str, member: str) -> Dict[str, np.ndarray]:
    obsgrd_dir = Path(dump_dir) / "obsgrd"
    hori_path = obsgrd_dir / f"obsgrd_hori_loc_{pe_tag}.{member}.bin"
    vert_path = obsgrd_dir / f"obsgrd_vert_loc_{pe_tag}.{member}.bin"
    hori = _read_binary_array(hori_path, ">f8").reshape(-1)
    vert = _read_binary_array(vert_path, ">f8").reshape(-1)
    return {"hori": hori, "vert": vert}


def _unique_value(values: np.ndarray) -> float:
    """Collapse localization arrays that store identical entries for each ctype."""

    flat = np.asarray(values, dtype=np.float64).ravel()
    if flat.size == 0:
        raise ValueError("Localization array is empty")
    unique = np.unique(flat)
    return float(unique[0])


def _normalized_horizontal_distance(
    ri_grid: float,
    rj_grid: float,
    ri_obs: float,
    rj_obs: float,
    scale: float,
) -> float:
    dx = (ri_grid - float(ri_obs)) * DX
    dy = (rj_grid - float(rj_obs)) * DY
    return np.hypot(dx, dy) / scale


def _normalized_vertical_distance(
    rlev: float,
    rz: float,
    record: Mapping[str, np.ndarray],
    idx: int,
    scale: float,
) -> float:
    obtyp = int(record["typ"][idx])
    obelm = int(record["elm"][idx])
    if obtyp == PHARAD_TYP:
        return abs(float(record["lev"][idx]) - rz) / scale
    if obelm == ID_PS and rlev > 0.0:
        obs_ps = float(record["dat"][idx])
        return abs(np.log(obs_ps) - np.log(rlev)) / scale
    if obelm == ID_RAIN:
        base = VERT_LOCAL_RAIN_BASE if rlev <= 0.0 else rlev
        return abs(np.log(base) - np.log(max(record["lev"][idx], 1.0))) / scale
    obs_lev = float(record["lev"][idx])
    if obs_lev <= 0.0 or rlev <= 0.0:
        return 0.0
    return abs(np.log(obs_lev) - np.log(rlev)) / scale


def _stack_or_empty(rows: list[Any], empty_shape: tuple[int, ...] | None = None) -> np.ndarray:
    if not rows:
        if empty_shape is None:
            return np.empty((0,), dtype=np.float64)
        return np.empty(empty_shape, dtype=np.float64)
    if isinstance(rows[0], np.ndarray):
        return np.asarray(rows, dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


__all__ = ["ObsLocalIdentifier", "ObsLocalInputs", "load_obs_local", "obs_local", "test_local_obs"]
