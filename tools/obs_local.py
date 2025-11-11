"""Python reproduction of the SCALE ``obs_local`` routine for test/SC23.

The implementation below mirrors the behavior of ``letkf_tools::obs_local``
for the SC23 experiment: two ensemble members, radar observations only,
distance-based observation capping, and no deterministic bookkeeping.
It operates directly on the LETKF dump files so that it can be validated
against the Fortran results contained in ``das_letkf/obs_local/after``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Sequence

import numpy as np

from .load_das_letkf import (
    load_das_obs_local_after,
    load_das_obs_local_before,
)
from .load_dumps import (
    _read_binary_array,
    _read_text_metadata,
    load_obs_raw,
    load_obsda_sorted,
)
from .params import DA_CONSTANTS, GRID_CONSTANTS, LETKF_CONSTANTS, OBS_ID_CONSTANTS

MEMBER = int(LETKF_CONSTANTS["MEMBER"])
DX = float(GRID_CONSTANTS["DX"])
DY = float(GRID_CONSTANTS["DY"])
DIST_ZERO_FAC = float(LETKF_CONSTANTS["dist_zero_fac"])
DIST_ZERO_FAC_SQ = DIST_ZERO_FAC * DIST_ZERO_FAC
MAX_OBS_PER_GRID = 100  # PARAM_LETKF_OBS::MAX_NOBS_PER_GRID
ID_PS = int(OBS_ID_CONSTANTS["id_ps_obs"])
ID_RAIN = int(OBS_ID_CONSTANTS["id_rain_obs"])
PHARAD_TYP = 22
VERT_LOCAL_RAIN_BASE = 1000.0
QUICKSELECT_EPS = np.finfo(np.float64).eps
PRC_NUM_X = 4
PRC_NUM_Y = 5
IHALO = int(DA_CONSTANTS["IHALO"])
JHALO = int(DA_CONSTANTS["JHALO"])
N_SEARCH_INCR = 8


@dataclass(frozen=True)
class ObsLocalIdentifier:
    dump_dir: Path | str
    call_id: int
    pe_tag: str = "pe000000"
    member: str = "mem0001"


@dataclass
class TargetGrid:
    ri: float
    rj: float
    rlev: float
    rz: float
    nvar: int


@dataclass
class ObsDataset:
    ensval: np.ndarray  # (nobs, MEMBER)
    dep: np.ndarray  # (nobs,)
    set_ids: np.ndarray  # (nobs,)
    obs_indices: np.ndarray  # (nobs,) zero-based into obs_raw
    ri: np.ndarray  # (nobs,)
    rj: np.ndarray  # (nobs,)
    lev: np.ndarray  # (nobs,)
    dat: np.ndarray  # (nobs,)
    err: np.ndarray  # (nobs,)
    elm: np.ndarray  # (nobs,)
    typ: np.ndarray  # (nobs,)

    @property
    def count(self) -> int:
        return self.ensval.shape[0]


@dataclass
class CTypeGroup:
    master: int
    ctypes: List[int]


@dataclass
class ObsGridInfo:
    ngrd_i: int
    ngrd_j: int
    ngrdsch_i: int
    ngrdsch_j: int
    ngrdext_i: int
    ngrdext_j: int
    grdspc_i: float
    grdspc_j: float
    ac_ext: np.ndarray
    tot_ext: int


@dataclass
class ObsLocalInputs:
    request: ObsLocalIdentifier
    target: TargetGrid
    obs: ObsDataset
    hori_loc: np.ndarray
    vert_loc: np.ndarray
    ctype_slices: List[tuple[int, int]]
    ctype_groups: List[CTypeGroup]
    ctype_meta: List[ObsGridInfo]
    before_meta: Dict[str, Any]
    before_arrays: Dict[str, np.ndarray]
    after_arrays: Dict[str, np.ndarray]


def load_obs_local(rank_identifier: ObsLocalIdentifier | Mapping[str, Any]) -> ObsLocalInputs:
    """Collect all data required to reproduce a single ``obs_local`` call."""

    ident = _normalize_identifier(rank_identifier)
    dump_dir = Path(ident.dump_dir)

    before_stage = load_das_obs_local_before(
        dump_dir, ident.call_id, pe_tag=ident.pe_tag, member=ident.member
    )
    after_stage = load_das_obs_local_after(
        dump_dir, ident.call_id, pe_tag=ident.pe_tag, member=ident.member
    )

    target = TargetGrid(
        ri=float(before_stage["meta"]["ri"]),
        rj=float(before_stage["meta"]["rj"]),
        rlev=float(before_stage["meta"]["rlev"]),
        rz=float(before_stage["meta"]["rz"]),
        nvar=int(before_stage["meta"]["nvar"]),
    )

    obs = _load_obsda_member(dump_dir, ident.pe_tag, ident.member)
    hori_loc, vert_loc, ctype_slices, ctype_groups, ctype_meta = _load_ctype_metadata(
        dump_dir, ident.pe_tag, ident.member, obs.elm, obs.typ, obs.count
    )

    return ObsLocalInputs(
        request=ident,
        target=target,
        obs=obs,
        hori_loc=hori_loc,
        vert_loc=vert_loc,
        ctype_slices=ctype_slices,
        ctype_groups=ctype_groups,
        ctype_meta=ctype_meta,
        before_meta=before_stage["meta"],
        before_arrays=before_stage["data"],
        after_arrays=after_stage["data"],
    )


def obs_local(
    data: ObsLocalInputs,
    search_q0: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    """Run a NumPy version of ``obs_local`` for the SC23 configuration.

    If ``search_q0`` is provided, it must be a 1-D array of length ``nctype``
    representing the incremental-search state for this grid/variable. The
    array is updated in place following the Fortran logic.
    """

    obs = data.obs
    nobs = obs.count
    dist_tmp = np.full(nobs, np.inf, dtype=np.float64)
    rloc_tmp = np.full(nobs, -1.0, dtype=np.float64)
    rdiag_tmp = np.zeros(nobs, dtype=np.float64)

    selected_hdxf: list[np.ndarray] = []
    selected_dep: list[float] = []
    selected_rdiag: list[float] = []
    selected_rloc: list[float] = []
    rank_coords = _rank_coordinates(data.request.pe_tag)

    if search_q0 is not None:
        search_vec = np.asarray(search_q0, dtype=np.int64)
        if search_vec.ndim != 1 or search_vec.shape[0] != len(data.ctype_meta):
            raise ValueError("search_q0 must be a 1-D array of length nctype")
    else:
        search_vec = None

    for group in data.ctype_groups:
        q_init = int(search_vec[group.master]) if search_vec is not None else 1
        kept_indices, q_final, nobsl_incr = _select_group_observations(
            data,
            group,
            dist_tmp,
            rloc_tmp,
            rdiag_tmp,
            rank_coords,
            q_init,
        )
        if search_vec is not None:
            if nobsl_incr == 0:
                pass
            else:
                if q_final == q_init and nobsl_incr > MAX_OBS_PER_GRID * 3:
                    search_vec[group.master] = max(q_final - 1, 1)
                elif q_final > q_init:
                    search_vec[group.master] = q_final
        if not kept_indices:
            continue
        for iob in kept_indices:
            selected_hdxf.append(obs.ensval[iob])
            selected_dep.append(float(obs.dep[iob]))
            selected_rdiag.append(float(rdiag_tmp[iob]))
            selected_rloc.append(float(rloc_tmp[iob]))

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

    inputs = load_obs_local(rank_identifier)
    before_arrays = inputs.before_arrays
    search_vec = None
    if before_arrays and "search_q0" in before_arrays:
        search_vec = np.array(before_arrays["search_q0"], dtype=np.int64, copy=True)
    numpy_output = obs_local(inputs, search_vec)
    reference = inputs.after_arrays

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
    if search_vec is not None and reference.get("search_q0") is not None:
        after_search = np.asarray(reference["search_q0"], dtype=np.int64)
        if not np.array_equal(search_vec, after_search):
            raise AssertionError("search_q0 mismatch: NumPy vs Fortran dump")
    return errors


def test_local_obs_from_global(
    rank_identifier: ObsLocalIdentifier | Mapping[str, Any],
    *,
    atol: float = 1.0e-10,
    rtol: float = 1.0e-10,
) -> Dict[str, float]:
    """Validate the global-replay loader by comparing to the Fortran dump."""

    inputs = load_obs_local_from_global(rank_identifier)
    before_arrays = inputs.before_arrays
    search_vec = None
    if before_arrays and "search_q0" in before_arrays:
        search_vec = np.array(before_arrays["search_q0"], dtype=np.int64, copy=True)
    numpy_output = obs_local(inputs, search_vec)
    reference = inputs.after_arrays

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
    if search_vec is not None and reference.get("search_q0") is not None:
        after_search = np.asarray(reference["search_q0"], dtype=np.int64)
        if not np.array_equal(search_vec, after_search):
            raise AssertionError("search_q0 mismatch: NumPy vs Fortran dump")
    return errors


def _load_obsda_member(
    dump_dir: Path,
    pe_tag: str,
    member: str,
) -> ObsDataset:
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
    obs_indices = ds["idx"].sel(member=member).values.astype(np.int64, copy=False) - 1

    ri_obs, rj_obs = _load_obs_coordinates(dump_dir, pe_tag, member)
    if ri_obs.shape[0] != ensval.shape[0]:
        raise ValueError("obsdanosort coordinate arrays do not match obsda size")

    records = _load_obs_records(dump_dir, pe_tag, member)
    lev = _gather_obs_field(set_ids, obs_indices, records, "lev", np.float64)
    dat = _gather_obs_field(set_ids, obs_indices, records, "dat", np.float64)
    err = _gather_obs_field(set_ids, obs_indices, records, "err", np.float64)
    elm = _gather_obs_field(set_ids, obs_indices, records, "elm", np.int64)
    typ = _gather_obs_field(set_ids, obs_indices, records, "typ", np.int64)

    return ObsDataset(
        ensval=ensval,
        dep=dep,
        set_ids=set_ids,
        obs_indices=obs_indices,
        ri=ri_obs,
        rj=rj_obs,
        lev=lev,
        dat=dat,
        err=err,
        elm=elm,
        typ=typ,
    )


def _load_obs_coordinates(
    dump_dir: Path,
    pe_tag: str,
    member: str,
) -> tuple[np.ndarray, np.ndarray]:
    stage_dir = dump_dir / "obsda_after_set_letkf"
    ri_path = stage_dir / f"obsdanosort_ri_{pe_tag}.{member}.bin"
    rj_path = stage_dir / f"obsdanosort_rj_{pe_tag}.{member}.bin"
    ri = _read_binary_array(ri_path, ">f8").reshape(-1).astype(np.float64, copy=False)
    rj = _read_binary_array(rj_path, ">f8").reshape(-1).astype(np.float64, copy=False)
    return ri, rj


def _load_obs_records(
    dump_dir: Path,
    pe_tag: str,
    member: str,
) -> Dict[int, Dict[str, np.ndarray]]:
    try:
        entries = load_obs_raw(dump_dir, pe_tag=pe_tag, member=member)
    except FileNotFoundError:
        entries = load_obs_raw(dump_dir)
    records: Dict[int, Dict[str, np.ndarray]] = {}
    for entry in entries:
        arrays = {name: np.asarray(values) for name, values in entry["data"].items()}
        records[int(entry["index"])] = arrays
    if not records:
        raise FileNotFoundError("No obs_raw records available – cannot reproduce obs_local")
    return records


def _gather_obs_field(
    set_ids: np.ndarray,
    obs_indices: np.ndarray,
    records: Mapping[int, Mapping[str, np.ndarray]],
    field: str,
    dtype: np.dtype[Any],
) -> np.ndarray:
    values = np.empty_like(obs_indices, dtype=dtype)
    for set_id in np.unique(set_ids):
        mask = set_ids == set_id
        record = records.get(int(set_id))
        if record is None:
            raise KeyError(f"Observation set {set_id} missing from obs_raw dumps")
        data = record[field]
        values[mask] = data[obs_indices[mask]]
    return values


def _load_ctype_metadata(
    dump_dir: Path,
    pe_tag: str,
    member: str,
    elm: np.ndarray,
    typ: np.ndarray,
    nobstotal: int,
) -> tuple[np.ndarray, np.ndarray, List[tuple[int, int]], List[CTypeGroup], List[ObsGridInfo]]:
    obsgrd_dir = dump_dir / "obsgrd"
    hori = _read_binary_array(obsgrd_dir / f"obsgrd_hori_loc_{pe_tag}.{member}.bin", ">f8").reshape(-1)
    vert = _read_binary_array(obsgrd_dir / f"obsgrd_vert_loc_{pe_tag}.{member}.bin", ">f8").reshape(-1)
    nctype = hori.shape[0]

    ctype_slices: List[tuple[int, int]] = []
    ctype_meta: List[ObsGridInfo] = []
    start = 0
    prev_total = 0
    for ic in range(1, nctype + 1):
        type_dir = obsgrd_dir / f"type_{ic:05d}"
        meta = _read_text_metadata(type_dir / f"obsgrd_meta_{pe_tag}.{member}.txt")
        ac_ext = _read_binary_array(type_dir / f"type{ic:04d}_ac_ext_{pe_tag}.{member}.bin", ">i4")
        total = int(ac_ext[-1, -1])
        ctype_slices.append((start, total))
        info = ObsGridInfo(
            ngrd_i=int(meta["ngrd_i"]),
            ngrd_j=int(meta["ngrd_j"]),
            ngrdsch_i=int(meta["ngrdsch_i"]),
            ngrdsch_j=int(meta["ngrdsch_j"]),
            ngrdext_i=int(meta["ngrdext_i"]),
            ngrdext_j=int(meta["ngrdext_j"]),
            grdspc_i=float(meta["grdspc_i"]),
            grdspc_j=float(meta["grdspc_j"]),
            ac_ext=ac_ext,
            tot_ext=total - prev_total,
        )
        ctype_meta.append(info)
        prev_total = total
        start = total
    if start != nobstotal:
        raise ValueError(f"ctype metadata mismatch: {start} != {nobstotal}")

    ctype_groups = _infer_ctype_groups(ctype_slices, elm, typ)

    return (
        hori.astype(np.float64),
        vert.astype(np.float64),
        ctype_slices,
        ctype_groups,
        ctype_meta,
    )


def _infer_ctype_groups(
    ctype_slices: Sequence[tuple[int, int]],
    elm: np.ndarray,
    typ: np.ndarray,
) -> List[CTypeGroup]:
    groups: List[CTypeGroup] = []
    used: set[int] = set()
    for ic, (start, end) in enumerate(ctype_slices):
        if ic in used or start >= end:
            continue
        key = (int(elm[start]), int(typ[start]))
        members = [ic]
        used.add(ic)
        for jc in range(ic + 1, len(ctype_slices)):
            if jc in used:
                continue
            s2, e2 = ctype_slices[jc]
            if s2 >= e2:
                continue
            if int(elm[s2]) == key[0] and int(typ[s2]) == key[1]:
                members.append(jc)
                used.add(jc)
        groups.append(CTypeGroup(master=ic, ctypes=members))
    return groups


def _obs_local_cal(
    target: TargetGrid,
    obs: ObsDataset,
    iob: int,
    ctype_idx: int,
    hori_loc: np.ndarray,
    vert_loc: np.ndarray,
) -> tuple[float, float, float]:
    nvar = target.nvar
    obelm = int(obs.elm[iob])
    obtyp = int(obs.typ[iob])
    nrloc = 1.0

    if nvar > 0 and nrloc < QUICKSELECT_EPS:
        return np.inf, 0.0, 0.0

    vert_scale = float(vert_loc[ctype_idx])
    if vert_scale == 0.0:
        nd_v = 0.0
    elif obelm == ID_PS and target.rlev > 0.0:
        nd_v = abs(np.log(obs.dat[iob]) - np.log(target.rlev)) / vert_scale
    elif obelm == ID_RAIN:
        base = VERT_LOCAL_RAIN_BASE if target.rlev <= 0.0 else target.rlev
        nd_v = abs(np.log(base) - np.log(max(obs.lev[iob], 1.0))) / vert_scale
    elif obtyp == PHARAD_TYP:
        nd_v = abs(obs.lev[iob] - target.rz) / vert_scale
    else:
        obs_lev = obs.lev[iob]
        if obs_lev <= 0.0 or target.rlev <= 0.0:
            nd_v = 0.0
        else:
            nd_v = abs(np.log(obs_lev) - np.log(target.rlev)) / vert_scale
    if nd_v > DIST_ZERO_FAC:
        return np.inf, 0.0, 0.0

    dx = (target.ri - obs.ri[iob]) * DX
    dy = (target.rj - obs.rj[iob]) * DY
    nd_h = np.hypot(dx, dy) / float(hori_loc[ctype_idx])
    if nd_h > DIST_ZERO_FAC:
        return np.inf, 0.0, 0.0

    ndist = nd_h * nd_h + nd_v * nd_v
    if ndist > DIST_ZERO_FAC_SQ:
        return np.inf, 0.0, 0.0

    loc_weight = nrloc * np.exp(-0.5 * ndist)
    if loc_weight <= QUICKSELECT_EPS:
        return np.inf, 0.0, 0.0

    err = obs.err[iob]
    rdiag = (err * err) / loc_weight
    return ndist, loc_weight, rdiag


def _quickselect_arg(distances: np.ndarray, indices: List[int], limit: int) -> None:
    """Replicate ``common_sort::QUICKSELECT_arg`` in Python (1-based version)."""

    if limit <= 0 or limit >= len(indices):
        return

    work = [None] + indices  # 1-based for faithful translation
    _quickselect_recursive(distances, work, 1, len(indices), limit)
    indices[:] = work[1:]


def _quickselect_recursive(
    distances: np.ndarray,
    indices: List[int | None],
    left: int,
    right: int,
    k: int,
) -> None:
    if left >= right:
        return
    if (right - left) // k >= 2:
        pivot = _sample_second_min_arg(distances, indices, left, right, k)
    else:
        middle = (left + right) // 2
        pivot = _median_of_three_arg(distances, indices, left, middle, right)
    pivot = _partition_arg(distances, indices, left, right, pivot)
    if k < pivot:
        _quickselect_recursive(distances, indices, left, pivot - 1, k)
    elif k > pivot:
        _quickselect_recursive(distances, indices, pivot + 1, right, k)


def _partition_arg(
    distances: np.ndarray,
    indices: List[int | None],
    left: int,
    right: int,
    pivot: int,
) -> int:
    pivot_val = distances[indices[pivot]]
    indices[pivot], indices[right] = indices[right], indices[pivot]
    store = left
    for idx in range(left, right):
        if distances[indices[idx]] < pivot_val:
            indices[store], indices[idx] = indices[idx], indices[store]
            store += 1
    indices[right], indices[store] = indices[store], indices[right]
    return store


def _median_of_three_arg(
    distances: np.ndarray,
    indices: List[int | None],
    i1: int,
    i2: int,
    i3: int,
) -> int:
    v1 = distances[indices[i1]]
    v2 = distances[indices[i2]]
    v3 = distances[indices[i3]]
    if v1 < v2:
        if v2 < v3:
            return i2
        if v1 < v3:
            return i3
        return i1
    if v1 < v3:
        return i1
    if v2 < v3:
        return i3
    return i2


def _sample_second_min_arg(
    distances: np.ndarray,
    indices: List[int | None],
    left: int,
    right: int,
    step: int,
) -> int:
    i = left
    i2 = left
    best = float("inf")
    second = float("inf")
    for j in range(left, right + 1, step):
        value = distances[indices[j]]
        if value < best:
            second = best
            best = value
            i2 = i
            i = j
        elif value < second:
            second = value
            i2 = j
    return i2


def _select_group_observations(
    data: ObsLocalInputs,
    group: CTypeGroup,
    dist_tmp: np.ndarray,
    rloc_tmp: np.ndarray,
    rdiag_tmp: np.ndarray,
    rank_coords: tuple[int, int],
    search_q0_value: int,
) -> tuple[List[int], int, int]:
    if not group.ctypes:
        return [], 1, 0

    cutoff_bounds = [
        _obs_local_range(data, ctype_idx, rank_coords) for ctype_idx in group.ctypes
    ]
    search_increments = _compute_search_increments(data, group)

    nobs_use: List[int] = []
    nobs_use2: List[int] = []
    nn_steps = [0] * (len(group.ctypes) + 1)
    q = max(search_q0_value - 1, 0)

    while True:
        q += 1
        reach_cutoff = True
        for icm, ctype_idx in enumerate(group.ctypes):
            nn_steps[icm] = len(nobs_use)
            info = data.ctype_meta[ctype_idx]
            if info.tot_ext <= 0:
                continue
            bounds = _expand_incremental_bounds(
                data,
                info,
                cutoff_bounds[icm],
                search_increments[icm],
                q,
                rank_coords,
            )
            if not bounds[4]:
                reach_cutoff = False
            nobs_use.extend(_obs_choose_ext(info, bounds[0], bounds[1], bounds[2], bounds[3]))
        nn_steps[-1] = len(nobs_use)

        if (not reach_cutoff) and len(nobs_use) < MAX_OBS_PER_GRID:
            continue
        if reach_cutoff and len(nobs_use) == 0:
            break

        nobs_use2.clear()
        dist_cutoff_sq = (search_increments[0] * q / data.hori_loc[group.master]) ** 2
        for icm, ctype_idx in enumerate(group.ctypes):
            for idx in range(nn_steps[icm], nn_steps[icm + 1]):
                iob = nobs_use[idx]
                if rloc_tmp[iob] == 0.0:
                    continue
                if rloc_tmp[iob] < 0.0:
                    ndist, loc_weight, rdiag = _obs_local_cal(
                        data.target,
                        data.obs,
                        iob,
                        ctype_idx,
                        data.hori_loc,
                        data.vert_loc,
                    )
                    if loc_weight <= 0.0:
                        rloc_tmp[iob] = 0.0
                        dist_tmp[iob] = np.inf
                        continue
                    dist_tmp[iob] = ndist
                    rloc_tmp[iob] = loc_weight
                    rdiag_tmp[iob] = rdiag
                if (not reach_cutoff) and dist_tmp[iob] > dist_cutoff_sq:
                    continue
                nobs_use2.append(iob)

        if len(nobs_use2) >= MAX_OBS_PER_GRID or reach_cutoff:
            break

    nobsl_incr = len(nobs_use2)
    if not nobs_use2:
        return [], q, nobsl_incr
    kept = list(nobs_use2)
    if len(kept) > MAX_OBS_PER_GRID:
        _quickselect_arg(dist_tmp, kept, MAX_OBS_PER_GRID)
        kept = kept[:MAX_OBS_PER_GRID]
    return kept, q, nobsl_incr


def _compute_search_increments(data: ObsLocalInputs, group: CTypeGroup) -> List[float]:
    increments: List[float] = []
    master_scale = float(data.hori_loc[group.master])
    master_info = data.ctype_meta[group.master]
    base = master_scale * DIST_ZERO_FAC / N_SEARCH_INCR
    base = max(base, master_info.grdspc_i, master_info.grdspc_j)
    increments.append(base)
    for ctype_idx in group.ctypes[1:]:
        scale = float(data.hori_loc[ctype_idx])
        increments.append(base / master_scale * scale)
    return increments


def _obs_local_range(
    data: ObsLocalInputs,
    ctype_idx: int,
    rank_coords: tuple[int, int],
) -> tuple[int, int, int, int]:
    info = data.ctype_meta[ctype_idx]
    ri = data.target.ri
    rj = data.target.rj
    dist_zero_i = data.hori_loc[ctype_idx] * DIST_ZERO_FAC / DX
    dist_zero_j = data.hori_loc[ctype_idx] * DIST_ZERO_FAC / DY
    imin, jmin = _ij_obsgrd_ext(info, ri - dist_zero_i, rj - dist_zero_j, rank_coords)
    imax, jmax = _ij_obsgrd_ext(info, ri + dist_zero_i, rj + dist_zero_j, rank_coords)
    return imin, imax, jmin, jmax


def _expand_incremental_bounds(
    data: ObsLocalInputs,
    info: ObsGridInfo,
    cutoff: tuple[int, int, int, int],
    search_incr: float,
    q: int,
    rank_coords: tuple[int, int],
) -> tuple[int, int, int, int, bool]:
    incr_i = search_incr / DX
    incr_j = search_incr / DY
    ri = data.target.ri
    rj = data.target.rj
    imin, jmin = _ij_obsgrd_ext(info, ri - incr_i * q, rj - incr_j * q, rank_coords)
    imax, jmax = _ij_obsgrd_ext(info, ri + incr_i * q, rj + incr_j * q, rank_coords)
    hit_cutoff = (
        imin <= cutoff[0]
        and imax >= cutoff[1]
        and jmin <= cutoff[2]
        and jmax >= cutoff[3]
    )
    if hit_cutoff:
        return cutoff[0], cutoff[1], cutoff[2], cutoff[3], True
    return imin, imax, jmin, jmax, False


def _ij_obsgrd_ext(
    info: ObsGridInfo,
    ri: float,
    rj: float,
    rank_coords: tuple[int, int],
) -> tuple[int, int]:
    rank_i, rank_j = rank_coords
    nlon = int(GRID_CONSTANTS["nlon"])
    nlat = int(GRID_CONSTANTS["nlat"])
    ril = ri - rank_i * nlon
    rjl = rj - rank_j * nlat
    ogi = math.ceil((ril - IHALO - 0.5) * info.ngrd_i / nlon) + info.ngrdsch_i
    ogj = math.ceil((rjl - JHALO - 0.5) * info.ngrd_j / nlat) + info.ngrdsch_j
    ogi = max(1, min(info.ngrdext_i, ogi))
    ogj = max(1, min(info.ngrdext_j, ogj))
    return ogi, ogj


def _obs_choose_ext(
    info: ObsGridInfo,
    imin: int,
    imax: int,
    jmin: int,
    jmax: int,
) -> List[int]:
    if imin > imax or jmin > jmax or info.tot_ext <= 0:
        return []
    imin = max(1, min(info.ngrdext_i, imin))
    imax = max(1, min(info.ngrdext_i, imax))
    jmin = max(1, min(info.ngrdext_j, jmin))
    jmax = max(1, min(info.ngrdext_j, jmax))
    ac = info.ac_ext
    picks: List[int] = []
    for j in range(jmin, jmax + 1):
        start = int(ac[imin - 1, j - 1])
        end = int(ac[imax, j - 1])
        if end > start:
            picks.extend(range(start, end))
    return picks


def _rank_coordinates(pe_tag: str) -> tuple[int, int]:
    rank = int(pe_tag[2:])
    return rank % PRC_NUM_X, rank // PRC_NUM_X


def _stack_or_empty(rows: list[Any], empty_shape: tuple[int, ...] | None = None) -> np.ndarray:
    if not rows:
        if empty_shape is None:
            return np.empty((0,), dtype=np.float64)
        return np.empty(empty_shape, dtype=np.float64)
    if isinstance(rows[0], np.ndarray):
        return np.asarray(rows, dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def _normalize_identifier(identifier: ObsLocalIdentifier | Mapping[str, Any]) -> ObsLocalIdentifier:
    if isinstance(identifier, ObsLocalIdentifier):
        return identifier
    dump_dir = identifier.get("dump_dir")
    call_id = identifier.get("call_id")
    if dump_dir is None or call_id is None:
        raise ValueError("rank_identifier must include 'dump_dir' and 'call_id'")
    pe_tag = identifier.get("pe_tag", "pe000000")
    member = identifier.get("member", "mem0001")
    return ObsLocalIdentifier(
        dump_dir=dump_dir,
        call_id=int(call_id),
        pe_tag=str(pe_tag),
        member=str(member),
    )


if TYPE_CHECKING:
    from .das_replay import ObsLocalReplay

_REPLAY_CACHE: Dict[tuple[str, str, str], "ObsLocalReplay"] = {}


def load_obs_local_from_global(rank_identifier: ObsLocalIdentifier | Mapping[str, Any]) -> ObsLocalInputs:
    """Same as ``load_obs_local`` but derives the 'before' state without das dumps."""

    ident = _normalize_identifier(rank_identifier)
    replay = _get_replay(ident)
    inputs = replay.build_inputs_for(ident.call_id, copy_search=True)
    after_stage = load_das_obs_local_after(
        ident.dump_dir, ident.call_id, pe_tag=ident.pe_tag, member=ident.member
    )
    inputs.after_arrays = after_stage["data"]
    return inputs


def _get_replay(ident: ObsLocalIdentifier) -> "ObsLocalReplay":
    from .das_replay import ObsLocalReplay

    dump_dir = str(Path(ident.dump_dir))
    key = (dump_dir, ident.pe_tag, ident.member)
    replay = _REPLAY_CACHE.get(key)
    if replay is None:
        replay = ObsLocalReplay(dump_dir, ident.pe_tag, ident.member)
        _REPLAY_CACHE[key] = replay
    return replay


__all__ = [
    "ObsLocalIdentifier",
    "ObsLocalInputs",
    "load_obs_local",
    "load_obs_local_from_global",
    "obs_local",
    "test_local_obs",
    "test_local_obs_from_global",
]
