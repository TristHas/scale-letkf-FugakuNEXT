"""Torch translation of the obs_local routine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Tuple

import numpy as np
import torch

from .obs_local import (
    MAX_OBS_PER_GRID,
    ObsLocalIdentifier,
    ObsLocalInputs,
    CTypeGroup,
    ObsGridInfo,
    load_obs_local_from_global,
    obs_local as obs_local_numpy,
    _stack_or_empty,
    _quickselect_arg,
    _rank_coordinates,
    _compute_search_increments,
    _obs_local_range,
    _expand_incremental_bounds,
    _obs_choose_ext,
    DX,
    DY,
    DIST_ZERO_FAC,
    DIST_ZERO_FAC_SQ,
    ID_PS,
    ID_RAIN,
    PHARAD_TYP,
    VERT_LOCAL_RAIN_BASE,
    QUICKSELECT_EPS,
)


@dataclass
class ObsDatasetTorch:
    ensval: torch.Tensor
    dep: torch.Tensor
    ri: torch.Tensor
    rj: torch.Tensor
    lev: torch.Tensor
    dat: torch.Tensor
    err: torch.Tensor
    elm: torch.Tensor
    typ: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.ensval.shape[0])


@dataclass
class TorchObsLocalContext:
    inputs: ObsLocalInputs
    obs: ObsDatasetTorch
    hori_loc: torch.Tensor
    vert_loc: torch.Tensor
    device: torch.device


def _to_tensor(array: np.ndarray, *, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array, dtype=np.float64)).to(device=device)


def _build_context(inputs: ObsLocalInputs, *, device: torch.device) -> TorchObsLocalContext:
    obs = inputs.obs
    dataset = ObsDatasetTorch(
        ensval=torch.from_numpy(np.asarray(obs.ensval, dtype=np.float64)).to(device),
        dep=torch.from_numpy(np.asarray(obs.dep, dtype=np.float64)).to(device),
        ri=torch.from_numpy(np.asarray(obs.ri, dtype=np.float64)).to(device),
        rj=torch.from_numpy(np.asarray(obs.rj, dtype=np.float64)).to(device),
        lev=torch.from_numpy(np.asarray(obs.lev, dtype=np.float64)).to(device),
        dat=torch.from_numpy(np.asarray(obs.dat, dtype=np.float64)).to(device),
        err=torch.from_numpy(np.asarray(obs.err, dtype=np.float64)).to(device),
        elm=torch.from_numpy(np.asarray(obs.elm, dtype=np.int64)).to(device),
        typ=torch.from_numpy(np.asarray(obs.typ, dtype=np.int64)).to(device),
    )
    return TorchObsLocalContext(
        inputs=inputs,
        obs=dataset,
        hori_loc=_to_tensor(np.asarray(inputs.hori_loc, dtype=np.float64), device=device),
        vert_loc=_to_tensor(np.asarray(inputs.vert_loc, dtype=np.float64), device=device),
        device=device,
    )


def obs_local_torch(
    inputs: ObsLocalInputs,
    search_q0: np.ndarray | None = None,
    *,
    device: torch.device | str | None = None,
) -> Mapping[str, np.ndarray]:
    torch_device = torch.device(device) if isinstance(device, str) \
    or device is not None else torch.device("cpu")
    ctx = _build_context(inputs, device=torch_device)
    obs = ctx.obs
    nobs = obs.count

    dist_tmp = torch.full((nobs,), float("inf"), dtype=torch.float64, device=torch_device)
    rloc_tmp = torch.full((nobs,), -1.0, dtype=torch.float64, device=torch_device)
    rdiag_tmp = torch.zeros((nobs,), dtype=torch.float64, device=torch_device)

    selected_hdxf: List[np.ndarray] = []
    selected_dep: List[float] = []
    selected_rdiag: List[float] = []
    selected_rloc: List[float] = []

    rank_coords = _rank_coordinates(inputs.request.pe_tag)
    if search_q0 is not None:
        search_vec = torch.as_tensor(search_q0, dtype=torch.int64, device=torch_device)
    else:
        search_vec = None

    for group in inputs.ctype_groups:
        q_init = int(search_vec[group.master].item()) if search_vec is not None else 1
        kept_indices, q_final, nobsl_incr = _select_group_observations_torch(
            ctx,
            group,
            dist_tmp,
            rloc_tmp,
            rdiag_tmp,
            rank_coords,
            q_init,
        )
        if search_vec is not None and nobsl_incr > 0:
            if q_final == q_init and nobsl_incr > MAX_OBS_PER_GRID * 3:
                search_vec[group.master] = max(q_final - 1, 1)
            elif q_final > q_init:
                search_vec[group.master] = q_final
        if not kept_indices:
            continue
        for iob in kept_indices:
            selected_hdxf.append(obs.ensval[iob].cpu().numpy())
            selected_dep.append(float(obs.dep[iob].cpu().numpy()))
            selected_rdiag.append(float(rdiag_tmp[iob].cpu().numpy()))
            selected_rloc.append(float(rloc_tmp[iob].cpu().numpy()))

    hdxf  = _stack_or_empty(selected_hdxf, (0, inputs.obs.ensval.shape[1]))
    dep   = _stack_or_empty(selected_dep)
    rdiag = _stack_or_empty(selected_rdiag)
    rloc  = _stack_or_empty(selected_rloc)

    if search_vec is not None and search_q0 is not None:
        search_q0[:] = search_vec.cpu().numpy()

    return {
        "hdxf": hdxf,
        "dep": dep,
        "rdiag": rdiag,
        "rloc": rloc,
    }


def _select_group_observations_torch(
    ctx: TorchObsLocalContext,
    group: CTypeGroup,
    dist_tmp: torch.Tensor,
    rloc_tmp: torch.Tensor,
    rdiag_tmp: torch.Tensor,
    rank_coords: tuple[int, int],
    search_q0_value: int,
) -> tuple[List[int], int, int]:
    data = ctx.inputs
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
        dist_cutoff_sq = (search_increments[0] * q / float(data.hori_loc[group.master])) ** 2
        for icm, ctype_idx in enumerate(group.ctypes):
            for idx in range(nn_steps[icm], nn_steps[icm + 1]):
                iob = nobs_use[idx]
                if float(rloc_tmp[iob].cpu().numpy()) == 0.0:
                    continue
                if float(rloc_tmp[iob].cpu().numpy()) < 0.0:
                    ndist, loc_weight, rdiag = _obs_local_cal_torch(
                        ctx,
                        iob,
                        ctype_idx,
                    )
                    if loc_weight <= 0.0:
                        rloc_tmp[iob] = 0.0
                        dist_tmp[iob] = float("inf")
                        continue
                    dist_tmp[iob] = ndist
                    rloc_tmp[iob] = loc_weight
                    rdiag_tmp[iob] = rdiag
                if (not reach_cutoff) and float(dist_tmp[iob].cpu().numpy()) > dist_cutoff_sq:
                    continue
                nobs_use2.append(iob)

        if len(nobs_use2) >= MAX_OBS_PER_GRID or reach_cutoff:
            break

    nobsl_incr = len(nobs_use2)
    if not nobs_use2:
        return [], q, nobsl_incr
    kept = list(nobs_use2)
    if len(kept) > MAX_OBS_PER_GRID:
        dist_vals = dist_tmp.cpu().numpy()
        _quickselect_arg(dist_vals, kept, MAX_OBS_PER_GRID)
        kept = kept[:MAX_OBS_PER_GRID]
    return kept, q, nobsl_incr


def _obs_local_cal_torch(
    ctx: TorchObsLocalContext,
    iob: int,
    ctype_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    obs = ctx.obs
    data = ctx.inputs
    target = data.target
    vert_scale = ctx.vert_loc[ctype_idx]
    hori_scale = ctx.hori_loc[ctype_idx]

    ri = obs.ri[iob]
    rj = obs.rj[iob]
    lev = obs.lev[iob]
    dat = obs.dat[iob]
    err = obs.err[iob]
    elm = obs.elm[iob]
    typ = obs.typ[iob]

    nd_v = torch.tensor(0.0, dtype=torch.float64, device=ctx.device)
    target_rlev = float(target.rlev)
    target_rz = float(target.rz)

    if float(vert_scale) != 0.0:
        if int(elm.item()) == ID_PS and target_rlev > 0.0:
            nd_v = torch.abs(torch.log(dat) - np.log(target_rlev)) / vert_scale
        elif int(elm.item()) == ID_RAIN:
            base = target_rlev if target_rlev > 0.0 else VERT_LOCAL_RAIN_BASE
            nd_v = torch.abs(torch.log(torch.clamp(lev, min=1.0)) - np.log(base)) / vert_scale
        elif int(typ.item()) == PHARAD_TYP:
            nd_v = torch.abs(lev - target_rz) / vert_scale
        else:
            if lev > 0.0 and target_rlev > 0.0:
                nd_v = torch.abs(torch.log(lev) - np.log(target_rlev)) / vert_scale
            else:
                nd_v = torch.tensor(0.0, dtype=torch.float64, device=ctx.device)

    if nd_v > DIST_ZERO_FAC:
        zero = torch.tensor(0.0, dtype=torch.float64, device=ctx.device)
        return torch.tensor(float("inf"), dtype=torch.float64, device=ctx.device), zero, zero

    dx = (float(target.ri) - ri) * DX
    dy = (float(target.rj) - rj) * DY
    nd_h = torch.hypot(dx, dy) / hori_scale
    if nd_h > DIST_ZERO_FAC:
        zero = torch.tensor(0.0, dtype=torch.float64, device=ctx.device)
        return torch.tensor(float("inf"), dtype=torch.float64, device=ctx.device), zero, zero

    ndist = nd_h * nd_h + nd_v * nd_v
    if ndist > DIST_ZERO_FAC_SQ:
        zero = torch.tensor(0.0, dtype=torch.float64, device=ctx.device)
        return torch.tensor(float("inf"), dtype=torch.float64, device=ctx.device), zero, zero

    loc_weight = torch.exp(-0.5 * ndist)
    if loc_weight <= QUICKSELECT_EPS:
        zero = torch.tensor(0.0, dtype=torch.float64, device=ctx.device)
        return torch.tensor(float("inf"), dtype=torch.float64, device=ctx.device), zero, zero

    rdiag = (err * err) / loc_weight
    return ndist, loc_weight, rdiag


def load_obs_local_torch(rank_identifier: ObsLocalIdentifier | Mapping[str, object]) -> ObsLocalInputs:
    return load_obs_local_from_global(rank_identifier)


__all__ = ["obs_local_torch", "load_obs_local_torch"]
