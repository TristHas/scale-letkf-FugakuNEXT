from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import xarray as xr

from pyletkf.io.das_letkf_dumps import (
    load_das_letkf_core_after,
    load_das_letkf_core_before,
    load_das_obs_local_after,
    load_das_postproc_after,
    load_das_postproc_before,
)
from pyletkf.io.letkf_dumps import (
    _normalize_member,
    _normalize_pe_tag,
    _read_binary_array,
    load_rank_members,
)
from pyletkf.params import (
    HORI_LOCAL_RADAR_OBSNOREF,
    MAX_OBS_PER_GRID,
    MEMBERS,
    NX_TILE,
    NY_TILE,
    DY,
    DX,
    VERT_LOCAL_RADAR_OBSNOREF,
    DIST_ZERO_FAC,
)
from pyletkf.spatial.map_obs_to_state import (
    assemble_cell_outputs,
    chunked_topk_neighbors,
)


LETKF_VARS = [
    "U",
    "V",
    "W",
    "T",
    "P",
    "QV",
    "QC",
    "QR",
    "QI",
    "QS",
    "QG",
]


def ij_to_yx(ij: int) -> tuple[int, int]:
    idx0 = ij - 1
    nx = NX_TILE
    iy = idx0 // nx
    ix = idx0 % nx
    return iy, ix

def cell_index(ij: int, ilev: int, nlev: int) -> int:
    horiz = ij - 1
    lev = ilev - 1
    return lev + horiz * nlev

def load_analysis_da(dump_root: Path, pe_tag: str) -> xr.DataArray:
    return load_rank_members(dump_root, "anal3d", pe_tag)

def load_obsdanosort_coords(
    dump_root: Path, pe_tag: str, member: str = "mem0001"
) -> tuple[np.ndarray, np.ndarray]:
    stage_dir = Path(dump_root) / "obsda_after_set_letkf"
    pe_norm = _normalize_pe_tag(pe_tag)
    mem_norm = _normalize_member(member)
    ri_path = stage_dir / f"obsdanosort_ri_{pe_norm}.{mem_norm}.bin"
    rj_path = stage_dir / f"obsdanosort_rj_{pe_norm}.{mem_norm}.bin"
    if not ri_path.exists() or not rj_path.exists():
        raise FileNotFoundError(
            f"obsdanosort files missing for {pe_norm}.{mem_norm}"
        )
    ri_vals = _read_binary_array(ri_path, ">f8").reshape(-1).astype(np.float64)
    rj_vals = _read_binary_array(rj_path, ">f8").reshape(-1).astype(np.float64)
    return ri_vals, rj_vals

def _bool_from_meta(meta_val: str | bool) -> bool:
    if isinstance(meta_val, bool):
        return meta_val
    token = str(meta_val).strip().lower()
    return token in {"true", "t", "1", "yes"}

def build_core_batch(call_ids: Sequence[int], dump_root: Path, pe_tag: str):
    hdxf_list: list[torch.Tensor] = []
    dep_list: list[torch.Tensor] = []
    rloc_list: list[torch.Tensor] = []
    rdiag_list: list[torch.Tensor] = []
    obs_counts: list[int] = []
    parm_infl: list[float] = []
    rdiag_wloc: list[bool] = []
    infl_update: list[bool] = []
    for call_id in call_ids:
        before = load_das_letkf_core_before(
            dump_root, call_id=call_id, pe_tag=pe_tag, member="mem0001"
        )
        data = before["data"]
        obs_counts.append(data["hdxf"].shape[0])
        parm_infl.append(float(before["meta"]["parm_infl"]))
        rdiag_wloc.append(_bool_from_meta(before["meta"]["rdiag_wloc"]))
        infl_update.append(_bool_from_meta(before["meta"]["infl_update"]))
        hdxf_list.append(
            torch.from_numpy(np.asarray(data["hdxf"], dtype=np.float64))
        )
        dep_list.append(
            torch.from_numpy(np.asarray(data["dep"], dtype=np.float64))
        )
        rloc_list.append(
            torch.from_numpy(np.asarray(data["rloc"], dtype=np.float64))
        )
        rdiag_list.append(
            torch.from_numpy(np.asarray(data["rdiag"], dtype=np.float64))
        )
    max_obs = max(obs_counts)
    batch = len(call_ids)

    def _pad_stack(src_list: list[torch.Tensor], ndim: int) -> torch.Tensor:
        shape = (batch, max_obs) + src_list[0].shape[1:]
        out = torch.zeros(shape, dtype=torch.float64)
        for idx, tensor in enumerate(src_list):
            nobs = obs_counts[idx]
            slicer = (slice(idx, idx + 1), slice(0, nobs))
            out[(slice(idx, idx + 1), slice(0, nobs))] = tensor
        return out

    hdxf = torch.zeros((batch, max_obs, len(MEMBERS)), dtype=torch.float64)
    dep = torch.zeros((batch, max_obs), dtype=torch.float64)
    rloc = torch.zeros((batch, max_obs), dtype=torch.float64)
    rdiag = torch.zeros((batch, max_obs), dtype=torch.float64)
    mask = torch.zeros((batch, max_obs), dtype=torch.bool)

    for idx, (hdxf_tensor, dep_tensor, rloc_tensor, rdiag_tensor) in enumerate(
        zip(hdxf_list, dep_list, rloc_list, rdiag_list)
    ):
        nobs = obs_counts[idx]
        hdxf[idx, :nobs, :] = hdxf_tensor
        dep[idx, :nobs] = dep_tensor
        rloc[idx, :nobs] = rloc_tensor
        rdiag[idx, :nobs] = rdiag_tensor
        mask[idx, :nobs] = True

    parm_infl_tensor = torch.tensor(parm_infl, dtype=torch.float64)
    rdiag_wloc_tensor = torch.tensor(rdiag_wloc, dtype=torch.bool)
    infl_update_tensor = torch.tensor(infl_update, dtype=torch.bool)

    class CoreBatch:
        pass

    batch_obj = CoreBatch()
    batch_obj.hdxf = hdxf
    batch_obj.dep = dep
    batch_obj.rloc = rloc
    batch_obj.rdiag = rdiag
    batch_obj.obs_mask = mask
    batch_obj.parm_infl = parm_infl_tensor
    batch_obj.rdiag_wloc = rdiag_wloc_tensor
    batch_obj.infl_update = infl_update_tensor
    return batch_obj, obs_counts


@dataclass
class PostprocData:
    trans: torch.Tensor
    transm: torch.Tensor
    gues_members: torch.Tensor
    gues_mean: torch.Tensor
    parm_infl: torch.Tensor

@dataclass
class PostprocParams:
    beta: torch.Tensor
    relax_alpha: torch.Tensor
    relax_alpha_spread: torch.Tensor
    relax_to_inflated: torch.Tensor
    relax_spread_out: torch.Tensor
    det_run: torch.Tensor
    nvar: torch.Tensor

def prepare_postproc_tensors(
    call_id: int, dump_root: Path, pe_tag: str
) -> tuple[PostprocData, PostprocParams, dict]:
    before = load_das_postproc_before(
        dump_root, call_id=call_id, pe_tag=pe_tag, member="mem0001"
    )
    after = load_das_postproc_after(
        dump_root, call_id=call_id, pe_tag=pe_tag, member="mem0001"
    )
    data_tensors = PostprocData(
        trans=torch.from_numpy(
            np.asarray(before["data"]["trans"], dtype=np.float64)
        ).unsqueeze(0),
        transm=torch.from_numpy(
            np.asarray(before["data"]["transm"], dtype=np.float64)
        ).unsqueeze(0),
        gues_members=torch.from_numpy(
            np.asarray(before["data"]["gues_members"], dtype=np.float64)
        ).unsqueeze(0),
        gues_mean=torch.tensor(
            [float(before["meta"]["gues_mean"])], dtype=torch.float64
        ),
        parm_infl=torch.tensor([float(before["meta"]["parm"])], dtype=torch.float64),
    )
    params = PostprocParams(
        beta=torch.tensor([float(before["meta"]["beta"])], dtype=torch.float64),
        relax_alpha=torch.tensor(
            [float(before["meta"]["relax_alpha"])], dtype=torch.float64
        ),
        relax_alpha_spread=torch.tensor(
            [float(before["meta"]["relax_alpha_spread"])], dtype=torch.float64
        ),
        relax_to_inflated=torch.tensor(
            [_bool_from_meta(before["meta"]["relax_to_inflated_prior"])],
            dtype=torch.bool,
        ),
        relax_spread_out=torch.tensor(
            [_bool_from_meta(before["meta"]["relax_spread_out"])],
            dtype=torch.bool,
        ),
        det_run=torch.tensor(
            [_bool_from_meta(before["meta"]["det_run"])], dtype=torch.bool
        ),
        nvar=torch.tensor([int(before["meta"]["nvar"])], dtype=torch.int64),
    )
    return data_tensors, params, after


def compute_cell_observation_dataset(
    state_ds,
    obs_ds,
    ij: int,
    ilev: int,
    *,
    device: torch.device,
):
    state = state_ds["state"]
    nx = state.sizes["x"]
    iy, ix = ij_to_yx(ij)
    grid_xy = torch.stack(
        (
            torch.as_tensor(state.coords["x"][ix], dtype=torch.float64, device=device),
            torch.as_tensor(state.coords["y"][iy], dtype=torch.float64, device=device),
        )
    ).unsqueeze(0)
    height_val = (
        state_ds["height"].data[ilev - 1, iy, ix]
        .to(device=device, dtype=torch.float64)
        .unsqueeze(0)
    )

    obs_xy = torch.stack(
        (
            obs_ds["ri_global"].data.to(device=device, dtype=torch.float64) * DX,
            obs_ds["rj_global"].data.to(device=device, dtype=torch.float64) * DY,
        ),
        dim=1,
    )
    obs_z = obs_ds["lev"].data.to(device=device, dtype=torch.float64)

    max_obs = min(MAX_OBS_PER_GRID, obs_ds.sizes["obs"])
    topk = chunked_topk_neighbors(
        grid_xy,
        height_val,
        obs_xy,
        obs_z,
        horiz_loc=HORI_LOCAL_RADAR_OBSNOREF,
        vert_loc=VERT_LOCAL_RADAR_OBSNOREF,
        dist_zero_fac=DIST_ZERO_FAC,
        max_obs_per_grid=max_obs,
        chunk_size=1,
    )

    assembled = assemble_cell_outputs(
        topk["topk_vals"],
        topk["topk_idx"],
        topk["valid_mask"],
        obs_ds,
        var_local_factor=1.0,
    )
    return assembled
