from __future__ import annotations
import numpy as np
import torch
import xarray as xr

from .grid_proj import (
    compute_grid_indices_from_state,
    compute_obs_grid_idx,
)
from .interp_obs_state import sample_state
from .obs_op import compute_all_hx
from .filter_obs import filter_sc23_obs

HORI_LOCAL_RADAR_OBSNOREF = 2000.0
VERT_LOCAL_RADAR_OBSNOREF = 2000.0
MAX_OBS_PER_GRID = 100
DIST_ZERO_FAC = 3.651483717
DX = 100.0
DY = 100.0
MEMBERS = ("0001", "0002")


def extract_coordinate_tensors(
    state_ds: xr.Dataset,
    obs_ds: xr.Dataset,
    pe_tag: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor | int]:
    """Build contiguous torch tensors for grid and observation coordinates."""

    grid_ri, grid_rj, grid_rz = compute_grid_indices_from_state(state_ds, pe_tag)
    grid_xy = torch.stack(
        (
            torch.as_tensor(grid_ri, device=device, dtype=dtype) * DX,
            torch.as_tensor(grid_rj, device=device, dtype=dtype) * DY,
        ),
        dim=1,
    )
    grid_z = torch.as_tensor(grid_rz, device=device, dtype=dtype)

    obs_xy = torch.stack(
        (
            torch.as_tensor(obs_ds["ri_global"].values, device=device, dtype=dtype) * DX,
            torch.as_tensor(obs_ds["rj_global"].values, device=device, dtype=dtype) * DY,
        ),
        dim=1,
    )
    obs_z = torch.as_tensor(obs_ds["lev"].values, device=device, dtype=dtype)
    horiz_len = state_ds.sizes["x"] * state_ds.sizes["y"]
    return {
        "grid_xy": grid_xy,
        "grid_z": grid_z,
        "obs_xy": obs_xy,
        "obs_z": obs_z,
        "horiz_len": horiz_len,
    }


def chunked_topk_neighbors(
    grid_xy: torch.Tensor,
    grid_z: torch.Tensor,
    obs_xy: torch.Tensor,
    obs_z: torch.Tensor,
    *,
    horiz_loc: float,
    vert_loc: float,
    dist_zero_fac: float,
    max_obs_per_grid: int,
    chunk_size: int,
) -> Dict[str, torch.Tensor]:
    """Return nearest observations for every grid cell using chunked torch.cdist."""

    device = grid_xy.device
    dtype = grid_xy.dtype
    topk_vals: list[torch.Tensor] = []
    topk_idx: list[torch.Tensor] = []
    valid_masks: list[torch.Tensor] = []

    for start in range(0, grid_xy.shape[0], chunk_size):
        end = min(start + chunk_size, grid_xy.shape[0])
        chunk_xy = grid_xy[start:end]
        horiz = torch.cdist(chunk_xy, obs_xy) / horiz_loc
        vert = torch.abs(grid_z[start:end].unsqueeze(1) - obs_z) / vert_loc
        ndist = horiz * horiz + vert * vert
        mask = (
            (horiz <= dist_zero_fac)
            & (vert <= dist_zero_fac)
            & (ndist <= dist_zero_fac * dist_zero_fac)
        )
        masked = torch.where(mask, ndist, torch.tensor(float("inf"), device=device, dtype=dtype))
        vals, idx = torch.topk(
            masked,
            k=max_obs_per_grid,
            dim=1,
            largest=False,
            sorted=True,
        )
        topk_vals.append(vals)
        topk_idx.append(idx)
        valid_masks.append(torch.isfinite(vals))

    return {
        "topk_vals": torch.cat(topk_vals, dim=0),
        "topk_idx": torch.cat(topk_idx, dim=0),
        "valid_mask": torch.cat(valid_masks, dim=0),
    }

def assemble_cell_outputs(
    topk_vals: torch.Tensor,
    topk_idx: torch.Tensor,
    valid_mask: torch.Tensor,
    hx_torch: torch.Tensor,
    hx_mean: torch.Tensor,
    obs_ds: xr.Dataset,
    *,
    var_local_factor: float,
) -> Dict[str, torch.Tensor]:
    """Gather hdxf/dep/rloc/rdiag for each grid cell using the neighbor indices."""

    device = topk_vals.device
    dtype = topk_vals.dtype
    hdxf = hx_torch - hx_mean
    
    dep = torch.as_tensor(obs_ds["dat"].values, device=device, dtype=dtype) - hx_mean.squeeze(1)
    err = torch.as_tensor(obs_ds["err"].values, device=device, dtype=dtype)
    obs_idx = torch.as_tensor(obs_ds.obs.values, device=device, dtype=torch.int64)

    hdxf_sel = hdxf[topk_idx]
    dep_sel = dep[topk_idx]
    err_sel = err[topk_idx]

    rloc_sel = var_local_factor * torch.exp(-0.5 * topk_vals)
    rdiag_sel = torch.where(rloc_sel > 0.0, err_sel * err_sel / rloc_sel, torch.zeros_like(rloc_sel))

    hdxf_sel *= valid_mask.unsqueeze(-1)#torch.where(valid_mask.unsqueeze(-1), hdxf_sel, torch.zeros_like(hdxf_sel))
    dep_sel  *= valid_mask#torch.where(valid_mask, dep_sel, torch.zeros_like(dep_sel))
    rloc_sel *= valid_mask#torch.where(valid_mask, rloc_sel, torch.zeros_like(rloc_sel))
    rdiag_sel*= valid_mask#torch.where(valid_mask, rdiag_sel, torch.zeros_like(rdiag_sel))
    
    idx_sel = torch.where(
        valid_mask,
        obs_idx[topk_idx],
        -1,
    )
    counts_sel = valid_mask.sum(dim=1, dtype=torch.int32)
    return {
        "hdxf": hdxf_sel,
        "dep": dep_sel,
        "rloc": rloc_sel,
        "rdiag": rdiag_sel,
        "counts": counts_sel,
        "indices": idx_sel,
        "mask":valid_mask
    }

def pre_letkf_pipeline(obs, state_ds, pe_tag, 
                       device = torch.device("cuda:2"),
                       chunk_size = 1024):
    state = (
        state_ds["state"]
        .transpose("y", "x", "z", "ens", "variable")
        .sel(ens=list(MEMBERS))
    )
    obs_filtered = compute_obs_grid_idx(obs, state_ds, pe_tag)

    samples, _ = sample_state(
        state,
        state_ds["height"].values,
        obs_filtered["ri_local"].values,
        obs_filtered["rj_local"].values,
        obs_filtered["lev"].values,
    )
    
    obs_state = torch.from_numpy(samples.transpose(2, 0, 1)).to(device=device, dtype=torch.float64)
    
    hx = compute_all_hx(obs_state, obs_filtered)
    hx_mean = hx.mean(dim=1, keepdim=True)
    
    obs_filtered, keep = filter_sc23_obs(obs_filtered, hx)
    hx = hx[keep]
    hx_mean = hx_mean[keep]
    
    coords = extract_coordinate_tensors(state_ds, obs_filtered, pe_tag, device=device, dtype=torch.float64)
    topk = chunked_topk_neighbors(
        coords["grid_xy"],
        coords["grid_z"],
        coords["obs_xy"],
        coords["obs_z"],
        horiz_loc=HORI_LOCAL_RADAR_OBSNOREF,
        vert_loc=VERT_LOCAL_RADAR_OBSNOREF,
        dist_zero_fac=DIST_ZERO_FAC,
        max_obs_per_grid=MAX_OBS_PER_GRID,
        chunk_size=chunk_size,
    )
    
    assembled = assemble_cell_outputs(
        topk["topk_vals"],
        topk["topk_idx"],
        topk["valid_mask"],
        hx,
        hx_mean,
        obs_filtered,
        var_local_factor=1.0,
    )
    
    results = {
            "hdxf": assembled["hdxf"].cpu().numpy(),
            "dep": assembled["dep"].cpu().numpy(),
            "rloc": assembled["rloc"].cpu().numpy(),
            "rdiag": assembled["rdiag"].cpu().numpy(),
            "counts": assembled["counts"].cpu().numpy(),
            "mask": assembled["mask"].cpu().numpy(),
            "obs_indices": assembled["indices"].cpu().numpy(),
            "grid_info": {
                "ri": coords["grid_xy"][:, 0].cpu().numpy() / DX,
                "rj": coords["grid_xy"][:, 1].cpu().numpy() / DY,
                "n_horiz": coords["horiz_len"],
            },
        }
    return results
