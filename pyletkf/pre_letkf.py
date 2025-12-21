from __future__ import annotations
import math

import numpy as np
import torch
import xarray as xr

from .spatial.grid_proj import compute_obs_grid_idx
from .spatial.map_obs_to_state import gather_obs
from .spatial.map_state_to_obs import read_all_inov

from .obs_op.filter_obs import filter_sc23_obs
from .params import (HORI_LOCAL_RADAR_OBSNOREF, VERT_LOCAL_RADAR_OBSNOREF,
                     MAX_OBS_PER_GRID, DIST_ZERO_FAC, DX, DY, MEMBERS)

def pre_letkf(obs, states, device, chunk_size=1024):
    # Step 1: Populate observations with hx
    obs = obs.to(device)
    obs = compute_obs_grid_idx(obs)
    obs = read_all_inov(obs, states, device=device)
    obs_valid = filter_sc23_obs(obs)

    # Step 2: Populate each state cell with the nearest observation hx
    halo_i = math.ceil(HORI_LOCAL_RADAR_OBSNOREF * DIST_ZERO_FAC / DX)
    halo_j = math.ceil(HORI_LOCAL_RADAR_OBSNOREF * DIST_ZERO_FAC / DY)
    results = { tile_index:gather_obs(state_ds, obs_valid, tile_index, 
                             halo_i, halo_j, 
                             device=device, chunk_size=chunk_size)\
                for tile_index, state_ds in states.items()}
    return results



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

