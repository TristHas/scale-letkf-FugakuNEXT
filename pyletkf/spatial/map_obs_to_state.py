from __future__ import annotations
import math
from typing import Mapping, Optional

import torch
from xtensor import Dataset

from ..params import (DX, DY, PRC_NUM_X, PRC_NUM_Y, NX_TILE, NY_TILE, DX, DY, TOTAL_NX, TOTAL_NY,
                      HORI_LOCAL_RADAR_OBSNOREF, VERT_LOCAL_RADAR_OBSNOREF, 
                      DIST_ZERO_FAC, MAX_OBS_PER_GRID)

def tile_bounds(tile_i: int, tile_j: int, halo_i: int = 0, halo_j: int = 0) -> tuple[float, float, float, float]:
    start_i = max(1.0, tile_i * NX_TILE + 1.0 - halo_i)
    end_i = min(TOTAL_NX, (tile_i + 1) * NX_TILE + halo_i)
    start_j = max(1.0, tile_j * NY_TILE + 1.0 - halo_j)
    end_j = min(TOTAL_NY, (tile_j + 1) * NY_TILE + halo_j)
    return start_i, end_i, start_j, end_j

def filter_obs_to_haloed_tile(obs: Dataset, tile_index: int, halo_i: int, halo_j: int) -> Dataset | None:
    tile_i = tile_index % PRC_NUM_X
    tile_j = tile_index // PRC_NUM_X
    start_i, end_i, start_j, end_j = tile_bounds(tile_i, tile_j, halo_i=halo_i, halo_j=halo_j)
    ri = obs["ri_global"].data
    rj = obs["rj_global"].data
    mask = (ri >= start_i) & (ri <= end_i) & (rj >= start_j) & (rj <= end_j)
    indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
    if indices.numel() == 0:
        return None
    return obs.isel(obs=indices)

def _coord_tensor(values, device, dtype):
    if isinstance(values, torch.Tensor):
        return values.to(device=device, dtype=dtype)
    return torch.as_tensor(values, device=device, dtype=dtype)

def extract_coordinate_tensors(state_ds: Dataset, obs_ds: Dataset, pe_tag: str, *, device, dtype):
    state = state_ds["state"]
    y = _coord_tensor(state.coords["y"], device, torch.float64)
    x = _coord_tensor(state.coords["x"], device, torch.float64)
    z = _coord_tensor(state.coords["z"], device, torch.float64)
    xi, yi = torch.meshgrid(y, x, indexing="ij")
    grid_ri = xi.reshape(-1).repeat_interleave(len(z))
    grid_rj = yi.reshape(-1).repeat_interleave(len(z))
    grid_z = state_ds["height"].data.permute(1, 2, 0).reshape(-1).to(device=device, dtype=dtype)
    grid_xy = torch.stack((grid_ri * DX, grid_rj * DY), dim=1).to(device=device, dtype=dtype)
    obs_xy = torch.stack(
        (
            obs_ds["ri_global"].data.to(device=device, dtype=dtype) * DX,
            obs_ds["rj_global"].data.to(device=device, dtype=dtype) * DY,
        ),
        dim=1,
    )
    obs_z = obs_ds["lev"].data.to(device=device, dtype=dtype)
    grid_z = grid_z
    horiz_len = state.sizes["x"] * state.sizes["y"]
    return {
        "grid_xy": grid_xy,
        "grid_z": grid_z,
        "obs_xy": obs_xy,
        "obs_z": obs_z,
        "horiz_len": horiz_len,
    }


def chunked_topk_neighbors(
    grid_xy,
    grid_z,
    obs_xy,
    obs_z,
    *,
    horiz_loc,
    vert_loc,
    dist_zero_fac,
    max_obs_per_grid,
    chunk_size,
):
    device = grid_xy.device
    dtype = grid_xy.dtype
    topk_vals = []
    topk_idx = []
    valid_masks = []
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
        masked = torch.where(mask, ndist, torch.full_like(ndist, float("inf")))
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
    topk_vals,
    topk_idx,
    valid_mask,
    obs_ds: Dataset,
    *,
    var_local_factor,
    ):
    hx = obs_ds["hx"].data
    hx_mean = obs_ds["hx_mean"].data
    hdxf = hx - hx_mean[:,None]
    dep = obs_ds["dat"].data - hx_mean
    err = obs_ds["err"].data
    obs_idx = obs_ds["obs"].data

    hdxf_sel = hdxf[topk_idx]
    dep_sel = dep[topk_idx]
    err_sel = err[topk_idx]

    rloc_sel = var_local_factor * torch.exp(-0.5 * topk_vals)
    rdiag_sel = torch.where(rloc_sel > 0.0, err_sel * err_sel / rloc_sel, 0)

    mask_expanded = valid_mask.unsqueeze(-1)
    hdxf_sel = torch.where(mask_expanded, hdxf_sel, 0)
    dep_sel = torch.where(valid_mask, dep_sel, 0)
    rloc_sel = torch.where(valid_mask, rloc_sel, 0)
    rdiag_sel = torch.where(valid_mask, rdiag_sel, 0)
    idx_sel = torch.where(valid_mask,obs_idx[topk_idx],-1)
    
    counts_sel = valid_mask.sum(dim=1, dtype=torch.int32)
    return format_to_dataset(hdxf_sel, dep_sel, rloc_sel, rdiag_sel, valid_mask, obs_ds)

def format_to_dataset(hdxf, dep, rloc, rdiag, obs_mask, obs):
    cell_dim = torch.arange(hdxf.shape[0], device=hdxf.device)
    obs_dim = torch.arange(hdxf.shape[1], device=hdxf.device)
    dataset = Dataset(coords={"cell":cell_dim, "obs":obs_dim, 
                              "ens":torch.arange(hdxf.shape[2], device=hdxf.device)})
    
    dataset["hdxf"]=(("cell", "obs", "ens"), hdxf)
    dataset["dep"]=(("cell", "obs"), dep)
    dataset["rloc"]=(("cell", "obs"), rloc)
    dataset["rdiag"]=(("cell", "obs"), rdiag)
    dataset["obs_mask"]   = (("cell", "obs"), obs_mask)
    dataset["parm_infl"]  = (("cell",), torch.ones_like(cell_dim))
    dataset["rdiag_wloc"] = (("cell",), torch.ones_like(cell_dim).bool())
    dataset["infl_update"]= (("cell",), torch.ones_like(cell_dim).bool())
    return dataset

def gather_obs(
        state_ds: Dataset,
        obs_valid: Dataset,
        tile_index: int,
        halo_i: int,
        halo_j: int,
        *,
        device: torch.device,
        chunk_size: int,
    ):
    """
    """
    obs_halo = filter_obs_to_haloed_tile(obs_valid, tile_index, 
                                         halo_i=halo_i, halo_j=halo_j)
    if obs_halo is None or obs_halo.sizes["obs"] == 0: return None

    coords = extract_coordinate_tensors(state_ds, obs_halo, f"pe{tile_index:06d}", 
                                        device=device, dtype=torch.float64)
    
    topk = chunked_topk_neighbors(
        coords["grid_xy"], coords["grid_z"],
        coords["obs_xy"],  coords["obs_z"],
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
        obs_halo,
        var_local_factor=1.0,
    )

    return assembled