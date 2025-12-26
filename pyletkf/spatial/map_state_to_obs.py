from __future__ import annotations
from typing import Mapping, Sequence
import math
from tqdm.auto import tqdm

import torch
from xtensor import DataTensor, Dataset

from ..obs_op import compute_all_hx
from ..params import PRC_NUM_X, NX_TILE, NY_TILE, IHALO, JHALO, KHALO

def _fractional_index_unit(size: int, coord: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    coord0 = torch.clamp(coord - 1.0, 0.0, size - 1.0 - 1.0e-6)
    lo = coord0.floor().long()
    frac = coord0 - lo.to(coord0.dtype)
    return lo, frac

def _interpolate_height_columns(height: torch.Tensor, ix0, fx, iy0, fy) -> torch.Tensor:
    # height shape: (z, y, x) -> transpose to (y, x, z)
    height_yx = height.permute(1, 2, 0)
    ix1 = torch.clamp(ix0 + 1, 0, height_yx.shape[1] - 1)
    iy1 = torch.clamp(iy0 + 1, 0, height_yx.shape[0] - 1)

    fx = fx.unsqueeze(1)
    fy = fy.unsqueeze(1)

    h00 = height_yx[iy0, ix0]
    h10 = height_yx[iy0, ix1]
    h01 = height_yx[iy1, ix0]
    h11 = height_yx[iy1, ix1]
    h0 = h00 * (1.0 - fx) + h10 * fx
    h1 = h01 * (1.0 - fx) + h11 * fx
    return h0 * (1.0 - fy) + h1 * fy

def _vertical_index_from_height(columns: torch.Tensor, lev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    nobs, nz = columns.shape
    expanded = lev.unsqueeze(1).expand(-1, nz)
    mask = expanded >= columns
    idx = torch.clamp(mask.sum(dim=1) - 1, 0, nz - 2)
    v0 = columns[torch.arange(nobs), idx]
    v1 = columns[torch.arange(nobs), idx + 1]
    denom = v1 - v0
    frac = torch.where(denom > 0.0, (lev - v0) / denom, torch.zeros_like(v0))
    return idx, frac

def _sample_cube(field5d, ix0, fx, iy0, fy, iz0, fz):
    ix1 = torch.clamp(ix0 + 1, 0, field5d.shape[1] - 1)
    iy1 = torch.clamp(iy0 + 1, 0, field5d.shape[0] - 1)
    iz1 = torch.clamp(iz0 + 1, 0, field5d.shape[2] - 1)

    fx = fx[:, None, None]
    fy = fy[:, None, None]
    fz = fz[:, None, None]

    c000 = field5d[iy0, ix0, iz0]
    c100 = field5d[iy0, ix1, iz0]
    c010 = field5d[iy1, ix0, iz0]
    c110 = field5d[iy1, ix1, iz0]
    c001 = field5d[iy0, ix0, iz1]
    c101 = field5d[iy0, ix1, iz1]
    c011 = field5d[iy1, ix0, iz1]
    c111 = field5d[iy1, ix1, iz1]

    c00 = c000 * (1.0 - fx) + c100 * fx
    c01 = c010 * (1.0 - fx) + c110 * fx
    c10 = c001 * (1.0 - fx) + c101 * fx
    c11 = c011 * (1.0 - fx) + c111 * fx

    c0 = c00 * (1.0 - fy) + c01 * fy
    c1 = c10 * (1.0 - fy) + c11 * fy

    return c0 * (1.0 - fz) + c1 * fz

def sample_state(
    state: DataTensor,
    height: DataTensor,
    ri_local: DataTensor,
    rj_local: DataTensor,
    lev: DataTensor,
):
    state_cube = state.data  # (y, x, z, ens, var)
    height_tensor = height.data  # (z, y, x)
    ri = ri_local.data.to(torch.float64)
    rj = rj_local.data.to(torch.float64)
    lev_tensor = lev.data.to(torch.float64)

    ix0, fx = _fractional_index_unit(state_cube.shape[1], ri)
    iy0, fy = _fractional_index_unit(state_cube.shape[0], rj)

    height_cols = _interpolate_height_columns(height_tensor, ix0, fx, iy0, fy)
    iz0, fz = _vertical_index_from_height(height_cols, lev_tensor)

    samples = _sample_cube(state_cube, ix0, fx, iy0, fy, iz0, fz)
    rk = iz0.to(torch.float64) + fz + KHALO
    return samples, rk

def _subset_obs(obs: Dataset, mask: torch.Tensor) -> Dataset | None:
    indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
    if indices.numel() == 0:
        return None
    return obs.isel(obs=indices)

def filter_obs_to_tile_index(obs: Dataset, tile_index: int) -> Dataset | None:
    target_i = tile_index % PRC_NUM_X
    target_j = tile_index // PRC_NUM_X

    ri_global = obs["ri_global"].data
    rj_global = obs["rj_global"].data

    rank_i = torch.floor((ri_global - 1.0) / NX_TILE).long()
    rank_j = torch.floor((rj_global - 1.0) / NY_TILE).long()

    mask = (rank_i == target_i) & (rank_j == target_j)
    return _subset_obs(obs, mask)

def read_all_tile_hx_sequentially(obs, states):
    hxs, obs_idxs = zip(*[read_tile_hx(obs, state_ds, tile_index) \
                        for tile_index, state_ds in tqdm(states.items())])
    return hxs, obs_idxs

def read_tile_hx(obs, state_ds, tile_index):
    """
        
    """
    obs_tile = filter_obs_to_tile_index(obs, tile_index)
    if obs_tile is None or obs_tile.sizes["obs"] == 0: return (None, None)
    
    state = state_ds["state"].transpose("y", "x", "z", "ens", "variable")
    device = state.device
    
    samples, _ = sample_state(
      state.to(device),
      state_ds["height"].to(device),
      obs_tile["ri_local"],
      obs_tile["rj_local"],
      obs_tile["lev"],
    )
    
    obs_state = samples.permute(2, 0, 1).to(device=device, dtype=torch.float64)
    hx = compute_all_hx(obs_state, obs_tile)
    obs_indices = obs_tile["obs"].data.long()
    return hx, obs_indices

def assemble_all_hx(obs, states, hxs, obs_idxs):
    """
    Build the full hx matrix (obs × members) by looping over tiles and
    running the observation operator on the locally interpolated state.
    """
    n_obs = obs.sizes["obs"]
    n_members = next(iter(states.values())).sizes["ens"]
    ens = next(iter(states.values()))._coords["ens"]

    hx_full = torch.full((n_obs, n_members), float("nan"), 
                         device=hxs[0].device, 
                         dtype=torch.float64)

    for hx, obs_idx in zip(hxs, obs_idxs):
        if obs_idx is not None:
            hx_full[obs_idx] = hx

    obs = obs.assign_coords(ens=ens)
    obs["hx"] = (("obs", "ens"), hx_full)
    obs["hx_mean"] = obs["hx"].mean("ens")
    return obs

def populate_all_hx_sequentially(
      obs: Dataset,
      states: Mapping[int, Dataset],
      *,
      device: torch.device,
  ) -> torch.Tensor:
    """
    Build the full hx matrix (obs × members) by looping over tiles and
    running the observation operator on the locally interpolated state.
    """
    n_obs = obs.sizes["obs"]
    n_members = next(iter(states.values())).sizes["ens"]
    ens = next(iter(states.values()))._coords["ens"]
    
    hx_full = torch.full((n_obs, n_members), float("nan"), device=device, dtype=torch.float64)
    
    for tile_index, state_ds in tqdm(states.items()):
        hx, obs_indices = read_tile_hx(obs, state_ds, tile_index)
        hx_full[obs_indices] = hx

    obs = obs.assign_coords(ens=ens)
    obs["hx"] = (("obs", "ens"), hx_full)
    obs["hx_mean"] = obs["hx"].mean("ens")
    return obs