from __future__ import annotations
from typing import Mapping, Sequence
from tqdm.auto import tqdm

import torch
from xtensor import DataTensor, Dataset

from .grid_proj import filter_obs_to_tile
from ..obs_op import compute_all_hx
from ..obs_op.filter_obs import (
    ID_RADAR_REF,
    ID_RADAR_VR,
    PHARAD_TYP,
    RADAR_ZMAX,
    RADAR_ZMIN,
)
from ..params import DX, DY, KHALO, NX_TILE, NY_TILE

def _fractional_index_unit(size: int, coord: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    coord = coord - .5
    lo = coord.floor().long()
    frac = coord - lo.to(coord.dtype)
    return lo, frac

def _interpolate_height_columns(height: torch.Tensor, ix0, fx, iy0, fy) -> torch.Tensor:
    height_yx = height.permute(1, 2, 0)
    ix1 = torch.clamp(ix0 + 1, 0, height_yx.shape[1] - 1)
    iy1 = torch.clamp(iy0 + 1, 0, height_yx.shape[0] - 1)

    fx = fx.unsqueeze(1)
    fy = fy.unsqueeze(1)

    h00 = height_yx[iy0, ix0]
    h10 = height_yx[iy0, ix1]
    h01 = height_yx[iy1, ix0]
    h11 = height_yx[iy1, ix1]
    h0  = h00 * (1.0 - fx) + h10 * fx
    h1  = h01 * (1.0 - fx) + h11 * fx
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

def _bilinear_sample(field_yx: torch.Tensor, ix0, fx, iy0, fy) -> torch.Tensor:
    ix1 = torch.clamp(ix0 + 1, 0, field_yx.shape[1] - 1)
    iy1 = torch.clamp(iy0 + 1, 0, field_yx.shape[0] - 1)
    #fx = fx.unsqueeze(1)
    #fy = fy.unsqueeze(1)
    f00 = field_yx[iy0, ix0]
    f10 = field_yx[iy0, ix1]
    f01 = field_yx[iy1, ix0]
    f11 = field_yx[iy1, ix1]
    f0 = f00 * (1.0 - fx) + f10 * fx
    f1 = f01 * (1.0 - fx) + f11 * fx
    return f0 * (1.0 - fy) + f1 * fy

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

def _radar_height_mask(obs_tile: Dataset) -> torch.Tensor:
    lev = obs_tile["lev"].data
    elm = obs_tile["elm"].data
    typ = obs_tile["typ"].data
    is_radar = (typ == PHARAD_TYP) & ((elm == ID_RADAR_REF) | (elm == ID_RADAR_VR))
    if not torch.any(is_radar):
        return torch.ones_like(lev, dtype=torch.bool)
    height_ok = (lev >= RADAR_ZMIN) & (lev <= RADAR_ZMAX)
    return (~is_radar) | height_ok

def sample_state(
    state: DataTensor,
    height: DataTensor,
    ri_local: DataTensor,
    rj_local: DataTensor,
    lev: DataTensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state_cube = state.data  # (y, x, z, ens, var)
    height_tensor = height.data  # (z, y, x)
    ri = ri_local.data.to(torch.float64) #- .5
    rj = rj_local.data.to(torch.float64) #- .5
    lev_tensor = lev.data.to(torch.float64)

    ix0, fx = _fractional_index_unit(state_cube.shape[1], ri)
    iy0, fy = _fractional_index_unit(state_cube.shape[0], rj)
 
    columns = _interpolate_height_columns(height_tensor, ix0, fx, iy0, fy)
    min_height = columns[:, 0]
    max_height = columns[:, -1]
    lev_clamped = torch.clamp(lev_tensor, min_height, max_height)
    iz0, fz = _vertical_index_from_height(columns, lev_clamped)

    samples = _sample_cube(state_cube, ix0, fx, iy0, fy, iz0, fz)
    rk = iz0.to(torch.float64) + fz + KHALO
    valid_mask = (lev_tensor >= min_height) & (lev_tensor <= max_height)
    return samples, rk, valid_mask

def compute_obs_local_coords(obs: Dataset, state_ds: Dataset) -> tuple[DataTensor, DataTensor]:
    tile_i = int(state_ds.attrs.get("tile_i", 0))
    tile_j = int(state_ds.attrs.get("tile_j", 0))
    start_i = tile_i * NX_TILE
    start_j = tile_j * NY_TILE
    
    ri_local = obs["ri_global"] - start_i 
    rj_local = obs["rj_global"] - start_j 

    ri_local, rj_local = apply_halo_to_obs_local_coords(state_ds, ri_local, rj_local)

    return ri_local, rj_local

def apply_halo_to_obs_local_coords(state_ds, ri_local, rj_local):
    halo_meta  = state_ds.attrs.get("spatial_halo", {"x": (0, 0), "y": (0, 0)})
    local_halo = state_ds.attrs.get("halo_map",     {"x": (0, 0), "y": (0, 0)})
    left = int(local_halo.get("x", (0, 0))[0]) + int(halo_meta.get("x", (0, 0))[0])
    top  = int(local_halo.get("y", (0, 0))[0]) + int(halo_meta.get("y", (0, 0))[0])
    ri_local = ri_local + left
    rj_local = rj_local + top
    return ri_local, rj_local

    
def read_tile_hx(obs, state_ds):
    """
        
    """
    obs_tile = filter_obs_to_tile(obs, state_ds)
    if obs_tile is None or obs_tile.sizes["obs"] == 0: return (None, None)

    ri_local, rj_local = compute_obs_local_coords(obs, state_ds)
     
    state  = state_ds["state"].transpose("y", "x", "z", "ens", "variable")
    device = state.device
    height = state_ds["height"]
    
    samples, _, valid_mask = sample_state(
        state.to(device),
        height.to(device),
        ri_local,
        rj_local,
        obs_tile["lev"],
    )
    valid_mask = valid_mask & _radar_height_mask(obs_tile)
    
    obs_state = samples.permute(2, 0, 1).to(device=device, dtype=torch.float64)
    
    hx = compute_all_hx(obs_state, obs_tile)
    
    invalid_mask = (~valid_mask).to(device=device)
    if invalid_mask.any():
        hx[invalid_mask] = 0.0
        
    obs_indices = obs_tile["obs"].data.long()
    return hx, obs_indices

def read_all_tile_hx_sequentially(obs, states):
    hxs, obs_idxs = zip(*[read_tile_hx(obs, state_ds, tile_index) \
                        for tile_index, state_ds in tqdm(states.items())])
    return hxs, obs_idxs

def assemble_all_hx(obs, states, hxs, obs_idxs):
    """
        Build the full hx matrix (obs × members) by looping over tiles and
        running the observation operator on the locally interpolated state.
    """
    n_obs = obs.sizes["obs"]
    n_members = next(iter(states.values())).sizes["ens"]
    ens = next(iter(states.values()))["ens"]

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
    ens = next(iter(states.values()))["ens"]
    
    hx_full = torch.full((n_obs, n_members), float("nan"), device=device, dtype=torch.float64)
    
    for tile_index, state_ds in tqdm(states.items()):
        hx, obs_indices = read_tile_hx(obs, state_ds, tile_index)
        hx_full[obs_indices] = hx

    obs = obs.assign_coords(ens=ens)
    obs["hx"] = (("obs", "ens"), hx_full)
    obs["hx_mean"] = obs["hx"].mean("ens")
    return obs
