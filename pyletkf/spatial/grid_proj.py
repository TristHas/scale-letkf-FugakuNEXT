import math
import torch
from xtensor import Dataset, DataTensor

from ..params import (RADIUS, FACT, BASE_LON,
                      DX, DY, GRID_PARAMS, 
                      NX_TILE, NY_TILE,
                      IHALO, JHALO,
                      PRC_NUM_X, PRC_NUM_Y,
                      TOTAL_NX, TOTAL_NY)

def tile_bounds(state_dataset, 
                halo_i: int = 0, 
                halo_j: int = 0) -> tuple[float, float, float, float]:
    """
    """
    tile_i  = state_dataset.attrs["tile_i"]
    tile_j  = state_dataset.attrs["tile_j"]
    start_i = tile_i * NX_TILE - halo_i
    end_i   = (tile_i + 1) * NX_TILE + halo_i
    start_j = tile_j * NY_TILE - halo_j
    end_j   = (tile_j + 1) * NY_TILE + halo_j
    return start_i, end_i, start_j, end_j

def filter_obs_to_tile(obs: Dataset, dataset_ds: Dataset,
                      halo_i: int = 0, 
                      halo_j: int = 0) -> Dataset | None:
    """
    """
    ri_min, ri_max, rj_min, rj_max = tile_bounds(dataset_ds, halo_i, halo_j)
    ri_global = obs["ri_global"].data 
    rj_global = obs["rj_global"].data
    mask = (
          (ri_global >= ri_min)
        & (ri_global <= ri_max)
        & (rj_global >= rj_min)
        & (rj_global <= rj_max)
    )
    subset = obs.isel(obs=mask)
    return subset
    
def _lonlat_to_grid_indices(
        lon_deg: torch.Tensor,
        lat_deg: torch.Tensor,
        *,
        base_x: float,
        param_y: float,
        cxg0: float,
        cyg0: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    lon_rad = torch.deg2rad(lon_deg)
    lat_rad = torch.deg2rad(lat_deg)
    x = base_x + RADIUS * FACT * (lon_rad - BASE_LON)
    latrot = 0.5 * math.pi - lat_rad
    dist = torch.reciprocal(torch.tan(0.5 * latrot))
    y = param_y + RADIUS * FACT * torch.log(dist)
    ri = (x - cxg0) / DX #+ 1.0
    rj = (y - cyg0) / DY #+ 1.0
    return ri-1.5, rj-1.5

def populate_obs_global_indices(obs: Dataset, grid_params=GRID_PARAMS) -> Dataset:
    lon = obs["lon"].data
    lat = obs["lat"].data
    ri, rj = _lonlat_to_grid_indices(
      lon,
      lat,
      base_x=grid_params["base_x"],
      param_y=grid_params["param_y"],
      cxg0=grid_params["cxg0"],
      cyg0=grid_params["cyg0"],
    )
    return obs.assign(ri_global=DataTensor(ri, obs["lon"].coords, ("obs",)),
                      rj_global=DataTensor(rj, obs["lat"].coords, ("obs",)))

def populate_obs_local_idx(obs: Dataset) -> Dataset:
    obs["ri_local"] = (("obs",), ((obs["ri_global"].values) % NX_TILE))
    obs["rj_local"] = (("obs",), ((obs["rj_global"].values) % NY_TILE))
    return obs

def compute_obs_grid_idx(obs: Dataset, grid_params=GRID_PARAMS) -> Dataset:
    obs = populate_obs_global_indices(obs, grid_params=grid_params)
    return obs