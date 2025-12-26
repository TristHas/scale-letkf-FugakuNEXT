import math
import torch
from xtensor import Dataset, DataTensor

from ..params import (RADIUS, FACT, BASE_LON,
                      DX, DY, GRID_PARAMS, 
                      NX_TILE, NY_TILE,
                      IHALO, JHALO)

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
    ri = (x - cxg0) / DX + 1.0
    rj = (y - cyg0) / DY + 1.0
    return ri, rj

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
    obs["ri_local"] = (("obs",), ((obs["ri_global"].values - 1) % NX_TILE) + 1)
    obs["rj_local"] = (("obs",), ((obs["rj_global"].values - 1) % NY_TILE) + 1)
    return obs

def compute_obs_grid_idx(obs: Dataset, grid_params=GRID_PARAMS) -> Dataset:
    obs = populate_obs_global_indices(obs, grid_params=grid_params)
    obs = populate_obs_local_idx(obs)
    return obs