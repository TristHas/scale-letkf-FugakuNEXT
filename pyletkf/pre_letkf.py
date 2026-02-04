from __future__ import annotations
import math
from tqdm.auto import tqdm

import numpy as np
import torch
import xarray as xr

from .spatial.grid_proj import compute_obs_grid_idx
from .spatial.map_obs_to_state import gather_obs
from .spatial.map_state_to_obs import read_tile_hx, assemble_all_hx

from .obs_op import filter_sc23_obs
from .params import (
    HORI_LOCAL_RADAR_OBSNOREF,
    VERT_LOCAL_RADAR_OBSNOREF,
    MAX_OBS_PER_GRID,
    DIST_ZERO_FAC,
    DX,
    DY,
    MEMBERS,
)
from .io.state import strip_state_halo

def pre_letkf(obs, states, device=None):
    
    # Step 0: Populate obs with their grid indices
    obs = compute_obs_grid_idx(obs)
    device = obs["dat"].device
    
    # Step 1: Populate observations with hx
    hxs, obs_idxs = zip(
        *[
            read_tile_hx(obs, state_ds.to(device), tile_index)
            for tile_index, state_ds in tqdm(states.items())
        ]
    )
    obs = assemble_all_hx(obs, states, hxs, obs_idxs)
    
    # Step 2: Filter obs
    obs_valid = filter_sc23_obs(obs)
    stripped_states = {tile_index: strip_state_halo(state_ds) for tile_index, state_ds in tqdm(states.items())}
    
    # Step 3: Populate each state cell with the nearest observation hx
    halo_i = math.ceil(HORI_LOCAL_RADAR_OBSNOREF * DIST_ZERO_FAC / DX)
    halo_j = math.ceil(HORI_LOCAL_RADAR_OBSNOREF * DIST_ZERO_FAC / DY)
    
    results = {
        tile_index: gather_obs(
            state_ds.to(device),
            obs_valid,
            tile_index,
            halo_i,
            halo_j,
        ).to(device="cpu")
        for tile_index, state_ds in tqdm(stripped_states.items())
    }
    
    return results
