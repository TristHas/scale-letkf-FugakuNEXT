from pathlib import Path
from tqdm.auto import tqdm

import torch
import xtensor as xt

import sys
sys.path.append("../../")

from sc23 import sc23_grid
from pyletkf import ScaleLetkfConverter, GridInterpolator, LETKF, ObsOperator
from pyletkf.obs_op.radar import ref_operator, vr_operator, filter_ref, filter_vr

def preload_all_states(rank=None):
    """
    """
    grid  = sc23_grid()
    ranks = list(range(20)) if rank is None else\
            [x[0] for x in grid.neighbor_ranks(tile)]
    for i in tqdm(ranks):
        state = grid[i].read_state()

# Initialize all neighbourhood tiles observations
def load_neighbour_ops(tile):
    """
    """
    neighbour_ops = {}
    state_conv = ScaleLetkfConverter()
    interp = GridInterpolator(tile.grid)
    obs_op = ObsOperator({4001: ref_operator, 4002: vr_operator}, 
                         {4001: filter_ref, 4002: filter_vr})

    for k,_,_ in tqdm(tile.neighbor_ranks()): 
        ctx_ = tile.grid[k]
        ctx_.device = "cuda:7"
    
        obs = ctx_.read_obs().to(ctx_.device)
        obs = ctx_.populate_obs_coords(obs)
        obs = ctx_.filter_obs_to_tile(obs)
    
        state = ctx_.read_state().to(ctx_.device)
        state = ctx_.share_halos(state)
        state = state_conv.model_to_da(state)
        
        obs = interp.map_state_to_obs(obs, state)
        obs = obs_op(obs.to(ctx_.device))
        obs = obs_op.filter(obs)
        
        neighbour_ops[k]=obs.to("cpu")
        obs_state = None
        state = None
        torch.cuda.empty_cache()
    return neighbour_ops

def init_sc23():
    grid = sc23_grid()
    state_conv = ScaleLetkfConverter()
    interp = GridInterpolator(grid)
    obs_op = ObsOperator({4001: ref_operator, 4002: vr_operator}, 
                         {4001: filter_ref, 4002: filter_vr})
    core = LETKF()
    return Pipeline(grid, obs_op, interp, core, state_conv)

rank = 5
device = "cuda:7"

ctx  = grid[rank].to(device)

# Preload the tiles needed around
preload_all_states()
neighbour_ops = load_neighbour_ops(ctx)

# Run the target tile workflow
obs = ctx.read_obs().to(ctx.device)
obs = ctx.populate_obs_coords(obs)
obs = ctx.filter_obs_to_tile(obs)

state = ctx.read_state().to(ctx.device)
state = ctx.share_halos(state)
state = state_conv.model_to_da(state)

obs_state = interp.map_state_to_obs(obs, state)
state = ctx.strip_halo(state)
obs = obs_op(obs_state.to(ctx.device))
obs = obs_op.filter(obs)

obs = ctx.share_obs(obs, neighbour_ops.values())

# Remaining steps to test
da_state  = interp.map_obs_to_state(obs, state)
da_params = core.infer_update_params(da_state)
output    = core.apply_update(state, da_params)