from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from tqdm import tqdm
import sys

import pytest
import torch

CURRENT_DIR = Path(__file__).resolve().parent
SC23_DIR = CURRENT_DIR.parent
PKG_ROOT = SC23_DIR.parents[1]
PROJECT_ROOT = CURRENT_DIR.parents[3]

for path in (SC23_DIR, PKG_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from sc23 import init_sc23
from pyletkf import ScaleLetkfConverter, GridInterpolator, ObsOperator
from pyletkf.obs_op.radar import ref_operator, vr_operator, filter_ref, filter_vr
from test_io.letkf_dumps import load_state, load_obsda_setletkf, load_obsda_obsop

def load_neighbour_ops(pipeline, tile, device):
    """
    """
    neighbour_ops = {}

    for k,_,_ in tqdm(tile.neighbor_ranks()): 
        obs = local_hx(pipeline, tile.grid[k], device)
        obs = pipeline.obs_op.filter(obs)
        
        neighbour_ops[k]=obs.to("cpu")
        obs_state = None
        state = None
        torch.cuda.empty_cache()
    return neighbour_ops
    
def local_hx(pipeline, tile, device, filter_lev=True):
    """
    """
    obs = tile.read_obs(filter_lev=filter_lev).to(device)
    obs = tile.populate_obs_coords(obs)
    obs = tile.filter_obs_to_tile(obs)

    state = tile.read_state().to(device)
    state = tile.share_halos(state)
    state = pipeline.state_conv.model_to_da(state)

    obs_state = pipeline.interp.map_state_to_obs(obs.to(device), state)
    return pipeline.obs_op(obs_state.to(device))

@pytest.mark.parametrize("rank", [0, 5, 19])
def test_state_loading_matches_scale_dump(rank: int):
    """
    """
    grid, pipeline = init_sc23()
    tile = grid[rank]
    
    state = tile.strip_halo(tile.read_state().to(DEVICE)).to("cpu")
    state = pipeline.state_conv.model_to_da(state)
    
    dump_state = load_state(rank)

    torch.testing.assert_close(
        state["state"].transpose("variable", "ens", "z", "y", "x")\
                          .values.to(torch.float64),
        dump_state["state"].transpose("variable", "ens", "z", "y", "x")\
                           .values.to(torch.float64),
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    torch.testing.assert_close(
        state["height"].values.to(torch.float64).permute(2, 0, 1),
        dump_state["height"].values.to(torch.float64),
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    torch.testing.assert_close(
        state["topo"].values.to(torch.float64),
        dump_state["topo"].values.to(torch.float64),
        atol=1.0e-6,
        rtol=1.0e-6,
    )

@pytest.mark.parametrize("rank", [0, 5, 19])
def test_share_halos_matches_scale_dump(rank: int):
    def _slice_for_coords(new_coord, old_coord):
        new = torch.as_tensor(new_coord, dtype=torch.float64)
        old = torch.as_tensor(old_coord, dtype=torch.float64)
        start = int(torch.nonzero(old == new[0], as_tuple=False)[0])
        end = int(torch.nonzero(old == new[-1], as_tuple=False)[0]) + 1
        return slice(start, end)
    
    grid, pipeline = init_sc23()
    tile = grid[rank]
    shared_state = tile.share_halos(tile.read_state().to(DEVICE)).to("cpu")
    # Should load_state of neighboring tiles and verify against them

def test_obs_op(rank, device, eps=10**-6):
    grid, pipeline = init_sc23()
    tile = grid[rank]

    obs = local_hx(pipeline, tile, device, filter_lev=True)
    obs = pipeline.obs_op.filter(obs)
    
    obs_dump = load_obsda_obsop(rank)
    
    mask = torch.isin(obs_dump["obs"].values.to(device), obs["obs"].values)
    idxs = obs_dump["obs"].values.to(device)[mask]
    comp = obs["hx"].sel(obs=idxs)
    
    dump = obs_dump["ensval"].sel(obs=idxs)
    dump = dump.assign_coords(ens=comp["ens"]).to(device)
    
    abs_err =  (torch.abs(dump - comp).sum("ens") /\
               (torch.abs(dump).sum("ens") + eps))    
    
    return abs_err

def test_map_state_to_obs(rank, device, eps=10**-6):
    grid, pipeline = init_sc23()
    tile = grid[rank]
    
    obs = local_hx(pipeline, tile, device, filter_lev=True)
    obs = pipeline.obs_op.filter(obs)
    other_obs = load_neighbour_ops(pipeline, tile, device)

    shared_ops = tile.share_obs(obs, other_obs)
    obs_dump = load_obsda_setletkf(rank)

    ref_set = set(obs_dump["obs"].values.cpu().numpy().tolist())
    my_set = set(obs_dump["obs"].values.cpu().numpy().tolist())
    # What we should test:
    # assert my_set==ref_set 

    return shared_ops, obs_dump
    
def test_obs_op_filtering(rank, device, eps=10**-6):
    grid, pipeline = init_sc23()
    tile = grid[rank]

    obs = local_hx(pipeline, tile, device, filter_lev=True)
    obs = pipeline.obs_op.filter(obs)
    # What should I test against and how? Not sure