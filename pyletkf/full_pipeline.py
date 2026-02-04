from pathlib import Path
from tqdm.auto import tqdm

import torch

from .params import MEMBERS, PRC_NUM_X, PRC_NUM_Y
from .io import load_haloed_letkf_state, load_obsda_sorted, load_radar
from .io.state import strip_state_halo
from .letkf_core import letkf_core
from .post_letkf import apply_analysis_update, create_config
from .pre_letkf import pre_letkf

def process_full_pipeline(tile_idx=0, device="cuda:1"):
    """
    """
    #scale_var = [ 'DENS', 'MOMZ', 'MOMX', 'MOMY', 'RHOT', 'QV', 'QC', 'QR', 'QI', 'QS', 'QG']
    letkf_var = [ 'U', 'V', 'W', 'T', 'P', 'QV', 'QC', 'QR', 'QI', 'QS', 'QG']
    n_tiles  = PRC_NUM_X * PRC_NUM_Y
    dump_dir = Path("result/SC23/20210730060030/letkf_dump")
    
    pe_tag = f"pe{str(tile_idx).zfill(6)}"
    obs = load_radar()
    obs["obs"] = obs["obs"]-1
    
    states = {
        pe: load_haloed_letkf_state(dump_dir, f"pe{pe:06d}", "anal_f").sel(ens=list(MEMBERS))
        for pe in tqdm(range(1))
    }

    # Loop over the different tiles are done inside this function.
    da_state = pre_letkf(obs, states)
    
    stripped_states = {pe: strip_state_halo(ds) for pe, ds in states.items()}
    
    # Core solver step
    da_params = letkf_core(da_state[tile_idx])
    
    # Format for applying the update step
    state = stripped_states[tile_idx]["state"].transpose("z", "y", "x", "ens", "variable").values.contiguous()
    
    gues_members = state.view(-1, state.shape[-2], state.shape[-1])
    gues_members = gues_members.permute(2,0,1)
    da_params["gues_members"] = (["variables", "cell", "member_col"], gues_members)
    da_params["gues_mean"] = da_params["gues_members"].mean("member_col")
    
    update_config = create_config(device=device)
    # Apply LETKF update
    outputs = torch.zeros_like(gues_members)
    
    for i,var in enumerate(tqdm(letkf_var)):
        data = da_params.isel(variables=i).to(device)
        param = update_config.isel(variables=0).to(device)
        out = apply_analysis_update(data, param)
        outputs[i]=out['anal_members']
        
    return outputs, obs, stripped_states


def postrprocess_outputs(outputs, state):
    import numpy as np
    import xarray as xr
    import hvplot.xarray
    
    nvar = state.sizes["variable"]
    nz = state.sizes["z"]
    ny = state.sizes["y"]
    nx = state.sizes["x"]
    
    outputs = outputs.view(nvar, nz, ny, nx, 2)
    
    coords = dict(state.coords)
    coords["ens"] = np.array(MEMBERS)
    coords["variable"] = np.array(coords["variable"])
    
    out_da = xr.DataArray(outputs.cpu(), name="out_da",
                          coords=coords, 
                          dims=["variable", "z", "y", "x", "ens"])
    
    return out_da
