from pathlib import Path

import torch

from pyletkf.io.state import load_haloed_letkf_state
from pyletkf.io.radar import load_radar
from pyletkf.io.letkf_dumps import load_obsda_sorted

from pyletkf.obs_op import filter_sc23_obs
from pyletkf.spatial.grid_proj import compute_obs_grid_idx, filter_obs_to_tile
from pyletkf.spatial.map_state_to_obs import read_tile_hx

from pyletkf.obs_op.filter_obs import ID_RADAR_REF, ID_RADAR_VR

def filter_sc23_obs(device="cuda:1", target_tile=5):
    # Settings
    DUMP_PATH  = Path("result/SC23/20210730060030/letkf_dump")
    RADAR_PATH = Path("result/SC23/obs_radar/radar_20210730060030.dat")
    target_pe = f"pe{str(target_tile).zfill(6)}"
    
    radar_dataset = load_radar(RADAR_PATH).to(device)
    radar_dataset["obs"] = radar_dataset["obs"] + 1
    radar_dataset = radar_dataset.to(device) 
    
    state_dataset = load_haloed_letkf_state(DUMP_PATH, target_pe, "anal_f", 
                                            halo_x=1, halo_y=1).to(device)
    
    obs = compute_obs_grid_idx(radar_dataset)
    
    obs_tile = filter_obs_to_tile(obs, state_dataset)
    obs_tile = read_tile_hx(obs_tile, state_dataset)
    
    x = filter_sc23_obs(obs)
    x_ref = x.isel(obs=x["elm"]==ID_RADAR_REF)
    x_vr  = x.isel(obs=x["elm"]==ID_RADAR_VR)
    
    sorted_ds = load_obsda_sorted(DUMP_PATH, target_pe)
    idx_fortran = torch.as_tensor(sorted_ds["idx"].values, dtype=torch.int64) + 1
    
    y = filter_obs_to_tile(obs.sel(obs=idx_fortran), state_dataset)
    y_vr = y.isel(obs=y["elm"]==ID_RADAR_VR)
    y_vr = y_vr.isel(obs=~torch.isnan(y_vr["dat"]))
    y_ref = y.isel(obs=y["elm"]==ID_RADAR_REF)
    y_ref = y_ref.isel(obs=~torch.isnan(y_ref["dat"]))
    
    assert (y_ref["obs"].to_pandas().sort_index().values==x_ref["obs"].to_pandas().sort_index().values).all()
    assert (y_vr["obs"].to_pandas().sort_index().values==x_vr["obs"].to_pandas().sort_index().values).all()