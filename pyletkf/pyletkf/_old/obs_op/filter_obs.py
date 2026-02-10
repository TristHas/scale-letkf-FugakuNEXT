from __future__ import annotations
import torch
import xtensor as xt

ID_RADAR_REF = 4001
ID_RADAR_VR = 4002
PHARAD_TYP = 22
GROSS_ERROR_RADAR_REF = 5.0
GROSS_ERROR_RADAR_VR = 5.0
RADAR_REF_THRES_DBZ = 10.0
MIN_RADAR_REF_MEMBER_OBSRAIN = 1
MIN_RADAR_REF_MEMBER_OBSNORAIN = 1
RADAR_ZMIN = 500.0
RADAR_ZMAX = 11000.0

def filter_obs_wrt_height(obs_ds):
    height_ok  = (obs_ds["lev"] >= RADAR_ZMIN)   & (obs_ds["lev"] <= RADAR_ZMAX)
    return obs_ds.isel(obs=height_ok)
    
def filter_ref_wrt_val(obs_ds):
    mask = (obs_ds["elm"] == ID_RADAR_REF) & (obs_ds["typ"] == PHARAD_TYP) 
    obs_ds = obs_ds.isel(obs=mask)
    
    mem_counts = (obs_ds["hx"] > RADAR_REF_THRES_DBZ).sum(dim="ens")
    mem_needed = torch.where(
        obs_ds["dat"].data > RADAR_REF_THRES_DBZ,
        MIN_RADAR_REF_MEMBER_OBSRAIN,
        MIN_RADAR_REF_MEMBER_OBSNORAIN
    )
    mem_good = mem_counts >= mem_needed
    return obs_ds.isel(obs=mem_good)

def filter_vr_wrt_val(obs_ds):
    mask = (obs_ds["elm"] == ID_RADAR_VR) & (obs_ds["typ"] == PHARAD_TYP) 
    obs_ds = obs_ds.isel(obs=mask)
    
    innov_vr = obs_ds["dat"] - obs_ds["hx_mean"]
    good_vr = torch.abs(innov_vr) <= GROSS_ERROR_RADAR_VR * obs_ds["err"]
    return obs_ds.isel(obs=good_vr)

def filter_sc23_obs(obs_ds):
    x = filter_obs_wrt_height(obs_ds)
    x_vr  = filter_vr_wrt_val(x)
    x_ref = filter_ref_wrt_val(x)
    return xt.concat([x_ref, x_vr], dim="obs")
