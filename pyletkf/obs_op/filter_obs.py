from __future__ import annotations
import torch
from xtensor import DataTensor, Dataset

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

def _to_tensor(value, device, dtype=torch.float64):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def filter_sc23_obs(obs_ds: Dataset):
    device = obs_ds["dat"].data.device
    nobs = len(obs_ds["obs"].values)
    
    elm = obs_ds["elm"].data
    typ = obs_ds["typ"].data
    dat = obs_ds["dat"].data
    err = obs_ds["err"].data
    lev = obs_ds["lev"].data
    hx_vals = obs_ds["hx_mean"].data
    hx_all = obs_ds["hx"].data
    
    keep = torch.zeros(nobs, dtype=torch.bool, device=device)
    mem_counts = (hx_all > RADAR_REF_THRES_DBZ).sum(dim=1)
    height_ok = (lev >= RADAR_ZMIN) & (lev <= RADAR_ZMAX)
    mask_ref = (elm == ID_RADAR_REF) & (typ == PHARAD_TYP) & height_ok
    
    ref_idx = torch.nonzero(mask_ref, as_tuple=False).squeeze(1)
    if ref_idx.numel() > 0:
        mem_good = torch.ones(ref_idx.numel(), dtype=torch.bool, device=device)
        if mem_counts is not None:
            mem_needed = torch.where(
                dat[mask_ref] > RADAR_REF_THRES_DBZ,
                torch.tensor(MIN_RADAR_REF_MEMBER_OBSRAIN, device=device),
                torch.tensor(MIN_RADAR_REF_MEMBER_OBSNORAIN, device=device),
            )
            mem_good = mem_counts[mask_ref] >= mem_needed
        innov_ref = dat[mask_ref] - hx_vals[mask_ref]
        good_ref = torch.abs(innov_ref) <= GROSS_ERROR_RADAR_REF * err[mask_ref]
        selected = mem_good & good_ref
        keep[ref_idx[selected]] = True

    mask_vr = (elm == ID_RADAR_VR) & (typ == PHARAD_TYP) & height_ok

    vr_idx = torch.nonzero(mask_vr, as_tuple=False).squeeze(1)
    if vr_idx.numel() > 0:
        innov_vr = dat[mask_vr] - hx_vals[mask_vr]
        good_vr = torch.abs(innov_vr) <= GROSS_ERROR_RADAR_VR * err[mask_vr]
        keep[vr_idx[good_vr]] = True

    kept_idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
    obs_valid = obs_ds.isel(obs=kept_idx.cpu())
    
    obs_valid = obs_valid.assign(obs_orig=DataTensor(obs_valid["obs"].data.clone(), {"obs": obs_valid["obs"].coords["obs"]}, ("obs",)))
    obs_valid = obs_valid.assign_coords(obs=torch.arange(obs_valid.sizes["obs"], dtype=torch.int64))

    return obs_valid
