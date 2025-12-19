from __future__ import annotations
import numpy as np
import torch
import xarray as xr

GROSS_ERROR_RADAR_REF = 5.0
PHARAD_TYP = 22
GROSS_ERROR_RADAR_VR = 5.0
RADAR_REF_THRES_DBZ = 10.0
MIN_RADAR_REF_MEMBER_OBSRAIN = 1
MIN_RADAR_REF_MEMBER_OBSNORAIN = 1
RADAR_ZMIN = 500.0
RADAR_ZMAX = 11000.0
ID_RADAR_REF = 4001
ID_RADAR_VR = 4002

def _to_numpy(arr) -> np.ndarray:
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)

def filter_sc23_obs(
    obs_ds: xr.Dataset, hx: xr.DataArray | np.ndarray | torch.Tensor | None = None
) -> xr.Dataset:
    """Apply SC23 reflectivity QC to the raw obs array."""
    elm = obs_ds["elm"].values
    typ = obs_ds["typ"].values
    dat = obs_ds["dat"].values
    err = obs_ds["err"].values
    lev = obs_ds["lev"].values
    nobs = obs_ds.sizes["obs"]
    keep = np.zeros(nobs, dtype=bool)

    has_hx = hx is not None
    hx_vals = np.zeros(nobs, dtype=np.float64)
    hx_all = None
    if hx is not None:
        hx_arr = _to_numpy(hx)
        if hx_arr.shape[0] != nobs:
            raise ValueError("hx first dimension must match number of observations")
        if hx_arr.ndim == 1:
            hx_vals = hx_arr.astype(np.float64, copy=False)
        elif hx_arr.ndim == 2:
            hx_all = hx_arr.astype(np.float64, copy=False)
            hx_vals = hx_all.mean(axis=1)
        else:
            raise ValueError("hx must be a 1D or 2D array")
    mem_counts = None
    if hx_all is not None:
        mem_counts = np.sum(hx_all > RADAR_REF_THRES_DBZ, axis=1)

    height_ok = (lev >= RADAR_ZMIN) & (lev <= RADAR_ZMAX)

    mask_ref = (elm == ID_RADAR_REF) & (typ == PHARAD_TYP) & height_ok
    if not has_hx:
        keep |= mask_ref
    else:
        mem_good = np.ones(np.count_nonzero(mask_ref), dtype=bool)
        if mem_counts is not None:
            mem_needed = np.where(
                dat[mask_ref] > RADAR_REF_THRES_DBZ,
                MIN_RADAR_REF_MEMBER_OBSRAIN,
                MIN_RADAR_REF_MEMBER_OBSNORAIN,
            )
            mem_good = mem_counts[mask_ref] >= mem_needed
        innov_ref = dat[mask_ref] - hx_vals[mask_ref]
        good_ref = np.abs(innov_ref) <= GROSS_ERROR_RADAR_REF * err[mask_ref]
        selected = mem_good & good_ref
        keep[np.where(mask_ref)[0][selected]] = True

    mask_vr = (elm == ID_RADAR_VR) & (typ == PHARAD_TYP) & height_ok
    if not has_hx:
        keep |= mask_vr
    else:
        innov_vr = dat[mask_vr] - hx_vals[mask_vr]
        good_vr = np.abs(innov_vr) <= GROSS_ERROR_RADAR_VR * err[mask_vr]
        keep[np.where(mask_vr)[0][good_vr]] = True

    obs_valid = obs_ds.isel(obs=keep)
    obs_valid = obs_valid.assign(
        obs_orig=("obs", obs_valid["obs"].values)
    )
    obs_valid = obs_valid.assign_coords(
        obs=np.arange(obs_valid.sizes["obs"])
    )

    return obs_valid, hx[keep]