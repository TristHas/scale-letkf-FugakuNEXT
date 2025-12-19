import torch
import xarray as xr

from .radar_reflectivity import ref_operator, convert_raw_to_dbz
from .radar_vr import vr_operator

ID_RADAR_VR = 4002
ID_RADAR_REF = 4001  # reflectivity observations

def compute_all_hx(obs_state: torch.Tensor, obs_radar: xr.Dataset) -> torch.Tensor:
    device = obs_state.device
    dtype = obs_state.dtype
    obs_type = torch.as_tensor(obs_radar["elm"].values, device=device, dtype=torch.int64)
    hx_out = torch.empty(obs_state.shape[1], obs_state.shape[2], device=device, dtype=dtype)

    mask_ref = obs_type == ID_RADAR_REF
    if mask_ref.any():
        hx_out[mask_ref] = ref_operator(obs_state[:, mask_ref])

    mask_vr = obs_type == ID_RADAR_VR
    
    if mask_vr.any():
        obs_vr = obs_radar.sel(obs=mask_vr.cpu().numpy())
        obs_state_vr = obs_state[:, mask_vr]
        hx = vr_operator(obs_state_vr, obs_vr)
        hx_out[mask_vr] = hx

    return hx_out
