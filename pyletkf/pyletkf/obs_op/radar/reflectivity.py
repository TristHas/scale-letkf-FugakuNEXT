import numpy as np
import torch
from .utils import linear_reflectivity_method3

def ref_operator(obs, use_melt=False, 
                 use_t08_rs2014=False,
                 min_radar_ref_dbz=10.,
                 low_res_shift=-5.):
    """
    """
    temp, press, qr, qs, qg = obs["state"].sel(variable=['T', 'P', 'QR', 'QS', 'QG'])\
                                          .values.permute(2,0,1)
    radar_lin = linear_reflectivity_method3(
        qr, qs, qg,
        temp, press,
        use_melt=use_melt,
        use_t08_rs2014=use_t08_rs2014,
    )

    hx = torch.full_like(radar_lin, min_radar_ref_dbz + low_res_shift)
    min_ref_linear = 10.0 ** (min_radar_ref_dbz / 10.0)
    valid = radar_lin >= min_ref_linear
    hx[valid] = 10.0 * torch.log10(radar_lin[valid])
    return hx

def filter_ref(obs, 
               radar_dbz_thres=10.,
               n_ens_rain_thres=1,
               n_ens_norain_thres=1,
               **kwargs
              ):
    """
    """
    n_ens_count = (obs["hx"] > radar_dbz_thres).sum("ens").data
    obs_rain = obs["dat"].data > radar_dbz_thres
    
    n_ens_needed = torch.where(
        obs_rain, 
        n_ens_rain_thres, 
        n_ens_norain_thres, 
    ) 
    return n_ens_count >= n_ens_needed
