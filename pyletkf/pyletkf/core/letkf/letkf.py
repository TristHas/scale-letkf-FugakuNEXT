import math
from tqdm import tqdm

import torch
import xtensor as xt

from .ops import letkf_core, letkf_update

RELAX_SPREAD_OUT = False
Q_UPDATE_TOP = 30000.0
Q_SPRD_MAX = 0.5
    
class LETKF():
    def __init__(self, 
                 param_infl=1,
                 eig_val_clamp=10**-12, 
                 eig_batch_size=50_000,
                 sigma_b=.04,
                 beta=0,
                 alpha_rtpp=0.95,
                 alpha_rtps=0,
            ):
        """
        """
        self.param_infl=param_infl
        self.eig_val_clamp=eig_val_clamp
        self.eig_batch_size=eig_batch_size
        self.sigma_b=sigma_b
        self.beta=beta
        self.alpha_rtpp=alpha_rtpp
        self.alpha_rtps=alpha_rtps

    def infer_update_params(self, ds):
        """
        """
        W_a, w_a = letkf_core(
            ds["dep"].values, ds["hx_d"].values,
            ds["rdiag"].values, ds["obs_mask"].values,
            param_infl=self.param_infl, 
            eival_clamp=self.eig_val_clamp,
            batch_size=self.eig_batch_size
        )
        result = xt.Dataset(coords={"cell":ds["cell"], "ens":ds["ens"], 
                                    "ens_row":ds["ens"], "ens_col":ds["ens"]})  
        result["wa"]= (("cell", "ens"), w_a)
        result["Wa"] = (("cell", "ens_row", "ens_col"), W_a)
        return result

    def apply_update(self, state, da_params):
        """
        """
        xb = state["state"].data.contiguous()
        state_shape = xb.shape
        xb = xb.view(xb.shape[0], xb.shape[1], -1)   # Flatten spatial dimensions
        xb = xb.permute(2, 1, 0).contiguous() # variable, cell, ens
        
        Wa = da_params["Wa"].values
        wa = da_params["wa"].values
        
        xa = letkf_update(xb, wa, Wa, self.alpha_rtpp, self.alpha_rtps, self.beta)
        xa = xa.permute(2, 1, 0).contiguous()
        xa = xa.view(state_shape)        
        return xt.DataTensor(xa, coords=state["state"].coords, 
                             dims=state["state"].dims)

    def __call__(self, state, da_state):
        """
        """
        da_params = self.infer_update_params(da_state)
        return core.apply_update(state, da_params)