import torch
import xtensor as xt

class ObsOperator():
    def __init__(self,
                 operator_map = None,
                 filter_map = None):
        """
            Observation Operator base class.
            Takes as input a dictionnary of {observation_type: function}
            Each observation type's function expects a xt.Dataset as input. 
        """
        self.operator_map = operator_map 
        self.filter_map = filter_map

    def __call__(self, obs: xt.Dataset) -> xt.Dataset:
        """
            
        """
        assert ("state" in obs) and ("elm" in obs)
        assert obs["state"].dims == ("obs", "ens", "variable")
        
        device = obs["state"].device 
        dtype  = obs["state"].dtype

        obs_type = obs["elm"].data.to(dtype=torch.int64, 
                                      device=device)
        
        hx_out = torch.zeros((obs.sizes["obs"], 
                              obs.sizes["ens"]),
                              device=device,
                              dtype=dtype)

        for obs_idx, obs_op in self.operator_map.items():
            mask = obs_type == obs_idx
            if mask.any(): hx_out[mask] = obs_op(obs.isel(obs=mask))

        mask = obs["mask"].values.to(hx_out.dtype) \
               if "mask" in obs else None
        if mask is not None: hx_out *= mask.unsqueeze(-1)
        
        hx_mean = hx_out.mean(-1)
        dhx = hx_out - hx_mean.unsqueeze(-1)
        innov = obs["dat"].values - hx_mean
        if mask is not None: innov *= mask

        obs["hx"]    = (("obs", "ens"), hx_out)
        obs["hx_d"]  = (("obs", "ens"), dhx)
        obs["hx_m"]  = (("obs",), hx_mean)
        obs["innov"] = (("obs",), innov)

        return obs
    
    def filter(self, obs: xt.Dataset) -> xt.Dataset:
        """
            
        """
        if not self.filter_map or "hx_m" not in obs:
            return obs
        
        obs_type = obs["elm"].data.to(torch.int64)
        keep_mask = torch.ones(obs.sizes["obs"], 
                               device=obs_type.device,
                               dtype=torch.bool)
        for obs_idx, obs_filter in self.filter_map.items():
            mask = obs_type == obs_idx
            if mask.any():
                keep_mask[mask] = obs_filter(obs.isel(obs=mask))
        return obs.isel(obs=keep_mask)