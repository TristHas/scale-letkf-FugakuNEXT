import math
import torch
import xtensor as xt

from .ops import TOPK_FUNC, sample_state

def extract_coordinate_tensors(state_ds: xt.Dataset, 
                               obs_ds: xt.Dataset, 
                               grid, 
                               dtype=torch.float32):
    """
    """
    state = state_ds["state"]
    device = state.device
    
    y = state["y"].data.to(device)
    x = state["x"].data.to(device)
    z = state["z"].data.to(device)
    
    grid_y, grid_x  = torch.meshgrid(y, x, indexing="ij")
    grid_x = grid_x.reshape(-1).repeat_interleave(len(z))
    grid_y = grid_y.reshape(-1).repeat_interleave(len(z))
    #grid_xy = torch.stack((grid_ri, grid_rj), dim=1).to(device=device)
    grid_xy = torch.stack((grid_x, grid_y), dim=1).to(device=device, dtype=dtype)
    #grid_z  = state_ds["height"].data.permute(1, 2, 0).reshape(-1).to(device=device)
    grid_z  = state_ds["height"].transpose("y", "x", "z")\
                                .values.contiguous()\
                                .reshape(-1).to(device=device)
    
    obs_xy = torch.stack(
        (obs_ds["ri_global"].data.to(device=device) * grid.dx,
         obs_ds["rj_global"].data.to(device=device) * grid.dy),
         dim=1,
    )
    obs_z = obs_ds["lev"].data.to(device=device)
    
    return {
        "grid_xy": grid_xy.to(dtype=dtype),
        "grid_z": grid_z.to(dtype=dtype),
        "obs_xy": obs_xy.to(dtype=dtype),
        "obs_z": obs_z.to(dtype=dtype),
    }

def assemble_cell_outputs(
        hdxf, dep, err, obs_idx,
        topk_vals, topk_idx, valid_mask,
        var_local_factor=1.,
    ):
    """
    """
    hdxf_sel = hdxf[topk_idx]
    dep_sel  = dep[topk_idx]
    err_sel  = err[topk_idx]

    rloc_sel = var_local_factor * torch.exp(-0.5 * topk_vals)
    rdiag_sel = err_sel ** 2 / rloc_sel

    mask_expanded = valid_mask.unsqueeze(-1)
    hdxf_sel = torch.where(mask_expanded, hdxf_sel, 0)
    dep_sel = torch.where(valid_mask, dep_sel, 0)
    rloc_sel = torch.where(valid_mask, rloc_sel, 0)
    rdiag_sel = torch.where(valid_mask, rdiag_sel, 0)
    idx_sel = torch.where(valid_mask, obs_idx[topk_idx], -1)

    return hdxf_sel, dep_sel, rdiag_sel, rloc_sel, valid_mask
    
class GridInterpolator():
    def __init__(self,
                 grid,
                 max_n_obs=100,
                 method="quad",
                 horizontal_loc=2000,
                 vertical_loc=2000,
                 dist_zero_fac=3.651483717,
                 var_local_factor=1.,
                ):
        """
        """
        self.grid = grid
        self.max_n_obs = max_n_obs
        self.horizontal_loc = horizontal_loc
        self.vertical_loc = vertical_loc
        self.topk_fn = TOPK_FUNC[method]
        self.dist_zero_fac = dist_zero_fac
        self.var_local_factor = var_local_factor
        self.obs_halo_x = math.ceil(self.horizontal_loc * self.dist_zero_fac / self.grid.dx)
        self.obs_halo_y = math.ceil(self.horizontal_loc * self.dist_zero_fac / self.grid.dy)
        setattr(self.grid, "obs_halo_x", self.obs_halo_x)
        setattr(self.grid, "obs_halo_y", self.obs_halo_y)
        self.extract_coordinate_tensors = extract_coordinate_tensors

    def map_state_to_obs(self, obs, state):
        """
        """
        device = obs["lev"].device
        state_var = state["state"].transpose("y", "x", "z", "ens", "variable").values
        height = state["height"].values
        variable_names = list(state["variable"].values)
            
        samples, valid_mask = sample_state(
            state_var.to(device),
            height.to(device),
            obs["ri_local"].values.to(device),
            obs["rj_local"].values.to(device),
            obs["lev"].values.to(device),
            variable_names=variable_names,
        )
        
        obs = obs.assign_coords(ens=state["ens"])
        obs = obs.assign_coords(variable=state["variable"])
        obs["state"] = (("obs", "ens", "variable"), samples)
        obs["mask"]  = (("obs",), valid_mask)
        return obs

    def map_obs_to_state(self, obs, state):
        """
        """
        coords = self.extract_coordinate_tensors(state, obs, self.grid)
        topk_vals, topk_idx, valid_mask = self.topk_fn(
            max_obs_per_grid=self.max_n_obs,
            horiz_loc=self.horizontal_loc,
            vert_loc=self.vertical_loc,
            zero_fac=self.dist_zero_fac,
             **coords
        )

        hdxf, dep, rdiag, rloc, mask = assemble_cell_outputs(
            obs["hx_d"].data,
            obs["innov"].data,
            obs["err"].data,
            obs["obs"].data,
            topk_vals, topk_idx,
            valid_mask,
            var_local_factor=self.var_local_factor,
        )

        dataset = xt.Dataset(
            coords={
                "cell": xt.arange_index(hdxf.shape[0], device=hdxf.device),
                "obs" : xt.arange_index(hdxf.shape[1], device=hdxf.device),
                "ens" : xt.arange_index(hdxf.shape[2], device=hdxf.device),
            }
        )
        dataset["hx_d"] = (("cell", "obs", "ens"), hdxf)
        dataset["dep"] = (("cell", "obs"), dep)
        dataset["rdiag"] = (("cell", "obs"), rdiag)
        dataset["rloc"] = (("cell", "obs"), rloc)
        dataset["obs_mask"] = (("cell", "obs"), mask)

        return dataset

