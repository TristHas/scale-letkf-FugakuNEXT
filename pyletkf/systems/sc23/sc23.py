import math

import torch
import torch.nn.functional as F
import xtensor as xt

from pyletkf import Grid, Tile, Pipeline, ScaleLetkfConverter, GridInterpolator, LETKF, ObsOperator
from pyletkf.obs_op.radar import ref_operator, vr_operator, filter_ref, filter_vr

from sc23io import load_scale_state, load_radar

SCALE_STATE_ORDER = ["DENS", "RHOT", "MOMX", "MOMY", "MOMZ", "QV", "QC", "QR", "QI", "QS", "QG"]

RADIUS = 6_371_220.0
BASE_LON_DEG = 139.609
BASE_LAT_DEG = 35.861
BASE_LON = math.radians(BASE_LON_DEG)
BASE_LAT = math.radians(BASE_LAT_DEG)
FACT = math.cos(BASE_LAT)

base_x = 64000.
param_y =  -3_402_148.0178926853
cxg0 = -150.
cyg0 = -150.
dx = 100.
dy = 100.

def init_sc23():
    """
    """
    grid = Grid(proj_fn=sc23_proj, tile=SC23Tile)
    state_conv = ScaleLetkfConverter()
    interp = GridInterpolator(grid)
    obs_op = ObsOperator({4001: ref_operator, 4002: vr_operator}, 
                         {4001: filter_ref, 4002: filter_vr})
    core = LETKF()
    return grid, Pipeline(obs_op, interp, core, state_conv, None)

def sc23_proj(lon, lat):
    """
    """
    lon_rad = torch.deg2rad(lon)
    lat_rad = torch.deg2rad(lat)
    x = base_x + RADIUS * FACT * (lon_rad - BASE_LON)
    latrot = .5 * math.pi - lat_rad
    dist = torch.reciprocal(torch.tan(.5 * latrot))
    y = param_y + RADIUS * FACT * torch.log(dist)
    ri = (x - cxg0) / dx 
    rj = (y - cyg0) / dy 
    return ri-2, rj-2
    
class SC23Tile(Tile):
    """
    """
    def read_state(self):
        """
        """
        pad_x, pad_y = self.pad_x, self.pad_y
        states, height, topo, x, y, z, ens = load_scale_state(self.rank, 
                                                              pad_x=pad_x, 
                                                              pad_y=pad_y)
        
        
        topo = F.pad(topo, pad_x + pad_y, value=float("nan"))
        height = F.pad(
            height,
            (0, 0) + pad_x + pad_y,
            value=float("nan"),
        )
        states = F.pad(
            states,
            (0, 0) + pad_x + pad_y,
            value=float("nan"),
        )

        x = F.pad(x, pad_x)
        x[0] = x[1]  - self.grid.dx
        x[-1]= x[-2] + self.grid.dx

        y = F.pad(y, pad_y)
        y[0] = y[1]  - self.grid.dy
        y[-1]= y[-2] + self.grid.dy
        
        ds = xt.Dataset(coords={"x":x,"y":y,"z":z,
                                "ens":ens,
                                "variable":SCALE_STATE_ORDER})
        ds["state"] =(("ens", "variable", "y", "x", "z"), states)
        ds["height"]=(("y", "x", "z"), height)
        ds["topo"]  =(("y", "x"), topo)
        return ds

    def read_state_halo(self, x=None, y=None):
        """
        """
        pad_x, pad_y = self.pad_x, self.pad_y
        states, height, topo, x, y, z, ens = load_scale_state(self.rank, 
                                                              pad_x=pad_x, 
                                                              pad_y=pad_y,
                                                              x=x, y=y)

        ds = xt.Dataset(coords={"x":x,"y":y,"z":z,"ens":ens,"variable":SCALE_STATE_ORDER})
        ds["state"] =(("ens","variable","y","x","z"), states)
        ds["height"]=(("y","x","z"), height)
        ds["topo"]  =(("y","x"), topo)
        return ds
        
    def share_halos(self, ds):
        """
        """
        for rank, dx, dy in self.neighbor_ranks():
            ctx  = type(self)(rank, self.grid)
            halo_ds = ctx.read_state_halo(x=self.lookup_neighbor[dx], 
                                          y=self.lookup_neighbor[dy])
        
            x = self.lookup_self_x[dx]
            y = self.lookup_self_y[dy]
        
            ds["state"].values[:,:,y,x] = halo_ds["state"].values
            ds["height"].values[y,x] = halo_ds["height"].values
        return ds
        
    def read_obs(self, filter_lev=True):
        """
        """
        return load_radar(filter_lev=filter_lev)

    def share_obs(self, obs, neighbour_obs, interp):
        """
        """
        combined = xt.concat([obs] + [x.to(obs["dat"].device) for x in neighbour_obs], dim="obs")
        filtered = self.grid.filter_obs_to_tile(
            combined,
            self.rank,
            halo_x=interp.obs_halo_x + 1.5,
            halo_y=interp.obs_halo_y + .5,
        )
        return filtered
