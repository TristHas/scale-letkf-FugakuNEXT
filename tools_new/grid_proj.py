import math
import numpy as np
import xarray as xr

RADIUS = 6_371_220.0
BASE_LON_DEG = 139.609
BASE_LAT_DEG = 35.861
BASE_LON = math.radians(BASE_LON_DEG)
BASE_LAT = math.radians(BASE_LAT_DEG)
FACT = math.cos(BASE_LAT)

PRC_NUM_X = 4
PRC_NUM_Y = 5
NX_TILE = 320
NY_TILE = 256
DX = 100.0
DY = 100.0

IHALO = 2
JHALO = 2

def grid_ij_from_petag(pe_tag):
    pe_index = int(pe_tag.replace("pe", ""))
    target_i = pe_index % PRC_NUM_X
    target_j = pe_index // PRC_NUM_X
    return (target_i, target_j)

def _lonlat_to_grid_indices(
        lon_deg: np.ndarray,
        lat_deg: np.ndarray,
        *,
        base_x: float,
        param_y: float,
        cxg0: float,
        cyg0: float,
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    """
    lon_rad = np.deg2rad(lon_deg)
    lat_rad = np.deg2rad(lat_deg)
    x = base_x + RADIUS * FACT * (lon_rad - BASE_LON)
    latrot = 0.5 * np.pi - lat_rad
    dist = 1.0 / np.tan(0.5 * latrot)
    y = param_y + RADIUS * FACT * np.log(dist)
    ri = (x - cxg0) / DX + 1.0
    rj = (y - cyg0) / DY + 1.0
    return ri, rj

def global_to_local_indices(ri, rj, grid_ij):
    """
        Convert global ri/rj to tile-local sample indices.
    """
    target_i, target_j = grid_ij
    ri_tile = ri - target_i * NX_TILE
    rj_tile = rj - target_j * NY_TILE

    ri_sample = np.clip(ri_tile - IHALO, 1.0, NX_TILE - 1.0e-6)
    rj_sample = np.clip(rj_tile - JHALO, 1.0, NY_TILE - 1.0e-6)

    return ri_sample, rj_sample

def local_to_global_indices(ri_sample, rj_sample, grid_ij):
    """
    """
    target_i, target_j = grid_ij
    
    ri_tile = ri_sample + IHALO
    rj_tile = rj_sample + JHALO

    ri = ri_tile + target_i * NX_TILE
    rj = rj_tile + target_j * NY_TILE

    return ri, rj

def compute_grid_indices_from_state(state_ds: xr.Dataset, pe_tag: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Infer global grid indices from the state coordinates."""
    nx = state_ds.sizes["x"]
    ny = state_ds.sizes["y"]
    nz = state_ds.sizes["z"]
    grid_ij = grid_ij_from_petag(pe_tag)
    x_idx = np.arange(1, nx + 1, dtype=np.float64)
    y_idx = np.arange(1, ny + 1, dtype=np.float64)
    xi, yi = np.meshgrid(x_idx, y_idx, indexing="xy")
    ri_horiz, rj_horiz = local_to_global_indices(xi.ravel(), yi.ravel(), grid_ij)
    grid_ri = np.tile(ri_horiz, nz)
    grid_rj = np.tile(rj_horiz, nz)
    heights = (
        state_ds["height"]
        .transpose("z", "y", "x")
        .values
        .reshape(-1)
    )
    return grid_ri, grid_rj, heights
    
def populate_obs_global_indices(obs, state_ds):
    """
    Convert obs lat/lon into global grid indices (ri, rj).
    """
    lon_obs = obs["lon"].values
    lat_obs = obs["lat"].values

    ri, rj = _lonlat_to_grid_indices(
        lon_obs, lat_obs,
        base_x=state_ds["base_x"].item(),
        param_y=state_ds["param_y"].item(),
        cxg0=state_ds["cxg0"].item(),
        cyg0=state_ds["cyg0"].item(),
    )
    obs["ri_global"]=("obs", ri)
    obs["rj_global"]=("obs", rj)
    return obs

def populate_obs_local_indices(obs, grid_ij):
    ri, rj = global_to_local_indices(obs["ri_global"], obs["rj_global"], grid_ij)
    obs["ri_local"]=ri
    obs["rj_local"]=rj
    return obs

def filter_obs_to_tile(obs, grid_ij):
    """
    Keep only obs that belong to the tile of the specified PE.
    Returns obs filtered + filtered ri/rj + tile coords.
    """
    rank_i = np.floor((obs["ri_global"] - 1.0) / NX_TILE).astype(int)
    rank_j = np.floor((obs["rj_global"] - 1.0) / NY_TILE).astype(int)

    target_i, target_j = grid_ij
    tile_mask = (rank_i == target_i) & (rank_j == target_j)

    if not np.any(tile_mask):
        raise RuntimeError("No observations reside in this tile")

    obs_filtered = obs.isel(obs=tile_mask)

    return obs_filtered

def compute_obs_grid_idx(obs, state_ds, pe_tag):
    """
    """
    # 1) lat/lon → global grid indices
    obs = populate_obs_global_indices(obs, state_ds)
    # 2) filter obs to tile
    grid_ij = grid_ij_from_petag(pe_tag)
    obs_filtered = filter_obs_to_tile(obs, grid_ij)
    # 3) convert global → local tile indices
    obs_filtered = populate_obs_local_indices(obs_filtered, grid_ij)
    return obs_filtered