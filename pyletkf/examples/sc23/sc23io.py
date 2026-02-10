from __future__ import annotations

from pathlib import Path
from array import array
from functools import lru_cache

import numpy as np
import xarray as xr

import torch
import torch.nn.functional as F
import xtensor as xt

from pyletkf.obs_op.radar import convert_raw_to_dbz

DATA_ROOT = Path("/home/tristan/workspace/scale_letkf/scale-letkf-FugakuNEXT/result/SC23")
SCALE_STATE_ORDER = ["DENS", "RHOT", "MOMX", "MOMY", "MOMZ", "QV", "QC", "QR", "QI", "QS", "QG"]
ADDITIONAL_VAR = ["height", "topo", "CZ"]

DEFAULT_RADAR_PATH = DATA_ROOT / "obs_radar/radar_20210730060030.dat"
RADAR_ZMIN = 500.0
RADAR_ZMAX = 11000.0

Z = torch.tensor([    55.     ,   165.     ,   275.     ,   385.     ,   495.     ,
                     608.08   ,   727.49245,   853.592  ,   986.75315,  1127.37135,
                    1275.86415,  1432.67255,  1598.2623 ,  1773.1251 ,  1957.78025,
                    2152.77615,  2358.6919 ,  2576.13905,  2805.7633 ,  3048.24655,
                    3304.30895,  3574.7108 ,  3860.2551 ,  4161.7898 ,  4480.2102 ,
                    4816.46215,  5171.5442 ,  5546.51075,  5942.47535,  6360.61375,
                    6802.16795,  7268.4492 ,  7760.842  ,  8280.80855,  8829.89305,
                    9409.7261 , 10022.0298 , 10668.62255, 11351.4243 , 12072.4629 ,
                   12842.8018 , 13642.8018 , 14442.8018 , 15242.8018 , 16042.8013 ],
                   dtype=torch.float64)

@lru_cache(maxsize=50)
def _load_tile_dataset(path_str):
    """
    Load and cache one full tile (all requested variables). SCALE tiles use
    a single chunk per state variable, so reading even small halo slices
    decompresses the entire array. By caching the fully decoded Dataset we
    can service halo requests via cheap indexing.
    """
    ds = xr.open_dataset(path_str)[SCALE_STATE_ORDER + ADDITIONAL_VAR]
    return ds.load()

def clear_state_cache():
    """
    Release all cached tile datasets (e.g., between analysis cycles).
    """
    _load_tile_dataset.cache_clear()

def load_scale_state(tile_idx=0, 
                     seconds="30", 
                     prefix="anal_f",
                     members=["0001", "0002"],
                     x=None,
                     y=None):#, "mean"]):
    """
    """
    tile_name = "pe" + str(tile_idx).zfill(6)

    states = []
    for member in members:
        fp = DATA_ROOT / f"202107300600{seconds}" / prefix / member \
                       / f"init_20210730-0600{seconds}.000.{tile_name}.nc"
        ds = load_state(fp, x=x, y=y)
        state = np.stack([ds[var].values[:,:,-45:] for var in SCALE_STATE_ORDER])
        states.append(state)
    states = np.stack(states)
    topo = ds["topo"].values
    height = ds["height"].values
    x = ds["x"].values
    y = ds["y"].values
    states, height, topo, x, y = map(torch.tensor, [states, height, topo, x, y])
    return states, height, topo, x, y, Z, members

def load_state(fp, x=None, y=None, cache=True):
    """
        Load an analysis tile slice. By default, cache the fully expanded Dataset
        for each tile path and serve halo slices from that cached object.
    """
    path = str(Path(fp))
    if cache: 
        ds = _load_tile_dataset(path)
    else: 
        ds = xr.open_dataset(path)[SCALE_STATE_ORDER + ADDITIONAL_VAR].load()

    indexers = {}
    if x is not None:
        indexers["x"] = [x]
        if "xh" in ds.dims:
            indexers["xh"] = [x]
    if y is not None:
        indexers["y"] = [y]
        if "yh" in ds.dims:
            indexers["yh"] = [y]
    if indexers:
        ds = ds.isel(**indexers)
    else:
        # Return a view so callers cannot mutate the cached dataset in-place.
        ds = ds.copy(deep=False)
    return ds

def load_radar(path=DEFAULT_RADAR_PATH,
               filter_lev=True):
    """
    """
    nvar = 9

    arr = array("f")
    arr.frombytes(path.read_bytes())
    arr.byteswap()

    data = torch.frombuffer(arr, dtype=torch.float32)

    radar_lon, radar_lat, radar_z = data[[1,4,7]]
    data = data[nvar:]
    nobs = len(data) // nvar
    _, elm, lon, lat, lev, dat, err, _, _ = data.view(nobs, nvar).t()
    dbz = convert_raw_to_dbz(dat)
    
    obs_coord = xt.arange_index(nobs, dtype=int)
    
    attrs = {"radar_lon": radar_lon, "radar_lat": radar_lat, "radar_z": radar_z}
    ds = xt.Dataset(coords={"obs": obs_coord}, attrs=attrs)
    for k,v in zip(["elm", "lon", "lat", "lev", "raw", "err", "dat"],
                   [elm, lon.double(), lat.double(), lev, dat, err, dbz]):
        ds[k]=(("obs",), v)
        
    if filter_lev: ds = filter_obs_height(ds)

    return ds

def filter_obs_height(obs: Dataset) -> torch.Tensor:
    lev = obs["lev"].data
    height_ok = (lev >= RADAR_ZMIN) & (lev <= RADAR_ZMAX)
    return obs.isel(obs=height_ok)
