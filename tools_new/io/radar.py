from pathlib import Path
import numpy as np
import xarray as xr

from ..obs_op import convert_raw_to_dbz

ID_RADAR_REF = 4001  # reflectivity observations
MIN_RADAR_REF_DBZ = 10.0  # from PARAM_LETKF_RADAR
LOW_REF_SHIFT = -5.0
MIN_RADAR_REF = 10.0 ** (MIN_RADAR_REF_DBZ / 10.0)  # linear threshold = 10
DEFAULT_RADAR_PATH = Path("/home/tristan/workspace/scale_letkf/scale-letkf-FugakuNEXT/result/SC23/obs_radar/radar_20210730060030.dat")

def convert_raw_to_dbz_old(raw_values):
    """
        Convert raw PAWR reflectivity (linear units) to dBZ.
    """
    raw = np.asarray(raw_values, dtype=np.float64)
    dbz = np.empty_like(raw, dtype=np.float64)
    valid = (raw >= 0.0) & (raw < 1.0e10)    
    # For out-of-range entries, mirror the Fortran behavior (set to NaN / undef)
    dbz[~valid] = np.nan
    low_mask = valid & (raw < MIN_RADAR_REF)
    high_mask = valid & ~low_mask
    # Below the threshold → fixed clear-sky value (MIN_RADAR_REF_DBZ + LOW_REF_SHIFT)
    dbz[low_mask] = MIN_RADAR_REF_DBZ + LOW_REF_SHIFT
    # Above the threshold → 10*log10(raw)
    dbz[high_mask] = 10.0 * np.log10(raw[high_mask])
    return dbz

def _read_radar_dat(path: Path = DEFAULT_RADAR_PATH, 
                    byteorder: str | None = None) -> xr.Dataset:
    """
    """
    path = Path(path)
    if byteorder is None:
        header_be = np.fromfile(path, dtype=">f4", count=9)
        header_le = np.fromfile(path, dtype="<f4", count=9)
        if header_be.size == 9 and np.all(np.abs(header_be) <= 360):
            order = ">"
        elif header_le.size == 9 and np.all(np.abs(header_le) <= 360):
            order = "<"
        else:
            order = ">"
    else:
        order = byteorder

    floats = np.fromfile(path, dtype=f"{order}f4")
    if floats.size < 9:
        raise ValueError(f"{path} too small to contain radar header")

    radar_lon = float(floats[1])
    radar_lat = float(floats[4])
    radar_z = float(floats[7])

    stride = 9
    payload = floats[9:]
    if payload.size % stride != 0:
        raise ValueError(f"{path}: payload not divisible by {stride}")
    nobs = payload.size // stride
    obs_coord = np.arange(1, nobs + 1, dtype=np.int64)
    if nobs == 0:
        data = {
            "elm": ("obs", np.empty(0, dtype=np.int32)),
            "lon": ("obs", np.empty(0)),
            "lat": ("obs", np.empty(0)),
            "lev": ("obs", np.empty(0)),
            "dat": ("obs", np.empty(0)),
            "err": ("obs", np.empty(0)),
            "typ": ("obs", np.empty(0, dtype=np.int32)),
            "dif": ("obs", np.empty(0)),
        }
    else:
        payload = payload.reshape(nobs, stride)
        elm = payload[:, 1].astype(np.int32)
        lon = payload[:, 2].astype(np.float64)
        lat = payload[:, 3].astype(np.float64)
        lev = payload[:, 4].astype(np.float64)
        dat = payload[:, 5].astype(np.float64)
        err = payload[:, 6].astype(np.float64)
        typ = np.full(nobs, 22, dtype=np.int32)
        dif = np.zeros(nobs, dtype=np.float64)
        data = {
            "elm": ("obs", elm),
            "lon": ("obs", lon),
            "lat": ("obs", lat),
            "lev": ("obs", lev),
            "dat": ("obs", dat),
            "err": ("obs", err),
            "typ": ("obs", typ),
            "dif": ("obs", dif),
        }

    attrs = {"radar_lon": radar_lon, "radar_lat": radar_lat, "radar_z": radar_z}
    return xr.Dataset(data, coords={"obs": obs_coord}, attrs=attrs)

def convert_radar_dbz(radar_obs):
    """
    """
    raw = radar_obs["dat"].to_pandas()
    elm = radar_obs["elm"].to_pandas()
    y_obs = raw.copy()
    ref_mask = elm == ID_RADAR_REF
    y_obs.loc[ref_mask] = convert_raw_to_dbz(raw.loc[ref_mask].values)
    radar_obs["raw"] = radar_obs["dat"]
    radar_obs["dat"]=y_obs
    
def load_radar(path=DEFAULT_RADAR_PATH):
    """
    """
    ds = _read_radar_dat(path)
    convert_radar_dbz(ds)
    return ds