from __future__ import annotations

import struct
import sys
from array import array
from pathlib import Path
from typing import Dict

import torch
from xtensor import DataTensor, Dataset

from ..obs_op import convert_raw_to_dbz

DEFAULT_RADAR_PATH = Path("/home/tristan/workspace/scale_letkf/scale-letkf-FugakuNEXT/result/SC23/obs_radar/radar_20210730060030.dat")
PHARAD_TYP = 22

def _decode_order(data: bytes) -> str:
    header = data[: 9 * 4]
    header_be = struct.unpack(">9f", header)
    header_le = struct.unpack("<9f", header)
    if all(abs(val) <= 360 for val in header_be):
        return ">"
    if all(abs(val) <= 360 for val in header_le):
        return "<"
    return ">"

def _tensor_from_bytes(data: bytes, order: str, code: str, dtype: torch.dtype) -> torch.Tensor:
    arr = array(code)
    arr.frombytes(data)
    native = ">" if sys.byteorder == "big" else "<"
    if order != native:
        arr.byteswap()
    return torch.tensor(arr, dtype=dtype)

def _read_radar_dat(path: Path = DEFAULT_RADAR_PATH, byteorder: str | None = None) -> Dataset:
    path = Path(path)
    raw = path.read_bytes()
    order = byteorder or _decode_order(raw)
    floats = _tensor_from_bytes(raw, order, "f", torch.float32)
    if floats.numel() < 9:
        raise ValueError(f"{path} too small to contain radar header")

    radar_lon = float(floats[1])
    radar_lat = float(floats[4])
    radar_z = float(floats[7])

    stride = 9
    payload = floats[9:]
    if payload.numel() % stride != 0:
        raise ValueError(f"{path}: payload not divisible by {stride}")
    nobs = payload.numel() // stride
    obs_coord = torch.arange(1, nobs + 1, dtype=torch.int64)
    if nobs == 0:
        elm = torch.empty(0, dtype=torch.int32)
        lon = torch.empty(0, dtype=torch.float64)
        lat = torch.empty(0, dtype=torch.float64)
        lev = torch.empty(0, dtype=torch.float64)
        dat = torch.empty(0, dtype=torch.float64)
        err = torch.empty(0, dtype=torch.float64)
        typ = torch.empty(0, dtype=torch.int32)
        dif = torch.empty(0, dtype=torch.float64)
    else:
        payload = payload.view(nobs, stride)
        elm = payload[:, 1].to(dtype=torch.int32)
        lon = payload[:, 2].to(dtype=torch.float64)
        lat = payload[:, 3].to(dtype=torch.float64)
        lev = payload[:, 4].to(dtype=torch.float64)
        dat = payload[:, 5].to(dtype=torch.float64)
        err = payload[:, 6].to(dtype=torch.float64)
        typ = torch.full((nobs,), PHARAD_TYP, dtype=torch.int32)
        dif = torch.zeros(nobs, dtype=torch.float64)

    coords = {"obs": obs_coord}
    data = {
        "elm": DataTensor(elm, coords, ("obs",)),
        "lon": DataTensor(lon, coords, ("obs",)),
        "lat": DataTensor(lat, coords, ("obs",)),
        "lev": DataTensor(lev, coords, ("obs",)),
        "dat": DataTensor(dat, coords, ("obs",)),
        "err": DataTensor(err, coords, ("obs",)),
        "typ": DataTensor(typ, coords, ("obs",)),
        "dif": DataTensor(dif, coords, ("obs",)),
    }
    attrs = {"radar_lon": radar_lon, "radar_lat": radar_lat, "radar_z": radar_z}
    return Dataset(data, coords={"obs": obs_coord}, attrs=attrs)

def convert_radar_dbz(radar_obs: Dataset) -> Dataset:
    raw_tensor = radar_obs["dat"]
    converted = convert_raw_to_dbz(raw_tensor.data)
    dat_tensor = DataTensor(converted, raw_tensor.coords, raw_tensor.dims)
    return radar_obs.assign(raw=raw_tensor, dat=dat_tensor)
    
def load_radar(path: Path = DEFAULT_RADAR_PATH) -> Dataset:
    ds = _read_radar_dat(path)
    return convert_radar_dbz(ds)
