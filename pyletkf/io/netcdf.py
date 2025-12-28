from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import torch
from netCDF4 import Dataset as NetCDFDataset

from ..params import MEMBERS

STATE_FIELD_SPECS: Mapping[str, Tuple[str, ...]] = {
    "DENS": ("z", "y", "x"),
    "RHOT": ("z", "y", "x"),
    "MOMX": ("z", "y", "xh"),
    "MOMY": ("z", "yh", "x"),
    "MOMZ": ("zh", "y", "x"),
    "QV": ("z", "y", "x"),
    "QC": ("z", "y", "x"),
    "QR": ("z", "y", "x"),
    "QI": ("z", "y", "x"),
    "QS": ("z", "y", "x"),
    "QG": ("z", "y", "x"),
    "height": ("z", "y", "x"),
    "topo": ("y", "x"),
    "lon": ("y", "x"),
    "lat": ("y", "x"),
}

COORD_VARS = ("x", "y", "z", "xh", "yh", "zh")
SHARED_STATE_VARS = {"lon", "lat", "topo", "height"}


def _parse_halo(value: Iterable[int] | int) -> Tuple[int, int]:
    if isinstance(value, Iterable) and not isinstance(value, (int, float)):
        vals = tuple(int(v) for v in value)
    else:
        vals = (int(value),)
    if len(vals) == 2:
        return vals
    if len(vals) == 1:
        return (vals[0], 0)
    return (0, 0)


def _normalize_slice(selector: slice | None, size: int) -> slice:
    if selector is None:
        return slice(None)
    start = selector.start
    stop = selector.stop
    step = selector.step
    if start is not None and start < 0:
        start += size
    if stop is not None and stop < 0:
        stop += size
    start = max(start, 0) if start is not None else None
    if stop is not None:
        stop = min(stop, size)
    return slice(start, stop, step)


def _build_indexer(curr_dims: Tuple[str, ...], shape: Tuple[int, ...], dim_slices: Mapping[str, slice] | None) -> Tuple[slice, ...]:
    dim_slices = dim_slices or {}
    indexer: list[slice] = []
    for dim, length in zip(curr_dims, shape):
        selector = dim_slices.get(dim)
        normalized = _normalize_slice(selector, length) if selector is not None else slice(None)
        indexer.append(normalized)
    return tuple(indexer)


def _read_member_file(path: Path, dim_slices: Mapping[str, slice] | None = None) -> tuple[Dict[str, torch.Tensor], Dict[str, Tuple[float, ...]], Dict[str, Tuple[int, int]], torch.Tensor, torch.Tensor, float, float, torch.Tensor, torch.Tensor]:
    arrays: Dict[str, torch.Tensor] = {}
    coords: Dict[str, Tuple[float, ...]] = {}
    halos: Dict[str, Tuple[int, int]] = {}
    fxg = torch.tensor([])
    fyg = torch.tensor([])
    cxg0 = 0.0
    cyg0 = 0.0
    cz_vals = torch.tensor([])
    fz_vals = torch.tensor([])

    with NetCDFDataset(path) as ds:
        for name, target_dims in STATE_FIELD_SPECS.items():
            var = ds.variables[name]
            curr_dims = tuple(var.dimensions)
            indexer = _build_indexer(curr_dims, var.shape, dim_slices)
            data = torch.as_tensor(var[indexer], dtype=torch.float64)
            if curr_dims != target_dims:
                perm = [curr_dims.index(dim) for dim in target_dims]
                data = data.permute(*perm)
            arrays[name] = data
        for coord in COORD_VARS:
            if coord not in ds.variables:
                continue
            var = ds.variables[coord]
            curr_dims = tuple(var.dimensions)
            if len(curr_dims) != 1:
                tensor = torch.as_tensor(var[:], dtype=torch.float64)
            else:
                indexer = _build_indexer(curr_dims, var.shape, dim_slices)
                tensor = torch.as_tensor(var[indexer], dtype=torch.float64)
            coords[coord] = tuple(float(v) for v in tensor.reshape(-1).tolist())
            halo_attr = var.getncattr("halo_local") if "halo_local" in var.ncattrs() else (0, 0)
            halos[coord] = _parse_halo(halo_attr)
        fxg = torch.as_tensor(ds.variables["FXG"][:], dtype=torch.float64)
        fyg = torch.as_tensor(ds.variables["FYG"][:], dtype=torch.float64)
        cxg0 = float(torch.as_tensor(ds.variables["CXG"][:], dtype=torch.float64)[0])
        cyg0 = float(torch.as_tensor(ds.variables["CYG"][:], dtype=torch.float64)[0])
        if "CZ" in ds.variables:
            cz_vals = torch.as_tensor(ds.variables["CZ"][:], dtype=torch.float64)
        if "FZ" in ds.variables:
            fz_vals = torch.as_tensor(ds.variables["FZ"][:], dtype=torch.float64)
    return arrays, coords, halos, fxg, fyg, cxg0, cyg0, cz_vals, fz_vals


def read_and_concat_members(dump_dir: str | Path, pe_tag: str, prefix: str = "anal_f", *, dim_slices: Mapping[str, slice] | None = None) -> "Dataset":
    from xtensor import Dataset  # local import to avoid circular

    dump_dir = Path(dump_dir)
    stacked: Dict[str, list[torch.Tensor]] = {}
    shared_vars: Dict[str, torch.Tensor] = {}
    base_coords: Dict[str, Tuple[float, ...]] | None = None
    halo_map: Dict[str, Tuple[int, int]] | None = None
    fxg = fyg = None
    cxg0 = cyg0 = 0.0
    cz = fz = None
    for mem in MEMBERS:
        fname = f"init_20210730-060030.000.{pe_tag}.nc"
        path = dump_dir / ".." / prefix / mem / fname
        arrays, coords, halos, fxg_vals, fyg_vals, cx_val, cy_val, cz_vals, fz_vals = _read_member_file(path, dim_slices)
        for name, tensor in arrays.items():
            if name in SHARED_STATE_VARS:
                shared_vars[name] = tensor
            else:
                stacked.setdefault(name, []).append(tensor)
        if base_coords is None:
            base_coords = coords
        if halo_map is None:
            halo_map = halos
        if fxg is None:
            fxg = fxg_vals
            fyg = fyg_vals
            cxg0 = cx_val
            cyg0 = cy_val
        if cz is None:
            cz = cz_vals
        if fz is None:
            fz = fz_vals
    data_vars = {}
    for name, tensors in stacked.items():
        dims = ("ens",) + STATE_FIELD_SPECS[name]
        data_vars[name] = (dims, torch.stack(tensors, dim=0))
    for name, tensor in shared_vars.items():
        data_vars[name] = (STATE_FIELD_SPECS[name], tensor)
    assert base_coords is not None and halo_map is not None and fxg is not None and fyg is not None
    assert cz is not None and fz is not None
    coords = dict(base_coords)
    coords["ens"] = tuple(MEMBERS)
    attrs = {
        "halo_map": halo_map,
        "fxg": fxg,
        "fyg": fyg,
        "cxg0": cxg0,
        "cyg0": cyg0,
        "cz": cz,
        "fz": fz,
    }
    return Dataset(data_vars, coords=coords, attrs=attrs)
