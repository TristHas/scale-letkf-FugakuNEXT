from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import torch
from netCDF4 import Dataset as NetCDFDataset

from xtensor import DataTensor, Dataset

from ..params import KHALO

RDRY = 287.04
CPDRY = 1004.64
CVDry = CPDRY - RDRY

RVAP = 461.50
CPVAP = 1846.00
CVVAP = CPVAP - RVAP

PRE00 = 100000.0
FILL = -9.9999e30

TRACER_SPECIES = ("QV", "QC", "QR", "QI", "QS", "QG")
TRACER_CV = torch.tensor([CVVAP, CVVAP, CVVAP, CVVAP, CVVAP, CVVAP], dtype=torch.float64)

CONTROL_ORDER = ("U", "V", "W", "T", "P", "QV", "QC", "QR", "QI", "QS", "QG")

RADIUS = 6_371_220.0
BASE_LON_DEG = 139.609
BASE_LAT_DEG = 35.861
BASE_LON = math.radians(BASE_LON_DEG)
BASE_LAT = math.radians(BASE_LAT_DEG)
FACT = math.cos(BASE_LAT)

MEMBERS = ("0001", "0002", "mean")
COORD_VARS = ("x", "y", "z", "xh", "yh", "zh")

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


@dataclass
class RawState:
    members: Tuple[str, ...]
    data: Dict[str, torch.Tensor]
    coords: Dict[str, Tuple[float, ...]]
    halo_map: Dict[str, Tuple[int, int]]
    fxg: torch.Tensor
    fyg: torch.Tensor
    cxg0: float
    cyg0: float
    cz: torch.Tensor
    fz: torch.Tensor


def _read_member_file(path: Path) -> tuple[Dict[str, torch.Tensor], Dict[str, Tuple[float, ...]], Dict[str, Tuple[int, int]], torch.Tensor, torch.Tensor, float, float, torch.Tensor, torch.Tensor]:
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
            data = torch.as_tensor(var[:], dtype=torch.float64)
            curr_dims = tuple(var.dimensions)
            if curr_dims != target_dims:
                perm = [curr_dims.index(dim) for dim in target_dims]
                data = data.permute(*perm)
            arrays[name] = data
        for coord in COORD_VARS:
            if coord not in ds.variables:
                continue
            var = ds.variables[coord]
            tensor = torch.as_tensor(var[:], dtype=torch.float64)
            coords[coord] = tuple(float(v) for v in tensor.tolist())
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


def read_and_concat_members(dump_dir: str | Path, pe_tag: str, prefix: str = "anal_f") -> RawState:
    dump_dir = Path(dump_dir)
    stacked: Dict[str, list[torch.Tensor]] = {name: [] for name in STATE_FIELD_SPECS}
    base_coords: Dict[str, Tuple[float, ...]] | None = None
    halo_map: Dict[str, Tuple[int, int]] | None = None
    fxg = fyg = None
    cxg0 = cyg0 = 0.0
    cz = fz = None
    for mem in MEMBERS:
        fname = f"init_20210730-060030.000.{pe_tag}.nc"
        path = dump_dir / ".." / prefix / mem / fname
        arrays, coords, halos, fxg_vals, fyg_vals, cx_val, cy_val, cz_vals, fz_vals = _read_member_file(path)
        for name, tensor in arrays.items():
            stacked[name].append(tensor)
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
    data = {name: torch.stack(parts, dim=0) for name, parts in stacked.items()}
    assert base_coords is not None and halo_map is not None and fxg is not None and fyg is not None
    assert cz is not None and fz is not None
    return RawState(tuple(MEMBERS), data, base_coords, halo_map, fxg, fyg, cxg0, cyg0, cz, fz)


def _clean_tensor(tensor: torch.Tensor) -> torch.Tensor:
    mask = torch.abs(tensor - FILL) > 1.0e20
    return torch.where(mask, tensor, torch.full_like(tensor, float("nan")))

def _compute_height_from_topo(
    topo: torch.Tensor,
    cz: torch.Tensor,
    fz: torch.Tensor,
    nlev: int,
) -> torch.Tensor:
    ks = 1 + KHALO
    ke = ks + nlev - 1
    ztop = fz[ke - 1] - fz[ks - 2]
    cz_slice = cz[ks - 1 : ks - 1 + nlev]
    scale = (ztop - topo) / ztop
    heights: list[torch.Tensor] = []
    for level_idx, cz_val in enumerate(cz_slice):
        heights.append(scale * cz_val + topo)
    return torch.stack(heights, dim=0)


def convert_scale_to_letkf(raw: RawState) -> DataTensor:
    rho = _clean_tensor(raw.data["DENS"])
    rhot = _clean_tensor(raw.data["RHOT"])
    momx = _clean_tensor(raw.data["MOMX"])
    momy = _clean_tensor(raw.data["MOMY"])
    momz = _clean_tensor(raw.data["MOMZ"])[:, 1:, :, :]

    u = momx / rho
    v = momy / rho
    w = momz / rho

    moist = torch.stack([_clean_tensor(raw.data[name]) for name in TRACER_SPECIES], dim=0)
    moist_cl = torch.nan_to_num(moist, nan=0.0)
    qdry = 1.0 - moist_cl.sum(dim=0)

    tracer_cv = TRACER_CV.view(-1, 1, 1, 1, 1)
    cv_tot = CVDry * qdry + (moist_cl * tracer_cv).sum(dim=0)
    species_index = {name: idx for idx, name in enumerate(TRACER_SPECIES)}
    rtot = RDRY * qdry + RVAP * moist_cl[species_index["QV"]]

    base = (rhot * rtot) / PRE00
    valid = (base > 0.0) & (rho > 0.0) & (cv_tot > 0.0) & (rtot > 0.0)

    gamma = torch.where(valid, (cv_tot + rtot) / cv_tot, torch.full_like(cv_tot, float("nan")))
    pressure = torch.where(valid, PRE00 * torch.pow(base, gamma), torch.full_like(base, float("nan")))
    temperature = torch.where(valid, pressure / (rho * rtot), torch.full_like(base, float("nan")))

    qv = moist[species_index["QV"]]
    qc = moist[species_index["QC"]]
    qr = moist[species_index["QR"]]
    qi = moist[species_index["QI"]]
    qs = moist[species_index["QS"]]
    qg = moist[species_index["QG"]]

    control_stack = torch.stack(
        [u, v, w, temperature, pressure, qv, qc, qr, qi, qs, qg],
        dim=0,
    )
    coords = {
        "variable": CONTROL_ORDER,
        "ens": raw.members,
        "z": raw.coords["z"],
        "y": raw.coords["y"],
        "x": raw.coords["x"],
    }
    return DataTensor(control_stack, coords, ("variable", "ens", "z", "y", "x"))


def strip_all_halos(ds: Dataset, halo_map: Dict[str, Tuple[int, int]] | None = None) -> Dataset:
    halo = halo_map or ds.attrs.get("halo_map", {})
    updated = {}
    for name, var in ds.data_vars.items():
        selectors = {}
        for dim, (L, R) in halo.items():
            if dim in var.dims and (L or R):
                selectors[dim] = slice(L, None if R == 0 else -R)
        updated[name] = var.isel(**selectors) if selectors else var
    return Dataset(updated, attrs=ds.attrs)


def compute_grid_params(raw: RawState) -> tuple[float, float, float, float]:
    fxg = raw.fxg
    fyg = raw.fyg
    cxg0 = raw.cxg0
    cyg0 = raw.cyg0
    base_x = float(0.5 * (fxg[0] + fxg[-1]))
    base_y = float(0.5 * (fyg[0] + fyg[-1]))
    latrot0 = 0.5 * math.pi - BASE_LAT
    dist0 = 1.0 / math.tan(0.5 * latrot0)
    param_y = base_y - RADIUS * FACT * math.log(dist0)
    return base_x, param_y, cxg0, cyg0


def _scalar_tensor(value: float) -> DataTensor:
    return DataTensor(torch.as_tensor(value, dtype=torch.float64), {}, ())


def load_letkf_state(dump_dir: str | Path, pe_tag: str, prefix: str, strip_hallow: bool = True) -> Dataset:
    raw = read_and_concat_members(dump_dir, pe_tag, prefix)
    letkf_state = convert_scale_to_letkf(raw)
    base_x, param_y, cxg0, cyg0 = compute_grid_params(raw)
    y_coords = raw.coords["y"]
    x_coords = raw.coords["x"]
    z_coords = raw.coords["z"]

    lon = DataTensor(raw.data["lon"][0], {"y": y_coords, "x": x_coords}, ("y", "x"))
    lat = DataTensor(raw.data["lat"][0], {"y": y_coords, "x": x_coords}, ("y", "x"))
    topo_field = raw.data["topo"][0]
    if "height" in raw.data and raw.data["height"].numel() > 0:
        height_tensor = raw.data["height"][0]
    else:
        height_tensor = _compute_height_from_topo(topo_field, raw.cz, raw.fz, len(z_coords))
    height = DataTensor(height_tensor, {"z": z_coords, "y": y_coords, "x": x_coords}, ("z", "y", "x"))
    topography = DataTensor(topo_field, {"y": y_coords, "x": x_coords}, ("y", "x"))
    ks = 1 + KHALO
    ke = ks + len(z_coords) - 1
    ztop = raw.fz[ke - 1] - raw.fz[ks - 2]
    cz_slice = raw.cz[ks - 1 : ks - 1 + len(z_coords)]
    cz_levels = DataTensor(cz_slice, {"z": z_coords}, ("z",))

    data_vars = {
        "state": letkf_state,
        "lon": lon,
        "lat": lat,
        "height": height,
        "topography": topography,
        "cz_levels": cz_levels,
        "base_x": _scalar_tensor(base_x),
        "param_y": _scalar_tensor(param_y),
        "cxg0": _scalar_tensor(cxg0),
        "cyg0": _scalar_tensor(cyg0),
    }
    coords = {
        "variable": CONTROL_ORDER,
        "ens": raw.members,
        "z": z_coords,
        "y": y_coords,
        "x": x_coords,
    }
    dataset = Dataset(data_vars, coords=coords, attrs={"halo_map": raw.halo_map, "ztop": float(ztop)})
    if strip_hallow:
        dataset = strip_all_halos(dataset, raw.halo_map)
    return dataset

def convert_letkf_to_scale(*_args, **_kwargs):
    raise NotImplementedError("convert_letkf_to_scale is not available without xarray.")

def convert_letkf_scale_var(*_args, **_kwargs):
    raise NotImplementedError("convert_letkf_scale_var is not available without xarray.")
