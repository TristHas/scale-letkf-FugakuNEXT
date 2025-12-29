from __future__ import annotations

import math
from itertools import chain
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import torch

from xtensor import DataTensor, Dataset

from ..params import IHALO, JHALO, KHALO, NX_TILE, NY_TILE, PRC_NUM_X, PRC_NUM_Y
from .netcdf import read_and_concat_members
from .utils import _normalize_pe_tag

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
SCALE_STATE_ORDER = ("DENS", "RHOT", "MOMX", "MOMY", "MOMZ", "QV", "QC", "QR", "QI", "QS", "QG")

RADIUS = 6_371_220.0
BASE_LON_DEG = 139.609
BASE_LAT_DEG = 35.861
BASE_LON = math.radians(BASE_LON_DEG)
BASE_LAT = math.radians(BASE_LAT_DEG)
FACT = math.cos(BASE_LAT)



def _clean_tensor(tensor: torch.Tensor) -> torch.Tensor:
    mask = torch.abs(tensor - FILL) > 1.0e20
    return torch.where(mask, tensor, torch.full_like(tensor, float("nan")))


def _build_dim_slices(x_slice: slice | None = None, y_slice: slice | None = None) -> Mapping[str, slice] | None:
    dim_slices: dict[str, slice] = {}
    if x_slice is not None:
        dim_slices["x"] = x_slice
        dim_slices["xh"] = x_slice
    if y_slice is not None:
        dim_slices["y"] = y_slice
        dim_slices["yh"] = y_slice
    return dim_slices or None


def _tile_indices(pe_tag: str) -> tuple[int, int, int]:
    pe_norm = _normalize_pe_tag(pe_tag)
    idx = int(pe_norm[2:])
    tile_i = idx % PRC_NUM_X
    tile_j = idx // PRC_NUM_X
    return idx, tile_i, tile_j


def _tile_tag(tile_i: int, tile_j: int) -> str | None:
    if 0 <= tile_i < PRC_NUM_X and 0 <= tile_j < PRC_NUM_Y:
        idx = tile_j * PRC_NUM_X + tile_i
        return f"pe{idx:06d}"
    return None


def _neighbor_dataset(
    dump_dir: str | Path,
    prefix: str,
    tile_i: int,
    tile_j: int,
    dx: int,
    dy: int,
    *,
    x_slice: slice | None = None,
    y_slice: slice | None = None,
) -> Dataset | None:
    tag = _tile_tag(tile_i + dx, tile_j + dy)
    if tag is None:
        return None
    dim_slices = _build_dim_slices(x_slice, y_slice)
    return load_scale_state(dump_dir, tag, prefix, dim_slices=dim_slices)


def _concat_row(blocks: Sequence[Dataset | None]) -> dict | None:
    available = [block for block in blocks if block is not None]
    if not available:
        return None
    tensors = [block["state"].values for block in blocks if block is not None]
    state_row = torch.cat(tensors, dim=-1)
    height_row = torch.cat([block["height"].values for block in blocks if block is not None], dim=-1)
    lon_row = torch.cat([block["lon"].values for block in blocks if block is not None], dim=-1)
    lat_row = torch.cat([block["lat"].values for block in blocks if block is not None], dim=-1)
    topo_row = torch.cat([block["topo"].values for block in blocks if block is not None], dim=-1)
    x_segments = [tuple(block.coords["x"]) for block in blocks if block is not None]
    xh_segments = [tuple(block.coords.get("xh", block.coords["x"])) for block in blocks if block is not None]
    y_coords = tuple(available[0].coords["y"])
    yh_coords = tuple(available[0].coords.get("yh", available[0].coords["y"]))
    return {
        "state": state_row,
        "height": height_row,
        "lon": lon_row,
        "lat": lat_row,
        "topo": topo_row,
        "x_segments": x_segments,
        "xh_segments": xh_segments,
        "y_coords": y_coords,
        "yh_coords": yh_coords,
    }


def _assemble_rows(rows: Sequence[Sequence[Dataset | None]]) -> list[dict]:
    assembled: list[dict] = []
    for blocks in rows:
        row = _concat_row(blocks)
        if row is not None:
            assembled.append(row)
    return assembled


def _flatten_segments(segments: Sequence[Sequence[float]]) -> Tuple[float, ...]:
    return tuple(chain.from_iterable(segments))


def _compute_extent(base_vals: Sequence[float], extended: Sequence[float]) -> tuple[int, int]:
    base_list = list(base_vals)
    extended_list = list(extended)
    start = extended_list.index(base_list[0])
    end = start + len(base_list)
    left = start
    right = len(extended_list) - end
    return left, right


def _replace_state(dataset: Dataset, new_state: DataTensor, variable_names: Sequence[str]) -> Dataset:
    data_vars = dict(dataset.data_vars)
    data_vars["state"] = new_state
    coords = dict(dataset.coords)
    coords["variable"] = tuple(variable_names)
    return Dataset(data_vars, coords=coords, attrs=dataset.attrs)


def _select_state_field(state: DataTensor, name: str) -> torch.Tensor:
    try:
        selected = state.sel(variable=name)
    except KeyError as error:
        raise KeyError(f"State variable '{name}' not found.") from error
    return selected.values

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


def convert_scale_to_letkf(scale_state: Dataset) -> Dataset:
    if "state" not in scale_state.data_vars:
        raise ValueError("scale_state dataset must include a 'state' variable.")
    state = scale_state["state"]
    rho = _clean_tensor(_select_state_field(state, "DENS"))
    rhot = _clean_tensor(_select_state_field(state, "RHOT"))
    momx = _clean_tensor(_select_state_field(state, "MOMX"))
    momy = _clean_tensor(_select_state_field(state, "MOMY"))
    momz = _clean_tensor(_select_state_field(state, "MOMZ"))

    u = momx / rho
    v = momy / rho
    w = momz / rho

    moist = torch.stack([_clean_tensor(_select_state_field(state, name)) for name in TRACER_SPECIES], dim=0)
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
        "ens": scale_state.coords["ens"],
        "z": scale_state.coords["z"],
        "y": scale_state.coords["y"],
        "x": scale_state.coords["x"],
    }
    control_tensor = DataTensor(control_stack, coords, ("variable", "ens", "z", "y", "x"))
    return _replace_state(scale_state, control_tensor, CONTROL_ORDER)


def compute_grid_params(scale_state: Dataset) -> tuple[float, float, float, float]:
    fxg = scale_state.attrs["fxg"]
    fyg = scale_state.attrs["fyg"]
    cxg0 = scale_state.attrs["cxg0"]
    cyg0 = scale_state.attrs["cyg0"]
    base_x = float(0.5 * (fxg[0] + fxg[-1]))
    base_y = float(0.5 * (fyg[0] + fyg[-1]))
    latrot0 = 0.5 * math.pi - BASE_LAT
    dist0 = 1.0 / math.tan(0.5 * latrot0)
    param_y = base_y - RADIUS * FACT * math.log(dist0)
    return base_x, param_y, cxg0, cyg0

def _scalar_tensor(value: float) -> DataTensor:
    return DataTensor(torch.as_tensor(value, dtype=torch.float64), {}, ())

def load_scale_state(dump_dir: str | Path, pe_tag: str, prefix: str, *, dim_slices: Mapping[str, slice] | None = None) -> Dataset:
    scale_raw = read_and_concat_members(dump_dir, pe_tag, prefix, dim_slices=dim_slices)
    idx, tile_i, tile_j = _tile_indices(pe_tag)
    base_x, param_y, cxg0, cyg0 = compute_grid_params(scale_raw)
    y_coords = scale_raw.coords["y"]
    x_coords = scale_raw.coords["x"]
    z_coords = scale_raw.coords["z"]
    state_arrays: list[torch.Tensor] = []
    for name in SCALE_STATE_ORDER:
        tensor = scale_raw[name].values
        if name == "MOMZ":
            tensor = tensor[:, 1:, :, :]
        state_arrays.append(tensor)

    state_stack = torch.stack(state_arrays, dim=0)
    state_tensor = DataTensor(
        state_stack,
        {
            "variable": SCALE_STATE_ORDER,
            "ens": scale_raw.coords["ens"],
            "z": z_coords,
            "y": y_coords,
            "x": x_coords,
        },
        ("variable", "ens", "z", "y", "x"),
    )

    lon = scale_raw["lon"]
    lat = scale_raw["lat"]
    topo = scale_raw["topo"]
    height = scale_raw["height"]

    ks = 1 + KHALO
    ke = ks + len(z_coords) - 1
    ztop = scale_raw.attrs["fz"][ke - 1] - scale_raw.attrs["fz"][ks - 2]
    cz_slice = scale_raw.attrs["cz"][ks - 1 : ks - 1 + len(z_coords)]
    cz_levels = DataTensor(cz_slice, {"z": z_coords}, ("z",))

    data_vars = {
        "state": state_tensor,
        "lon": lon,
        "lat": lat,
        "height": height,
        "topo": topo,
        "cz_levels": cz_levels,
        "base_x": _scalar_tensor(base_x),
        "param_y": _scalar_tensor(param_y),
        "cxg0": _scalar_tensor(cxg0),
        "cyg0": _scalar_tensor(cyg0),
    }
    coords = {
        "variable": SCALE_STATE_ORDER,
        "ens": scale_raw.coords["ens"],
        "z": z_coords,
        "y": y_coords,
        "x": x_coords,
        "xh": scale_raw.coords.get("xh", x_coords),
        "yh": scale_raw.coords.get("yh", y_coords),
        "zh": scale_raw.coords.get("zh", tuple(range(len(z_coords) + 1))),
    }
    attrs = dict(scale_raw.attrs)
    attrs["ztop"] = float(ztop)
    attrs["tile_index"] = idx
    attrs["tile_i"] = tile_i
    attrs["tile_j"] = tile_j
    return Dataset(data_vars, coords=coords, attrs=attrs)


def load_haloed_scale_state(
    dump_dir: str | Path,
    pe_tag: str,
    prefix: str,
    halo_x: int = IHALO,
    halo_y: int = JHALO,
) -> Dataset:
    base = load_scale_state(dump_dir, pe_tag, prefix)
    halo_x = max(int(halo_x), 0)
    halo_y = max(int(halo_y), 0)
    if halo_x == 0 and halo_y == 0:
        return base
    _, tile_i, tile_j = _tile_indices(pe_tag)
    return _apply_spatial_halos(dump_dir, prefix, base, tile_i, tile_j, halo_x, halo_y)


def _apply_spatial_halos(
    dump_dir: str | Path,
    prefix: str,
    base: Dataset,
    tile_i: int,
    tile_j: int,
    halo_x: int,
    halo_y: int,
) -> Dataset:
    nx = base["state"].sizes["x"]
    ny = base["state"].sizes["y"]
    halo_x = min(halo_x, nx)
    halo_y = min(halo_y, ny)
    left = halo_x if tile_i > 0 else 0
    right = halo_x if tile_i < PRC_NUM_X - 1 else 0
    top = halo_y if tile_j > 0 else 0
    bottom = halo_y if tile_j < PRC_NUM_Y - 1 else 0
    if left == 0 and right == 0 and top == 0 and bottom == 0:
        return base

    left_slice = slice(nx - left, nx) if left else None
    right_slice = slice(0, right) if right else None
    top_slice = slice(ny - top, ny) if top else None
    bottom_slice = slice(0, bottom) if bottom else None

    rows: list[list[Dataset | None]] = []
    if top:
        rows.append(
            [
                _neighbor_dataset(dump_dir, prefix, tile_i, tile_j, -1, -1, x_slice=left_slice, y_slice=top_slice)
                if left
                else None,
                _neighbor_dataset(dump_dir, prefix, tile_i, tile_j, 0, -1, y_slice=top_slice),
                _neighbor_dataset(dump_dir, prefix, tile_i, tile_j, 1, -1, x_slice=right_slice, y_slice=top_slice)
                if right
                else None,
            ]
        )
    rows.append(
        [
            _neighbor_dataset(dump_dir, prefix, tile_i, tile_j, -1, 0, x_slice=left_slice) if left else None,
            base,
            _neighbor_dataset(dump_dir, prefix, tile_i, tile_j, 1, 0, x_slice=right_slice) if right else None,
        ]
    )
    if bottom:
        rows.append(
            [
                _neighbor_dataset(dump_dir, prefix, tile_i, tile_j, -1, 1, x_slice=left_slice, y_slice=bottom_slice)
                if left
                else None,
                _neighbor_dataset(dump_dir, prefix, tile_i, tile_j, 0, 1, y_slice=bottom_slice),
                _neighbor_dataset(dump_dir, prefix, tile_i, tile_j, 1, 1, x_slice=right_slice, y_slice=bottom_slice)
                if right
                else None,
            ]
        )

    assembled_rows = _assemble_rows(rows)
    if not assembled_rows:
        return base

    state_full = torch.cat([row["state"] for row in assembled_rows], dim=-2)
    height_full = torch.cat([row["height"] for row in assembled_rows], dim=-2)
    lon_full = torch.cat([row["lon"] for row in assembled_rows], dim=-2)
    lat_full = torch.cat([row["lat"] for row in assembled_rows], dim=-2)
    topo_full = torch.cat([row["topo"] for row in assembled_rows], dim=-2)

    y_coords = tuple(chain.from_iterable(row["y_coords"] for row in assembled_rows))
    yh_coords = tuple(chain.from_iterable(row["yh_coords"] for row in assembled_rows))
    x_segments = next((row["x_segments"] for row in assembled_rows if row["x_segments"]), None)
    if x_segments is None:
        raise ValueError("Unable to determine x coordinates for halo assembly.")
    new_x = _flatten_segments(x_segments)
    xh_segments = next((row["xh_segments"] for row in assembled_rows if row["xh_segments"]), None)
    new_xh = _flatten_segments(xh_segments) if xh_segments else new_x

    variable_coords = tuple(base["state"].coords["variable"])
    ens_coords = tuple(base.coords["ens"])
    z_coords = tuple(base.coords["z"])

    state_tensor = DataTensor(
        state_full,
        {
            "variable": variable_coords,
            "ens": ens_coords,
            "z": z_coords,
            "y": y_coords,
            "x": new_x,
        },
        ("variable", "ens", "z", "y", "x"),
    )
    height_tensor = DataTensor(height_full, {"z": z_coords, "y": y_coords, "x": new_x}, ("z", "y", "x"))
    lon_tensor = DataTensor(lon_full, {"y": y_coords, "x": new_x}, ("y", "x"))
    lat_tensor = DataTensor(lat_full, {"y": y_coords, "x": new_x}, ("y", "x"))
    topo_tensor = DataTensor(topo_full, {"y": y_coords, "x": new_x}, ("y", "x"))

    base_x_coords = tuple(base.coords["x"])
    base_y_coords = tuple(base.coords["y"])
    left_extent, right_extent = _compute_extent(base_x_coords, new_x)
    top_extent, bottom_extent = _compute_extent(base_y_coords, y_coords)

    data_vars = dict(base.data_vars)
    data_vars.update(
        {
            "state": state_tensor,
            "height": height_tensor,
            "lon": lon_tensor,
            "lat": lat_tensor,
            "topo": topo_tensor,
        }
    )
    coords = dict(base.coords)
    coords.update({"x": new_x, "y": y_coords, "xh": new_xh, "yh": yh_coords})
    attrs = dict(base.attrs)
    attrs["spatial_halo"] = {"x": (left_extent, right_extent), "y": (top_extent, bottom_extent)}
    return Dataset(data_vars, coords=coords, attrs=attrs)


def load_letkf_state(dump_dir: str | Path, pe_tag: str, prefix: str) -> Dataset:
    scale_state = load_scale_state(dump_dir, pe_tag, prefix)
    return convert_scale_to_letkf(scale_state)


def load_haloed_letkf_state(
    dump_dir: str | Path,
    pe_tag: str,
    prefix: str,
    halo_x: int = IHALO,
    halo_y: int = JHALO,
) -> Dataset:
    scale_state = load_haloed_scale_state(dump_dir, pe_tag, prefix, halo_x=halo_x, halo_y=halo_y)
    return convert_scale_to_letkf(scale_state)


def strip_state_halo(dataset: Dataset) -> Dataset:
    halo_meta = dataset.attrs.get("spatial_halo")
    if not halo_meta:
        return dataset
    left, right = halo_meta.get("x", (0, 0))
    top, bottom = halo_meta.get("y", (0, 0))
    left = int(left)
    right = int(right)
    top = int(top)
    bottom = int(bottom)
    if left == 0 and right == 0 and top == 0 and bottom == 0:
        return dataset
    x_size = dataset["state"].sizes["x"]
    y_size = dataset["state"].sizes["y"]
    x_slice = slice(left, x_size - right if right > 0 else x_size)
    y_slice = slice(top, y_size - bottom if bottom > 0 else y_size)

    def _isel_tensor(dt: DataTensor) -> DataTensor:
        indexer: dict[str, slice] = {}
        if "x" in dt.dims:
            indexer["x"] = x_slice
        if "y" in dt.dims:
            indexer["y"] = y_slice
        if not indexer:
            return dt
        return dt.isel(**indexer)

    data_vars = {name: _isel_tensor(var) for name, var in dataset.data_vars.items()}
    coords = dict(dataset.coords)
    if "x" in coords:
        coords["x"] = coords["x"][x_slice]
    if "y" in coords:
        coords["y"] = coords["y"][y_slice]
    attrs = dict(dataset.attrs)
    attrs["spatial_halo"] = {"x": (0, 0), "y": (0, 0)}
    return Dataset(data_vars, coords=coords, attrs=attrs)

def convert_letkf_to_scale(dataset: Dataset) -> Dataset:
    if "state" not in dataset.data_vars:
        raise ValueError("Dataset must include a 'state' variable.")
    control = dataset["state"]
    if "variable" not in control.dims:
        raise ValueError("Control DataTensor must include a 'variable' dimension.")

    required_dims = ("variable", "ens", "z", "y", "x")
    missing = [dim for dim in required_dims if dim not in control.dims]
    if missing:
        raise ValueError(f"Control tensor missing dimensions: {missing}")
    if control.dims != required_dims:
        control = control.transpose(*required_dims)

    def _extract(name: str) -> torch.Tensor:
        try:
            return control.sel(variable=name).values
        except KeyError as error:
            raise KeyError(f"Control variable '{name}' not available.") from error

    moist = torch.stack([_extract(name) for name in TRACER_SPECIES], dim=0)
    moist_cl = torch.nan_to_num(moist, nan=0.0)
    qdry = 1.0 - moist_cl.sum(dim=0)

    tracer_cv = TRACER_CV.view(-1, 1, 1, 1, 1)
    cv_tot = CVDry * qdry + (moist_cl * tracer_cv).sum(dim=0)
    species_index = {name: idx for idx, name in enumerate(TRACER_SPECIES)}
    rtot = RDRY * qdry + RVAP * moist_cl[species_index["QV"]]

    pressure = _extract("P")
    temperature = _extract("T")

    rho = pressure / (rtot * temperature)
    cvovcp = cv_tot / (cv_tot + rtot)
    rhot = PRE00 / rtot * torch.pow(pressure / PRE00, cvovcp)

    momx_mass = rho * _extract("U")
    momy_mass = rho * _extract("V")
    momz_mass = rho * _extract("W")

    scale_stack = torch.stack(
        [
            rho,
            rhot,
            momx_mass,
            momy_mass,
            momz_mass,
            moist[species_index["QV"]],
            moist[species_index["QC"]],
            moist[species_index["QR"]],
            moist[species_index["QI"]],
            moist[species_index["QS"]],
            moist[species_index["QG"]],
        ],
        dim=0,
    )

    coords = {
        "variable": SCALE_STATE_ORDER,
        "ens": dataset.coords["ens"],
        "z": dataset.coords["z"],
        "y": dataset.coords["y"],
        "x": dataset.coords["x"],
    }
    scale_tensor = DataTensor(scale_stack, coords, ("variable", "ens", "z", "y", "x"))
    return _replace_state(dataset, scale_tensor, SCALE_STATE_ORDER)
