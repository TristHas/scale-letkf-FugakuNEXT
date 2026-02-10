from __future__ import annotations

import math
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

DUMP_DIR  = Path("result/SC23/20210730060030/letkf_dump")

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


def _tile_domain_halo(tile_i: int, tile_j: int) -> dict[str, tuple[int, int]]:
    halo_x_left = IHALO if tile_i == 0 else 0
    halo_x_right = IHALO if tile_i == PRC_NUM_X - 1 else 0
    halo_y_top = JHALO if tile_j == 0 else 0
    halo_y_bottom = JHALO if tile_j == PRC_NUM_Y - 1 else 0
    return {
        "x": (halo_x_left, halo_x_right),
        "xh": (halo_x_left, halo_x_right),
        "y": (halo_y_top, halo_y_bottom),
        "yh": (halo_y_top, halo_y_bottom),
    }


def _axis_interior_bounds(tile_i: int, tile_j: int, axis: str) -> tuple[int, int]:
    halo = _tile_domain_halo(tile_i, tile_j)[axis]
    extent = NX_TILE if axis in ("x", "xh") else NY_TILE
    start = halo[0]
    end = start + extent
    return start, end


def _neighbor_axis_slice(
    tile_i: int,
    tile_j: int,
    axis: str,
    section: str,
    width: int | None,
) -> slice | None:
    if section == "full":
        start, end = _axis_interior_bounds(tile_i, tile_j, axis)
        return slice(start, end)
    if not width or width <= 0:
        return None
    start, end = _axis_interior_bounds(tile_i, tile_j, axis)
    if section in ("left", "top"):
        return slice(start, start + width)
    if section in ("right", "bottom"):
        return slice(end - width, end)
    raise ValueError(f"Unsupported section '{section}' for axis '{axis}'")


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

def load_scale_state(
    tile_idx: int,
    halo_i: int = 0,
    halo_j: int = 0,
    dump_dir = DUMP_DIR,
    prefix: str = "anal_f",
    *,
    dim_slices: Mapping[str, slice] | None = None,
) -> Dataset:        
    pe_tag = f"pe{str(tile_idx).zfill(6)}"
    scale_raw = read_and_concat_members(
        dump_dir,
        pe_tag,
        prefix,
        halo_i=halo_i,
        halo_j=halo_j,
        dim_slices=dim_slices,
    )
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
    attrs["dump_dir"] = str(dump_dir)
    attrs["state_prefix"] = prefix
    attrs["pe_tag"] = pe_tag
    return Dataset(data_vars, coords=coords, attrs=attrs)


def load_haloed_scale_state(
    dump_dir: str | Path,
    pe_tag: str,
    prefix: str,
    halo_x: int = IHALO,
    halo_y: int = JHALO,
) -> Dataset:
    halo_x = max(int(halo_x), 0)
    halo_y = max(int(halo_y), 0)
    base = load_scale_state(pe_tag, 
                            halo_i=halo_x, halo_j=halo_y, 
                            prefix=prefix, dump_dir=dump_dir)
    if halo_x == 0 and halo_y == 0:
        return base
    return read_tile_halos(base)


def _assign_halo_block(target: Dataset, source: Dataset, x_slice: slice, y_slice: slice) -> None:
    if x_slice is None or y_slice is None or source is None:
        return
    for name in ("state", "height", "lon", "lat", "topo"):
        if name not in target.data_vars or name not in source.data_vars:
            continue
        target_tensor = target[name]
        source_tensor = source[name]
        indexer: list[slice] = []
        for dim in target_tensor.dims:
            if dim == "x":
                indexer.append(x_slice)
            elif dim == "y":
                indexer.append(y_slice)
            else:
                indexer.append(slice(None))
        target_tensor.values[tuple(indexer)] = source_tensor.values


def _load_neighbor_block(
    dump_dir: Path,
    prefix: str,
    tile_i: int,
    tile_j: int,
    dx: int,
    dy: int,
    x_section: str,
    x_width: int | None,
    y_section: str,
    y_width: int | None,
) -> Dataset | None:
    neighbor_i = tile_i + dx
    neighbor_j = tile_j + dy
    if _tile_tag(neighbor_i, neighbor_j) is None:
        return None
    x_slice = _neighbor_axis_slice(neighbor_i, neighbor_j, "x", x_section, x_width)
    y_slice = _neighbor_axis_slice(neighbor_i, neighbor_j, "y", y_section, y_width)
    try:
        return _neighbor_dataset(
            dump_dir,
            prefix,
            tile_i,
            tile_j,
            dx,
            dy,
            x_slice=x_slice,
            y_slice=y_slice,
        )
    except FileNotFoundError:
        return None


def read_tile_halos(state: Dataset) -> Dataset:
    halo_meta = state.attrs.get("spatial_halo")
    if not halo_meta:
        return state
    halo_x = halo_meta.get("x", (0, 0))
    halo_y = halo_meta.get("y", (0, 0))
    left_total = max(int(halo_x[0]), 0)
    right_total = max(int(halo_x[1]), 0)
    top_total = max(int(halo_y[0]), 0)
    bottom_total = max(int(halo_y[1]), 0)
    if left_total == right_total == top_total == bottom_total == 0:
        return state

    try:
        tile_i = int(state.attrs["tile_i"])
        tile_j = int(state.attrs["tile_j"])
    except KeyError as error:
        raise ValueError("State dataset missing tile indices required for halo filling.") from error
    dump_dir_attr = state.attrs.get("dump_dir")
    prefix = state.attrs.get("state_prefix")
    if dump_dir_attr is None or prefix is None:
        raise ValueError("State dataset missing dump metadata required for halo filling.")
    dump_dir = Path(dump_dir_attr)
    prefix = str(prefix)

    x_size = state["state"].sizes["x"]
    y_size = state["state"].sizes["y"]
    left_fill = min(left_total, x_size) if tile_i > 0 else 0
    right_fill = min(right_total, x_size) if tile_i < PRC_NUM_X - 1 else 0
    top_fill = min(top_total, y_size) if tile_j > 0 else 0
    bottom_fill = min(bottom_total, y_size) if tile_j < PRC_NUM_Y - 1 else 0

    def _make_slice(start: int, end: int) -> slice | None:
        if end <= start:
            return None
        return slice(start, end)

    x_left_slice = _make_slice(0, left_fill)
    x_right_slice = _make_slice(x_size - right_fill, x_size)
    y_top_slice = _make_slice(0, top_fill)
    y_bottom_slice = _make_slice(y_size - bottom_fill, y_size)
    x_center_slice = _make_slice(left_total, x_size - right_total)
    y_center_slice = _make_slice(top_total, y_size - bottom_total)

    operations = [
        # Top row
        (x_left_slice, y_top_slice, -1, -1, "right", left_fill, "bottom", top_fill),
        (x_center_slice, y_top_slice, 0, -1, "full", None, "bottom", top_fill),
        (x_right_slice, y_top_slice, 1, -1, "left", right_fill, "bottom", top_fill),
        # Middle row
        (x_left_slice, y_center_slice, -1, 0, "right", left_fill, "full", None),
        (x_right_slice, y_center_slice, 1, 0, "left", right_fill, "full", None),
        # Bottom row
        (x_left_slice, y_bottom_slice, -1, 1, "right", left_fill, "top", bottom_fill),
        (x_center_slice, y_bottom_slice, 0, 1, "full", None, "top", bottom_fill),
        (x_right_slice, y_bottom_slice, 1, 1, "left", right_fill, "top", bottom_fill),
    ]

    for x_slice, y_slice, dx, dy, x_section, x_width, y_section, y_width in operations:
        if x_slice is None or y_slice is None:
            continue
        if x_section != "full" and (x_width is None or x_width <= 0):
            continue
        if y_section != "full" and (y_width is None or y_width <= 0):
            continue
        neighbor = _load_neighbor_block(
            dump_dir,
            prefix,
            tile_i,
            tile_j,
            dx,
            dy,
            x_section,
            x_width,
            y_section,
            y_width,
        )
        if neighbor is None:
            continue
        _assign_halo_block(state, neighbor, x_slice, y_slice)
    return state


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
