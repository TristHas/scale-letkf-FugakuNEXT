import math
import numpy as np
import xarray as xr
from pathlib import Path

# ---------------------------------------------------------------------
# Shared constants and utilities
# ---------------------------------------------------------------------
RDRY = 287.04
CPDRY = 1004.64
CVDry = CPDRY - RDRY

RVAP = 461.50
CPVAP = 1846.00
CVVAP = CPVAP - RVAP

PRE00 = 100000.0
FILL = -9.9999e30

TRACER_CV = xr.DataArray(
    [CVVAP, CVVAP, CVVAP, CVVAP, CVVAP, CVVAP],
    dims=("species",),
    coords={"species": ["QV", "QC", "QR", "QI", "QS", "QG"]},
)

CONTROL_ORDER = ("U", "V", "W", "T", "P",
                 "QV", "QC", "QR", "QI", "QS", "QG")

RADIUS = 6_371_220.0
BASE_LON_DEG = 139.609
BASE_LAT_DEG = 35.861
BASE_LON = math.radians(BASE_LON_DEG)
BASE_LAT = math.radians(BASE_LAT_DEG)
FACT = math.cos(BASE_LAT)


def halo_width(coord):
    h = coord.attrs.get("halo_local", (0, 0))
    h = tuple(int(v) for v in np.atleast_1d(h))
    return h if len(h) == 2 else (h[0], 0)


def read_halo(ds):
    halo = {}
    for name in ("x", "xh", "y", "yh", "z", "zh"):
        if name in ds.coords:
            L, R = halo_width(ds.coords[name])
            if L or R:
                halo[name] = (L, R)
    return halo


def strip_halo(da, halo_map):
    out = da
    for dim, (L, R) in halo_map.items():
        if dim in out.dims:
            out = out.isel({dim: slice(L, None if R == 0 else -R)})
    return out


def deundef(da):
    return da.where(np.abs(da - FILL) > 1e20)

# ---------------------------------------------------------------------
# 1. Load each member and concatenate on ens dimension.
# ---------------------------------------------------------------------
def read_and_concat_members(dump_dir: str | Path, pe_tag: str, prefix: str="anal_f") -> xr.Dataset:
    dump_dir = Path(dump_dir)
    members = ["0001", "0002", "mean"]
    datasets = []

    for mem in members:
        fname = f"init_20210730-060030.000.{pe_tag}.nc"
        path = dump_dir / ".." / prefix / mem / fname
        ds = xr.open_dataset(path, engine="netcdf4")
        datasets.append(ds.expand_dims(ens=[mem]))

    raw = xr.concat(datasets, dim="ens")
    return raw

# ---------------------------------------------------------------------
# 2. Convert SCALE → LETKF 
# ---------------------------------------------------------------------
def convert_scale_to_letkf(ds: xr.Dataset) -> xr.DataArray:
    """
    Convert SCALE prognostic variables → LETKF control variables.
    Works directly on (ens, z, y, x).
    No halo stripping.
    """
    #height = deundef(ds["height"]).astype(np.float64).transpose("ens", "z", "y", "x")
    rho  = deundef(ds["DENS"]).astype(np.float64).transpose("ens", "z", "y", "x")
    rhot = deundef(ds["RHOT"]).astype(np.float64).transpose("ens", "z", "y", "x")

    # Momentum
    momx = deundef(ds["MOMX"]).astype(np.float64).transpose("ens", "z", "y", "xh")
    momy = deundef(ds["MOMY"]).astype(np.float64).transpose("ens", "z", "yh", "x")
    momz = deundef(ds["MOMZ"]).astype(np.float64).transpose("ens", "zh", "y", "x")

    x_mass = rho["x"]; y_mass = rho["y"]; z_mass = rho["z"]

    momx = momx.rename({"xh": "x"}).assign_coords(x=x_mass)
    momy = momy.rename({"yh": "y"}).assign_coords(y=y_mass)
    momz = momz.isel(zh=slice(1, None)).rename({"zh": "z"}).assign_coords(z=z_mass)

    u = momx / rho
    v = momy / rho
    w = momz / rho

    # Moisture (species, ens, z, y, x)
    moist = xr.concat(
        [deundef(ds[name]).astype(np.float64)
         for name in TRACER_CV["species"].values],
        dim="species",
    )
    moist = moist.assign_coords(species=TRACER_CV["species"])
    moist = moist.transpose("species", "ens", "z", "y", "x")

    # Thermodynamics
    moist_cl = moist.fillna(0.0)
    qdry = 1.0 - moist_cl.sum("species")

    cv_tot = CVDry * qdry + (moist_cl * TRACER_CV).sum("species")
    rtot   = RDRY * qdry + RVAP * moist_cl.sel(species="QV")

    base = (rhot * rtot) / PRE00
    valid = (base > 0) & (rho > 0) & (cv_tot > 0) & (rtot > 0)

    gamma = xr.where(valid, (cv_tot + rtot) / cv_tot, np.nan)

    pressure    = xr.where(valid, PRE00 * base ** gamma, np.nan)
    temperature = xr.where(valid, pressure / (rho * rtot), np.nan)

    # LETKF variable stack
    control = xr.concat(
        [
            u, v, w,
            temperature, pressure,
            moist.sel(species="QV", drop=True),
            moist.sel(species="QC", drop=True),
            moist.sel(species="QR", drop=True),
            moist.sel(species="QI", drop=True),
            moist.sel(species="QS", drop=True),
            moist.sel(species="QG", drop=True),
        ],
        dim="variable",
    )

    control = control.assign_coords(variable=list(CONTROL_ORDER))
    return control


def strip_all_halos(control: xr.DataArray, halo_map) -> xr.DataArray:
    out = control
    for dim, (L, R) in halo_map.items():
        if dim in out.dims:
            out = out.isel({dim: slice(L, None if R == 0 else -R)})
    return out

def compute_grid_params(ds):
    fxg = ds["FXG"].values
    fyg = ds["FYG"].values
    cxg0 = float(ds["CXG"].values[0])
    cyg0 = float(ds["CYG"].values[0])
    ds.close()
    base_x = 0.5 * (fxg[0] + fxg[-1])
    base_y = 0.5 * (fyg[0] + fyg[-1])
    latrot0 = 0.5 * math.pi - BASE_LAT
    dist0 = 1.0 / math.tan(0.5 * latrot0)
    param_y = base_y - RADIUS * FACT * math.log(dist0)
    return base_x, param_y, cxg0, cyg0
    
def load_letkf_state(dump_dir, prefix, pe_tag, strip_hallow=True):
    raw = read_and_concat_members(dump_dir, prefix, pe_tag)
    letkf_state = convert_scale_to_letkf(raw)
    base_x, param_y, cxg0, cyg0 = compute_grid_params(raw)
    converted = xr.Dataset({
        "state":letkf_state,
        "lon": raw["lon"].isel(ens=0),
        "lat": raw["lat"].isel(ens=0),
        "height": raw["height"].isel(ens=0),
        "base_x":base_x, 
        "param_y":param_y, 
        "cxg0":cxg0, "cyg0":cyg0
    })
    if strip_hallow:
        halo_map = read_halo(raw)
        converted = strip_all_halos(converted, halo_map)
    return converted

def convert_letkf_to_scale(control: xr.DataArray) -> xr.Dataset:
    """
    Exact inverse of convert_scale_to_letkf().
    Input:
        control: LETKF control array with shape
                 (variable, ens, z, y, x)
    Output:
        Dataset with fields:
            DENS, RHOT, MOMX, MOMY, MOMZ,
            QV, QC, QR, QI, QS, QG
    """

    if set(control["variable"].values) != set(CONTROL_ORDER):
        raise ValueError("LETKF variable order mismatch")

    ctrl = control.transpose("variable", "ens", "z", "y", "x")

    def _sel(name: str) -> xr.DataArray:
        return ctrl.sel(variable=name).reset_coords(drop=True)

    # ───────────────────────────────────────────────
    # 1. Extract basic variables
    # ───────────────────────────────────────────────
    U = _sel("U")
    V = _sel("V")
    W = _sel("W")
    T = _sel("T")
    P = _sel("P")

    # Moist species in correct order
    moist = xr.concat(
        [_sel(s) for s in TRACER_CV["species"].values],
        dim="species"
    )
    moist = moist.assign_coords(species=TRACER_CV["species"])
    moist = moist.transpose("species", "ens", "z", "y", "x")

    # ───────────────────────────────────────────────
    # 2. Thermodynamics – reverse of forward mapping
    # ───────────────────────────────────────────────
    qdry = (1.0 - moist.sum("species")).clip(min=1e-12)
    rtot = (RDRY * qdry + RVAP * moist.sel(species="QV")).reset_coords(drop=True)
    cv_tot = (CVDry * qdry + (moist * TRACER_CV).sum("species")).reset_coords(drop=True)

    rho = (P / (rtot * T)).reset_coords(drop=True)
    cvovcp = (cv_tot / (cv_tot + rtot)).reset_coords(drop=True)

    rhot = (PRE00 / rtot * (P / PRE00) ** cvovcp).reset_coords(drop=True)

    # ───────────────────────────────────────────────
    # 3. Momentum (reverse of renamings/staggers)
    # ───────────────────────────────────────────────
    momx_mass = (rho * U).reset_coords(drop=True)
    momy_mass = (rho * V).reset_coords(drop=True)
    momz_mass = (rho * W).reset_coords(drop=True)

    # MOMX: unstagger x (mass → xh)
    momx = momx_mass.rename({"x": "xh"}).transpose("ens", "z", "y", "xh")

    # MOMY: unstagger y (mass → yh)
    momy = momy_mass.rename({"y": "yh"}).transpose("ens", "z", "yh", "x")

    # MOMZ: restagger z → zh (one extra vertical face)
    momz_faces = momz_mass.pad(z=(1, 0), mode="edge").rename({"z": "zh"})
    momz = momz_faces.transpose("ens", "zh", "y", "x")

    # ───────────────────────────────────────────────
    # 4. Moisture species → individual prognostic vars
    # ───────────────────────────────────────────────
    moist_split = {
        name: moist.sel(species=name).reset_coords(drop=True).transpose("ens", "z", "y", "x")
        for name in TRACER_CV["species"].values
    }

    # ───────────────────────────────────────────────
    # 5. Build Dataset
    # ───────────────────────────────────────────────
    ds = xr.Dataset(
        data_vars={
            "DENS": rho.transpose("ens", "z", "y", "x"),
            "RHOT": rhot.transpose("ens", "z", "y", "x"),
            "MOMX": momx,
            "MOMY": momy,
            "MOMZ": momz,
            **{name: moist_split[name] for name in TRACER_CV["species"].values},
        },
        coords={
            "ens": ctrl["ens"],
            "x": ctrl["x"],
            "y": ctrl["y"],
            "z": ctrl["z"],
            "xh": momx["xh"],
            "yh": momy["yh"],
            "zh": momz["zh"],
        },
    )

    # Restore halo markers (consistent with forward version removing them)
    for c in ("x", "xh", "y", "yh", "z", "zh"):
        if c in ds:
            ds[c].attrs["halo_local"] = (0, 0)

    return ds

def convert_letkf_scale_var(control: xr.DataArray) -> xr.Dataset:
    """Map LETKF control variables back to SCALE prognostic fields."""
    if "ens" not in control.dims:
        raise ValueError("LETKF control array must include an 'ens' dimension.")
    ctrl = control.transpose("variable", "ens", "z", "y", "x")

    def _sel(name: str) -> xr.DataArray:
        return ctrl.sel(variable=name).reset_coords(drop=True)

    # Moisture block
    moist = xr.concat([_sel(name) for name in TRACER_CV["species"].values], dim="species")
    moist = moist.assign_coords(species=TRACER_CV["species"])
    moist = moist.transpose("species", "ens", "z", "y", "x")

    qdry = (1.0 - moist.sum("species")).clip(min=1e-12)
    rtot = (RDRY * qdry + RVAP * moist.sel(species="QV")).reset_coords(drop=True)
    cv_tot = (CVDry * qdry + (moist * TRACER_CV).sum("species")).reset_coords(drop=True)

    pressure = _sel("P").transpose("ens", "z", "y", "x")
    temperature = _sel("T").transpose("ens", "z", "y", "x")

    rho = (pressure / (rtot * temperature)).reset_coords(drop=True)
    cvovcp = (cv_tot / (cv_tot + rtot)).reset_coords(drop=True)
    rhot = (PRE00 / rtot * (pressure / PRE00) ** cvovcp).reset_coords(drop=True)

    momx_mass = (rho * _sel("U").transpose("ens", "z", "y", "x")).reset_coords(drop=True)
    momy_mass = (rho * _sel("V").transpose("ens", "z", "y", "x")).reset_coords(drop=True)
    momz_mass = (rho * _sel("W").transpose("ens", "z", "y", "x")).reset_coords(drop=True)
    momz_faces = momz_mass.pad(z=(1, 0), mode="edge").rename({"z": "zh"})

    x_mass = ctrl["x"].copy()
    y_mass = ctrl["y"].copy()
    z_mass = ctrl["z"].copy()
    for coord in (x_mass, y_mass, z_mass):
        coord.attrs["halo_local"] = (0, 0)

    xh = xr.DataArray(x_mass.values, dims=("xh",), coords={"xh": x_mass.values}, attrs={"halo_local": (0, 0)})
    yh = xr.DataArray(y_mass.values, dims=("yh",), coords={"yh": y_mass.values}, attrs={"halo_local": (0, 0)})
    zh_vals = np.arange(z_mass.size + 1, dtype=np.float64)
    zh = xr.DataArray(zh_vals, dims=("zh",), attrs={"halo_local": (0, 0)})

    momx_data = momx_mass.transpose("ens", "y", "x", "z").rename({"x": "xh"}).assign_coords(xh=xh.values)
    momy_data = momy_mass.transpose("ens", "y", "x", "z").rename({"y": "yh"}).assign_coords(yh=yh.values)
    momz_data = momz_faces.assign_coords(zh=zh.values).transpose("ens", "y", "x", "zh")

    def _moist(name: str) -> xr.DataArray:
        return moist.sel(species=name).reset_coords(drop=True)

    dataset = xr.Dataset(
        data_vars={
            "DENS": rho.transpose("ens", "y", "x", "z"),
            "RHOT": rhot.transpose("ens", "y", "x", "z"),
            "MOMX": momx_data,
            "MOMY": momy_data,
            "MOMZ": momz_data,
            "QV": _moist("QV").transpose("ens", "y", "x", "z"),
            "QC": _moist("QC").transpose("ens", "y", "x", "z"),
            "QR": _moist("QR").transpose("ens", "y", "x", "z"),
            "QI": _moist("QI").transpose("ens", "y", "x", "z"),
            "QS": _moist("QS").transpose("ens", "y", "x", "z"),
            "QG": _moist("QG").transpose("ens", "y", "x", "z"),
        },
        coords={
            "x": x_mass,
            "y": y_mass,
            "z": z_mass,
            "xh": xh,
            "yh": yh,
            "zh": zh,
            "ens": ctrl["ens"],
        },
    )

    return dataset

