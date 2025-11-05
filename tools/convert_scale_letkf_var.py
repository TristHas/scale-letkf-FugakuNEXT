import numpy as np
import xarray as xr
from pathlib import Path

# ------------------------------------------------------------------
# Physical constants (mirroring SCALE’s state_trans routine)
# ------------------------------------------------------------------
RDRY = 287.04
CPDRY = 1004.64
CVDry = CPDRY - RDRY

RVAP = 461.50
CPVAP = 1846.00
CVVAP = CPVAP - RVAP

PRE00 = 100000.0
FILL = -9.9999e30

# MATCH LETKF: all moist tracers use the vapor Cv in state_trans.
TRACER_CV = xr.DataArray(
  [CVVAP, CVVAP, CVVAP, CVVAP, CVVAP, CVVAP],
  dims=("species",),
  coords={"species": ["QV", "QC", "QR", "QI", "QS", "QG"]},
)

CONTROL_ORDER = ("U", "V", "W", "T", "P", "QV", "QC", "QR", "QI", "QS", "QG")

def halo_width(coord: xr.DataArray) -> tuple[int, int]:
      """Read SCALE’s halo metadata for a coordinate axis."""
      halo = coord.attrs.get("halo_local", (0, 0))
      halo = tuple(int(v) for v in np.atleast_1d(halo))
      return halo if len(halo) == 2 else (halo[0], 0)

def strip_halo(da: xr.DataArray, halo_map: dict[str, tuple[int, int]]) -> xr.DataArray:
    """Remove the local halo cells from an array."""
    out = da
    for dim, (left, right) in halo_map.items():
        if dim not in out.dims: continue
        start = left
        stop = None if right == 0 else -right
        out = out.isel({dim: slice(start, stop)})
    return out

def deundef(da: xr.DataArray) -> xr.DataArray:
    """Convert SCALE fill values to NaN."""
    return da.where(np.abs(da - FILL) > 1e20)

def read_halo(ds):
    halo = {}
    for coord_name in ("x", "xh", "y", "yh", "z", "zh"):
        if coord_name in ds.coords:
            left, right = halo_width(ds.coords[coord_name])
            if left or right:
                halo[coord_name] = (left, right)
    return halo

def convert_scale_letkf_var(ds):
    halo = read_halo(ds)
    # Mass-grid density / theta and moisture (trimmed halos, float64, NaNs for fill)
    rho = deundef(strip_halo(ds["DENS"], halo)).astype(np.float64).transpose("z", "y", "x")
    rhot = deundef(strip_halo(ds["RHOT"], halo)).astype(np.float64).transpose("z", "y", "x")
    # Mass-grid coordinates
    x_mass = rho["x"]
    y_mass = rho["y"]
    z_mass = rho["z"]
    ### Moisture
    moist = xr.concat(
      [deundef(strip_halo(ds[name], halo)).astype(np.float64) for name in TRACER_CV["species"].values],
      dim="species",
    )
    moist = moist.assign_coords(species=TRACER_CV["species"]).transpose("species", "z", "y", "x")
    # U winds (x-direction)
    momx = deundef(strip_halo(ds["MOMX"], halo)).astype(np.float64).transpose("z", "y", "xh")
    # state_trans uses the same index for MOMX and the mass grid; align coordinates to mirror that.
    momx_mass = momx.rename({"xh": "x"}).assign_coords(x=x_mass)
    u_mass = momx_mass / rho
    # V winds (y-direction)
    momy = deundef(strip_halo(ds["MOMY"], halo)).astype(np.float64).transpose("z", "yh", "x")
    # Same for MOMY: treat the staggered axis as colocated with the mass points.
    momy_mass = momy.rename({"yh": "y"}).assign_coords(y=y_mass)
    v_mass = momy_mass / rho
    # W winds (z-direction)
    momz = deundef(strip_halo(ds["MOMZ"], halo)).astype(np.float64).transpose("zh", "y", "x")
    # SCALE's state_trans pairs MOMZ(k+1/2) with mass level k; drop the bottom face to mirror that behaviour.
    momz_mass = momz.isel(zh=slice(1, None)).rename({"zh": "z"}).assign_coords(z=z_mass)
    w_mass = momz_mass / rho
    # Reorder wind dims
    u_mass = u_mass.transpose("z", "y", "x")
    v_mass = v_mass.transpose("z", "y", "x")
    w_mass = w_mass.transpose("z", "y", "x")
    # ------------------------------------------------------------------
    # Thermodynamic conversion: rhot -> (T, P), keep moisture as-is
    # ------------------------------------------------------------------
    moist_clean = moist.fillna(0.0)
    qdry = 1.0 - moist_clean.sum("species")
    
    cv_tot = CVDry * qdry + (moist_clean * TRACER_CV).sum("species")
    rtot = RDRY * qdry + RVAP * moist_clean.sel(species="QV")
    
    base = (rhot * rtot) / PRE00
    valid = (base > 0) & (rho > 0) & (cv_tot > 0) & (rtot > 0)
    
    gamma = xr.where(valid, (cv_tot + rtot) / cv_tot, np.nan)
    
    pressure = xr.where(valid, PRE00 * base ** gamma, np.nan)
    temperature = xr.where(valid, pressure / (rho * rtot), np.nan)
    # ------------------------------------------------------------------
    # Assemble LETKF control variables
    # ------------------------------------------------------------------
    control_stack = xr.concat(
      [
          u_mass,
          v_mass,
          w_mass,
          temperature,
          pressure,
          moist.sel(species="QV", drop=True),
          moist.sel(species="QC", drop=True),
          moist.sel(species="QR", drop=True),
          moist.sel(species="QI", drop=True),
          moist.sel(species="QS", drop=True),
          moist.sel(species="QG", drop=True),
      ],
      dim="variable",
    )
    control_stack = control_stack.assign_coords(variable=list(CONTROL_ORDER))
    drop_targets = [name for name in ("xh", "yh", "zh", "species") if name in control_stack.coords]
    return control_stack.drop_vars(drop_targets) if drop_targets else control_stack


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
