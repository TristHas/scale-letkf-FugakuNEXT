import numpy as np
import xarray as xr

KHALO = 2

def _fractional_index_unit(size: int, coord: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coord0 = np.clip(coord - 1.0, 0.0, size - 1.0 - 1.0e-6)
    lo = np.floor(coord0).astype(np.int64)
    frac = coord0 - lo
    return lo, frac

def _interpolate_height_columns(height: np.ndarray, ix0, fx, iy0, fy) -> np.ndarray:
    ix1 = np.clip(ix0 + 1, 0, height.shape[1] - 1)
    iy1 = np.clip(iy0 + 1, 0, height.shape[0] - 1)
    fx = fx[:, None]
    fy = fy[:, None]

    h00 = height[iy0, ix0, :]
    h10 = height[iy0, ix1, :]
    h01 = height[iy1, ix0, :]
    h11 = height[iy1, ix1, :]
    h0 = h00 * (1.0 - fx) + h10 * fx
    h1 = h01 * (1.0 - fx) + h11 * fx
    return h0 * (1.0 - fy) + h1 * fy

def _vertical_index_from_height(columns: np.ndarray, lev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nobs, nz = columns.shape
    rk = np.zeros(nobs, dtype=np.float64)
    for i in range(nobs):
        prof = columns[i]
        target = lev[i]
        if target <= prof[0]:
            k = 0
            frac = 0.0
        elif target >= prof[-1]:
            k = nz - 2
            frac = 0.999
        else:
            k = np.searchsorted(prof, target) - 1
            k = np.clip(k, 0, nz - 2)
            denom = prof[k + 1] - prof[k]
            frac = 0.0 if denom <= 0.0 else (target - prof[k]) / denom
        rk[i] = k + frac
    iz0 = np.clip(np.floor(rk).astype(np.int64), 0, nz - 2)
    fz = rk - iz0
    return iz0, fz

def sample_state(
    state: xr.DataArray,
    height: np.ndarray,
    ri_local: np.ndarray,
    rj_local: np.ndarray,
    lev: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    state_cube = state.transpose("y", "x", "z", "ens", "variable").values

    ix0, fx = _fractional_index_unit(state_cube.shape[1], ri_local)
    iy0, fy = _fractional_index_unit(state_cube.shape[0], rj_local)
    
    height_cols = _interpolate_height_columns(height, ix0, fx, iy0, fy)
    iz0, fz = _vertical_index_from_height(height_cols, lev)

    samples: dict[str, np.ndarray] = {}
    var_names = list(state["variable"].values)
    for ivar, name in enumerate(var_names):
        field = state_cube[..., ivar]
        samples[name] = _sample_cube(field, ix0, fx, iy0, fy, iz0, fz)
    rk = iz0.astype(np.float64) + fz + KHALO
    return samples, rk

def _sample_cube_old(field: np.ndarray, ix0, fx, iy0, fy, iz0, fz) -> np.ndarray:
    ix1 = np.clip(ix0 + 1, 0, field.shape[1] - 1)
    iy1 = np.clip(iy0 + 1, 0, field.shape[0] - 1)
    iz1 = np.clip(iz0 + 1, 0, field.shape[2] - 1)

    fx = fx[:, None]
    fy = fy[:, None]
    fz = fz[:, None]

    c000 = field[iy0, ix0, iz0]
    c100 = field[iy0, ix1, iz0]
    c010 = field[iy1, ix0, iz0]
    c110 = field[iy1, ix1, iz0]
    c001 = field[iy0, ix0, iz1]
    c101 = field[iy0, ix1, iz1]
    c011 = field[iy1, ix0, iz1]
    c111 = field[iy1, ix1, iz1]

    c00 = c000 * (1.0 - fx) + c100 * fx
    c01 = c010 * (1.0 - fx) + c110 * fx
    c10 = c001 * (1.0 - fx) + c101 * fx
    c11 = c011 * (1.0 - fx) + c111 * fx
    c0 = c00 * (1.0 - fy) + c01 * fy
    c1 = c10 * (1.0 - fy) + c11 * fy
    return c0 * (1.0 - fz) + c1 * fz

def sample_state(
    state: xr.DataArray,
    height: np.ndarray,
    ri_local: np.ndarray,
    rj_local: np.ndarray,
    lev: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:

    # shape: (y, x, z, ens, variable)
    state_cube = state.transpose("y", "x", "z", "ens", "variable").values

    ix0, fx = _fractional_index_unit(state_cube.shape[1], ri_local)
    iy0, fy = _fractional_index_unit(state_cube.shape[0], rj_local)

    height_cols = _interpolate_height_columns(height, ix0, fx, iy0, fy)
    iz0, fz = _vertical_index_from_height(height_cols, lev)

    samples = _sample_cube(state_cube, ix0, fx, iy0, fy, iz0, fz)

    # Vertical fractional index
    rk = iz0.astype(np.float64) + fz + KHALO
    return samples, rk

def _sample_cube(field5d, ix0, fx, iy0, fy, iz0, fz):
    """
    field5d: array of shape (y, x, z, ens, var)
    ix0, iy0, iz0: int arrays of shape (nobs,)
    fx, fy, fz: float arrays of shape (nobs,)
    """

    ix1 = np.clip(ix0 + 1, 0, field5d.shape[1] - 1)
    iy1 = np.clip(iy0 + 1, 0, field5d.shape[0] - 1)
    iz1 = np.clip(iz0 + 1, 0, field5d.shape[2] - 1)

    # Make broadcastable: (nobs,1,1)
    fx = fx[:, None, None]
    fy = fy[:, None, None]
    fz = fz[:, None, None]

    # Gather corner values for all ensembles & variables
    c000 = field5d[iy0, ix0, iz0]
    c100 = field5d[iy0, ix1, iz0]
    c010 = field5d[iy1, ix0, iz0]
    c110 = field5d[iy1, ix1, iz0]

    c001 = field5d[iy0, ix0, iz1]
    c101 = field5d[iy0, ix1, iz1]
    c011 = field5d[iy1, ix0, iz1]
    c111 = field5d[iy1, ix1, iz1]

    # Linear interpolation, fully vectorized
    c00 = c000 * (1 - fx) + c100 * fx
    c01 = c010 * (1 - fx) + c110 * fx
    c10 = c001 * (1 - fx) + c101 * fx
    c11 = c011 * (1 - fx) + c111 * fx

    c0 = c00 * (1 - fy) + c01 * fy
    c1 = c10 * (1 - fy) + c11 * fy

    out = c0 * (1 - fz) + c1 * fz  # shape: (nobs, ens, var)
    return out
