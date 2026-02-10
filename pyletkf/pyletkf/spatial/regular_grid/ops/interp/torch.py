from tqdm.auto import tqdm
import torch

KHALO = 2

def _fractional_index_unit(size: int, coord: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    coord = coord - .5
    lo = coord.floor().long()
    frac = coord - lo.to(coord.dtype)
    return lo, frac

def _interpolate_height_columns(height_yx: torch.Tensor, ix0, fx, iy0, fy) -> torch.Tensor:
    ix1 = torch.clamp(ix0 + 1, 0, height_yx.shape[1] - 1)
    iy1 = torch.clamp(iy0 + 1, 0, height_yx.shape[0] - 1)

    fx = fx.unsqueeze(1)
    fy = fy.unsqueeze(1)

    h00 = height_yx[iy0, ix0]
    h10 = height_yx[iy0, ix1]
    h01 = height_yx[iy1, ix0]
    h11 = height_yx[iy1, ix1]
    h0  = h00 * (1.0 - fx) + h10 * fx
    h1  = h01 * (1.0 - fx) + h11 * fx
    return h0 * (1.0 - fy) + h1 * fy

def _vertical_index_from_height(columns: torch.Tensor, 
                                lev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    nobs, nz = columns.shape
    expanded = lev.unsqueeze(1).expand(-1, nz)
    mask = expanded >= columns
    idx = torch.clamp(mask.sum(dim=1) - 1, 0, nz - 2)
    v0 = columns[torch.arange(nobs), idx]
    v1 = columns[torch.arange(nobs), idx + 1]
    denom = v1 - v0
    frac = torch.where(denom > 0.0, (lev - v0) / denom, torch.zeros_like(v0))
    return idx, frac

def _bilinear_sample(field_yx: torch.Tensor, ix0, fx, iy0, fy) -> torch.Tensor:
    ix1 = torch.clamp(ix0 + 1, 0, field_yx.shape[1] - 1)
    iy1 = torch.clamp(iy0 + 1, 0, field_yx.shape[0] - 1)
    f00 = field_yx[iy0, ix0]
    f10 = field_yx[iy0, ix1]
    f01 = field_yx[iy1, ix0]
    f11 = field_yx[iy1, ix1]
    f0  = f00 * (1.0 - fx) + f10 * fx
    f1  = f01 * (1.0 - fx) + f11 * fx
    return f0 * (1.0 - fy) + f1 * fy

def _sample_cube(field5d, ix0, fx, iy0, fy, iz0, fz):
    ix1 = torch.clamp(ix0 + 1, 0, field5d.shape[1] - 1)
    iy1 = torch.clamp(iy0 + 1, 0, field5d.shape[0] - 1)
    iz1 = torch.clamp(iz0 + 1, 0, field5d.shape[2] - 1)

    fx = fx[:, None, None]
    fy = fy[:, None, None]
    fz = fz[:, None, None]

    c000 = field5d[iy0, ix0, iz0]
    c100 = field5d[iy0, ix1, iz0]
    c010 = field5d[iy1, ix0, iz0]
    c110 = field5d[iy1, ix1, iz0]
    c001 = field5d[iy0, ix0, iz1]
    c101 = field5d[iy0, ix1, iz1]
    c011 = field5d[iy1, ix0, iz1]
    c111 = field5d[iy1, ix1, iz1]

    c00 = c000 * (1.0 - fx) + c100 * fx
    c01 = c010 * (1.0 - fx) + c110 * fx
    c10 = c001 * (1.0 - fx) + c101 * fx
    c11 = c011 * (1.0 - fx) + c111 * fx

    c0 = c00 * (1.0 - fy) + c01 * fy
    c1 = c10 * (1.0 - fy) + c11 * fy

    return c0 * (1.0 - fz) + c1 * fz

def sample_state(
        state,   # (y, x, z, ens, var)
        height,  # (z, y, x)
        ri,
        rj,
        lev,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lev_tensor = lev.data.to(torch.float64)

    ix0, fx = _fractional_index_unit(state.shape[1], ri)
    iy0, fy = _fractional_index_unit(state.shape[0], rj)

    columns = _interpolate_height_columns(height, ix0, fx, iy0, fy)
    min_height = columns[:, 0]
    max_height = columns[:, -1]
    lev_clamped = torch.clamp(lev, min_height, max_height)
    iz0, fz = _vertical_index_from_height(columns, lev_clamped)

    samples = _sample_cube(state, ix0, fx, iy0, fy, iz0, fz)
    rk = iz0.to(torch.float64) + fz + KHALO
    valid_mask = (lev >= min_height) & (lev <= max_height)
    return samples, rk, valid_mask