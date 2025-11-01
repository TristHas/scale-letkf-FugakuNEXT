from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import xarray as xr

from . import load_guess_rank, load_analysis_rank, load_obs_rank  # re-export convenience
from .letkf_dump_loader import _read_fortran_binary

UNDEF = -9.99e33  # SCALE missing-value sentinel


def reshape_state3d_raw(
    raw: np.ndarray,
    metadata: dict,
    fill_value: float = np.nan,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reshape a `(point, level, member, variable)` array onto the contiguous tile owned by one rank.

    Parameters
    ----------
    raw
        Output of `_read_fortran_binary` for a `*3d_peXXXXX.bin` file.
        Shape must be `(npoint, nlev, nens, nvar)`.
    metadata
        Parsed contents of `state_meta_peXXXXX.txt`. Needs keys:
        `myrank_e`, `nlon`, `nlat`, and either `nij1_rank` or `nij1`.
    fill_value
        Value used to initialise grid cells that the rank does not own.

    Returns
    -------
    grid : np.ndarray
        Array of shape `(nens, nlev, ny_local, nx_local, nvar)` containing the
        rank’s local tile (columns/rows sorted by global index).
    y_coords : np.ndarray
        Global latitude indices for each local row.
    x_global : np.ndarray
        `ny_local` × `nx_local` array giving the global longitude index of each column.
    """
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 4:
        raise ValueError(f"Expected 4-D raw array (point, level, member, variable); got {raw.shape}")

    npoint, nlev, nens, nvar = raw.shape

    nlon = int(metadata["nlon"])
    nlat = int(metadata["nlat"])
    nij1 = int(metadata.get("nij1_rank", metadata.get("nij1", npoint)))
    myrank_e = int(metadata["myrank_e"])

    total_points = nlon * nlat
    nprocs_e = max(1, int(round(total_points / nij1)))

    point_idx = np.arange(npoint, dtype=np.int64)
    linear_idx = myrank_e + nprocs_e * point_idx

    valid_mask = linear_idx < total_points
    if not np.all(valid_mask):
        raw = raw[valid_mask, ...]
        linear_idx = linear_idx[valid_mask]

    x_idx = (linear_idx % nlon).astype(np.int64)
    y_idx = (linear_idx // nlon).astype(np.int64)

    unique_y = np.unique(y_idx)
    ny_local = unique_y.size
    row_from_y = {int(val): idx for idx, val in enumerate(unique_y.tolist())}

    columns_per_row: dict[int, list[int]] = {row: [] for row in range(ny_local)}
    for x_val, y_val in zip(x_idx, y_idx):
        row = row_from_y[int(y_val)]
        columns_per_row[row].append(int(x_val))

    column_lookup: dict[tuple[int, int], int] = {}
    nx_local = 0
    for row, xs in columns_per_row.items():
        xs_sorted = sorted(set(xs))
        columns_per_row[row] = xs_sorted
        nx_local = max(nx_local, len(xs_sorted))
        for col, x_val in enumerate(xs_sorted):
            column_lookup[(row, x_val)] = col

    grid = np.full((nens, nlev, ny_local, nx_local, nvar), fill_value, dtype=np.float64)
    x_global = np.full((ny_local, nx_local), -1, dtype=np.int32)
    for row, xs in columns_per_row.items():
        x_global[row, : len(xs)] = xs

    raw_perm = raw.transpose(2, 1, 0, 3)  # (member, level, point, variable)
    for p, (x_val, y_val) in enumerate(zip(x_idx, y_idx)):
        row = row_from_y[int(y_val)]
        col = column_lookup[(row, int(x_val))]
        grid[:, :, row, col, :] = raw_perm[:, :, p, :]

    mask = np.abs(grid - UNDEF) >= 1e30
    grid = np.where(mask, grid, fill_value)

    return grid, unique_y.astype(np.int32), x_global


def state3d_to_xarray(
    raw: np.ndarray,
    metadata: dict,
    *,
    level_coord: Optional[Iterable] = None,
    member_labels: Optional[Iterable] = None,
    variable_names: Optional[Iterable] = None,
    fill_value: float = np.nan,
    attrs: Optional[dict] = None,
) -> xr.DataArray:
    """
    Convert raw LETKF 3D state data to an xarray DataArray laid out on the rank’s tile.
    """
    grid, y_coords, x_global = reshape_state3d_raw(raw, metadata, fill_value=fill_value)
    nens, nlev, ny_local, nx_local, nvar = grid.shape

    if level_coord is None:
        level_coord = np.arange(nlev, dtype=np.int32)
    if member_labels is None:
        member_labels = [f"{i + 1:04d}" for i in range(nens)]
    if variable_names is None:
        variable_names = np.arange(nvar, dtype=np.int32)

    return xr.DataArray(
        grid,
        dims=("member", "level", "y", "x", "variable"),
        coords={
            "member": np.asarray(list(member_labels)),
            "level": np.asarray(list(level_coord)),
            "y": y_coords,
            "x": np.arange(nx_local, dtype=np.int32),
            "x_global": (("y", "x"), x_global),
            "variable": np.asarray(list(variable_names)),
        },
        attrs={} if attrs is None else dict(attrs),
    )


def load_state3d_as_xarray(base_dir: str | Path, prefix: str, rank_suffix: str) -> xr.DataArray:
    """
    Convenience wrapper: load a per-rank state dump and return it as an xarray DataArray.

    Parameters
    ----------
    base_dir
        Directory containing the LETKF dump (e.g. `.../letkf_dump`).
    prefix
        Either `"gues"` or `"anal"` to pick guess/analysis files.
    rank_suffix
        Rank identifier such as `"pe000010"`.
    """
    base_path = Path(base_dir)
    bundle = (
        load_guess_rank(base_path, rank_suffix)
        if prefix == "gues"
        else load_analysis_rank(base_path, rank_suffix)
    )
    raw = _read_fortran_binary(base_path / f"{prefix}3d_{rank_suffix}.bin")
    return state3d_to_xarray(
        raw,
        bundle.metadata,
        level_coord=bundle.dataset.coords["level"].values,
        member_labels=bundle.dataset.coords["member"].values,
        variable_names=bundle.dataset.coords["variable"].values,
        attrs=bundle.metadata,
    )


__all__ = [
    "UNDEF",
    "reshape_state3d_raw",
    "state3d_to_xarray",
    "load_state3d_as_xarray",
    "load_guess_rank",
    "load_analysis_rank",
    "load_obs_rank",
]
