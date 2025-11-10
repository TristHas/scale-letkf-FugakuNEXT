"""High-level helpers for reconstructing das_letkf inputs from global dumps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Tuple

import numpy as np
import xarray as xr

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Tuple

import numpy as np

from .load_das_letkf import load_das_obs_local_before
from .load_dumps import load_grid_info, Z_LEVELS, _read_binary_array
from .params import GRID_CONSTANTS, LETKF_CONSTANTS

VAR_NAMES = ["U", "V", "W", "T", "P", "QV", "QC", "QR", "QI", "QS", "QG"]
VAR_INDEX = {name: idx for idx, name in enumerate(VAR_NAMES)}
MEMBER = int(LETKF_CONSTANTS["MEMBER"])
IHALO = 2
JHALO = 2


@dataclass(frozen=True)
class GridLocation:
    ij: int
    ilev: int
    nvar_global: int
    ri: float
    rj: float
    rlev: float
    rz: float


def _call_id_to_indices(call_id: int, nij1: int, nlev: int, var_count: int) -> Tuple[int, int, int]:
    call0 = call_id - 1
    grid_idx, nvar0 = divmod(call0, var_count)
    ilev_idx, ij0 = divmod(grid_idx, nij1)
    return ij0 + 1, ilev_idx + 1, nvar0 + 1


def derive_obs_local_before(dump_dir: Path | str, call_id: int, pe_tag: str, member: str = "mem0001") -> GridLocation:
    dump_path = Path(dump_dir)
    rig1, rjg1, topo1, _ = load_grid_info(dump_path, pe_tag)
    nij1 = rig1.size
    nlev = int(GRID_CONSTANTS["nlev"])
    nv3d = int(GRID_CONSTANTS["nv3d"])
    nv2d = int(GRID_CONSTANTS.get("nv2d", 0))
    ij, ilev, nvar_global = _call_id_to_indices(call_id, nij1, nlev, nv3d + nv2d)

    ri = float(rig1[ij - 1])
    rj = float(rjg1[ij - 1])
    rlev = float(_state_pressure_mean(dump_path, pe_tag, ij - 1, ilev - 1))
    if topo1 is not None:
        topo_val = topo1[ij - 1]
    else:
        topo_val = _sample_topography(dump_path, pe_tag, ri, rj)
    ztop = Z_LEVELS[-1]
    cz = Z_LEVELS[ilev - 1]
    rz = float(((ztop - topo_val) / ztop) * cz + topo_val)

    return GridLocation(
        ij=ij,
        ilev=ilev,
        nvar_global=nvar_global,
        ri=ri,
        rj=rj,
        rlev=rlev,
        rz=rz,
    )


def test_obs_local_before_global(
    dump_dir: Path | str,
    call_id: int,
    pe_tag: str,
    member: str = "mem0001",
) -> dict[str, float]:
    derived = derive_obs_local_before(dump_dir, call_id, pe_tag, member)
    stage = load_das_obs_local_before(dump_dir, call_id, pe_tag=pe_tag, member=member)
    meta = stage["meta"]
    errors = {
        "ri": abs(derived.ri - float(meta["ri"])),
        "rj": abs(derived.rj - float(meta["rj"])),
        "rlev": abs(derived.rlev - float(meta["rlev"])),
        "rz": abs(derived.rz - float(meta["rz"])),
    }
    tolerances = {"ri": 1e-6, "rj": 1e-6, "rlev": 1e-6, "rz": 5e-2}
    for key, tol in tolerances.items():
        if errors[key] > tol:
            raise AssertionError(f"{key} mismatch: error={errors[key]} > {tol}")
    return errors


@lru_cache(maxsize=None)
def _load_state_cube(dump_dir: Path, prefix: str, pe_tag: str) -> np.ndarray:
    pe_norm = _normalize_pe_tag(pe_tag)
    path = dump_dir / prefix / f"{prefix}_{pe_norm}.mem0001.bin"
    if not path.exists():
        raise FileNotFoundError(path)
    return _read_binary_array(path, ">f8")


def _state_pressure_mean(dump_dir: Path, pe_tag: str, ij_idx: int, ilev_idx: int) -> float:
    cube = _load_state_cube(dump_dir, "gues3d", pe_tag)
    values = cube[ij_idx, ilev_idx, :MEMBER, VAR_INDEX["P"]]
    return float(np.mean(values, dtype=np.float64))


def _normalize_pe_tag(pe_tag: str) -> str:
    token = str(pe_tag).lower()
    if token.startswith("pe"):
        token = token[2:]
    return f"pe{int(token):06d}"


@lru_cache(maxsize=None)
def _load_topography_array(dump_dir: Path, pe_tag: str) -> np.ndarray:
    base = dump_dir.parents[1] / "const" / "topo"
    path = base / f"topo.{_normalize_pe_tag(pe_tag)}.nc"
    ds = xr.open_dataset(path)
    arr = ds["topo"].values.astype(np.float64)
    ds.close()
    return arr


def _sample_topography(dump_dir: Path, pe_tag: str, ri: float, rj: float) -> float:
    topo = _load_topography_array(dump_dir, pe_tag)
    x_idx = int(round(ri)) - IHALO - 1 + IHALO
    y_idx = int(round(rj)) - JHALO - 1 + JHALO
    return float(topo[y_idx, x_idx])


__all__ = [
    "GridLocation",
    "derive_obs_local_before",
    "test_obs_local_before_global",
]
