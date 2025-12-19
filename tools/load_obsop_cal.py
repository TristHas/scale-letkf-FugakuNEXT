from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

_BIN_SUFFIX = ".bin"


def _read_binary_array(path: Path) -> np.ndarray:
    with path.open("rb") as fh:
        nd_raw = np.fromfile(fh, dtype=">i4", count=1)
        if nd_raw.size == 0:
            raise ValueError(f"{path} is empty")
        nd = int(nd_raw[0])
        dims = np.fromfile(fh, dtype=">i4", count=nd)
        if dims.size != nd:
            raise ValueError(f"{path}: expected {nd} dims, found {dims.size}")
        size = int(np.prod(dims, dtype=np.int64))
        data = np.fromfile(fh, dtype=">f8", count=size)
        if data.size != size:
            raise ValueError(f"{path}: expected {size} values, found {data.size}")
    return data.reshape(tuple(int(d) for d in dims), order="F")


def _read_integer_array(path: Path) -> np.ndarray:
    with path.open("rb") as fh:
        nd_raw = np.fromfile(fh, dtype=">i4", count=1)
        if nd_raw.size == 0:
            raise ValueError(f"{path} is empty")
        nd = int(nd_raw[0])
        dims = np.fromfile(fh, dtype=">i4", count=nd)
        if dims.size != nd:
            raise ValueError(f"{path}: expected {nd} dims, found {dims.size}")
        size = int(np.prod(dims, dtype=np.int64))
        data = np.fromfile(fh, dtype=">i4", count=size)
        if data.size != size:
            raise ValueError(f"{path}: expected {size} values, found {data.size}")
    return data.reshape(tuple(int(d) for d in dims), order="F")


def _read_stage_metadata(path: Path) -> Dict[str, int]:
    meta: Dict[str, int] = {}
    if not path.exists():
        return meta
    with path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key:
                try:
                    meta[key] = int(value)
                except ValueError:
                    pass
    return meta


def _stage_dir(base_dir: Path, section: str, domain: str, member: str, stage: str) -> Path:
    return Path(base_dir) / section / domain / member / stage


def list_domains(base_dir: Path, section: str = "state") -> List[str]:
    section_dir = Path(base_dir) / section
    if not section_dir.exists():
        return []
    return sorted(p.name for p in section_dir.iterdir() if p.is_dir())


def list_members(base_dir: Path, domain: str, section: str = "state") -> List[str]:
    dom_dir = Path(base_dir) / section / domain
    if not dom_dir.exists():
        return []
    return sorted(p.name for p in dom_dir.iterdir() if p.is_dir())


def list_stages(base_dir: Path, domain: str, member: str, section: str = "state") -> List[str]:
    mem_dir = Path(base_dir) / section / domain / member
    if not mem_dir.exists():
        return []
    return sorted(p.name for p in mem_dir.iterdir() if p.is_dir())


def load_obsop_state(
    base_dir: Path,
    domain: str,
    member: str,
    stage: str,
) -> Dict[str, np.ndarray]:
    stage_dir = _stage_dir(base_dir, "state", domain, member, stage)
    data: Dict[str, np.ndarray] = {}
    for key in ("v3dg", "v2dg", "mv3dg", "slope3dg"):
        path = stage_dir / f"{key}{_BIN_SUFFIX}"
        if path.exists():
            if key in ("v3dg", "v2dg", "mv3dg"):
                data[key] = _read_binary_array(path)
            else:
                data[key] = _read_binary_array(path)
    data["stage_meta"] = _read_stage_metadata(stage_dir / "stage_meta.txt")
    return data


def load_obsop_observations(
    base_dir: Path,
    domain: str,
    member: str,
    stage: str,
) -> Dict[str, np.ndarray]:
    stage_dir = _stage_dir(base_dir, "obs", domain, member, stage)
    if not stage_dir.exists():
        raise FileNotFoundError(stage_dir)

    int_fields = ["set", "idx", "nn", "elm", "typ", "qc"]
    real_fields = [
        "lon",
        "lat",
        "lev",
        "ri_global",
        "rj_global",
        "ril",
        "rjl",
        "rkz",
    ]

    data: Dict[str, np.ndarray] = {}
    for field in int_fields:
        path = stage_dir / f"{field}{_BIN_SUFFIX}"
        if path.exists():
            data[field] = _read_integer_array(path).reshape(-1)
        else:
            data[field] = np.empty(0, dtype=np.int32)
    for field in real_fields:
        path = stage_dir / f"{field}{_BIN_SUFFIX}"
        if path.exists():
            data[field] = _read_binary_array(path).reshape(-1)
        else:
            data[field] = np.empty(0, dtype=np.float64)

    meta_path = stage_dir / f"meta{_BIN_SUFFIX}"
    if meta_path.exists():
        data["meta"] = _read_binary_array(meta_path)
    else:
        data["meta"] = np.empty((0, 0))

    data["stage_meta"] = _read_stage_metadata(stage_dir / "stage_meta.txt")
    return data
