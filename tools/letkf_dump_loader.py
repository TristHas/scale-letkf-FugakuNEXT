from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import xarray as xr

from .letkf_parameters import get_state_variable_metadata


@dataclass(frozen=True)
class ObsdaBundle:
    """Observation-space data for one LETKF subdomain."""

    dataset: xr.Dataset

    @property
    def observation_count(self) -> int:
        return int(self.dataset.sizes.get("observation", 0))

    @property
    def member_ids(self) -> tuple[str, ...]:
        if "member" in self.dataset.coords:
            return tuple(str(v) for v in self.dataset.coords["member"].values)
        return ()

    @property
    def innovations(self) -> xr.DataArray:
        return self.dataset["innovation"]


@dataclass(frozen=True)
class LetkfDump:
    """Container for LETKF dump content."""

    guess: xr.Dataset
    analysis: xr.Dataset
    obsda: Dict[str, ObsdaBundle]
    dump_dir: Path

    @property
    def member_ids(self) -> tuple[str, ...]:
        for dataset in (self.guess, self.analysis):
            if "member" in dataset.coords:
                return tuple(str(v) for v in dataset.coords["member"].values)
        return ()


_F4_NATIVE = np.dtype("=f4")

_STATE_METADATA = get_state_variable_metadata()
_V3D_NAMES: list[str] = [str(v) for v in _STATE_METADATA.get("v3d_name", [])]
_V2D_NAMES: list[str] = [str(v) for v in _STATE_METADATA.get("v2d_name", [])]


def _read_obsda_file(path: Path) -> np.ndarray:
    """Read obsda_(mean|member) files written via sequential unformatted I/O."""
    records = []
    with path.open("rb") as fh:
        endian = None
        expected_len = _F4_NATIVE.itemsize * 4
        while True:
            header = fh.read(4)
            if not header:
                break
            if len(header) != 4:
                raise ValueError(f"{path} has incomplete record header")
            len_le = int.from_bytes(header, byteorder="little")
            len_be = int.from_bytes(header, byteorder="big")

            if endian is None:
                if len_le == expected_len and len_be != expected_len:
                    endian = "<"
                elif len_be == expected_len and len_le != expected_len:
                    endian = ">"
                else:
                    candidates = [
                        (ord_, length)
                        for ord_, length in (("<", len_le), (">", len_be))
                        if 0 < length <= 1_000_000
                    ]
                    if len(candidates) == 1:
                        endian = candidates[0][0]
                    else:
                        raise ValueError(
                            f"{path} record header has unexpected lengths {len_le} / {len_be}"
                        )

            record_len = len_le if endian == "<" else len_be
            if record_len != expected_len:
                raise ValueError(
                    f"{path} record size {record_len} does not match expected {expected_len}"
                )

            payload = fh.read(record_len)
            if len(payload) != record_len:
                raise ValueError(f"{path} truncated while reading payload")
            payload_dtype = np.dtype(f"{endian}f4")
            records.append(np.frombuffer(payload, dtype=payload_dtype).astype(_F4_NATIVE))
            trailer = fh.read(4)
            if len(trailer) != 4:
                raise ValueError(f"{path} has incomplete record trailer")
            trailer_len = int.from_bytes(trailer, byteorder="little" if endian == "<" else "big")
            if trailer_len != record_len:
                raise ValueError(f"{path} has inconsistent record trailer")
    if not records:
        return np.empty((0, 4), dtype=_F4_NATIVE)
    return np.vstack([rec.reshape(1, 4) for rec in records])


def _stack_state_3d(
    member_ids: Sequence[str],
    arrays: Sequence[np.ndarray],
    name: str,
    var_names: Sequence[str] | None = None,
) -> xr.DataArray:
    data = np.stack([np.swapaxes(arr, 1, 2) for arr in arrays], axis=0)
    levels, ny, nx, nvar = data.shape[1:]
    if var_names and len(var_names) == nvar:
        var_labels = list(var_names)
    else:
        var_labels = _V3D_NAMES if _V3D_NAMES else list(range(nvar))
    coords = {
        "member": list(member_ids),
        "level": np.arange(levels, dtype=np.int32),
        "y": np.arange(ny, dtype=np.int32),
        "x": np.arange(nx, dtype=np.int32),
        "variable": var_labels,
    }
    return xr.DataArray(
        data,
        dims=("member", "level", "y", "x", "variable"),
        coords=coords,
        name=name,
    )


def _stack_state_2d(
    member_ids: Sequence[str],
    arrays: Sequence[np.ndarray],
    name: str,
    var_names: Sequence[str] | None = None,
) -> xr.DataArray:
    data = np.stack([arr.swapaxes(0, 1) for arr in arrays], axis=0)
    ny, nx, nvar = data.shape[1:]
    if var_names and len(var_names) == nvar:
        labels = list(var_names)
    else:
        labels = _V2D_NAMES if _V2D_NAMES else list(range(nvar))
    coords = {
        "member": list(member_ids),
        "y": np.arange(ny, dtype=np.int32),
        "x": np.arange(nx, dtype=np.int32),
        "variable_2d": labels,
    }
    return xr.DataArray(
        data,
        dims=("member", "y", "x", "variable_2d"),
        coords=coords,
        name=name,
    )


def _select_state_file(member_dir: Path, base_name: str) -> Path:
    aggregated = member_dir / f"{base_name}.nc"
    if aggregated.exists():
        return aggregated
    tiles = sorted(member_dir.glob(f"{base_name}.pe*.nc"))
    if not tiles:
        raise FileNotFoundError(f"No restart files found in {member_dir}")
    if len(tiles) > 1:
        raise NotImplementedError(
            f"Multiple tiles found for {member_dir.name}; enable FILE_AGGREGATE to write a single file."
        )
    return tiles[0]


def _prepare_variable(array: xr.DataArray, expected_ndim: int, path: Path, name: str) -> np.ndarray:
    data = np.asarray(array.values, dtype=np.float64)
    if data.ndim == expected_ndim + 1 and data.shape[-1] == 1:
        data = np.squeeze(data, axis=-1)
    if data.ndim != expected_ndim:
        raise ValueError(f"{path} variable {name} has unexpected shape {data.shape}")
    return data


def _read_member_state(member_dir: Path, prefix: str) -> tuple[np.ndarray, np.ndarray | None, list[str], list[str]]:
    mem_tag = member_dir.name
    base_name = f"{prefix}.{mem_tag}"
    path = _select_state_file(member_dir, base_name)

    arrays3d: list[np.ndarray] = []
    arrays2d: list[np.ndarray] = []
    names3d: list[str] = []
    names2d: list[str] = []

    with xr.open_dataset(path) as ds:
        for name in _V3D_NAMES:
            if name not in ds:
                continue
            arrays3d.append(_prepare_variable(ds[name], 3, path, name))
            names3d.append(name)
        if not arrays3d:
            raise ValueError(f"{path} does not contain known 3D state variables")

        for name in _V2D_NAMES:
            if name not in ds:
                continue
            arrays2d.append(_prepare_variable(ds[name], 2, path, name))
            names2d.append(name)

    arr3d = np.stack(arrays3d, axis=-1)
    arr2d = np.stack(arrays2d, axis=-1) if arrays2d else None
    return arr3d, arr2d, names3d, names2d


def _load_state_dataset(base_dir: Path, prefix: str) -> xr.Dataset:
    prefix_dir = base_dir / prefix
    if not prefix_dir.exists():
        return xr.Dataset()

    member_dirs = sorted(p for p in prefix_dir.iterdir() if p.is_dir())
    if not member_dirs:
        return xr.Dataset()

    member_ids: list[str] = []
    arrays3d: list[np.ndarray] = []
    arrays2d: list[np.ndarray | None] = []
    names3d_ref: list[str] | None = None
    names2d_ref: list[str] | None = None
    has_2d = False

    for member_dir in member_dirs:
        arr3d, arr2d, names3d, names2d = _read_member_state(member_dir, prefix)
        if names3d_ref is None:
            names3d_ref = names3d
        elif names3d_ref != names3d:
            raise ValueError(f"Inconsistent 3D variable set for member {member_dir.name}")
        if names2d_ref is None:
            names2d_ref = names2d
        elif names2d_ref != names2d:
            raise ValueError(f"Inconsistent 2D variable set for member {member_dir.name}")

        member_ids.append(member_dir.name)
        arrays3d.append(arr3d)
        arrays2d.append(arr2d)
        if arr2d is not None:
            has_2d = True

    data_vars = {
        "state_3d": _stack_state_3d(member_ids, arrays3d, f"{prefix}_state_3d", names3d_ref or _V3D_NAMES),
    }

    if has_2d:
        if not all(arr is not None for arr in arrays2d):
            missing = [mem for mem, arr in zip(member_ids, arrays2d) if arr is None]
            raise ValueError(f"2D variables missing for members: {', '.join(missing)}")
        data_vars["state_2d"] = _stack_state_2d(
            member_ids,
            [arr for arr in arrays2d if arr is not None],
            f"{prefix}_state_2d",
            names2d_ref or _V2D_NAMES,
        )

    return xr.Dataset(data_vars)


def _load_obsda(dump_path: Path) -> Dict[str, ObsdaBundle]:
    obsda_dir = dump_path / "obsda"
    bundles: Dict[str, ObsdaBundle] = {}
    if not obsda_dir.exists():
        return bundles

    for meta in sorted(obsda_dir.glob("obsda_meta_*.txt")):
        suffix = meta.stem.replace("obsda_meta_", "")
        mean_file = obsda_dir / f"obsda_mean_{suffix}.dat"
        if not mean_file.exists():
            continue
        mean_raw = _read_obsda_file(mean_file)
        set_ids = mean_raw[:, 0].astype(np.int32)
        idx_ids = mean_raw[:, 1].astype(np.int32)
        innov = mean_raw[:, 2].astype(np.float64)
        qc = mean_raw[:, 3].astype(np.int32)
        n_obs = innov.shape[0]
        coords = {
            "observation": np.arange(n_obs, dtype=np.int32),
        }
        data_vars = {
            "set_id": ("observation", set_ids),
            "index_id": ("observation", idx_ids),
            "innovation": ("observation", innov),
            "qc_flag": ("observation", qc),
        }

        member_files = sorted(obsda_dir.glob(f"obsda_????_{suffix}.dat"))
        if member_files:
            member_ids = []
            anomalies = []
            for member_file in member_files:
                mem_tag = member_file.stem.split("_")[1]
                member_ids.append(str(mem_tag))
                member_raw = _read_obsda_file(member_file)
                if member_raw.shape[0] != n_obs:
                    raise ValueError(
                        f"Observation count mismatch between {mean_file.name} and {member_file.name}"
                    )
                anomalies.append(member_raw[:, 2].astype(np.float64))
            hx_anom = np.stack(anomalies, axis=0)
            coords["member"] = member_ids
            data_vars["hx_anomaly"] = (("member", "observation"), hx_anom)

        dataset = xr.Dataset(data_vars, coords=coords)
        bundles[suffix] = ObsdaBundle(dataset=dataset)
    return bundles


def load_letkf_dump(dump_dir: str | Path) -> LetkfDump:
    """Load ensemble state and observation-space dumps produced by letkf_dump."""
    dump_path = Path(dump_dir).expanduser().resolve()
    if not dump_path.exists():
        raise FileNotFoundError(f"Dump directory {dump_path} does not exist")

    guess = _load_state_dataset(dump_path, prefix="gues")
    analysis = _load_state_dataset(dump_path, prefix="anal")
    obsda = _load_obsda(dump_path)

    return LetkfDump(
        guess=guess,
        analysis=analysis,
        obsda=obsda,
        dump_dir=dump_path,
    )


__all__ = ["ObsdaBundle", "LetkfDump", "load_letkf_dump"]
