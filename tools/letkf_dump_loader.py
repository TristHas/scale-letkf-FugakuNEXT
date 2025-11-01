from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import xarray as xr

from .letkf_parameters import get_state_variable_metadata


@dataclass(frozen=True)
class ObsdaBundle:
    """Observation-space data for one LETKF rank."""

    dataset: xr.Dataset

    @property
    def rank(self) -> str:
        return str(self.dataset.attrs.get("rank", ""))

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
class StateBundle:
    """State-space dump (guess or analysis) for one LETKF rank."""

    rank: str
    dataset: xr.Dataset
    metadata: Dict[str, int | str]

    @property
    def member_ids(self) -> tuple[str, ...]:
        if "member" in self.dataset.coords:
            return tuple(str(v) for v in self.dataset.coords["member"].values)
        return ()


@dataclass(frozen=True)
class LetkfDump:
    """Container for LETKF dump content."""

    guess: Dict[str, StateBundle]
    analysis: Dict[str, StateBundle]
    obsda: Dict[str, ObsdaBundle]
    dump_dir: Path

    @property
    def ranks(self) -> tuple[str, ...]:
        rank_ids = set(self.guess.keys()) | set(self.analysis.keys()) | set(self.obsda.keys())
        return tuple(sorted(rank_ids))

    @property
    def member_ids(self) -> tuple[str, ...]:
        for collection in (self.guess, self.analysis):
            for bundle in collection.values():
                return bundle.member_ids
        return ()


_I4_NATIVE = np.dtype("=i4")
_F8_NATIVE = np.dtype("=f8")
_F4_NATIVE = np.dtype("=f4")

_STATE_METADATA = get_state_variable_metadata()
_V3D_NAMES: list[str] = [str(v) for v in _STATE_METADATA.get("v3d_name", [])]
_V2D_NAMES: list[str] = [str(v) for v in _STATE_METADATA.get("v2d_name", [])]


def _read_fortran_binary(path: Path) -> np.ndarray:
    """Read array written by letkf_dump (stream binary with int32 header)."""
    with path.open("rb") as fh:
        header = fh.read(4)
        if len(header) == 0:
            raise ValueError(f"{path} is empty")
        if len(header) != 4:
            raise ValueError(f"{path} truncated while reading rank header")

        nd_le = int.from_bytes(header, byteorder="little", signed=False)
        nd_be = int.from_bytes(header, byteorder="big", signed=False)
        candidates: list[tuple[str, int]] = []
        if 1 <= nd_le <= 8:
            candidates.append(("<", nd_le))
        if 1 <= nd_be <= 8 and nd_be != nd_le:
            candidates.append((">", nd_be))
        if not candidates:
            raise ValueError(f"{path} reports invalid rank header values {nd_le}/{nd_be}")

        endian, nd = candidates[0]

        dim_bytes = nd * 4
        dims_raw = fh.read(dim_bytes)
        int_fmt = "i"
        if len(dims_raw) != dim_bytes:
            fh.seek(4)
            dim_bytes = nd * 8
            dims_raw = fh.read(dim_bytes)
            if len(dims_raw) != dim_bytes:
                raise ValueError(f"{path} truncated while reading dimension array")
            int_fmt = "q"

        dims = struct.unpack(f"{endian}{nd}{int_fmt}", dims_raw)
        dims = tuple(int(v) for v in dims)
        if any(d <= 0 for d in dims):
            raise ValueError(f"{path} contains non-positive dimensions {dims}")

        payload = fh.read()
        total = int(np.prod(dims, dtype=np.int64))
        payload_len = len(payload)
        if total <= 0:
            raise ValueError(f"{path} reports zero element count")
        if payload_len <= 0:
            raise ValueError(f"{path} payload is empty")

        value_count = None
        itemsize = None

        # Try double precision payload first.
        if payload_len % _F8_NATIVE.itemsize == 0:
            count = payload_len // _F8_NATIVE.itemsize
            if count >= total:
                value_count = count
                itemsize = _F8_NATIVE.itemsize

        # Fallback to single precision.
        if itemsize is None and payload_len % _F4_NATIVE.itemsize == 0:
            count = payload_len // _F4_NATIVE.itemsize
            if count >= total:
                value_count = count
                itemsize = _F4_NATIVE.itemsize

        if itemsize is None:
            raise ValueError(
                f"{path} payload size {payload_len} is not compatible with shape {dims}"
            )

        prod_rest = int(np.prod(dims[1:], dtype=np.int64)) if nd > 1 else 1
        if value_count > total and prod_rest > 0:
            extra = value_count - total
            if extra % prod_rest == 0:
                payload = payload[: total * itemsize]
                value_count = total

        if value_count != total:
            raise ValueError(
                f"{path} unexpected payload length ({value_count} values, expected {total})"
            )

        dtype = np.dtype(f"{endian}f8") if itemsize == _F8_NATIVE.itemsize else np.dtype(
            f"{endian}f4"
        )
        data = np.frombuffer(payload, dtype=dtype, count=total)

    return np.asarray(data, dtype=np.float64).reshape(dims, order="F")


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


def _read_state_metadata(path: Path) -> Dict[str, int | str]:
    if not path.exists():
        raise FileNotFoundError(f"State metadata file {path} does not exist")
    metadata: Dict[str, int | str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key == "domain_suffix":
                metadata[key] = value
                continue
            try:
                metadata[key] = int(value)
            except ValueError:
                metadata[key] = value
    return metadata


def _member_labels(count: int) -> list[str]:
    width = max(4, len(str(count)))
    return [f"{idx:0{width}d}" for idx in range(1, count + 1)]


def _normalise_rank(rank: str | int) -> str:
    if isinstance(rank, int):
        if rank < 0:
            raise ValueError("Rank index must be non-negative")
        return f"pe{rank:06d}"
    rank_str = str(rank).strip()
    if not rank_str:
        raise ValueError("Rank identifier must be non-empty")
    if rank_str.startswith("pe") and len(rank_str) == 8 and rank_str[2:].isdigit():
        return rank_str
    if rank_str.isdigit():
        return f"pe{int(rank_str):06d}"
    raise ValueError(f"Unsupported rank identifier {rank!r}")


def _compute_point_indices(meta: Dict[str, int | str], nij1: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    try:
        isize = int(meta.get("tile_i_size", -1))
        jsize = int(meta.get("tile_j_size", -1))
        istart = int(meta.get("tile_i_start", -1))
        jstart = int(meta.get("tile_j_start", -1))
    except (TypeError, ValueError):
        return None, None

    if min(isize, jsize, istart, jstart) > 0:
        if isize * jsize == nij1:
            xs = np.empty(nij1, dtype=np.int32)
            ys = np.empty(nij1, dtype=np.int32)
            idx = 0
            for j in range(jsize):
                for i in range(isize):
                    xs[idx] = istart + i
                    ys[idx] = jstart + j
                    idx += 1
            return xs, ys

    nlon = int(meta.get("nlon", -1))
    nlat = int(meta.get("nlat", -1))
    if nlon > 0 and nlat > 0:
        xs = np.empty(nij1, dtype=np.int32)
        ys = np.empty(nij1, dtype=np.int32)
        base = int(meta.get("domain_offset", 0))
        for idx in range(nij1):
            linear = base + idx
            xs[idx] = (linear % nlon) + 1
            ys[idx] = (linear // nlon) + 1
        return xs, ys

    return None, None

    xs = np.empty(nij1, dtype=np.int32)
    ys = np.empty(nij1, dtype=np.int32)
    idx = 0
    for j in range(jsize):
        for i in range(isize):
            xs[idx] = istart + i
            ys[idx] = jstart + j
            idx += 1
    return xs, ys


def _build_state_dataset(rank: str, meta: Dict[str, int | str], raw3d: np.ndarray, raw2d: np.ndarray | None) -> xr.Dataset:
    if raw3d.ndim != 4:
        raise ValueError(f"{rank}: expected 4-D array for state_3d, got shape {raw3d.shape}")

    nij1, nlev, nens, nv3d = raw3d.shape
    member_labels = _member_labels(nens)

    state3d = np.transpose(raw3d, (2, 1, 0, 3))  # member, level, point, variable

    coords: Dict[str, Sequence[int] | Sequence[str] | np.ndarray] = {
        "member": member_labels,
        "level": np.arange(nlev, dtype=np.int32),
        "point": np.arange(nij1, dtype=np.int32),
        "variable": _V3D_NAMES if _V3D_NAMES else np.arange(nv3d, dtype=np.int32),
    }

    data_vars = {
        "state_3d": (("member", "level", "point", "variable"), state3d),
    }

    if raw2d is not None:
        if raw2d.ndim != 3:
            raise ValueError(f"{rank}: expected 3-D array for state_2d, got shape {raw2d.shape}")
        if raw2d.shape[0] != nij1 or raw2d.shape[1] != nens:
            raise ValueError(
                f"{rank}: state_2d dimensions {raw2d.shape} do not match state_3d {raw3d.shape}"
            )
        nv2d = raw2d.shape[2]
        state2d = np.transpose(raw2d, (1, 0, 2))  # member, point, variable_2d
        coords["variable_2d"] = _V2D_NAMES if _V2D_NAMES else np.arange(nv2d, dtype=np.int32)
        data_vars["state_2d"] = (("member", "point", "variable_2d"), state2d)

    dataset = xr.Dataset(data_vars, coords={key: (key, np.asarray(values)) for key, values in coords.items()})

    xs, ys = _compute_point_indices(meta, nij1)
    if xs is not None and ys is not None:
        dataset = dataset.assign_coords(
            x_index=("point", xs.astype(np.int32)),
            y_index=("point", ys.astype(np.int32)),
        )

    dataset.attrs["rank"] = rank
    for key, value in meta.items():
        dataset.attrs[key] = value

    return dataset


def _load_state_rank(base_dir: Path, prefix: str, suffix: str) -> StateBundle:
    file3d = base_dir / f"{prefix}3d_{suffix}.bin"
    if not file3d.exists():
        raise FileNotFoundError(f"State dump {file3d} does not exist")
    meta_path = base_dir / f"state_meta_{suffix}.txt"
    metadata = _read_state_metadata(meta_path)
    raw3d = _read_fortran_binary(file3d)
    file2d = base_dir / f"{prefix}2d_{suffix}.bin"
    raw2d = _read_fortran_binary(file2d) if file2d.exists() else None
    dataset = _build_state_dataset(suffix, metadata, raw3d, raw2d)
    return StateBundle(rank=suffix, dataset=dataset, metadata=metadata)


def _load_state_bundles(base_dir: Path, prefix: str) -> Dict[str, StateBundle]:
    bundles: Dict[str, StateBundle] = {}
    files3d = sorted(base_dir.glob(f"{prefix}3d_*.bin"))
    for file3d in files3d:
        suffix = file3d.stem.split("_", 1)[1]
        bundles[suffix] = _load_state_rank(base_dir, prefix, suffix)
    return bundles


def _load_single_obsda(dump_path: Path, suffix: str) -> ObsdaBundle | None:
    obsda_dir = dump_path / "obsda"
    mean_file = obsda_dir / f"obsda_mean_{suffix}.dat"
    if not mean_file.exists():
        return None

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

    obsda_dir_files = sorted(obsda_dir.glob(f"obsda_????_{suffix}.dat"))
    if obsda_dir_files:
        member_ids = []
        anomalies = []
        for member_file in obsda_dir_files:
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
    dataset.attrs["rank"] = suffix
    return ObsdaBundle(dataset=dataset)


def _load_obsda(dump_path: Path) -> Dict[str, ObsdaBundle]:
    obsda_dir = dump_path / "obsda"
    bundles: Dict[str, ObsdaBundle] = {}
    if not obsda_dir.exists():
        return bundles

    for meta_file in sorted(obsda_dir.glob("obsda_meta_*.txt")):
        suffix = meta_file.stem.replace("obsda_meta_", "")
        bundle = _load_single_obsda(dump_path, suffix)
        if bundle is not None:
            bundles[suffix] = bundle
    return bundles


def load_letkf_dump(dump_dir: str | Path) -> LetkfDump:
    """Load ensemble state and observation-space dumps produced by letkf_dump."""
    dump_path = Path(dump_dir).expanduser().resolve()
    if not dump_path.exists():
        raise FileNotFoundError(f"Dump directory {dump_path} does not exist")

    guess = _load_state_bundles(dump_path, prefix="gues")
    analysis = _load_state_bundles(dump_path, prefix="anal")
    obsda = _load_obsda(dump_path)

    return LetkfDump(
        guess=guess,
        analysis=analysis,
        obsda=obsda,
        dump_dir=dump_path,
    )


def load_guess_rank(dump_dir: str | Path, rank: str | int) -> StateBundle:
    """Load guess state dump for a single rank."""
    suffix = _normalise_rank(rank)
    base_dir = Path(dump_dir).expanduser().resolve()
    return _load_state_rank(base_dir, "gues", suffix)


def load_analysis_rank(dump_dir: str | Path, rank: str | int) -> StateBundle:
    """Load analysis state dump for a single rank."""
    suffix = _normalise_rank(rank)
    base_dir = Path(dump_dir).expanduser().resolve()
    return _load_state_rank(base_dir, "anal", suffix)


def load_obs_rank(dump_dir: str | Path, rank: str | int) -> ObsdaBundle:
    """Load observation-space dump for a single rank."""
    suffix = _normalise_rank(rank)
    base_dir = Path(dump_dir).expanduser().resolve()
    bundle = _load_single_obsda(base_dir, suffix)
    if bundle is None:
        raise FileNotFoundError(f"No observation dump found for rank {suffix}")
    return bundle


__all__ = [
    "ObsdaBundle",
    "StateBundle",
    "LetkfDump",
    "load_analysis_rank",
    "load_guess_rank",
    "load_letkf_dump",
    "load_obs_rank",
]
