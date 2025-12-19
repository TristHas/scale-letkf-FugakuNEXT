from pathlib import Path
import re
from typing import Dict, List
import numpy as np
import xarray as xr

from .convert_scale_letkf_var import convert_scale_letkf_var

_UNDEF = -9.99e33  # SCALE missing sentinel
Z_LEVELS = [   55.     ,   165.     ,   275.     ,   385.     ,   495.     ,
             608.08   ,   727.49245,   853.592  ,   986.75315,  1127.37135,
            1275.86415,  1432.67255,  1598.2623 ,  1773.1251 ,  1957.78025,
            2152.77615,  2358.6919 ,  2576.13905,  2805.7633 ,  3048.24655,
            3304.30895,  3574.7108 ,  3860.2551 ,  4161.7898 ,  4480.2102 ,
            4816.46215,  5171.5442 ,  5546.51075,  5942.47535,  6360.61375,
            6802.16795,  7268.4492 ,  7760.842  ,  8280.80855,  8829.89305,
            9409.7261 , 10022.0298 , 10668.62255, 11351.4243 , 12072.4629 ,
           12842.8018 , 13642.8018 , 14442.8018 , 15242.8018 , 16042.8013 ]

x_offset = {"pe" + str(i).zfill(6) : 50 + (100  * (i % 4)) * 320 for i in range(20)}
y_offset = {"pe" + str(i).zfill(6) : 50 + (100  * (i // 4)) * 256 for i in range(20)}

X_LEVELS =  {k: v+np.arange(320)*100 for k,v in x_offset.items()}
Y_LEVELS =  {k: v+np.arange(256)*100 for k,v in y_offset.items()}
_OBSDA_FILE_RE = re.compile(r"obsda_(?P<var>.+)_(?P<pe>pe[0-9a-z]+)\.(?P<member>[^.]+)\.bin$", re.IGNORECASE)
_OBS_RAW_FILE_RE = re.compile(
    r"(?P<prefix>obs\d{4})_(?P<field>[a-z0-9]+)_(?P<pe>pe[0-9a-z]+)\.(?P<member>[^.]+)\.bin$",
    re.IGNORECASE,
)
_OBSGRD_FILE_RE = re.compile(
    r"(?P<base>type\d{4})_(?P<field>[a-z_]+)_(?P<pe>pe[0-9a-z]+)\.(?P<member>[^.]+)\.bin$",
    re.IGNORECASE,
)
_SPECIAL_MEMBERS = {
    "mean": "memmean",
    "memmean": "memmean",
    "mdet": "memmdet",
    "memmdet": "memmdet",
    "mgue": "memmgue",
    "memmgue": "memmgue",
    "sprd": "memsprd",
    "memsprd": "memsprd",
}
_OBS_RAW_INT_FIELDS = {"elm", "typ", "rank"}
_OBSGRD_INT_ARRAYS = {
    "n": ">i4",
    "ac": ">i4",
    "tot": ">i4",
    "n_ext": ">i4",
    "ac_ext": ">i4",
    "tot_sub": ">i4",
    "tot_g": ">i4",
}
STATE_META_SUBDIR = "state_meta"

def load_and_convert_rank_members(
    dump_dir: str | Path,
    prefix: str,
    pe_tag: str,
) -> xr.DataArray:
    """Load all ensemble members for a rank and attach 1-D lon/lat vectors."""

    dump_dir = Path(dump_dir)
    pe_norm = _normalize_pe_tag(pe_tag)
    lat_vec: np.ndarray | None = None
    lon_vec: np.ndarray | None = None
    members = ["0001", "0002", "mean"]
    datasets: List[xr.DataArray] = []
    for member in members:
        fname = f"init_20210730-060030.000.{pe_norm}.nc"
        nc_path = dump_dir / ".." / "anal_f" / member / fname
        ds = xr.open_dataset(nc_path, engine="netcdf4")
        if lat_vec is None and "lat" in ds and "lon" in ds:
            lat2d = ds["lat"].astype(np.float64)
            lon2d = ds["lon"].astype(np.float64)
            lat_vec = lat2d.mean(dim="x").values
            lon_vec = lon2d.mean(dim="y").values
        da_comp = convert_scale_letkf_var(ds)
        ds.close()
        datasets.append(da_comp.expand_dims(ens=[member]))
    stacked = xr.concat(datasets, dim="ens").transpose("y", "x", "z", "ens", "variable")
    if lat_vec is not None:
        if lat_vec.shape[0] == stacked.sizes["y"] + 2:
            lat_vec = lat_vec[1:-1]
        elif lat_vec.shape[0] == stacked.sizes["y"] + 4:
            lat_vec = lat_vec[2:-2]
    if lon_vec is not None:
        if lon_vec.shape[0] == stacked.sizes["x"] + 2:
            lon_vec = lon_vec[1:-1]
        elif lon_vec.shape[0] == stacked.sizes["x"] + 4:
            lon_vec = lon_vec[2:-2]

    if lat_vec is not None and lat_vec.shape[0] == stacked.sizes["y"]:
        stacked = stacked.assign_coords(lat=("y", lat_vec))
    if lon_vec is not None and lon_vec.shape[0] == stacked.sizes["x"]:
        stacked = stacked.assign_coords(lon=("x", lon_vec))
    return stacked

def _read_binary_array(path: Path, dtype: str) -> np.ndarray:
    with path.open("rb") as fh:
        nd = np.fromfile(fh, dtype=">i4", count=1)[0]
        dims = tuple(np.fromfile(fh, dtype=">i4", count=nd))
        data = np.fromfile(fh, dtype=dtype)
    size = int(np.prod(dims, dtype=np.int64))
    if data.size < size:
        raise ValueError(f"{path} truncated: expected {size} values, found {data.size}")
    if data.size > size:
        data = data[:size]
    return data.reshape(dims, order="F")

def _read_state(path: Path) -> np.ndarray:
    return _read_binary_array(path, ">f8")

def _read_obs_component(path: Path) -> np.ndarray:
    base = path.name.split("_pe", 1)[0]
    component = base.rsplit("_", 1)[-1]
    int_components = {"set", "idx", "key", "qc"}
    dtype = ">i4" if component in int_components else ">f8"
    return _read_binary_array(path, dtype)

def load_rank_members(
    dump_dir: str | Path,
    prefix: str,
    pe_tag: str | int
    ) -> xr.DataArray:
    dump_dir = Path(dump_dir)
    pe_norm = _normalize_pe_tag(pe_tag)
    data_dir = _resolve_state_dir(dump_dir, prefix)
    mems = ["mem0001", "mem0002", "memmean"]
    x = [_read_state(data_dir / f"{prefix}_{pe_norm}.{mem}.bin") for mem in mems]

    y = np.zeros((256, 320, 45, 3, 11))
    y_flat = y.reshape(-1, 45, 3, 11)
    
    y_flat[::3] = x[0]
    y_flat[1::3] = x[1]
    y_flat[2::3] = x[2]
    
    y = y_flat.reshape((256, 320, 45, 3, 11))
    
    dims=["y", "x", "z", "ens", "variable"]
    da = xr.DataArray(y, dims=dims,
                      coords={"variable":['U', 'V', 'W', 'T', 'P', 'QV', 'QC', 'QR', 'QI', 'QS', 'QG'],
                              "ens":["0001", "0002", "mean"],
                              "x":X_LEVELS[pe_norm],
                              "y":Y_LEVELS[pe_norm],
                              "z":Z_LEVELS
                             }
                      )

    da = da.where(da>-10**30)
    return da

def load_grid_info(
    dump_dir: str | Path,
    pe_tag: str | int,
    member: str | int = "mem0001",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load rig1/rjg1/topo1/hgt1 arrays dumped per rank."""
    dump_dir = Path(dump_dir)
    base_dir = dump_dir / "grid"
    if not base_dir.exists():
        raise FileNotFoundError(f"{base_dir} not found")
    pe_norm = _normalize_pe_tag(pe_tag)
    mem_norm = _normalize_member(member)
    rig1_path = base_dir / f"rig1_{pe_norm}.{mem_norm}.bin"
    rjg1_path = base_dir / f"rjg1_{pe_norm}.{mem_norm}.bin"
    if not rig1_path.exists() or not rjg1_path.exists():
        raise FileNotFoundError(f"rig1/rjg1 files missing for {pe_norm}.{mem_norm}")
    rig1 = _read_binary_array(rig1_path, ">f8").reshape(-1)
    rjg1 = _read_binary_array(rjg1_path, ">f8").reshape(-1)
    topo1 = None
    hgt1 = None
    if (base_dir / f"topo1_{pe_norm}.{mem_norm}.bin").exists():
        topo1 = _read_binary_array(base_dir / f"topo1_{pe_norm}.{mem_norm}.bin", ">f8").reshape(-1)
    if (base_dir / f"hgt1_{pe_norm}.{mem_norm}.bin").exists():
        hgt1 = _read_binary_array(base_dir / f"hgt1_{pe_norm}.{mem_norm}.bin", ">f8")
    return rig1, rjg1, topo1, hgt1

def load_localization_tables(
    dump_dir: str | Path,
    pe_tag: str | int,
    member: str | int = "mem0001",
) -> dict[str, np.ndarray]:
    """Load variable-localization metadata dumped under letkf_dump/localization."""
    dump_dir = Path(dump_dir)
    base_dir = dump_dir / "localization"
    if not base_dir.exists():
        raise FileNotFoundError(f"{base_dir} not found")
    pe_norm = _normalize_pe_tag(pe_tag)
    mem_norm = _normalize_member(member)
    def _load(name: str, dtype: str):
        path = base_dir / f"{name}_{pe_norm}.{mem_norm}.bin"
        if not path.exists():
            raise FileNotFoundError(path)
        return _read_binary_array(path, dtype)
    return {
        "var_local": _load("var_local", ">f8"),
        "var_local_n2nc": _load("var_local_n2nc", ">i4"),
        "var_local_n2n": _load("var_local_n2n", ">i4"),
        "uid_obs_varlocal": _load("uid_obs_varlocal", ">i4"),
        "n_merge": _load("n_merge", ">i4"),
        "ic_merge": _load("ic_merge", ">i4"),
        "elm_u_ctype": _load("elm_u_ctype", ">i4"),
        "typ_ctype": _load("typ_ctype", ">i4"),
    }

def load_state_metadata(
    dump_dir: str | Path,
    pe_tag: str | int,
    member: str | int,
) -> dict[str, int | float | str]:
    """Load per-rank metadata dumped alongside the state fields."""

    dump_dir = Path(dump_dir)
    meta_dir = _resolve_state_meta_dir(dump_dir)
    pe_norm = _normalize_pe_tag(pe_tag)
    mem_norm = _normalize_member(member)
    meta_path = meta_dir / f"state_meta_{pe_norm}.{mem_norm}.txt"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    meta = _read_text_metadata(meta_path)
    if not meta:
        raise FileNotFoundError(f"{meta_path} is empty")
    return meta

def load_obsda_var(dump_dir: str | Path, pe_tag: str | int, member: str | int, variable: str) -> np.ndarray:
    return _load_obsda_var_from_stage(dump_dir, "obsda_after_obsope", pe_tag, member, variable)

def load_obsda_sorted_var(dump_dir: str | Path, pe_tag: str | int, member: str | int, variable: str) -> np.ndarray:
    return _load_obsda_var_from_stage(dump_dir, "obsda_after_set_letkf", pe_tag, member, variable)

def load_obsda(dump_dir: str | Path, pe_tag: str | int) -> xr.Dataset:
    return _load_obsda_dataset(dump_dir, "obsda_after_obsope", pe_tag)

def load_obsda_sorted(dump_dir: str | Path, pe_tag: str | int) -> xr.Dataset:
    return _load_obsda_dataset(dump_dir, "obsda_after_set_letkf", pe_tag)

def load_obs_raw(
    dump_dir: str | Path,
    *,
    pe_tag: str | int | None = None,
    member: str | int | None = None,
) -> list[dict]:
    """Load the raw observation dumps (obs_info arrays) as a list of dictionaries."""
    base_dir = Path(dump_dir) / "obs_raw"
    if not base_dir.exists():
        raise FileNotFoundError(f"{base_dir} not found")
    pe_norm = _normalize_pe_tag(pe_tag) if pe_tag is not None else None
    mem_norm = _normalize_member(member) if member is not None else None

    records: list[dict] = []
    for obs_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        obs_index = _extract_obs_index(obs_dir.name)
        groups: dict[tuple[str, str], dict[str, Path]] = {}
        for path in sorted(obs_dir.glob("*.bin")):
            match = _OBS_RAW_FILE_RE.match(path.name)
            if not match:
                continue
            group_key = (match.group("pe"), match.group("member"))
            field = match.group("field").lower()
            groups.setdefault(group_key, {})[field] = path
        if not groups:
            continue
        group_key, files = _select_obs_group(groups, pe_norm, mem_norm)
        pe_sel, mem_sel = group_key
        meta_path = obs_dir / f"obs_meta_{pe_sel}.{mem_sel}.txt"
        meta = _read_text_metadata(meta_path)
        data = {}
        for field, path in sorted(files.items()):
            dtype = ">i4" if field in _OBS_RAW_INT_FIELDS else ">f8"
            data[field] = _read_binary_array(path, dtype)
        records.append(
            {
                "index": obs_index,
                "pe": pe_sel,
                "member": mem_sel,
                "meta": meta,
                "data": data,
            }
        )
    if not records:
        raise FileNotFoundError(f"No observation dumps found under {base_dir}")
    return records

def load_obsgrd(
    dump_dir: str | Path,
    pe_tag: str | int,
    member: str | int,
) -> dict:
    """Load the observation-sorting grid diagnostics."""
    stage_dir = Path(dump_dir) / "obsgrd"
    if not stage_dir.exists():
        raise FileNotFoundError(f"{stage_dir} not found")
    pe_norm = _normalize_pe_tag(pe_tag)
    mem_norm = _normalize_member(member)
    summary_path = stage_dir / f"obsgrd_summary_{pe_norm}.{mem_norm}.txt"
    summary = _read_text_metadata(summary_path)
    if not summary:
        raise FileNotFoundError(f"{summary_path} not found or empty")
    nctype = int(summary.get("nctype", 0))
    ctypes: list[dict] = []
    for ictype in range(1, nctype + 1):
        cdir = stage_dir / f"type_{ictype:05d}"
        meta_path = cdir / f"obsgrd_meta_{pe_norm}.{mem_norm}.txt"
        meta = _read_text_metadata(meta_path)
        base_tag = f"type{ictype:04d}"
        arrays: dict[str, np.ndarray] = {}
        for field, dtype in _OBSGRD_INT_ARRAYS.items():
            path = cdir / f"{base_tag}_{field}_{pe_norm}.{mem_norm}.bin"
            if path.exists():
                arrays[field] = _read_binary_array(path, dtype)
        ctypes.append({"ctype": ictype, "meta": meta, "arrays": arrays})
    return {"summary": summary, "ctypes": ctypes}

def _load_obsda_var_from_stage(
    dump_dir: str | Path,
    stage_name: str,
    pe_tag: str | int,
    member: str | int,
    variable: str,
) -> np.ndarray:
    stage_dir, legacy_prefix = _resolve_obs_stage_dir(Path(dump_dir), stage_name)
    pe_norm = _normalize_pe_tag(pe_tag)
    mem_norm = _normalize_member(member)
    var_norm = variable.strip().lower()
    if legacy_prefix:
        filename = f"obsda_{legacy_prefix}_{var_norm}_{pe_norm}.{mem_norm}.bin"
    else:
        filename = f"obsda_{var_norm}_{pe_norm}.{mem_norm}.bin"
    path = stage_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    return _read_obs_component(path)

def _load_obsda_dataset(dump_dir: str | Path, stage_name: str, pe_tag: str | int) -> xr.Dataset:
    stage_dir, legacy_prefix = _resolve_obs_stage_dir(Path(dump_dir), stage_name)
    if not stage_dir.exists():
        raise FileNotFoundError(f"{stage_dir} not found")
    pe_norm = _normalize_pe_tag(pe_tag)
    grouped = _collect_obsda_files(stage_dir, pe_norm, legacy_prefix)
    members = sorted({mem for member_map in grouped.values() for mem in member_map})
    data_vars = {}
    for var_name, paths_per_member in grouped.items():
        arrays: list[np.ndarray] = []
        for mem in members:
            path = paths_per_member.get(mem)
            if path is None:
                raise FileNotFoundError(f"Missing {var_name} for member {mem} in {stage_dir}")
            arrays.append(_read_obs_component(path))
        first = arrays[0]
        if not all(arr.shape == first.shape for arr in arrays[1:]):
            raise ValueError(f"Inconsistent shapes for {var_name} in {stage_dir}")
        data = np.stack(arrays, axis=0)
        dims = _obs_dims(first.shape)
        coords = _obs_coords(first.shape, members)
        data_vars[var_name] = xr.DataArray(data, dims=dims, coords=coords)
    return xr.Dataset(data_vars)

def _collect_obsda_files(stage_dir: Path, pe_tag: str, legacy_prefix: str | None) -> dict[str, dict[str, Path]]:
    grouped: dict[str, dict[str, Path]] = {}
    if legacy_prefix:
        prefix = f"obsda_{legacy_prefix}_"
    else:
        prefix = "obsda_"
    pattern = f"{prefix}*_{pe_tag}.mem*.bin"
    for path in sorted(stage_dir.glob(pattern)):
        match = _OBSDA_FILE_RE.match(path.name)
        if not match:
            continue
        var_token = match.group("var").lower()
        if legacy_prefix:
            legacy_token = legacy_prefix.lower() + "_"
            if not var_token.startswith(legacy_token):
                continue
            var_name = var_token[len(legacy_token):]
        else:
            var_name = var_token
        member = match.group("member")
        grouped.setdefault(var_name, {})[member] = path
    if not grouped:
        raise FileNotFoundError(f"No observation files for {pe_tag} under {stage_dir}")
    return grouped

def _obs_dims(shape: tuple[int, ...]) -> list[str]:
    dims = ["member"]
    if not shape:
        return dims
    for idx in range(len(shape) - 1):
        dims.append(f"aux_dim_{idx}")
    dims.append("obs")
    return dims

def _obs_coords(shape: tuple[int, ...], members: list[str]) -> dict[str, np.ndarray]:
    coords: dict[str, np.ndarray] = {"member": np.array(members)}
    for idx, size in enumerate(shape[:-1]):
        coords[f"aux_dim_{idx}"] = np.arange(size)
    if shape:
        coords["obs"] = np.arange(shape[-1])
    return coords

def _resolve_state_dir(dump_dir: Path, prefix: str) -> Path:
    candidate = dump_dir / prefix
    if candidate.is_dir():
        return candidate
    return dump_dir

def _resolve_state_meta_dir(dump_dir: Path) -> Path:
    candidate = dump_dir / STATE_META_SUBDIR
    if candidate.is_dir():
        return candidate
    return dump_dir

def _resolve_obs_stage_dir(dump_dir: Path, stage_name: str) -> tuple[Path, str | None]:
    stage_dir = dump_dir / stage_name
    if stage_dir.exists():
        return stage_dir, None
    nested = dump_dir / "obsda" / stage_name
    if nested.exists():
        return nested, None
    legacy_root = dump_dir / "obsda"
    if legacy_root.exists():
        legacy_token = stage_name
        if legacy_token.startswith("obsda_"):
            legacy_token = legacy_token[len("obsda_"):]
        return legacy_root, legacy_token
    return stage_dir, None

def _normalize_pe_tag(pe_tag: str | int) -> str:
    if isinstance(pe_tag, int):
        if pe_tag < 0:
            raise ValueError("pe_tag must be non-negative")
        return f"pe{pe_tag:06d}"
    tag = pe_tag.strip().lower()
    if tag.startswith("pe"):
        suffix = tag[2:]
        if suffix.isdigit():
            return f"pe{int(suffix):06d}"
        return f"pe{suffix}"
    if tag.isdigit():
        return f"pe{int(tag):06d}"
    raise ValueError(f"Cannot parse pe_tag '{pe_tag}'")

def _normalize_member(member: str | int) -> str:
    if isinstance(member, int):
        if member < 0:
            raise ValueError("member must be non-negative")
        return f"mem{member:04d}"
    tag = member.strip().lower()
    if tag in _SPECIAL_MEMBERS:
        return _SPECIAL_MEMBERS[tag]
    if tag.startswith("mem"):
        suffix = tag[3:]
        if suffix.isdigit():
            return f"mem{int(suffix):04d}"
        return f"mem{suffix}"
    if tag.isdigit():
        return f"mem{int(tag):04d}"
    raise ValueError(f"Cannot parse member '{member}'")

def _extract_obs_index(dirname: str) -> int:
    match = re.match(r"obs(\d{4})", dirname.lower())
    if not match:
        raise ValueError(f"Unrecognized observation directory '{dirname}'")
    return int(match.group(1))

def _select_obs_group(
    groups: dict[tuple[str, str], dict[str, Path]],
    pe_norm: str | None,
    mem_norm: str | None,
) -> tuple[tuple[str, str], dict[str, Path]]:
    for (pe, mem), files in sorted(groups.items()):
        if pe_norm is not None and pe != pe_norm:
            continue
        if mem_norm is not None and mem != mem_norm:
            continue
        return (pe, mem), files
    raise FileNotFoundError(
        f"No observation dump matching pe={pe_norm} member={mem_norm}"
    )

def _read_text_metadata(path: Path) -> dict[str, int | float | str]:
    if not path.exists():
        return {}
    meta: dict[str, int | float | str] = {}
    with path.open() as fh:
        for line in fh:
            if "=" not in line:
                continue
            key, value = line.strip().split("=", 1)
            meta[key.strip()] = _coerce_value(value.strip())
    return meta

def _coerce_value(text: str) -> int | float | str:
    if not text:
        return text
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text
