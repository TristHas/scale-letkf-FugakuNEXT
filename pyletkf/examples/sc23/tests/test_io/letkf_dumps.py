from pathlib import Path
import re
from typing import Dict, List
import numpy as np
import xarray as xr

import torch
import xtensor as xt

from .utils import (
    _normalize_member,
    _normalize_pe_tag,
    _read_text_metadata,
    _read_binary_array
)

DUMP_ROOT = Path("/home/tristan/workspace/scale_letkf/scale-letkf-FugakuNEXT/result/SC23/20210730060030/letkf_dump")
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


###
### State
###
def load_state(
    rank: int,
    prefix: str = "gues3d",
    mems = ["mem0001", "mem0002", "memmean"]
    ) -> xr.DataArray:
    """
    """
    pe_norm = _normalize_pe_tag(rank)
    data_dir = DUMP_ROOT / prefix
    
    x = [_read_binary_array(data_dir / f"{prefix}_{pe_norm}.{mem}.bin", ">f8") for mem in mems]

    y = np.zeros((256, 320, 45, 3, 11))
    y_flat = y.reshape(-1, 45, 3, 11)
    
    y_flat[::3] = x[0]
    y_flat[1::3] = x[1]
    y_flat[2::3] = x[2]
    
    y = y_flat.reshape((256, 320, 45, 3, 11))
    
    dims=["y", "x", "z", "ens", "variable"]
    da = xr.DataArray(y, dims=dims,
                      coords={"variable":['U', 'V', 'W', 'T', 'P', 'QV', 'QC', 'QR', 'QI', 'QS', 'QG'],
                              "ens":[x.replace("mem", "") for x in mems],
                              "x":X_LEVELS[pe_norm],
                              "y":Y_LEVELS[pe_norm],
                              "z":Z_LEVELS
                             }
                      )

    da = da.where(da>-10**30)
    return da

def load_state_metadata(
    pe_tag: str | int,
    member: str | int,
) -> dict[str, int | float | str]:
    """
        Load per-rank metadata dumped alongside the state fields.
    """
    meta_dir = DUMP_ROOT / STATE_META_SUBDIR
    pe_norm  = _normalize_pe_tag(pe_tag)
    mem_norm = _normalize_member(member)
    meta_path = meta_dir / f"state_meta_{pe_norm}.{mem_norm}.txt"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    meta = _read_text_metadata(meta_path)
    if not meta:
        raise FileNotFoundError(f"{meta_path} is empty")
    return meta


###
### OBSDA
###

DATA_TYPES = {
  # integer components (per-observation)
  "set": ">i4",
  "idx": ">i4",
  "key": ">i4",
  "qc":  ">i4",

  # real-valued components (per-observation)
  "val":    ">f8",
  "ensval": ">f8",   # stacked per-member, so each file holds (nmem, nobs)
  "eqv":    ">f8",
  "tm":     ">f8",
  "pm":     ">f8",
  "qv":     ">f8",
  "pert":   ">f8",
  "epert":  ">f8",

  # nosort coordinates (observations only, no member axis)
  "ri": ">f8",
  "rj": ">f8",
}

def tile_name(rank):
    """
    """
    return "pe" + str(rank).zfill(6)

def bin_file_name(variable, rank, ens):
    """
    """
    prefix = "obsdanosort" if variable in {"ri", "rj"} else "obsda" 
    return f"{prefix}_{variable}_{tile_name(rank)}.{ens}.bin"

def list_obsda_var(root):
    return np.unique([x.stem.split("_")[1] for x in root.glob("*.bin")])    
    
def load_obsda_file(root, variable, rank, ens="mem0001"):
    """
    """
    fp = root / bin_file_name(variable, rank, ens)
    data_type = DATA_TYPES[variable]
    data = _read_binary_array(fp, data_type).astype(data_type[1:]) #.reshape(-1)
    return torch.from_numpy(data).squeeze().t()

def load_obsda_var(root, variable, rank, ens=["mem0001", "mem0002"]):
    """
    """
    return torch.stack([load_obsda_file(root, variable, rank, en) for en in ens], dim=-1)

def load_obsda_ds(root, rank, ens=["mem0001", "mem0002"]):
    """
    """
    idx = load_obsda_file(root, "idx", rank)
    ds = xt.Dataset(coords={"obs":idx-1, "ens":ens})
    variables = list_obsda_var(root)
    for variable in variables:
        if variable !="idx":
            data = load_obsda_file(root, variable, rank)#, ens=ens)
            if data.ndim == 2:
                ds[variable]=(("obs", "ens"), data)
            else:
                ds[variable]=(("obs",), data)
    return ds

def load_obsda_obsop(rank, ens=["mem0001", "mem0002"]):
    """
    """
    root = DUMP_ROOT / "obsda_after_obsope"

    idx = load_obsda_file(root, "idx", rank)
    ds = xt.Dataset(coords={"obs":idx-1, "ens":ens})
    variables = list_obsda_var(root)
    for variable in variables:
        if variable !="idx":
            data = load_obsda_var(root, variable, rank, ens=ens)
            if variable == "ensval":
                ds[variable]=(("obs", "ens"), data)
            else:
                ds[variable]=(("obs",), data.t()[0])
    return ds

def load_obsda_setletkf(rank, ens=["mem0001", "mem0002"]):
    """
    """
    root = DUMP_ROOT / "obsda_after_set_letkf"

    idx = load_obsda_file(root, "idx", rank)
    ds = xt.Dataset(coords={"obs":idx-1, "ens":ens})
    variables = list_obsda_var(root)
    for variable in variables:
        if variable !="idx":
            data = load_obsda_file(root, variable, rank)#, ens=ens)
            if data.ndim == 2:
                ds[variable]=(("obs", "ens"), data)
            else:
                ds[variable]=(("obs",), data)
    return ds


###
### Others
###

def load_grid_info(
    pe_tag: str | int,
    member: str | int = "mem0001",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
        Load rig1/rjg1/topo1/hgt1 arrays dumped per rank.
    """
    base_dir = DUMP_ROOT / "grid"
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
    rank: str | int,
    member: str | int = "mem0001",
) -> dict[str, np.ndarray]:
    """Load variable-localization metadata dumped under letkf_dump/localization."""
    base_dir = DUMP_ROOT / "localization"
    if not base_dir.exists():
        raise FileNotFoundError(f"{base_dir} not found")
    pe_norm = _normalize_pe_tag(rank)
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
    
def load_obs_raw(
    rank: str | int | None = None,
    member: str | int | None = None,
) -> list[dict]:
    """
        Load the raw observation dumps (obs_info arrays) as a list of dictionaries.
    """
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
    base_dir = DUMP_ROOT / "obs_raw"
    if not base_dir.exists():
        raise FileNotFoundError(f"{base_dir} not found")
    pe_norm = _normalize_pe_tag(rank) if rank is not None else None
    mem_norm = _normalize_member(member) if member is not None else None

    records: list[dict] = []
    for obs_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        obs_index = _extract_obs_index(obs_dir.name)
        groups: dict[tuple[str, str], dict[str, Path]] = {}
        for path in sorted(obs_dir.glob("*.bin")):
            match = _OBS_RAW_FILE_RE.match(path.name)
            if not match: continue
            group_key = (match.group("pe"), match.group("member"))
            field = match.group("field").lower()
            groups.setdefault(group_key, {})[field] = path
        if not groups: continue
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
        rank: int,
        member: str | int,
    ) -> dict:
    """Load the observation-sorting grid diagnostics."""
    stage_dir = DUMP_ROOT / "obsgrd"
    if not stage_dir.exists():
        raise FileNotFoundError(f"{stage_dir} not found")
    pe_norm = _normalize_pe_tag(rank)
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

def load_obsop_interp(
    stage: str = "radar_pre_trans",
    pe_tag: str | int = 0,
    member: str | int = "mem0001",
) -> dict[str, np.ndarray]:
    """
        Load the pre-observation-operator interpolation diagnostics dumped from
        LETKF_obs_operator.
    """
    stage_dir = DUMP_ROOT / "obsop_interp" / stage
    if not stage_dir.exists():
        raise FileNotFoundError(f"{stage_dir} not found")
    pe_norm = _normalize_pe_tag(pe_tag)
    mem_norm = _normalize_member(member)
    components = {
        "set": (">i4", f"obsop_interp_set_{pe_norm}.{mem_norm}.bin"),
        "idx": (">i4", f"obsop_interp_idx_{pe_norm}.{mem_norm}.bin"),
        "ri":  (">f8", f"obsop_interp_ri_{pe_norm}.{mem_norm}.bin"),
        "rj":  (">f8", f"obsop_interp_rj_{pe_norm}.{mem_norm}.bin"),
        "rk":  (">f8", f"obsop_interp_rk_{pe_norm}.{mem_norm}.bin"),
        "level_kind": (">i4", f"obsop_interp_levelkind_{pe_norm}.{mem_norm}.bin"),
    }
    data: dict[str, np.ndarray] = {}
    for key, (dtype, filename) in components.items():
        path = stage_dir / filename
        if not path.exists():
            if key == "level_kind":
                continue
            raise FileNotFoundError(path)
        data[key] = _read_binary_array(path, dtype)
    return data

_EXP_FIX = re.compile(r'^([+-]?\d+(?:\.\d+)?)([+-]\d+)$')

def _parse_float(token: str) -> float:
    tok = token.strip().replace('D', 'E')
    if 'E' not in tok:
        m = _EXP_FIX.match(tok)
        if m:
            tok = f"{m.group(1)}E{m.group(2)}"
    return float(tok)
    
def load_obs_interp_dump():
    """
    Load interpolated-state dumps from OBSOP_INTERP_DUMP.
    Parameters
    ----------
    root : str or Path
        Directory holding obs_interp_rank*.dat.
    Returns
    -------
    list[dict]
    """
    root = DUMP_ROOT / "obs_interp_dump"
    entries = []
    for path in sorted(root.glob("obs_interp_rank*.dat")):
        with path.open() as fh:
            for line in fh:
                if not line or line[0] == '#':
                    continue
                parts = line.split()
                entry = {
                    "obs_set": int(parts[0]),
                    "obs_idx": int(parts[1]),
                    "elm": int(parts[2]),
                    "typ": int(parts[3]),
                    "stage": parts[4].strip(),
                    "rank_global": int(parts[5]),
                    "rank_local": int(parts[6]),
                    "ri": _parse_float(parts[7]),
                    "rj": _parse_float(parts[8]),
                    "rk": _parse_float(parts[9]),
                    "lon": _parse_float(parts[10]),
                    "lat": _parse_float(parts[11]),
                    "lev": _parse_float(parts[12]),
                    "label": parts[13].strip(),
                    "values": np.array([_parse_float(tok) for tok in parts[14:]], dtype=float),
                    "path": path,
                }
                entries.append(entry)
    df = pd.DataFrame([[d["obs_idx"]] + d["values"].tolist() for d in dumps])
    df = df.set_index(0).sort_index()
    return df