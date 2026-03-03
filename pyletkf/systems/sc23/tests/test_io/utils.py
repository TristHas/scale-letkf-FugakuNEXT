import numpy as np
from pathlib import Path

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
