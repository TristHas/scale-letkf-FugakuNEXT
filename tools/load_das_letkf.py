"""Helpers to read das_letkf stage dumps."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .load_dumps import (
    _normalize_member,
    _normalize_pe_tag,
    _read_binary_array,
    _read_text_metadata,
)

_STAGE_FIELDS: dict[tuple[str, str], Mapping[str, str]] = {
    ("obs_local", "before"): {"search_q0": ">i4"},
    (
        "obs_local",
        "after",
    ): {
        "hdxf": ">f8",
        "rdiag": ">f8",
        "rloc": ">f8",
        "dep": ">f8",
        "search_q0": ">i4",
        "nobsl_t": ">i4",
        "cutd_t": ">f8",
    },
    ("letkf_core", "before"): {
        "hdxf": ">f8",
        "rdiag": ">f8",
        "rloc": ">f8",
        "dep": ">f8",
    },
    ("letkf_core", "after"): {
        "trans": ">f8",
        "transm": ">f8",
        "pa": ">f8",
        "transmd": ">f8",
    },
    ("postproc", "before"): {
        "trans": ">f8",
        "transm": ">f8",
        "gues_members": ">f8",
    },
    ("postproc", "after"): {
        "transrlx": ">f8",
        "anal_members": ">f8",
        "anal_det": ">f8",
    },
}


def list_das_calls(dump_dir: str | Path) -> list[int]:
    """Return sorted call IDs available under the das_letkf dump tree."""

    base = Path(dump_dir) / "das_letkf"
    if not base.exists():
        return []
    calls: set[int] = set()
    for meta_path in base.glob("*/*/meta_call*.txt"):
        name = meta_path.name
        if "meta_call" not in name:
            continue
        try:
            token = name.split("meta_call", 1)[1]
            token = token.split("_", 1)[0]
            calls.add(int(token))
        except (IndexError, ValueError):
            continue
    return sorted(calls)


def load_das_stage(
    dump_dir: str | Path,
    call_id: int | str,
    stage: str,
    phase: str,
    *,
    pe_tag: str | int = "pe000000",
    member: str | int = "mem0001",
) -> dict[str, Any]:
    """Load metadata and arrays for a given stage/phase dump."""

    phase_dir = _stage_dir(dump_dir, stage, phase)
    pe_norm = _normalize_pe_tag(pe_tag)
    mem_norm = _normalize_member(member)
    call_tag = _format_call_id(call_id)
    meta = _read_stage_metadata(phase_dir, call_tag, pe_norm, mem_norm)
    arrays = _read_stage_arrays(phase_dir, call_tag, pe_norm, mem_norm, stage, phase)
    return {"meta": meta, "data": arrays}


def load_das_obs_local_before(*args, **kwargs) -> dict[str, Any]:
    return load_das_stage(*args, stage="obs_local", phase="before", **kwargs)


def load_das_obs_local_after(*args, **kwargs) -> dict[str, Any]:
    return load_das_stage(*args, stage="obs_local", phase="after", **kwargs)


def load_das_letkf_core_before(*args, **kwargs) -> dict[str, Any]:
    return load_das_stage(*args, stage="letkf_core", phase="before", **kwargs)


def load_das_letkf_core_after(*args, **kwargs) -> dict[str, Any]:
    return load_das_stage(*args, stage="letkf_core", phase="after", **kwargs)


def load_das_postproc_before(*args, **kwargs) -> dict[str, Any]:
    return load_das_stage(*args, stage="postproc", phase="before", **kwargs)


def load_das_postproc_after(*args, **kwargs) -> dict[str, Any]:
    return load_das_stage(*args, stage="postproc", phase="after", **kwargs)


def _stage_dir(dump_dir: str | Path, stage: str, phase: str) -> Path:
    base = Path(dump_dir) / "das_letkf"
    phase_dir = base / stage / phase
    if not phase_dir.is_dir():
        raise FileNotFoundError(phase_dir)
    return phase_dir


def _read_stage_metadata(phase_dir: Path, call_tag: str, pe_tag: str, member: str) -> dict[str, Any]:
    meta_path = phase_dir / f"meta_call{call_tag}_{pe_tag}.{member}.txt"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    return _read_text_metadata(meta_path)


def _read_stage_arrays(
    phase_dir: Path,
    call_tag: str,
    pe_tag: str,
    member: str,
    stage: str,
    phase: str,
) -> dict[str, Any]:
    arrays: dict[str, Any] = {}
    fields = _STAGE_FIELDS.get((stage, phase), {})
    for name, dtype in fields.items():
        path = phase_dir / f"{name}_call{call_tag}_{pe_tag}.{member}.bin"
        if not path.exists():
            continue
        arrays[name] = _read_binary_array(path, dtype)
    return arrays


def _format_call_id(call_id: int | str) -> str:
    if isinstance(call_id, int):
        if call_id < 0:
            raise ValueError("call_id must be non-negative")
        return f"{call_id:012d}"
    token = str(call_id).strip()
    if token.lower().startswith("call"):
        token = token[4:]
    if token.startswith("_"):
        token = token[1:]
    if not token.isdigit():
        raise ValueError(f"Cannot parse call_id '{call_id}'")
    return f"{int(token):012d}"
