"""NumPy reproduction of the SCALE LETKF core routine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .load_das_letkf import (
    load_das_letkf_core_after,
    load_das_letkf_core_before,
)

SIGMA_B = 0.04


@dataclass(frozen=True)
class LetkfCoreIdentifier:
    dump_dir: Path | str
    call_id: int
    pe_tag: str = "pe000000"
    member: str = "mem0001"


@dataclass
class LetkfCoreInputs:
    identifier: LetkfCoreIdentifier
    before_meta: Mapping[str, Any]
    before_arrays: Mapping[str, np.ndarray]
    after_meta: Mapping[str, Any]
    after_arrays: Mapping[str, np.ndarray]


def load_letkf_core_data(identifier: LetkfCoreIdentifier | Mapping[str, Any]) -> LetkfCoreInputs:
    ident = _normalize_identifier(identifier)
    dump_dir = Path(ident.dump_dir)
    before = load_das_letkf_core_before(
        dump_dir,
        ident.call_id,
        pe_tag=ident.pe_tag,
        member=ident.member,
    )
    after = load_das_letkf_core_after(
        dump_dir,
        ident.call_id,
        pe_tag=ident.pe_tag,
        member=ident.member,
    )
    return LetkfCoreInputs(
        identifier=ident,
        before_meta=before["meta"],
        before_arrays=before["data"],
        after_meta=after["meta"],
        after_arrays=after["data"],
    )


def letkf_core(data: LetkfCoreInputs) -> dict[str, np.ndarray | float | None]:
    hdxf = np.asarray(data.before_arrays["hdxf"], dtype=np.float64)
    rdiag = np.asarray(data.before_arrays["rdiag"], dtype=np.float64)
    rloc = np.asarray(data.before_arrays["rloc"], dtype=np.float64)
    dep = np.asarray(data.before_arrays["dep"], dtype=np.float64)

    ne = hdxf.shape[1]
    nobsl = int(data.before_meta.get("nobsl", 0))
    if nobsl > hdxf.shape[0]:
        raise ValueError("nobsl exceeds hdxf rows")
    hdxf_use = hdxf[:nobsl, :]
    rdiag_use = rdiag[:nobsl]
    rloc_use = rloc[:nobsl]
    dep_use = dep[:nobsl]

    parm_infl = float(data.before_meta.get("parm_infl", 1.0))
    rdiag_wloc = _meta_bool(data.before_meta, "rdiag_wloc", default=False)
    infl_update = _meta_bool(data.before_meta, "infl_update", default=False)
    depd_use = None  # Not available in dumps

    if nobsl == 0:
        trans = np.zeros((ne, ne), dtype=np.float64)
        np.fill_diagonal(trans, np.sqrt(parm_infl))
        transm = np.zeros(ne, dtype=np.float64)
        pa = (parm_infl / max(ne - 1, 1)) * np.eye(ne, dtype=np.float64)
        transmd = np.zeros(ne, dtype=np.float64) if depd_use is not None else None
        return {
            "trans": trans,
            "transm": transm,
            "pa": pa,
            "parm_infl": parm_infl,
            "transmd": transmd,
        }

    if rdiag_wloc:
        factors = 1.0 / rdiag_use
    else:
        factors = np.where(rdiag_use != 0.0, rloc_use / rdiag_use, 0.0)
    hdxb_rinv = hdxf_use * factors[:, None]

    work1 = hdxb_rinv.T @ hdxf_use
    rho = 1.0 / parm_infl
    work1[np.diag_indices(ne)] += (ne - 1) * rho

    eival, eivec = np.linalg.eigh(work1)
    eival = np.maximum(eival, 1.0e-12)

    inv_diag = (1.0 / eival).astype(np.float64)
    pa = eivec @ (inv_diag[:, None] * eivec.T)

    work2 = hdxb_rinv.T @ dep_use
    work3 = pa @ work2

    trans_base = _compute_transform(eivec, eival, ne)
    transm = work3.copy()

    transmd = None
    if depd_use is not None:
        work2d = hdxb_rinv.T @ depd_use
        transmd = pa @ work2d

    if infl_update:
        parm_infl = _update_inflation(
            parm_infl,
            dep_use,
            rdiag_use,
            rloc_use,
            hdxb_rinv,
            hdxf_use,
            rdiag_wloc,
        )

    return {
        "trans": trans_base,
        "transm": transm,
        "pa": pa,
        "parm_infl": parm_infl,
        "transmd": transmd,
    }


def test_letkf_core(identifier: LetkfCoreIdentifier | Mapping[str, Any], *, atol: float = 1.0e-10, rtol: float = 1.0e-10) -> dict[str, float]:
    data = load_letkf_core_data(identifier)
    result = letkf_core(data)
    errors: dict[str, float] = {}

    for key in ("trans", "transm", "pa"):
        ref = data.after_arrays.get(key)
        if ref is None:
            continue
        ref_np = np.asarray(ref, dtype=np.float64)
        if key == "pa" and np.max(np.abs(ref_np)) < 1.0e-12:
            continue
        np.testing.assert_allclose(result[key], ref_np, atol=atol, rtol=rtol)
        errors[key] = float(np.max(np.abs(result[key] - ref_np)))

    ref_parm = data.after_meta.get("parm_infl_post")
    if ref_parm is not None:
        np.testing.assert_allclose(result["parm_infl"], float(ref_parm), atol=atol, rtol=rtol)
        errors["parm_infl"] = abs(result["parm_infl"] - float(ref_parm))

    if "transmd" in data.after_arrays:
        errors["transmd"] = float("nan")  # Missing depd prevents reproduction

    return errors


def _compute_transform(eivec: np.ndarray, eival: np.ndarray, ne: int) -> np.ndarray:
    scales = np.sqrt((ne - 1) / eival)
    work = eivec * scales
    return work @ eivec.T


def _update_inflation(
    parm_infl: float,
    dep: np.ndarray,
    rdiag: np.ndarray,
    rloc: np.ndarray,
    hdxb_rinv: np.ndarray,
    hdxb: np.ndarray,
    rdiag_wloc: bool,
) -> float:
    if rdiag_wloc:
        parm1 = np.sum((dep * dep) / rdiag)
    else:
        parm1 = np.sum(((dep * dep) / rdiag) * rloc)
    parm2 = np.sum(hdxb_rinv * hdxb) / (hdxb.shape[1] - 1)
    parm3 = np.sum(rloc)
    parm4 = (parm1 - parm3) / parm2 - parm_infl
    sigma_o = 2.0 / parm3 * ((parm_infl * parm2 + parm3) / parm2) ** 2
    gain = SIGMA_B ** 2 / (sigma_o + SIGMA_B ** 2)
    return parm_infl + gain * parm4


def _meta_bool(meta: Mapping[str, Any], key: str, default: bool = False) -> bool:
    value = meta.get(key)
    if value is None:
        return default
    if isinstance(value, str):
        token = value.strip().lower()
        if token in ("true", ".true.", "1", "yes"):
            return True
        if token in ("false", ".false.", "0", "no"):
            return False
    return bool(value)


def _normalize_identifier(identifier: LetkfCoreIdentifier | Mapping[str, Any]) -> LetkfCoreIdentifier:
    if isinstance(identifier, LetkfCoreIdentifier):
        return identifier
    dump_dir = identifier.get("dump_dir")
    call_id = identifier.get("call_id")
    if dump_dir is None or call_id is None:
        raise ValueError("identifier must contain dump_dir and call_id")
    pe_tag = identifier.get("pe_tag", "pe000000")
    member = identifier.get("member", "mem0001")
    return LetkfCoreIdentifier(dump_dir=dump_dir, call_id=int(call_id), pe_tag=str(pe_tag), member=str(member))


__all__ = ["LetkfCoreIdentifier", "LetkfCoreInputs", "load_letkf_core_data", "letkf_core", "test_letkf_core"]
