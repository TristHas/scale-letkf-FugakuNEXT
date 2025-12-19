from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from .letkf_core import (
    LetkfCoreIdentifier,
    load_letkf_core_from_global,
    letkf_core,
)
from .obs_local import PRC_NUM_X, PRC_NUM_Y
from .load_das_letkf import (
    load_das_postproc_after,
    load_das_postproc_before,
)
from .load_dumps import _read_binary_array
from .params import DA_CONSTANTS, GRID_CONSTANTS, LETKF_CONSTANTS

MEMBER = int(LETKF_CONSTANTS["MEMBER"])
Q_SPRD_MAX = float(LETKF_CONSTANTS.get("Q_SPRD_MAX", 0.0))
IV3D_Q = 6  # matches common_scale::iv3d_q


@dataclass(frozen=True)
class PostprocIdentifier:
    dump_dir: Path | str
    call_id: int
    pe_tag: str = "pe000000"
    member: str = "mem0001"


@dataclass
class PostprocInputs:
    identifier: PostprocIdentifier
    before_meta: Mapping[str, Any]
    before_arrays: Mapping[str, np.ndarray]
    after_meta: Mapping[str, Any]
    after_arrays: Mapping[str, np.ndarray]


def load_postproc_data(identifier: PostprocIdentifier | Mapping[str, Any]) -> PostprocInputs:
    ident = _normalize_identifier(identifier)
    dump_dir = Path(ident.dump_dir)
    before = load_das_postproc_before(
        dump_dir,
        ident.call_id,
        pe_tag=ident.pe_tag,
        member=ident.member,
    )
    after = load_das_postproc_after(
        dump_dir,
        ident.call_id,
        pe_tag=ident.pe_tag,
        member=ident.member,
    )
    return PostprocInputs(
        identifier=ident,
        before_meta=before["meta"],
        before_arrays=before["data"],
        after_meta=after["meta"],
        after_arrays=after["data"],
    )


def load_postproc_from_global(identifier: PostprocIdentifier | Mapping[str, Any]) -> PostprocInputs:
    ident = _normalize_identifier(identifier)
    core_inputs = load_letkf_core_from_global(
        LetkfCoreIdentifier(
            dump_dir=ident.dump_dir,
            call_id=ident.call_id,
            pe_tag=ident.pe_tag,
            member=ident.member,
        )
    )
    core_outputs = letkf_core(core_inputs)
    meta_core = core_inputs.before_meta
    ij = int(meta_core["ij"])
    ilev = int(meta_core["ilev"])
    nvar = int(meta_core["nvar"])
    ri = float(meta_core["ri"])
    rj = float(meta_core["rj"])
    rz = float(meta_core["rz"])
    beta = _relax_beta(ri, rj, rz)
    g_mean, g_pert = _sample_background(
        ident.dump_dir,
        ident.pe_tag,
        ij - 1,
        ilev - 1,
        nvar - 1,
    )
    before_meta = {
        "stage": "postproc",
        "phase": "before",
        "kind": meta_core["kind"],
        "call_id": ident.call_id,
        "ij": ij,
        "ilev": ilev,
        "nvar": nvar,
        "n2nc": int(meta_core["n2nc"]),
        "n2n": int(meta_core["n2n"]),
        "beta": beta,
        "parm": float(core_outputs["parm_infl"]),
        "relax_alpha": LETKF_CONSTANTS["RELAX_ALPHA"],
        "relax_alpha_spread": LETKF_CONSTANTS["RELAX_ALPHA_SPREAD"],
        "relax_spread_out": str(LETKF_CONSTANTS["RELAX_SPREAD_OUT"]).lower(),
        "relax_to_inflated_prior": str(LETKF_CONSTANTS["RELAX_ALPHA"] > 0.0).lower(),
        "det_run": str(LETKF_CONSTANTS.get("DET_RUN", False)).lower(),
        "gues_mean": g_mean,
    }
    before_arrays = {
        "trans": np.asarray(core_outputs["trans"], dtype=np.float64),
        "transm": np.asarray(core_outputs["transm"], dtype=np.float64),
        "gues_members": np.asarray(g_pert, dtype=np.float64),
    }
    after = load_das_postproc_after(
        ident.dump_dir,
        ident.call_id,
        pe_tag=ident.pe_tag,
        member=ident.member,
    )
    return PostprocInputs(
        identifier=ident,
        before_meta=before_meta,
        before_arrays=before_arrays,
        after_meta=after["meta"],
        after_arrays=after["data"],
    )


def postproc(data: PostprocInputs) -> dict[str, Any]:
    meta = data.before_meta
    trans = np.asarray(data.before_arrays["trans"], dtype=np.float64)
    transm = np.asarray(data.before_arrays["transm"], dtype=np.float64)
    gues_members = np.asarray(data.before_arrays["gues_members"], dtype=np.float64)
    gues_mean = float(meta.get("gues_mean", 0.0))
    beta = float(meta.get("beta", 1.0))
    parm = float(meta.get("parm", 1.0))
    relax_to_inflated = _meta_bool(meta, "relax_to_inflated_prior", False)
    if not relax_to_inflated:
        parm = 1.0
    relax_alpha = float(meta.get("relax_alpha", 0.0))
    relax_alpha_spread = float(meta.get("relax_alpha_spread", 0.0))
    relax_spread_out = _meta_bool(meta, "relax_spread_out", False)
    det_run = _meta_bool(meta, "det_run", False)

    wrlx, infl_out = _apply_relaxation(
        trans,
        np.asarray(data.before_arrays["trans"], dtype=np.float64),
        gues_members,
        parm,
        relax_alpha,
        relax_alpha_spread,
    )

    workda_present = relax_spread_out and (relax_alpha_spread != 0.0)
    workda_value = infl_out if workda_present else 0.0

    total = (wrlx + transm[:, None]) * beta
    diag_idx = np.diag_indices_from(total)
    total[diag_idx] += (1.0 - beta)

    anal_members = gues_mean + gues_members @ total

    q_mean = 0.0
    q_sprd = 0.0
    q_limited = False
    nvar = int(meta.get("nvar", -1))
    if nvar == IV3D_Q and Q_SPRD_MAX > 0.0:
        q_mean = np.mean(anal_members)
        if q_mean > 0.0:
            anomalies = anal_members - q_mean
            q_sprd = np.sqrt(np.sum(anomalies ** 2) / max(MEMBER - 1, 1)) / q_mean
            if q_sprd > Q_SPRD_MAX:
                scale = Q_SPRD_MAX / q_sprd
                anal_members = q_mean + anomalies * scale
                q_limited = True

    result = {
        "transrlx": total,
        "anal_members": anal_members,
        "q_mean": q_mean if q_limited else 0.0,
        "q_sprd": q_sprd if q_limited else 0.0,
        "q_limited": q_limited,
        "workda_value": workda_value,
        "workda_present": workda_present,
        "anal_det": None,
    }

    if det_run and "anal_det" in data.after_arrays:
        # Missing deterministic inputs: need transmd and deterministic background state.
        result["anal_det"] = None

    return result


def test_postproc(identifier: PostprocIdentifier | Mapping[str, Any], *, atol: float = 1.0e-10, rtol: float = 1.0e-10) -> dict[str, float]:
    data = load_postproc_data(identifier)
    result = postproc(data)
    errors: dict[str, float] = {}

    for key in ("transrlx", "anal_members"):
        ref = np.asarray(data.after_arrays[key], dtype=np.float64)
        np.testing.assert_allclose(result[key], ref, atol=atol, rtol=rtol)
        errors[key] = float(np.max(np.abs(result[key] - ref)))

    meta_after = data.after_meta
    np.testing.assert_allclose(float(meta_after.get("beta", 1.0)), float(data.before_meta.get("beta", 1.0)))
    workda_ref = float(meta_after.get("workda_value", 0.0))
    np.testing.assert_allclose(result["workda_value"], workda_ref)
    errors["workda_value"] = abs(result["workda_value"] - workda_ref)
    errors["q_mean"] = abs(result["q_mean"] - float(meta_after.get("q_mean", 0.0)))
    errors["q_sprd"] = abs(result["q_sprd"] - float(meta_after.get("q_sprd", 0.0)))

    if _meta_bool(meta_after, "q_limited", False) != result["q_limited"]:
        raise AssertionError("q_limited mismatch")
    if _meta_bool(meta_after, "workda_present", False) != result["workda_present"]:
        raise AssertionError("workda_present mismatch")

    return errors


def test_postproc_from_global(
    identifier: PostprocIdentifier | Mapping[str, Any],
    *,
    atol: float = 1.0e-10,
    rtol: float = 1.0e-10,
) -> dict[str, float]:
    data = load_postproc_from_global(identifier)
    result = postproc(data)
    errors: dict[str, float] = {}

    for key in ("transrlx", "anal_members"):
        ref = np.asarray(data.after_arrays[key], dtype=np.float64)
        np.testing.assert_allclose(result[key], ref, atol=atol, rtol=rtol)
        errors[key] = float(np.max(np.abs(result[key] - ref)))

    meta_after = data.after_meta
    np.testing.assert_allclose(float(meta_after.get("beta", 1.0)), data.before_meta["beta"])
    workda_ref = float(meta_after.get("workda_value", 0.0))
    np.testing.assert_allclose(result["workda_value"], workda_ref)
    errors["workda_value"] = abs(result["workda_value"] - workda_ref)
    errors["q_mean"] = abs(result["q_mean"] - float(meta_after.get("q_mean", 0.0)))
    errors["q_sprd"] = abs(result["q_sprd"] - float(meta_after.get("q_sprd", 0.0)))

    if _meta_bool(meta_after, "q_limited", False) != result["q_limited"]:
        raise AssertionError("q_limited mismatch")
    if _meta_bool(meta_after, "workda_present", False) != result["workda_present"]:
        raise AssertionError("workda_present mismatch")

    return errors


def _apply_relaxation(
    w: np.ndarray,
    trans_raw: np.ndarray,
    xb: np.ndarray,
    parm: float,
    relax_alpha: float,
    relax_alpha_spread: float,
) -> tuple[np.ndarray, float]:
    if relax_alpha != 0.0:
        wrlx = (1.0 - relax_alpha) * w
        diag_idx = np.diag_indices_from(wrlx)
        wrlx[diag_idx] += relax_alpha * np.sqrt(parm)
        return wrlx, 1.0
    if relax_alpha_spread != 0.0:
        pa = (trans_raw @ trans_raw.T) / max(MEMBER - 1, 1)
        var_g = float(np.sum(xb * xb))
        var_a = float(xb @ (pa @ xb))
        if var_g > 0.0 and var_a > 0.0:
            infl_out = relax_alpha_spread * np.sqrt(var_g * parm / (var_a * max(MEMBER - 1, 1))) - relax_alpha_spread + 1.0
        else:
            infl_out = 1.0
        return w * infl_out, infl_out
    return w.copy(), 1.0


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


def _normalize_identifier(identifier: PostprocIdentifier | Mapping[str, Any]) -> PostprocIdentifier:
    if isinstance(identifier, PostprocIdentifier):
        return identifier
    dump_dir = identifier.get("dump_dir")
    call_id = identifier.get("call_id")
    if dump_dir is None or call_id is None:
        raise ValueError("identifier must contain dump_dir and call_id")
    pe_tag = identifier.get("pe_tag", "pe000000")
    member = identifier.get("member", "mem0001")
    return PostprocIdentifier(dump_dir=dump_dir, call_id=int(call_id), pe_tag=str(pe_tag), member=str(member))


_BACKGROUND_CACHE: Dict[tuple[str, str], np.ndarray] = {}


def _load_background_cube(dump_dir: Path, pe_tag: str) -> np.ndarray:
    key = (str(dump_dir), pe_tag)
    cached = _BACKGROUND_CACHE.get(key)
    if cached is not None:
        return cached
    path = dump_dir / "gues3d" / f"gues3d_{pe_tag}.mem0001.bin"
    arr = _read_binary_array(path, ">f8").astype(np.float64, copy=False)
    members = arr[:, :, :MEMBER, :]
    _BACKGROUND_CACHE[key] = members
    return members


def _sample_background(
    dump_dir: Path | str,
    pe_tag: str,
    ij_idx: int,
    ilev_idx: int,
    nvar_idx: int,
) -> Tuple[float, np.ndarray]:
    cube = _load_background_cube(Path(dump_dir), pe_tag)
    members = cube[ij_idx, ilev_idx, :MEMBER, nvar_idx]
    mean = float(np.mean(members, dtype=np.float64))
    perturb = members - mean
    return mean, perturb


def _relax_beta(ri: float, rj: float, rz: float) -> float:
    beta = _boundary_taper(ri, rj)
    if beta <= 0.0:
        return 0.0
    radar_only = bool(DA_CONSTANTS.get("RADAR_ONLY", False))
    if radar_only:
        zmax = float(DA_CONSTANTS.get("RADAR_ZMAX", 0.0))
        vert_local = float(DA_CONSTANTS.get("VERT_LOCAL_RADAR", 0.0))
        if rz > zmax + vert_local * float(LETKF_CONSTANTS["dist_zero_fac"]):
            return 0.0
    return beta


def _boundary_taper(ri: float, rj: float) -> float:
    buffer_width = float(DA_CONSTANTS.get("BOUNDARY_BUFFER_WIDTH", 0.0))
    if buffer_width <= 0.0:
        return 1.0

    dx = float(GRID_CONSTANTS["DX"])
    dy = float(GRID_CONSTANTS["DY"])
    nlon_total = int(GRID_CONSTANTS["nlon"]) * int(PRC_NUM_X)
    nlat_total = int(GRID_CONSTANTS["nlat"]) * int(PRC_NUM_Y)

    ihalo = int(DA_CONSTANTS.get("IHALO", 0))
    jhalo = int(DA_CONSTANTS.get("JHALO", 0))

    dist_x = min(max(ri - ihalo, 0.0), max(nlon_total + ihalo + 1 - ri, 0.0)) * dx
    dist_y = min(max(rj - jhalo, 0.0), max(nlat_total + jhalo + 1 - rj, 0.0)) * dy
    dist = min(dist_x, dist_y)
    if dist <= 0.0:
        return 0.0
    beta = dist / buffer_width
    return max(0.0, min(beta, 1.0))


__all__ = [
    "PostprocIdentifier",
    "PostprocInputs",
    "load_postproc_data",
    "load_postproc_from_global",
    "postproc",
    "test_postproc",
    "test_postproc_from_global",
]
