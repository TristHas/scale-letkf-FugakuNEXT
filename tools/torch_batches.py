"""Torch-ready batch containers for the LETKF pipeline.

This module keeps the existing NumPy reference implementations untouched while
providing a clean set of dataclasses that mirror their inputs/outputs in a
batched, padded form that is convenient for PyTorch vectorisation.

Only new files are added so that legacy Python modules stay unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch

from .letkf_core import LetkfCoreInputs
from .postproc import PostprocInputs
from .params import LETKF_CONSTANTS


MEMBER = int(LETKF_CONSTANTS["MEMBER"])


@dataclass
class CoreBatchInputs:
    """Padded batch of obs-local outputs consumed by the LETKF core."""

    hdxf: torch.Tensor  # (batch, max_obs, MEMBER)
    dep: torch.Tensor  # (batch, max_obs)
    rdiag: torch.Tensor  # (batch, max_obs)
    rloc: torch.Tensor  # (batch, max_obs)
    obs_mask: torch.Tensor  # (batch, max_obs) boolean
    parm_infl: torch.Tensor  # (batch,)
    rdiag_wloc: torch.Tensor  # (batch,) boolean
    infl_update: torch.Tensor  # (batch,) boolean
    call_ids: torch.Tensor  # (batch,) int64 for bookkeeping

    @property
    def batch_size(self) -> int:
        return int(self.hdxf.shape[0])

    @property
    def max_obs(self) -> int:
        return int(self.hdxf.shape[1])


@dataclass
class CoreBatchOutputs:
    """Batched LETKF core outputs, ready for post-processing."""

    trans: torch.Tensor  # (batch, MEMBER, MEMBER)
    transm: torch.Tensor  # (batch, MEMBER)
    pa: torch.Tensor  # (batch, MEMBER, MEMBER)
    parm_infl: torch.Tensor  # (batch,)


@dataclass
class PostprocBatchInputs:
    """Inputs required by the relaxation/post-processing step."""

    trans: torch.Tensor  # (batch, MEMBER, MEMBER)
    transm: torch.Tensor  # (batch, MEMBER)
    gues_members: torch.Tensor  # (batch, MEMBER)
    gues_mean: torch.Tensor  # (batch,)
    beta: torch.Tensor  # (batch,)
    parm: torch.Tensor  # (batch,)
    relax_alpha: torch.Tensor  # (batch,)
    relax_alpha_spread: torch.Tensor  # (batch,)
    relax_to_inflated: torch.Tensor  # (batch,) boolean
    relax_spread_out: torch.Tensor  # (batch,) boolean
    det_run: torch.Tensor  # (batch,) boolean
    nvar: torch.Tensor  # (batch,) int64
    call_ids: torch.Tensor  # (batch,) int64


@dataclass
class PostprocBatchOutputs:
    """Outputs of the torch post-processing step."""

    anal_members: torch.Tensor  # (batch, MEMBER)
    transrlx: torch.Tensor  # (batch, MEMBER, MEMBER)
    q_mean: torch.Tensor  # (batch,)
    q_sprd: torch.Tensor  # (batch,)
    q_limited: torch.Tensor  # (batch,) boolean
    workda_value: torch.Tensor  # (batch,)
    workda_present: torch.Tensor  # (batch,) boolean


def _as_float_tensor(array: np.ndarray, *, device: torch.device | None = None) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(array, dtype=np.float64))
    return tensor.to(device=device)


def _as_bool_tensor(value: Iterable[bool], *, device: torch.device | None = None) -> torch.Tensor:
    tensor = torch.tensor(list(value), dtype=torch.bool)
    return tensor.to(device=device)


def build_core_batch_from_inputs(
    inputs: Sequence[LetkfCoreInputs],
    *,
    device: torch.device | None = None,
) -> CoreBatchInputs:
    """Stack a list of ``LetkfCoreInputs`` into padded torch tensors."""

    if not inputs:
        raise ValueError("inputs must contain at least one LetkfCoreInputs instance")

    nobsl_values = [
        int(data.before_meta.get("nobsl", data.before_arrays["hdxf"].shape[0]))
        for data in inputs
    ]
    max_obs = max(nobsl_values)
    batch = len(inputs)

    hdxf = torch.zeros((batch, max_obs, MEMBER), dtype=torch.float64, device=device)
    dep = torch.zeros((batch, max_obs), dtype=torch.float64, device=device)
    rdiag = torch.zeros((batch, max_obs), dtype=torch.float64, device=device)
    rloc = torch.zeros((batch, max_obs), dtype=torch.float64, device=device)
    obs_mask = torch.zeros((batch, max_obs), dtype=torch.bool, device=device)

    parm_infl: List[float] = []
    rdiag_wloc: List[bool] = []
    infl_update: List[bool] = []
    call_ids: List[int] = []

    for idx, data in enumerate(inputs):
        nobsl = nobsl_values[idx]
        slice_obj = slice(0, nobsl)
        hdxf_np = np.asarray(data.before_arrays["hdxf"], dtype=np.float64)
        dep_np = np.asarray(data.before_arrays["dep"], dtype=np.float64)
        rdiag_np = np.asarray(data.before_arrays["rdiag"], dtype=np.float64)
        rloc_np = np.asarray(data.before_arrays["rloc"], dtype=np.float64)

        hdxf[idx, slice_obj, :] = _as_float_tensor(hdxf_np[:nobsl, :], device=device)
        dep[idx, slice_obj] = _as_float_tensor(dep_np[:nobsl], device=device)
        rdiag[idx, slice_obj] = _as_float_tensor(rdiag_np[:nobsl], device=device)
        rloc[idx, slice_obj] = _as_float_tensor(rloc_np[:nobsl], device=device)
        obs_mask[idx, slice_obj] = True

        meta = data.before_meta
        parm_infl.append(float(meta.get("parm_infl", 1.0)))
        rdiag_wloc.append(_meta_bool(meta, "rdiag_wloc", default=False))
        infl_update.append(_meta_bool(meta, "infl_update", default=False))
        call_ids.append(int(meta.get("call_id", -1)))

    return CoreBatchInputs(
        hdxf=hdxf,
        dep=dep,
        rdiag=rdiag,
        rloc=rloc,
        obs_mask=obs_mask,
        parm_infl=torch.tensor(parm_infl, dtype=torch.float64, device=device),
        rdiag_wloc=_as_bool_tensor(rdiag_wloc, device=device),
        infl_update=_as_bool_tensor(infl_update, device=device),
        call_ids=torch.tensor(call_ids, dtype=torch.int64, device=device),
    )


def build_postproc_batch_from_inputs(
    inputs: Sequence[PostprocInputs],
    *,
    device: torch.device | None = None,
) -> PostprocBatchInputs:
    """Stack ``PostprocInputs`` from the NumPy pipeline into torch tensors."""

    if not inputs:
        raise ValueError("inputs must contain at least one PostprocInputs instance")

    batch = len(inputs)
    trans = torch.zeros((batch, MEMBER, MEMBER), dtype=torch.float64, device=device)
    transm = torch.zeros((batch, MEMBER), dtype=torch.float64, device=device)
    gues_members = torch.zeros((batch, MEMBER), dtype=torch.float64, device=device)
    gues_mean: List[float] = []
    beta: List[float] = []
    parm: List[float] = []
    relax_alpha: List[float] = []
    relax_alpha_spread: List[float] = []
    relax_to_inflated: List[bool] = []
    relax_spread_out: List[bool] = []
    det_run: List[bool] = []
    nvar: List[int] = []
    call_ids: List[int] = []

    for idx, data in enumerate(inputs):
        arr_trans = np.asarray(data.before_arrays["trans"], dtype=np.float64)
        arr_transm = np.asarray(data.before_arrays["transm"], dtype=np.float64)
        arr_gues = np.asarray(data.before_arrays["gues_members"], dtype=np.float64)
        if arr_gues.ndim == 1:
            arr_gues = arr_gues[:MEMBER]
        else:
            arr_gues = arr_gues[:MEMBER]

        trans[idx] = _as_float_tensor(arr_trans, device=device)
        transm[idx] = _as_float_tensor(arr_transm, device=device)
        gues_members[idx] = _as_float_tensor(arr_gues, device=device)

        meta = data.before_meta
        gues_mean.append(float(meta.get("gues_mean", 0.0)))
        beta.append(float(meta.get("beta", 1.0)))
        parm.append(float(meta.get("parm", 1.0)))
        relax_alpha.append(float(meta.get("relax_alpha", 0.0)))
        relax_alpha_spread.append(float(meta.get("relax_alpha_spread", 0.0)))
        relax_to_inflated.append(_meta_bool(meta, "relax_to_inflated_prior", False))
        relax_spread_out.append(_meta_bool(meta, "relax_spread_out", False))
        det_run.append(_meta_bool(meta, "det_run", False))
        nvar.append(int(meta.get("nvar", -1)))
        call_ids.append(int(meta.get("call_id", -1)))

    return PostprocBatchInputs(
        trans=trans,
        transm=transm,
        gues_members=gues_members,
        gues_mean=torch.tensor(gues_mean, dtype=torch.float64, device=device),
        beta=torch.tensor(beta, dtype=torch.float64, device=device),
        parm=torch.tensor(parm, dtype=torch.float64, device=device),
        relax_alpha=torch.tensor(relax_alpha, dtype=torch.float64, device=device),
        relax_alpha_spread=torch.tensor(relax_alpha_spread, dtype=torch.float64, device=device),
        relax_to_inflated=_as_bool_tensor(relax_to_inflated, device=device),
        relax_spread_out=_as_bool_tensor(relax_spread_out, device=device),
        det_run=_as_bool_tensor(det_run, device=device),
        nvar=torch.tensor(nvar, dtype=torch.int64, device=device),
        call_ids=torch.tensor(call_ids, dtype=torch.int64, device=device),
    )


def _meta_bool(meta: dict, key: str, default: bool = False) -> bool:
    value = meta.get(key)
    if value is None:
        return default
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", ".true.", "1", "yes"}:
            return True
        if token in {"false", ".false.", "0", "no"}:
            return False
    return bool(value)


__all__ = [
    "CoreBatchInputs",
    "CoreBatchOutputs",
    "PostprocBatchInputs",
    "PostprocBatchOutputs",
    "build_core_batch_from_inputs",
    "build_postproc_batch_from_inputs",
]
