"""Torch implementation of the LETKF post-processing step."""

from __future__ import annotations

import torch

from .postproc import IV3D_Q
from .params import LETKF_CONSTANTS
from .torch_batches import PostprocBatchInputs, PostprocBatchOutputs

MEMBER = int(LETKF_CONSTANTS["MEMBER"])
Q_SPRD_MAX = float(LETKF_CONSTANTS.get("Q_SPRD_MAX", 0.0))
IDENTITY = None  # lazily initialised


def _eye(device: torch.device) -> torch.Tensor:
    global IDENTITY
    if IDENTITY is None or IDENTITY.device != device:
        IDENTITY = torch.eye(MEMBER, dtype=torch.float64, device=device)
    return IDENTITY


def postproc_torch(batch: PostprocBatchInputs) -> PostprocBatchOutputs:
    """Vectorised relaxation and member update."""

    device = batch.trans.device
    parm = torch.where(batch.relax_to_inflated, batch.parm, torch.ones_like(batch.parm))

    wrlx, infl_out = _apply_relaxation_torch(
        batch.trans,
        batch.trans,
        batch.gues_members,
        parm,
        batch.relax_alpha,
        batch.relax_alpha_spread,
    )

    beta = batch.beta
    transm_matrix = batch.transm.unsqueeze(-1).expand(-1, MEMBER, MEMBER)
    total = (wrlx + transm_matrix) * beta.view(-1, 1, 1)
    total = total + _eye(device).unsqueeze(0) * (1.0 - beta).view(-1, 1, 1)

    anal_members = batch.gues_mean.unsqueeze(-1) + torch.bmm(
        batch.gues_members.unsqueeze(1), total
    ).squeeze(1)

    q_mean_raw = anal_members.mean(dim=1)
    anomalies = anal_members - q_mean_raw.unsqueeze(-1)
    sprd_num = torch.sum(anomalies * anomalies, dim=1) / max(MEMBER - 1, 1)
    sprd = torch.sqrt(torch.clamp(sprd_num, min=0.0))
    safe_mean = torch.where(q_mean_raw > 0.0, q_mean_raw, torch.ones_like(q_mean_raw))
    sprd_ratio = torch.zeros_like(q_mean_raw)
    positive_mask = q_mean_raw > 0.0
    sprd_ratio[positive_mask] = sprd[positive_mask] / safe_mean[positive_mask]

    limit_mask = (
        (batch.nvar == IV3D_Q)
        & positive_mask
        & (Q_SPRD_MAX > 0.0)
        & (sprd_ratio > Q_SPRD_MAX)
    )
    if torch.any(limit_mask):
        ratio = torch.where(
            sprd_ratio > 0.0, sprd_ratio, torch.ones_like(sprd_ratio)
        )
        scale = torch.ones_like(q_mean_raw)
        scale[limit_mask] = Q_SPRD_MAX / ratio[limit_mask]
        anal_members = torch.where(
            limit_mask.unsqueeze(-1),
            q_mean_raw.unsqueeze(-1) + anomalies * scale.unsqueeze(-1),
            anal_members,
        )

    q_mean = torch.where(limit_mask, q_mean_raw, torch.zeros_like(q_mean_raw))
    q_sprd = torch.where(limit_mask, sprd_ratio, torch.zeros_like(sprd_ratio))
    workda_present = batch.relax_spread_out & (batch.relax_alpha_spread != 0.0)
    workda_value = torch.where(workda_present, infl_out, torch.zeros_like(infl_out))

    return PostprocBatchOutputs(
        anal_members=anal_members,
        transrlx=total,
        q_mean=q_mean,
        q_sprd=q_sprd,
        q_limited=limit_mask,
        workda_value=workda_value,
        workda_present=workda_present,
    )


def _apply_relaxation_torch(
    w: torch.Tensor,
    trans_raw: torch.Tensor,
    xb: torch.Tensor,
    parm: torch.Tensor,
    relax_alpha: torch.Tensor,
    relax_alpha_spread: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched equivalent of postproc._apply_relaxation."""

    device = w.device
    batch = w.shape[0]
    wrlx = w.clone()
    infl_out = torch.ones(batch, dtype=torch.float64, device=device)

    mask_alpha = relax_alpha != 0.0
    if torch.any(mask_alpha):
        w_sel = w[mask_alpha]
        factor = (1.0 - relax_alpha[mask_alpha]).view(-1, 1, 1)
        w_sel = w_sel * factor
        diag = (
            relax_alpha[mask_alpha]
            * torch.sqrt(torch.clamp(parm[mask_alpha], min=0.0))
        ).view(-1, 1, 1)
        w_sel = w_sel + _eye(device).unsqueeze(0) * diag
        wrlx[mask_alpha] = w_sel

    mask_spread = (~mask_alpha) & (relax_alpha_spread != 0.0)
    if torch.any(mask_spread):
        w_sel = w[mask_spread]
        xb_sel = xb[mask_spread]
        parm_sel = parm[mask_spread]
        ral = relax_alpha_spread[mask_spread]
        denom = float(max(MEMBER - 1, 1))
        pa = torch.bmm(w_sel, w_sel.transpose(1, 2)) / denom
        pa_xb = torch.bmm(pa, xb_sel.unsqueeze(-1)).squeeze(-1)
        var_g = torch.sum(xb_sel * xb_sel, dim=1)
        var_a = torch.sum(xb_sel * pa_xb, dim=1)
        valid = (var_g > 0.0) & (var_a > 0.0)
        infl_values = torch.ones_like(var_g)
        if torch.any(valid):
            term = torch.sqrt(
                torch.clamp(var_g[valid] * parm_sel[valid] / (var_a[valid] * denom), min=0.0)
            )
            infl_values[valid] = ral[valid] * term - ral[valid] + 1.0
        wrlx[mask_spread] = w_sel * infl_values.view(-1, 1, 1)
        infl_out[mask_spread] = infl_values

    return wrlx, infl_out


__all__ = ["postproc_torch"]
