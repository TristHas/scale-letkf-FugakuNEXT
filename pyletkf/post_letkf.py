from __future__ import annotations

import numpy as np
import xarray as xr

import torch
import xtensor as xt

from .params import MEMBERS, Q_SPRD_MAX

IV3D_Q = 6  
MEMBER = len(MEMBERS)
IDENTITY = None

def create_config(letkf_var = ['U', 'V', 'W', 'T', 'P', 'QV', 'QC', 'QR', 'QI', 'QS', 'QG'], device="cpu"):
    ds = xr.Dataset(data_vars={}, coords={"variables":letkf_var})
    zeros = np.zeros(len(letkf_var))
    ds["beta"]=(("variables",), zeros)
    ds["relax_alpha"]=(("variables",), zeros)
    ds["relax_alpha_spread"]=(("variables",), zeros)
    ds["relax_to_inflated"]=(("variables",), zeros.astype(bool))
    ds["relax_spread_out"]=(("variables",), zeros.astype(bool))
    ds["det_run"]=(("variables",), zeros.astype(bool))
    ds["nvar"]=(("variables",), zeros.astype(int))
    return xt.Dataset.from_xarray(ds)
    
def _eye(device: torch.device) -> torch.Tensor:
    global IDENTITY
    if IDENTITY is None or IDENTITY.device != device:
        IDENTITY = torch.eye(MEMBER, dtype=torch.float64, device=device)
    return IDENTITY

def apply_relaxation(data, params) -> tuple[torch.Tensor, torch.Tensor]:
    """
    
    """
    w = data["trans"].values
    xb = data["gues_members"].values
    parm = torch.where(params["relax_to_inflated"].values, data["parm_infl"].values, 1)
    relax_alpha = params["relax_alpha"].values
    relax_alpha_spread = params["relax_alpha_spread"].values
    
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

def update_transform(data, params):
    wrlx, infl_out = apply_relaxation(data, params)
    beta = params["beta"].values
    transm_matrix = data["transm"].values.unsqueeze(-1).expand(-1, MEMBER, MEMBER)
    total = (wrlx + transm_matrix) * beta.view(-1, 1, 1)
    
    total = total + _eye(beta.device).unsqueeze(0) * (1.0 - beta).view(-1, 1, 1)
    return total, infl_out

def postprocess_analysis_members(anal_members, params):
    """
    """
    q_mean_raw = anal_members.mean(dim=1)
    anomalies  = anal_members - q_mean_raw.unsqueeze(-1)
    
    sprd_num = torch.sum(anomalies * anomalies, dim=1) / max(MEMBER - 1, 1)
    sprd = torch.sqrt(torch.clamp(sprd_num, min=0.0))
    
    safe_mean = torch.where(q_mean_raw > 0.0, q_mean_raw, 1)
    sprd_ratio = torch.zeros_like(q_mean_raw)
    positive_mask = q_mean_raw > 0.0
    sprd_ratio[positive_mask] = sprd[positive_mask] / safe_mean[positive_mask]

    limit_mask = (
        (params["nvar"].values == IV3D_Q)
        & positive_mask
        & (Q_SPRD_MAX > 0.0)
        & (sprd_ratio > Q_SPRD_MAX)
    )
    
    if torch.any(limit_mask):
        ratio = torch.where(
            sprd_ratio > 0.0, sprd_ratio, 1
        )
        scale = torch.ones_like(q_mean_raw)
        scale[limit_mask] = Q_SPRD_MAX / ratio[limit_mask]
        anal_members = torch.where(
            limit_mask.unsqueeze(-1),
            q_mean_raw.unsqueeze(-1) + anomalies * scale.unsqueeze(-1),
            anal_members,
        )

    q_mean = torch.where(limit_mask, q_mean_raw, 0)
    q_sprd = torch.where(limit_mask, sprd_ratio, 0)
    return anal_members, limit_mask, q_mean, q_sprd

def apply_analysis_update(data, params) -> PostprocBatchOutputs:
    """
        Vectorised relaxation and member update.
    """
    device = data["trans"].values.device
    # 1. Preprocess transm matrix
    transrlx, infl_out = update_transform(data, params)
    # 2. Apply transform
    anal_members = data["gues_mean"].values.unsqueeze(-1) +\
                   torch.bmm(data["gues_members"].values.unsqueeze(1), 
                             transrlx).squeeze(1)
    # 3. Updates 
    anal_members, limit_mask, q_mean, q_sprd = postprocess_analysis_members(anal_members, params)
    # 4. what else
    workda_present = params["relax_spread_out"].values &\
                    (params["relax_alpha_spread"].values != 0.0)
    workda_value = torch.where(workda_present, infl_out, 0)

    return dict(
        anal_members=anal_members,
        transrlx=transrlx,
        q_mean=q_mean,
        q_sprd=q_sprd,
        q_limited=limit_mask,
        workda_value=workda_value,
        workda_present=workda_present,
    )

__all__ = ["apply_analysis_update", "create_config"]
