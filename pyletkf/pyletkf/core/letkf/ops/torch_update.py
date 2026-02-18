from __future__ import annotations

import torch


def _as_tensor(value, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _prepare_param(
    value, length: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    tensor = _as_tensor(value, device=device, dtype=dtype)
    if tensor.ndim == 0:
        return tensor.expand(length)
    if tensor.shape[0] != length:
        raise ValueError(
            f"Parameter has incompatible length {tensor.shape[0]}, expected {length}"
        )
    return tensor


def letkf_update(
    xb: torch.Tensor,
    wa: torch.Tensor,
    Wa: torch.Tensor,
    alpha_pert=0.0,
    alpha_spread=0.0,
    beta=1.0,
    *,
    parm_infl: torch.Tensor | float | None = None,
    return_trans: bool = False,
):
    """
    Apply LETKF weight update to the background ensemble.

    xb: [cell, state_dim, ensemble] background ensemble members.
    wa: [cell, ensemble] mean weight vector (transm in SCALE dumps).
    Wa: [cell, ensemble, ensemble] ensemble transform (trans in SCALE dumps).
    """

    if xb.ndim != 3:
        raise ValueError(f"xb must be [cell, state_dim, ens], got shape {xb.shape}")

    device = xb.device
    dtype = xb.dtype
    n_cells, _, n_ens = xb.shape

    wa = _as_tensor(wa, device=device, dtype=dtype)
    Wa = _as_tensor(Wa, device=device, dtype=dtype)

    if wa.shape != (n_cells, n_ens):
        raise ValueError(f"wa must have shape {(n_cells, n_ens)}, got {wa.shape}")
    if Wa.shape != (n_cells, n_ens, n_ens):
        raise ValueError(f"Wa must have shape {(n_cells, n_ens, n_ens)}, got {Wa.shape}")

    alpha_pert_t = _prepare_param(
        alpha_pert, n_cells, device=device, dtype=dtype
    ).view(n_cells, 1, 1)
    alpha_spread_t = _prepare_param(
        alpha_spread, n_cells, device=device, dtype=dtype
    ).view(n_cells, 1, 1)
    beta_t = _prepare_param(beta, n_cells, device=device, dtype=dtype).view(
        n_cells, 1, 1
    )
    if parm_infl is None:
        parm = torch.ones(n_cells, device=device, dtype=dtype)
    else:
        parm = _prepare_param(parm_infl, n_cells, device=device, dtype=dtype)

    eye = torch.eye(n_ens, device=device, dtype=dtype).unsqueeze(0)
    trans = Wa.clone()

    if torch.any(alpha_pert_t != 0):
        trans = trans * (1.0 - alpha_pert_t)
        diag_scale = alpha_pert_t * torch.sqrt(torch.clamp(parm, min=0.0)).view(
            n_cells, 1, 1
        )
        trans = trans + eye * diag_scale

    if torch.any(alpha_spread_t != 0):
        raise NotImplementedError("RTPS relaxation requires analysis covariance input")

    total = trans + wa.unsqueeze(-1)
    total = total * beta_t + eye * (1.0 - beta_t)

    xb_mean = xb.mean(dim=-1, keepdim=True)
    xb_anom = xb - xb_mean
    xa = xb_mean + torch.bmm(xb_anom, total)

    if return_trans:
        return xa, total
    return xa


__all__ = ["letkf_update"]
