from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np
import torch
import xarray as xr

from .letkf_dump_loader import LetkfDump, ObsdaBundle, load_letkf_dump


@dataclass
class PreparedObservationSet:
    """Container for observation-space data formatted for LETKF."""

    hx_anom: torch.Tensor  # shape: (n_obs, n_members)
    innov: torch.Tensor  # shape: (n_obs,)
    obs_var: torch.Tensor  # shape: (n_obs,)


def flatten_state_ensemble(
    state: xr.DataArray,
    *,
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, Sequence[str]]:
    """Flatten an ensemble state DataArray into a 2-D tensor.

    Parameters
    ----------
    state
        xarray DataArray containing a ``member`` dimension and arbitrary
        feature dimensions describing the model state.
    device
        Optional torch device that the resulting tensors should be moved to.

    Returns
    -------
    xb
        Background ensemble as a tensor of shape ``(n_state, n_members)``.
    mean
        Ensemble mean with shape ``(n_state,)``.
    member_order
        Sequence of member identifiers in the order represented along the
        second axis of ``xb``.
    """
    if not isinstance(state, xr.DataArray):
        raise TypeError("state must be an xarray.DataArray")
    if "member" not in state.dims:
        raise ValueError("state must contain a 'member' dimension")

    feature_dims = [dim for dim in state.dims if dim != "member"]
    if not feature_dims:
        raise ValueError("state must have at least one non-member dimension")

    stacked = state.stack(state_index=feature_dims)
    matrix = stacked.transpose("state_index", "member")

    xb = torch.from_numpy(np.asarray(matrix.values, dtype=np.float64)).to(
        device=device, dtype=torch.float64
    )
    mean = xb.mean(dim=1)
    member_order = [str(v) for v in matrix.coords["member"].values]
    return xb, mean, member_order




def prepare_observations(
    bundle: ObsdaBundle,
    member_order: Sequence[str],
    default_obs_error: float = 1.0,
    device: torch.device | None = None,
) -> PreparedObservationSet:
    """Convert an :class:`ObsdaBundle` into tensors for LETKF computations."""
    ds = bundle.dataset
    n_obs = bundle.observation_count
    if n_obs == 0:
        raise ValueError("Observation bundle is empty")

    innov = torch.from_numpy(np.asarray(ds["innovation"].values, dtype=np.float64)).to(
        device=device, dtype=torch.float64
    )

    member_labels = [str(m) for m in member_order]
    hx_da = ds["hx_anomaly"] if "hx_anomaly" in ds.data_vars else None
    if hx_da is None:
        hx_anom = torch.zeros((n_obs, len(member_labels)), dtype=torch.float64, device=device)
    else:
        available = [str(v) for v in hx_da.coords["member"].values]
        missing = [mem for mem in member_labels if mem not in available]
        if missing:
            raise KeyError(f"Observation bundle missing ensemble anomalies for {missing}")
        hx_aligned = hx_da.sel(member=member_labels).transpose("observation", "member")
        hx_anom = torch.from_numpy(np.asarray(hx_aligned.values, dtype=np.float64)).to(
            device=device, dtype=torch.float64
        )

    obs_var = torch.full(
        (n_obs,),
        float(default_obs_error),
        dtype=torch.float64,
        device=device,
    )
    return PreparedObservationSet(hx_anom=hx_anom, innov=innov, obs_var=obs_var)




def das_letkf(
    xb: torch.Tensor,
    hx_anom: torch.Tensor,
    innov: torch.Tensor,
    obs_var: torch.Tensor,
    inflation: float = 1.0,
) -> torch.Tensor:
    """Perform a single global LETKF analysis using PyTorch."""
    if xb.ndim != 2:
        raise ValueError("xb must be 2-D (n_state, n_members)")
    if hx_anom.ndim != 2:
        raise ValueError("hx_anom must be 2-D (n_obs, n_members)")
    if innov.ndim != 1 or obs_var.ndim != 1:
        raise ValueError("innov and obs_var must be 1-D")

    n_state, n_members = xb.shape
    n_obs = hx_anom.shape[0]

    if hx_anom.shape[1] != n_members:
        raise ValueError("hx_anom member dimension mismatch with xb")
    if innov.shape[0] != n_obs or obs_var.shape[0] != n_obs:
        raise ValueError("Observation vectors must align along the first dimension")

    if n_members < 2:
        raise ValueError("LETKF requires at least two ensemble members")

    xb_mean = xb.mean(dim=1, keepdim=True)
    xb_pert = xb - xb_mean

    obs_var = obs_var.clone()
    if torch.any(obs_var <= 0):
        raise ValueError("Observation variances must be positive")

    weight_matrix = hx_anom / obs_var.sqrt().unsqueeze(1)
    innov_weighted = innov / obs_var.sqrt()

    m_eye = torch.eye(n_members, dtype=xb.dtype, device=xb.device)
    gram = weight_matrix.T @ weight_matrix
    gram = gram + (n_members - 1) * m_eye

    rhs = weight_matrix.T @ innov_weighted
    mean_weights = torch.linalg.solve(gram, rhs)

    eigvals, eigvecs = torch.linalg.eigh(gram)
    eigvals = torch.clamp(eigvals, min=1e-12)
    gram_inv_sqrt = eigvecs @ torch.diag(eigvals.rsqrt()) @ eigvecs.T
    pert_transform = gram_inv_sqrt * (n_members - 1) ** 0.5

    if inflation != 1.0:
        if inflation <= 0:
            raise ValueError("inflation must be positive")
        pert_transform = pert_transform * inflation ** 0.5

    xa_pert = xb_pert @ pert_transform
    xa_mean = xb_mean + xb_pert @ mean_weights.unsqueeze(1)

    return xa_pert + xa_mean


def load_and_prepare(
    dump_dir: str,
    obs_suffix: str,
    device: torch.device | None = None,
    default_obs_error: float = 1.0,
) -> Tuple[torch.Tensor, PreparedObservationSet]:
    """Load a LETKF dump and prepare tensors for assimilation experiments."""
    dump = load_letkf_dump(dump_dir)
    if "state_3d" not in dump.guess:
        raise KeyError("Background dataset does not include 'state_3d'")

    xb, _, member_order = flatten_state_ensemble(
        dump.guess["state_3d"], device=device
    )

    if obs_suffix not in dump.obsda:
        available = ", ".join(sorted(dump.obsda.keys())) or "<none>"
        raise KeyError(
            f"Observation suffix '{obs_suffix}' not found in dump (available: {available})"
        )

    obs_bundle = dump.obsda[obs_suffix]
    obs = prepare_observations(
        obs_bundle,
        member_order,
        default_obs_error=default_obs_error,
        device=device,
    )
    return xb, obs


__all__ = [
    "PreparedObservationSet",
    "flatten_state_ensemble",
    "prepare_observations",
    "das_letkf",
    "load_and_prepare",
]
