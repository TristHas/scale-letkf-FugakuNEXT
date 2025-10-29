from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

from .letkf_dump_loader import LetkfDump, ObsdaBundle


@dataclass
class LetkfAnalysis:
    """Analysis ensemble and associated weights from LETKF."""

    members_3d: Dict[str, np.ndarray]
    members_2d: Dict[str, np.ndarray]
    mean_weights: np.ndarray
    cov_sqrt_weights: np.ndarray
    mem_order: Tuple[str, ...]


def _stack_state(dump: LetkfDump) -> Tuple[np.ndarray, Tuple[str, ...], Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]]]:
    """Flatten ensemble state fields into column-stacked matrix."""

    if "state_3d" not in dump.guess:
        raise KeyError("Background dataset lacks 'state_3d' variable")

    state3d = dump.guess["state_3d"]
    mem_order = tuple(str(m) for m in state3d.coords["member"].values)
    if len(mem_order) < 2:
        raise ValueError("LETKF requires at least two ensemble members")

    has_2d = "state_2d" in dump.guess.data_vars
    state2d = dump.guess["state_2d"] if has_2d else None

    state_vectors = []
    shapes: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]] = {}

    for mem in mem_order:
        arr3d = np.asarray(state3d.sel(member=mem).values, dtype=np.float64)
        shape3d = arr3d.shape
        parts = [arr3d.reshape(-1, order="F")]

        if has_2d and state2d is not None:
            arr2d = np.asarray(state2d.sel(member=mem).values, dtype=np.float64)
            shape2d = arr2d.shape
            parts.append(arr2d.reshape(-1, order="F"))
        else:
            shape2d = tuple()

        state_vectors.append(np.concatenate(parts))
        shapes[mem] = (shape3d, shape2d)

    state_matrix = np.column_stack(state_vectors)
    return state_matrix, mem_order, shapes




def _stack_observations(dump: LetkfDump, mem_order: Tuple[str, ...]) -> Tuple[np.ndarray, np.ndarray]:
    """Assemble observation innovations and ensemble anomalies."""

    if not dump.obsda:
        raise ValueError("Dump does not contain observation-space data")

    innov_list = []
    obs_cols = {mem: [] for mem in mem_order}

    for suffix in sorted(dump.obsda.keys()):
        bundle: ObsdaBundle = dump.obsda[suffix]
        ds = bundle.dataset
        innov_list.append(np.asarray(ds["innovation"].values, dtype=np.float64))

        if "hx_anomaly" not in ds.data_vars:
            raise ValueError(f"Observation bundle {suffix} lacks ensemble anomalies")
        hx_da = ds["hx_anomaly"].sel(member=list(mem_order))
        hx_vals = np.asarray(hx_da.values, dtype=np.float64)
        for idx, mem in enumerate(mem_order):
            obs_cols[mem].append(hx_vals[idx, :])

    innov = np.concatenate(innov_list)
    y_matrix = np.column_stack([np.concatenate(obs_cols[mem]) for mem in mem_order])
    return innov, y_matrix




def _prepare_obs_error(observation_error: float | np.ndarray, n_obs: int) -> np.ndarray:
    """Broadcast scalar/array observation error variance to 1D array."""

    if np.isscalar(observation_error):
        obs_var = np.full(n_obs, float(observation_error), dtype=np.float64)
    else:
        obs_var = np.asarray(observation_error, dtype=np.float64)
        if obs_var.shape != (n_obs,):
            raise ValueError(
                f"Observation error variance must have shape ({n_obs},), got {obs_var.shape}"
            )
    if np.any(obs_var <= 0.0):
        raise ValueError("Observation error variances must be positive")
    return obs_var


def _symmetric_matrix_sqrt(mat: np.ndarray) -> np.ndarray:
    """Return symmetric matrix square root using eigen-decomposition."""

    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, 0.0, None)
    sqrt_vals = np.sqrt(vals)
    return (vecs * sqrt_vals) @ vecs.T


def letkf_das(
    dump: LetkfDump,
    observation_error_variance: float | np.ndarray = 1.0,
    ridge: float = 1e-9,
    inflation: float = 1.0,
) -> LetkfAnalysis:
    """Perform a simplified LETKF analysis using loaded dump data.

    Parameters
    ----------
    dump:
        LETKF dump container returned by :func:`load_letkf_dump`.
    observation_error_variance:
        Scalar or array of length ``n_obs`` containing diagonal elements of the
        observation error covariance matrix ``R``.  If a scalar is supplied, it
        is broadcast to all observations.
    ridge:
        Small positive value added to the diagonal of intermediate matrices to
        improve numerical stability during inversion.
    inflation:
        Optional multiplicative inflation factor applied to the background
        perturbations prior to the analysis.
    """

    xb_matrix, mem_order, shapes = _stack_state(dump)
    innov, y_matrix = _stack_observations(dump, mem_order)

    n_obs = innov.size
    obs_var = _prepare_obs_error(observation_error_variance, n_obs)
    r_inv = 1.0 / obs_var

    k = xb_matrix.shape[1]
    xb_mean = xb_matrix.mean(axis=1, keepdims=True)
    xb_pert = xb_matrix - xb_mean
    if inflation != 1.0:
        if inflation <= 0.0:
            raise ValueError("inflation must be positive")
        xb_pert = xb_pert * float(inflation) ** 0.5

    # Observation-space anomalies (already mean removed in dump)
    a_matrix = y_matrix

    # Compute analysis weights following Hunt et al. (2007)
    eye_k = np.eye(k)
    gain_matrix = (k - 1) * eye_k + a_matrix.T @ (r_inv[:, None] * a_matrix)
    gain_matrix += ridge * eye_k
    gain_inv = np.linalg.inv(gain_matrix)

    weight_mean = gain_inv @ (a_matrix.T @ (r_inv * innov))
    weight_sqrt = _symmetric_matrix_sqrt((k - 1) * gain_inv)

    xa_pert = xb_pert @ weight_sqrt
    xa_mean = xb_mean + xb_pert @ weight_mean[:, None]
    xa_matrix = xa_pert + xa_mean

    analysis_3d: Dict[str, np.ndarray] = {}
    analysis_2d: Dict[str, np.ndarray] = {}

    for col_idx, mem in enumerate(mem_order):
        vector = xa_matrix[:, col_idx]
        shape3d, shape2d = shapes[mem]
        size3d = int(np.prod(shape3d))
        offset = 0
        analysis_3d[mem] = vector[offset : offset + size3d].reshape(shape3d, order="F")
        offset += size3d
        if shape2d:
            size2d = int(np.prod(shape2d))
            analysis_2d[mem] = vector[offset : offset + size2d].reshape(shape2d, order="F")

    return LetkfAnalysis(
        members_3d=analysis_3d,
        members_2d=analysis_2d,
        mean_weights=weight_mean,
        cov_sqrt_weights=weight_sqrt,
        mem_order=mem_order,
    )


__all__ = ["LetkfAnalysis", "letkf_das"]
