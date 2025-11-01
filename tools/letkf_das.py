from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from .letkf_dump_loader import LetkfDump, ObsdaBundle, StateBundle


@dataclass
class LetkfAnalysis:
    """
    Analysis ensemble and associated weights from LETKF.

    Members are organised as nested dictionaries: ``members_3d[member_id][rank_suffix]``.
    """

    members_3d: Dict[str, Dict[str, np.ndarray]]
    members_2d: Dict[str, Dict[str, np.ndarray]]
    mean_weights: np.ndarray
    cov_sqrt_weights: np.ndarray
    mem_order: Tuple[str, ...]


def _resolve_dataset(entry: Any) -> Tuple[str, Any]:
    """Return (suffix, dataset-like) pair from stored state entries."""

    if isinstance(entry, StateBundle):
        return entry.rank, entry.dataset

    # Fallback for older structures where the value is already an xarray Dataset
    if hasattr(entry, "data_vars"):
        suffix = getattr(entry, "attrs", {}).get("rank", "")
        return str(suffix), entry

    raise TypeError("Unsupported state bundle entry")


def _select_var(dataset: Any, base_name: str) -> str:
    """Find a variable in dataset by exact name or suffix match."""

    data_vars = getattr(dataset, "data_vars", {})
    if base_name in data_vars:
        return base_name
    for name in data_vars:
        if name.endswith(base_name):
            return name
    raise KeyError(f"Background dataset lacks '{base_name}' variable")


def _stack_state(
    dump: LetkfDump,
) -> Tuple[np.ndarray, Tuple[str, ...], Dict[str, List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]]]]:
    """Flatten ensemble state fields into column-stacked matrix."""

    if not dump.guess:
        raise ValueError("Dump does not contain background state bundles")

    state_items_raw = sorted(dump.guess.items())
    suffix0, dataset0 = _resolve_dataset(state_items_raw[0][1])
    key_state3d = _select_var(dataset0, "state_3d")

    mem_order = tuple(str(m) for m in dataset0.coords["member"].values)
    if len(mem_order) < 2:
        raise ValueError("LETKF requires at least two ensemble members")

    state_vectors: Dict[str, List[np.ndarray]] = {mem: [] for mem in mem_order}
    shapes: Dict[str, List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]]] = {
        mem: [] for mem in mem_order
    }

    for suffix, raw_entry in state_items_raw:
        rank_suffix, dataset = _resolve_dataset(raw_entry)
        bundle_mem_order = tuple(str(m) for m in dataset.coords["member"].values)
        if bundle_mem_order != mem_order:
            raise ValueError(
                f"Member ordering mismatch in bundle {rank_suffix or suffix}: {bundle_mem_order} != {mem_order}"
            )

        state_key = _select_var(dataset, "state_3d")
        has_2d = False
        state2d_key = None
        try:
            state2d_key = _select_var(dataset, "state_2d")
            has_2d = True
        except KeyError:
            has_2d = False

        for mem in mem_order:
            arr3d = np.asarray(dataset[state_key].sel(member=mem).values, dtype=np.float64)
            shape3d = arr3d.shape
            parts = [arr3d.reshape(-1, order="F")]

            if has_2d:
                arr2d = np.asarray(dataset[state2d_key].sel(member=mem).values, dtype=np.float64)
                shape2d = arr2d.shape
                parts.append(arr2d.reshape(-1, order="F"))
            else:
                shape2d = tuple()

            state_vectors[mem].append(np.concatenate(parts))
            shapes[mem].append((rank_suffix or suffix, shape3d, shape2d))

    state_matrix = np.column_stack([np.concatenate(state_vectors[mem]) for mem in mem_order])
    return state_matrix, mem_order, shapes


def _normalise_member_label(value: Any, width: int) -> str:
    """Normalise various member identifiers to zero-padded strings."""

    if isinstance(value, (bytes, np.bytes_)):
        text = value.decode("utf-8").strip()
    else:
        text = str(value).strip()

    if text.isdigit():
        return f"{int(text):0{width}d}"
    # handle values like '0001 ' or 'mem0001'
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        return f"{int(digits):0{width}d}"
    return text


def _stack_observations(dump: LetkfDump, mem_order: Tuple[str, ...]) -> Tuple[np.ndarray, np.ndarray]:
    """Assemble observation innovations and ensemble anomalies."""

    if not dump.obsda:
        raise ValueError("Dump does not contain observation-space data")

    innov_list: List[np.ndarray] = []
    obs_cols: Dict[str, List[np.ndarray]] = {mem: [] for mem in mem_order}

    member_width = max(len(mem) for mem in mem_order)
    normalised_order = [mem.zfill(member_width) for mem in mem_order]

    for suffix in sorted(dump.obsda.keys()):
        bundle: ObsdaBundle = dump.obsda[suffix]
        ds = bundle.dataset
        innov_list.append(np.asarray(ds["innovation"].values, dtype=np.float64))

        if "hx_anomaly" not in ds.data_vars:
            raise ValueError(f"Observation bundle {suffix} lacks ensemble anomalies")

        hx_da = ds["hx_anomaly"]
        hx_vals = np.asarray(hx_da.values, dtype=np.float64)

        if hx_da.dims[0] != "member" and "member" in hx_da.coords:
            hx_da = hx_da.rename({hx_da.dims[0]: "member"})
            hx_vals = np.asarray(hx_da.values, dtype=np.float64)

        if "member" in hx_da.coords:
            hx_labels = [
                _normalise_member_label(val, member_width) for val in hx_da.coords["member"].values
            ]
        else:
            hx_labels = [f"{idx+1:0{member_width}d}" for idx in range(hx_vals.shape[0])]

        label_to_index: Dict[str, int] = {}
        for idx, label in enumerate(hx_labels):
            label_to_index[label] = idx

        for mem, norm_label in zip(mem_order, normalised_order):
            label = norm_label
            if label not in label_to_index:
                alt_label = _normalise_member_label(mem, member_width)
                if alt_label in label_to_index:
                    label = alt_label
                elif "mean" in label_to_index:
                    # Use mean anomaly as a crude fallback when individual perturbations are missing.
                    label = "mean"
                else:
                    obs_cols[mem].append(np.zeros(hx_vals.shape[1], dtype=np.float64))
                    continue
            obs_cols[mem].append(hx_vals[label_to_index[label], :])

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
    """Perform a simplified LETKF analysis using loaded dump data."""

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

    a_matrix = y_matrix

    eye_k = np.eye(k)
    gain_matrix = (k - 1) * eye_k + a_matrix.T @ (r_inv[:, None] * a_matrix)
    gain_matrix += ridge * eye_k
    gain_inv = np.linalg.inv(gain_matrix)

    weight_mean = gain_inv @ (a_matrix.T @ (r_inv * innov))
    weight_sqrt = _symmetric_matrix_sqrt((k - 1) * gain_inv)

    xa_pert = xb_pert @ weight_sqrt
    xa_mean = xb_mean + xb_pert @ weight_mean[:, None]
    xa_matrix = xa_pert + xa_mean

    analysis_3d: Dict[str, Dict[str, np.ndarray]] = {mem: {} for mem in mem_order}
    analysis_2d: Dict[str, Dict[str, np.ndarray]] = {mem: {} for mem in mem_order}

    for col_idx, mem in enumerate(mem_order):
        vector = xa_matrix[:, col_idx]
        offset = 0
        for suffix, shape3d, shape2d in shapes[mem]:
            size3d = int(np.prod(shape3d, dtype=np.int64))
            block3d = vector[offset : offset + size3d].reshape(shape3d, order="F")
            analysis_3d[mem][suffix] = block3d
            offset += size3d
            if shape2d:
                size2d = int(np.prod(shape2d, dtype=np.int64))
                block2d = vector[offset : offset + size2d].reshape(shape2d, order="F")
                analysis_2d[mem][suffix] = block2d
                offset += size2d

    return LetkfAnalysis(
        members_3d=analysis_3d,
        members_2d=analysis_2d,
        mean_weights=weight_mean,
        cov_sqrt_weights=weight_sqrt,
        mem_order=mem_order,
    )


__all__ = ["LetkfAnalysis", "letkf_das"]
