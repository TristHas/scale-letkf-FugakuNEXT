from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import xarray as xr

from .letkf_das import letkf_das
from .letkf_dump_loader import LetkfDump, load_letkf_dump
from .letkf_parameters import LetkfParameters, load_letkf_parameters


def _resolve_inflation(params: LetkfParameters, override: float | None) -> float:
    if override is not None:
        return float(override)
    infl = params.letkf.get("infl_mul", 1.0)
    try:
        infl_value = float(infl)
    except (TypeError, ValueError):
        infl_value = 1.0
    if infl_value <= 0.0:
        return 1.0
    return infl_value


def _resolve_obs_error(params: LetkfParameters, override: float | None) -> float:
    if override is not None:
        return float(override)
    candidates = [
        "obserr_radar_ref",
        "obserr_radar_vr",
        "obserr_t",
        "obserr_q",
        "obserr_u",
    ]
    errors = params.observation.errors
    for key in candidates:
        value = errors.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 1.0


def _build_analysis_array(analysis, reference: xr.DataArray, members: Sequence[str]) -> xr.DataArray:
    stack = np.stack([analysis.members_3d[mem] for mem in members], axis=0)
    coords = {
        "member": list(members),
        "level": reference.coords.get("level", np.arange(stack.shape[1])),
        "y": reference.coords.get("y", np.arange(stack.shape[2])),
        "x": reference.coords.get("x", np.arange(stack.shape[3])),
        "variable": reference.coords.get("variable", np.arange(stack.shape[4])),
    }
    return xr.DataArray(
        stack,
        dims=reference.dims,
        coords=coords,
        name="analysis_computed",
    )


def compare_analysis(
    dump: LetkfDump
    obs_suffix: str | None,
    inflation: float,
    obs_error: float,
    ridge: float,
) -> None:
    if obs_suffix is not None:
        if obs_suffix not in dump.obsda:
            available = ", ".join(sorted(dump.obsda)) or "<none>"
            raise KeyError(
                f"Observation suffix '{obs_suffix}' not found in dump (available: {available})"
            )
        obsda = {obs_suffix: dump.obsda[obs_suffix]}
    else:
        obsda = dump.obsda

    working_dump = LetkfDump(
        guess=dump.guess,
        analysis=dump.analysis,
        obsda=obsda,
        dump_dir=dump.dump_dir,
    )

    obs_variance = float(obs_error) ** 2
    analysis = letkf_das(
        working_dump,
        observation_error_variance=obs_variance,
        ridge=ridge,
        inflation=inflation,
    )

    if "state_3d" not in dump.analysis.data_vars:
        raise RuntimeError("Reference analysis state is not available in dump")

    reference = dump.analysis["state_3d"]
    ref_members = [str(m) for m in reference.coords["member"].values]
    members = [mem for mem in analysis.mem_order if mem in ref_members]
    if not members:
        raise RuntimeError("No common members between computed analysis and reference")

    computed = _build_analysis_array(analysis, reference.sel(member=members), members)
    ref_aligned = reference.sel(member=members)
    diff = computed - ref_aligned

    diff_sq = diff ** 2
    rmse_global = float(np.sqrt(diff_sq.mean().item()))
    max_abs_global = float(np.abs(diff).max().item())

    rmse_per_var = np.sqrt(diff_sq.mean(dim=("member", "level", "y", "x"))).to_series()
    rmse_per_mem = np.sqrt(diff_sq.mean(dim=("level", "y", "x", "variable"))).to_series()

    print("LETKF replay summary")
    print("======================")
    print(f"Dump directory: {dump.dump_dir}")
    print(f"Observation suffix: {obs_suffix or '<all>'}")
    print(f"Inflation factor: {inflation:.3f}")
    print(f"Observation error σ: {obs_error:.3f} (variance {obs_variance:.3f})")
    print(f"Ridge parameter: {ridge:.2e}")
    print(f"Global RMSE: {rmse_global:.6e}")
    print(f"Max |difference|: {max_abs_global:.6e}")
    print()
    print("RMSE per variable:")
    for var, value in rmse_per_var.items():
        print(f"  {var}: {float(value):.6e}")
    print()
    print("RMSE per member:")
    for mem, value in rmse_per_mem.items():
        print(f"  {mem}: {float(value):.6e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a LETKF dump and compare against the stored analysis."
    )
    parser.add_argument("dump_dir", type=Path, help="Path to letkf_dump directory")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Configuration file used for the LETKF cycle",
    )
    parser.add_argument(
        "--obs-suffix",
        type=str,
        default=None,
        help="Restrict comparison to a specific obs suffix (default: use all)",
    )
    parser.add_argument(
        "--inflation",
        type=float,
        default=None,
        help="Override multiplicative inflation factor",
    )
    parser.add_argument(
        "--obs-error",
        type=float,
        default=None,
        help="Override observation error standard deviation",
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=1e-9,
        help="Ridge regularisation added to weight matrix inversion",
    )

    args = parser.parse_args()

    dump = load_letkf_dump(args.dump_dir)
    params = load_letkf_parameters(args.config)

    inflation = _resolve_inflation(params, args.inflation)
    obs_error = _resolve_obs_error(params, args.obs_error)

    compare_analysis(
        dump=dump,
        obs_suffix=args.obs_suffix,
        inflation=inflation,
        obs_error=obs_error,
        ridge=float(args.ridge),
    )


if __name__ == "__main__":
    main()
