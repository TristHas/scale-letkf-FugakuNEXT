"""Minimal smoke test for the PyTorch LETKF helper.

Run with:

    python -m tools.letkf_pytorch_demo /path/to/dump pe000002

The script expects PyTorch to be installed and will raise an informative
error otherwise.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - guarded import
    raise SystemExit(
        "PyTorch is required to run the LETKF demo (pip install torch)."
    ) from exc

from .letkf_pytorch import das_letkf, load_and_prepare


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump_dir", type=Path, help="Path to letkf_dump directory")
    parser.add_argument(
        "obs_suffix",
        help="Observation bundle suffix (e.g. 'pe000002')",
    )
    parser.add_argument(
        "--obs-error",
        type=float,
        default=1.0,
        help="Default observation variance (used when not available in dump)",
    )
    parser.add_argument(
        "--inflation",
        type=float,
        default=1.0,
        help="Optional multiplicative inflation applied to analysis perturbations",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    xb, obs = load_and_prepare(
        args.dump_dir, args.obs_suffix, default_obs_error=args.obs_error
    )

    print(f"Loaded background state of shape {tuple(xb.shape)}")
    print(f"Observation set: {obs.hx_anom.shape[0]} obs, {obs.hx_anom.shape[1]} members")

    xa = das_letkf(
        xb,
        obs.hx_anom,
        obs.innov,
        obs.obs_var,
        inflation=args.inflation,
    )

    xb_mean = xb.mean(dim=1)
    xa_mean = xa.mean(dim=1)
    delta_norm = torch.linalg.norm(xa_mean - xb_mean).item()
    print(f"Analysis mean shift (L2 norm): {delta_norm:.3e}")

    spread_bg = xb.std(dim=1).mean().item()
    spread_an = xa.std(dim=1).mean().item()
    print(f"Average ensemble spread: background={spread_bg:.3e}, analysis={spread_an:.3e}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
