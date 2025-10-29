#!/usr/bin/env python3
"""Convenience script to execute the LETKF workflow via the Python bindings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .letkf_api import LetkfAPI


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools_python.run_letkf",
        description="Run the SCALE-LETKF workflow through the Python C-API bindings.",
    )
    parser.add_argument(
        "config",
        help="Path to the LETKF configuration file (e.g. config.nml.letkf).",
    )
    parser.add_argument(
        "--stdout-dir",
        help="Optional directory prefix for per-rank stdout files (matches the Fortran optional second argument).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"LETKF config file not found: {config_path}")

    new_cli = [sys.argv[0], str(config_path)]
    if args.stdout_dir:
        new_cli.append(args.stdout_dir)
    sys.argv = new_cli

    api = LetkfAPI()
    api.set_config_path(str(config_path))
    api.initialize()
    try:
        api.run_letkf()
    finally:
        api.finalize()


if __name__ == "__main__":
    main()
