#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm


CURRENT_DIR = Path(__file__).resolve().parent
SC23_DIR = CURRENT_DIR.parent
PKG_ROOT = SC23_DIR.parents[1]

for path in (SC23_DIR, PKG_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from pyletkf.core.letkf.ops import letkf_core  # type: ignore  # noqa: E402
from test_io.das_letkf_dumps import (  # type: ignore  # noqa: E402
    list_das_calls,
    load_das_letkf_core_after,
    load_das_letkf_core_before,
)
from test_io.letkf_dumps import DUMP_ROOT  # type: ignore  # noqa: E402


def _to_float_tensor(array: np.ndarray | Any) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array, dtype=np.float64))


def convert_to_letkf_format(data: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """Prepare tensors expected by letkf_core."""

    hdxf = _to_float_tensor(data["hdxf"]).unsqueeze(0)
    dep = _to_float_tensor(data["dep"]).unsqueeze(0)
    rdiag = _to_float_tensor(data["rdiag"]).unsqueeze(0)
    mask = torch.ones_like(rdiag, dtype=torch.bool)
    return {"hdx": hdxf, "dep": dep, "rdiag": rdiag, "mask": mask}


def compare_outputs(
    result: tuple[torch.Tensor, torch.Tensor],
    expected: dict[str, np.ndarray],
    *,
    atol: float = 1.0e-10,
    rtol: float = 1.0e-10,
) -> dict[str, float | bool]:
    """Compare Wa/wa tensors against SCALE outputs."""

    trans_expected = np.asarray(expected["trans"], dtype=np.float64)
    transm_expected = np.asarray(expected["transm"], dtype=np.float64)

    Wa, wa = result
    Wa_np = Wa.squeeze(0).cpu().numpy()
    wa_np = wa.squeeze(0).cpu().numpy()

    trans_abs = float(np.max(np.abs(Wa_np - trans_expected)))
    transm_abs = float(np.max(np.abs(wa_np - transm_expected)))

    trans_ok = bool(np.allclose(Wa_np, trans_expected, atol=atol, rtol=rtol))
    transm_ok = bool(np.allclose(wa_np, transm_expected, atol=atol, rtol=rtol))

    return {
        "trans_abs_max": trans_abs,
        "transm_abs_max": transm_abs,
        "trans_ok": trans_ok,
        "transm_ok": transm_ok,
        "ok": bool(trans_ok and transm_ok),
    }


def run_validation(limit: int | None = None) -> dict[str, Any]:
    call_ids = list_das_calls(DUMP_ROOT)
    if limit is not None:
        call_ids = call_ids[:limit]

    results: list[dict[str, Any]] = []
    missing_inputs = 0
    for call_id in tqdm(call_ids, desc="Validating letkf_core"):
        try:
            before = load_das_letkf_core_before(DUMP_ROOT, call_id)
            after = load_das_letkf_core_after(DUMP_ROOT, call_id)
        except FileNotFoundError:
            missing_inputs += 1
            continue
        data = before.get("data", {})
        hdxf = data.get("hdxf")
        dep = data.get("dep")
        rdiag = data.get("rdiag")
        if hdxf is None or dep is None or rdiag is None or hdxf.size == 0:
            missing_inputs += 1
            continue
        tensors = convert_to_letkf_format(data)
        with torch.no_grad():
            computed = letkf_core(**tensors)
        summary = compare_outputs(computed, after["data"])
        summary["call_id"] = call_id
        results.append(summary)

    return {
        "results": results,
        "missing_inputs": missing_inputs,
        "requested_calls": len(call_ids),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate pyletkf letkf_core outputs.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only validate the first N call_ids (useful for quick spot checks).",
    )
    args = parser.parse_args()

    summary = run_validation(limit=args.limit)
    results: list[dict[str, Any]] = summary["results"]
    missing_inputs = summary["missing_inputs"]
    requested_calls = summary["requested_calls"]

    if not results:
        print(
            f"No calls produced comparable data "
            f"(checked {requested_calls}, missing {missing_inputs})."
        )
        return 1

    failures = [res for res in results if not res["ok"]]
    max_trans = max(res["trans_abs_max"] for res in results)
    max_transm = max(res["transm_abs_max"] for res in results)

    print(f"Validated {len(results)} calls (missing inputs: {missing_inputs}).")
    print(
        f"Max |trans - SCALE| = {max_trans:.3e}, "
        f"max |transm - SCALE| = {max_transm:.3e}."
    )

    if failures:
        failing_ids = ", ".join(str(res["call_id"]) for res in failures[:10])
        print(f"{len(failures)} calls failed: {failing_ids}")
        return 2

    print("All compared calls match SCALE outputs within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
