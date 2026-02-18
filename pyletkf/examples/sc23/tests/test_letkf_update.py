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

from pyletkf.core.letkf.ops import letkf_update  # type: ignore  # noqa: E402
from test_io.das_letkf_dumps import (  # type: ignore  # noqa: E402
    list_das_calls,
    load_das_postproc_after,
    load_das_postproc_before,
)
from test_io.letkf_dumps import DUMP_ROOT  # type: ignore  # noqa: E402


def _to_float_tensor(array: np.ndarray | Any) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array, dtype=np.float64))


def _bool_from_meta(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"true", "t", "1", "yes"}:
        return True
    if token in {"false", "f", "0", "no"}:
        return False
    return default


def run_update_validation(limit: int | None = None) -> dict[str, Any]:
    call_ids = list_das_calls(DUMP_ROOT)
    if limit is not None:
        call_ids = call_ids[:limit]

    records: list[dict[str, Any]] = []
    missing = 0
    for call_id in tqdm(call_ids, desc="Validating letkf_update"):
        try:
            before = load_das_postproc_before(DUMP_ROOT, call_id, return_meta=True)
            after = load_das_postproc_after(DUMP_ROOT, call_id, return_meta=True)
        except FileNotFoundError:
            missing += 1
            continue
        data = before["data"]
        gues = data.get("gues_members")
        if gues is None or gues.size == 0:
            missing += 1
            continue

        gues_mean = float(before["meta"]["gues_mean"])
        xb_vals = np.asarray(gues, dtype=np.float64) + gues_mean
        xb = torch.from_numpy(xb_vals).view(1, 1, -1)
        Wa = _to_float_tensor(data["trans"]).unsqueeze(0)
        wa = _to_float_tensor(data["transm"]).unsqueeze(0)

        beta = float(before["meta"].get("beta", after["meta"].get("beta", 1.0)))
        alpha = float(before["meta"].get("relax_alpha", 0.0))
        alpha_spread = float(before["meta"].get("relax_alpha_spread", 0.0))
        parm = (
            float(before["meta"].get("parm", 1.0))
            if _bool_from_meta(before["meta"].get("relax_to_inflated_prior"), True)
            else 1.0
        )

        with torch.no_grad():
            xa, trans = letkf_update(
                xb,
                wa,
                Wa,
                alpha_pert=alpha,
                alpha_spread=alpha_spread,
                beta=beta,
                parm_infl=parm,
                return_trans=True,
            )

        xa_np = xa.squeeze(0).squeeze(0).cpu().numpy()
        trans_np = trans.squeeze(0).cpu().numpy()
        anal_expected = np.asarray(after["data"]["anal_members"], dtype=np.float64)
        trans_expected = np.asarray(after["data"]["transrlx"], dtype=np.float64)

        members_abs = float(np.max(np.abs(xa_np - anal_expected)))
        trans_abs = float(np.max(np.abs(trans_np - trans_expected)))
        ok = bool(np.allclose(xa_np, anal_expected, atol=1.0e-10, rtol=1.0e-10)) and bool(
            np.allclose(trans_np, trans_expected, atol=1.0e-10, rtol=1.0e-10)
        )
        records.append(
            {
                "call_id": call_id,
                "members_abs_max": members_abs,
                "trans_abs_max": trans_abs,
                "ok": ok,
            }
        )

    return {"results": records, "missing": missing, "requested": len(call_ids)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate pyletkf letkf_update outputs against SCALE dumps."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N calls (useful for smoke tests).",
    )
    args = parser.parse_args()

    summary = run_update_validation(limit=args.limit)
    results = summary["results"]
    missing = summary["missing"]
    requested = summary["requested"]

    if not results:
        print(f"No postproc data found (checked {requested}, missing {missing}).")
        return 1

    failures = [res for res in results if not res["ok"]]
    max_members = max(res["members_abs_max"] for res in results)
    max_trans = max(res["trans_abs_max"] for res in results)

    print(f"Validated {len(results)} postproc calls (missing inputs: {missing}).")
    print(
        f"Max |anal_members - SCALE| = {max_members:.3e}, "
        f"max |transrlx - SCALE| = {max_trans:.3e}."
    )

    if failures:
        failing_ids = ", ".join(str(res["call_id"]) for res in failures[:10])
        print(f"{len(failures)} calls failed: {failing_ids}")
        return 2

    print("All compared calls match SCALE postproc outputs within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
