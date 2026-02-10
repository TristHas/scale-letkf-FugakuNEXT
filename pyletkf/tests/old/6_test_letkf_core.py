from __future__ import annotations

from pathlib import Path

import numpy as np

from pyletkf.letkf_core import letkf_core
from pyletkf.io.das_letkf_dumps import load_das_letkf_core_after

from pyletkf_tests.helpers import build_core_batch


def _discover_call_ids(dump_root: Path, pe_tag: str, limit: int = 3) -> list[int]:
    stage_dir = dump_root.parent / "das_letkf" / "letkf_core" / "before"
    pattern = f"meta_call*_{pe_tag}.mem0001.txt"
    call_ids = sorted(
        int(path.name.split("_")[1][4:])
        for path in stage_dir.glob(pattern)
    )
    return call_ids[:limit]


def test_letkf_core_matches_fortran(dump_root, target_pe):
    call_ids = _discover_call_ids(dump_root, target_pe, limit=3)
    batch, obs_counts = build_core_batch(call_ids, dump_root, target_pe)
    result = letkf_core(batch)
    for idx, call_id in enumerate(call_ids):
        after = load_das_letkf_core_after(
            dump_root, call_id=call_id, pe_tag=target_pe, member="mem0001"
        )
        np.testing.assert_allclose(
            result["trans"].data[idx].cpu().numpy(),
            after["data"]["trans"],
            atol=1.0e-10,
            rtol=1.0e-10,
        )
        np.testing.assert_allclose(
            result["transm"].data[idx].cpu().numpy(),
            after["data"]["transm"],
            atol=1.0e-10,
            rtol=1.0e-10,
        )
