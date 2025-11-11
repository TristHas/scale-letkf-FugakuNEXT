from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.das_letkf import DasLetkfIdentifier, call_id_to_indices, run_rank_analysis
from tools.das_letkf_torch import run_rank_analysis_torch
from tools.params import GRID_CONSTANTS

DUMP_ROOT = Path("result/SC23/20210730060030/letkf_dump")


def _dump_available() -> bool:
    return DUMP_ROOT.exists()


def run_test() -> None:
    if not _dump_available():
        print(f"Skipping das_letkf_torch test: dump folder missing ({DUMP_ROOT})")
        return

    pe_tag = "pe000005"
    ident = DasLetkfIdentifier(dump_dir=DUMP_ROOT, pe_tag=pe_tag, member="mem0001")
    call_ids = [1, 2]

    baseline = run_rank_analysis(ident, call_ids=call_ids)
    torch_result = run_rank_analysis_torch(ident, call_ids=call_ids, chunk_size=4)

    nv3d = int(GRID_CONSTANTS["nv3d"])
    nv2d = int(GRID_CONSTANTS.get("nv2d", 0))
    var_count = max(1, nv3d + nv2d)
    for call_id in call_ids:
        ij, ilev, nvar = call_id_to_indices(call_id, baseline.analysis.shape[0], baseline.analysis.shape[1], var_count=var_count)
        np.testing.assert_allclose(
            torch_result.analysis[ij - 1, ilev - 1, :, nvar - 1],
            baseline.analysis[ij - 1, ilev - 1, :, nvar - 1],
            atol=1.0e-10,
            rtol=1.0e-10,
        )

    assert torch_result.processed_call_count == baseline.processed_call_count == len(call_ids)
    print(f"das_letkf_torch matches NumPy pipeline for {pe_tag} calls:", call_ids)


if __name__ == "__main__":
    run_test()
