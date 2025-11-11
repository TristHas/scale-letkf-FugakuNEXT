from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.das_replay import ObsLocalReplay
from tools.obs_local import obs_local
from tools.obs_local import ObsLocalIdentifier
from tools.obs_local_torch import obs_local_torch

DUMP_ROOT = Path("result/SC23/20210730060030/letkf_dump")


def _dump_available() -> bool:
    return DUMP_ROOT.exists()


def run_test() -> None:
    if not _dump_available():
        print(f"Skipping torch obs_local test: dump folder missing ({DUMP_ROOT})")
        return

    pe_tag = "pe000000"
    member = "mem0001"
    call_ids = [1, 2, 3]

    replay = ObsLocalReplay(DUMP_ROOT, pe_tag, member)

    for call_id in call_ids:
        ident = ObsLocalIdentifier(dump_dir=DUMP_ROOT, call_id=call_id, pe_tag=pe_tag, member=member)
        inputs = replay.build_inputs_for(call_id, copy_search=True)
        before_arrays = inputs.before_arrays
        search_np = None
        if before_arrays and "search_q0" in before_arrays:
            search_np = np.array(before_arrays["search_q0"], dtype=np.int64, copy=True)

        torch_search = search_np.copy() if search_np is not None else None
        numpy_search = search_np.copy() if search_np is not None else None

        torch_output = obs_local_torch(inputs, torch_search)
        numpy_output = obs_local(inputs, numpy_search)

        for key in ("hdxf", "dep", "rdiag", "rloc"):
            np.testing.assert_allclose(
                torch_output[key],
                numpy_output[key],
                atol=1.0e-10,
                rtol=1.0e-10,
            )

    print("obs_local_torch matches NumPy reference for calls:", call_ids)


if __name__ == "__main__":
    run_test()
