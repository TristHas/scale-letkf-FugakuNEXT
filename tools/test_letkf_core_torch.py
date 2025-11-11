from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from tools.das_letkf import DasLetkfIdentifier, _make_letkf_inputs
from tools.das_replay import ObsLocalReplay
from tools.letkf_core import letkf_core
from tools.letkf_core_torch import letkf_core_torch
from tools.torch_batches import build_core_batch_from_inputs


DUMP_ROOT = Path("result/SC23/20210730060030/letkf_dump")


def _dump_available() -> bool:
    return DUMP_ROOT.exists()


def run_test() -> None:
    if not _dump_available():
        print(f"Skipping torch LETKF test: dump folder missing ({DUMP_ROOT})")
        return

    pe_tag = "pe000000"
    member = "mem0001"
    call_ids = [1, 2, 3, 4, 5]

    ident = DasLetkfIdentifier(dump_dir=DUMP_ROOT, pe_tag=pe_tag, member=member)
    replay = ObsLocalReplay(DUMP_ROOT, pe_tag, member)

    numpy_outputs = []
    inputs = []
    for call_id in call_ids:
        outputs, meta = replay.run_call(call_id)
        lc_inputs = _make_letkf_inputs(ident, replay, meta, outputs)
        inputs.append(lc_inputs)
        numpy_outputs.append(letkf_core(lc_inputs))

    batch = build_core_batch_from_inputs(inputs)
    torch_result = letkf_core_torch(batch)

    for idx, ref in enumerate(numpy_outputs):
        trans_ref = np.asarray(ref["trans"], dtype=np.float64)
        np.testing.assert_allclose(
            torch_result.trans[idx].cpu().numpy(),
            trans_ref,
            atol=1.0e-10,
            rtol=1.0e-10,
        )
        np.testing.assert_allclose(
            torch_result.transm[idx].cpu().numpy(),
            np.asarray(ref["transm"], dtype=np.float64),
            atol=1.0e-10,
            rtol=1.0e-10,
        )
        pa_ref = np.asarray(ref["pa"], dtype=np.float64)
        if np.max(np.abs(pa_ref)) > 1.0e-12:
            np.testing.assert_allclose(
                torch_result.pa[idx].cpu().numpy(),
                pa_ref,
                atol=1.0e-10,
                rtol=1.0e-10,
            )

        parm_infl_torch = float(torch_result.parm_infl[idx].cpu().numpy())
        np.testing.assert_allclose(parm_infl_torch, float(ref["parm_infl"]), atol=1.0e-12, rtol=1.0e-12)

    print("letkf_core_torch matches NumPy reference for calls:", call_ids)


if __name__ == "__main__":
    run_test()
