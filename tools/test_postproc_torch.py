from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.das_letkf import DasLetkfIdentifier, _make_letkf_inputs, _make_postproc_inputs
from tools.das_replay import ObsLocalReplay
from tools.letkf_core import letkf_core
from tools.postproc import postproc
from tools.postproc_torch import postproc_torch
from tools.torch_batches import build_postproc_batch_from_inputs

DUMP_ROOT = Path("result/SC23/20210730060030/letkf_dump")


def _dump_available() -> bool:
    return DUMP_ROOT.exists()


def run_test() -> None:
    if not _dump_available():
        print(f"Skipping torch postproc test: dump folder missing ({DUMP_ROOT})")
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
        core_outputs = letkf_core(lc_inputs)
        pp_inputs = _make_postproc_inputs(DUMP_ROOT, ident, meta, lc_inputs, core_outputs)
        numpy_outputs.append(postproc(pp_inputs))
        inputs.append(pp_inputs)

    batch = build_postproc_batch_from_inputs(inputs)
    torch_result = postproc_torch(batch)

    for idx, ref in enumerate(numpy_outputs):
        np.testing.assert_allclose(
            torch_result.transrlx[idx].cpu().numpy(),
            np.asarray(ref["transrlx"], dtype=np.float64),
            atol=1.0e-10,
            rtol=1.0e-10,
        )
        np.testing.assert_allclose(
            torch_result.anal_members[idx].cpu().numpy(),
            np.asarray(ref["anal_members"], dtype=np.float64),
            atol=1.0e-10,
            rtol=1.0e-10,
        )
        np.testing.assert_allclose(
            float(torch_result.workda_value[idx].cpu().numpy()),
            float(ref["workda_value"]),
            atol=1.0e-10,
            rtol=1.0e-10,
        )
        assert bool(torch_result.workda_present[idx].cpu().numpy()) == bool(ref["workda_present"])
        assert bool(torch_result.q_limited[idx].cpu().numpy()) == bool(ref["q_limited"])
        np.testing.assert_allclose(
            float(torch_result.q_mean[idx].cpu().numpy()),
            float(ref["q_mean"]),
            atol=1.0e-10,
            rtol=1.0e-10,
        )
        np.testing.assert_allclose(
            float(torch_result.q_sprd[idx].cpu().numpy()),
            float(ref["q_sprd"]),
            atol=1.0e-10,
            rtol=1.0e-10,
        )

    print("postproc_torch matches NumPy reference for calls:", call_ids)


if __name__ == "__main__":
    run_test()
