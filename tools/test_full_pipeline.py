from __future__ import annotations

from pathlib import Path
import sys
from typing import List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.das_letkf import DasLetkfIdentifier, _make_letkf_inputs, _make_postproc_inputs
from tools.das_replay import ObsLocalReplay
from tools.letkf_core import letkf_core
from tools.postproc import postproc
from tools.letkf_core_torch import letkf_core_torch
from tools.postproc_torch import postproc_torch
from tools.torch_batches import CoreBatchInputs, build_postproc_batch_from_inputs

from tools_new.io.letkf_dumps import load_obsda_sorted
from tools_new.io.radar import load_radar
from tools_new.io.state import load_letkf_state
from tools_new.pre_letkf import pre_letkf_pipeline

DUMP_ROOT = Path("result/SC23/20210730060030/letkf_dump")
PE_TAG = "pe000000"
MEMBER_TAG = "mem0001"
CALL_IDS = [1, 2, 3]


def _dump_available() -> bool:
    return DUMP_ROOT.exists()


def _select_row(results: dict, ij: int, ilev: int) -> dict:
    n_horiz = results["grid_info"]["n_horiz"]
    idx = (ilev - 1) * n_horiz + (ij - 1)
    return {
        "hdxf": results["hdxf"][idx],
        "dep": results["dep"][idx],
        "rloc": results["rloc"][idx],
        "rdiag": results["rdiag"][idx],
        "mask": results["mask"][idx].astype(bool),
        "indices": results["obs_indices"][idx],
    }


def run_test() -> None:
    if not _dump_available():
        print(f"Skipping full-pipeline test: dump folder missing ({DUMP_ROOT})")
        return

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    obs = load_radar()
    obsda_sorted = load_obsda_sorted(DUMP_ROOT, PE_TAG)
    obs = obs.sel(obs=obs.obs.isin(obsda_sorted.obs))
    state_ds = load_letkf_state(DUMP_ROOT, PE_TAG, "anal_f", strip_hallow=True)

    pre_results = pre_letkf_pipeline(obs, state_ds, PE_TAG, device=device)

    ident = DasLetkfIdentifier(dump_dir=DUMP_ROOT, pe_tag=PE_TAG, member=MEMBER_TAG)
    replay = ObsLocalReplay(DUMP_ROOT, PE_TAG, MEMBER_TAG)

    core_rows = []
    numpy_core_outputs: List[dict] = []
    lc_inputs_list = []
    metas = []

    for call_id in CALL_IDS:
        outputs, meta = replay.run_call(call_id)
        lc_inputs = _make_letkf_inputs(ident, replay, meta, outputs)
        lc_inputs_list.append(lc_inputs)
        metas.append(meta)

        row = _select_row(pre_results, meta.ij, meta.ilev)
        nobsl = lc_inputs.before_arrays["hdxf"].shape[0]
        sel = np.flatnonzero(row["mask"])
        if sel.size != nobsl:
            raise AssertionError(f"Mask/obs count mismatch for call {call_id}")

        np.testing.assert_allclose(
            row["hdxf"][sel, :],
            lc_inputs.before_arrays["hdxf"],
            atol=1.0e-10,
            rtol=1.0e-10,
        )
        np.testing.assert_allclose(
            row["dep"][sel],
            lc_inputs.before_arrays["dep"],
            atol=1.0e-10,
            rtol=1.0e-10,
        )
        np.testing.assert_allclose(
            row["rloc"][sel],
            lc_inputs.before_arrays["rloc"],
            atol=1.0e-10,
            rtol=1.0e-10,
        )
        np.testing.assert_allclose(
            row["rdiag"][sel],
            lc_inputs.before_arrays["rdiag"],
            atol=1.0e-10,
            rtol=1.0e-10,
        )

        numpy_core_outputs.append(letkf_core(lc_inputs))
        core_rows.append(row)

    hdxf_stack = np.stack([row["hdxf"] for row in core_rows], axis=0)
    dep_stack = np.stack([row["dep"] for row in core_rows], axis=0)
    rloc_stack = np.stack([row["rloc"] for row in core_rows], axis=0)
    rdiag_stack = np.stack([row["rdiag"] for row in core_rows], axis=0)
    mask_stack = np.stack([row["mask"] for row in core_rows], axis=0)

    core_batch = CoreBatchInputs(
        hdxf=torch.from_numpy(hdxf_stack).to(device=device, dtype=torch.float64),
        dep=torch.from_numpy(dep_stack).to(device=device, dtype=torch.float64),
        rdiag=torch.from_numpy(rdiag_stack).to(device=device, dtype=torch.float64),
        rloc=torch.from_numpy(rloc_stack).to(device=device, dtype=torch.float64),
        obs_mask=torch.from_numpy(mask_stack).to(device=device),
        parm_infl=torch.ones(len(CALL_IDS), dtype=torch.float64, device=device),
        rdiag_wloc=torch.ones(len(CALL_IDS), dtype=torch.bool, device=device),
        infl_update=torch.zeros(len(CALL_IDS), dtype=torch.bool, device=device),
        call_ids=torch.tensor(CALL_IDS, dtype=torch.int64, device=device),
    )

    torch_core = letkf_core_torch(core_batch)

    for idx, ref in enumerate(numpy_core_outputs):
        np.testing.assert_allclose(
            torch_core.trans[idx].cpu().numpy(),
            np.asarray(ref["trans"], dtype=np.float64),
            atol=1.0e-10,
            rtol=1.0e-10,
        )
        np.testing.assert_allclose(
            torch_core.transm[idx].cpu().numpy(),
            np.asarray(ref["transm"], dtype=np.float64),
            atol=1.0e-10,
            rtol=1.0e-10,
        )
        np.testing.assert_allclose(
            torch_core.pa[idx].cpu().numpy(),
            np.asarray(ref["pa"], dtype=np.float64),
            atol=1.0e-10,
            rtol=1.0e-10,
        )
        np.testing.assert_allclose(
            float(torch_core.parm_infl[idx].cpu().numpy()),
            float(ref["parm_infl"]),
            atol=1.0e-12,
            rtol=1.0e-12,
        )

    pp_inputs = []
    numpy_post = []
    for idx, (meta, lc_inputs) in enumerate(zip(metas, lc_inputs_list)):
        core_dict = {
            "trans": torch_core.trans[idx].cpu().numpy(),
            "transm": torch_core.transm[idx].cpu().numpy(),
            "parm_infl": float(torch_core.parm_infl[idx].cpu().numpy()),
        }
        pp_in = _make_postproc_inputs(DUMP_ROOT, ident, meta, lc_inputs, core_dict)
        pp_inputs.append(pp_in)
        numpy_post.append(postproc(pp_in))

    post_batch = build_postproc_batch_from_inputs(pp_inputs, device=device)
    torch_post = postproc_torch(post_batch)

    for idx, ref in enumerate(numpy_post):
        np.testing.assert_allclose(
            torch_post.transrlx[idx].cpu().numpy(),
            np.asarray(ref["transrlx"], dtype=np.float64),
            atol=1.0e-10,
            rtol=1.0e-10,
        )
        np.testing.assert_allclose(
            torch_post.anal_members[idx].cpu().numpy(),
            np.asarray(ref["anal_members"], dtype=np.float64),
            atol=1.0e-10,
            rtol=1.0e-10,
        )
        np.testing.assert_allclose(
            float(torch_post.workda_value[idx].cpu().numpy()),
            float(ref["workda_value"]),
            atol=1.0e-10,
            rtol=1.0e-10,
        )
        assert bool(torch_post.workda_present[idx].cpu().numpy()) == bool(
            ref["workda_present"]
        )
        assert bool(torch_post.q_limited[idx].cpu().numpy()) == bool(ref["q_limited"])
        np.testing.assert_allclose(
            float(torch_post.q_mean[idx].cpu().numpy()),
            float(ref["q_mean"]),
            atol=1.0e-10,
            rtol=1.0e-10,
        )
        np.testing.assert_allclose(
            float(torch_post.q_sprd[idx].cpu().numpy()),
            float(ref["q_sprd"]),
            atol=1.0e-10,
            rtol=1.0e-10,
        )

    print("Full pipeline matches Fortran reference for calls:", CALL_IDS)


if __name__ == "__main__":
    run_test()
