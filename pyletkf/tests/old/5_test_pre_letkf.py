from __future__ import annotations

import numpy as np
import torch
import pytest

from pyletkf.io.das_letkf_dumps import load_das_obs_local_after
from pyletkf_tests.helpers import cell_index

CALL_IDS = [1, 12, 23]


@pytest.mark.parametrize("call_id", CALL_IDS)
def test_pre_letkf_matches_obs_local(pre_letkf_dataset, dump_root, target_pe, state_dataset, call_id):
    record = load_das_obs_local_after(
        dump_root, call_id=call_id, pe_tag=target_pe, member="mem0001"
    )
    meta = record["meta"]
    data = record["data"]
    nlev = state_dataset["state"].sizes["z"]
    cell_idx = cell_index(int(meta["ij"]), int(meta["ilev"]), nlev)

    hdxf = pre_letkf_dataset["hdxf"].data[cell_idx]
    dep = pre_letkf_dataset["dep"].data[cell_idx]
    rloc = pre_letkf_dataset["rloc"].data[cell_idx]
    rdiag = pre_letkf_dataset["rdiag"].data[cell_idx]
    mask = pre_letkf_dataset["obs_mask"].data[cell_idx].to(torch.bool)

    nobs = data["hdxf"].shape[0]
    indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
    assert indices.numel() >= nobs
    sel = indices[:nobs]

    np.testing.assert_allclose(
        hdxf[sel].cpu().numpy(),
        data["hdxf"],
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    np.testing.assert_allclose(
        dep[sel].cpu().numpy(),
        data["dep"],
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    np.testing.assert_allclose(
        rloc[sel].cpu().numpy(),
        data["rloc"],
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    np.testing.assert_allclose(
        rdiag[sel].cpu().numpy(),
        data["rdiag"],
        atol=1.0e-10,
        rtol=1.0e-10,
    )
