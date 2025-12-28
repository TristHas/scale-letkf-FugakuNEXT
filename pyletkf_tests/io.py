from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

from pyletkf.io.letkf_dumps import load_rank_members
from pyletkf.io.state import convert_letkf_to_scale, convert_scale_to_letkf, load_letkf_state
from pyletkf.params import MEMBERS

def test_conversion_loop(state_dataset):
    control = state_dataset["state"]
    scale_dataset = convert_letkf_to_scale(state_dataset)
    recon_control = convert_scale_to_letkf(scale_dataset)["state"]
    torch.testing.assert_close(
        control.values,
        recon_control.values,
        atol=1.0e-9,
        rtol=1.0e-9,
    )


@pytest.mark.skipif(
    not Path("result/SC23/20210730060030/letkf_dump").exists(),
    reason="SC23 dump directory not available",
)
def test_load_state_against_dump():
    dump_root = Path("result/SC23/20210730060030/letkf_dump")
    target_pe = "pe000005"

    dump_da = load_rank_members(dump_root, "gues3d", target_pe)
    state_np = dump_da.transpose("variable", "ens", "z", "y", "x").values

    state_dataset = load_letkf_state(dump_root, target_pe, "anal_f").sel(
        ens=list(MEMBERS)
    )
    val = state_dataset["state"].values
    max_diff = np.abs(val - state_np[:, : len(MEMBERS)]).max()
    assert max_diff < 1.0e-9, f"maximum discrepancy {max_diff}"
