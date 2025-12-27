from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from pyletkf.io.letkf_dumps import load_rank_members
from pyletkf.io.state import (
    convert_scale_letkf_var,
    convert_letkf_scale_var,
    load_letkf_state,
)
from pyletkf.params import MEMBERS


def test_load_and_conversion(dump_dir, prefix, pe_tag, tol=10**-2):
    da_dumped = load_rank_members(dump_dir, prefix, pe_tag)
    da_comp = load_letkf_state(dump_dir, "anal_f", pe_tag)
    diff = np.abs(da_dumped - da_comp)
    err = diff.mean(("y", "x", "z", "ens")) / np.abs(da_dumped).mean(
        ("y", "x", "z", "ens")
    )
    print(err)
    assert (err < tol).all().item()


def test_conversion_loop(da_control: xr.DataArray):
    ds_scale = convert_letkf_scale_var(da_control)
    recon_members: list[xr.DataArray] = []
    for ens_val in ds_scale.coords["ens"].values:
        ds_member = ds_scale.sel(ens=ens_val).reset_coords(drop=True)
        recon = convert_scale_letkf_var(ds_member).expand_dims(ens=[ens_val])
        recon_members.append(recon)

    da_roundtrip = xr.concat(recon_members, dim="ens").transpose(
        "variable", "ens", "z", "y", "x"
    )
    xr.testing.assert_allclose(
        da_control.transpose("variable", "ens", "z", "y", "x"),
        da_roundtrip,
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

    state_dataset = load_letkf_state(dump_root, target_pe, "anal_f", strip_hallow=True).sel(
        ens=list(MEMBERS)
    )
    val = state_dataset["state"].values
    max_diff = np.abs(val - state_np[:, : len(MEMBERS)]).max()
    assert max_diff < 1.0e-9, f"maximum discrepancy {max_diff}"
