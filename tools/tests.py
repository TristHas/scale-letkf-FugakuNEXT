import numpy as np
import xarray as xr
from pathlib import Path

from . import load_rank_members, load_and_convert_rank_members
from .convert_scale_letkf_var import convert_scale_letkf_var, convert_letkf_scale_var


def test_load_and_conversion(dump_dir, prefix, pe_tag, tol=10**-2):
    da_dumped = load_rank_members(dump_dir, prefix, pe_tag)
    da_comp = load_and_convert_rank_members(dump_dir, prefix, pe_tag)
    diff = np.abs(da_dumped - da_comp)
    err = diff.mean(("y", "x", "z", "ens")) / np.abs(da_dumped).mean(("y", "x", "z", "ens"))
    print(err)
    assert (err < tol).all().item()


def test_conversion_loop(da_control: xr.DataArray):
    ds_scale = convert_letkf_scale_var(da_control)
    recon_members: list[xr.DataArray] = []
    for ens_val in ds_scale.coords["ens"].values:
        ds_member = ds_scale.sel(ens=ens_val).reset_coords(drop=True)
        recon = convert_scale_letkf_var(ds_member).expand_dims(ens=[ens_val])
        recon_members.append(recon)

    da_roundtrip = xr.concat(recon_members, dim="ens").transpose("variable", "ens", "z", "y", "x")
    xr.testing.assert_allclose(
        da_control.transpose("variable", "ens", "z", "y", "x"),
        da_roundtrip,
    )


if __name__ == "__main__":
    dump_dir = Path("./result/SC23/20210730060030/letkf_dump/")
    prefix = "gues3d"
    pe_tag = "pe000000"

    test_load_and_conversion(dump_dir, prefix, pe_tag)
    da = load_and_convert_rank_members(dump_dir, prefix, pe_tag)
    test_conversion_loop(da)
