import numpy as np
import xarray as xr
from pathlib import Path

from . import load_rank_members, load_and_convert_rank_members
from .convert_scale_letkf_var import convert_scale_letkf_var, convert_letkf_scale_var
from .das_letkf import (
    DasLetkfIdentifier,
    call_id_to_indices,
    run_rank_analysis,
)
from .load_dumps import _read_binary_array
from .params import GRID_CONSTANTS


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


def test_das_letkf_pipeline_subset():
    dump_dir = Path("result/SC23/20210730060030/letkf_dump/")
    pe_tag = "pe000000"
    ident = DasLetkfIdentifier(dump_dir=dump_dir, pe_tag=pe_tag, member="mem0001")
    subset_calls = list(range(1, 65))
    result = run_rank_analysis(ident, call_ids=subset_calls)
    assert result.processed_call_count == len(subset_calls)

    ref = _read_binary_array(dump_dir / "anal3d" / f"anal3d_{pe_tag}.mem0001.bin", ">f8")
    var_count = int(GRID_CONSTANTS["nv3d"]) + int(GRID_CONSTANTS.get("nv2d", 0))
    member_count = result.analysis.shape[2]

    for call_id in subset_calls:
        ij, ilev, nvar = call_id_to_indices(call_id, ref.shape[0], ref.shape[1], var_count=var_count)
        assert result.updated_mask[ij - 1, ilev - 1, nvar - 1]
        np.testing.assert_allclose(
            result.analysis[ij - 1, ilev - 1, :, nvar - 1],
            ref[ij - 1, ilev - 1, :member_count, nvar - 1],
            atol=1.0e-10,
            rtol=1.0e-10,
        )


if __name__ == "__main__":
    dump_dir = Path("./result/SC23/20210730060030/letkf_dump/")
    prefix = "gues3d"
    pe_tag = "pe000000"

    test_load_and_conversion(dump_dir, prefix, pe_tag)
    da = load_and_convert_rank_members(dump_dir, prefix, pe_tag)
    test_conversion_loop(da)
