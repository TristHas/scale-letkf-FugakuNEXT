from __future__ import annotations

from pathlib import Path

import xarray as xr

from pyletkf.full_pipeline import run_full_pipeline
from pyletkf.io import load_letkf_state
from pyletkf.params import MEMBERS


def test_full_pipeline_reproduces_fortran_analysis():
    dump_dir = Path("result/SC23/20210730060030/letkf_dump")
    pe_tag = "pe000005"
    analysis = run_full_pipeline(pe_tag, dump_dir=dump_dir, device="cpu", chunk_size=512)
    truth = (
        load_letkf_state(dump_dir, pe_tag, "anal", strip_hallow=True)
        .sel(ens=list(MEMBERS))["state"]
        .to_dataarray()
    )
    xr.testing.assert_allclose(analysis, truth)
