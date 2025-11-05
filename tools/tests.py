import numpy as np
from . import load_rank_members, load_and_convert_rank_members
from pathlib import Path

def test_load_and_conversion(dump_dir, prefix, pe_tag, tol=10**-2):
    da_dumped = load_rank_members(dump_dir, prefix, pe_tag)
    da_comp = load_and_convert_rank_members(dump_dir, prefix, pe_tag)
    diff = np.abs(da_dumped - da_comp)
    err = diff.mean(("y", "x", "z", "ens")) / np.abs(da_dumped).mean(("y", "x", "z", "ens"))
    print(err)
    assert (err < tol).all().item()

if __name__ == "__main__":
    dump_dir = Path("./result/SC23/20210730060030/letkf_dump/")
    prefix = "gues3d"
    pe_tag = "pe000000"
    test_load_and_conversion(dump_dir, prefix, pe_tag)