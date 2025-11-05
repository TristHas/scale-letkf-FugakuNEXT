def test_load_and_conversion(dump_dir, prefix, pe_tag, tol=10**-2):
    da_dumped = load_rank_members(dump_dir, prefix, pe_tag)
    da_comp = load_and_convert_rank_members(dump_dir, prefix, pe_tag)
    diff = np.abs(da_dumped - da_comp)
    err = diff.mean(("y", "x", "z", "ens")) / np.abs(da_dumped).mean(("y", "x", "z", "ens"))
    assert (err < tol).all().item()