from __future__ import annotations

import torch

from pyletkf.io.letkf_dumps import load_obsda_sorted
from pyletkf.params import NX_TILE, NY_TILE
from pyletkf.spatial.grid_proj import compute_obs_grid_idx
from pyletkf_tests.helpers import load_obsdanosort_coords


def test_grid_projection_matches_fortran(radar_dataset, dump_root, target_pe):
    obs = compute_obs_grid_idx(radar_dataset)
    ri_global = obs["ri_global"].data.to(torch.float64)
    rj_global = obs["rj_global"].data.to(torch.float64)
    ri_local = obs["ri_local"].data.to(torch.float64)
    rj_local = obs["rj_local"].data.to(torch.float64)

    sorted_ds = load_obsda_sorted(dump_root, target_pe)
    idx_fortran = torch.as_tensor(sorted_ds["idx"].values, dtype=torch.int64)
    ri_ref_np, rj_ref_np = load_obsdanosort_coords(dump_root, target_pe)
    assert (
        idx_fortran.numel()
        == ri_ref_np.shape[0]
        == rj_ref_np.shape[0]
    ), "Mismatch between obsda metadata and obsdanosort dumps"

    idx0 = idx_fortran - 1
    ri_ref = torch.from_numpy(ri_ref_np).to(torch.float64)
    rj_ref = torch.from_numpy(rj_ref_np).to(torch.float64)

    torch.testing.assert_close(
        ri_global[idx0], ri_ref, atol=1.0e-6, rtol=1.0e-6
    )
    torch.testing.assert_close(
        rj_global[idx0], rj_ref, atol=1.0e-6, rtol=1.0e-6
    )

    ri_local_expected = ((ri_ref - 1.0) % NX_TILE) + 1.0
    rj_local_expected = ((rj_ref - 1.0) % NY_TILE) + 1.0
    torch.testing.assert_close(
        ri_local[idx0], ri_local_expected, atol=1.0e-5, rtol=1.0e-6
    )
    torch.testing.assert_close(
        rj_local[idx0], rj_local_expected, atol=1.0e-5, rtol=1.0e-6
    )
