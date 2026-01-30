from __future__ import annotations

import torch

from pyletkf.io.letkf_dumps import load_obsda
from pyletkf.spatial.grid_proj import compute_obs_grid_idx, filter_obs_to_tile
from pyletkf.spatial.map_state_to_obs import read_tile_hx


def test_read_tile_hx_matches_fortran(
    radar_dataset, state_dataset, dump_root, target_pe, target_tile
):
    obs = compute_obs_grid_idx(radar_dataset)
    obs_tile = filter_obs_to_tile(obs, state_dataset)
    hx, obs_idx = read_tile_hx(obs_tile, state_dataset)
    assert hx is not None and obs_idx is not None

    hx = hx.to(torch.float64).cpu()
    obs_idx = obs_idx.to(torch.int64).cpu()

    obsda = load_obsda(dump_root, target_pe)
    idx_fortran = torch.as_tensor(
        obsda["idx"].isel(member=0).values, dtype=torch.int64
    )
    hx_fortran = torch.from_numpy(obsda["ensval"].values.T).to(
        torch.float64
    )

    idx_lookup = {int(idx): pos for pos, idx in enumerate(obs_idx.tolist())}
    missing = [int(idx) for idx in idx_fortran.tolist() if idx not in idx_lookup]
    assert (
        not missing
    ), f"Missing hx values for {len(missing)} obs present in Fortran dumps"

    gather_rows = torch.tensor(
        [idx_lookup[int(idx)] for idx in idx_fortran.tolist()],
        dtype=torch.long,
    )
    hx_python = hx[gather_rows]
    torch.testing.assert_close(
        hx_python, hx_fortran, atol=1.0e-6, rtol=1.0e-6
    )
