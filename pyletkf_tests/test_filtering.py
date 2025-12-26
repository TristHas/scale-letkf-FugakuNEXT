from __future__ import annotations

import torch

from pyletkf.io.letkf_dumps import load_obsda_sorted
from pyletkf.obs_op import filter_sc23_obs
from pyletkf.spatial.grid_proj import compute_obs_grid_idx
from pyletkf.spatial.map_state_to_obs import (
    filter_obs_to_tile_index,
    read_tile_hx,
)


def test_filter_sc23_obs_matches_fortran(
    radar_dataset, state_dataset, dump_root, target_tile, target_pe
):
    obs = compute_obs_grid_idx(radar_dataset)
    obs_tile = filter_obs_to_tile_index(obs, target_tile)
    assert obs_tile is not None

    hx, _ = read_tile_hx(obs, state_dataset, target_tile)
    assert hx is not None
    hx = hx.to(torch.float64)
    hx_mean = hx.mean(dim=1)

    ens_coord = torch.arange(hx.shape[1], dtype=torch.int64)
    obs_with_hx = obs_tile.assign_coords(ens=ens_coord)
    obs_with_hx["hx"] = (("obs", "ens"), hx)
    obs_with_hx["hx_mean"] = (("obs",), hx_mean)

    filtered = filter_sc23_obs(obs_with_hx)
    idx_python = filtered["obs_orig"].data.to(torch.int64).cpu()

    sorted_ds = load_obsda_sorted(dump_root, target_pe)
    idx_fortran = torch.as_tensor(sorted_ds["idx"].values, dtype=torch.int64)

    assert idx_python.numel() == idx_fortran.numel()
    torch.testing.assert_close(
        torch.sort(idx_python).values,
        torch.sort(idx_fortran).values,
    )
