from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

# Ensure the repository root (which contains the `pyletkf` package directory)
# is on sys.path even though the package itself is namespace-style.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyletkf.io import load_haloed_letkf_state, load_radar
from pyletkf.params import MEMBERS, NX_TILE, NY_TILE, PRC_NUM_X
import pyletkf.pre_letkf as _pre_letkf_mod
import pyletkf.spatial.map_state_to_obs as _ms2o_mod
import pyletkf.spatial.map_obs_to_state as _mos_mod
from pyletkf.spatial.grid_proj import filter_obs_to_tile

# The production code still exposes pre_letkf.read_tile_hx with the legacy
# signature that included a tile index. Newer map_state_to_obs.read_tile_hx no
# longer accepts that parameter, so shim a compatibility wrapper for tests.
def _compat_read_tile_hx(obs, state_ds, *_unused_tile_index):
    obs_tile = filter_obs_to_tile(obs, state_ds)
    if obs_tile is None or obs_tile.sizes["obs"] == 0:
        return (None, None)
    return _ms2o_mod.read_tile_hx(obs_tile, state_ds)

_pre_letkf_mod.read_tile_hx = _compat_read_tile_hx
def _compat_filter_obs_to_haloed_tile(obs, tile_index, halo_i, halo_j):
    tile_i = tile_index % PRC_NUM_X
    tile_j = tile_index // PRC_NUM_X
    start_i = tile_i * NX_TILE - halo_i
    end_i = (tile_i + 1) * NX_TILE + halo_i
    start_j = tile_j * NY_TILE - halo_j
    end_j = (tile_j + 1) * NY_TILE + halo_j
    ri = obs["ri_global"].data
    rj = obs["rj_global"].data
    mask = (ri >= start_i) & (ri <= end_i) & (rj >= start_j) & (rj <= end_j)
    indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
    if indices.numel() == 0:
        return None
    return obs.isel(obs=indices.cpu())

_mos_mod.filter_obs_to_haloed_tile = _compat_filter_obs_to_haloed_tile

from pyletkf.pre_letkf import pre_letkf

TARGET_DUMP_ROOT = Path("result/SC23/20210730060030/letkf_dump")
TARGET_PE = "pe000005"
TARGET_TILE_INDEX = 5
RADAR_PATH = Path("result/SC23/obs_radar/radar_20210730060030.dat")


def _ensure_data_available():
    if not TARGET_DUMP_ROOT.exists():
        pytest.skip(f"LETKF dumps not found at {TARGET_DUMP_ROOT}")
    if not RADAR_PATH.exists():
        pytest.skip(f"Radar observations not found at {RADAR_PATH}")

@pytest.fixture(scope="session")
def dump_root():
    _ensure_data_available()
    return TARGET_DUMP_ROOT


@pytest.fixture(scope="session")
def target_tile():
    return TARGET_TILE_INDEX


@pytest.fixture(scope="session")
def target_pe():
    return TARGET_PE


@pytest.fixture(scope="session")
def state_dataset(dump_root):
    ds = load_haloed_letkf_state(dump_root, TARGET_PE, "anal_f", halo_x=1, halo_y=1)
    return ds.sel(ens=list(MEMBERS))


@pytest.fixture(scope="session")
def radar_dataset():
    return load_radar(RADAR_PATH)


@pytest.fixture(scope="session")
def pre_letkf_dataset(radar_dataset, state_dataset, target_tile):
    device = torch.device("cpu")
    results = pre_letkf(radar_dataset, {target_tile: state_dataset}, device=device)
    return results[target_tile]
