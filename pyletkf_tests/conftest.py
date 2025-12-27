from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyletkf.io import load_letkf_state, load_letkf_dump_state, load_radar
from pyletkf.params import MEMBERS
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
    ds = load_letkf_dump_state(dump_root, TARGET_PE, "gues3d", strip_hallow=True)
    return ds.sel(ens=list(MEMBERS))


@pytest.fixture(scope="session")
def radar_dataset():
    return load_radar(RADAR_PATH)


@pytest.fixture(scope="session")
def pre_letkf_dataset(radar_dataset, state_dataset, target_tile):
    device = torch.device("cpu")
    results = pre_letkf(radar_dataset, {target_tile: state_dataset}, device=device, chunk_size=2048)
    return results[target_tile]
