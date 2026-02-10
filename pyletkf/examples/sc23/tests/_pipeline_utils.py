from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys

import torch


CURRENT_DIR = Path(__file__).resolve().parent
SC23_DIR = CURRENT_DIR.parent
PKG_ROOT = SC23_DIR.parents[1]
PROJECT_ROOT = CURRENT_DIR.parents[3]

for path in (SC23_DIR, PKG_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from sc23 import SC23GridManager, SC23OfflineSequentialCommManager
from pyletkf import ScaleLetkfConverter, GridInterpolator, ObsOperator, LETKF
from pyletkf.obs_op.radar import ref_operator, vr_operator, filter_ref, filter_vr


DUMP_ROOT = (
    PROJECT_ROOT / "result" / "SC23" / "20210730060030" / "letkf_dump"
)
DEVICE = torch.device("cuda:7")
PATCH_SIZE = 64

GRID = SC23GridManager()
STATE_CONV = ScaleLetkfConverter()
INTERP = GridInterpolator(GRID)
OBS_OP = ObsOperator(
    {4001: ref_operator, 4002: vr_operator},
    {4001: filter_ref, 4002: filter_vr},
)
GLOBAL_LETKF = LETKF()


def make_ctx(rank: int) -> SC23OfflineSequentialCommManager:
    ctx = SC23OfflineSequentialCommManager(rank, GRID)
    ctx.device = DEVICE
    return ctx


@lru_cache(maxsize=32)
def _local_products(rank: int):
    ctx = make_ctx(rank)
    obs = ctx.read_obs().to(DEVICE)
    obs = ctx.populate_obs_coords(obs)
    obs = ctx.filter_obs_to_tile(obs)
    if obs is None or obs.sizes.get("obs", 0) == 0:
        return None, None
    state = ctx.read_state().to(DEVICE)
    state = ctx.share_halos(state)
    state_da = STATE_CONV.model_to_da(state)
    obs_state = INTERP.map_state_to_obs(obs, state_da)
    obs_proc = OBS_OP(obs_state)
    obs_proc = OBS_OP.filter(obs_proc)
    state_core = ctx.strip_halo(state_da)
    return obs_proc.to("cpu"), state_core.to("cpu")


def _select_patch(dataset):
    if dataset is None or dataset.sizes.get("obs", 0) == 0:
        return dataset
    ri = dataset["ri_local"].values.to(torch.float64)
    rj = dataset["rj_local"].values.to(torch.float64)
    mask = (
        (ri >= 0.0)
        & (ri < PATCH_SIZE)
        & (rj >= 0.0)
        & (rj < PATCH_SIZE)
    )
    idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
    if idx.numel() == 0:
        return dataset
    return dataset.isel(obs=idx)


def build_shared_context(rank: int, obs_limit: int | None = None):
    ctx = make_ctx(rank)
    local, state_core = _local_products(rank)
    if local is None or state_core is None:
        raise RuntimeError(f"Rank {rank} produced no observations")
    local = _select_patch(local)
    state_core = state_core.isel(
        x=slice(0, PATCH_SIZE), y=slice(0, PATCH_SIZE)
    )
    neighbors = []
    for nbr, _, _ in ctx.neighbor_ranks():
        obs_nb, _ = _local_products(nbr)
        if obs_nb is not None:
            neighbors.append(_select_patch(obs_nb))
    shared = ctx.share_obs(
        local.to(DEVICE),
        [nb.to(DEVICE) for nb in neighbors],
    )
    if obs_limit is not None:
        shared = shared.isel(obs=slice(0, obs_limit))
    return ctx, shared, state_core.to(DEVICE)


def _cache_key(obs_limit: int | None) -> int:
    return -1 if obs_limit is None else int(obs_limit)


@lru_cache(maxsize=8)
def _cached_obs_state(rank: int, obs_limit_key: int):
    obs_limit = None if obs_limit_key < 0 else obs_limit_key
    _, shared, state_core = build_shared_context(rank, obs_limit)
    da_state = INTERP.map_obs_to_state(shared, state_core)
    return state_core.to("cpu"), da_state.to("cpu")


def get_state_and_da(rank: int, obs_limit: int | None = None):
    key = _cache_key(obs_limit)
    state_core, da_state = _cached_obs_state(rank, key)
    return state_core.to(DEVICE), da_state.to(DEVICE)


def cell_index(ij: int, ilev: int, nlev: int) -> int:
    horiz = ij - 1
    lev = ilev - 1
    return lev + horiz * nlev


def ij_to_yx(ij: int) -> tuple[int, int]:
    idx0 = ij - 1
    nx = GRID.tiles_size_x
    iy = idx0 // nx
    ix = idx0 % nx
    return int(iy), int(ix)
