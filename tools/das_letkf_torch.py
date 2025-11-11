"""Torch-parallel reproduction of ``das_letkf`` for a single PE tile."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping

import numpy as np
import torch

from .das_letkf import (
    DasLetkfIdentifier,
    RankAnalysis,
    _make_letkf_inputs,
    _make_postproc_inputs,
    _normalize_identifier,
)
from .das_replay import ObsLocalReplay
from .letkf_core_torch import letkf_core_torch
from .obs_local_torch import obs_local_torch
from .postproc import _load_background_cube
from .postproc_torch import postproc_torch
from .torch_batches import (
    build_core_batch_from_inputs,
    build_postproc_batch_from_inputs,
)
from .params import GRID_CONSTANTS, LETKF_CONSTANTS


NV3D = int(GRID_CONSTANTS["nv3d"])
MEMBER = int(LETKF_CONSTANTS["MEMBER"])


class ObsLocalTorchReplay(ObsLocalReplay):
    """Replay helper that executes ``obs_local_torch`` instead of NumPy."""

    def __init__(self, dump_dir: Path | str, pe_tag: str, member: str, *, device: torch.device) -> None:
        super().__init__(dump_dir, pe_tag, member)
        self.device = device

    def run_call(self, call_id: int):
        self._advance_to(call_id)
        if self._next_call != call_id:
            raise ValueError(f"Call order mismatch: expected {self._next_call}, got {call_id}")
        meta = self._build_call_meta(call_id)
        inputs = self._build_inputs(meta, include_search_copy=False)
        search_view = self._search_view(meta)
        outputs = obs_local_torch(inputs, search_view, device=self.device)
        self._next_call += 1
        self._last_call_id = call_id
        self._last_outputs = outputs
        self._last_meta = meta
        return outputs, meta


def run_rank_analysis_torch(
    identifier: DasLetkfIdentifier | Mapping[str, object],
    *,
    call_ids: Iterable[int] | None = None,
    max_calls: int | None = None,
    chunk_size: int = 16,
    device: str | torch.device | None = None,
) -> RankAnalysis:
    """Torch analogue of ``run_rank_analysis`` supporting batched LETKF operations."""

    ident = _normalize_identifier(identifier)
    dump_dir = Path(ident.dump_dir)
    torch_device = torch.device(device) if device is not None else torch.device("cpu")
    replay = ObsLocalTorchReplay(dump_dir, ident.pe_tag, ident.member, device=torch_device)
    var_count = max(1, replay.var_count)
    total_calls = replay.nij1 * replay.nlev * var_count

    analysis = np.array(_load_background_cube(dump_dir, ident.pe_tag), copy=True)
    analysis = analysis[:, :, :MEMBER, :NV3D]
    updated_mask = np.zeros((replay.nij1, replay.nlev, NV3D), dtype=bool)

    processed = 0
    call_iter = _iter_call_ids(call_ids, total_calls, max_calls)
    member_tags = tuple(f"mem{idx + 1:04d}" for idx in range(MEMBER))

    buffer: List[_ChunkItem] = []

    def flush_buffer() -> None:
        nonlocal buffer, analysis, updated_mask, processed
        if not buffer:
            return
        core_inputs = [item.core_inputs for item in buffer]
        core_batch = build_core_batch_from_inputs(core_inputs, device=torch_device)
        core_outputs = letkf_core_torch(core_batch)
        post_inputs = []
        for idx, item in enumerate(buffer):
            core_dict = {
                "trans": core_outputs.trans[idx].cpu().numpy(),
                "transm": core_outputs.transm[idx].cpu().numpy(),
                "parm_infl": float(core_outputs.parm_infl[idx].cpu().numpy()),
            }
            pp_inputs = _make_postproc_inputs(
                dump_dir,
                ident,
                item.meta,
                item.core_inputs,
                core_dict,
            )
            post_inputs.append(pp_inputs)
        post_batch = build_postproc_batch_from_inputs(post_inputs, device=torch_device)
        post_outputs = postproc_torch(post_batch)
        for idx, item in enumerate(buffer):
            ij_idx = item.meta.ij - 1
            ilev_idx = item.meta.ilev - 1
            nvar_idx = item.meta.nvar - 1
            analysis[ij_idx, ilev_idx, :, nvar_idx] = post_outputs.anal_members[idx].cpu().numpy()
            updated_mask[ij_idx, ilev_idx, nvar_idx] = True
        processed += len(buffer)
        buffer = []

    for call_id in call_iter:
        outputs, meta = replay.run_call(call_id)
        lc_inputs = _make_letkf_inputs(ident, replay, meta, outputs)
        buffer.append(_ChunkItem(meta=meta, core_inputs=lc_inputs))
        if len(buffer) >= chunk_size:
            flush_buffer()

    flush_buffer()

    if processed == 0:
        raise ValueError("No calls processed; increase max_calls or supply call_ids")

    return RankAnalysis(
        pe_tag=ident.pe_tag,
        analysis=analysis,
        member_tags=member_tags,
        updated_mask=updated_mask,
        processed_call_count=processed,
        total_call_count=total_calls,
    )


@dataclass
class _ChunkItem:
    meta: object
    core_inputs: object


def _iter_call_ids(
    call_ids: Iterable[int] | None,
    total_calls: int,
    max_calls: int | None,
) -> Iterable[int]:
    if call_ids is None:
        iterator: Iterable[int] = range(1, total_calls + 1)
    else:
        normalized = sorted({int(cid) for cid in call_ids})
        iterator = normalized
    count = 0
    for call_id in iterator:
        if call_id < 1 or call_id > total_calls:
            raise ValueError(f"call_id {call_id} outside valid range 1..{total_calls}")
        yield call_id
        count += 1
        if max_calls is not None and count >= max_calls:
            break


__all__ = ["run_rank_analysis_torch"]
