from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple
import numpy as np
import torch

@dataclass
class CoreBatchInputs:
    """Padded batch of obs-local outputs consumed by the LETKF core."""

    hdxf: torch.Tensor        # (batch, max_obs, MEMBER)
    dep: torch.Tensor         # (batch, max_obs)
    rdiag: torch.Tensor       # (batch, max_obs)
    rloc: torch.Tensor        # (batch, max_obs)
    obs_mask: torch.Tensor    # (batch, max_obs) boolean
    parm_infl: torch.Tensor   # (batch,)
    rdiag_wloc: torch.Tensor  # (batch,) boolean
    infl_update: torch.Tensor # (batch,) boolean
    call_ids: torch.Tensor    # (batch,) int64 for bookkeeping

    @property
    def batch_size(self) -> int:
        return int(self.hdxf.shape[0])

    @property
    def max_obs(self) -> int:
        return int(self.hdxf.shape[1])

@dataclass
class CoreBatchOutputs:
    """Batched LETKF core outputs, ready for post-processing."""

    trans: torch.Tensor  # (batch, MEMBER, MEMBER)
    transm: torch.Tensor  # (batch, MEMBER)
    pa: torch.Tensor  # (batch, MEMBER, MEMBER)
    parm_infl: torch.Tensor  # (batch,)

@dataclass
class PostprocBatchInputs:
    """Inputs required by the relaxation/post-processing step."""
    trans: torch.Tensor         # (batch, MEMBER, MEMBER)
    transm: torch.Tensor        # (batch, MEMBER)
    gues_members: torch.Tensor  # (batch, MEMBER)
    gues_mean: torch.Tensor     # (batch,)
    beta: torch.Tensor          # (batch,)
    parm: torch.Tensor          # (batch,)
    relax_alpha: torch.Tensor   # (batch,)
    relax_alpha_spread: torch.Tensor  # (batch,)
    relax_to_inflated: torch.Tensor   # (batch,) boolean
    relax_spread_out: torch.Tensor    # (batch,) boolean
    det_run: torch.Tensor             # (batch,) boolean
    nvar: torch.Tensor                # (batch,) int64
    call_ids: torch.Tensor            # (batch,) int64


@dataclass
class PostprocBatchOutputs:
    """Outputs of the torch post-processing step."""

    anal_members: torch.Tensor  # (batch, MEMBER)
    transrlx: torch.Tensor      # (batch, MEMBER, MEMBER)
    q_mean: torch.Tensor        # (batch,)
    q_sprd: torch.Tensor        # (batch,)
    q_limited: torch.Tensor     # (batch,) boolean
    workda_value: torch.Tensor  # (batch,)
    workda_present: torch.Tensor# (batch,) boolean