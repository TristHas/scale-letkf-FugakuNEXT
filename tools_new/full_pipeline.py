from __future__ import annotations
from tqdm.auto import tqdm

from pathlib import Path
import numpy as np
import time
import torch

from tools_new.io.letkf_dumps import load_obsda_sorted
from tools_new.io.radar import load_radar
from tools_new.io.state import load_letkf_state
from tools_new.pre_letkf import pre_letkf_pipeline

# Important: move to tools_new implem instead (same but probably need some import debugs, etc.)
from tools_new.letkf_core import letkf_core
from tools.postproc_torch import postproc_torch
from tools_new.structs import CoreBatchInputs, PostprocBatchInputs
from tools_new.params import MEMBERS

def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def _build_core_batch(results: dict[str, np.ndarray], device: torch.device) -> CoreBatchInputs:
    hdxf = torch.from_numpy(results["hdxf"]).to(device=device, dtype=torch.float64)
    dep = torch.from_numpy(results["dep"]).to(device=device, dtype=torch.float64)
    rloc = torch.from_numpy(results["rloc"]).to(device=device, dtype=torch.float64)
    rdiag = torch.from_numpy(results["rdiag"]).to(device=device, dtype=torch.float64)
    obs_mask = torch.from_numpy(results["mask"]).to(device=device, dtype=torch.bool)
    batch_size = hdxf.shape[0]

    parm_infl = torch.ones(batch_size, dtype=torch.float64, device=device)
    rdiag_wloc = torch.ones(batch_size, dtype=torch.bool, device=device)
    infl_update = torch.zeros(batch_size, dtype=torch.bool, device=device)
    call_ids = torch.arange(batch_size, dtype=torch.int64, device=device)

    return CoreBatchInputs(
        hdxf=hdxf,
        dep=dep,
        rdiag=rdiag,
        rloc=rloc,
        obs_mask=obs_mask,
        parm_infl=parm_infl,
        rdiag_wloc=rdiag_wloc,
        infl_update=infl_update,
        call_ids=call_ids,
    )

def _extract_background_members(state_ds, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    state = (
        state_ds["state"]
        .transpose("z", "y", "x", "ens", "variable")
        .sel(ens=list(MEMBERS))
        .values
    )
    members = torch.from_numpy(state.reshape(-1, len(MEMBERS), state.shape[-1])).to(device=device, dtype=torch.float64)
    mean = members.mean(dim=1)
    return members, mean

def _build_post_batch(
        core_outputs,
        gues_members: torch.Tensor,
        gues_mean: torch.Tensor,
        call_ids: torch.Tensor,
    ) -> PostprocBatchInputs:
    device = gues_members.device
    batch_size = gues_members.shape[0]
    zeros = torch.zeros(batch_size, dtype=torch.float64, device=device)

    return PostprocBatchInputs(
        trans=core_outputs.trans,
        transm=core_outputs.transm,
        gues_members=gues_members,
        gues_mean=gues_mean,
        beta=torch.ones(batch_size, dtype=torch.float64, device=device),
        parm=core_outputs.parm_infl,
        relax_alpha=zeros,
        relax_alpha_spread=zeros,
        relax_to_inflated=torch.zeros(batch_size, dtype=torch.bool, device=device),
        relax_spread_out=torch.zeros(batch_size, dtype=torch.bool, device=device),
        det_run=torch.zeros(batch_size, dtype=torch.bool, device=device),
        nvar=torch.zeros(batch_size, dtype=torch.int64, device=device),
        call_ids=call_ids,
    )
    
def full_pipeline(obs, state_ds, pe_tag, device="cuda:1"):
    nz, ny, nx = 45, 256, 320
    
    pre_results = pre_letkf_pipeline(obs, state_ds, pe_tag, device=device)
    core_batch = _build_core_batch(pre_results, device)
    torch.cuda.empty_cache()
    
    core_outputs = letkf_core(core_batch)
    torch.cuda.empty_cache()
    
    gues_members, gues_mean = _extract_background_members(state_ds, device)
    torch.cuda.empty_cache()
    
    post_batch = _build_post_batch(core_outputs, gues_members, gues_mean, core_batch.call_ids)
    post_outputs = postproc_torch(post_batch)

    output = post_outputs.anal_members.view(nz, ny, nx, 2)

    return output
