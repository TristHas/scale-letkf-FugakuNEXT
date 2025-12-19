import math
from tqdm.auto import tqdm
import torch
from .structs import CoreBatchInputs, CoreBatchOutputs
from .params import LETKF_CONSTANTS

BATCH = 50_000
SIGMA_B = 0.04
MEMBER = int(LETKF_CONSTANTS["MEMBER"])
EYE = None  # lazy-initialised identity cache

def _identity(device: torch.device) -> torch.Tensor:
    global EYE
    if EYE is None or EYE.device != device:
        EYE = torch.eye(MEMBER, dtype=torch.float64, device=device)
    return EYE


def letkf_core_torch(batch: CoreBatchInputs) -> CoreBatchOutputs:
    """Vectorised LETKF core following tools.letkf_core.letkf_core."""

    device = batch.hdxf.device
    mask = batch.obs_mask
    mask_f = mask.to(torch.float64)
    hdxf = batch.hdxf * mask_f.unsqueeze(-1)
    dep = batch.dep * mask_f
    rloc = batch.rloc * mask_f
    safe_rdiag = torch.where(mask, batch.rdiag, torch.ones_like(batch.rdiag))
    nz = safe_rdiag != 0.0

    inv_rdiag = torch.where(nz, 1.0 / safe_rdiag, torch.zeros_like(safe_rdiag))
    loc_over_rdiag = torch.where(nz, rloc / safe_rdiag, torch.zeros_like(safe_rdiag))

    factors = torch.where(batch.rdiag_wloc[:, None], inv_rdiag, loc_over_rdiag) * mask_f
    hdxb_rinv = hdxf * factors.unsqueeze(-1)

    work1 = torch.bmm(hdxb_rinv.transpose(1, 2), hdxf)
    rho = (MEMBER - 1) / batch.parm_infl
    work1 = work1 + _identity(device).unsqueeze(0) * rho.unsqueeze(-1).unsqueeze(-1)

    eival, eivec = torch.linalg.eigh(work1)
    eival = torch.clamp(eival, min=1.0e-12)
    inv_diag = torch.reciprocal(eival)
    pa = torch.bmm(eivec, inv_diag.unsqueeze(-1) * eivec.transpose(1, 2))

    dep_vec = dep.unsqueeze(-1)
    work2 = torch.bmm(hdxb_rinv.transpose(1, 2), dep_vec).squeeze(-1)
    transm = torch.bmm(pa, work2.unsqueeze(-1)).squeeze(-1)
    trans = _compute_transform(eivec, eival)

    parm_infl = batch.parm_infl.clone()
    if torch.any(batch.infl_update):
        parm_infl = torch.where(
            batch.infl_update,
            _update_inflation(
                parm_infl,
                dep,
                safe_rdiag,
                rloc,
                hdxb_rinv,
                hdxf,
                batch.rdiag_wloc,
                mask_f,
            ),
            parm_infl,
        )

    return CoreBatchOutputs(
        trans=trans,
        transm=transm,
        pa=pa,
        parm_infl=parm_infl,
    )

def letkf_core(batch):
    """
    """
    hdxb_rinv, work1  = comp_work1(batch)
    torch.cuda.empty_cache()
    
    einval, einvec = chunked_eigh(work1)
    einval = torch.clamp(einval, min=1.0e-12)
    inv_diag = torch.reciprocal(einval)
    
    pa = torch.bmm(einvec, inv_diag.unsqueeze(-1) * einvec.transpose(1, 2))
    
    dep_vec = batch.dep.unsqueeze(-1)
    work2 = torch.bmm(hdxb_rinv.transpose(1, 2), dep_vec).squeeze(-1)
    transm = torch.bmm(pa, work2.unsqueeze(-1)).squeeze(-1)
    trans = _compute_transform(einvec, einval)
    
    return CoreBatchOutputs(
        trans=trans,
        transm=transm,
        pa=pa,
        parm_infl=batch.parm_infl,
    )

def comp_work1(batch):
    device = batch.hdxf.device
    mask = batch.obs_mask
    dep = batch.dep
    rloc = batch.rloc 
    hdxf = batch.hdxf
    
    mask_f = mask#.to(torch.float64)
    hdxf *= mask_f.unsqueeze(-1)
    dep  *= mask_f
    rloc *= mask_f
    
    safe_rdiag = torch.where(mask, batch.rdiag, 1)
    nz = safe_rdiag != 0.0
    
    inv_rdiag = torch.where(nz, 1.0 / safe_rdiag, 0)
    loc_over_rdiag = torch.where(nz, rloc / safe_rdiag, 0)
    
    factors = torch.where(batch.rdiag_wloc[:, None], inv_rdiag, loc_over_rdiag) * mask_f
    hdxb_rinv = hdxf * factors.unsqueeze(-1)

    work1 = torch.bmm(hdxb_rinv.transpose(1, 2), hdxf)
    rho = (MEMBER - 1) / batch.parm_infl
    work1 = work1 + _identity(device).unsqueeze(0) * rho.unsqueeze(-1).unsqueeze(-1)

    return hdxb_rinv, work1


def chunked_eigh(work1, batch=BATCH):
    n_batch=math.ceil(work1.shape[0] / batch)
    einval, einvec = zip(*[torch.linalg.eigh(work1[i*batch:(i+1)*batch])
                         for i in tqdm(range(n_batch))])
    return torch.cat(einval), torch.cat(einvec)
def _compute_transform(eivec: torch.Tensor, eival: torch.Tensor) -> torch.Tensor:
    scales = torch.sqrt(torch.clamp((MEMBER - 1) / eival, min=0.0))
    work = eivec * scales.unsqueeze(-2)
    return torch.bmm(work, eivec.transpose(1, 2))


def _update_inflation(
    parm_infl: torch.Tensor,
    dep: torch.Tensor,
    rdiag: torch.Tensor,
    rloc: torch.Tensor,
    hdxb_rinv: torch.Tensor,
    hdxb: torch.Tensor,
    rdiag_wloc: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    dep_sq = dep * dep
    base = torch.where(rdiag != 0.0, dep_sq / rdiag, torch.zeros_like(dep_sq))
    parm1 = torch.where(rdiag_wloc[:, None], base, base * rloc).sum(dim=1)
    parm2 = (hdxb_rinv * hdxb).sum(dim=(1, 2)) / max(MEMBER - 1, 1)
    parm3 = (rloc * mask).sum(dim=1)
    valid = (parm2 != 0.0) & (parm3 != 0.0)
    parm2_safe = torch.where(valid, parm2, torch.ones_like(parm2))
    parm3_safe = torch.where(valid, parm3, torch.ones_like(parm3))
    parm4 = (parm1 - parm3_safe) / parm2_safe - parm_infl
    sigma_o = 2.0 / parm3_safe * ((parm_infl * parm2_safe + parm3_safe) / parm2_safe) ** 2
    gain = SIGMA_B**2 / (sigma_o + SIGMA_B**2)
    updated = parm_infl + gain * parm4
    return torch.where(valid, updated, parm_infl)


__all__ = ["letkf_core_torch"]