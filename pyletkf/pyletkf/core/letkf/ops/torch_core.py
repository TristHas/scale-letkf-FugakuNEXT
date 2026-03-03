import math

from tqdm.auto import tqdm

import torch

def localize_hdx(hdx, rdiag, mask):
    """
        Inputs
            hdx:   n_cells x n_ensemble x n_obs
            rdiag: n_cells x n_ensemble x n_obs
            mask:  n_cells x n_ensemble x n_obs
        Return:
            hdx_loc: n_cells x n_obs x n_ensemble
    """
    nz = rdiag != 0.0
    hdx *= mask.unsqueeze(-1)
    rdiag = torch.where(mask, rdiag, 1)
    inv_rdiag = torch.where(nz, 1. / rdiag, 0)
    hdx_loc = hdx * inv_rdiag.unsqueeze(-1)
    return hdx_loc.transpose(1, 2)

def covariance_hdx(hdx, hdx_loc, param_infl=1):
    """
        Inputs
            hdx:     n_cells x n_ensemble x n_obs
            hdx_loc: n_cells x n_obs x n_ensemble
            param_infl: int
        Return:
            cov: n_cells x n_ensemble x n_ensemble
    """
    n_ens = hdx.shape[-1]
    cov = torch.bmm(hdx_loc, hdx)
    rho = (n_ens - 1) / param_infl
    eye = torch.eye(n_ens, dtype=cov.dtype, device=cov.device)[None]
    cov = cov + eye * rho
    return cov

def letkf_core(dep, hdx, rdiag, mask, 
               param_infl=1, 
               eival_clamp=1.0e-12,
               batch_size=50_000):
    """
        Inputs
            dep:   n_cells x n_ensemble x n_obs
            hdx:   n_cells x n_ensemble x n_obs
            rdiag: n_cells x n_ensemble x n_obs
            mask:  n_cells x n_ensemble x n_obs
            param_infl:  int, 
            eival_clamp: float,
            batch_size:  int):
        Return:
            hdx_loc: n_cells x n_obs x n_ensemble
    """
    hdx_loc = localize_hdx(hdx, rdiag, mask)
    hdx_cov = covariance_hdx(hdx, hdx_loc, param_infl)
    
    eig_val, eig_vec = chunked_eigen_solver(hdx_cov, batch_size)
    eig_val = torch.clamp(eig_val, min=eival_clamp)
    eig_inv = torch.reciprocal(eig_val)

    n = hdx.shape[-1] - 1
    W_a = torch.bmm(eig_vec, ((n*eig_inv).sqrt()).unsqueeze(-1) *\
                               eig_vec.transpose(1, 2))
    
    pa = torch.bmm(eig_vec, eig_inv.unsqueeze(-1) *\
                            eig_vec.transpose(1, 2))
    work2 = torch.bmm(hdx_loc, dep.unsqueeze(-1)).squeeze(-1)
    w_a = torch.bmm(pa, work2.unsqueeze(-1)).squeeze(-1)

    return W_a, w_a

def chunked_eigen_solver(work1: torch.Tensor, 
                         chunk_size: int = 50_000, 
                         device = None):
    """
        work1: [B, N, N] (symmetric / Hermitian per batch item)
        returns:
          einval: [B, N]
          einvec: [B, N, N]
    """
    host_device = work1.device
    device = device or host_device
    B, N, N2 = work1.shape
    assert N == N2, f"Expected square matrices, got {work1.shape}"

    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    def _solve(mat: torch.Tensor):
        mat_dev = mat if mat.device == device else mat.to(device)
        evals, evecs = torch.linalg.eigh(mat_dev)
        return evals.to(host_device), evecs.to(host_device)

    if B <= chunk_size:
        return _solve(work1)
        
    val_dtype = work1.dtype 
    einval_out = work1.new_empty((B, N), dtype=val_dtype, 
                                 device=host_device)
    einvec_out = work1.new_empty((B, N, N), dtype=work1.dtype, 
                                 device=host_device)

    offset = 0
    for w in work1.split(chunk_size, dim=0):
        # w: [b, N, N]
        eival, eivec = _solve(w)
        b = w.shape[0]
        einval_out[offset:offset + b].copy_(eival)
        einvec_out[offset:offset + b].copy_(eivec)
        offset += b

    return einval_out, einvec_out

import torch

@torch.no_grad()
def chunked_letkf_core(dep: torch.Tensor,
                       hdx: torch.Tensor,
                       rdiag: torch.Tensor,
                       mask: torch.Tensor,
                       param_infl: float = 1.0,
                       eival_clamp: float = 1.0e-12,
                       chunk_size: int = 50_000,
                       device=None):
    """
        Chunked wrapper around letkf_core(dep, hdx, rdiag, mask, ...).

        Expected shapes (typical LETKF):
          hdx  : [B, N, N]   (per-batch work matrix / ensemble-space matrix)
          dep  : [B, N] or [B, N, ...]  (must be splittable on dim=0)
          rdiag: [B, N] or [B, 1] or broadcastable on dim=0
          mask : [B, N] or broadcastable on dim=0

        Returns:
          w_a: [B, N]
          W_a: [B, N, N]
    """
    host_device = hdx.device
    device = device or host_device

    B, N, N2 = hdx.shape
    assert N == N2, f"Expected square matrices, got {hdx.shape}"

    if B <= chunk_size:
        if device == host_device:
            return letkf_core(dep, hdx, rdiag, mask,
                              param_infl=param_infl,
                              eival_clamp=eival_clamp)
        return letkf_core(dep.to(device),
                          hdx.to(device),
                          rdiag.to(device),
                          mask.to(device),
                          param_infl=param_infl,
                          eival_clamp=eival_clamp)

    # allocate outputs on host
    w_a = hdx.new_empty((B, N), dtype=hdx.dtype, device=host_device)
    W_a = hdx.new_empty((B, N, N), dtype=hdx.dtype, device=host_device)

    offset = 0
    for sl in (slice(i, min(i + chunk_size, B)) for i in range(0, B, chunk_size)):
        dep_  = dep[sl]
        hdx_  = hdx[sl]
        rdiag_ = rdiag[sl] if rdiag.shape[0] == B else rdiag
        mask_  = mask[sl]  if mask.shape[0]  == B else mask

        if device != host_device:
            dep_   = dep_.to(device)
            hdx_   = hdx_.to(device)
            rdiag_ = rdiag_.to(device)
            mask_  = mask_.to(device)

        W_a_, w_a_ = letkf_core(dep_, hdx_, rdiag_, mask_,
                                param_infl=param_infl,
                                eival_clamp=eival_clamp)

        b = int(hdx_.shape[0])
        w_a[offset:offset + b].copy_(w_a_.to(host_device))
        W_a[offset:offset + b].copy_(W_a_.to(host_device))
        offset += b

    return w_a, W_a
