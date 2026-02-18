import math

from tqdm.auto import tqdm

import torch

def localize_hdx(hdx, rdiag, mask):
    """
    """
    nz = rdiag != 0.0
    hdx *= mask.unsqueeze(-1)
    rdiag = torch.where(mask, rdiag, 1)
    inv_rdiag = torch.where(nz, 1. / rdiag, 0)
    hdx_loc = hdx * inv_rdiag.unsqueeze(-1)
    return hdx_loc.transpose(1, 2)

def covariance_hdx(hdx, hdx_loc, param_infl=1):
    """
    """
    n_ens = hdx.shape[-1]
    cov = torch.bmm(hdx_loc, hdx)
    rho = (n_ens - 1) / param_infl
    eye = torch.eye(n_ens, device=cov.device)[None]
    cov = cov + eye * rho
    return cov

def letkf_core(dep, hdx, rdiag, mask, 
                param_infl=1, 
                eival_clamp=1.0e-12,
                batch_size=50_000):
    """
    """
    hdx_loc = localize_hdx(hdx, rdiag, mask)
    hdx_cov = covariance_hdx(hdx, hdx_loc, param_infl)
    
    eig_val, eig_vec = chunked_eigen_solver(hdx_cov, batch_size)
    eig_val = torch.clamp(eig_val, min=eival_clamp)
    eig_inv = torch.reciprocal(eig_val)

    n = hdx.shape[-1] - 1
    W_a  = torch.bmm(eig_vec, ((n*eig_inv).sqrt()).unsqueeze(-1) *\
                                 eig_vec.transpose(1, 2))
    
    pa     = torch.bmm(eig_vec,  eig_inv.unsqueeze(-1) *\
                                 eig_vec.transpose(1, 2))
    work2   = torch.bmm(hdx_loc, dep.unsqueeze(-1)).squeeze(-1)
    w_a = torch.bmm(pa, work2.unsqueeze(-1)).squeeze(-1)

    return W_a, w_a

def chunked_eigen_solver(work1, chunk_size=50_000):
    """
    """
    n_chunks = math.ceil(work1.shape[0] / chunk_size)
    eigen_pairs = [
        torch.linalg.eigh(work1[i * chunk_size:(i + 1) * chunk_size])
        for i in tqdm(range(n_chunks), disable=(n_chunks == 1))
    ]
    einval, einvec = zip(*eigen_pairs)
    return torch.cat(einval), torch.cat(einvec)
