import math
from tqdm.auto import tqdm
import torch
from xtensor import DataTensor, Dataset

from .params import MEMBERS, SIGMA_B

MEMBER = len(MEMBERS)
BATCH = 50_000
EYE = None  # lazy-initialised identity cache

def _identity(device: torch.device) -> torch.Tensor:
    global EYE
    if EYE is None or EYE.device != device:
        EYE = torch.eye(MEMBER, dtype=torch.float64, device=device)
    return EYE

CORE_VAR_DIMS = {
    "hdxf": ("batch", "obs", "member"),
    "dep": ("batch", "obs"),
    "rloc": ("batch", "obs"),
    "rdiag": ("batch", "obs"),
    "obs_mask": ("batch", "obs"),
    "parm_infl": ("batch",),
    "rdiag_wloc": ("batch",),
    "infl_update": ("batch",),
}

def _default_coords(shape, dims, device):
    coords = {}
    for axis, dim in enumerate(dims):
        coords[dim] = torch.arange(shape[axis], dtype=torch.float64, device=device)
    return coords


def _ensure_datatensor(value: torch.Tensor, dims: tuple[str, ...]) -> DataTensor:
    coords = _default_coords(value.shape, dims, value.device)
    return DataTensor(value, coords, dims)


def _ensure_core_dataset(batch) -> Dataset:
    if isinstance(batch, Dataset):
        return batch
    data_vars = {}
    for name, dims in CORE_VAR_DIMS.items():
        tensor = getattr(batch, name, None)
        if tensor is None:
            continue
        data_vars[name] = _ensure_datatensor(tensor, dims)
    coords = {}
    if "hdxf" in data_vars:
        for dim in data_vars["hdxf"].dims:
            coords[dim] = data_vars["hdxf"].coords[dim]
    return Dataset(data_vars, coords=coords)


def _tensor(ds: Dataset, name: str) -> torch.Tensor:
    return ds[name].data


def _dim_coords(dt: DataTensor, dim: str) -> torch.Tensor:
    coord = dt.coords.get(dim)
    if coord is None:
        size = dt.sizes[dim]
        return torch.arange(size, dtype=torch.float64, device=dt.data.device)
    if isinstance(coord, torch.Tensor):
        return coord.to(dt.data.device)
    return torch.as_tensor(coord, dtype=torch.float64, device=dt.data.device)


def letkf_core_torch(batch) -> Dataset:
    """Vectorised LETKF core following tools.letkf_core.letkf_core."""

    ds = _ensure_core_dataset(batch)

    hdxf_dt = ds["hdxf"]
    batch_dim, obs_dim, member_dim = hdxf_dt.dims
    batch_coords = _dim_coords(hdxf_dt, batch_dim)
    member_coords = _dim_coords(hdxf_dt, member_dim)

    device = hdxf_dt.data.device

    mask = ds["obs_mask"].data
    mask_f = mask.to(torch.float64)
    hdxf = hdxf_dt.data * mask_f.unsqueeze(-1)
    dep = ds["dep"].data * mask_f
    rloc = ds["rloc"].data * mask_f
    safe_rdiag = torch.where(mask, ds["rdiag"].data, torch.ones_like(ds["rdiag"].data))
    nz = safe_rdiag != 0.0

    inv_rdiag = torch.where(nz, 1.0 / safe_rdiag, torch.zeros_like(safe_rdiag))
    loc_over_rdiag = torch.where(nz, rloc / safe_rdiag, torch.zeros_like(safe_rdiag))

    factors = torch.where(ds["rdiag_wloc"].data[:, None], inv_rdiag, loc_over_rdiag) * mask_f
    hdxb_rinv = hdxf * factors.unsqueeze(-1)

    work1 = torch.bmm(hdxb_rinv.transpose(1, 2), hdxf)
    rho = (MEMBER - 1) / ds["parm_infl"].data
    work1 = work1 + _identity(device).unsqueeze(0) * rho.unsqueeze(-1).unsqueeze(-1)

    eival, eivec = torch.linalg.eigh(work1)
    eival = torch.clamp(eival, min=1.0e-12)
    inv_diag = torch.reciprocal(eival)
    pa = torch.bmm(eivec, inv_diag.unsqueeze(-1) * eivec.transpose(1, 2))

    dep_vec = dep.unsqueeze(-1)
    work2 = torch.bmm(hdxb_rinv.transpose(1, 2), dep_vec).squeeze(-1)
    transm = torch.bmm(pa, work2.unsqueeze(-1)).squeeze(-1)
    trans = _compute_transform(eivec, eival)

    parm_infl = ds["parm_infl"].data.clone()
    if torch.any(ds["infl_update"].data):
        parm_infl = torch.where(
            ds["infl_update"].data,
            _update_inflation(
                parm_infl,
                dep,
                safe_rdiag,
                rloc,
                hdxb_rinv,
                hdxf,
                ds["rdiag_wloc"].data,
                mask_f,
            ),
            parm_infl,
        )

    row_dim = f"{member_dim}_row"
    col_dim = f"{member_dim}_col"
    result = {
        "trans": DataTensor(
            trans,
            {batch_dim: batch_coords, row_dim: member_coords, col_dim: member_coords},
            (batch_dim, row_dim, col_dim),
        ),
        "transm": DataTensor(
            transm,
            {batch_dim: batch_coords, member_dim: member_coords},
            (batch_dim, member_dim),
        ),
        "pa": DataTensor(
            pa,
            {batch_dim: batch_coords, row_dim: member_coords, col_dim: member_coords},
            (batch_dim, row_dim, col_dim),
        ),
        "parm_infl": DataTensor(parm_infl, {batch_dim: batch_coords}, (batch_dim,)),
    }
    return Dataset(result, coords={batch_dim: batch_coords, member_dim: member_coords})


def letkf_core(batch):
    """
    """
    ds = _ensure_core_dataset(batch)
    hdxb_rinv, work1, batch_coords, member_coords, dims = comp_work1(ds)
    torch.cuda.empty_cache()
    
    einval, einvec = chunked_eigh(work1)
    einval = torch.clamp(einval, min=1.0e-12)
    inv_diag = torch.reciprocal(einval)
    
    pa = torch.bmm(einvec, inv_diag.unsqueeze(-1) * einvec.transpose(1, 2))
    
    dep_vec = ds["dep"].data.unsqueeze(-1)
    work2 = torch.bmm(hdxb_rinv.transpose(1, 2), dep_vec).squeeze(-1)
    transm = torch.bmm(pa, work2.unsqueeze(-1)).squeeze(-1)
    trans = _compute_transform(einvec, einval)
    
    batch_dim, obs_dim, member_dim = dims
    row_dim = f"{member_dim}_row"
    col_dim = f"{member_dim}_col"
    result = {
        "trans": DataTensor(
            trans,
            {batch_dim: batch_coords, row_dim: member_coords, col_dim: member_coords},
            (batch_dim, row_dim, col_dim),
        ),
        "transm": DataTensor(
            transm,
            {batch_dim: batch_coords, member_dim: member_coords},
            (batch_dim, member_dim),
        ),
        "pa": DataTensor(
            pa,
            {batch_dim: batch_coords, row_dim: member_coords, col_dim: member_coords},
            (batch_dim, row_dim, col_dim),
        ),
        "parm_infl": ds["parm_infl"],
    }
    return Dataset(result, coords={batch_dim: batch_coords, member_dim: member_coords})


def comp_work1(batch: Dataset):
    hdxf_dt = batch["hdxf"]
    batch_dim, obs_dim, member_dim = hdxf_dt.dims
    batch_coords = _dim_coords(hdxf_dt, batch_dim)
    member_coords = _dim_coords(hdxf_dt, member_dim)

    device = hdxf_dt.data.device
    mask = batch["obs_mask"].data
    dep = batch["dep"].data
    rloc = batch["rloc"].data 
    hdxf = hdxf_dt.data
    
    mask_f = mask#.to(torch.float64)
    hdxf *= mask_f.unsqueeze(-1)
    dep  *= mask_f
    rloc *= mask_f
    
    safe_rdiag = torch.where(mask, batch["rdiag"].data, torch.ones_like(batch["rdiag"].data))
    nz = safe_rdiag != 0.0
    
    inv_rdiag = torch.where(nz, 1.0 / safe_rdiag, 0)
    loc_over_rdiag = torch.where(nz, rloc / safe_rdiag, 0)
    
    factors = torch.where(batch["rdiag_wloc"].data[:, None], inv_rdiag, loc_over_rdiag) * mask_f
    hdxb_rinv = hdxf * factors.unsqueeze(-1)

    work1 = torch.bmm(hdxb_rinv.transpose(1, 2), hdxf)
    rho = (MEMBER - 1) / batch["parm_infl"].data
    work1 = work1 + _identity(device).unsqueeze(0) * rho.unsqueeze(-1).unsqueeze(-1)

    return hdxb_rinv, work1, batch_coords, member_coords, (batch_dim, obs_dim, member_dim)


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
