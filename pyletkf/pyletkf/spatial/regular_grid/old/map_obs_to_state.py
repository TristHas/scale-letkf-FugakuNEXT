import torch

HORI_LOCAL_RADAR_OBSNOREF = 2000.0
VERT_LOCAL_RADAR_OBSNOREF = 2000.0
DIST_ZERO_FAC = 3.651483717
MAX_OBS_PER_GRID = 100
NN_CHUNK_SIZE = 1024
CELL_SIZE = 2000

def chunked_topk_neighbors(grid_xy, grid_z, obs_xy, obs_z,
                           max_obs_per_grid=MAX_OBS_PER_GRID,
                           horiz_loc=HORI_LOCAL_RADAR_OBSNOREF,
                           vert_loc=VERT_LOCAL_RADAR_OBSNOREF,
                           zero_fac=DIST_ZERO_FAC,
                           chunk_size = NN_CHUNK_SIZE,
                           **kwargs):
    """
    """
    device = grid_xy.device
    dtype = grid_xy.dtype
    topk_vals, topk_idx, valid_masks = [], [], []
    
    for start in range(0, grid_xy.shape[0], chunk_size):
        end = min(start + chunk_size, grid_xy.shape[0])
        chunk_xy = grid_xy[start:end]
        horiz = torch.cdist(chunk_xy, obs_xy) / horiz_loc
        vert  = torch.abs(grid_z[start:end].unsqueeze(1) - obs_z) / vert_loc
        ndist = horiz * horiz + vert * vert
        
        mask = (
              (horiz <= zero_fac)
            & (vert  <= zero_fac)
            & (ndist <= zero_fac * zero_fac)
        )
        masked = torch.where(mask, ndist, torch.full_like(ndist, float("inf")))
        vals, idx = torch.topk(
            masked,
            k=max_obs_per_grid,
            dim=1,
            largest=False,
            sorted=True,
        )
        topk_vals.append(vals)
        topk_idx.append(idx)
        valid_masks.append(torch.isfinite(vals))

    topk_vals = torch.cat(topk_vals, dim=0)
    topk_idx  = torch.cat(topk_idx, dim=0)
    valid_mask= torch.cat(valid_masks, dim=0)

    return topk_vals, topk_idx, valid_mask

def topk_quad(grid_xy, grid_z, obs_xy, obs_z,     
              max_obs_per_grid=MAX_OBS_PER_GRID,
              horiz_loc = HORI_LOCAL_RADAR_OBSNOREF,
              vert_loc  = VERT_LOCAL_RADAR_OBSNOREF,
              zero_fac  = DIST_ZERO_FAC,
              cell_size = CELL_SIZE,
              **kwargs
            ):
    """
    """
    device = obs_xy.device
    dtype = obs_xy.dtype
    n_cells = zero_fac
    nobs = obs_xy.shape[0]
    ngrid = grid_xy.shape[0]

    grid_quad = grid_xy // cell_size
    obs_quad  = obs_xy // cell_size
    
    obs_idxs = torch.arange(nobs, device=device)
    grid_idxs = torch.arange(ngrid, device=device)
    topk_idxs   = -torch.ones(ngrid, max_obs_per_grid, 
                              device=device, dtype=torch.int64)
    topk_vals   = torch.full((ngrid, max_obs_per_grid), float("inf"),
                             device=device, dtype=dtype)
    
    for quad in grid_quad.unique(dim=0):
        msk_obs = (obs_quad - quad).abs().max(1).values <= n_cells
        lobs = obs_xy[msk_obs]
        nlobs = lobs.shape[0]
        
        if nlobs == 0: continue

        msk_grid = (grid_quad == quad).all(1)
        lgrid = grid_xy[msk_grid]
        lgrid_z = grid_z[msk_grid]
        lobs_z  = obs_z[msk_obs]
        
        horiz = torch.cdist(lgrid, lobs) / horiz_loc
        vert  = torch.abs(lgrid_z.unsqueeze(1) - lobs_z) / vert_loc
        ndist = horiz * horiz + vert * vert
        
        mask = (
              (horiz <= zero_fac)
            & (vert  <= zero_fac)
            & (ndist <= zero_fac * zero_fac)
        )
        ndist = torch.where(mask, ndist, torch.full_like(ndist, float("inf")))
        
        k = min(max_obs_per_grid, nlobs)
        if k == 0: continue
            
        vals, idx = torch.topk(ndist, k=k, dim=1, largest=False, sorted=True)
        
        lobs_idxs = obs_idxs[msk_obs][idx]
        lgrid_idxs = grid_idxs[msk_grid]

        topk_idxs[lgrid_idxs, :k] = lobs_idxs
        topk_vals[lgrid_idxs, :k] = vals
            
    mask = ~torch.isinf(topk_vals)
    return topk_vals, topk_idxs, mask

TOPK_FUNC = {
    "quad"    : topk_quad,
    "chunked" : chunked_topk_neighbors,
}