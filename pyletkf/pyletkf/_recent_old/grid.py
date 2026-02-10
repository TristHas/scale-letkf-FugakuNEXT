import torch
import xtensor as xt

class GridManager():
    def __init__(self, 
                 n_tiles_x=4, 
                 n_tiles_y=5, 
                 tiles_size_x=320, 
                 tiles_size_y=256, 
                 dx=100., dy=100.,
                 halo_x=1, halo_y=1
                 ):
        """
        """
        self.n_tiles_x = n_tiles_x
        self.n_tiles_y = n_tiles_y
        
        self.tiles_size_x = tiles_size_x
        self.tiles_size_y = tiles_size_y
        
        self.dx = dx
        self.dy = dy
        
        self.total_n_x = self.n_tiles_x * self.tiles_size_x
        self.total_n_y = self.n_tiles_y * self.tiles_size_y

        self.halo_x = halo_x
        self.halo_y = halo_y

    def __getitem__(self, idx):
        """
        """
        pass
        
    def proj_fn(self, lon, lat):
        """
        """
        raise NotImplementedError
    
    def neighbor_ranks(self, rank, k_neighbors=1):
        """
        """
        tile_x, tile_y = self.tile_idx_from_rank(rank)
        diffs = list(range(-k_neighbors,k_neighbors+1))
        neighbors = []
        for dx in diffs:
            nx = tile_x + dx
            if nx < 0 or nx >= self.n_tiles_x:
                continue
            for dy in diffs:
                ny = tile_y + dy
                if ny < 0 or ny >= self.n_tiles_y:
                    continue
                if dx == 0 and dy == 0:
                    continue
                neighbors.append((self.rank_from_tile_idx(nx, ny), dx, dy))
        return neighbors
        
    def tile_idx_from_rank(self, rank):
        """
        """
        tile_x = rank % self.n_tiles_x
        tile_y = rank // self.n_tiles_x
        return tile_x, tile_y

    def rank_from_tile_idx(self, tile_x, tile_y):
        """
        """
        return tile_x + tile_y * self.n_tiles_x

    def tile_bounds(self, rank, halo_x=1, halo_y=1):
        """
        """
        tile_x, tile_y = self.tile_idx_from_rank(rank)
        return self._tile_bounds(tile_x, tile_y, halo_x, halo_y)

    def tile_coords(self, rank, halo_x=1, halo_y=1, device="cpu"):
        """
        """
        start_x, end_x, start_y, end_y = self.tile_bounds(rank, halo_x, halo_y)
        x = (torch.arange(start_x, end_x, device=device) -.5 ) * self.dx
        y = (torch.arange(start_y, end_y, device=device) -.5 ) * self.dy
        return x, y
        
    def _tile_bounds(self, tile_x, tile_y, halo_x, halo_y):
        """
        """
        start_x = tile_x * self.tiles_size_x - halo_x
        end_x   = (tile_x + 1) * self.tiles_size_x + halo_x
        start_y = tile_y * self.tiles_size_y - halo_y
        end_y   = (tile_y + 1) * self.tiles_size_y + halo_y
        return start_x, end_x, start_y, end_y

    def compute_obs_grid_coords(self, obs):
        """
        """
        return self.proj_fn(
            obs["lon"].data,
            obs["lat"].data,
        )

    def compute_obs_tile_coords(self, obs, rank, halo_x=1, halo_y=1):
        """
        """
        start_x, _, start_y, _ = self.tile_bounds(rank, halo_x, halo_y)
        ri_local = obs["ri_global"] - start_x
        rj_local = obs["rj_global"] - start_y
        return ri_local, rj_local
        
    def filter_obs_to_tile(self, 
                           obs: xt.Dataset,
                           rank: int,
                           halo_x=1, halo_y=1) -> xt.Dataset | None:
        """
        """
        ri_min, ri_max, rj_min, rj_max = self.tile_bounds(rank, 
                                                          halo_x=halo_x, 
                                                          halo_y=halo_y)
        ri_global = obs["ri_global"].data 
        rj_global = obs["rj_global"].data
        mask = (
              (ri_global >= ri_min)
            & (ri_global <= ri_max)
            & (rj_global >= rj_min)
            & (rj_global <= rj_max)
        )
        subset = obs.isel(obs=mask)
        return subset

    def populate_obs_coords(self, obs: xt.Dataset,
                            rank=None, halo_x=1, halo_y=1) -> xt.Dataset:
        """
        """
        ri, rj = self.proj_fn(
              obs["lon"].data,
              obs["lat"].data,
        )
        obs["ri_global"]=(("obs",), ri)
        obs["rj_global"]=(("obs",), rj)

        if rank is not None:
            start_x, _, start_y, _ = self.tile_bounds(rank, halo_x, halo_y)
            obs["ri_local"] = obs["ri_global"] - start_x
            obs["rj_local"] = obs["rj_global"] - start_y

        return obs
