import torch

class Tile:
    def __init__(self, rank, grid, device=None):
        """
        """
        self.rank = rank
        self.grid = grid
        self.device = torch.device(f"cuda:{rank}") or device
        self._init_halo_comm()

    def to(self, device):
        self.device = device
        return self
        
    def _init_halo_comm(self):
        """
        """
        tile_x, tile_y = self.grid.tile_idx_from_rank(self.rank)
        
        self.pad_x = (
            self.grid.halo_x if tile_x != 0 else 0,
            self.grid.halo_x if tile_x != self.grid.n_tiles_x - 1 else 0,
        )
        self.pad_y = (
            self.grid.halo_y if tile_y != 0 else 0,
            self.grid.halo_y if tile_y != self.grid.n_tiles_y - 1 else 0,
        )

        def _stop(pad):
            return None if pad == 0 else -pad

        self.slice_x = slice(self.pad_x[0], _stop(self.pad_x[1]))
        self.slice_y = slice(self.pad_y[0], _stop(self.pad_y[1]))

        left_halo_x = slice(0, self.pad_x[0])
        right_halo_x = slice(-self.pad_x[1], None) if self.pad_x[1] else slice(0, 0)
        bottom_halo_y = slice(0, self.pad_y[0])
        top_halo_y = slice(-self.pad_y[1], None) if self.pad_y[1] else slice(0, 0)

        self.lookup_self_x = {
            0: self.slice_x,
            -1: left_halo_x,
            1: right_halo_x,
        }
        self.lookup_self_y = {
            0: self.slice_y,
            -1: bottom_halo_y,
            1: top_halo_y,
        }

        left_edge_x = slice(
            self.pad_x[0],
            self.pad_x[0] + (self.grid.halo_x if self.pad_x[0] or self.grid.halo_x else 0),
        )
        right_edge_x = slice(
            -self.grid.halo_x - self.pad_x[1] if (self.grid.halo_x or self.pad_x[1]) else 0,
            -self.pad_x[1] or None,
        )
        bottom_edge_y = slice(
            self.pad_y[0],
            self.pad_y[0] + (self.grid.halo_y if self.pad_y[0] or self.grid.halo_y else 0),
        )
        top_edge_y = slice(
            -self.grid.halo_y - self.pad_y[1] if (self.grid.halo_y or self.pad_y[1]) else 0,
            -self.pad_y[1] or None,
        )

        self.lookup_send_x = {0: self.slice_x, -1: left_edge_x, 1: right_edge_x}
        self.lookup_send_y = {0: self.slice_y, -1: bottom_edge_y, 1: top_edge_y}
        self.lookup_neighbor = {0: None, 1: 0, -1: -1}

    def read_state(self, *args, **kwargs):
        """
        """
        raise NotImplementedError

    def share_halos(self, state):
        """
        """
        raise NotImplementedError
        
    def read_obs(self, *args, **kwargs):
        """
        """
        raise NotImplementedError

    def share_obs(self, obs, *args, **kwargs):
        """
        """
        raise NotImplementedError
        
    def strip_halo(self, ds):
        """
        """
        return ds.isel(x=self.slice_x, y=self.slice_y)

    def populate_obs_coords(self, obs, halo_x=1, halo_y=1):
        """
        """
        return self.grid.populate_obs_coords(obs, rank=self.rank,
                                             halo_x=halo_x,
                                             halo_y=halo_y)

    def filter_obs_to_tile(self, obs):
        """
        """
        return self.grid.filter_obs_to_tile(obs, self.rank)

    def neighbor_ranks(self, k_neighbors=1):
        """
        """
        return self.grid.neighbor_ranks(self.rank, k_neighbors=1)
