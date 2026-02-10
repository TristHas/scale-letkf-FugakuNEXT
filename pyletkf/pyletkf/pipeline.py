from __future__ import annotations

import math
from typing import Callable

import torch
import xtensor as xt
from tqdm.auto import tqdm
from xtensor import DataTensor, Dataset

class Pipeline():
    """
    """
    def __init__(self, obs_op, interp, core, state_conv, logger, sequential=False):
        """
        """
        self.obs_op = obs_op
        self.interp = interp
        self.core = core
        self.state_conv = state_conv
        self.logger = logger
        self.sequential = False

    def pre_letkf_timed(self, tile, device):
        with self.logger.time("Reading observations"):
            obs = tile.read_obs().to(tile.device)
            obs = tile.populate_obs_coords(obs)
            obs = tile.filter_obs_to_tile(obs)
        
        with self.logger.time("Loading state"):
            state = tile.read_state().to(tile.device)
        
        with self.logger.time("Sharing halos"):
            state = tile.share_halos(state)
        
        with self.logger.time("State Conversion"):
            state = self.state_conv.model_to_da(state)
        
        with self.logger.time("Interpolation"):
            obs_state = self.interp.map_state_to_obs(obs.to(tile.device), state)
        
        with self.logger.time("Halo Striping"):
            state = tile.strip_halo(state)
        
        with self.logger.time("Observation Operator"):
            obs = self.obs_op(obs_state.to(tile.device))
        
        with self.logger.time("Obs Filter"):
            obs = self.obs_op.filter(obs)
            logger.info("Remaining observations: %d", obs["obs"].shape[0])
        
        return state, obs
        
    def run_timed(self, tile, device):
        """
        """
        device = device or torch.device("cpu")
        state, obs = self.pre_letkf(tile, device)
        
        with self.logger.time("Obs Sharing"):
            obs = tile.share_obs(obs)
            logger.info("Shared observations: %d", obs["obs"].shape[0])
        
        with self.logger.time("Obs mapping"):
            da_state = self.interp.map_obs_to_state(obs, state)
        
        with self.logger.time("LETKF Solver"):
            da_params = self.core.infer_update_params(da_state)
        
        with self.logger.time("LETKF Update"):
            output = self.core.apply_update(state, da_params)
        
        return output

    def share_obs(self, tile, obs):
        """
        """
        if self.sequential:
            pass
        else:
            return tile.share_obs(obs)

    def share_halos(self, tile, state):
        """
        """
        if self.sequential:
            pass
        else:
            return tile.share_halos(state)