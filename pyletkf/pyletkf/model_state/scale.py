import pandas as pd
import torch
import xtensor as xt
from .ops import scale_to_letkf, letkf_to_scale

LETKF_VARIABLES = pd.Index(("U", "V", "W", "T", "P", "QV", "QC", "QR", "QI", "QS", "QG"), name="variable")
SCALE_VARIABLES = pd.Index(("DENS", "RHO", "MOMX", "MOMY", "MOMZ", "QV", "QC", "QR", "QI", "QS", "QG"), name="variable")

class ScaleLetkfConverter():
    def __init__(self, device=None, chunk_size=10**9):
        """
        """
        self.rdry  = 287.04
        self.cpdry = 1004.64
        self.cvdry = self.cpdry - self.rdry
        
        self.rvap  = 461.50
        self.cpvap = 1846.00
        self.cvvap = self.cpvap - self.rvap
        
        self.pre00 = 100000.0
        self.fill  = -9.9999e30

        self.device = device
        self.chunk_size = chunk_size
        
    def to(device):
        """
        """
        self.device=device
        return self
        
    def model_to_da(self, state):
        """
        """
        scale_state = state["state"].values
        letkf_state = self.chunked_apply(scale_state, scale_to_letkf)
        out_state = state.assign_coords(variable=LETKF_VARIABLES)
        out_state["height"]= (state["height"].dims, state["height"].values)
        out_state["state"] = (state["state"].dims, letkf_state)
        return out_state

    def da_to_model(self, state):
        """
        """
        letkf_state = state["state"].values
        scale_state = self.chunked_apply(letkf_state, letkf_to_scale)
        out_state = state.assign_coords(variable=SCALE_VARIABLES)
        out_state["height"]= (state["height"].dims, state["height"].values)
        out_state["state"] = (state["state"].dims, scale_state)
        return out_state

    def chunked_apply(self, state, fn):
        """
        """
        host_device = state.device
        device = self.device or host_device
        chunk_size = self.chunk_size
        if state.shape[0] <= chunk_size:
            return fn(state.to(device),
                      self.rdry, self.cvvap,
                      self.rvap, self.cvdry,
                      self.pre00).to(host_device)

        output = torch.empty_like(state)
        offset = 0
        for chunk in torch.split(state, chunk_size, dim=0):
            chunk_size_local = chunk.shape[0]
            transformed = fn(chunk.to(device),
                             self.rdry, self.cvvap,
                             self.rvap, self.cvdry,
                             self.pre00).to(host_device)
            output[offset:offset + chunk_size_local] = transformed
            offset += chunk_size_local
        return output
