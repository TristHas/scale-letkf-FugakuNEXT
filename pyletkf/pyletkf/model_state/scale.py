import pandas as pd
import torch
import xtensor as xt
from .ops import scale_to_letkf, letkf_to_scale

LETKF_VARIABLES = pd.Index(("U", "V", "W", "T", "P", "QV", "QC", "QR", "QI", "QS", "QG"), name="variable")
SCALE_VARIABLES = pd.Index(("DENS", "RHO", "MOMX", "MOMY", "MOMZ", "QV", "QC", "QR", "QI", "QS", "QG"), name="variable")

class ScaleLetkfConverter():
    def __init__(self):
        """
        """
        self.rdry = 287.04
        self.cpdry = 1004.64
        self.cvdry = self.cpdry - self.rdry
        
        self.rvap = 461.50
        self.cpvap = 1846.00
        self.cvvap = self.cpvap - self.rvap
        
        self.pre00 = 100000.0
        self.fill  = -9.9999e30
        
    def model_to_da(self, state):
        """
        """
        scale_state = state["state"].values.permute(1, 0, 2, 3, 4).contiguous()
        letkf_state = scale_to_letkf(scale_state, self.rdry, self.cvvap, 
                                     self.rvap, self.cvdry, self.pre00)
        
        letkf_state = letkf_state.permute(1, 0, 2, 3, 4).contiguous()
        out_state = state.assign_coords(variable=LETKF_VARIABLES)
        out_state["state"] = (state["state"].dims, letkf_state)
        return out_state

    def da_to_model(self, state):
        """
        """
        letkf_state = state["state"].values.transpose(0,1).contiguous()
        scale_state = letkf_to_scale(letkf_state, self.rdry, self.cvvap, 
                                     self.rvap, self.cvdry, self.pre00)
        
        out_state = state.assign_coords(variable=SCALE_VARIABLES)
        out_state["height"]=state["height"]
        out_state["state"]= (state["state"].dims, scale_state.transpose(0,1).contiguous())
        return out_state