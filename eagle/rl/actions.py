from typing import Dict, Tuple
import torch

class ActionSpace:
    def __init__(self, depth_choices, k_main_choices, k_expand_choices, R_choices):
        self.depth_choices = depth_choices
        self.k_main_choices = k_main_choices
        self.k_expand_choices = k_expand_choices
        self.R_choices = R_choices

    def sizes(self):
        return {
            "depth": len(self.depth_choices),
            "k_main": len(self.k_main_choices),
            "k_expand": len(self.k_expand_choices),
            "R": len(self.R_choices),
        }

    def decode(self, idx_depth: int, idx_km: int, idx_ke: int, idx_R: int):
        d  = self.depth_choices[idx_depth]
        km = self.k_main_choices[idx_km]
        ke = self.k_expand_choices[idx_ke]
        R  = self.R_choices[idx_R]
        return d, km, ke, R

    def encode(self, d: int, km: int, ke: int, R: int):
        return (
            self.depth_choices.index(d),
            self.k_main_choices.index(km),
            self.k_expand_choices.index(ke),
            self.R_choices.index(R),
        )
