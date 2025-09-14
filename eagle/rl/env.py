from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Tuple
import torch
from .features import build_state_from_logits
from .constraints import feasible, compute_K
from .config import RLConfig

@dataclass
class EpisodeResult:
    delta_al: float
    K: int
    R: int
    meters: Dict[str, Any]
    draft_logits: torch.Tensor
    target_logits: torch.Tensor

class EagleEnv:
    def __init__(self, runner, cfg: RLConfig):
        self.runner = runner  # must implement .draft_and_verify(...)
        self.cfg = cfg
        self.delta_hist = deque(maxlen=cfg.delta_al_window)
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    def reset(self, prompt: str) -> torch.Tensor:
        # Obtain initial logits for state before choosing action if needed
        # Many designs use last-step logits; here we let runner provide initial pair
        dlogits, tlogits = self.runner.peek_logits(prompt)
        if isinstance(dlogits, torch.Tensor) is False:
            dlogits = torch.tensor(dlogits, dtype=torch.float32, device=self.device)
        if isinstance(tlogits, torch.Tensor) is False:
            tlogits = torch.tensor(tlogits, dtype=torch.float32, device=self.device)
        state = build_state_from_logits(dlogits, tlogits, self.cfg.topM, self.cfg.tau_mass,
                                        torch.tensor(list(self.delta_hist), device=self.device))
        self._cached_prompt = prompt
        return state.to(self.device)

    def step(self, action: torch.Tensor):
        idx_depth, idx_km, idx_ke, idx_R = action.tolist()
        d = self.runner.action_space.depth_choices[idx_depth]
        km = self.runner.action_space.k_main_choices[idx_km]
        ke = self.runner.action_space.k_expand_choices[idx_ke]
        R  = self.runner.action_space.R_choices[idx_R]

        ok, K = feasible(d, km, ke, R)
        if not ok:
            r = torch.tensor(self.cfg.invalid_penalty, device=self.device)
            return None, r, True, {"invalid": True}

        res: EpisodeResult = self.runner.draft_and_verify(self._cached_prompt, d, km, ke, R)

        # reward
        lv, ld, mu, B = self.cfg.lambda_v, self.cfg.lambda_d, self.cfg.mu_budget, self.cfg.budget_target
        budget = lv*res.R + ld*res.K
        r = res.delta_al - lv*res.R - ld*res.K - mu*((budget - B)**2)
        r = torch.tensor(r, dtype=torch.float32, device=self.device)

        # next state from resulting logits
        state = build_state_from_logits(res.draft_logits, res.target_logits, self.cfg.topM, self.cfg.tau_mass,
                                        torch.tensor(list(self.delta_hist), device=self.device))
        self.delta_hist.append(res.delta_al)
        done = True
        info = {"K": res.K, "R": res.R, **res.meters}
        return state, r, done, info
