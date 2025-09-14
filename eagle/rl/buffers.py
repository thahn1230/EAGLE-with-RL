import torch
from dataclasses import dataclass

@dataclass
class Transition:
    s: torch.Tensor
    a: torch.Tensor
    logp: torch.Tensor
    r: torch.Tensor
    v: torch.Tensor
    done: torch.Tensor

class RolloutBuffer:
    def __init__(self, device="cuda"):
        self.device = device
        self.clear()

    def add(self, s,a,logp,r,v,done):
        self.S.append(s.detach())
        self.A.append(a.detach())
        self.logP.append(logp.detach())
        self.R.append(r.detach())
        self.V.append(v.detach())
        self.D.append(done.detach())

    def compute_gae(self, gamma=0.99, lam=0.95):
        S = torch.stack(self.S).to(self.device)
        A = torch.stack(self.A).to(self.device)
        logP = torch.stack(self.logP).to(self.device)
        R = torch.stack(self.R).to(self.device)
        V = torch.stack(self.V).to(self.device)
        D = torch.stack(self.D).to(self.device)

        # one-step rollouts; but keep GAE API
        advantages = R - V
        returns = R
        self._cached = (S, A, logP, returns, advantages)
        return returns, advantages

    def get(self):
        return self._cached

    def clear(self):
        self.S, self.A, self.logP, self.R, self.V, self.D = [], [], [], [], [], []
        self._cached = None
