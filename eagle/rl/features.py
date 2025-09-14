import torch
import torch.nn.functional as F
from typing import Dict, Tuple

def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    # p, q: (..., M) probabilities sum to 1
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).log()).sum(dim=-1)
    kl_qm = (q * (q / m).log()).sum(dim=-1)
    js = 0.5 * (kl_pm + kl_qm)
    # Normalize to [0,1] approximately by dividing by log 2
    return (js / torch.log(torch.tensor(2.0, device=js.device))).clamp(0, 1)

def topk_margins(probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # probs: (..., M), returns (p1-p2, p1)
    top2 = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1).values
    if top2.shape[-1] == 1:
        margin = top2[..., 0]
        p1 = top2[..., 0]
    else:
        margin = top2[..., 0] - top2[..., 1]
        p1 = top2[..., 0]
    return margin, p1

def min_k_for_mass(probs: torch.Tensor, tau: float) -> torch.Tensor:
    # probs: (..., M) sorted descending is better but not required
    sorted_p, _ = torch.sort(probs, dim=-1, descending=True)
    csum = torch.cumsum(sorted_p, dim=-1)
    hits = (csum >= tau).float()
    idx = hits.argmax(dim=-1)
    return idx + 1  # k_tau is count, so +1 for index->count

def build_state_from_logits(draft_logits: torch.Tensor,
                            target_logits: torch.Tensor,
                            topM: int,
                            tau: float,
                            delta_al_hist: torch.Tensor) -> torch.Tensor:
    # logits: (V,), select topM
    d_top = torch.topk(draft_logits, k=topM).values
    t_top = torch.topk(target_logits, k=topM).values

    d_idx = torch.topk(draft_logits, k=topM).indices
    t_idx = torch.topk(target_logits, k=topM).indices

    # align indices: gather probs on union? to keep code simple, softmax independently
    d_probs = F.softmax(d_top, dim=-1)
    t_probs = F.softmax(t_top, dim=-1)

    js = js_divergence(d_probs, t_probs).unsqueeze(0)

    m_draft, p1_draft = topk_margins(d_probs)
    m_tgt, p1_tgt = topk_margins(t_probs)

    k_tau_draft = min_k_for_mass(d_probs, tau).float().unsqueeze(0)
    k_tau_tgt   = min_k_for_mass(t_probs, tau).float().unsqueeze(0)

    # delta_al_hist: (H,), include mean/std/last
    if delta_al_hist.numel() == 0:
        mu, sigma, last = 0.0, 0.0, 0.0
    else:
        mu = float(delta_al_hist.float().mean().item())
        sigma = float(delta_al_hist.float().std(unbiased=False).item())
        last = float(delta_al_hist[-1].item())
    trend = torch.tensor([mu, sigma, last], device=draft_logits.device).float()

    state = torch.cat([
        js,
        m_draft.unsqueeze(0), m_tgt.unsqueeze(0),
        p1_draft.unsqueeze(0), p1_tgt.unsqueeze(0),
        k_tau_draft, k_tau_tgt,
        trend
    ], dim=0).float()
    return state  # shape (1 + 1+1 + 1+1 + 1+1 + 3,) = 10 dims
