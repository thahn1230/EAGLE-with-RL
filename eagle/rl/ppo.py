# eagle/rl/ppo.py
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from .actions import ActionSpace
from .constraints import mask_k_expand, mask_R, compute_K


class CategoricalMasked(torch.distributions.Categorical):
    def __init__(self, logits: torch.Tensor, mask: torch.Tensor | None = None):
        if mask is not None:
            # 모든 항목이 False면 전체 -inf가 되어 NaN이 날 수 있음 → 폴백 처리
            if mask.dtype != torch.bool:
                mask = mask.bool()
            if not torch.any(mask):
                super().__init__(logits=logits)  # 폴백(마스크 미적용)
                return
            logits = logits.masked_fill(~mask, float('-inf'))
            # 혹시 전부 -inf가 되면 폴백
            if not torch.isfinite(logits).any():
                super().__init__(logits=logits.new_zeros(logits.shape))
                return
        super().__init__(logits=logits)


class Policy(nn.Module):
    def __init__(self, state_dim: int, action_space: ActionSpace):
        super().__init__()
        self.action_space = action_space
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 512), nn.Tanh(),
            nn.Linear(512, 512), nn.Tanh(),
        )
        self.head_depth  = nn.Linear(512, action_space.sizes()["depth"])
        self.head_kmain  = nn.Linear(512, action_space.sizes()["k_main"])
        self.head_kexp   = nn.Linear(512, action_space.sizes()["k_expand"])
        self.head_R      = nn.Linear(512, action_space.sizes()["R"])
        self.value_head  = nn.Linear(512, 1)

    def forward(self, s: torch.Tensor) -> Dict[str, torch.Tensor]:
        if s.dim() == 1: 
            s = s.unsqueeze(0)
        h = self.backbone(s)
        return {
          "depth":   self.head_depth(h),
          "k_main":  self.head_kmain(h),
          "k_expand":self.head_kexp(h),
          "R":       self.head_R(h),
          "value":   self.value_head(h).squeeze(-1)
        }

    @torch.no_grad()
    def act(self, s: torch.Tensor):
        """
        순차 샘플링 + 제약 마스킹:
          depth → k_main → (mask) k_expand → (mask) R
        """
        out = self.forward(s)  # B=1 가정
        device = s.device

        # depth
        ld = out["depth"].squeeze(0)
        dist_d = CategoricalMasked(logits=ld)
        idx_d = dist_d.sample()
        logp_d = dist_d.log_prob(idx_d)
        d_val = self.action_space.depth_choices[idx_d.item()]

        # k_main
        lkm = out["k_main"].squeeze(0)
        dist_km = CategoricalMasked(logits=lkm)
        idx_km = dist_km.sample()
        logp_km = dist_km.log_prob(idx_km)
        km_val = self.action_space.k_main_choices[idx_km.item()]

        # k_expand (mask: ke ≤ km)
        lke_raw = out["k_expand"].squeeze(0)
        ke_mask_bool = mask_k_expand(self.action_space.k_expand_choices, km_val)
        ke_mask = torch.tensor(ke_mask_bool, device=device, dtype=torch.bool)
        dist_ke = CategoricalMasked(logits=lke_raw, mask=ke_mask)
        idx_ke = dist_ke.sample()
        logp_ke = dist_ke.log_prob(idx_ke)
        ke_val = self.action_space.k_expand_choices[idx_ke.item()]

        # R (mask: R ≤ K(depth,km,ke))
        lR_raw = out["R"].squeeze(0)
        R_mask_bool = mask_R(self.action_space.R_choices, d_val, km_val, ke_val)
        R_mask = torch.tensor(R_mask_bool, device=device, dtype=torch.bool)
        dist_R = CategoricalMasked(logits=lR_raw, mask=R_mask)
        idx_R = dist_R.sample()
        logp_R = dist_R.log_prob(idx_R)

        logp = (logp_d + logp_km + logp_ke + logp_R).detach()
        v = out["value"].squeeze(0)
        a = torch.stack([idx_d, idx_km, idx_ke, idx_R]).to(device)

        return a, logp, v

    def logprob_value(self, s: torch.Tensor, a: torch.Tensor):
        """
        PPO 재평가(배치): 저장된 action에 대해 동일한 제약 마스킹으로
        depth/k_main/k_expand/R 의 logp를 합산해서 반환.
        """
        out = self.forward(s)  # (B, ...)
        if a.dim() == 1:
            a = a.unsqueeze(0)
        B = a.size(0)
        device = s.device

        logp_list = []
        for i in range(B):
            idx_d  = a[i, 0].long().item()
            idx_km = a[i, 1].long().item()
            idx_ke = a[i, 2].long().item()
            idx_R  = a[i, 3].long().item()

            d_val  = self.action_space.depth_choices[idx_d]
            km_val = self.action_space.k_main_choices[idx_km]
            ke_val = self.action_space.k_expand_choices[idx_ke]
            # R 값은 마스킹만 필요하고 실제 값은 로그확률 계산에 index로만 사용

            # depth
            ld = out["depth"][i]
            dist_d = CategoricalMasked(logits=ld)
            lp_d = dist_d.log_prob(torch.tensor(idx_d, device=device))

            # k_main
            lkm = out["k_main"][i]
            dist_km = CategoricalMasked(logits=lkm)
            lp_km = dist_km.log_prob(torch.tensor(idx_km, device=device))

            # k_expand (mask)
            lke = out["k_expand"][i]
            ke_mask_bool = mask_k_expand(self.action_space.k_expand_choices, km_val)
            ke_mask = torch.tensor(ke_mask_bool, device=device, dtype=torch.bool)
            dist_ke = CategoricalMasked(logits=lke, mask=ke_mask)
            lp_ke = dist_ke.log_prob(torch.tensor(idx_ke, device=device))

            # R (mask)
            lR = out["R"][i]
            R_mask_bool = mask_R(self.action_space.R_choices, d_val, km_val, ke_val)
            R_mask = torch.tensor(R_mask_bool, device=device, dtype=torch.bool)
            dist_R = CategoricalMasked(logits=lR, mask=R_mask)
            lp_R = dist_R.log_prob(torch.tensor(idx_R, device=device))

            logp_list.append(lp_d + lp_km + lp_ke + lp_R)

        logp = torch.stack(logp_list, dim=0)
        v = out["value"].squeeze(-1)
        return logp, v


def ppo_update(policy: Policy, buffer, cfg):
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    _, _ = buffer.compute_gae(cfg.gamma, cfg.gae_lambda)
    S, A, old_logP, returns, adv = buffer.get()
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

    B = S.size(0)
    idx = torch.randperm(B, device=S.device)
    S, A, old_logP, returns, adv = S[idx], A[idx], old_logP[idx], returns[idx], adv[idx]

    for _ in range(cfg.update_epochs):
        for start in range(0, B, cfg.minibatch_size):
            end = start + cfg.minibatch_size
            s_b, a_b = S[start:end], A[start:end]
            old_logp_b, ret_b, adv_b = old_logP[start:end], returns[start:end], adv[start:end]

            # === 동일 마스킹으로 재평가 ===
            logp_b, v_b = policy.logprob_value(s_b, a_b)

            ratio = (logp_b - old_logp_b).exp()
            surr1 = ratio * adv_b
            surr2 = torch.clamp(ratio, 1.0 - cfg.ppo_clip, 1.0 + cfg.ppo_clip) * adv_b
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(v_b, ret_b)

            # (선택) 마스크된 분포의 entropy를 추가할 수 있음. 일단 0으로 둡니다.
            entropy_term = 0.0

            loss = policy_loss + cfg.value_coef * value_loss + cfg.entropy_coef * entropy_term
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()
