from dataclasses import dataclass, field
from typing import List

@dataclass
class RLConfig:
    # ----- Action Space (V0: uniform across depth) -----
    depth_choices: List[int] = field(default_factory=lambda: [2,3,4,5,6,7,8,9,10])
    k_main_choices: List[int] = field(default_factory=lambda: [2,4,8,12,16,20,24,32])
    k_expand_choices: List[int] = field(default_factory=lambda: [2,4,8,12,16,20])
    R_choices: List[int] = field(default_factory=lambda: [8,12,16,20,24,32,40,48,56,64,96,128])

    # ----- Reward weights -----
    lambda_v: float = 0.002  # cost per verification token
    lambda_d: float = 0.001  # cost per draft token
    mu_budget: float = 0.0002  # budget penalty multiplier
    budget_target: float = 140.0  # target for (lambda_v*R + lambda_d*K)

    invalid_penalty: float = -1.0  # penalty if invalid action

    # ----- Features -----
    topM: int = 32
    tau_mass: float = 0.9  # cumulative probability threshold for k_tau
    delta_al_window: int = 16

    # ----- PPO hparams -----
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip: float = 0.2
    lr: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 1.0

    episodes_per_update: int = 512
    update_epochs: int = 4
    minibatch_size: int = 256

    # ----- Misc -----
    seed: int = 1337
    device: str = "cuda"

    # Dataset / runner
    dataset_path: str = ""  # set via CLI
    save_dir: str = "runs/ppo_v1"
