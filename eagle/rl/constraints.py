# eagle/rl/constraints.py
from typing import Tuple, List

def compute_K(depth: int, k_main: int, k_expand: int) -> int:
    # K = 1 + sum_{i=1..depth} (k_expand(i-1)) * k_main(i)
    # V0 uniform per depth: k_expand(0)=1; k_main(i)=k_main; k_expand(i)=k_expand
    # depth=1 -> 1 + k_main
    # depth=2 -> 1 + k_main + k_expand*k_main
    K = 1 + k_main + sum((k_expand if i > 0 else 1) * k_main for i in range(1, depth))
    return K

def feasible(depth: int, k_main: int, k_expand: int, R: int) -> Tuple[bool, int]:
    if k_main < k_expand:  # 보수적 제약: ke ≤ km
        return (False, 0)
    K = compute_K(depth, k_main, k_expand)
    return (R <= K, K)

def mask_k_expand(k_expand_choices: List[int], k_main: int):
    # k_expand ≤ k_main
    return [ke <= k_main for ke in k_expand_choices]

def mask_R(R_choices: List[int], depth: int, k_main: int, k_expand: int):
    # R ≤ K(depth, k_main, k_expand)
    K = compute_K(depth, k_main, k_expand)
    return [r <= K for r in R_choices]
