import os, json, random, torch, time, threading
from typing import List
from .config import RLConfig
from .ppo import Policy, ppo_update
from .actions import ActionSpace
from .buffers import RolloutBuffer
from .env import EagleEnv
from .integration.eagle_runner import EagleRunner
from .features import build_state_from_logits

# tqdm이 없으면 조용히 비활성화
try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

def _wrap_iter(it, total=None, desc=None, progress=False, ncols=None, leave=True):
    if progress and _HAS_TQDM:
        return tqdm(it, total=total, desc=desc, dynamic_ncols=True,
                    ncols=ncols, leave=leave)
    return it

def _tqdm_write(msg: str):
    if _HAS_TQDM:
        try:
            from tqdm.auto import tqdm as _t
            _t.write(msg)
            return
        except Exception:
            pass
    print(msg, flush=True)

def load_prompts_from_jsonl(path: str):
    # cli.py의 _load_prompts_flex와 동일 로직을 간단화하여 포함
    import json
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            if "prompt" in obj:
                prompts.append(obj["prompt"])
            elif "turns" in obj:
                roles = ["USER", "ASSISTANT"]
                msg = []
                for i, t in enumerate(obj["turns"]):
                    role = roles[i % 2]
                    msg.append(f"{role}: {t}")
                prompts.append("\n".join(msg))
            elif "instruction" in obj:
                ins = obj["instruction"]
                inp = obj.get("input", "")
                prompt = ins if not inp else f"{ins}\n\nINPUT:\n{inp}"
                prompts.append(prompt)
    return prompts


class PromptSampler:
    def __init__(self, prompts: List[str], seed=1337):
        self.prompts = prompts
        random.Random(seed).shuffle(self.prompts)
        self.idx = 0

    def next(self):
        p = self.prompts[self.idx % len(self.prompts)]
        self.idx += 1
        return p

from collections import Counter  # 파일 상단 import에 추가

def train_loop(
    runner_init: dict,
    cfg: RLConfig,
    dataset_path: str,
    save_dir: str,
    progress: bool = False,
    heartbeat_sec: float = 0.0,
    progress_ncols: int | None = None,
):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    prompts = load_prompts_from_jsonl(dataset_path)
    sampler = PromptSampler(prompts, seed=cfg.seed)

    # Runner 생성 (모델 로딩 포함)
    runner = EagleRunner(**runner_init)
    action_space = ActionSpace(cfg.depth_choices, cfg.k_main_choices, cfg.k_expand_choices, cfg.R_choices)
    runner.action_space = action_space

    # 상태 차원 산출
    t0 = time.time()
    dlog, tlog = runner.peek_logits(sampler.next())
    hist = torch.tensor([], device=dlog.device)
    dummy_state = build_state_from_logits(dlog, tlog, cfg.topM, cfg.tau_mass, hist)
    state_dim = dummy_state.numel()
    if progress:
        _tqdm_write(f"[boot] peek_logits ready in {time.time()-t0:.2f}s; state_dim={state_dim}")

    policy = Policy(state_dim, action_space).to(device)
    buffer = RolloutBuffer(device=device)

    os.makedirs(save_dir, exist_ok=True)

    # --- NEW: 지수이동평균(EMA) 트래커 ---
    ema = {"r": 0.0, "dal": 0.0, "K": 0.0, "R": 0.0}
    beta = 0.95  # 더 느리게(안정적으로) 보고 싶으면 0.98~0.995 정도로 올려도 됨

    # 하트비트 쓰레드 (옵션)
    last_tick = time.time()
    _stop = False
    def _beat():
        if heartbeat_sec <= 0:
            return
        period = max(0.5, min(heartbeat_sec, 2.0))
        while not _stop:
            time.sleep(period)
            if (time.time() - last_tick) >= heartbeat_sec:
                _tqdm_write(f"[HB] training alive; idle {int(time.time()-last_tick)}s")

    th = None
    if heartbeat_sec > 0:
        th = threading.Thread(target=_beat, daemon=True)
        th.start()

    try:
        for it in range(1, 999999):
            # --- NEW: 이번 iteration 누적 통계 초기화 ---
            n_ep = 0
            sum_dal = 0.0; sum_K = 0.0; sum_R = 0.0
            sum_dv = 0.0;  sum_peek = 0.0
            ah_d = Counter(); ah_km = Counter(); ah_ke = Counter(); ah_R = Counter()

            # 수집
            collector = _wrap_iter(
                range(cfg.episodes_per_update),
                total=cfg.episodes_per_update,
                desc=f"Collect it={it}",
                progress=progress,
                ncols=progress_ncols or None,
                leave=True,
            )
            for _ in collector:
                prompt = sampler.next()

                # 상태 구성
                t1 = time.time()
                dlog, tlog = runner.peek_logits(prompt)
                s = build_state_from_logits(dlog, tlog, cfg.topM, cfg.tau_mass,
                                            torch.tensor([], device=dlog.device))
                t_peek = time.time() - t1

                # 정책 행위
                a, logp, v = policy.act(s)
                idx_depth, idx_km, idx_ke, idx_R = a.tolist()
                d  = action_space.depth_choices[idx_depth]
                km = action_space.k_main_choices[idx_km]
                ke = action_space.k_expand_choices[idx_ke]
                R  = action_space.R_choices[idx_R]

                # 한 에피소드 실행
                t2 = time.time()
                res = runner.draft_and_verify(prompt, d, km, ke, R)
                t_dv = time.time() - t2

                # 보상 계산(기존 로직 유지)
                r = torch.tensor(res.delta_al - cfg.lambda_v*res.R - cfg.lambda_d*res.K,
                                 dtype=torch.float32, device=device)
                buffer.add(s, a, logp, r, v, torch.tensor([True], device=device))

                # --- NEW: 누적, EMA, 히스토그램 갱신 ---
                n_ep += 1
                sum_dal += float(res.delta_al)
                sum_K   += float(res.K)
                sum_R   += float(res.R)
                sum_dv  += t_dv
                sum_peek+= t_peek

                r_scalar = float(r.item())
                ema["r"]   = beta*ema["r"]   + (1-beta)*r_scalar
                ema["dal"] = beta*ema["dal"] + (1-beta)*float(res.delta_al)
                ema["K"]   = beta*ema["K"]   + (1-beta)*float(res.K)
                ema["R"]   = beta*ema["R"]   + (1-beta)*float(res.R)

                ah_d[d]   += 1
                ah_km[km] += 1
                ah_ke[ke] += 1
                ah_R[R]   += 1

                # 진행바/로그
                last_tick = time.time()
                if progress and _HAS_TQDM:
                    _tqdm_write(
                        f"[it {it}] dv={t_dv:.2f}s peek={t_peek:.2f}s | "
                        f"ΔAL={res.delta_al:.4f} K={res.K} R={res.R} "
                        f"(d={d}, km={km}, ke={ke}, R={R})"
                    )

            # 업데이트
            t3 = time.time()
            ppo_update(policy, buffer, cfg)
            t_upd = time.time() - t3
            buffer.clear()
            last_tick = time.time()

            # --- NEW: progress.jsonl에 리치 로그 쓰기(매 iter) ---
            mean_row = {}
            if n_ep > 0:
                mean_row = {
                    "dal":  (sum_dal / n_ep),
                    "K":    (sum_K   / n_ep),
                    "R":    (sum_R   / n_ep),
                }
            time_row = {
                "dv_sec":   ((sum_dv / n_ep) if n_ep > 0 else None),
                "peek_sec": ((sum_peek / n_ep) if n_ep > 0 else None),
            }
            action_hist = {
                "d":   dict(ah_d),
                "km":  dict(ah_km),
                "ke":  dict(ah_ke),
                "R":   dict(ah_R),
            }
            with open(os.path.join(save_dir, "progress.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "iter": it,
                    "t_update_s": t_upd,
                    "ema": ema,
                    "mean": mean_row,
                    "time": time_row,
                    "action_hist": action_hist,
                    "episodes": n_ep,
                }) + "\n")

            if progress:
                _tqdm_write(
                    f"[it {it}] ppo_update {t_upd:.2f}s | "
                    f"mean ΔAL={mean_row.get('dal',0):.3f}  mean K={mean_row.get('K',0):.1f}  "
                    f"mean R={mean_row.get('R',0):.2f} | EMA(r)={ema['r']:.3f}"
                )

            # 체크포인트 (기존 주기 유지)
            if it % 5 == 0:
                torch.save(policy.state_dict(), os.path.join(save_dir, f"policy_{it}.pt"))
                with open(os.path.join(save_dir, "progress.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps({"iter": it, "checkpoint": f"policy_{it}.pt"}) + "\n")
                if progress:
                    _tqdm_write(f"[ckpt] saved policy_{it}.pt")

    finally:
        _stop = True
        if th is not None:
            th.join(timeout=1.0)

    torch.save(policy.state_dict(), os.path.join(save_dir, "policy_final.pt"))
