import argparse, os, torch, time, json
from .config import RLConfig
from .trainer import train_loop

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    # 공통 모델 경로 인자
    def add_model_args(p):
        p.add_argument("--base-model-path", type=str, required=True,
                       help="HF 경로 또는 로컬 폴더")
        p.add_argument("--ea-model-path", type=str, required=True,
                       help="EAGLE 계층 경로(가중치 + config) 또는 HF repo id")
        p.add_argument("--device", type=str, default="cuda")
        p.add_argument("--use-eagle3", action="store_true", help="EAGLE3")

    tr = sub.add_parser("train")
    tr.add_argument("--dataset", type=str, required=True, help="JSONL (prompt 필드 또는 turns 적용)")
    tr.add_argument("--save-dir", type=str, default="runs/ppo_v1")
    # --- 진행바/하트비트 옵션 (기본 꺼짐) ---
    tr.add_argument("--progress", action="store_true",
                    help="학습 중 tqdm 진행바를 표시합니다.")
    tr.add_argument("--heartbeat-sec", type=float, default=float(os.environ.get("EAGLE_HEARTBEAT", "0") or 0.0),
                    help="N초 동안 스텝 진행이 없으면 하트비트 로그를 찍습니다. 0=off")
    tr.add_argument("--progress-ncols", type=int, default=int(os.environ.get("EAGLE_PROGRESS_NCOLS", "0") or 0),
                    help="tqdm bar 폭 (0=auto)")
    add_model_args(tr)

    ev = sub.add_parser("eval")
    ev.add_argument("--dataset", type=str, required=True)
    ev.add_argument("--policy", type=str, required=True)
    ev.add_argument("--out", type=str, default="logs/eval.jsonl")
    # --- 추가 옵션 (EAGLE 벤치마크 스타일) ---
    ev.add_argument("--temperature", type=float, default=0.0)
    ev.add_argument("--limit", type=int, default=0, help="앞에서 N개만 평가(0=전부)")
    # 정책을 무시하고 강제 파라미터로 덮어쓰기
    ev.add_argument("--force-depth", type=int)
    ev.add_argument("--force-k-main", type=int)
    ev.add_argument("--force-k-expand", type=int)
    ev.add_argument("--force-R", type=int)
    add_model_args(ev)

    args = parser.parse_args()

    cfg = RLConfig()
    cfg.device = args.device

    # Runner 생성은 trainer 내부에서 하도록 인자 전달
    runner_init = {
        "base_model_path": args.base_model_path,
        "ea_model_path": args.ea_model_path,
        "device": cfg.device,
        "use_eagle3": args.use_eagle3,
    }

    if args.cmd == "train":
        # 환경변수 플래그도 허용(EAGLE_PROGRESS=1)
        progress_flag = args.progress or bool(int(os.environ.get("EAGLE_PROGRESS", "0") or 0))
        progress_ncols = (args.progress_ncols or None)
        train_loop(
            runner_init, cfg,
            dataset_path=args.dataset,
            save_dir=args.save_dir,
            progress=progress_flag,
            heartbeat_sec=args.heartbeat_sec,
            progress_ncols=progress_ncols,
        )

    elif args.cmd == "eval":
        from .actions import ActionSpace
        from .ppo import Policy
        from .integration.eagle_runner import EagleRunner
        from .features import build_state_from_logits

        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        runner = EagleRunner(**runner_init)

        # 정책 로딩
        dummy_state, _, _ = _one_state_for_shape(runner, dataset_path=args.dataset)
        state_dim = dummy_state.numel()
        action_space = ActionSpace(cfg.depth_choices, cfg.k_main_choices, cfg.k_expand_choices, cfg.R_choices)
        policy = Policy(state_dim, action_space).to(device)
        policy.load_state_dict(torch.load(args.policy, map_location=device))
        policy.eval()

        # 평가용 프롬프트 로딩
        prompts = _load_prompts_flex(args.dataset)
        if args.limit and args.limit > 0:
            prompts = prompts[:args.limit]

        os.makedirs(os.path.dirname(args.out), exist_ok=True)

        # === TPS/통계 누적자 (EAGLE 스크립트와 동일 포맷) ===
        tps_from_new_tokens = []   # per-sample (new_tokens / time)
        tps_from_tokenizer = []    # per-sample (tokenizer_count / time)
        total_time_sec = 0.0
        total_tokens_tokenizer = 0
        tot_n_phase = 0
        tot_n_accept = 0

        with open(args.out, "w", encoding="utf-8") as fout:
            for p in prompts:
                # 상태 1회
                dlog, tlog = runner.peek_logits(p)
                hist = torch.tensor([], device=dlog.device)
                s = build_state_from_logits(dlog, tlog, cfg.topM, cfg.tau_mass, hist).to(device)

                # 정책 argmax
                with torch.no_grad():
                    logits = policy.forward(s.unsqueeze(0))
                    idx_depth = torch.argmax(logits["depth"], dim=-1).item()
                    idx_km    = torch.argmax(logits["k_main"], dim=-1).item()
                    idx_ke    = torch.argmax(logits["k_expand"], dim=-1).item()
                    idx_R     = torch.argmax(logits["R"], dim=-1).item()
                d  = action_space.depth_choices[idx_depth]
                km = action_space.k_main_choices[idx_km]
                ke = action_space.k_expand_choices[idx_ke]
                R  = action_space.R_choices[idx_R]

                # 강제 파라미터 적용(원하면 길게 생성)
                if args.force_depth is not None:    d  = int(args.force_depth)
                if args.force_k_main is not None:   km = int(args.force_k_main)
                if args.force_k_expand is not None: ke = int(args.force_k_expand)
                if args.force_R is not None:        R  = int(args.force_R)

                # 실행(풀 디코드) + 시간
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.time()
                res = runner.draft_and_verify(
                    p, d, km, ke, R,
                    temperature=args.temperature,
                    full_decode=True,        # <-- 길게 생성
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                dt = time.time() - t0  # 초 단위

                # tokenizer 기반 토큰 카운트(검증용)
                tok_cnt = 0
                if res.output_text is not None and len(res.output_text) > 0:
                    tok = runner.model.get_tokenizer()
                    tok_cnt = max(0, len(tok(res.output_text).input_ids) - 1)
                elif res.output_ids is not None:
                    tok_cnt = len(res.output_ids)
                else:
                    tok_cnt = int(res.new_tokens)

                # TPS per-sample
                if dt > 0:
                    tps_from_new_tokens.append(res.new_tokens / dt)
                    tps_from_tokenizer.append(tok_cnt / dt)
                total_time_sec += dt
                total_tokens_tokenizer += tok_cnt

                # 누적 수용 통계
                tot_n_phase  += int(res.n_phase)
                tot_n_accept += int(res.n_accept)

                # JSONL 저장(필드 확장)
                fout.write(json.dumps({
                    "prompt": p[:160],
                    "depth": d, "k_main": km, "k_expand": ke, "R": R,
                    "delta_al": float(res.delta_al), "K": int(res.K),
                    "R_used": int(res.R), "t_ms": float(res.meters.get("t_ms", 0.0)),
                    "new_tokens": int(res.new_tokens),
                    "n_phase": int(res.n_phase), "n_accept": int(res.n_accept),
                    "al_sum": int(res.al_sum),
                    "out_len_tok": int(tok_cnt)
                }, ensure_ascii=False) + "\n")

        # === 요약 출력 (EAGLE 스크립트 동일 형식) ===
        def _avg(xs): return (sum(xs)/len(xs)) if xs else 0.0
        avg_tps_new = _avg(tps_from_new_tokens)
        avg_tps_tok = _avg(tps_from_tokenizer)
        print(f"\nTPS (avg, from new_tokens): {avg_tps_new:.3f} tok/s")
        print(f"TPS (avg, from tokenizer):  {avg_tps_tok:.3f} tok/s")
        if avg_tps_tok > 0:
            print(f"TPS ratio (new_tokens/tokenizer): {avg_tps_new/avg_tps_tok:.4f}")
        if total_time_sec > 0:
            print(f"Overall TPS (tokenizer total): {total_tokens_tokenizer/total_time_sec:.3f} tok/s")
        print(f"Accumulated: tokens={total_tokens_tokenizer}, time={total_time_sec:.3f}s")

        print("\nFinal Statistics:")
        print(f"Total n_phase: {tot_n_phase}")
        print(f"Total n_accept: {tot_n_accept}")
        if tot_n_phase > 0:
            al = (tot_n_accept / tot_n_phase) + 1.0
            print(f"Average acceptance length: {al:.4f}")

        print(f"[eval] wrote \u2192 {args.out}", flush=True)

def _load_prompts_flex(path: str):
    """데이터셋 jsonl에서 'prompt' 또는 'turns'를 유연히 읽어 prompt 문자열을 만든다."""
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

def _build_state(runner, prompt, cfg):
    dlog, tlog = runner.peek_logits(prompt)
    from .features import build_state_from_logits
    import torch
    hist = torch.tensor([], device=dlog.device)  # 평가 1회는 empty trend
    state = build_state_from_logits(dlog, tlog, cfg.topM, cfg.tau_mass, hist)
    return state

def _one_state_for_shape(runner, dataset_path):
    prompts = _load_prompts_flex(dataset_path)
    p = prompts[0] if len(prompts) else "Hello"
    s = _build_state(runner, p, RLConfig())
    return s, p, prompts

if __name__ == "__main__":
    main()
