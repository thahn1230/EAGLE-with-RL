from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional, List
import time
import torch

@dataclass
class RunResult:
    # --- 기존 필드 ---
    delta_al: float
    K: int
    R: int
    meters: Dict
    draft_logits: Optional[torch.Tensor] = None
    target_logits: Optional[torch.Tensor] = None
    # --- 통계/평가용 추가 필드 ---
    new_tokens: int = 0
    n_phase: int = 0
    n_accept: int = 0
    al_sum: int = 0
    output_ids: Optional[List[int]] = None
    output_text: Optional[str] = None

from eagle.model.ea_model import (
    EaModel,
    initialize_past_key_values,
    initialize_tree,
    tree_decoding,
    evaluate_posterior,
)
from eagle.model.utils import reset_tree_mode
from eagle.rl.constraints import compute_K


class EagleRunner:
    def __init__(self,
                 base_model_path: str,
                 ea_model_path: str,
                 device: str = "cuda",
                 use_eagle3: bool = False,
                 dtype: torch.dtype = torch.float16):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_eagle3 = use_eagle3

        self.model = EaModel.from_pretrained(
            use_eagle3=use_eagle3,
            base_model_path=base_model_path,
            ea_model_path=ea_model_path,
            total_token=32,
            depth=4,
            top_k=12,
            expand_k=8,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        self.tokenizer = self.model.get_tokenizer()
        self.action_space = None
        self.model.eval()

    def _encode(self, prompt: str):
        enc = self.tokenizer([prompt], return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)
        return input_ids, attn

    def _last_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 3:
            return logits[0, -1, :].detach()
        elif logits.dim() == 2:
            return logits[0, :].detach()
        else:
            return logits.detach().view(-1)

    @torch.inference_mode()
    def peek_logits(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids, attn = self._encode(prompt)
        outputs, orig_logits, _hidden = self.model.forward(
            input_ids=input_ids,
            attention_mask=attn,
            output_orig=True,
        )
        tgt = self._last_logits(orig_logits)
        drf = tgt
        return drf, tgt

    @torch.inference_mode()
    def draft_and_verify(
        self,
        prompt: str,
        depth: int,
        k_main: int,
        k_expand: int,
        R: int,
        *,
        temperature: float = 0.0,
        full_decode: bool = False,
    ) -> RunResult:
        # 1) 행동 파라미터 주입
        self.model.ea_layer.depth = depth
        self.model.ea_layer.top_k = k_main
        try:
            self.model.ea_layer.expand_k = k_expand
        except Exception:
            pass

        # 2) 예산 안전화
        #   - 총 상한: R_req ≤ K(depth, km, ke)
        #   - level-1 상한: total_tokens ≤ k_main  (topK_genrate가 level-1에서 topk(total_tokens) 호출)
        K_total = compute_K(depth=depth, k_main=k_main, k_expand=k_expand)
        R_req = int(min(int(R), int(K_total)))
        R_level1 = max(1, min(R_req, int(k_main)))
        self.model.ea_layer.total_tokens = R_level1

        # 3) 입력
        input_ids, attn = self._encode(prompt)

        # 4) 누적자 스냅샷
        prev_phase = int(getattr(self.model, "n_phase", 0))
        prev_accept = int(getattr(self.model, "n_accept", 0))
        prev_al_sum = int(getattr(self.model, "al_sum", 0)) if hasattr(self.model, "al_sum") else 0

        if hasattr(self.model, "current_length_data") and self.model.current_length_data is not None:
            try:
                self.model.current_length_data.zero_()
            except Exception:
                pass

        # 5) 실행
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()

        if full_decode:
            out_ids, new_token, _ = self.model.eagenerate(
                input_ids.to(self.device),
                temperature=temperature,
                log=True,
            )
        else:
            out_ids, new_token, _ = self.model.eagenerate(
                input_ids.to(self.device),
                temperature=0.0,
                max_new_tokens=1,
                max_length=2048,
                log=True,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()

        # 6) accept 길이 증분
        delta_al = int(getattr(self.model, "n_accept", 0) - prev_accept)

        # 7) 출력 텍스트 후처리
        tok = self.model.get_tokenizer()
        gen_ids = out_ids[0][len(input_ids[0]):]
        stop_token_ids = getattr(self.model, "stop_token_ids", None)
        if stop_token_ids:
            stop_idx = [i for i, t in enumerate(gen_ids) if int(t) in set(stop_token_ids)]
            if len(stop_idx) > 0:
                gen_ids = gen_ids[: stop_idx[0]]

        output_text = tok.decode(gen_ids, spaces_between_special_tokens=False)
        for st in tok.special_tokens_map.values():
            if isinstance(st, list):
                for s in st:
                    output_text = output_text.replace(s, "")
            else:
                output_text = output_text.replace(st, "")
        output_text = output_text.strip()

        # 8) 상태용 로짓(학습 경로)
        if not full_decode:
            base_out = self.model.base_model(out_ids.to(self.device))
            last_logits = base_out.logits[0, -1, :].detach()
            target_logits = last_logits
            draft_logits = last_logits
        else:
            draft_logits = None
            target_logits = None

        # 9) 메트릭 기록
        R_used = R_req  # 총 예산은 R_req로 간주(K 상한 반영)
        meters = {
            "t_ms": (t1 - t0) * 1000.0,
            "accept_length": float(delta_al),
            "depth": depth,
            "k_main": k_main,
            "k_expand": k_expand,
            "R": int(R_used),
            "R_level1": int(R_level1),   # 디버그: level-1에 실제 세팅된 topk
            "R_req": int(R_req),         # 디버그: K 상한 적용 후 예산
        }

        n_phase_delta  = int(getattr(self.model, "n_phase", 0)  - prev_phase)
        n_accept_delta = int(getattr(self.model, "n_accept", 0) - prev_accept)
        al_sum_delta   = int(getattr(self.model, "al_sum", prev_al_sum) - prev_al_sum) if hasattr(self.model, "al_sum") else 0

        return RunResult(
            delta_al=float(delta_al),
            K=int(K_total),
            R=int(R_used),
            meters=meters,
            draft_logits=draft_logits,
            target_logits=target_logits,
            new_tokens=int(new_token),
            n_phase=int(n_phase_delta),
            n_accept=int(n_accept_delta),
            al_sum=int(al_sum_delta),
            output_ids=[int(x) for x in gen_ids.tolist()] if hasattr(gen_ids, "tolist") else [int(x) for x in gen_ids],
            output_text=output_text,
        )
