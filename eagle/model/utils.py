import copy
import random

# typing 
from typing import List, Tuple
import time
import torch

# TODO
# from transformers import LlamaTokenizer
# tokenizer=LlamaTokenizer.from_pretrained("/home/lyh/weights/hf/vicuna_v13/7B/")

TOPK = 10  # topk for sparse tree

from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


class Timer:
    def __init__(self,name):
        self.name = name
    def __enter__(self):
        torch.cuda.synchronize()
        self.start = time.perf_counter()


    def __exit__(self, exc_type, exc_value, traceback):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start
        print(f'{self.name} took {elapsed} seconds')


def prepare_logits_processor(
        temperature: float = 0.0,
        repetition_penalty: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    if temperature > 1e-5:
        if temperature >= 1e-5 and temperature != 1.0:
            processor_list.append(TemperatureLogitsWarper(temperature))
        if repetition_penalty > 1.0:
            processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        if 1e-8 <= top_p < 1.0:
            processor_list.append(TopPLogitsWarper(top_p))
        if top_k > 0:
            processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


# test_processor = prepare_logits_processor(
#         0.0, 0.0, -1, 1
#     )


def pad_path(path: List[int], length: int, pad_value: int = -2) -> List[int]:
    """
    Pad the given path list with a specific value up to a specified length.

    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.

    Returns:
    - list: A new list based on the original path but padded to the desired length.

    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]

    Note:
    If the given path is already longer than the specified length,
    then no padding occurs, and the original path is returned.
    """

    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))


def generate_tree_buffers(tree_choices, device="cuda"):
    def custom_sort(lst):
        # sort_keys=[len(list)]
        sort_keys = []
        for i in range(len(lst)):
            sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
        return sort_keys
    with Timer("sort"):

        sorted_tree_choices = sorted(tree_choices, key=lambda x: (len(x), x))
        tree_len = len(sorted_tree_choices) + 1

    # Initialize depth_counts to keep track of how many choices have a particular depth
        depth_counts = []
        prev_depth = 0
        for path in sorted_tree_choices:
            depth = len(path)
            if depth != prev_depth:
                depth_counts.append(0)
            depth_counts[depth - 1] += 1
            prev_depth = depth

        tree_attn_mask = torch.eye(tree_len, tree_len)
        tree_attn_mask[:, 0] = 1
        start = 0
        for i in range(len(depth_counts)):
            for j in range(depth_counts[i]):
                cur_tree_choice = sorted_tree_choices[start + j]
                # retrieve ancestor position
                if len(cur_tree_choice) == 1:
                    continue
                ancestor_idx = []
                for c in range(len(cur_tree_choice) - 1):
                    ancestor_idx.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]) + 1)
                tree_attn_mask[j + start + 1, ancestor_idx] = 1
            start += depth_counts[i]

        tree_indices = torch.zeros(tree_len, dtype=torch.long)
        p_indices = [0 for _ in range(tree_len - 1)]
        b_indices = [[] for _ in range(tree_len - 1)]
        tree_indices[0] = 0
        start = 0
        bias = 0
        for i in range(len(depth_counts)):
            inlayer_bias = 0
            b = []
            for j in range(depth_counts[i]):
                cur_tree_choice = sorted_tree_choices[start + j]
                cur_parent = cur_tree_choice[:-1]
                if j != 0:
                    if cur_parent != parent:
                        bias += 1
                        inlayer_bias += 1
                        parent = cur_parent
                        b = []
                else:
                    parent = cur_parent
                tree_indices[start + j + 1] = cur_tree_choice[-1] + TOPK * (i + bias) + 1
                p_indices[start + j] = inlayer_bias
                if len(b) > 0:
                    b_indices[start + j] = copy.deepcopy(b)
                else:
                    b_indices[start + j] = []
                b.append(cur_tree_choice[-1] + TOPK * (i + bias) + 1)
            start += depth_counts[i]

        p_indices = [-1] + p_indices
        tree_position_ids = torch.zeros(tree_len, dtype=torch.long)
        start = 0
        for i in range(len(depth_counts)):
            tree_position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
            start += depth_counts[i]

        retrieve_indices_nest = []
        retrieve_paths = []
        for i in range(len(sorted_tree_choices)):
            cur_tree_choice = sorted_tree_choices[-i - 1]
            retrieve_indice = []
            if cur_tree_choice in retrieve_paths:
                continue
            else:
                for c in range(len(cur_tree_choice)):
                    retrieve_indice.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]))
                    retrieve_paths.append(cur_tree_choice[:c + 1])
            retrieve_indices_nest.append(retrieve_indice)
        max_length = max([len(x) for x in retrieve_indices_nest])
        retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        retrieve_indices = retrieve_indices + 1
        retrieve_indices = torch.cat([torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices],
                                     dim=1)

        maxitem = retrieve_indices.max().item() + 5



        retrieve_indices = retrieve_indices.tolist()
        retrieve_indices = sorted(retrieve_indices, key=custom_sort)
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)



    # Aggregate the generated buffers into a dictionary
    tree_buffers = {
        "tree_attn_mask": tree_attn_mask.unsqueeze(0).unsqueeze(0),
        "tree_indices": tree_indices,
        "tree_position_ids": tree_position_ids,
        "retrieve_indices": retrieve_indices,
    }

    # Move the tensors in the dictionary to the specified device
    tree_buffers = {
        k: v.clone().to(device)
        if isinstance(v, torch.Tensor)
        else torch.tensor(v, device=device)
        for k, v in tree_buffers.items()
    }

    return tree_buffers


def initialize_tree0(input_ids, model, past_key_values, logits_processor):
    draft_tokens, retrieve_indices,tree_mask,tree_position_ids, outputs, logits, hidden_state, sample_token = model(
        input_ids, past_key_values=past_key_values, output_orig=True, logits_processor=logits_processor
    )

    #     if logits_processor is not None:
    #         logits = orig[:, -1]
    #         logits = logits_processor(None, logits)
    #         probabilities = torch.nn.functional.softmax(logits, dim=1)
    #         token = torch.multinomial(probabilities, 1)
    #     else:
    #         token = torch.argmax(orig[:, -1])
    #         token = token[None, None]
    #     input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
    #     # Clone the output hidden states
    #
    #     draft_tokens, retrieve_indices,tree_mask,tree_position_ids = self.ea_layer.topK_genrate(hidden_states, input_ids, self.base_model.lm_head)
    #     if output_orig:
    #         return draft_tokens, retrieve_indices,tree_mask,tree_position_ids, outputs, orig, hidden_states, token
    #     return draft_tokens, retrieve_indices,tree_mask,tree_position_ids, hidden_states, token
    return draft_tokens, retrieve_indices,tree_mask,tree_position_ids, logits, hidden_state, sample_token

@torch.no_grad()
def initialize_tree(input_ids, model, past_key_values, logits_processor=None):
    """
    EAGLE2(cnets1) 호환:
    - prefill로 (B, T, H) hidden 확보
    - 다음 토큰 1개 샘플 → input_ids에 append (길이 T+1)
    - 그 1스텝을 KV로 전진하여 hidden을 (B, T+1, H)로 확장
    - ★ cnets1.topK_genrate는 내부에서 input_ids[:,1:]로 한 칸 자르므로,
      hidden_states도 앞에서 1토큰 잘라 (B, T, H)로 맞춰 전달
    반환: draft_tokens, retrieve_indices, tree_mask, tree_position_ids, orig_logits, hidden_states2, token
    """
    device = input_ids.device

    # 1) Prefill: 현재 prompt까지 한 번 전진
    outputs, orig, hidden_states = model(
        input_ids, past_key_values=past_key_values, output_orig=True
    )   # hidden_states: (B, T, H)

    # 2) 다음 토큰 1개 샘플
    last_logits = orig[:, -1]  # (B, V)
    if logits_processor is not None:
        last_logits = logits_processor(None, last_logits)
        probs = torch.softmax(last_logits, dim=-1)
        token = torch.multinomial(probs, 1)                 # (B,1)
    else:
        token = torch.argmax(last_logits, dim=-1, keepdim=True)  # (B,1)

    # 3) input_ids에 붙이고…
    input_ids2 = torch.cat([input_ids, token.to(device)], dim=1)  # (B, T+1)

    # 4) 그 토큰 1스텝을 KV로 전진시켜 hidden을 (B, T+1, H)로 확장
    outputs2, orig2, hidden_last = model(
        token, past_key_values=past_key_values, output_orig=True
    )   # hidden_last: (B, 1, H)
    hidden_states2 = torch.cat([hidden_states, hidden_last], dim=1)  # (B, T+1, H)

    # 5) ★ cnets1.topK_genrate는 내부에서 input_ids[:,1:] 합니다.
    #    따라서 hidden_states도 BOS를 제외해 길이를 맞춰 전달해야 합니다.
    hidden_states2_trim = hidden_states2[:, 1:, :]  # (B, T, H)  ← T = (T+1) - 1

    # 6) EAGLE2(cnets1)용 topK_genrate 호출 (4-인자 시그니처)
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = model.ea_layer.topK_genrate(
        hidden_states2_trim,          # ✅ (B, T, H)  ← input_ids2[:,1:]와 길이 정합
        input_ids2,                   #  (B, T+1)    ← 함수 내부에서 [:,1:]로 맞춤
        model.base_model.lm_head,
        logits_processor
    )

    # 반환 포맷 유지 (ea_model.eagenerate가 기대)
    return draft_tokens, retrieve_indices, tree_mask, tree_position_ids, orig, hidden_states2, token




def reset_tree_mode(
        model,
):
    model.base_model.model.tree_mask = None
    model.base_model.model.tree_mode = None


def reset_past_key_values(passed_key_values: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Resets the current lengths in the passed key-values to zero.

    This function is designed to be used during the evaluation of a baseline model.
    It iterates through each layer's key-values and sets their current lengths to zero,
    effectively resetting their state.

    Args:
    - passed_key_values (list of torch.Tensor): Contains past hidden states and past attention values for each layer.

    Returns:
    - passed_key_values (list of torch.Tensor): Updated past hidden states and past attention values with reset lengths.
    """
    for i in range(len(passed_key_values)):
        for j in range(2):
            passed_key_values[i][j].current_length.fill_(0)
    return passed_key_values


def generate_candidates(tree_logits, tree_indices, retrieve_indices, sample_token, logits_processor):
    sample_token = sample_token.to(tree_indices.device)

    candidates_logit = sample_token[0]

    candidates_tree_logits = tree_logits

    candidates = torch.cat([candidates_logit, candidates_tree_logits.view(-1)], dim=-1)

    tree_candidates = candidates[tree_indices]

    tree_candidates_ext = torch.cat(
        [tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device) - 1], dim=0)

    cart_candidates = tree_candidates_ext[retrieve_indices]


    # Unsqueeze the tree candidates for dimension consistency.
    tree_candidates = tree_candidates.unsqueeze(0)
    return cart_candidates,  tree_candidates


def tree_decoding(
        model,
        tree_candidates,
        past_key_values,
        tree_position_ids,
        input_ids,
        retrieve_indices,
):
    position_ids = tree_position_ids + input_ids.shape[1]
    if position_ids is not None and position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
    outputs, tree_logits, hidden_state = model(
        tree_candidates,
        output_orig=True,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    if model.use_eagle3:
        ea_device = model.ea_layer.lm_head.weight.device
        if outputs["hidden_states"][0].device != ea_device:
            outputs["hidden_states"] = [x.to(ea_device) for x in outputs["hidden_states"]]
        hidden_state = torch.cat(outputs["hidden_states"], dim=-1)

    logits = tree_logits[0, retrieve_indices]
    return logits, hidden_state, outputs





def evaluate_posterior(
        logits: torch.Tensor,
        candidates: torch.Tensor,
        logits_processor,
):
    """
    Evaluate the posterior probabilities of the candidates based on the provided logits and choose the best candidate.

    Depending on the temperature value, the function either uses greedy decoding or evaluates posterior
    probabilities to select the best candidate.

    Args:
    - logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
    - candidates (torch.Tensor): Candidate token sequences.
    - temperature (float): Softmax temperature for probability scaling. A value of 0 indicates greedy decoding.
    - posterior_threshold (float): Threshold for posterior probability.
    - posterior_alpha (float): Scaling factor for the threshold.

    Returns:
    - best_candidate (torch.Tensor): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    """
    # Greedy decoding based on temperature value
    if logits_processor is None:
        # Find the tokens that match the maximum logits for each position in the sequence
        posterior_mask = (
                candidates[:, 1:].to(logits.device) == torch.argmax(logits[:, :-1], dim=-1)
        ).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, accept_length, logits[best_candidate, accept_length]

    else:
        accept_length = 1
        accept_cand = candidates[0][:1]
        best_candidate = 0
        for i in range(1, candidates.shape[1]):
            if i != accept_length:
                break
            adjustflag = False
            is_eq = (candidates[:, :accept_length] == accept_cand).all(dim=1)
            fi = torch.nonzero(is_eq, as_tuple=True)[0][0]
            gt_logits = logits[fi, i - 1][None]
            gt_logits = logits_processor(None, gt_logits)[0]
            gtp = torch.softmax(gt_logits, dim=0)
            candidates_set = []
            for j in range(candidates.shape[0]):
                if is_eq[j]:
                    x = candidates[j, i]
                    xi = x.item()
                    if xi in candidates_set or xi == -1:
                        continue
                    candidates_set.append(xi)
                    r = random.random()
                    px = gtp[xi]
                    qx = 1.0
                    acp = px / qx
                    if r <= acp:
                        accept_cand = torch.cat((accept_cand, x[None]), dim=0)
                        accept_length += 1
                        best_candidate = j
                        break
                    else:
                        gtp[xi] = 0
                        gtp = gtp / gtp.sum()
                        adjustflag = True
        if adjustflag and accept_length != candidates.shape[1]:
            sample_p = gtp
        else:
            gt_logits = logits[best_candidate, accept_length - 1][None]
            gt_logits = logits_processor(None, gt_logits)[0]
            sample_p = torch.softmax(gt_logits, dim=0)
        return torch.tensor(best_candidate), accept_length - 1, sample_p


@torch.no_grad()
def update_inference_inputs(
        input_ids,
        candidates,
        best_candidate,
        accept_length,
        retrieve_indices,
        logits_processor,
        new_token,
        past_key_values_data_list,
        current_length_data,
        model,
        hidden_state_new,
        sample_p
):
    """
    EAGLE2(cnets1) 호환을 위해:
    - cnets1.topK_genrate는 input_ids[:,1:]를 사용하므로 hidden_states 길이 == input_ids[:,1:].shape[1] 이어야 함
    - (기존) 전체 시퀀스를 넘기면 길이 불일치가 발생 -> 로컬 세그먼트만 구성해서 전달
    - 또한 sample_p로 뽑은 token에 대해 1스텝 히든을 구해 hidden_states 길이를 맞춰 준다.
    """

    device = input_ids.device
    prev_input_len = input_ids.shape[1]

    # 1) 선택된 후보 토큰(accept_length+1개)을 본 시퀀스에 append
    seg_tokens = candidates[None, best_candidate, : accept_length + 1].to(device)   # (1, L)
    input_ids = torch.cat([input_ids, seg_tokens], dim=-1)                          # (1, T + L)

    # 2) past_key_values 데이터에 해당 토큰들의 KV를 복사/반영
    #    (retrieve_indices는 전역 인덱스 기준이므로 prev_input_len 오프셋을 더해줌)
    select_indices = retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len  # (L,)
    for past_key_values_data in past_key_values_data_list:
        tgt = past_key_values_data[..., select_indices.to(past_key_values_data.device), :]                 # (..., L, D)
        dst = past_key_values_data[..., prev_input_len: prev_input_len + tgt.shape[-2], :]                 # (..., L, D)
        dst.copy_(tgt, non_blocking=True)
    current_length_data.fill_(prev_input_len + select_indices.shape[0])  # 현재 길이 갱신

    # 3) 이번 스텝에서 사용할 로컬 히든 선택 (검증 단계에서 얻은 hidden_state_new에서 인덱싱)
    #    shape: (1, L, H)
    retrieve_hidden_state_new = hidden_state_new[:, retrieve_indices]                   # (1, num_paths, depth, H) 였던 걸
    accept_hidden_state_new   = retrieve_hidden_state_new[:, best_candidate, : accept_length + 1]  # (1, L, H)

    # 4) sample_p에서 다음 토큰 1개 샘플
    if isinstance(sample_p, list):
        prob = torch.stack(sample_p)[0]
    else:
        prob = sample_p
    if logits_processor is not None:
        token = torch.multinomial(prob, 1)             # (1,)
        token = token[None]                            # (1,1)
    else:
        token = torch.argmax(prob)
        token = token[None, None]                      # (1,1)

    # 5) 토큰을 본 시퀀스에도 append (전역 시퀀스)
    input_ids = torch.cat([input_ids, token.to(device)], dim=1)   # (1, T + L + 1)

    # 6) ★ EAGLE2의 topK_genrate에 넘길 "로컬" pair 구성
    #    - input_ids_for_topk: [직전 1토큰] + [방금 수용된 L토큰] + [sample 토큰]  → 길이 L+2
    #      내부에서 [:,1:]로 잘리면 길이 L+1
    #    - hidden_for_topk:    [수용된 L토큰의 히든] + [sample 토큰 1스텝 히든]      → 길이 L+1
    #
    #    sample 토큰 1스텝 히든을 얻기 위해 EaModel.forward를 1스텝 호출
    #    (EaModel은 self.past_key_values를 들고 있으므로 여기서 호출 가능)
    #    반환 hidden은 (B,1,H)
    _, _, hidden_step = model(
        token.to(device),
        past_key_values=model.past_key_values,   # ea_model.eagenerate에서 세팅됨
        output_orig=True
    )
    hidden_for_topk = torch.cat([accept_hidden_state_new, hidden_step], dim=1)  # (1, L+1, H)

    # 로컬 input_ids: 직전 1토큰 + 방금 수용된 L토큰 + sample 토큰
    last_prev_token = input_ids[:, -(accept_length + 1 + 1 + 1)]  # 직전 1토큰 = 새로 붙인 L+1(수용+샘플) 앞의 토큰
    input_ids_for_topk = torch.cat([last_prev_token[:, None], seg_tokens, token.to(device)], dim=1)  # (1, L+2)

    # 7) EAGLE2(cnets1)용 topK_genrate 호출  (⚠ attention_mask 인자 절대 넘기지 말 것)
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = model.ea_layer.topK_genrate(
        hidden_for_topk,                    # (1, L+1, H)
        input_ids=input_ids_for_topk,       # (1, L+2)  → 내부에서 [:,1:] 되어 (1, L+1)
        head=model.base_model.lm_head,
        logits_processor=logits_processor
    )

    new_token += accept_length + 1

    # ea_model.eagenerate가 기대하는 반환 형식 유지
    return input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, None, token


if __name__ == "__main__":
    logits = torch.randn(1, 5)
    tp = prepare_logits_processor(0.9, 0, 0.9, 0)
    l = tp(None, logits)
    if tp is None:
        print(tp)
