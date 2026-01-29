from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class LogProbBatch:
    token_logp: "torch.Tensor"
    token_mask: "torch.Tensor"
    token_count: "torch.Tensor"


def token_logprobs_for_completions(
    *,
    model,
    input_ids: "torch.Tensor",
    attention_mask: "torch.Tensor",
    prompt_lens: "torch.Tensor",
    pad_token_id: int,
) -> LogProbBatch:
    import torch

    if input_ids.ndim != 2:
        raise ValueError("input_ids must be [B,T]")
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask shape mismatch")
    if prompt_lens.ndim != 1 or prompt_lens.shape[0] != input_ids.shape[0]:
        raise ValueError("prompt_lens must be [B]")

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_attn = attention_mask[:, 1:]

    logp_all = torch.log_softmax(shift_logits, dim=-1).gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

    bsz, t1 = shift_labels.shape
    pos = torch.arange(t1, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
    start = (prompt_lens - 1).clamp_min(0).unsqueeze(1)
    completion_mask = (pos >= start) & (shift_attn.bool())

    token_count = completion_mask.sum(dim=1)
    token_logp = logp_all * completion_mask
    return LogProbBatch(token_logp=token_logp, token_mask=completion_mask, token_count=token_count)


def pad_batch(
    *,
    seqs: list[list[int]],
    prompt_lens: list[int],
    pad_token_id: int,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    import torch

    bsz = len(seqs)
    if bsz == 0:
        raise ValueError("empty batch")
    if len(prompt_lens) != bsz:
        raise ValueError("prompt_lens length mismatch")
    max_len = max(len(s) for s in seqs)
    ids = torch.full((bsz, max_len), int(pad_token_id), dtype=torch.long)
    attn = torch.zeros((bsz, max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        attn[i, : len(s)] = 1
    pl = torch.tensor(prompt_lens, dtype=torch.long)
    return ids, attn, pl

