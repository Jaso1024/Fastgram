from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    import torch

from .draft import draft_best_token_id, draft_sample_token_id


@dataclass
class GramDecodingStats:
    accepted: int = 0
    proposed: int = 0
    target_tokens: int = 0
    # For probe-based gating: log probs of accept decisions
    probe_log_probs: Optional[list[float]] = None
    probe_accepts: Optional[list[bool]] = None
    probe_positions: Optional[list[int]] = None  # positions in sequence where probe decided


def _attn_importance_scores(attentions, draft_start: int, draft_len: int) -> "torch.Tensor":
    import torch

    last = attentions[-1]
    attn_mean = last.mean(dim=1)
    key_mass = attn_mean[0].sum(dim=0)
    key_mass = key_mass / key_mass.numel()
    return key_mass[draft_start : draft_start + draft_len]


def _sample_from_logits(logits, *, topk: int, temperature: float) -> int:
    import torch

    if temperature <= 0:
        return int(torch.argmax(logits, dim=-1).item())
    if topk > 0:
        v, ix = torch.topk(logits, k=min(int(topk), int(logits.shape[-1])), dim=-1)
        probs = torch.softmax(v / float(temperature), dim=-1)
        j = int(torch.multinomial(probs, num_samples=1).item())
        return int(ix[j].item())
    probs = torch.softmax(logits / float(temperature), dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def _accept_by_gate(
    *,
    gate: Literal["none", "attn", "spec", "topk", "margin", "prob"],
    gate_topk: int,
    gate_margin: float,
    gate_prob: float,
    draft_id: int,
    logits_pos,
) -> bool:
    import torch

    if gate == "none":
        return int(torch.argmax(logits_pos, dim=-1).item()) == int(draft_id)
    if gate == "spec":
        raise RuntimeError("spec gate is handled separately")
    if gate == "topk":
        if gate_topk <= 0:
            raise ValueError("gate_topk must be > 0")
        _, ix = torch.topk(logits_pos, k=min(int(gate_topk), int(logits_pos.shape[-1])), dim=-1)
        return bool((ix == int(draft_id)).any().item())
    if gate == "margin":
        if gate_margin < 0:
            raise ValueError("gate_margin must be >= 0")
        max_logit = float(torch.max(logits_pos).item())
        tok_logit = float(logits_pos[int(draft_id)].item())
        return (max_logit - tok_logit) <= float(gate_margin)
    if gate == "prob":
        if gate_prob <= 0 or gate_prob > 1:
            raise ValueError("gate_prob must be in (0,1]")
        lse = torch.logsumexp(logits_pos, dim=-1)
        logp = logits_pos[int(draft_id)] - lse
        return float(torch.exp(logp).item()) >= float(gate_prob)
    if gate == "attn":
        raise RuntimeError("attn gate is handled separately")
    raise ValueError("unknown gate")


def gram_decode(
    *,
    model,
    tokenizer,
    engine,
    input_ids: "torch.Tensor",
    max_new_tokens: int,
    draft_k: int,
    max_support: int,
    draft_mode: Literal["infgram", "primitive"] = "infgram",
    draft_sample: bool = False,
    draft_topk: int = 0,
    draft_temperature: float = 1.0,
    draft_seed: Optional[int] = None,
    add_one_when_all_accepted: bool = True,
    raw_engine: bool = False,
    gate: Literal["none", "attn", "spec", "topk", "margin", "prob", "probe"] = "none",
    gate_topk: int = 10,
    gate_margin: float = 2.0,
    gate_prob: float = 0.1,
    attn_threshold: float = 0.02,
    reject_mode: Literal["greedy", "sample"] = "greedy",
    target_topk: int = 0,
    target_temperature: float = 1.0,
    accept_probe: Optional["torch.nn.Module"] = None,
    probe_deterministic: bool = False,
    probe_threshold: float = 0.5,
) -> tuple["torch.Tensor", GramDecodingStats]:
    import torch

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must be shape [1, seq]")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be > 0")
    if draft_k <= 0:
        raise ValueError("draft_k must be > 0")
    if max_support <= 0:
        raise ValueError("max_support must be > 0")
    if gate == "attn" and attn_threshold <= 0:
        raise ValueError("attn_threshold must be > 0")
    if reject_mode not in ("greedy", "sample"):
        raise ValueError("unknown reject_mode")
    if gate == "probe" and accept_probe is None:
        raise ValueError("accept_probe must be provided when gate='probe'")

    stats = GramDecodingStats()
    if gate == "probe":
        stats.probe_log_probs = []
        stats.probe_accepts = []
        stats.probe_positions = []
    device = input_ids.device

    cur_ids = input_ids
    prefix_ids = cur_ids[0].tolist()
    rng = None
    if draft_sample:
        import random

        rng = random.Random(draft_seed)

    for _ in range(max_new_tokens):
        draft_ids: list[int] = []
        draft_prefix = prefix_ids.copy()
        for _ in range(draft_k):
            if rng is None:
                next_id = draft_best_token_id(
                    engine=engine,
                    prefix_ids=draft_prefix,
                    max_support=max_support,
                    mode=draft_mode,
                    raw_engine=raw_engine,
                )
            else:
                next_id = draft_sample_token_id(
                    engine=engine,
                    prefix_ids=draft_prefix,
                    max_support=max_support,
                    mode=draft_mode,
                    raw_engine=raw_engine,
                    topk=int(draft_topk),
                    temperature=float(draft_temperature),
                    rng=rng,
                )
            if next_id is None:
                break
            draft_ids.append(int(next_id))
            draft_prefix.append(int(next_id))

        stats.proposed += len(draft_ids)

        if not draft_ids:
            with torch.no_grad():
                logits = model(cur_ids).logits[:, -1, :]
                next_id = torch.argmax(logits, dim=-1)
            stats.target_tokens += 1
            cur_ids = torch.cat([cur_ids, next_id[:, None]], dim=1)
            prefix_ids.append(int(next_id.item()))
            if tokenizer.eos_token_id is not None and int(next_id.item()) == int(tokenizer.eos_token_id):
                break
            continue

        draft_tensor = torch.tensor([draft_ids], dtype=cur_ids.dtype, device=device)
        need_hidden = gate == "probe"
        with torch.no_grad():
            out = model(
                torch.cat([cur_ids, draft_tensor], dim=1),
                output_attentions=(gate == "attn"),
                output_hidden_states=need_hidden,
            )
            logits = out.logits
            hidden_states = out.hidden_states[-1] if need_hidden else None

        gate_idx: Optional[int] = None
        if gate == "attn":
            draft_start = cur_ids.shape[1]
            scores = _attn_importance_scores(out.attentions, draft_start, len(draft_ids))
            for j, score in enumerate(scores.tolist()):
                if float(score) >= attn_threshold:
                    gate_idx = j
                    break

        accepted_all = True
        for j, draft_id in enumerate(draft_ids):
            pos = cur_ids.shape[1] + j - 1
            if pos < 0:
                pos = 0
            logits_pos = logits[0, pos, :]
            if reject_mode == "sample":
                target_next = _sample_from_logits(logits_pos, topk=int(target_topk), temperature=float(target_temperature))
            else:
                target_next = int(torch.argmax(logits_pos, dim=-1).item())
            if gate_idx is not None and j == gate_idx:
                accepted_all = False
                stats.target_tokens += 1
                cur_ids = torch.cat([cur_ids, torch.tensor([[target_next]], device=device, dtype=cur_ids.dtype)], dim=1)
                prefix_ids.append(target_next)
                break

            # Probe-based gating
            if gate == "probe":
                assert accept_probe is not None and hidden_states is not None
                assert stats.probe_log_probs is not None
                assert stats.probe_accepts is not None
                assert stats.probe_positions is not None
                h = hidden_states[0, pos, :].unsqueeze(0)  # [1, H]
                probe_out = accept_probe(h, deterministic=probe_deterministic, threshold=probe_threshold)
                accept_decision = bool(probe_out.accept.item())
                stats.probe_log_probs.append(float(probe_out.log_prob.item()))
                stats.probe_accepts.append(accept_decision)
                stats.probe_positions.append(pos)
                if accept_decision and target_next == int(draft_id):
                    stats.accepted += 1
                    continue
                # Probe rejected or draft doesn't match target
                accepted_all = False
                stats.target_tokens += 1
                cur_ids = torch.cat([cur_ids, torch.tensor([[target_next]], device=device, dtype=cur_ids.dtype)], dim=1)
                prefix_ids.append(target_next)
                break
            elif gate == "spec":
                if target_next == int(draft_id):
                    stats.accepted += 1
                    continue
            elif gate not in ("attn", "probe"):
                if _accept_by_gate(
                    gate=gate,  # type: ignore
                    gate_topk=int(gate_topk),
                    gate_margin=float(gate_margin),
                    gate_prob=float(gate_prob),
                    draft_id=int(draft_id),
                    logits_pos=logits_pos,
                ):
                    stats.accepted += 1
                    continue
            else:
                if target_next == int(draft_id):
                    stats.accepted += 1
                    continue
            accepted_all = False
            stats.target_tokens += 1
            cur_ids = torch.cat([cur_ids, torch.tensor([[target_next]], device=device, dtype=cur_ids.dtype)], dim=1)
            prefix_ids.append(target_next)
            break

        if accepted_all:
            cur_ids = torch.cat([cur_ids, draft_tensor], dim=1)
            prefix_ids.extend(draft_ids)
            if add_one_when_all_accepted:
                logits_pos = logits[0, cur_ids.shape[1] - 1, :]
                if reject_mode == "sample":
                    next_id = _sample_from_logits(logits_pos, topk=int(target_topk), temperature=float(target_temperature))
                else:
                    next_id = int(torch.argmax(logits_pos, dim=-1).item())
                stats.target_tokens += 1
                cur_ids = torch.cat([cur_ids, torch.tensor([[next_id]], device=device, dtype=cur_ids.dtype)], dim=1)
                prefix_ids.append(next_id)

        if tokenizer.eos_token_id is not None and int(cur_ids[0, -1].item()) == int(tokenizer.eos_token_id):
            break

    return cur_ids, stats
