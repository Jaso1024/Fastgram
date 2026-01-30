#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "gram-decoding"))
sys.path.insert(0, str(_ROOT / "gram-rl"))

from fastgram import GramEngine
from gram_decoding import gram_decode
from gram_rl.gcs import sync_index_from_gcs
from gram_rl.gsm8k import gsm8k_format_instruction, gsm8k_prompt, gsm8k_reward, gsm8k_reward_components
from gram_rl.grpo import grpo_loss
from gram_rl.logprobs import pad_batch, token_logprobs_for_completions
from gram_rl.probe import AcceptProbe, create_probe_for_model


def _adv_from_group(rewards: list[float], eps: float = 1e-6) -> list[float]:
    if not rewards:
        return []
    m = sum(rewards) / len(rewards)
    v = sum((r - m) ** 2 for r in rewards) / max(1, len(rewards) - 1)
    s = (v ** 0.5) if v > 0 else 0.0
    if s <= 0:
        return [0.0 for _ in rewards]
    return [(r - m) / (s + eps) for r in rewards]


def _make_chat_prompt(tok, question: str, *, format_instruction: str) -> str:
    base = gsm8k_prompt(question=question)
    if format_instruction:
        base = base + "\n" + str(format_instruction).strip()
    if hasattr(tok, "apply_chat_template"):
        msgs = [{"role": "user", "content": base}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return base


def _infer_max_context_len(tok, model) -> int:
    vals = []
    for v in [
        getattr(getattr(model, "config", None), "max_position_embeddings", None),
        getattr(getattr(model, "config", None), "max_sequence_length", None),
        getattr(getattr(model, "config", None), "max_seq_len", None),
        getattr(tok, "model_max_length", None),
    ]:
        if isinstance(v, int) and 0 < v < 1_000_000:
            vals.append(int(v))
    return min(vals) if vals else 4096


def _generate_raw(
    *,
    model,
    input_ids,
    pad_token_id: int,
    eos_token_id: int,
    max_new: int,
    topk: int,
    temperature: float,
):
    import torch

    m = model.module if hasattr(model, "module") else model
    m.eval()
    with torch.no_grad():
        out = m.generate(
            input_ids=input_ids,
            do_sample=True,
            top_k=int(topk),
            temperature=float(temperature),
            max_new_tokens=min(int(max_new), 1024),
            pad_token_id=int(pad_token_id),
            eos_token_id=int(eos_token_id),
        )
    m.train()
    return out


def _ddp_env() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world, local


def _ddp_reduce(t, op):
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=op)
    return t


def main() -> int:
    p = argparse.ArgumentParser(description="GRPO-ish training on GSM8K with gram-assisted decoding")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--index-gcs", default="gs://fastgram-indices-jaso1024/reasoning")
    p.add_argument("--index-dir", default="index/reasoning")
    p.add_argument("--sync-index", action="store_true", default=False)
    p.add_argument("--wandb", action="store_true", default=False)
    p.add_argument("--wandb-project", default="fastgram-rl")
    p.add_argument("--wandb-entity", default="")
    p.add_argument("--wandb-name", default="")
    p.add_argument("--wandb-tags", default="")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--decode-mode", choices=["gram", "model"], default="gram")
    p.add_argument("--accept-bonus", type=float, default=0.0)
    p.add_argument("--answer-format", choices=["none", "hash4", "cot", "deepseek"], default="none")
    p.add_argument("--format-weight", type=float, default=0.0)
    p.add_argument("--format-instruction", default="")
    p.add_argument("--draft-k", type=int, default=16)
    p.add_argument("--max-support", type=int, default=500)
    p.add_argument("--draft-topk", type=int, default=50)
    p.add_argument("--draft-temperature", type=float, default=1.0)
    p.add_argument("--gate", choices=["none", "spec", "topk", "margin", "prob", "probe"], default="margin")
    p.add_argument("--gate-topk", type=int, default=20)
    p.add_argument("--gate-margin", type=float, default=2.0)
    p.add_argument("--gate-prob", type=float, default=0.1)
    p.add_argument("--probe-lr", type=float, default=1e-4, help="Learning rate for accept probe")
    p.add_argument("--probe-weight", type=float, default=1.0, help="Weight for probe policy gradient loss")
    p.add_argument("--reject-mode", choices=["greedy", "sample"], default="sample")
    p.add_argument("--target-topk", type=int, default=50)
    p.add_argument("--target-temperature", type=float, default=0.8)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--kl-beta", type=float, default=0.02)
    p.add_argument("--ratio-mode", choices=["sequence", "sequence_sum", "token"], default="sequence")
    p.add_argument("--kl-estimator", choices=["nonneg", "sample"], default="nonneg")
    p.add_argument("--update-epochs", type=int, default=1)
    p.add_argument("--minibatch-size", type=int, default=0)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-dir", default="runs/gram_rl")
    p.add_argument("--save-every", type=int, default=200)
    args = p.parse_args()

    rank, world, local_rank = _ddp_env()
    is_main = rank == 0

    if args.seed:
        random.seed(int(args.seed) + rank)
        os.environ["PYTHONHASHSEED"] = str(int(args.seed) + rank)

    try:
        import torch
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        raise SystemExit(f"missing dependency: {e}")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if world > 1:
        dist.init_process_group(backend=backend)
    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    wandb_run = None
    if args.wandb and is_main:
        import wandb

        tags = [t.strip() for t in str(args.wandb_tags).split(",") if t.strip()]
        wandb_run = wandb.init(
            project=str(args.wandb_project),
            entity=(str(args.wandb_entity) if str(args.wandb_entity) else None),
            name=(str(args.wandb_name) if str(args.wandb_name) else None),
            config=vars(args),
            tags=tags if tags else None,
        )

    if args.sync_index:
        sync_index_from_gcs(gcs_uri=args.index_gcs, local_dir=args.index_dir)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch_dtype, trust_remote_code=True)
    model.to(device)
    model.train()
    if world > 1:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    ref = None
    if float(args.kl_beta) > 0.0:
        ref = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch_dtype, trust_remote_code=True)
        ref.to(device)
        ref.eval()
        for p_ in ref.parameters():
            p_.requires_grad_(False)

    eos = tok.eos_token_id
    if eos is None:
        raise SystemExit("tokenizer has no eos_token_id")
    vocab = getattr(tok, "vocab_size", None) or len(tok)

    if int(vocab) <= 2**8:
        token_dtype = "u8"
    elif int(vocab) <= 2**16:
        token_dtype = "u16"
    else:
        token_dtype = "u32"

    engine = None
    if str(args.decode_mode) == "gram":
        engine = GramEngine(
            index_dir=args.index_dir,
            eos_token_id=int(eos),
            vocab_size=int(vocab),
            version=4,
            token_dtype=token_dtype,
            threads=0,
        )

    # Initialize accept probe if using probe gate
    accept_probe = None
    probe_opt = None
    if str(args.gate) == "probe":
        base_model = model.module if hasattr(model, "module") else model
        accept_probe = create_probe_for_model(base_model)
        accept_probe.to(device=device, dtype=torch_dtype)  # Match model dtype
        accept_probe.train()
        probe_opt = torch.optim.AdamW(accept_probe.parameters(), lr=float(args.probe_lr))

    ds = load_dataset("gsm8k", "main", split="train")
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    max_ctx = _infer_max_context_len(tok, model.module if hasattr(model, "module") else model)

    save_dir = Path(args.save_dir)
    if is_main:
        save_dir.mkdir(parents=True, exist_ok=True)

    fmt_style = str(args.answer_format)
    fmt_inst = str(args.format_instruction).strip() or gsm8k_format_instruction(fmt_style)
    fmt_w = float(args.format_weight)

    t0 = time.perf_counter()
    step = 0
    while step < int(args.steps):
        q_ixs = [random.randrange(0, len(ds)) for _ in range(int(args.batch_size))]
        batch = [ds[i] for i in q_ixs]

        all_seqs: list[list[int]] = []
        all_prompt_lens: list[int] = []
        all_rewards: list[float] = []
        all_task_rewards: list[float] = []
        all_bonus_rewards: list[float] = []
        all_format_rewards: list[float] = []
        all_groups: list[int] = []
        all_probe_log_probs: list[list[float]] = []  # probe log probs per sample
        all_probe_positions: list[list[int]] = []  # positions where probe decided
        all_probe_accepts: list[list[bool]] = []  # probe accept decisions
        totals = {"proposed": 0, "accepted": 0, "target_tokens": 0}

        for b, ex in enumerate(batch):
            prompt_text = _make_chat_prompt(tok, ex["question"], format_instruction=fmt_inst)
            prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
            prompt_len = len(prompt_ids)
            max_new = max(1, int(max_ctx) - int(prompt_len) - 1)
            for g in range(int(args.group_size)):
                seed = (int(args.seed) if int(args.seed) != 0 else 0) + step * 100000 + b * 1000 + g
                input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                if str(args.decode_mode) == "gram":
                    out_ids, stats = gram_decode(
                        model=model,
                        tokenizer=tok,
                        engine=engine,
                        input_ids=input_ids,
                        max_new_tokens=int(max_new),
                        draft_k=int(args.draft_k),
                        max_support=int(args.max_support),
                        draft_mode="infgram",
                        draft_sample=True,
                        draft_topk=int(args.draft_topk),
                        draft_temperature=float(args.draft_temperature),
                        draft_seed=int(seed),
                        gate=args.gate,
                        gate_topk=int(args.gate_topk),
                        gate_margin=float(args.gate_margin),
                        gate_prob=float(args.gate_prob),
                        reject_mode=args.reject_mode,
                        target_topk=int(args.target_topk),
                        target_temperature=float(args.target_temperature),
                        add_one_when_all_accepted=True,
                        raw_engine=False,
                        accept_probe=accept_probe,
                    )
                else:
                    from gram_decoding.decoding import GramDecodingStats

                    stats = GramDecodingStats()
                    out_ids = _generate_raw(
                        model=model,
                        input_ids=input_ids,
                        pad_token_id=int(tok.pad_token_id),
                        eos_token_id=int(tok.eos_token_id),
                        max_new=int(max_new),
                        topk=int(args.target_topk),
                        temperature=float(args.target_temperature),
                    )
                full = out_ids[0].tolist()
                text = tok.decode(full[prompt_len:], skip_special_tokens=True)
                if fmt_style == "none":
                    ans_r = gsm8k_reward(completion_text=text, answer_text=ex["answer"])
                    fmt_r = 0.0
                else:
                    ans_r, fmt_r = gsm8k_reward_components(completion_text=text, answer_text=ex["answer"], style=fmt_style)
                acc = (float(stats.accepted) / float(stats.proposed)) if stats.proposed else 0.0
                bonus_r = float(args.accept_bonus) * acc if float(args.accept_bonus) != 0.0 else 0.0
                r = float(ans_r) + float(bonus_r) + float(fmt_w) * float(fmt_r)
                totals["proposed"] += int(stats.proposed)
                totals["accepted"] += int(stats.accepted)
                totals["target_tokens"] += int(stats.target_tokens)
                all_seqs.append(full)
                all_prompt_lens.append(prompt_len)
                all_rewards.append(float(r))
                all_task_rewards.append(float(ans_r))
                all_bonus_rewards.append(float(bonus_r))
                all_format_rewards.append(float(fmt_r))
                all_groups.append(b)
                # Collect probe data if using probe gate
                if stats.probe_log_probs is not None:
                    all_probe_log_probs.append(stats.probe_log_probs)
                    all_probe_positions.append(stats.probe_positions or [])
                    all_probe_accepts.append(stats.probe_accepts or [])
                else:
                    all_probe_log_probs.append([])
                    all_probe_positions.append([])
                    all_probe_accepts.append([])

        # Compute advantages across entire batch (not per-question)
        # This ensures variance even when individual questions have consistent outcomes
        adv_by_sample = _adv_from_group(all_rewards)

        ids_cpu, attn_cpu, pl_cpu = pad_batch(seqs=all_seqs, prompt_lens=all_prompt_lens, pad_token_id=int(tok.pad_token_id))
        ids = ids_cpu.to(device)
        attn = attn_cpu.to(device)
        pl = pl_cpu.to(device)
        adv_t = torch.tensor(adv_by_sample, dtype=torch.float32, device=device)

        with torch.no_grad():
            old = token_logprobs_for_completions(
                model=model,
                input_ids=ids,
                attention_mask=attn,
                prompt_lens=pl,
                pad_token_id=int(tok.pad_token_id),
            )
            old_logp = old.token_logp.detach()
            ref_logp = None
            if ref is not None:
                ref_lp = token_logprobs_for_completions(
                    model=ref,
                    input_ids=ids,
                    attention_mask=attn,
                    prompt_lens=pl,
                    pad_token_id=int(tok.pad_token_id),
                )
                ref_logp = ref_lp.token_logp.detach()
            token_mask = old.token_mask.detach()

        bsz = ids.shape[0]
        mb = int(args.minibatch_size)
        if mb <= 0 or mb > bsz:
            mb = bsz
        epochs = max(1, int(args.update_epochs))
        idx_all = torch.arange(bsz, device=device)
        loss_sum = 0.0
        policy_sum = 0.0
        kl_sum = 0.0
        upd = 0
        for _ep in range(epochs):
            perm = idx_all[torch.randperm(bsz, device=device)]
            for start in range(0, bsz, mb):
                ix = perm[start : start + mb]
                new = token_logprobs_for_completions(
                    model=model,
                    input_ids=ids.index_select(0, ix),
                    attention_mask=attn.index_select(0, ix),
                    prompt_lens=pl.index_select(0, ix),
                    pad_token_id=int(tok.pad_token_id),
                )
                loss_obj = grpo_loss(
                    logp_new=new.token_logp,
                    logp_old=old_logp.index_select(0, ix),
                    token_mask=token_mask.index_select(0, ix),
                    advantages=adv_t.index_select(0, ix),
                    ref_logp=(ref_logp.index_select(0, ix) if ref_logp is not None else None),
                    clip_eps=float(args.clip_eps),
                    kl_beta=float(args.kl_beta),
                    ratio_mode=str(args.ratio_mode),
                    kl_estimator=str(args.kl_estimator),
                )
                opt.zero_grad(set_to_none=True)
                loss_obj.loss.backward()
                opt.step()
                loss_sum += float(loss_obj.loss.item())
                policy_sum += float(loss_obj.policy_loss.item())
                kl_sum += float(loss_obj.kl_loss.item())
                upd += 1

        # Update probe if using probe gate
        probe_loss_sum = 0.0
        probe_upd = 0
        if accept_probe is not None and probe_opt is not None and any(len(p) > 0 for p in all_probe_positions):
            # For each sequence with probe decisions, recompute log probs with gradients
            for i in range(len(all_seqs)):
                positions = all_probe_positions[i]
                accepts = all_probe_accepts[i]
                if len(positions) == 0:
                    continue
                adv_i = adv_by_sample[i]
                # Forward pass to get hidden states
                seq_tensor = torch.tensor([all_seqs[i]], dtype=torch.long, device=device)
                with torch.no_grad():
                    out = model(seq_tensor, output_hidden_states=True)
                    hidden = out.hidden_states[-1]  # [1, seq_len, H]
                # Compute probe log probs for stored positions and actions
                probe_log_probs_sum = torch.tensor(0.0, device=device)
                for pos, accept in zip(positions, accepts):
                    if pos >= hidden.shape[1]:
                        continue
                    h = hidden[0, pos, :].unsqueeze(0)  # [1, H]
                    action = torch.tensor([accept], dtype=torch.bool, device=device)
                    log_prob = accept_probe.log_prob_of_action(h, action)
                    probe_log_probs_sum = probe_log_probs_sum + log_prob.squeeze()
                # REINFORCE: maximize advantage-weighted log prob
                probe_loss = -float(args.probe_weight) * adv_i * probe_log_probs_sum
                probe_opt.zero_grad(set_to_none=True)
                probe_loss.backward()
                probe_opt.step()
                probe_loss_sum += float(probe_loss.abs().item())
                probe_upd += 1

        elapsed = time.perf_counter() - t0
        count = max(1, len(all_rewards))
        sums = torch.tensor(
            [
                float(sum(all_rewards)),
                float(sum(all_task_rewards)),
                float(sum(all_bonus_rewards)),
                float(sum(all_format_rewards)),
                float(totals["proposed"]),
                float(totals["accepted"]),
                float(totals["target_tokens"]),
                float(count),
            ],
            dtype=torch.float64,
            device=device,
        )
        _ddp_reduce(sums, dist.ReduceOp.SUM if world > 1 else None)
        sum_r, sum_task_r, sum_bonus_r, sum_fmt_r, proposed, accepted, target_tokens, total_count = [float(x) for x in sums.tolist()]
        mean_r = sum_r / max(1.0, total_count)
        mean_task_r = sum_task_r / max(1.0, total_count)
        mean_bonus_r = sum_bonus_r / max(1.0, total_count)
        mean_fmt_r = sum_fmt_r / max(1.0, total_count)
        acc_rate = (accepted / proposed) if proposed else 0.0

        denom = float(max(1, upd))
        losses = torch.tensor([loss_sum / denom, policy_sum / denom, kl_sum / denom], dtype=torch.float64, device=device)
        _ddp_reduce(losses, dist.ReduceOp.SUM if world > 1 else None)
        loss_mean, policy_mean, kl_mean = [float(x) / float(world) for x in losses.tolist()]
        probe_loss_mean = (probe_loss_sum / max(1, probe_upd)) if probe_upd > 0 else 0.0

        if is_main:
            probe_str = f" probe {probe_loss_mean:.4f}" if probe_upd > 0 else ""
            print(
                f"step {step} loss {loss_mean:.4f} policy {policy_mean:.4f} kl {kl_mean:.4f}{probe_str} "
                f"reward {mean_r:.3f} elapsed {elapsed:.1f}s"
            )
        if wandb_run is not None:
            log = {
                "step": step,
                "loss": float(loss_mean),
                "policy_loss": float(policy_mean),
                "kl_loss": float(kl_mean),
                "reward_mean": float(mean_r),
                "reward_task_mean": float(mean_task_r),
                "reward_bonus_mean": float(mean_bonus_r),
                "reward_format_mean": float(mean_fmt_r),
                "draft_proposed": int(proposed),
                "draft_accepted": int(accepted),
                "draft_acceptance_rate": float(acc_rate),
                "target_tokens": int(target_tokens),
                "elapsed_sec": float(elapsed),
                "update_steps": int(upd),
            }
            if device.type == "cuda":
                log["gpu_mem_allocated_gb"] = float(torch.cuda.memory_allocated() / (1024**3))
                log["gpu_mem_reserved_gb"] = float(torch.cuda.memory_reserved() / (1024**3))
            if probe_upd > 0:
                log["probe_loss"] = float(probe_loss_mean)
                log["probe_updates"] = int(probe_upd)
            wandb_run.log(log, step=step)

        step += 1
        if int(args.save_every) > 0 and step % int(args.save_every) == 0:
            if is_main:
                out = save_dir / f"step_{step}"
                out.mkdir(parents=True, exist_ok=True)
                m = model.module if hasattr(model, "module") else model
                m.save_pretrained(out)
                tok.save_pretrained(out)
                if accept_probe is not None:
                    torch.save(accept_probe.state_dict(), out / "accept_probe.pt")

    if world > 1:
        dist.barrier()
    if is_main:
        out = save_dir / "final"
        out.mkdir(parents=True, exist_ok=True)
        m = model.module if hasattr(model, "module") else model
        m.save_pretrained(out)
        tok.save_pretrained(out)
        if accept_probe is not None:
            torch.save(accept_probe.state_dict(), out / "accept_probe.pt")
    if wandb_run is not None:
        wandb_run.finish()
    if world > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
