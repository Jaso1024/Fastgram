#!/usr/bin/env python
"""Production-grade RL training script for LLMs.

Supports GRPO, GSPO, and DAPO algorithms with configurable models and rewards.

Usage:
    # Single GPU
    python train.py --model Qwen/Qwen2.5-1.5B-Instruct --algorithm grpo

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=4 train.py --model Qwen/Qwen2.5-1.5B-Instruct

    # With wandb logging
    python train.py --model Qwen/Qwen2.5-1.5B-Instruct --wandb --wandb-project my-project
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

# Add gram-rl to path for imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_GRAM_RL_DIR = _SCRIPT_DIR.parent
if str(_GRAM_RL_DIR) not in sys.path:
    sys.path.insert(0, str(_GRAM_RL_DIR))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RL training for LLMs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                   help="HuggingFace model name or path")
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None,
                   help="Model dtype (default: bfloat16 on CUDA, float32 on CPU)")

    # Algorithm
    p.add_argument("--algorithm", choices=["grpo", "gspo", "dapo"], default="grpo",
                   help="RL algorithm to use")
    p.add_argument("--clip-eps", type=float, default=0.2,
                   help="PPO clipping epsilon")
    p.add_argument("--kl-beta", type=float, default=0.02,
                   help="KL penalty coefficient (0 to disable)")

    # Training
    p.add_argument("--steps", type=int, default=1000,
                   help="Number of training steps")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Number of prompts per batch")
    p.add_argument("--group-size", type=int, default=8,
                   help="Number of completions per prompt")
    p.add_argument("--max-new-tokens", type=int, default=256,
                   help="Maximum tokens to generate")
    p.add_argument("--lr", type=float, default=5e-6,
                   help="Learning rate")
    p.add_argument("--max-grad-norm", type=float, default=1.0,
                   help="Gradient clipping norm (0 to disable)")
    p.add_argument("--update-epochs", type=int, default=1,
                   help="Number of epochs per batch")

    # Generation
    p.add_argument("--temperature", type=float, default=0.8,
                   help="Sampling temperature")
    p.add_argument("--top-k", type=int, default=50,
                   help="Top-k sampling")

    # Task/Reward
    p.add_argument("--task", choices=["gsm8k"], default="gsm8k",
                   help="Task/dataset to train on")
    p.add_argument("--format-style", choices=["none", "hash4", "cot"], default="none",
                   help="Answer format style for GSM8K")
    p.add_argument("--format-weight", type=float, default=0.0,
                   help="Weight for format reward")

    # Checkpointing
    p.add_argument("--save-dir", default="runs/rl",
                   help="Directory to save checkpoints")
    p.add_argument("--save-every", type=int, default=200,
                   help="Save checkpoint every N steps (0 to disable)")

    # Logging
    p.add_argument("--log-every", type=int, default=1,
                   help="Log metrics every N steps")
    p.add_argument("--wandb", action="store_true",
                   help="Enable wandb logging")
    p.add_argument("--wandb-project", default="rl-training",
                   help="Wandb project name")
    p.add_argument("--wandb-entity", default="",
                   help="Wandb entity/team")
    p.add_argument("--wandb-name", default="",
                   help="Wandb run name")
    p.add_argument("--wandb-tags", default="",
                   help="Comma-separated wandb tags")

    # Reproducibility
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed")

    return p.parse_args()


def create_algorithm(args: argparse.Namespace):
    """Create the RL algorithm based on args."""
    from rl.config import AlgorithmConfig

    if args.algorithm == "grpo":
        config = AlgorithmConfig.grpo(
            clip_eps=args.clip_eps,
            kl_beta=args.kl_beta,
        )
    elif args.algorithm == "gspo":
        config = AlgorithmConfig.gspo(
            clip_eps=args.clip_eps,
            kl_beta=args.kl_beta,
        )
    elif args.algorithm == "dapo":
        config = AlgorithmConfig.dapo(
            clip_eps=args.clip_eps,
            kl_beta=args.kl_beta,
        )
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")

    from rl.algorithm import PolicyGradientAlgorithm
    return PolicyGradientAlgorithm(config)


def create_reward_fn(args: argparse.Namespace):
    """Create the reward function based on task."""
    if args.task == "gsm8k":
        from rl.rewards import GSM8KReward
        return GSM8KReward(
            format_style=args.format_style,
            format_weight=args.format_weight,
        )
    else:
        raise ValueError(f"Unknown task: {args.task}")


def load_dataset(args: argparse.Namespace):
    """Load the training dataset."""
    from datasets import load_dataset

    if args.task == "gsm8k":
        return load_dataset("gsm8k", "main", split="train")
    else:
        raise ValueError(f"Unknown task: {args.task}")


def format_prompt(tokenizer, example: dict, args: argparse.Namespace) -> str:
    """Format a dataset example as a prompt."""
    if args.task == "gsm8k":
        from rl.rewards import gsm8k_prompt, gsm8k_format_instruction

        base = gsm8k_prompt(example["question"])
        instruction = gsm8k_format_instruction(args.format_style)
        if instruction:
            base = base + "\n" + instruction

        if hasattr(tokenizer, "apply_chat_template"):
            msgs = [{"role": "user", "content": base}]
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        return base
    else:
        raise ValueError(f"Unknown task: {args.task}")


def main() -> int:
    args = parse_args()

    # Imports
    import torch
    from rl.models import HuggingFaceModel, GenerationConfig, pad_sequences
    from rl.algorithm import compute_advantages
    from rl.utils import (
        init_distributed,
        cleanup_distributed,
        barrier,
        gather_scalars,
    )

    # Initialize distributed
    dist_info = init_distributed()
    is_main = dist_info.is_main
    device = dist_info.device

    # Seed
    if args.seed:
        seed = args.seed + dist_info.rank
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    # Wandb
    wandb_run = None
    if args.wandb and is_main:
        import wandb
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.wandb_name or None,
            config=vars(args),
            tags=tags or None,
        )

    # Create model
    if is_main:
        print(f"Loading model: {args.model}")

    model = HuggingFaceModel(
        model_name=args.model,
        device=str(device),
        dtype=args.dtype,
        use_ddp=(dist_info.world_size > 1),
        local_rank=dist_info.local_rank,
    )
    model.load()
    model.train()

    # Create reference model for KL penalty
    ref_model = None
    if args.kl_beta > 0:
        if is_main:
            print("Loading reference model for KL penalty")
        ref_model = model.create_reference()

    # Create algorithm and reward
    algorithm = create_algorithm(args)
    reward_fn = create_reward_fn(args)

    # Load dataset
    dataset = load_dataset(args)
    if is_main:
        print(f"Loaded {len(dataset)} examples")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Generation config
    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        do_sample=True,
    )

    # Save directory
    save_dir = Path(args.save_dir)
    if is_main:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    t0 = time.perf_counter()

    for step in range(args.steps):
        step_start = time.perf_counter()

        # Sample prompts
        prompt_indices = [random.randrange(len(dataset)) for _ in range(args.batch_size)]
        examples = [dataset[i] for i in prompt_indices]

        # Generate completions
        all_sequences: list[list[int]] = []
        all_prompt_lens: list[int] = []
        all_rewards: list[float] = []
        all_reward_components: list[dict] = []
        all_completion_lens: list[int] = []
        all_groups: list[int] = []

        for prompt_idx, example in enumerate(examples):
            prompt_text = format_prompt(model.tokenizer, example, args)
            prompt_ids = model.tokenizer.encode(prompt_text, add_special_tokens=False)
            prompt_len = len(prompt_ids)

            # Batch generation: create batch of identical prompts
            batch_input = torch.tensor(
                [prompt_ids] * args.group_size, dtype=torch.long, device=device
            )
            gen_config.seed = random.randint(0, 2**31)
            output = model.generate(batch_input, config=gen_config)

            # Process all completions from batch
            for seq_idx in range(args.group_size):
                full_seq = output.sequences[seq_idx].tolist()
                completion_ids = full_seq[prompt_len:]
                completion_text = model.tokenizer.decode(
                    completion_ids, skip_special_tokens=True
                )

                # Compute reward
                reward_output = reward_fn(
                    completion_text=completion_text,
                    prompt_text=prompt_text,
                    completion_ids=completion_ids,
                    prompt_ids=prompt_ids,
                    example=example,
                )

                all_sequences.append(full_seq)
                all_prompt_lens.append(prompt_len)
                all_rewards.append(reward_output.total)
                all_reward_components.append(reward_output.components)
                all_completion_lens.append(len(completion_ids))
                all_groups.append(prompt_idx)

        # Compute advantages
        advantages = compute_advantages(all_rewards, normalize=True)

        # Prepare batch
        input_ids, attention_mask, prompt_lens = pad_sequences(
            sequences=all_sequences,
            pad_token_id=model.pad_token_id,
            prompt_lens=all_prompt_lens,
        )
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        prompt_lens = prompt_lens.to(device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)

        # Compute old log probs (detached)
        with torch.no_grad():
            old_logprobs = model.compute_logprobs(input_ids, attention_mask, prompt_lens)
            old_logp = old_logprobs.token_logp.detach()
            token_mask = old_logprobs.token_mask.detach()

            ref_logp = None
            if ref_model is not None:
                ref_out = ref_model.compute_logprobs(input_ids, attention_mask, prompt_lens)
                ref_logp = ref_out.token_logp.detach()

        # Policy gradient updates
        loss_sum = 0.0
        policy_loss_sum = 0.0
        kl_loss_sum = 0.0
        num_updates = 0

        for _ in range(args.update_epochs):
            # Forward pass
            new_logprobs = model.compute_logprobs(input_ids, attention_mask, prompt_lens)

            # Compute loss
            output = algorithm.compute_loss(
                logp_new=new_logprobs.token_logp,
                logp_old=old_logp,
                token_mask=token_mask,
                advantages=advantages_t,
                ref_logp=ref_logp,
            )

            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            output.loss.backward()

            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()

            loss_sum += output.loss.item()
            policy_loss_sum += output.policy_loss.item()
            kl_loss_sum += output.kl_loss.item()
            num_updates += 1

        # Compute local statistics
        correctness_rewards = [c.get("correctness", 0.0) for c in all_reward_components]
        format_rewards = [c.get("format", 0.0) for c in all_reward_components]
        mean_completion_len = sum(all_completion_lens) / max(len(all_completion_lens), 1)
        max_completion_len = max(all_completion_lens) if all_completion_lens else 0
        min_completion_len = min(all_completion_lens) if all_completion_lens else 0

        # Aggregate metrics across processes
        metrics = gather_scalars(
            [
                sum(all_rewards),
                len(all_rewards),
                loss_sum,
                policy_loss_sum,
                kl_loss_sum,
                num_updates,
                sum(correctness_rewards),
                sum(format_rewards),
                sum(all_completion_lens),
            ],
            device=device,
        )
        (reward_sum, count, loss_agg, policy_agg, kl_agg, updates_agg,
         correctness_sum, format_sum, completion_len_sum) = metrics.tolist()

        mean_reward = reward_sum / max(count, 1)
        mean_loss = loss_agg / max(updates_agg, 1)
        mean_policy_loss = policy_agg / max(updates_agg, 1)
        mean_kl_loss = kl_agg / max(updates_agg, 1)
        mean_correctness = correctness_sum / max(count, 1)
        mean_format = format_sum / max(count, 1)
        avg_completion_len = completion_len_sum / max(count, 1)

        # Advantage statistics
        adv_mean = sum(advantages) / max(len(advantages), 1)
        adv_std = (sum((a - adv_mean) ** 2 for a in advantages) / max(len(advantages), 1)) ** 0.5
        adv_min = min(advantages) if advantages else 0.0
        adv_max = max(advantages) if advantages else 0.0

        step_time = time.perf_counter() - step_start
        total_time = time.perf_counter() - t0

        # Logging
        if is_main and (step + 1) % args.log_every == 0:
            print(
                f"step {step + 1}/{args.steps} | "
                f"loss {mean_loss:.4f} | "
                f"policy {mean_policy_loss:.4f} | "
                f"kl {mean_kl_loss:.4f} | "
                f"reward {mean_reward:.3f} | "
                f"correct {mean_correctness:.3f} | "
                f"time {step_time:.1f}s"
            )

        if wandb_run is not None:
            log_dict = {
                # Core metrics
                "train/loss": mean_loss,
                "train/policy_loss": mean_policy_loss,
                "train/kl_loss": mean_kl_loss,
                # Rewards
                "reward/total": mean_reward,
                "reward/correctness": mean_correctness,
                "reward/format": mean_format,
                "reward/min": min(all_rewards) if all_rewards else 0.0,
                "reward/max": max(all_rewards) if all_rewards else 0.0,
                # Advantages
                "advantage/mean": adv_mean,
                "advantage/std": adv_std,
                "advantage/min": adv_min,
                "advantage/max": adv_max,
                # Generation stats
                "generation/completion_len_mean": avg_completion_len,
                "generation/completion_len_local_max": max_completion_len,
                "generation/completion_len_local_min": min_completion_len,
                "generation/num_samples": int(count),
                # Algorithm diagnostics
                "algorithm/clip_fraction": output.clip_fraction,
                "algorithm/mean_ratio": output.mean_ratio,
                # Timing
                "timing/step_time": step_time,
                "timing/total_time": total_time,
                "timing/samples_per_sec": count / max(step_time, 0.001),
            }
            if device.type == "cuda":
                log_dict["system/gpu_mem_gb"] = torch.cuda.memory_allocated() / (1024**3)
                log_dict["system/gpu_mem_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
            wandb_run.log(log_dict, step=step + 1)

        # Checkpointing
        if args.save_every > 0 and (step + 1) % args.save_every == 0:
            barrier()
            if is_main:
                ckpt_dir = save_dir / f"step_{step + 1}"
                model.save(str(ckpt_dir))
                print(f"Saved checkpoint to {ckpt_dir}")

    # Final save
    barrier()
    if is_main:
        final_dir = save_dir / "final"
        model.save(str(final_dir))
        print(f"Saved final model to {final_dir}")

    if wandb_run is not None:
        wandb_run.finish()

    cleanup_distributed()

    if is_main:
        print(f"Training complete. Total time: {time.perf_counter() - t0:.1f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
