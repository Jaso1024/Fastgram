#!/usr/bin/env python
"""Inspect what the gram draft model proposes vs what the target model wants."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "gram-decoding"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastgram import GramEngine
from gram_decoding.draft import draft_best_token_id, draft_sample_token_id
import random


def main():
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    index_dir = "index/reasoning"

    print(f"Loading model: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    eos = tok.eos_token_id
    vocab = tok.vocab_size

    if vocab <= 2**16:
        token_dtype = "u16"
    else:
        token_dtype = "u32"

    print(f"Loading gram index: {index_dir}")
    engine = GramEngine(
        index_dir=index_dir,
        eos_token_id=eos,
        vocab_size=vocab,
        version=4,
        token_dtype=token_dtype,
        threads=0,
    )

    # Test prompt with DeepSeek-style reasoning instruction
    question = "What is 15 + 27?"
    base_prompt = f"Q: {question}\nA:"
    # Add DeepSeek-style instruction to encourage matching reasoning patterns
    format_instruction = "Let me work through this problem step by step.\n\nFirst, I need to understand what the problem is asking."
    prompt = base_prompt + "\n" + format_instruction

    if hasattr(tok, "apply_chat_template"):
        msgs = [{"role": "user", "content": prompt}]
        prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        prompt_text = prompt

    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}\n")

    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
    prefix_ids = list(prompt_ids)

    rng = random.Random(42)
    draft_k = 16
    draft_topk = 20

    # Generate a few tokens and compare draft vs target
    for step in range(10):
        # Get draft proposals
        draft_ids = []
        draft_prefix = prefix_ids.copy()
        for _ in range(draft_k):
            next_id = draft_sample_token_id(
                engine=engine,
                prefix_ids=draft_prefix,
                max_support=500,
                mode="infgram",
                raw_engine=False,
                topk=draft_topk,
                temperature=1.0,
                rng=rng,
            )
            if next_id is None:
                break
            draft_ids.append(next_id)
            draft_prefix.append(next_id)

        if not draft_ids:
            print(f"Step {step}: No draft proposals")
            # Fall back to model
            input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(input_ids).logits[0, -1, :]
                next_id = int(torch.argmax(logits).item())
            prefix_ids.append(next_id)
            print(f"  Model chose: {repr(tok.decode([next_id]))}")
            if next_id == eos:
                break
            continue

        # Get model's preference for each position
        input_ids = torch.tensor([prefix_ids + draft_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_ids).logits[0]

        print(f"Step {step}: Draft proposed {len(draft_ids)} tokens")
        accepted = 0
        for j, draft_id in enumerate(draft_ids):
            pos = len(prefix_ids) + j - 1
            if pos < 0:
                pos = 0
            logits_pos = logits[pos]
            target_id = int(torch.argmax(logits_pos).item())

            # Get probabilities
            probs = torch.softmax(logits_pos, dim=-1)
            draft_prob = float(probs[draft_id].item())
            target_prob = float(probs[target_id].item())

            # Get ranks
            sorted_indices = torch.argsort(probs, descending=True)
            draft_rank = int((sorted_indices == draft_id).nonzero(as_tuple=True)[0].item()) + 1

            draft_tok = repr(tok.decode([draft_id]))
            target_tok = repr(tok.decode([target_id]))

            match = "✓" if draft_id == target_id else "✗"
            print(f"  [{j}] Draft: {draft_tok:20s} (p={draft_prob:.4f}, rank={draft_rank:4d}) | "
                  f"Target: {target_tok:20s} (p={target_prob:.4f}) {match}")

            if draft_id == target_id:
                accepted += 1
            else:
                # Stop at first mismatch
                prefix_ids.append(target_id)
                break
        else:
            # All accepted
            prefix_ids.extend(draft_ids)

        print(f"  Accepted: {accepted}/{len(draft_ids)}")

        if prefix_ids[-1] == eos:
            break

    print(f"\n{'='*60}")
    print(f"Final output: {tok.decode(prefix_ids[len(prompt_ids):])}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
