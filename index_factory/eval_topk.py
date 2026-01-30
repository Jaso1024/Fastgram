#!/usr/bin/env python
"""
Evaluate gram index top-k alignment with Qwen 2.5.

Tests how well the gram index predicts the model's next token distribution.
"""
import argparse
import json
import time
from collections import defaultdict
from typing import List, Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastgram import gram


def get_model_topk(model, tokenizer, prompt_ids: List[int], k: int = 10) -> Dict[int, float]:
    """Get model's top-k predictions for next token."""
    with torch.no_grad():
        input_ids = torch.tensor([prompt_ids], device=model.device)
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]  # Last position
        probs = torch.softmax(logits, dim=-1)
        topk_probs, topk_ids = torch.topk(probs, k)
        return {int(tok): float(p) for tok, p in zip(topk_ids.cpu(), topk_probs.cpu())}


def get_gram_topk(engine, prompt_ids: List[int], k: int = 10) -> Dict[int, float]:
    """Get gram's top-k predictions for next token."""
    result = engine.ntd(prompt_ids, max_support=10000)
    if "error" in result:
        return {}

    items = list(result['result_by_token_id'].items())
    items.sort(key=lambda x: x[1]['prob'], reverse=True)
    return {int(tok): float(v['prob']) for tok, v in items[:k]}


def compute_topk_overlap(model_topk: Dict[int, float], gram_topk: Dict[int, float], k: int) -> Dict:
    """Compute overlap metrics between model and gram top-k."""
    model_set = set(list(model_topk.keys())[:k])
    gram_set = set(list(gram_topk.keys())[:k])

    overlap = model_set & gram_set

    # Check if model's top-1 is in gram's top-k
    model_top1 = list(model_topk.keys())[0] if model_topk else None
    top1_in_gram = model_top1 in gram_set if model_top1 else False

    # Check gram's rank of model's top-1
    gram_rank_of_top1 = None
    if model_top1 and model_top1 in gram_topk:
        gram_sorted = sorted(gram_topk.items(), key=lambda x: x[1], reverse=True)
        for i, (tok, _) in enumerate(gram_sorted):
            if tok == model_top1:
                gram_rank_of_top1 = i + 1
                break

    return {
        "overlap_count": len(overlap),
        "overlap_pct": len(overlap) / k if k > 0 else 0,
        "model_top1_in_gram_topk": top1_in_gram,
        "gram_rank_of_model_top1": gram_rank_of_top1,
        "gram_has_predictions": len(gram_topk) > 0,
        "gram_prediction_count": len(gram_topk),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dirs", nargs="+", required=True, help="Index directories")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="Model name")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--topk", type=int, default=10, help="Top-k to compare")
    parser.add_argument("--min-prompt-len", type=int, default=10, help="Min prompt length")
    parser.add_argument("--max-prompt-len", type=int, default=100, help="Max prompt length")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    print(f"Loading gram index from: {args.index_dirs}")
    engine = gram(
        index_dir=args.index_dirs,
        eos_token_id=tokenizer.eos_token_id or 151643,
        vocab_size=tokenizer.vocab_size or 151643,
        version=4,
        token_dtype="u32",
    )

    # Generate test prompts from common patterns
    test_prompts = [
        "The answer is",
        "Let me think step by step",
        "First, we need to",
        "To solve this problem",
        "The solution is",
        "We can see that",
        "Therefore, the answer",
        "In conclusion",
        "The result is",
        "By calculating",
        "If we consider",
        "Given that",
        "Since we know",
        "This means that",
        "We can conclude",
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nSolve: 3x + 5 = 20<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nExplain how to",
        "The derivative of",
        "To calculate the",
    ]

    # Expand with tokenized versions at different lengths
    all_prompts = []
    for prompt in test_prompts:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) >= args.min_prompt_len:
            all_prompts.append(ids[:args.max_prompt_len])
        # Also try with just partial prompts
        for end in range(args.min_prompt_len, min(len(ids), args.max_prompt_len) + 1, 5):
            all_prompts.append(ids[:end])

    # Dedupe
    seen = set()
    unique_prompts = []
    for p in all_prompts:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            unique_prompts.append(p)

    prompts = unique_prompts[:args.num_samples]
    print(f"Testing {len(prompts)} prompts...")

    results = []
    stats = defaultdict(list)

    for i, prompt_ids in enumerate(prompts):
        model_topk = get_model_topk(model, tokenizer, prompt_ids, args.topk)
        gram_topk = get_gram_topk(engine, prompt_ids, args.topk)

        metrics = compute_topk_overlap(model_topk, gram_topk, args.topk)
        results.append(metrics)

        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                stats[k].append(v)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(prompts)}] Avg overlap: {sum(stats['overlap_pct'])/len(stats['overlap_pct']):.1%}")

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    print(f"\nTop-{args.topk} Overlap:")
    print(f"  Average overlap: {sum(stats['overlap_pct'])/len(stats['overlap_pct']):.1%}")
    print(f"  Min overlap: {min(stats['overlap_pct']):.1%}")
    print(f"  Max overlap: {max(stats['overlap_pct']):.1%}")

    top1_hits = sum(1 for r in results if r['model_top1_in_gram_topk'])
    print(f"\nModel Top-1 in Gram Top-{args.topk}: {top1_hits}/{len(results)} ({top1_hits/len(results):.1%})")

    has_preds = sum(1 for r in results if r['gram_has_predictions'])
    print(f"Gram has predictions: {has_preds}/{len(results)} ({has_preds/len(results):.1%})")

    avg_pred_count = sum(stats['gram_prediction_count']) / len(stats['gram_prediction_count'])
    print(f"Avg gram predictions per prompt: {avg_pred_count:.1f}")

    ranks = [r['gram_rank_of_model_top1'] for r in results if r['gram_rank_of_model_top1'] is not None]
    if ranks:
        print(f"\nGram rank of model's top-1 (when found):")
        print(f"  Average rank: {sum(ranks)/len(ranks):.1f}")
        print(f"  Median rank: {sorted(ranks)[len(ranks)//2]}")
        print(f"  In top-1: {sum(1 for r in ranks if r == 1)}/{len(ranks)}")
        print(f"  In top-3: {sum(1 for r in ranks if r <= 3)}/{len(ranks)}")
        print(f"  In top-5: {sum(1 for r in ranks if r <= 5)}/{len(ranks)}")


if __name__ == "__main__":
    main()
