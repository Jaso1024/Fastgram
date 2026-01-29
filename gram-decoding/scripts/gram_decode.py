#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
_GRAM_DECODING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_GRAM_DECODING))

from fastgram import GramEngine

from gram_decoding import gram_decode


def _infer_token_dtype(vocab_size: int) -> str:
    if vocab_size <= 2**8:
        return "u8"
    if vocab_size <= 2**16:
        return "u16"
    return "u32"


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


def main() -> int:
    p = argparse.ArgumentParser(description="Gram decoding (fastgram draft, HF verifier)")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--index-dir", required=True)
    p.add_argument("--prompt", default="Write a short paragraph about fast text search.")
    p.add_argument("--draft-k", type=int, default=8)
    p.add_argument("--max-support", type=int, default=200)
    p.add_argument("--version", type=int, default=4)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--token-dtype", default="")
    p.add_argument("--draft-mode", choices=["infgram", "primitive"], default="infgram")
    p.add_argument("--draft-sample", action="store_true", default=False)
    p.add_argument("--draft-topk", type=int, default=0)
    p.add_argument("--draft-temperature", type=float, default=1.0)
    p.add_argument("--draft-seed", type=int, default=0)
    p.add_argument("--add-one-when-all-accepted", action="store_true", default=True)
    p.add_argument("--raw-engine", action="store_true", default=False)
    p.add_argument("--device", default="")
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    p.add_argument("--gate", choices=["none", "attn", "spec", "topk", "margin", "prob"], default="none")
    p.add_argument("--gate-topk", type=int, default=10)
    p.add_argument("--gate-margin", type=float, default=2.0)
    p.add_argument("--gate-prob", type=float, default=0.1)
    p.add_argument("--attn-threshold", type=float, default=0.02)
    p.add_argument("--reject-mode", choices=["greedy", "sample"], default="greedy")
    p.add_argument("--target-topk", type=int, default=0)
    p.add_argument("--target-temperature", type=float, default=1.0)
    p.add_argument("--attn-implementation", default="")
    args = p.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        raise SystemExit(f"missing dependency: {e}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dtype == "auto":
        torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    elif args.dtype == "float32":
        torch_dtype = torch.float32
    elif args.dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.bfloat16

    model_kwargs = {"torch_dtype": torch_dtype, "trust_remote_code": True}
    if args.gate == "attn" and args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.to(device)
    model.eval()

    prompt_ids = tok.encode(args.prompt, add_special_tokens=False)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    max_new = max(1, _infer_max_context_len(tok, model) - len(prompt_ids) - 1)

    eos = tok.eos_token_id
    if eos is None:
        raise SystemExit("tokenizer has no eos_token_id")
    vocab = getattr(tok, "vocab_size", None) or len(tok)
    token_dtype = args.token_dtype or _infer_token_dtype(int(vocab))

    engine = GramEngine(
        index_dir=args.index_dir,
        eos_token_id=int(eos),
        vocab_size=int(vocab),
        version=int(args.version),
        token_dtype=token_dtype,
        threads=int(args.threads),
    )

    t0 = time.perf_counter()
    out_ids, stats = gram_decode(
        model=model,
        tokenizer=tok,
        engine=engine,
        input_ids=input_ids,
        max_new_tokens=int(max_new),
        draft_k=int(args.draft_k),
        max_support=int(args.max_support),
        draft_mode=args.draft_mode,
        draft_sample=bool(args.draft_sample),
        draft_topk=int(args.draft_topk),
        draft_temperature=float(args.draft_temperature),
        draft_seed=(int(args.draft_seed) if int(args.draft_seed) != 0 else None),
        add_one_when_all_accepted=bool(args.add_one_when_all_accepted),
        raw_engine=bool(args.raw_engine),
        gate=args.gate,
        gate_topk=int(args.gate_topk),
        gate_margin=float(args.gate_margin),
        gate_prob=float(args.gate_prob),
        attn_threshold=float(args.attn_threshold),
        reject_mode=args.reject_mode,
        target_topk=int(args.target_topk),
        target_temperature=float(args.target_temperature),
    )
    t1 = time.perf_counter()

    print("=== output ===")
    print(tok.decode(out_ids[0], skip_special_tokens=True))
    print("=== stats ===")
    print(f"proposed: {stats.proposed} accepted: {stats.accepted} target_tokens: {stats.target_tokens}")
    if stats.proposed:
        print(f"acceptance_rate: {stats.accepted / stats.proposed:.3f}")
    print(f"elapsed_sec: {t1 - t0:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
