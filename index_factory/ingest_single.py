#!/usr/bin/env python
"""
Single-dataset ingest script for parallel index building.

Usage:
  python ingest_single.py --dataset openthoughts --output-dir indices/openthoughts
  python ingest_single.py --dataset tulu --output-dir indices/tulu
  python ingest_single.py --dataset magpie --output-dir indices/magpie
  python ingest_single.py --dataset fineweb --output-dir indices/fineweb --limit 8000000
"""
import argparse
import multiprocessing as mp
import os
import struct
import time
from pathlib import Path
from typing import Iterator, Optional

import datasets
from transformers import AutoTokenizer


# ============================================================================
# Format handlers
# ============================================================================

def format_openthoughts(item: dict) -> str:
    """Format OpenThoughts3 conversation to training text."""
    convs = item.get("conversations", [])
    parts = []
    for msg in convs:
        role = msg.get("from", "")
        content = msg.get("value", "")
        if role == "human":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "gpt":
            parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
    return "\n".join(parts)


def format_messages(item: dict) -> str:
    """Format Tulu-3/Magpie messages to training text."""
    msgs = item.get("messages", [])
    parts = []
    for msg in msgs:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        elif role == "system":
            parts.append(f"<|im_start|>system\n{content}<|im_end|>")
    return "\n".join(parts)


def format_text(item: dict) -> str:
    """Format plain text field."""
    return item.get("text", "")


# ============================================================================
# Dataset iterators
# ============================================================================

DATASETS = {
    "openthoughts": {
        "hf_id": "open-thoughts/OpenThoughts3-1.2M",
        "formatter": format_openthoughts,
        "default_limit": None,
    },
    "tulu": {
        "hf_id": "allenai/tulu-3-sft-mixture",
        "formatter": format_messages,
        "default_limit": None,
    },
    "magpie": {
        "hf_id": "Magpie-Align/Magpie-Qwen2.5-Pro-300K-Filtered",
        "formatter": format_messages,
        "default_limit": None,
    },
    "fineweb": {
        "hf_id": "HuggingFaceFW/fineweb-edu",
        "hf_name": "sample-10BT",
        "formatter": format_text,
        "default_limit": 8_000_000,
        "min_length": 100,
    },
}


def iter_dataset(dataset_key: str, limit: Optional[int] = None) -> Iterator[str]:
    """Iterate over a dataset."""
    cfg = DATASETS[dataset_key]
    hf_id = cfg["hf_id"]
    hf_name = cfg.get("hf_name")
    formatter = cfg["formatter"]
    min_length = cfg.get("min_length", 0)

    if limit is None:
        limit = cfg.get("default_limit")

    print(f"Loading {hf_id}...")

    load_kwargs = {"split": "train", "streaming": True}
    if hf_name:
        load_kwargs["name"] = hf_name

    ds = datasets.load_dataset(hf_id, **load_kwargs)

    count = 0
    for item in ds:
        text = formatter(item)
        if text and len(text) >= min_length:
            yield text
            count += 1
            if limit and count >= limit:
                return


# ============================================================================
# Tokenization
# ============================================================================

class TokenizerWrapper:
    def __init__(self, tokenizer_id: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
        self.eos_token_id = self.tokenizer.eos_token_id
        if self.eos_token_id is None:
            if "Qwen" in tokenizer_id:
                self.eos_token_id = 151643
            else:
                raise ValueError("Tokenizer has no EOS token ID")
        self.vocab_size = self.tokenizer.vocab_size

    def encode(self, text: str) -> list[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        ids.append(self.eos_token_id)
        return ids


_TOKENIZER: Optional[TokenizerWrapper] = None


def worker_init(tokenizer_id: str):
    global _TOKENIZER
    _TOKENIZER = TokenizerWrapper(tokenizer_id)


def worker_process(text: str) -> list[int]:
    assert _TOKENIZER is not None
    return _TOKENIZER.encode(text)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Ingest single dataset for gram index")
    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()), help="Dataset to ingest")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B-Instruct", help="Tokenizer ID")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of docs")
    parser.add_argument("--proc", type=int, default=min(8, os.cpu_count() or 4), help="Processes")
    parser.add_argument("--buffer-size", type=int, default=500, help="Write buffer size")
    parser.add_argument("--shard-size", type=int, default=500_000_000, help="Tokens per shard")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {out_dir}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Limit: {args.limit or 'none'}")
    print(f"Processes: {args.proc}")
    print(f"{'='*60}")

    # Setup multiprocessing
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(args.proc, initializer=worker_init, initargs=(args.tokenizer,))

    start_time = time.time()
    total_tokens = 0
    total_docs = 0
    shard_idx = 0
    current_shard_tokens = 0
    current_f = open(out_dir / f"tokenized.{shard_idx}", "wb")

    def flush_batch(batch_docs: list[str]):
        nonlocal total_tokens, total_docs, current_shard_tokens, shard_idx, current_f

        if not batch_docs:
            return

        results = pool.map(worker_process, batch_docs)

        for ids in results:
            if current_shard_tokens + len(ids) > args.shard_size:
                current_f.close()
                shard_idx += 1
                current_f = open(out_dir / f"tokenized.{shard_idx}", "wb")
                current_shard_tokens = 0
                print(f"\n>>> Started shard {shard_idx}")

            for token in ids:
                current_f.write(struct.pack("<I", token))

            current_shard_tokens += len(ids)
            total_tokens += len(ids)

        total_docs += len(results)
        elapsed = time.time() - start_time
        rate = total_tokens / max(1, elapsed)
        print(f"\rDocs: {total_docs:,} | Tokens: {total_tokens:,} | Shard: {shard_idx} | {rate:,.0f} tok/s", end="", flush=True)

    try:
        batch_docs: list[str] = []
        for text in iter_dataset(args.dataset, args.limit):
            batch_docs.append(text)
            if len(batch_docs) >= args.buffer_size:
                flush_batch(batch_docs)
                batch_docs = []
        flush_batch(batch_docs)

    except KeyboardInterrupt:
        print("\n\nInterrupted!")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if not current_f.closed:
            current_f.close()
        pool.close()
        pool.join()

    elapsed = time.time() - start_time
    print(f"\n\n{'='*60}")
    print(f"DONE: {args.dataset}")
    print(f"Docs: {total_docs:,}")
    print(f"Tokens: {total_tokens:,}")
    print(f"Shards: {shard_idx + 1}")
    print(f"Time: {elapsed/60:.1f} min")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
