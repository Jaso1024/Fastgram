#!/usr/bin/env python
"""
Unified ingest script for building a medium-sized gram index.

Handles multiple dataset formats:
- OpenThoughts3: conversations with from/value
- Tulu-3-SFT: messages with role/content
- Magpie: messages with role/content
- FineWeb-Edu: plain text field
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


def format_tulu_messages(item: dict) -> str:
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


def format_fineweb(item: dict) -> str:
    """Format FineWeb-Edu plain text."""
    return item.get("text", "")


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


def iter_openthoughts(limit: Optional[int] = None) -> Iterator[str]:
    """Iterate OpenThoughts3-1.2M dataset."""
    print("Loading OpenThoughts3-1.2M...")
    ds = datasets.load_dataset(
        "open-thoughts/OpenThoughts3-1.2M",
        split="train",
        streaming=True,
    )
    count = 0
    for item in ds:
        text = format_openthoughts(item)
        if text:
            yield text
            count += 1
            if limit and count >= limit:
                return


def iter_tulu3(limit: Optional[int] = None) -> Iterator[str]:
    """Iterate Tulu-3-SFT-Mixture dataset."""
    print("Loading Tulu-3-SFT-Mixture...")
    ds = datasets.load_dataset(
        "allenai/tulu-3-sft-mixture",
        split="train",
        streaming=True,
    )
    count = 0
    for item in ds:
        text = format_tulu_messages(item)
        if text:
            yield text
            count += 1
            if limit and count >= limit:
                return


def iter_magpie(limit: Optional[int] = None) -> Iterator[str]:
    """Iterate Magpie-Qwen2.5-Pro-300K dataset."""
    print("Loading Magpie-Qwen2.5-Pro-300K-Filtered...")
    ds = datasets.load_dataset(
        "Magpie-Align/Magpie-Qwen2.5-Pro-300K-Filtered",
        split="train",
        streaming=True,
    )
    count = 0
    for item in ds:
        text = format_tulu_messages(item)
        if text:
            yield text
            count += 1
            if limit and count >= limit:
                return


def iter_fineweb_edu(limit: Optional[int] = None) -> Iterator[str]:
    """Iterate FineWeb-Edu sample."""
    print("Loading FineWeb-Edu (sample-10BT)...")
    ds = datasets.load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    count = 0
    for item in ds:
        text = format_fineweb(item)
        if text and len(text) > 100:  # Skip very short docs
            yield text
            count += 1
            if limit and count >= limit:
                return


def main():
    parser = argparse.ArgumentParser(description="Build medium gram index")
    parser.add_argument("--output-dir", default="indices/medium", help="Output directory")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B-Instruct", help="Tokenizer ID")
    parser.add_argument("--proc", type=int, default=min(8, os.cpu_count() or 4), help="Number of processes")
    parser.add_argument("--buffer-size", type=int, default=500, help="Write buffer size")
    parser.add_argument("--shard-size", type=int, default=500_000_000, help="Tokens per shard (~2GB)")
    # Dataset limits
    parser.add_argument("--openthoughts-limit", type=int, default=None, help="Limit OpenThoughts docs")
    parser.add_argument("--tulu-limit", type=int, default=None, help="Limit Tulu docs")
    parser.add_argument("--magpie-limit", type=int, default=None, help="Limit Magpie docs")
    parser.add_argument("--fineweb-limit", type=int, default=8_000_000, help="Limit FineWeb docs")
    # Skip flags
    parser.add_argument("--skip-openthoughts", action="store_true")
    parser.add_argument("--skip-tulu", action="store_true")
    parser.add_argument("--skip-magpie", action="store_true")
    parser.add_argument("--skip-fineweb", action="store_true")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Tokenizer: {args.tokenizer}")
    print(f"Output: {out_dir}")
    print(f"Processes: {args.proc}")

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
        print(f"\rDocs: {total_docs:,} | Tokens: {total_tokens:,} | Shard: {shard_idx} | Rate: {rate:,.0f} tok/s", end="", flush=True)

    try:
        batch_docs: list[str] = []

        # 1. OpenThoughts3 (reasoning)
        if not args.skip_openthoughts:
            print("\n\n=== Ingesting OpenThoughts3-1.2M ===")
            for text in iter_openthoughts(args.openthoughts_limit):
                batch_docs.append(text)
                if len(batch_docs) >= args.buffer_size:
                    flush_batch(batch_docs)
                    batch_docs = []
            flush_batch(batch_docs)
            batch_docs = []
            print(f"\n>>> OpenThoughts done: {total_docs:,} docs, {total_tokens:,} tokens")

        # 2. Tulu-3-SFT (instructions)
        if not args.skip_tulu:
            print("\n\n=== Ingesting Tulu-3-SFT-Mixture ===")
            for text in iter_tulu3(args.tulu_limit):
                batch_docs.append(text)
                if len(batch_docs) >= args.buffer_size:
                    flush_batch(batch_docs)
                    batch_docs = []
            flush_batch(batch_docs)
            batch_docs = []
            print(f"\n>>> Tulu-3 done: {total_docs:,} docs, {total_tokens:,} tokens")

        # 3. Magpie (synthetic instructions)
        if not args.skip_magpie:
            print("\n\n=== Ingesting Magpie-Qwen2.5-Pro-300K ===")
            for text in iter_magpie(args.magpie_limit):
                batch_docs.append(text)
                if len(batch_docs) >= args.buffer_size:
                    flush_batch(batch_docs)
                    batch_docs = []
            flush_batch(batch_docs)
            batch_docs = []
            print(f"\n>>> Magpie done: {total_docs:,} docs, {total_tokens:,} tokens")

        # 4. FineWeb-Edu (general knowledge)
        if not args.skip_fineweb:
            print("\n\n=== Ingesting FineWeb-Edu (sample-10BT) ===")
            for text in iter_fineweb_edu(args.fineweb_limit):
                batch_docs.append(text)
                if len(batch_docs) >= args.buffer_size:
                    flush_batch(batch_docs)
                    batch_docs = []
            flush_batch(batch_docs)
            batch_docs = []
            print(f"\n>>> FineWeb-Edu done: {total_docs:,} docs, {total_tokens:,} tokens")

    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving progress...")
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
    print(f"DONE!")
    print(f"Total docs: {total_docs:,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Shards: {shard_idx + 1}")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
