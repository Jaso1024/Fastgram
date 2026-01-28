import argparse
import multiprocessing as mp
import os
import struct
import time
from pathlib import Path
from typing import Dict, List, Optional

import datasets
from transformers import AutoTokenizer


def _write_u32(f, val):
    f.write(struct.pack("<I", val))


class TokenizerWrapper:
    def __init__(self, tokenizer_id: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
        # Qwen usually doesn't have a default BOS, but has EOS.
        self.eos_token_id = self.tokenizer.eos_token_id
        if self.eos_token_id is None:
            # Fallback for Qwen if not set? Qwen2.5 usually has 151643
            if "Qwen" in tokenizer_id:
                 self.eos_token_id = 151643 # <|endoftext|>
            else:
                 # Try to guess or fail
                 if hasattr(self.tokenizer, "eod_id"):
                     self.eos_token_id = self.tokenizer.eod_id
                 else:
                     raise ValueError("Tokenizer has no EOS token ID")
        
        self.vocab_size = self.tokenizer.vocab_size

    def encode(self, text: str) -> List[int]:
        # We want clean tokenization.
        # For general text, we usually want to append EOS to separate documents.
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        ids.append(self.eos_token_id)
        return ids


def worker_init(tokenizer_id):
    global _TOKENIZER
    _TOKENIZER = TokenizerWrapper(tokenizer_id)


def worker_process(text):
    return _TOKENIZER.encode(text)


def main():
    parser = argparse.ArgumentParser(description="Ingest HF datasets into fastgram tokenized.0 format")
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset name")
    parser.add_argument("--subset", default=None, help="Dataset subset")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--column", default="text", help="Text column name")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-72B-Instruct", help="Tokenizer ID")
    parser.add_argument("--proc", type=int, default=os.cpu_count(), help="Number of processes")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of docs")
    parser.add_argument("--buffer-size", type=int, default=1000, help="Write buffer size (number of docs)")
    parser.add_argument("--shard-size", type=int, default=2_000_000_000, help="Tokens per shard")
    
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading tokenizer: {args.tokenizer}")
    # Initialize tokenizer in main process to check it works
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size} -> using u32")
    
    print(f"Loading dataset: {args.dataset}")
    ds = datasets.load_dataset(args.dataset, args.subset, split=args.split, streaming=True)
    
    if args.limit:
        ds = ds.take(args.limit)
    
    # We use a pool of workers to tokenize
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(args.proc, initializer=worker_init, initargs=(args.tokenizer,))
    
    print(f"Writing to {out_dir} (shard size: {args.shard_size})...")
    
    start_time = time.time()
    total_tokens = 0
    total_docs = 0
    shard_idx = 0
    current_shard_tokens = 0
    
    # Open first shard
    current_f = open(out_dir / f"tokenized.{shard_idx}", "wb")
    
    try:
        # We process in batches to keep the pool fed
        batch_docs = []
        
        def flush_batch():
            nonlocal total_tokens, total_docs, current_shard_tokens, shard_idx, current_f
            if not batch_docs:
                return
            
            # Parallel tokenize
            results = pool.map(worker_process, batch_docs)
            
            # Write to disk
            for ids in results:
                # Check if we need to rotate shard
                if current_shard_tokens + len(ids) > args.shard_size:
                    current_f.close()
                    shard_idx += 1
                    current_f = open(out_dir / f"tokenized.{shard_idx}", "wb")
                    current_shard_tokens = 0

                # Write u32 little endian
                for token in ids:
                    current_f.write(struct.pack("<I", token))
                
                current_shard_tokens += len(ids)
                total_tokens += len(ids)
            
            total_docs += len(results)
            batch_docs.clear()
            
            elapsed = time.time() - start_time
            print(f"\rDocs: {total_docs} | Tokens: {total_tokens} | Shard: {shard_idx} | Rate: {total_tokens/max(1, elapsed):.2f} tok/s", end="")

        for i, item in enumerate(ds):
            text = item.get(args.column, "")
            if not text:
                continue
            
            batch_docs.append(text)
            
            if len(batch_docs) >= args.buffer_size:
                flush_batch()
                
        flush_batch() # Flush remaining

    finally:
        if not current_f.closed:
            current_f.close()
        pool.close()
        pool.join()
    
    print(f"\nDone. Total tokens: {total_tokens}")
    print(f"Saved to {out_dir}")

if __name__ == "__main__":
    main()