import argparse
import multiprocessing as mp
import os
import struct
import time
from pathlib import Path
from typing import Dict, List, Optional
import io
import json

import datasets
from transformers import AutoTokenizer

# Optional imports for raw ingestion
try:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    import zstandard as zstd
except ImportError:
    pq = None
    HfFileSystem = None
    zstd = None


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
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        ids.append(self.eos_token_id)
        return ids


def worker_init(tokenizer_id):
    global _TOKENIZER
    _TOKENIZER = TokenizerWrapper(tokenizer_id)


def worker_process(text):
    return _TOKENIZER.encode(text)


def format_deepseek_chat(messages: List[Dict]) -> str:
    """Formats a list of messages into a single training string."""
    text = ""
    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')
        
        # DeepSeek R1 specific handling for 'info'/'think_content' if present in struct
        # Structure seen: {'content': '...', 'info': {'think_content': '...'}, 'role': '...'}
        info = msg.get('info')
        think = ""
        if isinstance(info, dict):
            think = info.get('think_content', '')
        
        if role == 'user':
            text += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == 'assistant':
            text += "<|im_start|>assistant\n"
            if think:
                text += f"<think>\n{think}\n</think>\n"
            text += f"{content}<|im_end|>\n"
        else:
            # System or other
            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    
    text += "<|endoftext|>"
    return text


def iter_raw_files(dataset: str, subset: Optional[str] = None, split: str = "train", limit: Optional[int] = None):
    """
    Iterates over a dataset by reading raw files (parquet, jsonl, jsonl.zst) via HfFileSystem.
    """
    if HfFileSystem is None:
        raise ImportError("huggingface_hub, pyarrow, zstandard are required for raw ingestion")
    
    fs = HfFileSystem()
    base_path = f"hf://datasets/{dataset}"
    
    # Heuristic patterns for finding data files
    patterns = []
    if subset:
        # e.g. am_0.9M.jsonl.zst at root matching subset name
        patterns.append(f"{base_path}/{subset}.jsonl.zst")
        patterns.append(f"{base_path}/{subset}.jsonl")
        patterns.append(f"{base_path}/{subset}/*.parquet")
        patterns.append(f"{base_path}/{subset}/data/*.parquet")
    else:
        patterns.append(f"{base_path}/*.jsonl.zst")
        patterns.append(f"{base_path}/*.jsonl")
        patterns.append(f"{base_path}/*.parquet")
        patterns.append(f"{base_path}/data/*.parquet")
        
    files = []
    for p in patterns:
        try:
            found = fs.glob(p)
            if found:
                files.extend(found)
        except Exception:
            pass
            
    if not files:
        print(f"Warning: No data files found for {dataset} with patterns {patterns}")
        return

    # Deduplicate
    files = sorted(list(set(files)))
    print(f"Found {len(files)} files for raw ingestion: {files}")
    
    count = 0
    for file_path in files:
        print(f"Processing raw file: {file_path}")
        file_type = "parquet" if file_path.endswith(".parquet") else "jsonl"
        is_zst = file_path.endswith(".zst")
        
        with fs.open(file_path, "rb") as f:
            try:
                if file_type == "parquet":
                    pq_file = pq.ParquetFile(f)
                    for batch in pq_file.iter_batches():
                        pylist = batch.to_pylist()
                        for row in pylist:
                            yield row
                            count += 1
                            if limit and count >= limit:
                                return
                elif file_type == "jsonl":
                    stream = f
                    if is_zst:
                        dctx = zstd.ZstdDecompressor()
                        stream = dctx.stream_reader(f)
                        stream = io.TextIOWrapper(stream, encoding='utf-8')
                    else:
                        stream = io.TextIOWrapper(f, encoding='utf-8')
                        
                    for line in stream:
                        if not line.strip(): continue
                        try:
                            row = json.loads(line)
                            yield row
                            count += 1
                            if limit and count >= limit:
                                return
                        except json.JSONDecodeError:
                            pass
                            
            except Exception as e:
                print(f"Error reading {file_path}: {e}. Skipping file.")


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
    parser.add_argument("--force-raw", action="store_true", help="Force raw ingestion")
    
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size} -> using u32")
    
    # Determine ingestion method
    iterable_ds = None
    
    # Heuristic: DeepSeek datasets often have schema issues with 'datasets' lib
    is_deepseek = "DeepSeek" in args.dataset
    
    if args.force_raw or is_deepseek:
        print(f"Using RAW FILE ingestion for {args.dataset}")
        iterable_ds = iter_raw_files(args.dataset, args.subset, args.split, args.limit)
    else:
        print(f"Using STANDARD ingestion for {args.dataset}")
        try:
            iterable_ds = datasets.load_dataset(
                args.dataset, args.subset, split=args.split, streaming=True, verification_mode="no_checks"
            )
            if args.limit:
                iterable_ds = iterable_ds.take(args.limit)
        except Exception as e:
            print(f"Standard loading failed ({e}), falling back to RAW ingestion.")
            iterable_ds = iter_raw_files(args.dataset, args.subset, args.split, args.limit)

    # Workers
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(args.proc, initializer=worker_init, initargs=(args.tokenizer,))
    
    print(f"Writing to {out_dir} (shard size: {args.shard_size})...")
    
    start_time = time.time()
    total_tokens = 0
    total_docs = 0
    shard_idx = 0
    current_shard_tokens = 0
    current_f = open(out_dir / f"tokenized.{shard_idx}", "wb")
    
    try:
        batch_docs = []
        
        def flush_batch():
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

                for token in ids:
                    current_f.write(struct.pack("<I", token))
                
                current_shard_tokens += len(ids)
                total_tokens += len(ids)
            
            total_docs += len(results)
            batch_docs.clear()
            elapsed = time.time() - start_time
            print(f"\rDocs: {total_docs} | Tokens: {total_tokens} | Shard: {shard_idx} | Rate: {total_tokens/max(1, elapsed):.2f} tok/s", end="")

        for i, item in enumerate(iterable_ds):
            # Extraction logic
            text = ""
            if "messages" in item and isinstance(item["messages"], list):
                # Chat format
                text = format_deepseek_chat(item["messages"])
            else:
                # Standard text format
                text = item.get(args.column, "")
            
            if not text:
                continue
            
            batch_docs.append(text)
            if len(batch_docs) >= args.buffer_size:
                flush_batch()
                
        flush_batch()

    except Exception as e:
        print(f"\nCRITICAL ERROR during ingestion: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if not current_f.closed:
            current_f.close()
        pool.close()
        pool.join()
    
    print(f"\nDone. Total tokens: {total_tokens}")
    print(f"Saved to {out_dir}")

if __name__ == "__main__":
    main()