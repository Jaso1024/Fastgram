#!/usr/bin/env python
"""Sample text from the gram index to see what patterns it contains."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from transformers import AutoTokenizer
from fastgram import GramEngine


def main():
    index_dir = "index/reasoning"
    # Use Qwen tokenizer since the index was built with it
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)

    eos = tok.eos_token_id
    vocab = tok.vocab_size

    if vocab <= 2**16:
        token_dtype = "u16"
    else:
        token_dtype = "u32"

    print(f"Loading gram index: {index_dir}")
    print(f"EOS token id: {eos}")
    print(f"Vocab size: {vocab}")

    engine = GramEngine(
        index_dir=index_dir,
        eos_token_id=eos,
        vocab_size=vocab,
        version=4,
        token_dtype=token_dtype,
        threads=0,
    )

    # Try different starting sequences and see what the index suggests
    test_prefixes = [
        "Let me",
        "First,",
        "The answer",
        "I need to",
        "15 + 27",
        "<think>",
    ]

    for prefix in test_prefixes:
        print(f"\n{'='*60}")
        print(f"Prefix: {repr(prefix)}")
        prefix_ids = tok.encode(prefix, add_special_tokens=False)
        print(f"Token IDs: {prefix_ids}")

        # Get next token distribution
        ntd = engine.ntd(prompt_ids=prefix_ids, max_support=100)

        if "error" in ntd:
            print(f"  Error: {ntd['error']}")
            continue

        print(f"  Prompt count in index: {ntd.get('prompt_cnt', 0)}")
        print(f"  Suffix len used: {ntd.get('suffix_len', 0)}")

        result = ntd.get("result_by_token_id", {})
        if result:
            # Sort by count
            sorted_items = sorted(result.items(), key=lambda x: -x[1]["cont_cnt"])[:10]
            print(f"  Top 10 continuations:")
            for tok_id, info in sorted_items:
                tok_str = tok.decode([int(tok_id)])
                print(f"    {repr(tok_str):20s} count={info['cont_cnt']:6d} prob={info['prob']:.4f}")
        else:
            print("  No continuations found in index")

    # Try with empty prefix to see what's most common
    print(f"\n{'='*60}")
    print("Empty prefix (unigram distribution):")
    ntd = engine.ntd(prompt_ids=[], max_support=100)
    if "error" not in ntd:
        print(f"  Total documents/sequences: {ntd.get('prompt_cnt', 0)}")
        result = ntd.get("result_by_token_id", {})
        if result:
            sorted_items = sorted(result.items(), key=lambda x: -x[1]["cont_cnt"])[:20]
            print(f"  Top 20 starting tokens:")
            for tok_id, info in sorted_items:
                tok_str = tok.decode([int(tok_id)])
                print(f"    {repr(tok_str):20s} count={info['cont_cnt']:6d}")


if __name__ == "__main__":
    main()
