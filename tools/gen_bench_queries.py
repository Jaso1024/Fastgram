import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from fastgram import cpp_engine


def _bucket_for_count(cnt: int, buckets: List[dict]) -> str | None:
    for b in buckets:
        lo = b["min"]
        hi = b.get("max")
        if hi is None:
            if cnt >= lo:
                return b["name"]
        else:
            if lo <= cnt <= hi:
                return b["name"]
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--dtype", default="u16", choices=["u8", "u16", "u32"])
    parser.add_argument("--eos", type=int, required=True)
    parser.add_argument("--vocab", type=int, required=True)
    parser.add_argument("--version", type=int, default=4)
    parser.add_argument("--max-docs", type=int, default=2000)
    parser.add_argument("--max-doc-len", type=int, default=4000)
    parser.add_argument("--per-bucket", type=int, default=20)
    parser.add_argument("--out", default="bench/queries_v4_pileval_gpt2.txt")
    args = parser.parse_args()

    lengths = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]
    buckets = [
        {"name": "ultra_rare", "min": 1, "max": 1},
        {"name": "rare", "min": 2, "max": 5},
        {"name": "low", "min": 6, "max": 20},
        {"name": "mid", "min": 21, "max": 100},
        {"name": "high", "min": 101, "max": 1000},
        {"name": "very_high", "min": 1001},
    ]

    engine_cls = {"u8": cpp_engine.Engine_U8, "u16": cpp_engine.Engine_U16, "u32": cpp_engine.Engine_U32}[args.dtype]
    engine = engine_cls([args.index_dir], args.eos, args.vocab, args.version, 1, False, set(), 512)

    need: Dict[Tuple[int, str], int] = {}
    for L in lengths:
        for b in buckets:
            need[(L, b["name"])] = args.per_bucket

    seen = set()
    out_rows = []

    total_docs = engine.get_total_doc_cnt()
    max_docs = min(args.max_docs, total_docs)

    for doc_ix in range(max_docs):
        if doc_ix % 100 == 0:
            remaining = sum(1 for v in need.values() if v > 0)
            print(f"progress: {doc_ix}/{max_docs} docs, {len(out_rows)} queries, {remaining} buckets remaining", file=sys.stderr)
        doc = engine.get_doc_by_ix(doc_ix, args.max_doc_len)
        toks = doc.token_ids
        if not toks:
            continue
        for L in lengths:
            if all(need[(L, b["name"])] <= 0 for b in buckets):
                continue
            if len(toks) < L:
                continue
            for i in range(0, len(toks) - L + 1):
                q = toks[i : i + L]
                if args.eos in q:
                    continue
                key = (L, tuple(q))
                if key in seen:
                    continue
                seen.add(key)
                cnt = engine.count(q).count
                bucket = _bucket_for_count(cnt, buckets)
                if bucket is None:
                    continue
                if need[(L, bucket)] <= 0:
                    continue
                out_rows.append((bucket, L, cnt, q))
                need[(L, bucket)] -= 1
        if all(v <= 0 for v in need.values()):
            break
    print(f"finished: {len(out_rows)} queries generated", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        header = {
            "index_dir": args.index_dir,
            "dtype": args.dtype,
            "eos": args.eos,
            "vocab": args.vocab,
            "version": args.version,
            "lengths": lengths,
            "buckets": buckets,
            "per_bucket": args.per_bucket,
            "max_docs": max_docs,
            "max_doc_len": args.max_doc_len,
        }
        f.write("# " + json.dumps(header) + "\n")
        for bucket, L, cnt, q in out_rows:
            toks = " ".join(str(x) for x in q)
            f.write(f"{bucket} {L} {cnt} {toks}\n")

    remaining = {k: v for k, v in need.items() if v > 0}
    if remaining:
        print("incomplete buckets:")
        for (L, b), v in sorted(remaining.items()):
            print(f"  L={L} bucket={b} missing={v}")
    print(f"wrote {len(out_rows)} queries to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
