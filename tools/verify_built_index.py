import argparse
import shutil
import tempfile
from pathlib import Path

from fastgram import gram


def _load_queries(path: Path, max_queries: int) -> list[list[int]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            toks = [int(x) for x in parts[3:]]
            out.append(toks)
            if max_queries and len(out) >= max_queries:
                break
    return out


def _prepare_dir(source_dir: Path, built_dir: Path) -> tuple[Path, Path | None]:
    if (built_dir / "tokenized.0").exists():
        return built_dir, None
    tmp = Path(tempfile.mkdtemp(prefix="gram_build_verify_"))
    for name in ["tokenized.0", "offset.0", "metadata.0", "metaoff.0", "unigram.0"]:
        src = source_dir / name
        if src.exists():
            shutil.copyfile(src, tmp / name)
    for name in ["table.0"]:
        src = built_dir / name
        if not src.exists():
            raise RuntimeError(f"missing {name} in built dir")
        shutil.copyfile(src, tmp / name)
    return tmp, tmp


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--built-dir", required=True)
    parser.add_argument("--dtype", default="u16", choices=["u8", "u16", "u32"])
    parser.add_argument("--eos", type=int, required=True)
    parser.add_argument("--vocab", type=int, required=True)
    parser.add_argument("--version", type=int, default=4)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--max-queries", type=int, default=200)
    parser.add_argument("--ref-dir", default="")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    built_dir = Path(args.built_dir)
    queries = _load_queries(Path(args.queries), args.max_queries)
    if not queries:
        print("no queries to verify")
        return 2

    verify_dir, tmp = _prepare_dir(source_dir, built_dir)
    try:
        built = gram(str(verify_dir), args.eos, args.vocab, args.version, args.dtype)
        if args.ref_dir:
            ref = gram(args.ref_dir, args.eos, args.vocab, args.version, args.dtype)
        else:
            ref = gram(str(source_dir), args.eos, args.vocab, args.version, args.dtype)
        for idx, q in enumerate(queries):
            a = ref.count(q)
            b = built.count(q)
            if a.get("count") != b.get("count"):
                print(f"error: count mismatch at query {idx}")
                print(f"  query length: {len(q)}")
                print(f"  query (first 32): {q[:32]}")
                print(f"  ref count: {a.get('count')}")
                print(f"  built count: {b.get('count')}")
                return 2
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
