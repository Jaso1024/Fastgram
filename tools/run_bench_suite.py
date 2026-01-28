import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


def _get_build_dir() -> Path:
    if build_env := os.environ.get("GRAM_BUILD_DIR"):
        return Path(build_env)
    for candidate in ["build", "cmake-build-release", "cmake-build-debug", "out"]:
        p = Path(candidate)
        if p.exists() and p.is_dir():
            return p
    print("error: build directory not found, set GRAM_BUILD_DIR", file=sys.stderr)
    sys.exit(2)


def _run_one(cmd: list[str]) -> float:
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"error: command failed: {' '.join(cmd)}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        raise
    for line in out.splitlines():
        if line.startswith("per_call_ns\t"):
            return float(line.split("\t", 1)[1])
    raise RuntimeError("per_call_ns not found")


def _summarize(vals: list[float]) -> dict:
    vals_sorted = sorted(vals)
    def pct(p: float) -> float:
        return vals_sorted[int(p * (len(vals_sorted) - 1))]
    return {
        "mean_ns": statistics.mean(vals),
        "p50_ns": pct(0.50),
        "p95_ns": pct(0.95),
        "p99_ns": pct(0.99),
    }


def _run(cfg: dict, build_dir: Path, bucket: str | None, length: int | None) -> list[float]:
    cmd = [
        str(build_dir / "tg_bench_suite"),
        cfg["index_dir"],
        cfg["dtype"],
        str(cfg["eos"]),
        str(cfg["vocab"]),
        str(cfg["version"]),
        str(cfg["max_support"]),
        cfg["query_file"],
        cfg["op"],
        str(cfg["iters"]),
    ]
    if bucket:
        cmd.append(f"--bucket={bucket}")
    if length:
        cmd.append(f"--length={length}")
    return [_run_one(cmd) for _ in range(cfg["runs"])]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="bench/bench_suite_config.json")
    parser.add_argument("--build-dir", help="build directory (default: auto-detect)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    build_dir = Path(args.build_dir) if args.build_dir else _get_build_dir()
    if not (build_dir / "tg_bench_suite").exists():
        print(f"error: tg_bench_suite not found in {build_dir}", file=sys.stderr)
        return 2

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"error: config file not found: {cfg_path}", file=sys.stderr)
        return 2

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    warmup = int(cfg["warmup_runs"])
    query_file = cfg["query_file"]

    buckets = []
    lengths = []
    with open(query_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            buckets.append(parts[0])
            lengths.append(int(parts[1]))
    uniq_buckets = sorted(set(buckets))
    uniq_lengths = sorted(set(lengths))

    for _ in range(warmup):
        _run_one(
            [
                str(build_dir / "tg_bench_suite"),
                cfg["index_dir"],
                cfg["dtype"],
                str(cfg["eos"]),
                str(cfg["vocab"]),
                str(cfg["version"]),
                str(cfg["max_support"]),
                cfg["query_file"],
                cfg["op"],
                str(cfg["iters"]),
            ]
        )

    vals = _run(cfg, build_dir, None, None)
    s = _summarize(vals)
    print("== suite ==")
    print(f"runs\t{cfg['runs']}")
    print(f"mean_ns\t{s['mean_ns']:.3f}")
    print(f"p50_ns\t{s['p50_ns']:.3f}")
    print(f"p95_ns\t{s['p95_ns']:.3f}")
    print(f"p99_ns\t{s['p99_ns']:.3f}")

    if args.verbose:
        for b in uniq_buckets:
            vals = _run(cfg, build_dir, b, None)
            s = _summarize(vals)
            print(f"== bucket {b} ==")
            print(f"runs\t{cfg['runs']}")
            print(f"mean_ns\t{s['mean_ns']:.3f}")
            print(f"p50_ns\t{s['p50_ns']:.3f}")
            print(f"p95_ns\t{s['p95_ns']:.3f}")
            print(f"p99_ns\t{s['p99_ns']:.3f}")
        for L in uniq_lengths:
            vals = _run(cfg, build_dir, None, L)
            s = _summarize(vals)
            print(f"== length {L} ==")
            print(f"runs\t{cfg['runs']}")
            print(f"mean_ns\t{s['mean_ns']:.3f}")
            print(f"p50_ns\t{s['p50_ns']:.3f}")
            print(f"p95_ns\t{s['p95_ns']:.3f}")
            print(f"p99_ns\t{s['p99_ns']:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
