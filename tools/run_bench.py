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


def _bench_find(cfg: dict, build_dir: Path) -> list[float]:
    cmd = [
        str(build_dir / "tg_bench_find"),
        cfg["index_dir"],
        cfg["dtype"],
        str(cfg["eos"]),
        str(cfg["vocab"]),
        str(cfg["version"]),
        str(cfg["iters"]),
    ]
    return [_run_one(cmd) for _ in range(cfg["runs"])]


def _bench_ntd(cfg: dict, build_dir: Path) -> list[float]:
    cmd = [
        str(build_dir / "tg_bench_ntd"),
        cfg["index_dir"],
        cfg["dtype"],
        str(cfg["eos"]),
        str(cfg["vocab"]),
        str(cfg["version"]),
        str(cfg["max_support"]),
        str(cfg["iters"]),
    ]
    return [_run_one(cmd) for _ in range(cfg["runs"])]


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="bench/bench_config.json")
    parser.add_argument("--build-dir", help="build directory (default: auto-detect)")
    args = parser.parse_args()

    build_dir = Path(args.build_dir) if args.build_dir else _get_build_dir()
    if not (build_dir / "tg_bench_find").exists():
        print(f"error: tg_bench_find not found in {build_dir}", file=sys.stderr)
        return 2

    path = Path(args.config)
    if not path.exists():
        print(f"error: config file not found: {path}", file=sys.stderr)
        return 2

    data = json.loads(path.read_text(encoding="utf-8"))

    warmup = int(data["warmup_runs"])
    runs = int(data["runs"])
    iters = int(data["iters"])

    find_cfg = dict(data["find"])
    find_cfg.update({"warmup_runs": warmup, "runs": runs, "iters": iters})
    ntd_cfg = dict(data["ntd"])
    ntd_cfg.update({"warmup_runs": warmup, "runs": runs, "iters": iters})

    print("== find ==")
    for _ in range(warmup):
        _run_one(
            [
                str(build_dir / "tg_bench_find"),
                find_cfg["index_dir"],
                find_cfg["dtype"],
                str(find_cfg["eos"]),
                str(find_cfg["vocab"]),
                str(find_cfg["version"]),
                str(find_cfg["iters"]),
            ]
        )
    vals = _bench_find(find_cfg, build_dir)
    s = _summarize(vals)
    print(f"runs\t{runs}")
    print(f"mean_ns\t{s['mean_ns']:.3f}")
    print(f"p50_ns\t{s['p50_ns']:.3f}")
    print(f"p95_ns\t{s['p95_ns']:.3f}")
    print(f"p99_ns\t{s['p99_ns']:.3f}")

    print("== ntd ==")
    for _ in range(warmup):
        _run_one(
            [
                str(build_dir / "tg_bench_ntd"),
                ntd_cfg["index_dir"],
                ntd_cfg["dtype"],
                str(ntd_cfg["eos"]),
                str(ntd_cfg["vocab"]),
                str(ntd_cfg["version"]),
                str(ntd_cfg["max_support"]),
                str(ntd_cfg["iters"]),
            ]
        )
    vals = _bench_ntd(ntd_cfg, build_dir)
    s = _summarize(vals)
    print(f"runs\t{runs}")
    print(f"mean_ns\t{s['mean_ns']:.3f}")
    print(f"p50_ns\t{s['p50_ns']:.3f}")
    print(f"p95_ns\t{s['p95_ns']:.3f}")
    print(f"p99_ns\t{s['p99_ns']:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
