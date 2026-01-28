import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
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


def run(cmd: list[str]) -> dict:
    start = time.time()
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"error: command failed: {' '.join(cmd)}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        raise
    wall = time.time() - start
    data = {"wall_s": wall}
    for line in out.splitlines():
        if "\t" not in line:
            continue
        k, v = line.split("\t", 1)
        try:
            data[k] = int(v)
        except ValueError:
            try:
                data[k] = float(v)
            except ValueError:
                data[k] = v
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="bench/build_bench_config.json")
    parser.add_argument("--build-dir", help="build directory (default: auto-detect)")
    args = parser.parse_args()

    build_dir = Path(args.build_dir) if args.build_dir else _get_build_dir()
    if not (build_dir / "tg_build_index").exists():
        print(f"error: tg_build_index not found in {build_dir}", file=sys.stderr)
        return 2

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"error: config file not found: {cfg_path}", file=sys.stderr)
        return 2

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    out_root = Path(cfg["out_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    results = []
    for inp in cfg["inputs"]:
        name = inp["name"]
        in_dir = inp["dir"]
        for mode in cfg["modes"]:
            for i in range(cfg["runs"]):
                out_dir = out_root / f"{name}_{mode}_{i}"
                if out_dir.exists():
                    if not out_dir.is_relative_to(out_root):
                        print(f"error: refusing to delete {out_dir} (not under {out_root})", file=sys.stderr)
                        return 2
                    shutil.rmtree(out_dir)
                cmd = [
                    str(build_dir / "tg_build_index"),
                    in_dir,
                    str(out_dir),
                    str(cfg["token_width"]),
                    str(cfg["version"]),
                    mode,
                    str(cfg["ram_cap_bytes"]),
                ]
                data = run(cmd)
                data["input"] = name
                data["mode"] = mode
                data["run"] = i
                if cfg.get("verify"):
                    vcmd = [
                        "python",
                        "tools/verify_built_index.py",
                        "--source-dir",
                        in_dir,
                        "--built-dir",
                        str(out_dir),
                        "--dtype",
                        cfg["dtype"],
                        "--eos",
                        str(cfg["eos"]),
                        "--vocab",
                        str(cfg["vocab"]),
                        "--version",
                        str(cfg["version"]),
                        "--queries",
                        cfg["verify_queries"],
                        "--max-queries",
                        str(cfg.get("verify_max_queries", 200)),
                    ]
                    ref_dir = inp.get("ref_dir", "")
                    if ref_dir:
                        vcmd += ["--ref-dir", ref_dir]
                    subprocess.check_call(vcmd)
                results.append(data)
                print(json.dumps(data))

    out_json = out_root / "build_bench_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out_json}")
    summary = {}
    by_key = {}
    for r in results:
        key = (r["input"], r["mode"])
        by_key.setdefault(key, []).append(r)
    for (inp, mode), items in by_key.items():
        vals = [x.get("total_ns", x["wall_s"] * 1e9) for x in items]
        vals_sorted = sorted(vals)
        def pct(p: float) -> float:
            return vals_sorted[int(p * (len(vals_sorted) - 1))]
        mean_ns = statistics.mean(vals)
        p50_ns = pct(0.50)
        p95_ns = pct(0.95)
        p99_ns = pct(0.99)
        summary[f"{inp}:{mode}"] = {
            "runs": len(vals),
            "mean_ns": mean_ns,
            "p50_ns": p50_ns,
            "p95_ns": p95_ns,
            "p99_ns": p99_ns,
            "mean_s": mean_ns / 1e9,
            "p50_s": p50_ns / 1e9,
            "p95_s": p95_ns / 1e9,
            "p99_s": p99_ns / 1e9,
        }
    out_sum = out_root / "build_bench_summary.json"
    out_sum.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out_sum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
