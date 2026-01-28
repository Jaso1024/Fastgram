import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

from .index_catalog import list_official_indices, load_official_s3_indices


def _die(msg: str) -> int:
    print(msg, file=sys.stderr)
    return 2


def _resolve_index_spec(name_or_url: str) -> Tuple[str, str]:
    if name_or_url.startswith("s3://"):
        url = name_or_url
        if not url.startswith("s3://infini-gram-lite/"):
            raise KeyError(name_or_url)
        name = url.rstrip("/").split("/")[-1]
        return name, url
    d = load_official_s3_indices()
    if name_or_url in d:
        return name_or_url, d[name_or_url]
    raise KeyError(name_or_url)


def _aws_bin(path: str) -> str:
    if os.path.sep in path:
        return path
    resolved = shutil.which(path)
    if not resolved:
        raise FileNotFoundError(path)
    return resolved


def _run_aws_sync(
    aws: str,
    s3_url: str,
    dest: Path,
    delete: bool,
    quiet: bool,
    extra_args: List[str],
    dry_run: bool,
) -> int:
    cmd = [_aws_bin(aws), "s3", "sync", s3_url, str(dest)]
    cmd.append("--no-sign-request")
    if delete:
        cmd.append("--delete")
    if quiet:
        cmd.append("--only-show-errors")
    cmd.extend(extra_args)
    if dry_run:
        print(" ".join(cmd))
        return 0
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)
    return 0


def cmd_list(_args: argparse.Namespace) -> int:
    for name, url in list_official_indices():
        print(f"{name}\t{url}")
    return 0


def cmd_download(args: argparse.Namespace, extra_aws_args: List[str]) -> int:
    try:
        name, url = _resolve_index_spec(args.index)
    except KeyError:
        return _die("unknown index; run `gram list`")

    if args.to:
        dest = Path(args.to)
    else:
        dest = Path("index") / name

    return _run_aws_sync(
        aws=args.aws,
        s3_url=url,
        dest=dest,
        delete=args.delete,
        quiet=args.quiet,
        extra_args=extra_aws_args,
        dry_run=args.dry_run,
    )


def main(argv: List[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(prog="gram")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List official indices")
    p_list.set_defaults(_fn=cmd_list)

    p_dl = sub.add_parser("download", help="Download an index (name or s3://...)")
    p_dl.add_argument("index")
    p_dl.add_argument("--to", default="")
    p_dl.add_argument("--aws", default="aws")
    p_dl.add_argument("--delete", action="store_true")
    p_dl.add_argument("--quiet", action="store_true")
    p_dl.add_argument("--dry-run", action="store_true")
    p_dl.set_defaults(_fn=cmd_download)

    ns, extra = parser.parse_known_args(argv)
    try:
        return ns._fn(ns) if ns.cmd == "list" else ns._fn(ns, extra)
    except FileNotFoundError as e:
        return _die(f"missing dependency: {e}")
    except subprocess.CalledProcessError as e:
        return e.returncode


if __name__ == "__main__":
    raise SystemExit(main())
