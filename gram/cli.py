import argparse
import os
import shlex
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


def _interactive_download(aws: str, base: Path) -> bool:
    indices = list_official_indices()
    if not indices:
        print("no indices in catalog", file=sys.stderr)
        return False

    flt = ""
    while True:
        print()
        print("Download:")
        shown: List[Tuple[str, str]] = []
        for name, url in indices:
            if flt and flt not in name:
                continue
            shown.append((name, url))
        if not shown:
            print("(no matches)")
        else:
            for i, (name, url) in enumerate(shown, 1):
                print(f"{i}\t{name}\t{url}")

        s = input("select number, filter text, b, or q: ").strip()
        if s in ("q", "quit", "exit"):
            return True
        if s in ("b", "back"):
            return False
        if not s:
            flt = ""
            continue
        if s.isdigit():
            ix = int(s) - 1
            if ix < 0 or ix >= len(shown):
                print("invalid selection", file=sys.stderr)
                continue
            name, url = shown[ix]

            default_dest = base / name
            dest_in = input(f"dest [{default_dest}]: ").strip()
            dest = Path(dest_in) if dest_in else default_dest

            delete = input("delete extra local files? [y/N]: ").strip().lower() in ("y", "yes")
            quiet = input("quiet? [y/N]: ").strip().lower() in ("y", "yes")
            extra = input("extra aws args (optional): ").strip()
            extra_args = shlex.split(extra) if extra else []

            try:
                _run_aws_sync(
                    aws=aws,
                    s3_url=url,
                    dest=dest,
                    delete=delete,
                    quiet=quiet,
                    extra_args=extra_args,
                    dry_run=True,
                )
            except FileNotFoundError as e:
                print(f"missing dependency: {e}", file=sys.stderr)
                continue

            run = input("run download? [y/N]: ").strip().lower() in ("y", "yes")
            if not run:
                continue
            try:
                _run_aws_sync(
                    aws=aws,
                    s3_url=url,
                    dest=dest,
                    delete=delete,
                    quiet=quiet,
                    extra_args=extra_args,
                    dry_run=False,
                )
            except FileNotFoundError as e:
                print(f"missing dependency: {e}", file=sys.stderr)
                continue
            except subprocess.CalledProcessError as e:
                print(f"download failed: {e.returncode}", file=sys.stderr)
                continue
            continue

        flt = s


def _interactive_main() -> int:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return _die("no command provided; run `gram list` or `gram download ...`")

    aws = os.environ.get("GRAM_AWS", "aws")
    base = Path(os.environ.get("GRAM_INDEX_DIR", "index"))

    while True:
        print()
        print("gram:")
        print("1\tdownload")
        print("q\tquit")
        s = input("select: ").strip()
        if s in ("q", "quit", "exit"):
            return 0
        if s in ("1", "d", "download"):
            if _interactive_download(aws=aws, base=base):
                return 0
            continue
        print("unknown command", file=sys.stderr)


def main(argv: List[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        return _interactive_main()

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
