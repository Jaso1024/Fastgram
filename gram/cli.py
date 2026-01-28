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


def _use_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(s: str, code: str) -> str:
    if not _use_color():
        return s
    return f"\x1b[{code}m{s}\x1b[0m"


def _bold(s: str) -> str:
    return _c(s, "1")


def _dim(s: str) -> str:
    return _c(s, "2")


def _cyan(s: str) -> str:
    return _c(s, "36")


def _clear() -> None:
    if sys.stdout.isatty():
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.flush()


def _term_width() -> int:
    try:
        return shutil.get_terminal_size(fallback=(100, 20)).columns
    except Exception:
        return 100


def _hr() -> str:
    return _dim("─" * max(10, _term_width()))


def _prompt(s: str) -> str:
    return input(_bold(s))


def _yn(s: str, default: bool = False) -> bool:
    suf = " [Y/n]: " if default else " [y/N]: "
    v = _prompt(s + suf).strip().lower()
    if not v:
        return default
    return v in ("y", "yes")


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
        print(shlex.join(cmd))
        return 0
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)
    return 0


def cmd_list(_args: argparse.Namespace) -> int:
    rows = list_official_indices()
    if not sys.stdout.isatty():
        for name, url in rows:
            print(f"{name}\t{url}")
        return 0

    name_w = max((len(name) for name, _ in rows), default=4)
    _clear()
    print(_bold("gram") + " " + _dim("(official free indices)"))
    print(_hr())
    for name, url in rows:
        print(f"{_cyan(name.ljust(name_w))}  {_dim(url)}")
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
        _clear()
        print(_bold("Download indices") + " " + _dim("(infini-gram-lite, no-sign)"))
        print(_dim(f"AWS: {aws}   Base: {base}"))
        if flt:
            print(_dim(f"Filter: {flt}"))
        print(_hr())
        shown: List[Tuple[str, str]] = []
        for name, url in indices:
            if flt and flt.lower() not in name.lower():
                continue
            shown.append((name, url))
        if not shown:
            print(_dim("(no matches)"))
        else:
            name_w = max((len(n) for n, _ in shown), default=4)
            num_w = len(str(len(shown)))
            for i, (name, url) in enumerate(shown, 1):
                n = str(i).rjust(num_w)
                print(f"{_dim(n)}  {_cyan(name.ljust(name_w))}  {_dim(url)}")

        print(_hr())
        print(_dim("number=select  text=filter  enter=clear filter  b=back  q=quit"))
        s = _prompt("download> ").strip()
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
                _prompt("press enter: ")
                continue
            name, url = shown[ix]
            _clear()
            print(_bold("Selected") + ": " + _cyan(name))
            print(_dim(url))
            print(_hr())
            print(_dim("Press enter to accept defaults."))

            default_dest = base / name
            dest_in = _prompt(f"destination [{default_dest}]: ").strip()
            dest = Path(dest_in) if dest_in else default_dest

            delete = _yn("delete extra local files?", default=False)
            quiet = _yn("quiet?", default=False)
            extra = _prompt("extra aws args (optional): ").strip()
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
                _prompt("press enter: ")
                continue

            run = _yn("run download?", default=False)
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
                _prompt("press enter: ")
                continue
            except subprocess.CalledProcessError as e:
                print(f"download failed: {e.returncode}", file=sys.stderr)
                _prompt("press enter: ")
                continue
            _prompt("done; press enter: ")
            continue

        flt = s


def _interactive_main() -> int:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return _die("no command provided; run `gram list` or `gram download ...`")

    aws = os.environ.get("GRAM_AWS", "aws")
    base = Path(os.environ.get("GRAM_INDEX_DIR", "index"))

    while True:
        _clear()
        print(_bold("gram"))
        print(_hr())
        print("1  " + _cyan("download") + "  " + _dim("download an official free index"))
        print("q  " + _dim("quit"))
        print(_hr())
        s = _prompt("gram> ").strip()
        if s in ("q", "quit", "exit"):
            return 0
        if s in ("1", "d", "download"):
            if _interactive_download(aws=aws, base=base):
                return 0
            continue
        print("unknown command", file=sys.stderr)
        _prompt("press enter: ")


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
