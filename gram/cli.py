import argparse
import os
import random
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from .index_catalog import get_official_index, list_official_indices, load_official_s3_indices


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


def _print_settings(settings: dict) -> None:
    print(_bold("Settings"))
    print(_hr())
    print(f"{_cyan('topk'):>6}  {_dim(str(settings['topk']))}")
    print(f"{_cyan('max_support'):>6}  {_dim(str(settings['max_support']))}")
    print(f"{_cyan('threads'):>6}  {_dim(str(settings['threads']))}")
    print(f"{_cyan('steps'):>6}  {_dim(str(settings['steps']))}")
    print(f"{_cyan('temp'):>6}  {_dim(str(settings['temperature']))}")
    print(_hr())


def _apply_setting(settings: dict, key: str, value: str) -> Optional[str]:
    if key not in settings:
        return f"unknown setting: {key}"
    try:
        if key == "temperature":
            fv = float(value)
            if fv <= 0:
                return "value must be > 0"
            settings[key] = fv
            return None
        iv = int(value)
    except ValueError:
        return "invalid value"
    if key in ("steps", "max_support") and iv <= 0:
        return "value must be > 0"
    if key in ("topk", "threads") and iv < 0:
        return "value must be >= 0"
    settings[key] = iv
    return None


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


def _list_local_indices(base: Path) -> List[Path]:
    if not base.exists() or not base.is_dir():
        return []
    out: List[Path] = []
    for p in sorted(base.iterdir()):
        if not p.is_dir():
            continue
        out.append(p)
    return out


def _pick_local_index(base: Path) -> Optional[Path]:
    choices = _list_local_indices(base)
    if not choices:
        s = _prompt("index path: ").strip()
        return Path(s) if s else None
    while True:
        print(_bold("Local indices") + " " + _dim(f"(base: {base})"))
        print(_hr())
        num_w = len(str(len(choices)))
        for i, p in enumerate(choices, 1):
            n = str(i).rjust(num_w)
            print(f"{_dim(n)}  {_cyan(p.name)}  {_dim(str(p))}")
        print(_hr())
        print(_dim("number=select  p=path  b=back  q=quit"))
        s = _prompt("index> ").strip()
        if s in ("q", "quit", "exit"):
            return None
        if s in ("b", "back"):
            return None
        if s in ("p", "path"):
            s = _prompt("index path: ").strip()
            return Path(s) if s else None
        if s.isdigit():
            ix = int(s) - 1
            if 0 <= ix < len(choices):
                return choices[ix]
        print("invalid selection", file=sys.stderr)
        continue


def _load_tokenizer(cfg: dict) -> "AutoTokenizer":
    try:
        from transformers import AutoTokenizer
    except Exception as e:
        raise RuntimeError(f"missing dependency: transformers ({e})")

    hf_id = cfg.get("hf_id")
    if not isinstance(hf_id, str) or not hf_id:
        raise RuntimeError("tokenizer config missing hf_id")
    use_fast = bool(cfg.get("use_fast", False))
    add_bos_token = bool(cfg.get("add_bos_token", False))
    add_eos_token = bool(cfg.get("add_eos_token", False))
    token = os.environ.get("HF_TOKEN") if cfg.get("requires_hf_token") else None
    if cfg.get("requires_hf_token") and not token:
        raise RuntimeError("HF_TOKEN is required to load this tokenizer")
    try:
        if token is None:
            return AutoTokenizer.from_pretrained(
                hf_id,
                use_fast=use_fast,
                add_bos_token=add_bos_token,
                add_eos_token=add_eos_token,
            )
        return AutoTokenizer.from_pretrained(
            hf_id,
            use_fast=use_fast,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            token=token,
        )
    except TypeError:
        if token is None:
            return AutoTokenizer.from_pretrained(
                hf_id,
                use_fast=use_fast,
                add_bos_token=add_bos_token,
                add_eos_token=add_eos_token,
            )
        return AutoTokenizer.from_pretrained(
            hf_id,
            use_fast=use_fast,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            use_auth_token=token,
        )


def _tokenizer_info(tokenizer) -> Tuple[int, int]:
    eos = tokenizer.eos_token_id
    if eos is None:
        raise RuntimeError("tokenizer has no eos_token_id")
    vocab = getattr(tokenizer, "vocab_size", None)
    if vocab is None:
        vocab = len(tokenizer)
    return int(eos), int(vocab)


def _pick_token_dtype(vocab_size: int) -> List[str]:
    if vocab_size <= 2**8:
        return ["u8", "u16", "u32"]
    if vocab_size <= 2**16:
        return ["u16", "u32"]
    return ["u32"]


def _sample_token(dist: dict, topk: int, temperature: float) -> int:
    items = [(int(tok), float(info["prob"])) for tok, info in dist.items() if info["prob"] > 0]
    if not items:
        return 0
    items.sort(key=lambda x: x[1], reverse=True)
    if topk > 0:
        items = items[:topk]
    if temperature <= 0:
        return items[0][0]
    weights = [p ** (1.0 / temperature) for _, p in items]
    total = sum(weights)
    if total <= 0:
        return items[0][0]
    r = random.random() * total
    acc = 0.0
    for (tok, _), w in zip(items, weights):
        acc += w
        if r <= acc:
            return tok
    return items[-1][0]


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
    dry_run: bool,
) -> int:
    cmd = [_aws_bin(aws), "s3", "sync", s3_url, str(dest)]
    cmd.append("--no-sign-request")
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


def cmd_download(args: argparse.Namespace) -> int:
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
        dry_run=args.dry_run,
    )


def _make_engine(index_dir: str, tokenizer, version: int, max_support: int, threads: int):
    from .engine import GramEngine

    eos, vocab = _tokenizer_info(tokenizer)
    last_err: Optional[Exception] = None
    for dtype in _pick_token_dtype(vocab):
        try:
            engine = GramEngine(
                index_dir=index_dir,
                eos_token_id=eos,
                vocab_size=vocab,
                version=version,
                token_dtype=dtype,
                threads=threads,
                max_support=max_support,
            )
            return engine, dtype, eos, vocab
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"failed to load index with inferred dtype ({last_err})")


def _run_once(
    index_dir: str,
    tokenizer,
    version: int,
    prompt_text: str,
    max_support: int,
    topk: int,
    threads: int,
    steps: int,
    temperature: float,
    sample: bool,
) -> int:
    engine, dtype, eos, vocab = _make_engine(
        index_dir=index_dir,
        tokenizer=tokenizer,
        version=version,
        max_support=max_support,
        threads=threads,
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    if steps > 1 or sample:
        gen_ids: List[int] = []
        cur_ids = list(prompt_ids)
        for _ in range(max(1, steps)):
            r = engine.ntd(cur_ids, max_support=max_support)
            if "error" in r:
                print(r["error"], file=sys.stderr)
                return 2
            tok = _sample_token(r["result_by_token_id"], topk=topk, temperature=temperature)
            cur_ids.append(tok)
            gen_ids.append(tok)
        text = tokenizer.decode(gen_ids, clean_up_tokenization_spaces=False)
        text = text.replace("\n", "\\n").replace("\t", "\\t")
        print(_bold("Generated"))
        print(_dim(f"dtype={dtype} eos={eos} vocab={vocab}"))
        print(_hr())
        print(text)
        print(_hr())
        return 0
    r = engine.ntd(prompt_ids, max_support=max_support)
    if "error" in r:
        print(r["error"], file=sys.stderr)
        return 2
    dist = r["result_by_token_id"]
    rows = []
    for tok, info in dist.items():
        text = tokenizer.decode([tok], clean_up_tokenization_spaces=False)
        text = text.replace("\n", "\\n").replace("\t", "\\t")
        rows.append((tok, info["prob"], info["cont_cnt"], text))
    rows.sort(key=lambda x: x[1], reverse=True)
    if topk > 0:
        rows = rows[:topk]
    if sys.stdout.isatty():
        print(_bold("Next tokens"))
        print(_dim(f"dtype={dtype} eos={eos} vocab={vocab}"))
        print(_dim(f"prompt_cnt={r['prompt_cnt']}  approx={r['approx']}  total={len(dist)}"))
        print(_hr())
        tok_w = max((len(str(t)) for t, _, _, _ in rows), default=1)
        for tok, prob, cnt, text in rows:
            print(f"{_cyan(str(tok).rjust(tok_w))}  {_dim(f'{prob:.6g}')}  {_dim(str(cnt))}  {text}")
        print(_hr())
    else:
        for tok, prob, cnt, text in rows:
            print(f"{tok}\t{prob}\t{cnt}\t{text}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    if not args.prompt:
        return _die("prompt text is required")
    if args.topk < 0:
        return _die("topk must be >= 0")
    if args.max_support <= 0:
        return _die("max_support must be > 0")
    if args.steps <= 0:
        return _die("steps must be > 0")
    if args.temperature <= 0:
        return _die("temperature must be > 0")
    tokenizer_cfg = None
    if args.tokenizer:
        tokenizer_cfg = {"hf_id": args.tokenizer, "use_fast": False, "add_bos_token": False, "add_eos_token": False}
    else:
        entry = get_official_index(Path(args.index).name)
        if entry:
            tokenizer_cfg = entry.get("tokenizer")
    if not tokenizer_cfg:
        return _die("tokenizer not found; pass --tokenizer <hf-id>")
    try:
        tokenizer = _load_tokenizer(tokenizer_cfg)
    except Exception as e:
        return _die(str(e))
    version = int(args.version)
    return _run_once(
        index_dir=args.index,
        tokenizer=tokenizer,
        version=version,
        prompt_text=args.prompt,
        max_support=args.max_support,
        topk=args.topk,
        threads=args.threads,
        steps=args.steps,
        temperature=args.temperature,
        sample=args.sample,
    )


def _interactive_download(aws: str, base: Path) -> bool:
    indices = list_official_indices()
    if not indices:
        print("no indices in catalog", file=sys.stderr)
        return False

    flt = ""
    while True:
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
                continue
            name, url = shown[ix]
            print(_bold("Selected") + ": " + _cyan(name))
            print(_dim(url))
            print(_hr())
            dest = base / name
            print(_dim(f"Destination: {dest}"))
            print(_dim("Change with GRAM_INDEX_DIR or `gram download --to ...`"))

            try:
                _run_aws_sync(
                    aws=aws,
                    s3_url=url,
                    dest=dest,
                    dry_run=True,
                )
            except FileNotFoundError as e:
                print(f"missing dependency: {e}", file=sys.stderr)
                continue
            run = _yn("run download?", default=False)
            if not run:
                continue
            try:
                _run_aws_sync(
                    aws=aws,
                    s3_url=url,
                    dest=dest,
                    dry_run=False,
                )
            except FileNotFoundError as e:
                print(f"missing dependency: {e}", file=sys.stderr)
                continue
            except subprocess.CalledProcessError as e:
                print(f"download failed: {e.returncode}", file=sys.stderr)
                continue
            print("done")
            continue

        flt = s


def _interactive_run(base: Path) -> bool:
    index_path = _pick_local_index(base)
    if not index_path:
        return False

    entry = get_official_index(index_path.name)
    tokenizer_cfg = entry.get("tokenizer") if entry else None
    if not tokenizer_cfg:
        s = _prompt("tokenizer hf id: ").strip()
        if not s:
            return False
        tokenizer_cfg = {"hf_id": s, "use_fast": False, "add_bos_token": False, "add_eos_token": False}
    try:
        tokenizer = _load_tokenizer(tokenizer_cfg)
    except Exception as e:
        print(str(e), file=sys.stderr)
        return False

    version = int(entry.get("version", 4)) if entry else 4
    settings = {"topk": 20, "max_support": 1000, "threads": 0, "steps": 1, "temperature": 1.0}

    while True:
        print(_bold("Run") + " " + _dim(index_path.name))
        print(_dim(str(index_path)))
        print(_dim(f"tokenizer={tokenizer.name_or_path}  version={version}"))
        print(_dim("enter text, /gen N <text>, /set key value, /settings, b=back, q=quit"))
        print(_hr())
        s = _prompt("prompt> ").strip()
        if s in ("q", "quit", "exit"):
            return True
        if s in ("b", "back"):
            return False
        if s in ("/settings", "settings"):
            _print_settings(settings)
            continue
        if s.startswith("/set "):
            parts = s.split(maxsplit=2)
            if len(parts) < 3:
                print("usage: /set <key> <value>", file=sys.stderr)
                continue
            err = _apply_setting(settings, parts[1], parts[2])
            if err:
                print(err, file=sys.stderr)
            else:
                _print_settings(settings)
            continue
        gen_steps: Optional[int] = None
        if s.startswith("/gen "):
            parts = s.split(maxsplit=2)
            if len(parts) < 2 or not parts[1].isdigit():
                print("usage: /gen <n> <text>", file=sys.stderr)
                continue
            gen_steps = int(parts[1])
            if len(parts) == 2:
                s = _prompt("text> ").strip()
            else:
                s = parts[2]
        _run_once(
            index_dir=str(index_path),
            tokenizer=tokenizer,
            version=version,
            prompt_text=s,
            max_support=settings["max_support"],
            topk=settings["topk"],
            threads=settings["threads"],
            steps=gen_steps or settings["steps"],
            temperature=settings["temperature"],
            sample=True if gen_steps is not None else False,
        )
        continue


def _interactive_main() -> int:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return _die("no command provided; run `gram list` or `gram download ...`")

    aws = os.environ.get("GRAM_AWS", "aws")
    base = Path(os.environ.get("GRAM_INDEX_DIR", "index"))

    while True:
        print(_bold("gram"))
        print(_hr())
        print("1  " + _cyan("run") + "  " + _dim("next-token distribution from a local index"))
        print("2  " + _cyan("download") + "  " + _dim("download an official free index"))
        print("q  " + _dim("quit"))
        print(_hr())
        s = _prompt("gram> ").strip()
        if s in ("q", "quit", "exit"):
            return 0
        if s in ("1", "r", "run"):
            if _interactive_run(base=base):
                return 0
            continue
        if s in ("2", "d", "download"):
            if _interactive_download(aws=aws, base=base):
                return 0
            continue
        print("unknown command", file=sys.stderr)
        continue


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
    p_dl.add_argument("--dry-run", action="store_true")
    p_dl.set_defaults(_fn=cmd_download)

    p_run = sub.add_parser("run", help="Next-token distribution from a local index")
    p_run.add_argument("--index", required=True)
    p_run.add_argument("--tokenizer", default="")
    p_run.add_argument("--version", type=int, default=4)
    p_run.add_argument("--max-support", type=int, default=1000)
    p_run.add_argument("--topk", type=int, default=20)
    p_run.add_argument("--threads", type=int, default=0)
    p_run.add_argument("--steps", type=int, default=1)
    p_run.add_argument("--temperature", type=float, default=1.0)
    p_run.add_argument("--sample", action="store_true")
    p_run.add_argument("--prompt", default="")
    p_run.set_defaults(_fn=cmd_run)

    ns = parser.parse_args(argv)
    try:
        return ns._fn(ns)
    except FileNotFoundError as e:
        return _die(f"missing dependency: {e}")
    except subprocess.CalledProcessError as e:
        return e.returncode


if __name__ == "__main__":
    raise SystemExit(main())
