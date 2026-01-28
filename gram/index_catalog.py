import json
from functools import lru_cache
from importlib.resources import files
from typing import Any, Dict, List, Tuple


@lru_cache(maxsize=1)
def load_official_catalog() -> Dict[str, Dict[str, Any]]:
    p = files(__package__).joinpath("index_catalog.json")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("index catalog must be a JSON object")
    out: Dict[str, Dict[str, Any]] = {}
    for name, entry in data.items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise TypeError("index catalog entries must be str -> object")
        s3_url = entry.get("s3_url")
        if not isinstance(s3_url, str):
            raise TypeError("index catalog entry missing s3_url")
        tok = entry.get("tokenizer")
        if tok is not None and not isinstance(tok, dict):
            raise TypeError("index catalog tokenizer must be object")
        out[name] = entry
    return out


def load_official_s3_indices() -> Dict[str, str]:
    catalog = load_official_catalog()
    return {name: entry["s3_url"] for name, entry in catalog.items()}


def get_official_index(name: str) -> Dict[str, Any] | None:
    catalog = load_official_catalog()
    return catalog.get(name)


def list_official_indices() -> List[Tuple[str, str]]:
    d = load_official_s3_indices()
    return sorted(d.items())
