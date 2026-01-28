import json
from functools import lru_cache
from importlib.resources import files
from typing import Dict, List, Tuple


@lru_cache(maxsize=1)
def load_official_s3_indices() -> Dict[str, str]:
    p = files(__package__).joinpath("index_catalog.json")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("index catalog must be a JSON object")
    out: Dict[str, str] = {}
    for k, v in data.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise TypeError("index catalog entries must be str -> str")
        out[k] = v
    return out


def list_official_indices() -> List[Tuple[str, str]]:
    d = load_official_s3_indices()
    return sorted(d.items())

