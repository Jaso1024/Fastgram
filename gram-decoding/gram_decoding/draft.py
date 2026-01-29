from __future__ import annotations

import random
from typing import Literal, Optional


def _best_id_from_raw_map(by_id):
    best_id = None
    best_p = -1.0
    for tid, info in by_id.items():
        p = float(info.prob)
        if p > best_p:
            best_p = p
            best_id = int(tid)
    return best_id


def _sample_id_from_items(items: list[tuple[int, float]], *, topk: int, temperature: float, rng: random.Random) -> Optional[int]:
    if not items:
        return None
    items = [(int(t), float(p)) for t, p in items if p > 0]
    if not items:
        return None
    items.sort(key=lambda x: x[1], reverse=True)
    if topk > 0:
        items = items[:topk]
    if temperature <= 0:
        return items[0][0]
    weights = [p ** (1.0 / temperature) for _, p in items]
    total = sum(weights)
    if total <= 0:
        return items[0][0]
    r = rng.random() * total
    acc = 0.0
    for (tok, _), w in zip(items, weights):
        acc += w
        if r <= acc:
            return tok
    return items[-1][0]


def _best_id_from_dict(by_id: dict):
    best_id = None
    best_p = -1.0
    for tid, info in by_id.items():
        p = float(info.get("prob", 0.0))
        if p > best_p:
            best_p = p
            best_id = int(tid)
    return best_id


def _items_from_raw_map(by_id) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    for tid, info in by_id.items():
        out.append((int(tid), float(info.prob)))
    return out


def _items_from_dict(by_id: dict) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    for tid, info in by_id.items():
        out.append((int(tid), float(info.get("prob", 0.0))))
    return out


def draft_best_token_id(
    *,
    engine,
    prefix_ids: list[int],
    max_support: int,
    mode: Literal["infgram", "primitive"],
    raw_engine: bool,
) -> Optional[int]:
    if raw_engine:
        if mode == "infgram":
            res = engine.engine.ntd(prompt_ids=prefix_ids, max_support=max_support)
            return _best_id_from_raw_map(res.result_by_token_id)
        if mode == "primitive":
            res = engine.engine.primitive_ntd(prompt_ids=prefix_ids, max_support=max_support)
            return _best_id_from_raw_map(res.result_by_token_id)
        raise ValueError("unknown mode")

    if mode == "infgram":
        res = engine.ntd(prefix_ids, max_support=max_support)
    elif mode == "primitive":
        res = engine.primitive_ntd(prefix_ids, max_support=max_support)
    else:
        raise ValueError("unknown mode")

    by_id = res.get("result_by_token_id", {}) if isinstance(res, dict) else {}
    if not by_id:
        return None
    return _best_id_from_dict(by_id)


def draft_sample_token_id(
    *,
    engine,
    prefix_ids: list[int],
    max_support: int,
    mode: Literal["infgram", "primitive"],
    raw_engine: bool,
    topk: int,
    temperature: float,
    rng: random.Random,
) -> Optional[int]:
    if raw_engine:
        if mode == "infgram":
            res = engine.engine.ntd(prompt_ids=prefix_ids, max_support=max_support)
            return _sample_id_from_items(_items_from_raw_map(res.result_by_token_id), topk=topk, temperature=temperature, rng=rng)
        if mode == "primitive":
            res = engine.engine.primitive_ntd(prompt_ids=prefix_ids, max_support=max_support)
            return _sample_id_from_items(_items_from_raw_map(res.result_by_token_id), topk=topk, temperature=temperature, rng=rng)
        raise ValueError("unknown mode")

    if mode == "infgram":
        res = engine.ntd(prefix_ids, max_support=max_support)
    elif mode == "primitive":
        res = engine.primitive_ntd(prefix_ids, max_support=max_support)
    else:
        raise ValueError("unknown mode")

    by_id = res.get("result_by_token_id", {}) if isinstance(res, dict) else {}
    if not by_id:
        return None
    return _sample_id_from_items(_items_from_dict(by_id), topk=topk, temperature=temperature, rng=rng)
