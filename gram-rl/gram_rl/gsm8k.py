from __future__ import annotations

import re
from typing import Optional


_NUM_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def _extract_number(s: str) -> Optional[str]:
    if not s:
        return None
    if "####" in s:
        tail = s.rsplit("####", 1)[-1]
        m = _NUM_RE.search(tail)
        if m:
            return m.group(0)
    ms = list(_NUM_RE.finditer(s))
    if not ms:
        return None
    return ms[-1].group(0)


def _norm_num(s: str) -> str:
    return s.replace(",", "").strip()


def gsm8k_reward(*, completion_text: str, answer_text: str) -> float:
    pred = _extract_number(completion_text)
    gold = _extract_number(answer_text)
    if pred is None or gold is None:
        return 0.0
    return 1.0 if _norm_num(pred) == _norm_num(gold) else 0.0


def gsm8k_prompt(*, question: str) -> str:
    q = question.strip()
    return f"Q: {q}\nA:"

