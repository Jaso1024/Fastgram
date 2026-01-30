from __future__ import annotations

import re
from typing import Optional


_NUM_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def gsm8k_format_instruction(style: str) -> str:
    s = str(style)
    if s == "hash4":
        return "Give the final answer on its own line as: #### <number>"
    if s == "cot":
        return "Think through this step by step, showing your reasoning. Then give the final answer as: #### <number>"
    if s == "deepseek":
        return "Let me work through this problem step by step.\n\nFirst, I need to understand what the problem is asking."
    if s == "none":
        return ""
    raise ValueError("unknown format style")


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


def gsm8k_parse_answer(*, completion_text: str, style: str) -> tuple[Optional[str], bool]:
    s = str(style)
    if s in ("hash4", "cot", "deepseek"):
        # All these styles expect #### <number> format
        if "####" not in completion_text:
            return None, False
        tail = completion_text.rsplit("####", 1)[-1]
        m = _NUM_RE.search(tail)
        if not m:
            return None, False
        return _norm_num(m.group(0)), True
    if s == "none":
        pred = _extract_number(completion_text)
        return (_norm_num(pred) if pred is not None else None), True
    raise ValueError("unknown parse style")


def gsm8k_reward_components(*, completion_text: str, answer_text: str, style: str) -> tuple[float, float]:
    gold = _extract_number(answer_text)
    if gold is None:
        return 0.0, 0.0
    gold_n = _norm_num(gold)
    pred_n, fmt_ok = gsm8k_parse_answer(completion_text=completion_text, style=style)
    ans = 1.0 if (pred_n is not None and pred_n == gold_n) else 0.0
    fmt = 1.0 if bool(fmt_ok) else 0.0
    return ans, fmt


def gsm8k_reward(*, completion_text: str, answer_text: str) -> float:
    pred = _extract_number(completion_text)
    gold = _extract_number(answer_text)
    if pred is None or gold is None:
        return 0.0
    return 1.0 if _norm_num(pred) == _norm_num(gold) else 0.0


def gsm8k_prompt(*, question: str) -> str:
    q = question.strip()
    return f"Q: {q}\nA:"
