"""GSM8K-specific reward functions."""
from __future__ import annotations

import re
from typing import Any, Optional

from .base import BaseRewardFunction, RewardOutput


_NUM_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def extract_number(s: str) -> Optional[str]:
    """Extract a number from text, preferring #### format."""
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


def normalize_number(s: str) -> str:
    """Normalize number string (remove commas, strip whitespace)."""
    return s.replace(",", "").strip()


class GSM8KCorrectnessReward(BaseRewardFunction):
    """Binary reward for correct answers on GSM8K.

    Extracts the final number from completion and compares to ground truth.
    """

    def compute(
        self,
        *,
        completion_text: str,
        prompt_text: str,
        completion_ids: list[int],
        prompt_ids: list[int],
        example: dict[str, Any],
        **kwargs: Any,
    ) -> RewardOutput:
        answer_text = example.get("answer", "")

        pred = extract_number(completion_text)
        gold = extract_number(answer_text)

        if pred is None or gold is None:
            correct = False
        else:
            correct = normalize_number(pred) == normalize_number(gold)

        reward = 1.0 if correct else 0.0

        return RewardOutput(
            total=reward,
            components={"correct": reward},
            metadata={
                "predicted": pred,
                "gold": gold,
                "correct": correct,
            },
        )


class GSM8KFormatReward(BaseRewardFunction):
    """Reward for following the expected answer format.

    Supports multiple format styles:
    - "hash4": Answer should contain "#### <number>"
    - "cot": Should have chain-of-thought then "#### <number>"
    - "deepseek": Should have step-by-step reasoning
    """

    def __init__(self, style: str = "hash4"):
        self.style = style

    def compute(
        self,
        *,
        completion_text: str,
        prompt_text: str,
        completion_ids: list[int],
        prompt_ids: list[int],
        example: dict[str, Any],
        **kwargs: Any,
    ) -> RewardOutput:
        style = self.style

        if style in ("hash4", "cot", "deepseek"):
            # All these styles expect #### <number> format
            if "####" not in completion_text:
                format_ok = False
            else:
                tail = completion_text.rsplit("####", 1)[-1]
                m = _NUM_RE.search(tail)
                format_ok = m is not None
        elif style == "none":
            format_ok = True
        else:
            raise ValueError(f"Unknown format style: {style}")

        reward = 1.0 if format_ok else 0.0

        return RewardOutput(
            total=reward,
            components={"format_ok": reward},
            metadata={"style": style, "format_ok": format_ok},
        )


class GSM8KReward(BaseRewardFunction):
    """Combined GSM8K reward with correctness and optional format checking.

    This is a convenience class that combines correctness and format rewards.

    Example:
        reward = GSM8KReward(format_style="hash4", format_weight=0.1)
    """

    def __init__(
        self,
        format_style: str = "none",
        format_weight: float = 0.0,
    ):
        self.correctness = GSM8KCorrectnessReward()
        self.format_reward = GSM8KFormatReward(style=format_style)
        self.format_weight = format_weight

    def compute(
        self,
        *,
        completion_text: str,
        prompt_text: str,
        completion_ids: list[int],
        prompt_ids: list[int],
        example: dict[str, Any],
        **kwargs: Any,
    ) -> RewardOutput:
        args = dict(
            completion_text=completion_text,
            prompt_text=prompt_text,
            completion_ids=completion_ids,
            prompt_ids=prompt_ids,
            example=example,
            **kwargs,
        )

        correctness_out = self.correctness.compute(**args)
        format_out = self.format_reward.compute(**args)

        total = correctness_out.total + self.format_weight * format_out.total

        return RewardOutput(
            total=total,
            components={
                "correctness": correctness_out.total,
                "format": format_out.total,
                "format_weighted": self.format_weight * format_out.total,
            },
            metadata={
                **correctness_out.metadata,
                **format_out.metadata,
            },
        )


def gsm8k_prompt(question: str) -> str:
    """Format a GSM8K question as a prompt."""
    q = question.strip()
    return f"Q: {q}\nA:"


def gsm8k_format_instruction(style: str) -> str:
    """Get format instruction for a given style."""
    if style == "hash4":
        return "Be concise. Give the final answer on its own line as: #### <number>"
    if style == "cot":
        return "Think through this step by step, showing your reasoning. Then give the final answer as: #### <number>"
    if style == "deepseek":
        return "Let me work through this problem step by step.\n\nFirst, I need to understand what the problem is asking."
    if style == "none":
        return ""
    raise ValueError(f"Unknown format style: {style}")
