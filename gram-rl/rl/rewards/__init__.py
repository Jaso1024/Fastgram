"""Reward functions for RL training.

Rewards are completely decoupled from the algorithm. Any function that
implements the RewardFunction protocol can be used.

Example of creating a custom reward:
    from rl.rewards.base import BaseRewardFunction, RewardOutput

    class MyReward(BaseRewardFunction):
        def compute(self, *, completion_text, example, **kwargs):
            # Your reward logic here
            score = evaluate(completion_text, example["expected"])
            return RewardOutput(
                total=score,
                components={"score": score},
                metadata={},
            )
"""

from .base import (
    RewardOutput,
    RewardFunction,
    BaseRewardFunction,
    CompositeReward,
    LengthPenaltyReward,
)
from .gsm8k import (
    GSM8KCorrectnessReward,
    GSM8KFormatReward,
    GSM8KReward,
    gsm8k_prompt,
    gsm8k_format_instruction,
    extract_number,
    normalize_number,
)

__all__ = [
    # Base
    "RewardOutput",
    "RewardFunction",
    "BaseRewardFunction",
    "CompositeReward",
    "LengthPenaltyReward",
    # GSM8K
    "GSM8KCorrectnessReward",
    "GSM8KFormatReward",
    "GSM8KReward",
    "gsm8k_prompt",
    "gsm8k_format_instruction",
    "extract_number",
    "normalize_number",
]
