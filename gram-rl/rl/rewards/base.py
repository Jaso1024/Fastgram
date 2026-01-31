"""Base reward function interface.

Rewards are completely decoupled from the RL algorithm. Any reward function
that implements the RewardFunction protocol can be plugged in.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable


@dataclass
class RewardOutput:
    """Output from a reward function.

    Supports multi-component rewards for analysis and weighted combination.
    """
    total: float
    components: dict[str, float]
    metadata: dict[str, Any]

    @classmethod
    def simple(cls, reward: float) -> "RewardOutput":
        """Create a simple single-value reward."""
        return cls(total=reward, components={"reward": reward}, metadata={})


@runtime_checkable
class RewardFunction(Protocol):
    """Protocol for reward functions.

    Any callable that takes completion info and returns RewardOutput
    can be used as a reward function.
    """

    def __call__(
        self,
        *,
        completion_text: str,
        prompt_text: str,
        completion_ids: list[int],
        prompt_ids: list[int],
        example: dict[str, Any],
        **kwargs: Any,
    ) -> RewardOutput:
        """Compute reward for a completion.

        Args:
            completion_text: Decoded completion text
            prompt_text: The prompt that was given
            completion_ids: Token IDs of the completion
            prompt_ids: Token IDs of the prompt
            example: The original dataset example
            **kwargs: Additional task-specific arguments

        Returns:
            RewardOutput with total reward and component breakdown
        """
        ...


class BaseRewardFunction(ABC):
    """Abstract base class for reward functions.

    Provides a structured interface for implementing reward functions.
    """

    @abstractmethod
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
        """Compute the reward. Implement this in subclasses."""
        ...

    def __call__(
        self,
        *,
        completion_text: str,
        prompt_text: str,
        completion_ids: list[int],
        prompt_ids: list[int],
        example: dict[str, Any],
        **kwargs: Any,
    ) -> RewardOutput:
        """Make the class callable."""
        return self.compute(
            completion_text=completion_text,
            prompt_text=prompt_text,
            completion_ids=completion_ids,
            prompt_ids=prompt_ids,
            example=example,
            **kwargs,
        )


class CompositeReward(BaseRewardFunction):
    """Combines multiple reward functions with weights.

    Example:
        reward = CompositeReward([
            (correctness_reward, 1.0),
            (format_reward, 0.1),
            (length_penalty, 0.05),
        ])
    """

    def __init__(self, rewards: list[tuple[RewardFunction, float]]):
        """Initialize with list of (reward_fn, weight) pairs."""
        self.rewards = rewards

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
        total = 0.0
        components: dict[str, float] = {}
        metadata: dict[str, Any] = {}

        for i, (reward_fn, weight) in enumerate(self.rewards):
            output = reward_fn(
                completion_text=completion_text,
                prompt_text=prompt_text,
                completion_ids=completion_ids,
                prompt_ids=prompt_ids,
                example=example,
                **kwargs,
            )

            weighted = output.total * weight
            total += weighted

            # Prefix component names to avoid collision
            prefix = f"r{i}_"
            for k, v in output.components.items():
                components[f"{prefix}{k}"] = v
            components[f"{prefix}weighted"] = weighted

            for k, v in output.metadata.items():
                metadata[f"{prefix}{k}"] = v

        components["total"] = total
        return RewardOutput(total=total, components=components, metadata=metadata)


class LengthPenaltyReward(BaseRewardFunction):
    """Penalizes long completions."""

    def __init__(
        self,
        max_length: int,
        penalty_per_token: float = 0.001,
        threshold_ratio: float = 0.8,
    ):
        """
        Args:
            max_length: Maximum expected completion length
            penalty_per_token: Penalty applied per token over threshold
            threshold_ratio: Start penalizing at this fraction of max_length
        """
        self.max_length = max_length
        self.penalty_per_token = penalty_per_token
        self.threshold = int(max_length * threshold_ratio)

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
        length = len(completion_ids)
        if length <= self.threshold:
            penalty = 0.0
        else:
            excess = length - self.threshold
            penalty = -excess * self.penalty_per_token

        return RewardOutput(
            total=penalty,
            components={"length": float(length), "penalty": penalty},
            metadata={"completion_length": length},
        )
