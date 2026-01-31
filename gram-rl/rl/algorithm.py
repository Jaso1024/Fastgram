"""Configurable policy gradient algorithm for LLM RL.

This module provides a unified implementation that can be configured to
behave like GRPO, GSPO, DAPO, or custom hybrid approaches.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

from .config import AlgorithmConfig, RatioMode, KLEstimator, LossAggregation


@dataclass
class PolicyGradientOutput:
    """Output from a policy gradient step."""
    loss: Any  # torch.Tensor
    policy_loss: Any  # torch.Tensor
    kl_loss: Any  # torch.Tensor

    # Diagnostics
    mean_ratio: float = 0.0
    clip_fraction: float = 0.0
    mean_advantage: float = 0.0


class PolicyGradientAlgorithm:
    """Configurable policy gradient algorithm.

    Supports GRPO, GSPO, DAPO, and hybrid configurations through AlgorithmConfig.

    Example usage:
        config = AlgorithmConfig.gspo()
        algo = PolicyGradientAlgorithm(config)
        output = algo.compute_loss(
            logp_new=new_logprobs,
            logp_old=old_logprobs,
            token_mask=mask,
            advantages=advs,
            ref_logp=ref_logprobs,
        )
    """

    def __init__(self, config: AlgorithmConfig):
        self.config = config

    def compute_loss(
        self,
        *,
        logp_new: "torch.Tensor",
        logp_old: "torch.Tensor",
        token_mask: "torch.Tensor",
        advantages: "torch.Tensor",
        ref_logp: Optional["torch.Tensor"] = None,
    ) -> PolicyGradientOutput:
        """Compute the policy gradient loss.

        Args:
            logp_new: [B, T] log probabilities under current policy
            logp_old: [B, T] log probabilities under old policy (detached)
            token_mask: [B, T] boolean mask for valid completion tokens
            advantages: [B] per-sequence advantages
            ref_logp: [B, T] log probabilities under reference policy (for KL)

        Returns:
            PolicyGradientOutput with loss and diagnostics
        """
        import torch

        cfg = self.config

        if token_mask.dtype != torch.bool:
            token_mask = token_mask.bool()
        if advantages.ndim != 1:
            raise ValueError("advantages must be [B]")

        # Compute log ratio
        log_ratio = logp_new - logp_old
        token_count = token_mask.sum(dim=1).clamp_min(1)

        # Compute importance ratio based on mode
        if cfg.ratio_mode == RatioMode.TOKEN:
            policy_loss, diagnostics = self._token_level_loss(
                log_ratio=log_ratio,
                token_mask=token_mask,
                token_count=token_count,
                advantages=advantages,
            )
        elif cfg.ratio_mode == RatioMode.SEQUENCE:
            policy_loss, diagnostics = self._sequence_level_loss(
                log_ratio=log_ratio,
                token_mask=token_mask,
                token_count=token_count,
                advantages=advantages,
                normalize_by_length=True,
            )
        elif cfg.ratio_mode == RatioMode.SEQUENCE_SUM:
            policy_loss, diagnostics = self._sequence_level_loss(
                log_ratio=log_ratio,
                token_mask=token_mask,
                token_count=token_count,
                advantages=advantages,
                normalize_by_length=False,
            )
        else:
            raise ValueError(f"Unknown ratio_mode: {cfg.ratio_mode}")

        # Compute KL loss
        kl_loss = self._compute_kl_loss(
            logp_new=logp_new,
            ref_logp=ref_logp,
            token_mask=token_mask,
        )

        total_loss = policy_loss + kl_loss

        return PolicyGradientOutput(
            loss=total_loss,
            policy_loss=policy_loss,
            kl_loss=kl_loss,
            mean_ratio=diagnostics.get("mean_ratio", 0.0),
            clip_fraction=diagnostics.get("clip_fraction", 0.0),
            mean_advantage=float(advantages.mean().item()),
        )

    def _get_clip_bounds(self) -> tuple[float, float]:
        """Get clipping bounds, supporting asymmetric clipping (DAPO)."""
        cfg = self.config
        if cfg.clip_eps_low is not None and cfg.clip_eps_high is not None:
            return (1.0 - cfg.clip_eps_low, 1.0 + cfg.clip_eps_high)
        return (1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)

    def _token_level_loss(
        self,
        *,
        log_ratio: "torch.Tensor",
        token_mask: "torch.Tensor",
        token_count: "torch.Tensor",
        advantages: "torch.Tensor",
    ) -> tuple["torch.Tensor", dict]:
        """Compute token-level policy loss (standard GRPO/DAPO)."""
        import torch

        cfg = self.config
        clip_low, clip_high = self._get_clip_bounds()

        # Expand advantages to token level
        adv = advantages.unsqueeze(1)  # [B, 1]

        # Compute ratios
        ratio = torch.exp(log_ratio)  # [B, T]
        ratio_clipped = torch.clamp(ratio, clip_low, clip_high)

        # PPO-style clipped objective
        obj1 = ratio * adv
        obj2 = ratio_clipped * adv
        obj = torch.minimum(obj1, obj2)

        # Apply mask
        obj = obj * token_mask

        # Aggregate based on config
        if cfg.loss_aggregation == LossAggregation.TOKEN_MEAN:
            policy_loss = -(obj.sum() / token_mask.sum().clamp_min(1))
        elif cfg.loss_aggregation == LossAggregation.SEQ_MEAN_TOKEN_SUM:
            obj_seq = obj.sum(dim=1)
            policy_loss = -obj_seq.mean()
        elif cfg.loss_aggregation == LossAggregation.SEQ_MEAN_TOKEN_MEAN:
            obj_seq = obj.sum(dim=1) / token_count
            policy_loss = -obj_seq.mean()
        else:
            raise ValueError(f"Unknown loss_aggregation: {cfg.loss_aggregation}")

        # Diagnostics
        with torch.no_grad():
            mean_ratio = float((ratio * token_mask).sum() / token_mask.sum().clamp_min(1))
            clipped = ((ratio < clip_low) | (ratio > clip_high)) & token_mask
            clip_fraction = float(clipped.sum() / token_mask.sum().clamp_min(1))

        return policy_loss, {"mean_ratio": mean_ratio, "clip_fraction": clip_fraction}

    def _sequence_level_loss(
        self,
        *,
        log_ratio: "torch.Tensor",
        token_mask: "torch.Tensor",
        token_count: "torch.Tensor",
        advantages: "torch.Tensor",
        normalize_by_length: bool,
    ) -> tuple["torch.Tensor", dict]:
        """Compute sequence-level policy loss (GSPO style)."""
        import torch

        cfg = self.config
        clip_low, clip_high = self._get_clip_bounds()

        # Compute sequence-level log ratio
        if normalize_by_length:
            # GSPO: geometric mean of ratios
            log_ratio_seq = (log_ratio * token_mask).sum(dim=1) / token_count
        else:
            # Sum of log ratios (product of ratios)
            log_ratio_seq = (log_ratio * token_mask).sum(dim=1)

        # Convert to ratio
        ratio = torch.exp(log_ratio_seq)  # [B]
        ratio_clipped = torch.clamp(ratio, clip_low, clip_high)

        # PPO-style clipped objective
        obj1 = ratio * advantages
        obj2 = ratio_clipped * advantages
        obj = torch.minimum(obj1, obj2)

        policy_loss = -obj.mean()

        # Diagnostics
        with torch.no_grad():
            mean_ratio = float(ratio.mean())
            clip_fraction = float(((ratio < clip_low) | (ratio > clip_high)).float().mean())

        return policy_loss, {"mean_ratio": mean_ratio, "clip_fraction": clip_fraction}

    def _compute_kl_loss(
        self,
        *,
        logp_new: "torch.Tensor",
        ref_logp: Optional["torch.Tensor"],
        token_mask: "torch.Tensor",
    ) -> "torch.Tensor":
        """Compute KL divergence penalty."""
        import torch

        cfg = self.config

        if cfg.kl_beta <= 0.0 or ref_logp is None:
            return logp_new.sum() * 0.0  # Zero loss that maintains grad structure

        log_r = logp_new - ref_logp

        if cfg.kl_estimator == KLEstimator.SAMPLE:
            # Simple estimator: E[log(p/q)]
            kl_tok = log_r
        elif cfg.kl_estimator == KLEstimator.NONNEG:
            # Non-negative estimator: E[p/q - 1 - log(p/q)]
            r = torch.exp(log_r)
            kl_tok = (r - 1.0) - log_r
        else:
            raise ValueError(f"Unknown kl_estimator: {cfg.kl_estimator}")

        kl_loss = kl_tok[token_mask].mean() * cfg.kl_beta
        return kl_loss


def compute_advantages(
    rewards: list[float],
    normalize: bool = True,
    eps: float = 1e-6,
) -> list[float]:
    """Compute group-relative advantages from rewards.

    Args:
        rewards: List of rewards for each sample in the group
        normalize: Whether to normalize by std (True = GRPO style)
        eps: Small constant for numerical stability

    Returns:
        List of advantage values
    """
    if not rewards:
        return []

    mean = sum(rewards) / len(rewards)

    if not normalize:
        return [r - mean for r in rewards]

    # Compute std with Bessel's correction
    if len(rewards) > 1:
        var = sum((r - mean) ** 2 for r in rewards) / (len(rewards) - 1)
        std = var ** 0.5 if var > 0 else 0.0
    else:
        std = 0.0

    if std <= 0:
        return [0.0 for _ in rewards]

    return [(r - mean) / (std + eps) for r in rewards]


def filter_zero_variance_groups(
    rewards_by_group: list[list[float]],
    group_indices: list[list[int]],
) -> tuple[list[list[float]], list[list[int]]]:
    """Filter out groups with zero variance (DAPO dynamic sampling).

    Args:
        rewards_by_group: List of reward lists, one per prompt group
        group_indices: Corresponding indices into the full batch

    Returns:
        Filtered rewards and indices with zero-variance groups removed
    """
    filtered_rewards = []
    filtered_indices = []

    for rewards, indices in zip(rewards_by_group, group_indices):
        if len(set(rewards)) > 1:  # Has variance
            filtered_rewards.append(rewards)
            filtered_indices.append(indices)

    return filtered_rewards, filtered_indices


def apply_overlong_penalty(
    rewards: list[float],
    completion_lengths: list[int],
    max_length: int,
    buffer_len: int,
    penalty_factor: float,
) -> list[float]:
    """Apply DAPO-style overlong reward shaping.

    Linearly penalizes completions that approach max_length.

    Args:
        rewards: Original rewards
        completion_lengths: Length of each completion
        max_length: Maximum allowed length
        buffer_len: Start penalizing this many tokens before max
        penalty_factor: Scale of penalty

    Returns:
        Adjusted rewards
    """
    if penalty_factor <= 0:
        return rewards

    threshold = max_length - buffer_len
    adjusted = []

    for r, length in zip(rewards, completion_lengths):
        if length > threshold:
            exceed = length - threshold
            penalty = min(exceed / buffer_len * penalty_factor, penalty_factor)
            adjusted.append(r - penalty)
        else:
            adjusted.append(r)

    return adjusted
