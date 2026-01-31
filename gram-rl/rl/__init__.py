"""Modular RL framework for LLM training.

This package provides a configurable implementation of modern policy gradient
algorithms including GRPO, GSPO, and DAPO.

Example usage:
    from rl.config import AlgorithmConfig, TrainingConfig
    from rl.algorithm import PolicyGradientAlgorithm
    from rl.rewards.gsm8k import GSM8KReward

    # Configure algorithm
    config = AlgorithmConfig.gspo(kl_beta=0.01)
    algo = PolicyGradientAlgorithm(config)

    # Create reward function
    reward_fn = GSM8KReward(format_style="hash4", format_weight=0.1)

    # Use in training loop
    output = algo.compute_loss(
        logp_new=new_logprobs,
        logp_old=old_logprobs,
        token_mask=mask,
        advantages=advs,
    )
"""

from .config import (
    AlgorithmConfig,
    TrainingConfig,
    ExperimentConfig,
    RatioMode,
    KLEstimator,
    LossAggregation,
)
from .algorithm import (
    PolicyGradientAlgorithm,
    PolicyGradientOutput,
    compute_advantages,
    filter_zero_variance_groups,
    apply_overlong_penalty,
)

__all__ = [
    # Config
    "AlgorithmConfig",
    "TrainingConfig",
    "ExperimentConfig",
    "RatioMode",
    "KLEstimator",
    "LossAggregation",
    # Algorithm
    "PolicyGradientAlgorithm",
    "PolicyGradientOutput",
    "compute_advantages",
    "filter_zero_variance_groups",
    "apply_overlong_penalty",
]
