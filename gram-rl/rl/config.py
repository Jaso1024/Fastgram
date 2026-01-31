"""Configuration classes for RL algorithms.

These configs are designed to be serializable and hashable, making it easy to
associate experimental results with their exact configuration.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Any
import json
import hashlib


class RatioMode(str, Enum):
    """How to compute importance sampling ratios."""
    TOKEN = "token"           # Per-token ratios (standard GRPO)
    SEQUENCE = "sequence"     # Sequence-level ratio normalized by length (GSPO)
    SEQUENCE_SUM = "sequence_sum"  # Sequence-level ratio without normalization


class KLEstimator(str, Enum):
    """KL divergence estimation method."""
    SAMPLE = "sample"    # Simple: log(p/q) - high variance but unbiased
    NONNEG = "nonneg"    # exp(log(p/q)) - log(p/q) - 1 - always non-negative


class LossAggregation(str, Enum):
    """How to aggregate loss across tokens/sequences."""
    TOKEN_MEAN = "token_mean"       # Mean over all tokens in batch
    SEQ_MEAN_TOKEN_SUM = "seq_mean_token_sum"   # Sum tokens per seq, mean over seqs
    SEQ_MEAN_TOKEN_MEAN = "seq_mean_token_mean" # Mean tokens per seq, mean over seqs


@dataclass(frozen=True)
class AlgorithmConfig:
    """Configuration for the policy gradient algorithm.

    This config can represent GRPO, GSPO, DAPO, or hybrid approaches
    depending on the parameter settings.

    Frozen dataclass ensures immutability for reliable experiment tracking.
    """
    # Core algorithm parameters
    clip_eps: float = 0.2
    clip_eps_low: Optional[float] = None   # If set, use asymmetric clipping (DAPO)
    clip_eps_high: Optional[float] = None  # If set, use asymmetric clipping (DAPO)

    # Importance ratio computation
    ratio_mode: RatioMode = RatioMode.SEQUENCE

    # KL regularization
    kl_beta: float = 0.02
    kl_estimator: KLEstimator = KLEstimator.NONNEG

    # Loss aggregation
    loss_aggregation: LossAggregation = LossAggregation.TOKEN_MEAN

    # Advantage computation
    advantage_eps: float = 1e-6  # Numerical stability for std normalization
    advantage_normalize: bool = True  # Whether to normalize advantages

    # DAPO-specific: dynamic sampling
    filter_zero_variance_groups: bool = False  # Filter groups with identical rewards

    # DAPO-specific: overlong penalty
    overlong_penalty_factor: float = 0.0  # 0 = disabled
    overlong_buffer_len: int = 100  # Start penalizing this many tokens before max

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        d["ratio_mode"] = self.ratio_mode.value
        d["kl_estimator"] = self.kl_estimator.value
        d["loss_aggregation"] = self.loss_aggregation.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AlgorithmConfig":
        """Create from dictionary."""
        d = d.copy()
        if "ratio_mode" in d:
            d["ratio_mode"] = RatioMode(d["ratio_mode"])
        if "kl_estimator" in d:
            d["kl_estimator"] = KLEstimator(d["kl_estimator"])
        if "loss_aggregation" in d:
            d["loss_aggregation"] = LossAggregation(d["loss_aggregation"])
        return cls(**d)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "AlgorithmConfig":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(s))

    def config_hash(self) -> str:
        """Return a short hash identifying this config."""
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:12]

    # Preset configurations for common algorithms
    @classmethod
    def grpo(cls, **overrides) -> "AlgorithmConfig":
        """Standard GRPO configuration (DeepSeekMath style)."""
        defaults = dict(
            ratio_mode=RatioMode.TOKEN,
            kl_beta=0.02,
            kl_estimator=KLEstimator.NONNEG,
            clip_eps=0.2,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def gspo(cls, **overrides) -> "AlgorithmConfig":
        """GSPO configuration (Qwen3 style)."""
        defaults = dict(
            ratio_mode=RatioMode.SEQUENCE,
            kl_beta=0.0,  # GSPO typically uses no KL
            clip_eps=0.2,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def dapo(cls, **overrides) -> "AlgorithmConfig":
        """DAPO configuration (ByteDance style)."""
        defaults = dict(
            ratio_mode=RatioMode.TOKEN,
            clip_eps_low=0.2,
            clip_eps_high=0.28,  # Clip-Higher
            kl_beta=0.0,
            loss_aggregation=LossAggregation.TOKEN_MEAN,
            filter_zero_variance_groups=True,
            overlong_penalty_factor=1.0,
            overlong_buffer_len=100,
        )
        defaults.update(overrides)
        return cls(**defaults)


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for the training loop."""
    # Batch structure
    batch_size: int = 1          # Number of prompts per batch
    group_size: int = 8          # Completions per prompt

    # Optimization
    learning_rate: float = 5e-6
    update_epochs: int = 1       # Epochs per rollout batch
    minibatch_size: int = 0      # 0 = use full batch
    max_grad_norm: float = 1.0   # Gradient clipping (0 = disabled)

    # Training duration
    total_steps: int = 1000

    # Checkpointing
    save_every: int = 200
    save_dir: str = "runs/gram_rl"

    # Logging
    log_every: int = 1
    wandb_project: str = ""
    wandb_entity: str = ""
    wandb_name: str = ""
    wandb_tags: str = ""

    # Reproducibility
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainingConfig":
        return cls(**d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "TrainingConfig":
        return cls.from_dict(json.loads(s))


@dataclass
class ExperimentConfig:
    """Full experiment configuration combining algorithm, training, and task settings.

    This is the top-level config that should be logged with every experiment.
    """
    algorithm: AlgorithmConfig
    training: TrainingConfig

    # Model settings
    model_name: str = ""

    # Task-specific settings stored as dict for flexibility
    task_config: dict[str, Any] = field(default_factory=dict)

    # Metadata
    experiment_name: str = ""
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm.to_dict(),
            "training": self.training.to_dict(),
            "model_name": self.model_name,
            "task_config": self.task_config,
            "experiment_name": self.experiment_name,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentConfig":
        return cls(
            algorithm=AlgorithmConfig.from_dict(d["algorithm"]),
            training=TrainingConfig.from_dict(d["training"]),
            model_name=d.get("model_name", ""),
            task_config=d.get("task_config", {}),
            experiment_name=d.get("experiment_name", ""),
            description=d.get("description", ""),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "ExperimentConfig":
        return cls.from_dict(json.loads(s))

    def save(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        """Load config from JSON file."""
        with open(path) as f:
            return cls.from_json(f.read())

    def experiment_id(self) -> str:
        """Generate a unique ID for this experiment configuration."""
        algo_hash = self.algorithm.config_hash()
        return f"{self.experiment_name}_{algo_hash}" if self.experiment_name else algo_hash
