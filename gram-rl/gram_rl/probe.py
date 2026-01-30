"""Learned acceptance probe for gram-assisted decoding."""
from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


@dataclass
class ProbeOutput:
    """Output from the acceptance probe."""
    accept: torch.Tensor  # [B] bool - whether to accept
    log_prob: torch.Tensor  # [B] float - log prob of the decision
    prob: torch.Tensor  # [B] float - probability of accepting


class AcceptProbe(nn.Module):
    """
    A learned probe that decides whether to accept draft tokens.

    Takes the model's hidden state and outputs an accept/reject decision.
    Trained via RL to maximize reward (correctness + acceptance bonus).
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        deterministic: bool = False,
        threshold: float = 0.5,
    ) -> ProbeOutput:
        """
        Decide whether to accept based on hidden states.

        Args:
            hidden_states: [B, H] or [B, T, H] - if 3D, uses last position
            deterministic: if True, use threshold instead of sampling
            threshold: acceptance threshold when deterministic=True

        Returns:
            ProbeOutput with accept decisions and log probs
        """
        if hidden_states.ndim == 3:
            # Use last token's hidden state
            hidden_states = hidden_states[:, -1, :]

        logits = self.net(hidden_states).squeeze(-1)  # [B]
        probs = torch.sigmoid(logits)

        if deterministic:
            accept = probs >= threshold
        else:
            accept = torch.bernoulli(probs).bool()

        # Compute log prob of the decision
        # log P(decision) = accept * log(p) + (1-accept) * log(1-p)
        log_probs = torch.where(
            accept,
            torch.log(probs + 1e-8),
            torch.log(1 - probs + 1e-8),
        )

        return ProbeOutput(accept=accept, log_prob=log_probs, prob=probs)

    def log_prob_of_action(
        self,
        hidden_states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probability of given actions (for policy gradient).

        Args:
            hidden_states: [B, H] or [B, T, H]
            actions: [B] bool - the accept/reject decisions taken

        Returns:
            [B] log probabilities
        """
        if hidden_states.ndim == 3:
            hidden_states = hidden_states[:, -1, :]

        logits = self.net(hidden_states).squeeze(-1)
        probs = torch.sigmoid(logits)

        log_probs = torch.where(
            actions,
            torch.log(probs + 1e-8),
            torch.log(1 - probs + 1e-8),
        )

        return log_probs


def create_probe_for_model(model: nn.Module) -> AcceptProbe:
    """Create an AcceptProbe sized for the given model."""
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("Model must have a config with hidden_size")

    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        # Try alternative names
        hidden_size = getattr(config, "d_model", None)
        if hidden_size is None:
            hidden_size = getattr(config, "n_embd", None)

    if hidden_size is None:
        raise ValueError("Could not determine hidden_size from model config")

    return AcceptProbe(hidden_size=hidden_size)
