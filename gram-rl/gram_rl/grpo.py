from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GrpoLoss:
    loss: "torch.Tensor"
    policy_loss: "torch.Tensor"
    kl_loss: "torch.Tensor"


def grpo_loss(
    *,
    logp_new: "torch.Tensor",
    logp_old: "torch.Tensor",
    token_mask: "torch.Tensor",
    advantages: "torch.Tensor",
    ref_logp: "torch.Tensor",
    clip_eps: float,
    kl_beta: float,
) -> GrpoLoss:
    import torch

    if token_mask.dtype != torch.bool:
        token_mask = token_mask.bool()
    if advantages.ndim != 1:
        raise ValueError("advantages must be [B]")

    adv = advantages.unsqueeze(1)
    ratio = torch.exp(logp_new - logp_old)
    ratio_clipped = torch.clamp(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps))
    obj = torch.minimum(ratio * adv, ratio_clipped * adv)
    policy_loss = -(obj[token_mask]).mean()

    kl = (logp_new - ref_logp)
    kl_loss = (kl[token_mask]).mean() * float(kl_beta)

    return GrpoLoss(loss=policy_loss + kl_loss, policy_loss=policy_loss, kl_loss=kl_loss)

