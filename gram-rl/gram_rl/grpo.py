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
    ref_logp: "torch.Tensor | None" = None,
    clip_eps: float,
    kl_beta: float,
    ratio_mode: str = "sequence",
    kl_estimator: str = "nonneg",
) -> GrpoLoss:
    import torch

    if token_mask.dtype != torch.bool:
        token_mask = token_mask.bool()
    if advantages.ndim != 1:
        raise ValueError("advantages must be [B]")

    log_ratio = logp_new - logp_old
    token_count = token_mask.sum(dim=1).clamp_min(1)
    if str(ratio_mode) == "token":
        adv = advantages.unsqueeze(1)
        ratio = torch.exp(log_ratio)
        ratio_clipped = torch.clamp(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps))
        obj = torch.minimum(ratio * adv, ratio_clipped * adv)
        obj = obj * token_mask
        obj_seq = obj.sum(dim=1) / token_count
        policy_loss = -(obj_seq).mean()
    elif str(ratio_mode) == "sequence":
        log_ratio_seq = (log_ratio * token_mask).sum(dim=1) / token_count
        ratio = torch.exp(log_ratio_seq)
        ratio_clipped = torch.clamp(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps))
        obj = torch.minimum(ratio * advantages, ratio_clipped * advantages)
        policy_loss = -(obj).mean()
    elif str(ratio_mode) == "sequence_sum":
        log_ratio_seq = (log_ratio * token_mask).sum(dim=1)
        ratio = torch.exp(log_ratio_seq)
        ratio_clipped = torch.clamp(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps))
        obj = torch.minimum(ratio * advantages, ratio_clipped * advantages)
        policy_loss = -(obj).mean()
    else:
        raise ValueError("ratio_mode must be 'sequence', 'sequence_sum', or 'token'")

    if float(kl_beta) <= 0.0 or ref_logp is None:
        kl_loss = logp_new.sum() * 0.0
    else:
        log_r = (logp_new - ref_logp)
        if str(kl_estimator) == "sample":
            kl_tok = log_r
        elif str(kl_estimator) == "nonneg":
            r = torch.exp(log_r)
            kl_tok = (r - 1.0) - log_r
        else:
            raise ValueError("kl_estimator must be 'nonneg' or 'sample'")
        kl_loss = (kl_tok[token_mask]).mean() * float(kl_beta)

    return GrpoLoss(loss=policy_loss + kl_loss, policy_loss=policy_loss, kl_loss=kl_loss)
