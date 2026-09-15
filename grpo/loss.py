"""Group-relative advantages and assistant-only clipped/KL objective."""
import math


def group_advantages(rewards: list[float], std_floor: float = 0.1) -> list[float]:
    if len(rewards) < 2 or std_floor <= 0 or not all(math.isfinite(x) for x in rewards):
        raise ValueError("Invalid reward group/std floor")
    mean = sum(rewards)/len(rewards)
    if max(rewards)-min(rewards) <= 1e-8:
        return [0.0]*len(rewards)
    std = math.sqrt(sum((x-mean)**2 for x in rewards)/len(rewards))
    return [(x-mean)/max(std,std_floor) for x in rewards]


def token_loss(current, old, reference, advantage: float, beta: float = 0.02, epsilon: float = 0.2):
    """Inputs contain ONLY sampled assistant positions; no context/observation logprobs."""
    import torch
    if current.ndim != 1 or current.shape != old.shape or current.shape != reference.shape or not current.numel():
        raise ValueError("Sampled-token shapes differ")
    ratio = torch.exp(current-old.detach())
    pg = torch.minimum(ratio*advantage,ratio.clamp(1-epsilon,1+epsilon)*advantage)
    u = reference.detach()-current
    kl = torch.expm1(u)-u
    loss = -pg + beta*kl
    if not torch.isfinite(loss).all():
        raise ValueError("Non-finite GRPO loss; no silent clamp")
    return loss.sum(), {"pg_sum":(-pg).sum().detach().item(),"kl_sum":kl.sum().detach().item(),
                        "clip_count":((ratio<1-epsilon)|(ratio>1+epsilon)).sum().item(),
                        "token_count":current.numel()}
