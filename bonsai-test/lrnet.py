"""LR-nets: CLT reparameterization for ternary distribution optimization.

Based on Shayer et al. (ICLR 2018, arXiv:1710.07739).

Each weight has a categorical distribution over {-1, 0, 1} parameterized
by logits (theta_neg, theta_pos; theta_zero=0). Instead of REINFORCE
(high variance), uses the Central Limit Theorem:

  Pre-activation z = sum_j w_j * x_j  where w_j ~ Categorical
  By CLT: z ~ Normal(mu, sigma^2) where:
    mu = sum_j E[w_j] * x_j = sum_j (p_pos_j - p_neg_j) * x_j
    sigma^2 = sum_j Var[w_j] * x_j^2

Reparameterization: z = mu + sigma * eps,  eps ~ N(0,1)

This is DIFFERENTIABLE — gradients flow to theta_neg, theta_pos via
standard backprop. No population, no score function, no variance issues.

Memory: ~10P bytes (2 logits + optimizer state, depends on optimizer).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from bitlinear import RMSNorm, ternary_quantize


class LRNetLinear(nn.Module):
    """Linear layer with CLT reparameterization for ternary distributions.

    Maintains per-weight logits theta_neg, theta_pos (theta_zero = 0).
    Forward uses CLT to compute mean and variance of pre-activations,
    then reparameterizes with Gaussian noise.

    At eval: uses mean only (deterministic).
    At train: adds calibrated noise via reparameterization trick.
    """

    def __init__(
        self,
        out_features: int,
        in_features: int,
        initial_ternary: Tensor,
        initial_scales: Tensor,
        weight_mask: Tensor | None = None,
        group_size: int = 128,
    ):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.group_size = group_size

        # Distribution parameters (trainable)
        # Moderate peaking so CLT variance term is nonzero at init
        INIT_PEAK = 2.0
        th_neg = torch.full((out_features, in_features), -INIT_PEAK)
        th_pos = torch.full((out_features, in_features), -INIT_PEAK)
        th_neg[initial_ternary == -1] = INIT_PEAK
        th_pos[initial_ternary == 1] = INIT_PEAK

        if weight_mask is not None:
            dead = ~weight_mask.bool()
            th_neg[dead] = 0.0
            th_pos[dead] = 0.0

        self.theta_neg = nn.Parameter(th_neg)
        self.theta_pos = nn.Parameter(th_pos)

        # Fixed scales from Phase 1 (not trainable)
        self.register_buffer("scales", initial_scales.clone())

        # Pre-compute expanded scales
        n_groups = (out_features * in_features) // group_size
        scales_exp = (
            self.scales.unsqueeze(1)
            .expand(n_groups, group_size)
            .reshape(out_features, in_features)
        )
        self.register_buffer("scales_exp", scales_exp)

        if weight_mask is not None:
            self.register_buffer("weight_mask", weight_mask.clone())
        else:
            self.register_buffer("weight_mask", None)

    def forward(self, x: Tensor) -> Tensor:
        logits = torch.stack(
            [self.theta_neg, torch.zeros_like(self.theta_neg), self.theta_pos],
            dim=-1,
        )
        p = F.softmax(logits, dim=-1)
        p_neg = p[..., 0]
        p_pos = p[..., 2]

        # E[w] = p_pos - p_neg
        w_mean = (p_pos - p_neg) * self.scales_exp

        # Var[w] = p_pos + p_neg - (p_pos - p_neg)^2
        diff = p_pos - p_neg
        w_var = (p_pos + p_neg - diff * diff) * (self.scales_exp * self.scales_exp)

        if self.weight_mask is not None:
            w_mean = w_mean * self.weight_mask
            w_var = w_var * self.weight_mask

        mu = F.linear(x, w_mean)
        sigma_sq = F.linear(x * x, w_var)
        sigma = torch.sqrt(sigma_sq.clamp(min=1e-8))

        if self.training:
            eps = torch.randn_like(mu)
            return mu + sigma * eps
        return mu

    def get_ternary(self) -> Tensor:
        """Return argmax ternary weights for inference."""
        with torch.no_grad():
            logits = torch.stack(
                [self.theta_neg, torch.zeros_like(self.theta_neg), self.theta_pos],
                dim=-1,
            )
            argmax = logits.argmax(dim=-1)
            ternary = (argmax - 1).to(torch.int8)
            if self.weight_mask is not None:
                ternary = ternary * self.weight_mask.to(torch.int8)
            return ternary

    def distribution_entropy(self) -> Tensor:
        """Per-weight entropy of the distribution."""
        with torch.no_grad():
            logits = torch.stack(
                [self.theta_neg, torch.zeros_like(self.theta_neg), self.theta_pos],
                dim=-1,
            )
            p = F.softmax(logits, dim=-1)
            entropy = -(p * (p + 1e-8).log()).sum(dim=-1)
            if self.weight_mask is not None:
                entropy = entropy * self.weight_mask
            return entropy


def create_lrnet_model(p1_model, pruned_data, cfg, device):
    """Create LRNet model from Phase 1 TernaryTransformer.

    Replaces BitLinear layers in transformer blocks with LRNetLinear.
    Keeps embed and lm_head as frozen BitLinear/BitEmbedding (STE).

    Args:
        p1_model: TernaryTransformer with pruned shadow weights
        pruned_data: dict from perlayer_prune (ternary, scales, mask per layer)
        cfg: TernaryConfig
        device: torch device

    Returns:
        model with LRNetLinear layers (only theta_neg/theta_pos are trainable)
    """
    from model import TernaryTransformer
    from bitlinear import BitLinear, BitEmbedding

    model = TernaryTransformer(cfg).to(device)
    model.load_state_dict(p1_model.state_dict())

    # Freeze embed and lm_head
    for param in model.embed.parameters():
        param.requires_grad = False
    for param in model.lm_head.parameters():
        param.requires_grad = False
    # Also freeze norm parameters in embed/lm_head if any
    for param in model.final_norm.parameters():
        param.requires_grad = False

    # Replace transformer BitLinear with LRNetLinear
    replacements = []
    for name, mod in model.named_modules():
        if not isinstance(mod, BitLinear):
            continue
        if not name.startswith("layers."):
            continue

        data = pruned_data.get(name)
        if data is not None:
            initial_ternary = data["ternary"]
            initial_scales = data["scales"]
            mask = data["mask"]
        else:
            w_q, scales = ternary_quantize(mod.weight.detach(), mod.group_size)
            initial_ternary = w_q.to(torch.int8)
            initial_scales = scales
            mask = None

        lrnet = LRNetLinear(
            out_features=mod.out_features,
            in_features=mod.in_features,
            initial_ternary=initial_ternary.to(device),
            initial_scales=initial_scales.to(device),
            weight_mask=mask.to(device) if mask is not None else None,
            group_size=mod.group_size,
        )
        replacements.append((name, lrnet))

    # Apply replacements
    for name, lrnet in replacements:
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], lrnet.to(device))

    # Freeze block norms (they were trained in Phase 1)
    for name, mod in model.named_modules():
        if isinstance(mod, RMSNorm) and name.startswith("layers."):
            for param in mod.parameters():
                param.requires_grad = False

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"LRNet model: {n_total/1e6:.2f}M total, "
        f"{n_trainable/1e3:.1f}k trainable (distribution params only)"
    )

    return model


class LRNetTrainer:
    """Trainer for LR-nets style ternary distribution optimization.

    Uses standard backprop through CLT reparameterization.
    Only theta_neg and theta_pos are trainable.

    Experiment configs:
      B1: Standard LRNet (AdamW on distribution params)
      B2: LRNet + entropy regularization (prevent distribution collapse)
    """

    def __init__(
        self,
        model,
        *,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        entropy_reg: float = 0.0,
        total_steps: int = 5000,
        warmup_steps: int = 200,
    ):
        self.model = model
        self.entropy_reg = entropy_reg
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.lr_max = lr

        # Only optimize theta_neg, theta_pos parameters
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable, lr=lr, betas=(0.9, 0.95),
            weight_decay=weight_decay, eps=1e-8,
        )

        # Collect LRNetLinear modules for stats/regularization
        self.lrnet_modules: list[tuple[str, LRNetLinear]] = []
        for name, mod in model.named_modules():
            if isinstance(mod, LRNetLinear):
                self.lrnet_modules.append((name, mod))

        n_dist_params = sum(
            m.theta_neg.numel() + m.theta_pos.numel()
            for _, m in self.lrnet_modules
        )
        print(
            f"LRNet trainer: {len(self.lrnet_modules)} modules, "
            f"{n_dist_params/1e3:.1f}k distribution params, "
            f"lr={lr}, entropy_reg={entropy_reg}"
        )

        self._step_count = 0

    def _cosine_lr(self) -> float:
        if self._step_count < self.warmup_steps:
            return self.lr_max * self._step_count / max(1, self.warmup_steps)
        progress = (self._step_count - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        return self.lr_max * 0.1 + 0.5 * self.lr_max * 0.9 * (
            1 + math.cos(math.pi * progress)
        )

    def step(self, x: Tensor, y: Tensor, device: torch.device) -> dict:
        """One training step with CLT reparameterized forward + backprop."""
        self._step_count += 1

        lr = self._cosine_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        x, y = x.to(device), y.to(device)

        self.model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.model(x, labels=y)
            loss = out["loss"]

            if self.entropy_reg > 0:
                entropy_sum = sum(
                    m.distribution_entropy().sum() for _, m in self.lrnet_modules
                )
                n_weights = sum(
                    m.theta_neg.numel() for _, m in self.lrnet_modules
                )
                loss = loss - self.entropy_reg * entropy_sum / n_weights

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            "loss": out["loss"].item(),
            "lr": lr,
        }

    def distribution_stats(self) -> dict:
        """Return distribution statistics for logging."""
        total_entropy = 0.0
        total_params = 0

        for _name, mod in self.lrnet_modules:
            entropy = mod.distribution_entropy()
            if mod.weight_mask is not None:
                alive = mod.weight_mask.bool()
                total_entropy += entropy[alive].sum().item()
                total_params += alive.sum().item()
            else:
                total_entropy += entropy.sum().item()
                total_params += entropy.numel()

        return {
            "mean_entropy": total_entropy / max(1, total_params),
            "n_params": total_params,
        }
