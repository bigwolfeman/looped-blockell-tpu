"""Pressure Flip: minimal-memory ternary training via gradient sign accumulation.

STE shadow weights are floats that drift toward quantization boundaries.
For ternary {-1, 0, 1}, we don't need a float — just a counter that tracks
"how much evidence have I accumulated for flipping?"

Algorithm:
  1. Forward with ternary * scales (effective weight, detached as leaf)
  2. Backward (gradient flows to the effective weight leaf)
  3. pressure += sign(grad)  — accumulate directional evidence into int8
  4. Where |pressure| > threshold: flip ternary one step, reset pressure

Memory: 2 bytes/param (int8 ternary + int8 pressure) + transient grad during backward
vs STE+Adam: 14 bytes/param (bf16 shadow + fp32 m1 + fp32 m2 + fp32 grad)

7x memory savings with backprop-quality directional signal.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from bitlinear import RMSNorm, ternary_quantize


class PressureFlipLinear(nn.Module):
    """Linear layer trained via pressure accumulation and threshold flipping.

    No shadow weights. No optimizer state. Just ternary + pressure counter.
    Forward creates a detached leaf for gradient capture, backward populates .grad,
    then pressure_update() extracts sign(grad) and accumulates into int8 pressure.
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

        self.register_buffer("ternary", initial_ternary.clone().to(torch.int8))
        self.register_buffer("scales", initial_scales.clone().to(torch.bfloat16))
        self.register_buffer(
            "pressure", torch.zeros_like(initial_ternary, dtype=torch.int8)
        )
        if weight_mask is not None:
            self.register_buffer("weight_mask", weight_mask.clone())
        else:
            self.register_buffer("weight_mask", None)

        # Transient: holds the leaf tensor whose .grad we read after backward
        self._w_leaf: Tensor | None = None

    def _effective_weight(self) -> Tensor:
        gs = self.group_size
        n_groups = self.ternary.numel() // gs
        scales_exp = (
            self.scales.unsqueeze(1)
            .expand(n_groups, gs)
            .reshape(self.out_features, self.in_features)
        )
        w = self.ternary.float() * scales_exp
        if self.weight_mask is not None:
            w = w * self.weight_mask
        return w

    def forward(self, x: Tensor) -> Tensor:
        # Create detached leaf that captures gradient
        w = self._effective_weight().detach().requires_grad_(True)
        self._w_leaf = w
        return F.linear(x, w)

    @torch.no_grad()
    def pressure_update(self, threshold: int = 10, scale_lr: float = 0.0) -> dict:
        """Read gradient from leaf, accumulate sign into pressure, flip on threshold.

        Must be called AFTER backward() and BEFORE zero_grad().
        """
        if self._w_leaf is None or self._w_leaf.grad is None:
            return {"n_flipped": 0, "n_up": 0, "n_down": 0,
                    "mean_pressure": 0.0, "max_pressure": 0}

        grad = self._w_leaf.grad

        if self.weight_mask is not None:
            grad = grad * self.weight_mask

        # Accumulate negative gradient sign into pressure
        # Negative gradient = "weight should increase" = pressure toward +1
        grad_sign = (-grad).sign().to(torch.int8)

        # Saturating add to int8 pressure
        new_p = (self.pressure.to(torch.int16) + grad_sign.to(torch.int16)).clamp(-127, 127)
        self.pressure.copy_(new_p.to(torch.int8))

        # Flip where pressure exceeds threshold
        flip_pos = self.pressure > threshold
        flip_neg = self.pressure < -threshold

        # Step toward +1 (only if not already at +1)
        step_up = flip_pos & (self.ternary < 1)
        self.ternary[step_up] += 1

        # Step toward -1 (only if not already at -1)
        step_down = flip_neg & (self.ternary > -1)
        self.ternary[step_down] -= 1

        # Reset pressure for flipped weights
        flipped = step_up | step_down
        self.pressure[flipped] = 0

        # Enforce mask
        if self.weight_mask is not None:
            dead = ~self.weight_mask.bool()
            self.ternary[dead] = 0
            self.pressure[dead] = 0

        # Optional: update scales via gradient magnitude EMA
        if scale_lr > 0:
            gs = self.group_size
            grad_mag = grad.abs().reshape(-1, gs).mean(dim=1)
            self.scales.lerp_(grad_mag.to(self.scales.dtype), scale_lr)
            self.scales.clamp_(min=1e-6)

        # Free the leaf reference
        self._w_leaf = None

        return {
            "n_flipped": flipped.sum().item(),
            "n_up": step_up.sum().item(),
            "n_down": step_down.sum().item(),
            "mean_pressure": self.pressure.float().abs().mean().item(),
            "max_pressure": self.pressure.abs().max().item(),
        }


class PressureFlipBlock(nn.Module):
    """Transformer block with PressureFlipLinear layers."""

    def __init__(self, cfg, base_attn, base_mlp, base_norms, pruned_layers, device):
        super().__init__()
        self.norm_attn = base_norms[0]
        self.norm_mlp = base_norms[1]
        for p in self.norm_attn.parameters():
            p.requires_grad = False
        for p in self.norm_mlp.parameters():
            p.requires_grad = False

        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads

        # Attention projections
        self.W_q = _build_pf(base_attn.W_q, pruned_layers, device)
        self.W_k = _build_pf(base_attn.W_k, pruned_layers, device)
        self.W_v = _build_pf(base_attn.W_v, pruned_layers, device)
        self.W_o = _build_pf(base_attn.W_o, pruned_layers, device)

        if base_attn.alpha is not None:
            self.register_buffer("alpha", base_attn.alpha.detach().clone())
        else:
            self.alpha = None

        # MLP
        self.w_gate = _build_pf(base_mlp.w_gate, pruned_layers, device)
        self.w_up = _build_pf(base_mlp.w_up, pruned_layers, device)
        self.w_down = _build_pf(base_mlp.w_down, pruned_layers, device)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        B, S, D = x.shape
        H, Dh = self.n_heads, self.d_head

        normed = self.norm_attn(x)
        q = self.W_q(normed).reshape(B, S, H, Dh).transpose(1, 2)
        k = self.W_k(normed).reshape(B, S, H, Dh).transpose(1, 2)
        v = self.W_v(normed).reshape(B, S, H, Dh).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        if self.alpha is not None:
            x_heads = normed.reshape(B, S, H, Dh).transpose(1, 2)
            attn_out = attn_out + self.alpha * x_heads

        attn_out = self.W_o(attn_out.transpose(1, 2).reshape(B, S, D))
        x = x + attn_out

        normed = self.norm_mlp(x)
        x = x + self.w_down(F.silu(self.w_gate(normed)) * self.w_up(normed))
        return x


def _build_pf(bitlinear, pruned_layers: dict, device) -> PressureFlipLinear:
    """Create PressureFlipLinear from BitLinear + pruned data."""
    w = bitlinear.weight.detach()
    gs = bitlinear.group_size
    shape = tuple(w.shape)

    # Find matching pruned data
    mask = None
    w_q, scales = ternary_quantize(w, gs)
    for name in list(pruned_layers.keys()):
        if pruned_layers[name]["shape"] == shape:
            data = pruned_layers.pop(name)
            w_q = data["ternary"]
            scales = data["scales"]
            mask = data["mask"]
            break

    return PressureFlipLinear(
        out_features=bitlinear.out_features,
        in_features=bitlinear.in_features,
        initial_ternary=w_q.to(torch.int8).to(device),
        initial_scales=scales.to(device),
        weight_mask=mask.to(device) if mask is not None else None,
        group_size=gs,
    )


def create_pressure_flip_model(p1_model, pruned_data, cfg, device):
    """Build PressureFlipTransformer from Phase 1 model + pruning data."""
    from model import TernaryTransformer

    # Make mutable copy of pruned_data (we pop entries during construction)
    pd = dict(pruned_data)

    model = nn.Module()
    model.cfg = cfg

    # Frozen embed + lm_head + final_norm (use base model directly)
    model.embed = p1_model.embed
    model.final_norm = p1_model.final_norm
    model.lm_head = p1_model.lm_head
    for p in model.embed.parameters():
        p.requires_grad = False
    for p in model.final_norm.parameters():
        p.requires_grad = False
    for p in model.lm_head.parameters():
        p.requires_grad = False

    # Build PressureFlip blocks
    model.layers = nn.ModuleList()
    for layer in p1_model.layers:
        block = PressureFlipBlock(
            cfg,
            base_attn=layer.attention,
            base_mlp=layer.mlp,
            base_norms=(layer.norm_attn, layer.norm_mlp),
            pruned_layers=pd,
            device=device,
        )
        model.layers.append(block)

    # Causal mask
    model.register_buffer(
        "causal_mask",
        torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len)).unsqueeze(0).unsqueeze(0),
        persistent=False,
    )

    # Collect PF modules
    model.pf_modules = []
    for name, mod in model.named_modules():
        if isinstance(mod, PressureFlipLinear):
            model.pf_modules.append((name, mod))

    n_params = sum(m.ternary.numel() for _, m in model.pf_modules)
    n_alive = sum(
        int(m.weight_mask.sum().item()) if m.weight_mask is not None else m.ternary.numel()
        for _, m in model.pf_modules
    )
    print(f"PressureFlip model: {len(model.pf_modules)} modules, "
          f"{n_params/1e3:.1f}k params, {n_alive/1e3:.1f}k alive, "
          f"memory: {n_params * 2 / 1024:.0f}KB (ternary+pressure)")

    # Attach forward method
    def forward(self, input_ids, labels=None):
        B, S = input_ids.shape
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size), labels.view(-1), ignore_index=-100)
        return out

    import types
    model.forward = types.MethodType(forward, model)

    return model.to(device)


class PressureFlipTrainer:
    """Trainer for PressureFlip models.

    Each step: forward → backward → accumulate sign(grad) → flip on threshold.
    No optimizer. The pressure buffer IS the optimization state.
    """

    def __init__(
        self,
        model,
        *,
        threshold_start: int = 5,
        threshold_end: int = 30,
        threshold_schedule: str = "cosine",
        scale_lr: float = 0.0,
        total_steps: int = 5000,
        grad_clip: float = 1.0,
    ):
        self.model = model
        self.threshold_start = threshold_start
        self.threshold_end = threshold_end
        self.threshold_schedule = threshold_schedule
        self.scale_lr = scale_lr
        self.total_steps = total_steps
        self.grad_clip = grad_clip
        self._step_count = 0

        print(f"PressureFlip trainer: threshold {threshold_start}->{threshold_end} "
              f"({threshold_schedule}), scale_lr={scale_lr}")

    def _threshold(self) -> int:
        progress = self._step_count / max(1, self.total_steps)
        if self.threshold_schedule == "constant":
            return self.threshold_start
        elif self.threshold_schedule == "linear":
            t = self.threshold_start + (self.threshold_end - self.threshold_start) * progress
        else:
            t = self.threshold_start + 0.5 * (self.threshold_end - self.threshold_start) * (
                1 - math.cos(math.pi * progress))
        return int(t)

    def step(self, x: Tensor, y: Tensor, device: torch.device) -> dict:
        """One training step."""
        self._step_count += 1
        threshold = self._threshold()

        x, y = x.to(device), y.to(device)

        # Forward
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.model(x, labels=y)
            loss = out["loss"]

        # Backward — populates .grad on each PressureFlipLinear._w_leaf
        loss.backward()

        # Pressure update for all modules
        total_flipped = 0
        total_pressure = 0.0
        n_mods = 0

        for name, mod in self.model.pf_modules:
            stats = mod.pressure_update(threshold=threshold, scale_lr=self.scale_lr)
            total_flipped += stats["n_flipped"]
            total_pressure += stats["mean_pressure"]
            n_mods += 1

        return {
            "loss": loss.item(),
            "threshold": threshold,
            "n_flipped": total_flipped,
            "mean_pressure": total_pressure / max(1, n_mods),
        }
