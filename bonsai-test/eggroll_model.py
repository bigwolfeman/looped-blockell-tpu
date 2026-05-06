"""EGGROLL: Evolution Strategies for Ternary Training.

Pure ES — no backprop, no gradients, no STE. Every parameter is updated
via population-based search with low-rank perturbations.

Based on "Evolution Strategies at the Hyperscale" (arXiv:2511.16652).
Their EGG (int8-only RNN) BEATS backprop at 3.40 vs 3.58 bits/byte.

Algorithm per step:
  1. Sample N rank-1 perturbations per layer from seeds
  2. Forward pass each member (shared base + rank-1 correction)
  3. Compute scalar fitness = -loss for each member
  4. ES gradient: delta_W = sum(centered_fitness * A_i @ B_i^T) / (N * sigma)
  5. Apply to ternary: W += sign(delta_W) where |delta_W| > threshold

No optimizer state. No shadow weights. No backward pass.
Memory: just the model weights (ternary + scales).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from bitlinear import RMSNorm


class EggrollLinear(nn.Module):
    """Linear layer updated via EGGROLL ES."""

    def __init__(self, in_features: int, out_features: int, group_size: int = 128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        total = out_features * in_features
        if total % group_size != 0:
            self.group_size = self._find_group_size(total, group_size)

        init = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(init, a=5 ** 0.5)
        beta = init.reshape(-1, self.group_size).abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
        ternary = (init.reshape(-1, self.group_size) / beta).round().clamp(-1, 1)

        self.register_buffer("ternary", ternary.reshape(out_features, in_features).to(torch.int8))
        self.register_buffer("scales", beta.squeeze(1))
        # weight_mask: 1=alive, 0=pruned (set externally after pruning; None = no mask)
        self.register_buffer("weight_mask", None)

    @staticmethod
    def _find_group_size(total: int, target: int) -> int:
        for gs in range(target, 0, -1):
            if total % gs == 0:
                return gs
        return 1

    def get_weight(self) -> Tensor:
        gs = self.group_size
        n_groups = self.ternary.numel() // gs
        s = self.scales.unsqueeze(1).expand(n_groups, gs).reshape(self.out_features, self.in_features)
        w = self.ternary.float() * s
        if self.weight_mask is not None:
            w = w * self.weight_mask
        return w

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.get_weight())

    def forward_perturbed(self, x: Tensor, a: Tensor, b: Tensor, sigma: float) -> Tensor:
        """Forward with rank-1 perturbation: y = x @ W^T + sigma * (x @ b) @ a^T"""
        base = F.linear(x, self.get_weight())
        # a: (out_features,), b: (in_features,)
        # Mask perturbation vectors so dead weights contribute zero
        a_eff = a * self._get_out_mask() if self.weight_mask is not None else a
        b_eff = b * self._get_in_mask() if self.weight_mask is not None else b
        correction = sigma * (x @ b_eff).unsqueeze(-1) * a_eff.unsqueeze(0).unsqueeze(0)
        return base + correction

    def _get_out_mask(self) -> Tensor:
        """Per-output-row alive indicator: row is alive if any weight in it is alive."""
        if self.weight_mask is None:
            return None
        return self.weight_mask.any(dim=1).float()

    def _get_in_mask(self) -> Tensor:
        """Per-input-col alive indicator: col is alive if any weight in it is alive."""
        if self.weight_mask is None:
            return None
        return self.weight_mask.any(dim=0).float()

    def es_update(self, es_grad: Tensor, alpha: float) -> dict:
        """Apply ES gradient via thresholded ±1 steps (paper's int8 recipe).

        alpha ≈ fraction of weights to update per step.
        Weights move by ±1 in ternary space where evidence is strongest.
        Dead weights (mask==0) are never updated.
        """
        # If mask exists, zero out gradient at dead positions so they never rank highly
        if self.weight_mask is not None:
            es_grad = es_grad * self.weight_mask

        # Threshold: top alpha fraction by |es_grad| get updated
        flat_grad = es_grad.reshape(-1)
        n_total_weights = flat_grad.numel()
        n_update = max(1, min(n_total_weights, int(alpha * n_total_weights)))
        k = n_total_weights - n_update  # rank of the smallest "keep" value
        if k < 1:
            # Update all weights — threshold is effectively -inf
            threshold = -1e38
        else:
            threshold = flat_grad.abs().kthvalue(k).values.item()

        # Apply ±1 steps where |es_grad| > threshold (and mask==1 if mask exists)
        do_update = es_grad.abs() > threshold
        if self.weight_mask is not None:
            do_update = do_update & (self.weight_mask.bool())
        direction = es_grad.sign().to(torch.int8)

        old = self.ternary.clone()
        new_ternary = (self.ternary + direction * do_update.to(torch.int8)).clamp(-1, 1)

        changed = new_ternary != old
        was_zero = old == 0
        is_zero = new_ternary == 0
        n_sign = (changed & ~was_zero & ~is_zero).sum().item()
        n_structural = (changed & (was_zero | is_zero)).sum().item()

        self.ternary = new_ternary

        # Update scales: move in direction of es_grad for alive weights
        gs = self.group_size
        grad_scale = (es_grad * self.ternary.float()).reshape(-1, gs).mean(dim=1)
        self.scales.add_(grad_scale, alpha=0.001)
        self.scales.clamp_(min=1e-6)

        return {"n_flips": n_sign + n_structural, "n_sign": n_sign, "n_structural": n_structural}


class EggrollEmbedding(nn.Module):
    """Embedding updated via EGGROLL ES."""

    def __init__(self, num_embeddings: int, embedding_dim: int, group_size: int = 128):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.group_size = group_size

        total = num_embeddings * embedding_dim
        if total % group_size != 0:
            self.group_size = EggrollLinear._find_group_size(total, group_size)

        init = torch.empty(num_embeddings, embedding_dim)
        nn.init.normal_(init, std=0.02)
        beta = init.reshape(-1, self.group_size).abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
        ternary = (init.reshape(-1, self.group_size) / beta).round().clamp(-1, 1)

        self.register_buffer("ternary", ternary.reshape(num_embeddings, embedding_dim).to(torch.int8))
        self.register_buffer("scales", beta.squeeze(1))
        # weight_mask: same semantics as EggrollLinear (None = no mask)
        self.register_buffer("weight_mask", None)

    def get_weight(self) -> Tensor:
        gs = self.group_size
        n_groups = self.ternary.numel() // gs
        s = self.scales.unsqueeze(1).expand(n_groups, gs).reshape(self.num_embeddings, self.embedding_dim)
        w = self.ternary.float() * s
        if self.weight_mask is not None:
            w = w * self.weight_mask
        return w

    def forward(self, input_ids: Tensor) -> Tensor:
        return F.embedding(input_ids, self.get_weight())

    def es_update(self, es_grad: Tensor, alpha: float) -> dict:
        """Thresholded ±1 steps — same algorithm as EggrollLinear."""
        if self.weight_mask is not None:
            es_grad = es_grad * self.weight_mask

        flat_grad = es_grad.reshape(-1)
        n_total = flat_grad.numel()
        n_update = max(1, min(n_total, int(alpha * n_total)))
        k = max(1, n_total - n_update)
        threshold = flat_grad.abs().kthvalue(k).values.item()

        do_update = es_grad.abs() > threshold
        if self.weight_mask is not None:
            do_update = do_update & self.weight_mask.bool()
        direction = es_grad.sign().to(torch.int8)

        old = self.ternary.clone()
        new_ternary = (self.ternary + direction * do_update.to(torch.int8)).clamp(-1, 1)

        changed = new_ternary != old
        was_zero = old == 0
        is_zero = new_ternary == 0
        n_sign = (changed & ~was_zero & ~is_zero).sum().item()
        n_structural = (changed & (was_zero | is_zero)).sum().item()

        self.ternary = new_ternary

        gs = self.group_size
        grad_scale = (es_grad * self.ternary.float()).reshape(-1, gs).mean(dim=1)
        self.scales.add_(grad_scale, alpha=0.001)
        self.scales.clamp_(min=1e-6)

        return {"n_flips": n_sign + n_structural, "n_sign": n_sign, "n_structural": n_structural}


class EggrollAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.scale = self.d_head ** -0.5

        self.W_q = EggrollLinear(cfg.d_model, cfg.d_model, group_size=cfg.group_size)
        self.W_k = EggrollLinear(cfg.d_model, cfg.d_model, group_size=cfg.group_size)
        self.W_v = EggrollLinear(cfg.d_model, cfg.d_model, group_size=cfg.group_size)
        self.W_o = EggrollLinear(cfg.d_model, cfg.d_model, group_size=cfg.group_size)

        self.register_buffer("alpha", torch.full((cfg.n_heads, 1, 1), 0.1))

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        B, S, D = x.shape
        H, Dh = self.n_heads, self.d_head

        q = self.W_q(x).reshape(B, S, H, Dh).transpose(1, 2)
        k = self.W_k(x).reshape(B, S, H, Dh).transpose(1, 2)
        v = self.W_v(x).reshape(B, S, H, Dh).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        x_heads = x.reshape(B, S, H, Dh).transpose(1, 2)
        out = out + self.alpha * x_heads
        out = out.transpose(1, 2).reshape(B, S, D)
        return self.W_o(out)


class EggrollMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w_gate = EggrollLinear(cfg.d_model, cfg.d_ff, group_size=cfg.group_size)
        self.w_up = EggrollLinear(cfg.d_model, cfg.d_ff, group_size=cfg.group_size)
        self.w_down = EggrollLinear(cfg.d_ff, cfg.d_model, group_size=cfg.group_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class EggrollBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm_attn = RMSNorm(cfg.d_model)
        self.attention = EggrollAttention(cfg)
        self.norm_mlp = RMSNorm(cfg.d_model)
        self.mlp = EggrollMLP(cfg)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = x + self.attention(self.norm_attn(x), mask)
        x = x + self.mlp(self.norm_mlp(x))
        return x


class EggrollConfig:
    def __init__(self, **kwargs):
        self.d_model = kwargs.get("d_model", 512)
        self.n_heads = kwargs.get("n_heads", 8)
        self.d_ff = kwargs.get("d_ff", 1376)
        self.n_layers = kwargs.get("n_layers", 6)
        self.vocab_size = kwargs.get("vocab_size", 49152)
        self.max_seq_len = kwargs.get("max_seq_len", 1024)
        self.group_size = kwargs.get("group_size", 128)
        self.pop_size = kwargs.get("pop_size", 64)
        self.sigma = kwargs.get("sigma", 0.001)
        self.es_lr = kwargs.get("es_lr", 0.01)


class EggrollTransformer(nn.Module):
    """Ternary transformer trained purely via Evolution Strategies.

    No backprop. No gradients. No STE. No shadow weights.
    Just perturbation → evaluation → statistical credit assignment.

    Training loop:
      1. Sample N rank-1 perturbations (seeded, never stored)
      2. Forward pass each member: base + correction
      3. Fitness = -loss per member
      4. ES gradient per layer = sum(centered_fitness * outer(a_i, b_i))
      5. Apply to ternary via rounding
    """

    def __init__(self, cfg: EggrollConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = EggrollEmbedding(cfg.vocab_size, cfg.d_model, group_size=cfg.group_size)
        self.layers = nn.ModuleList([EggrollBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = EggrollLinear(cfg.d_model, cfg.vocab_size, group_size=cfg.group_size)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len)).unsqueeze(0).unsqueeze(0),
            persistent=False,
        )

        # Collect all ES-updated modules and their shapes for perturbation
        self._es_modules = []
        for name, m in self.named_modules():
            if isinstance(m, (EggrollLinear, EggrollEmbedding)):
                shape = (m.ternary.shape[0], m.ternary.shape[1])
                self._es_modules.append((name, m, shape))

        n_params = sum(m.ternary.numel() for _, m, _ in self._es_modules)
        print(f"EggrollTransformer: {n_params/1e6:.1f}M ternary params, "
              f"pop_size={cfg.pop_size}, sigma={cfg.sigma}, "
              f"NO backprop, NO gradients")

    def forward(self, input_ids: Tensor, labels: Tensor | None = None) -> dict:
        B, S = input_ids.shape
        x = self.embed(input_ids)
        mask = self.causal_mask[:, :, :S, :S]
        for layer in self.layers:
            x = layer(x, mask)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        out = {"logits": logits}
        if labels is not None:
            out["loss"] = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size), labels.view(-1), ignore_index=-100)
        return out

    @torch.no_grad()
    def es_step(self, x: Tensor, y: Tensor, step: int = 0) -> dict:
        """Vectorized EGGROLL step with antithetical sampling and fitness shaping.

        Uses mirrored pairs: member i uses +E, member i+N/2 uses -E.
        Fitness = sign(loss_minus - loss_plus) ∈ {-1, 0, +1} (ternary shaping).
        """
        N = self.cfg.pop_size
        assert N % 2 == 0, "pop_size must be even for antithetical pairs"
        N_half = N // 2
        sigma = self.cfg.sigma
        # Decaying alpha (fraction of params to flip): 1 / (0.015*t + 1)
        alpha = self.cfg.es_lr / (0.015 * step + 1)
        device = x.device
        B, S = x.shape

        # Generate perturbation vectors for N/2 base directions (antithetical doubles them)
        layer_perturbations = []
        for name, mod, (rows, cols) in self._es_modules:
            A = torch.randn(N_half, rows, device=device)
            Bmat = torch.randn(N_half, cols, device=device)
            layer_perturbations.append((A, Bmat))

        # Evaluate in chunks to fit in VRAM
        # Attention is O(N*B*H*S^2) — chunk population to control memory
        chunk = min(N_half, max(1, 24 * 1024 // (B * S)))  # ~24GB budget for attention

        losses_pos = self._eval_population_chunked(x, y, layer_perturbations, sigma, N_half, B, S, chunk)
        losses_neg = self._eval_population_chunked(x, y, layer_perturbations, -sigma, N_half, B, S, chunk)

        # Ternary fitness shaping: sign(loss_neg - loss_pos)
        # Positive fitness = positive perturbation was better (lower loss)
        fitness = torch.sign(losses_neg - losses_pos)  # (N_half,) ∈ {-1, 0, +1}

        # Compute ES gradient and apply updates
        total_stats = {"n_flips": 0, "n_sign": 0, "n_structural": 0}

        for (name, mod, (rows, cols)), (A, Bmat) in zip(self._es_modules, layer_perturbations):
            # ES gradient from antithetical pairs with ternary fitness
            es_grad = (A * fitness.unsqueeze(1)).T @ Bmat  # (rows, cols)
            es_grad.div_(N * sigma)  # N = full pop (antithetical pairs counted)

            stats = mod.es_update(es_grad, alpha)
            total_stats["n_flips"] += stats["n_flips"]
            total_stats["n_sign"] += stats["n_sign"]
            total_stats["n_structural"] += stats["n_structural"]

        return total_stats

    def _eval_population_chunked(self, x: Tensor, y: Tensor,
                                layer_perturbations: list, sigma: float,
                                N: int, B: int, S: int, chunk: int) -> Tensor:
        """Evaluate population in chunks to control VRAM usage."""
        device = x.device
        all_losses = []
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            C = end - start
            chunk_perts = [(A[start:end], Bm[start:end]) for A, Bm in layer_perturbations]
            losses = self._eval_population(x, y, chunk_perts, sigma, C, B, S)
            all_losses.append(losses)
        return torch.cat(all_losses)

    def _eval_population(self, x: Tensor, y: Tensor,
                         layer_perturbations: list, sigma: float,
                         N: int, B: int, S: int) -> Tensor:
        """Forward all N population members in one batched pass. Returns per-member losses."""
        device = x.device

        # Expand input: (N*B, S)
        x_pop = x.unsqueeze(0).expand(N, -1, -1).reshape(N * B, S)
        y_pop = y.unsqueeze(0).expand(N, -1, -1).reshape(N * B, S)

        # Embedding (no perturbation for now — embedding updates via scale only)
        w_embed = self.embed.get_weight()
        h = F.embedding(x_pop, w_embed)

        mask = self.causal_mask[:, :, :S, :S]
        layer_idx = 1  # skip embedding

        for block in self.layers:
            normed = block.norm_attn(h)
            attn = block.attention
            H, Dh = attn.n_heads, attn.d_head

            q = self._perturbed_linear(attn.W_q, normed, layer_perturbations[layer_idx], sigma, N, B)
            layer_idx += 1
            k = self._perturbed_linear(attn.W_k, normed, layer_perturbations[layer_idx], sigma, N, B)
            layer_idx += 1
            v = self._perturbed_linear(attn.W_v, normed, layer_perturbations[layer_idx], sigma, N, B)
            layer_idx += 1

            q = q.reshape(N * B, S, H, Dh).transpose(1, 2)
            k = k.reshape(N * B, S, H, Dh).transpose(1, 2)
            v = v.reshape(N * B, S, H, Dh).transpose(1, 2)

            scores = torch.matmul(q, k.transpose(-2, -1)) * attn.scale
            scores = scores.masked_fill(mask == 0, float('-inf'))
            scores = F.softmax(scores, dim=-1)
            attn_out = torch.matmul(scores, v)

            x_heads = normed.reshape(N * B, S, H, Dh).transpose(1, 2)
            attn_out = attn_out + attn.alpha * x_heads
            attn_out = attn_out.transpose(1, 2).reshape(N * B, S, -1)

            attn_out = self._perturbed_linear(attn.W_o, attn_out, layer_perturbations[layer_idx], sigma, N, B)
            layer_idx += 1

            h = h + attn_out

            normed = block.norm_mlp(h)
            gate = F.silu(self._perturbed_linear(block.mlp.w_gate, normed, layer_perturbations[layer_idx], sigma, N, B))
            layer_idx += 1
            up = self._perturbed_linear(block.mlp.w_up, normed, layer_perturbations[layer_idx], sigma, N, B)
            layer_idx += 1
            down = self._perturbed_linear(block.mlp.w_down, gate * up, layer_perturbations[layer_idx], sigma, N, B)
            layer_idx += 1

            h = h + down

        h = self.final_norm(h)
        logits = self._perturbed_linear(self.lm_head, h, layer_perturbations[layer_idx], sigma, N, B)

        # Per-member loss
        logits_flat = logits.reshape(N, B * S, -1)
        y_flat = y_pop.reshape(N, B * S)

        losses = torch.zeros(N, device=device)
        for i in range(N):
            losses[i] = F.cross_entropy(logits_flat[i], y_flat[i], ignore_index=-100)

        return losses

    def _perturbed_linear(self, mod: EggrollLinear, x: Tensor,
                          perturbation: tuple, sigma: float, N: int, B: int) -> Tensor:
        """Compute y = x @ W^T + sigma * (x @ b_i) * a_i^T, vectorized over population.

        x: (N*B, S, in_features)
        Returns: (N*B, S, out_features)

        When mod has a weight_mask, perturbation vectors are masked so dead weight
        positions contribute zero perturbation — preserving the rank-1 structure
        over the alive subspace.
        """
        A, Bmat = perturbation  # A: (N, out), B: (N, in)
        S = x.shape[1]

        # Apply mask to perturbation directions at dead positions
        if mod.weight_mask is not None:
            # out-mask: (out,) — row i is alive if any weight in row i is alive
            out_alive = mod.weight_mask.any(dim=1).float()   # (out,)
            in_alive  = mod.weight_mask.any(dim=0).float()   # (in,)
            A    = A    * out_alive.unsqueeze(0)   # (N, out)
            Bmat = Bmat * in_alive.unsqueeze(0)    # (N, in)

        # Base: shared across all members
        base = F.linear(x, mod.get_weight())  # (N*B, S, out)

        # Per-member rank-1 correction
        x_reshaped = x.reshape(N, B, S, -1)  # (N, B, S, in)
        # inner = x @ b_i for each member: (N, B, S, in) @ (N, in) -> (N, B, S)
        inner = torch.einsum('nbsi,ni->nbs', x_reshaped, Bmat)
        # outer = inner * a_i: (N, B, S) x (N, out) -> (N, B, S, out)
        correction = sigma * torch.einsum('nbs,no->nbso', inner, A)
        correction = correction.reshape(N * B, S, -1)

        return base + correction

    def ternary_stats(self) -> dict:
        all_t = torch.cat([m.ternary.flatten().float() for _, m, _ in self._es_modules])
        return {
            "neg_frac": (all_t == -1).float().mean().item(),
            "zero_frac": (all_t == 0).float().mean().item(),
            "pos_frac": (all_t == 1).float().mean().item(),
        }
