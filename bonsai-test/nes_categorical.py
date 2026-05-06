"""NES-Categorical: gradient-free ternary weight distribution optimization.

Maintains per-weight categorical distribution over {-1, 0, 1} with 2-DOF
logit parameterization: theta_neg, theta_pos (theta_zero = 0 fixed).

Gradient signal via score-function estimator (REINFORCE-style) with:
- Block-coordinate cycling: partition into G groups, cycle through
- SGD + momentum on logit parameters
- CDF-inversion antithetical sampling (u and 1-u)
- Fitness shaping (rank-based utilities)
- Cosine temperature schedule (exploration -> exploitation)
- Optional natural gradient (advantage form, Fisher is free)

Memory: 8P bytes with momentum (2 logits + 2 momentum in fp32)
vs AdamW's 13P bytes. No optimizer state needed.

Works with EggrollTransformer (explicit ternary/scales buffers).
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from kernels.eggroll_fast import chunked_cross_entropy


class NESCategoricalTrainer:
    """NES-Categorical optimizer for ternary weight distributions.

    Experiment configurations (from NES-Ternary-Experiment-Plan.md):
      A1: cycling='full',       pop=2048, momentum=0.0  (full-space NES)
      A2: cycling='full',       pop=2048, momentum=0.9  (+ momentum)
      A3: cycling='per_matrix', pop=256,  momentum=0.0  (cycling, no mom)
      A4: cycling='per_matrix', pop=256,  momentum=0.9  (main hypothesis)
      A5: cycling='per_layer',  pop=512,  momentum=0.9  (coarser groups)
      A6: cycling='per_matrix', pop=256,  momentum=0.9, natural_gradient=True
    """

    def __init__(
        self,
        model,
        *,
        pop_size: int = 256,
        momentum: float = 0.9,
        lr: float = 0.1,
        tau_start: float = 2.0,
        tau_end: float = 0.05,
        total_steps: int = 5000,
        cycling: str = "per_matrix",
        natural_gradient: bool = False,
        fitness_shaping: bool = True,
        baseline_ema: float = 0.99,
    ):
        self.model = model
        self.device = next(
            (p.device for p in model.parameters()), torch.device("cuda")
        )

        assert pop_size % 2 == 0, "pop_size must be even for antithetical pairs"
        self.pop_size = pop_size
        self.mom_coeff = momentum
        self.lr = lr
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.total_steps = total_steps
        self.natural_gradient = natural_gradient
        self.fitness_shaping = fitness_shaping
        self.baseline_ema_coeff = baseline_ema

        # Collect trainable modules (skip embed/lm_head)
        self.modules: list[tuple[str, object]] = []
        self._mod_id_set: set[int] = set()
        for name, mod, _shape in model._es_modules:
            if mod is model.embed or mod is model.lm_head:
                continue
            self.modules.append((name, mod))
            self._mod_id_set.add(id(mod))

        n_total = sum(m.ternary.numel() for _, m in self.modules)
        n_alive = sum(
            int(m.weight_mask.sum().item()) if m.weight_mask is not None else m.ternary.numel()
            for _, m in self.modules
        )

        # Initialize logits peaked at current ternary values.
        # Use moderate peaking (theta=2) so score-function gradient
        # has exploration room. At tau=2.0: p_peak ≈ 0.67, exploration ≈ 33%.
        # Temperature schedule sharpens to near-deterministic by end of training.
        INIT_PEAK = 2.0

        self.logit_neg: list[Tensor] = []
        self.logit_pos: list[Tensor] = []
        self.mom_neg: list[Tensor] = []
        self.mom_pos: list[Tensor] = []

        for _name, mod in self.modules:
            t = mod.ternary
            th_neg = torch.full(t.shape, -INIT_PEAK, dtype=torch.float32, device=self.device)
            th_pos = torch.full(t.shape, -INIT_PEAK, dtype=torch.float32, device=self.device)
            th_neg[t == -1] = INIT_PEAK
            th_pos[t == 1] = INIT_PEAK

            if mod.weight_mask is not None:
                dead = ~mod.weight_mask.bool()
                th_neg[dead] = 0.0
                th_pos[dead] = 0.0

            self.logit_neg.append(th_neg)
            self.logit_pos.append(th_pos)
            self.mom_neg.append(torch.zeros_like(th_neg))
            self.mom_pos.append(torch.zeros_like(th_pos))

        # Cycling groups
        if cycling == "per_matrix":
            self.group_indices = [[i] for i in range(len(self.modules))]
        elif cycling == "per_layer":
            n_layers = model.cfg.n_layers
            per_layer = len(self.modules) // n_layers
            self.group_indices = [
                list(range(l * per_layer, (l + 1) * per_layer))
                for l in range(n_layers)
            ]
        elif cycling == "full":
            self.group_indices = [list(range(len(self.modules)))]
        else:
            raise ValueError(f"Unknown cycling: {cycling}")

        print(
            f"NES-Cat: {len(self.modules)} modules, {n_total/1e3:.1f}k total, "
            f"{n_alive/1e3:.1f}k alive, {len(self.group_indices)} groups"
        )
        print(
            f"  pop={pop_size}, mom={momentum}, lr={lr}, "
            f"tau={tau_start:.1f}->{tau_end:.2f}, "
            f"{'natural' if natural_gradient else 'score-fn'} gradient"
        )

        self._step_count = 0
        self._baseline = None

    # ------------------------------------------------------------------
    # Temperature schedule
    # ------------------------------------------------------------------

    def _tau(self) -> float:
        progress = self._step_count / max(1, self.total_steps)
        return self.tau_end + 0.5 * (self.tau_start - self.tau_end) * (
            1 + math.cos(math.pi * progress)
        )

    # ------------------------------------------------------------------
    # Distribution utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _probs(theta_neg: Tensor, theta_pos: Tensor, tau: float):
        """Categorical probs from logits with temperature.
        Returns (p_neg, p_zero, p_pos) each same shape as theta_neg.
        """
        logits = torch.stack(
            [theta_neg / tau, torch.zeros_like(theta_neg), theta_pos / tau], dim=-1
        )
        p = F.softmax(logits, dim=-1)
        return p[..., 0], p[..., 1], p[..., 2]

    @staticmethod
    def _cdf_inv(p_neg: Tensor, p_zero: Tensor, u: Tensor) -> Tensor:
        """CDF inversion: u ~ Uniform(0,1) -> ternary {-1, 0, 1}."""
        result = torch.ones_like(u, dtype=torch.int8)
        result[u < (p_neg + p_zero)] = 0
        result[u < p_neg] = -1
        return result

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample(self, group_mod_indices: list[int], tau: float):
        """Sample N population members via CDF-inversion + antithetical pairs.

        Returns:
            all_samples: list of (N, out, in) int8 per active module
            all_u:       list of (N_half, out, in) float32 uniform samples
        """
        N_half = self.pop_size // 2
        all_samples = []
        all_u = []

        for mod_idx in group_mod_indices:
            _name, mod = self.modules[mod_idx]
            th_neg = self.logit_neg[mod_idx]
            th_pos = self.logit_pos[mod_idx]
            p_neg, p_zero, _p_pos = self._probs(th_neg, th_pos, tau)

            u = torch.rand(N_half, *th_neg.shape, device=self.device)
            w_pos = self._cdf_inv(
                p_neg.unsqueeze(0), p_zero.unsqueeze(0), u
            )
            w_neg = self._cdf_inv(
                p_neg.unsqueeze(0), p_zero.unsqueeze(0), 1.0 - u
            )
            samples = torch.cat([w_pos, w_neg], dim=0)  # (N, out, in)

            if mod.weight_mask is not None:
                samples = samples * mod.weight_mask.unsqueeze(0).to(torch.int8)

            all_samples.append(samples)
            all_u.append(u)

        return all_samples, all_u

    # ------------------------------------------------------------------
    # Gradient computation
    # ------------------------------------------------------------------

    def _score_fn_grad(
        self, advantages: Tensor, samples: Tensor, mod_idx: int, tau: float
    ) -> tuple[Tensor, Tensor]:
        """Score function gradient: nabla_theta = E[advantage * (1_{w=k} - p_k)]."""
        th_neg = self.logit_neg[mod_idx]
        th_pos = self.logit_pos[mod_idx]
        p_neg, _p_zero, p_pos = self._probs(th_neg, th_pos, tau)

        ind_neg = (samples == -1).float()
        ind_pos = (samples == 1).float()
        adv_sum = advantages.sum()

        grad_neg = torch.einsum("n,noi->oi", advantages, ind_neg) - p_neg * adv_sum
        grad_pos = torch.einsum("n,noi->oi", advantages, ind_pos) - p_pos * adv_sum
        grad_neg /= self.pop_size
        grad_pos /= self.pop_size

        _name, mod = self.modules[mod_idx]
        if mod.weight_mask is not None:
            dead = ~mod.weight_mask.bool()
            grad_neg[dead] = 0.0
            grad_pos[dead] = 0.0

        return grad_neg, grad_pos

    def _natural_grad(
        self, losses: Tensor, samples: Tensor, mod_idx: int, _tau: float
    ) -> tuple[Tensor, Tensor]:
        """Natural gradient: nabla_tilde_theta_k = E[L] - E[L|w=k]."""
        mean_loss = losses.mean()

        mask_neg = (samples == -1).float()
        mask_pos = (samples == 1).float()

        count_neg = mask_neg.sum(dim=0).clamp(min=1)
        count_pos = mask_pos.sum(dim=0).clamp(min=1)

        avg_neg = torch.einsum("n,noi->oi", losses, mask_neg) / count_neg
        avg_pos = torch.einsum("n,noi->oi", losses, mask_pos) / count_pos

        grad_neg = mean_loss - avg_neg
        grad_pos = mean_loss - avg_pos

        grad_neg[mask_neg.sum(dim=0) == 0] = 0
        grad_pos[mask_pos.sum(dim=0) == 0] = 0

        _name, mod = self.modules[mod_idx]
        if mod.weight_mask is not None:
            dead = ~mod.weight_mask.bool()
            grad_neg[dead] = 0.0
            grad_pos[dead] = 0.0

        return grad_neg, grad_pos

    def _fitness_shape(self, losses: Tensor) -> Tensor:
        """Rank-based fitness shaping utilities.
        Returns (N,) advantages: positive = good (low loss).
        """
        N = losses.shape[0]
        ranks = losses.argsort().argsort().float()
        log_base = math.log(N / 2 + 1)
        utilities = torch.clamp(log_base - torch.log(ranks + 1), min=0)
        utilities /= utilities.sum() + 1e-8
        utilities -= 1.0 / N
        return utilities * N

    # ------------------------------------------------------------------
    # Set model weights to distribution argmax
    # ------------------------------------------------------------------

    def _set_all_argmax(self, tau: float):
        for i, (_name, mod) in enumerate(self.modules):
            p_neg, p_zero, p_pos = self._probs(
                self.logit_neg[i], self.logit_pos[i], tau
            )
            probs = torch.stack([p_neg, p_zero, p_pos], dim=-1)
            argmax_idx = probs.argmax(dim=-1)
            ternary = (argmax_idx - 1).to(torch.int8)
            if mod.weight_mask is not None:
                ternary = ternary * mod.weight_mask.to(torch.int8)
            mod.ternary = ternary

    # ------------------------------------------------------------------
    # Batched forward pass
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _batched_forward(
        self,
        x: Tensor,
        y: Tensor,
        active_mod_indices: list[int],
        all_samples: list[Tensor],
    ) -> Tensor:
        """Forward N population members with per-member weights for active modules.

        Inactive modules use shared (argmax) weights. Active modules use
        per-member sampled ternary via batched matmul.

        Returns: (N,) per-member losses.
        """
        model = self.model
        N = all_samples[0].shape[0]
        B, S = x.shape

        # Pre-compute per-member effective weights for active modules
        active_weights: dict[int, Tensor] = {}
        for mod_idx, samples in zip(active_mod_indices, all_samples):
            _name, mod = self.modules[mod_idx]
            gs = mod.group_size
            n_groups = mod.ternary.numel() // gs
            scales_exp = (
                mod.scales.unsqueeze(1)
                .expand(n_groups, gs)
                .reshape(mod.ternary.shape)
            )
            w_eff = samples.float() * scales_exp.unsqueeze(0)
            if mod.weight_mask is not None:
                w_eff = w_eff * mod.weight_mask.unsqueeze(0)
            active_weights[id(mod)] = w_eff.to(torch.bfloat16)

        return self._batched_forward_inner(
            x, y, active_weights, N, B, S
        )

    @torch.no_grad()
    @torch.autocast("cuda", dtype=torch.bfloat16)
    def _batched_forward_inner(self, x, y, active_weights, N, B, S):
        model = self.model

        # Expand input across population
        x_pop = x.unsqueeze(0).expand(N, -1, -1).reshape(N * B, S)
        y_pop = y.unsqueeze(0).expand(N, -1, -1).reshape(N * B, S)

        # Embedding (shared, no perturbation)
        w_embed = model.embed.get_weight()
        h = F.embedding(x_pop, w_embed)

        # Transformer layers
        for block in model.layers:
            normed = block.norm_attn(h)
            attn = block.attention
            H, Dh = attn.n_heads, attn.d_head

            q = self._linear_or_batched(attn.W_q, normed, active_weights, N, B)
            k = self._linear_or_batched(attn.W_k, normed, active_weights, N, B)
            v = self._linear_or_batched(attn.W_v, normed, active_weights, N, B)

            q = q.reshape(N * B, S, H, Dh).transpose(1, 2)
            k = k.reshape(N * B, S, H, Dh).transpose(1, 2)
            v = v.reshape(N * B, S, H, Dh).transpose(1, 2)

            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

            x_heads = normed.reshape(N * B, S, H, Dh).transpose(1, 2)
            attn_out = attn_out + attn.alpha * x_heads
            attn_out = attn_out.transpose(1, 2).reshape(N * B, S, -1)

            attn_out = self._linear_or_batched(
                attn.W_o, attn_out, active_weights, N, B
            )
            h = h + attn_out

            normed = block.norm_mlp(h)
            gate = F.silu(
                self._linear_or_batched(
                    block.mlp.w_gate, normed, active_weights, N, B
                )
            )
            up = self._linear_or_batched(
                block.mlp.w_up, normed, active_weights, N, B
            )
            down = self._linear_or_batched(
                block.mlp.w_down, gate * up, active_weights, N, B
            )
            h = h + down

        h = model.final_norm(h)

        # LM head (shared) + chunked cross-entropy
        w_lm = model.lm_head.get_weight()
        h_flat = h.reshape(N * B * S, -1)
        y_flat = y_pop.reshape(-1)

        M = N * B * S
        max_ce_bytes = 1 * 1024**3
        v_chunk = min(4096, max(256, int(max_ce_bytes / (M * 4))))

        per_token = chunked_cross_entropy(h_flat, w_lm, y_flat, v_chunk=v_chunk)
        return per_token.reshape(N, B * S).mean(dim=1)

    def _linear_or_batched(
        self,
        mod,
        x: Tensor,
        active_weights: dict[int, Tensor],
        N: int,
        B: int,
    ) -> Tensor:
        """Shared-weight linear or per-member batched matmul."""
        if id(mod) in active_weights:
            S = x.shape[1]
            w_eff = active_weights[id(mod)]
            x_r = x.reshape(N, B, S, -1)
            out = torch.einsum("nbsi,nio->nbso", x_r, w_eff.transpose(1, 2))
            return out.reshape(N * B, S, -1)
        else:
            return F.linear(x, mod.get_weight())

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, x: Tensor, y: Tensor) -> dict:
        """One NES-Categorical step with block-coordinate cycling.

        1. Pick active group (cycling)
        2. Set all modules to argmax of distributions
        3. Sample population for active group (CDF + antithetical)
        4. Batched forward to get per-member losses
        5. Compute gradient (score-function or natural)
        6. Update logits with momentum (gradient ascent on fitness)
        7. Set ternary to updated argmax

        Returns dict with loss, tau, group info.
        """
        self._step_count += 1
        tau = self._tau()
        group_idx = (self._step_count - 1) % len(self.group_indices)
        active_mod_indices = self.group_indices[group_idx]

        self._set_all_argmax(tau)

        all_samples, _all_u = self._sample(active_mod_indices, tau)

        losses = self._batched_forward(x, y, active_mod_indices, all_samples)

        mean_loss = losses.mean().item()

        if self._baseline is None:
            self._baseline = mean_loss
        else:
            self._baseline = (
                self.baseline_ema_coeff * self._baseline
                + (1 - self.baseline_ema_coeff) * mean_loss
            )

        if self.fitness_shaping:
            advantages = self._fitness_shape(losses)
        else:
            advantages = self._baseline - losses

        for i, mod_idx in enumerate(active_mod_indices):
            samples = all_samples[i]

            if self.natural_gradient:
                g_neg, g_pos = self._natural_grad(losses, samples, mod_idx, tau)
            else:
                g_neg, g_pos = self._score_fn_grad(
                    advantages, samples, mod_idx, tau
                )

            self.mom_neg[mod_idx].mul_(self.mom_coeff).add_(g_neg)
            self.mom_pos[mod_idx].mul_(self.mom_coeff).add_(g_pos)

            # Gradient ascent on fitness (positive grad -> increase probability)
            self.logit_neg[mod_idx].add_(self.lr * self.mom_neg[mod_idx])
            self.logit_pos[mod_idx].add_(self.lr * self.mom_pos[mod_idx])

        self._set_all_argmax(tau)

        return {
            "loss": mean_loss,
            "tau": tau,
            "group": group_idx,
            "group_name": ",".join(
                self.modules[i][0] for i in active_mod_indices
            ),
            "min_loss": losses.min().item(),
            "max_loss": losses.max().item(),
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def distribution_stats(self) -> dict:
        """Return current distribution statistics for logging."""
        tau = self._tau()
        total_entropy = 0.0
        total_params = 0
        total_changed = 0

        for i, (_name, mod) in enumerate(self.modules):
            p_neg, p_zero, p_pos = self._probs(
                self.logit_neg[i], self.logit_pos[i], tau
            )
            probs = torch.stack([p_neg, p_zero, p_pos], dim=-1)

            entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)

            if mod.weight_mask is not None:
                alive = mod.weight_mask.bool()
                entropy = entropy[alive]
                n_alive = alive.sum().item()
            else:
                n_alive = entropy.numel()

            total_entropy += entropy.sum().item()
            total_params += n_alive

        return {
            "mean_entropy": total_entropy / max(1, total_params),
            "n_params": total_params,
            "tau": tau,
        }
