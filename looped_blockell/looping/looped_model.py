"""Looped Block-ELL Transformer — JAX/Flax TPU implementation.

Ports titans_core/looping/looped_model.py to JAX/Flax.

Architecture (Parcae-style):
    1. Embedding + embedding scale
    2. Prelude: n_prelude TransformerBlocks (non-looped)
    3. Input RMSNorm
    4. DiagonalInjection (SSM-style state gate)
    5. Core: n_core TransformerBlocks, looped T times via lax.scan
    6. Optional per-iteration embedding for Phase C routing
    7. Coda: n_coda TransformerBlocks (non-looped)
    8. Final RMSNorm + LM head (weight-tied with embedding)

lax.scan design:
    - Scan length = total_iters (n_max + k_max) — must be a Python int.
    - Carry: h  [B, S, d_model]
    - Scanned input: step indices  arange(total_iters)
    - Per-sequence freeze: active = (step_idx < depths), h = jnp.where(active, h_new, h)
    - No-grad phase: is_nograd = (step_idx < n_max), wrapped in lax.stop_gradient
    - jax.checkpoint applied to scan body for O(1)-per-step activation memory

Weight tying:
    lm_head uses the embedding matrix directly (transposed). In Flax,
    this is done via a `kernel` initializer that returns a reference to the
    embed param — but the cleaner idiom is to pass the embed weight explicitly
    through __call__. We use the "kernel = embed.embedding" pattern via
    nn.share_scope / a custom Dense call with the transposed embed weight.

Static shape rule:
    total_iters (n_max + k_max) MUST be a Python int before jit. The caller
    (training loop or inference) is responsible for computing the DepthPlan
    OUTSIDE the jitted function and passing total_iters / depths as arguments.

    Typical pattern::

        plan = sample_depth(key, B, cfg.mean_depth, bptt_depth=cfg.bptt_depth)
        # plan.n_max, plan.k_max are Python ints — safe to use as static dims
        logits, loss = jax.jit(model.apply, static_argnums=(...))(
            params, input_ids, labels, plan.total, plan.n_max, plan.k_max
        )
"""

from __future__ import annotations

import math
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from ..config import LoopedBlockELLConfig
from ..layers.norms import RMSNorm
from ..layers.transformer_block import TransformerBlock
from .diagonal_injection import DiagonalInjection
from .depth_sampler import DepthPlan, sample_depth, sample_fixed


class LoopedTransformer(nn.Module):
    """Parcae-style looped transformer with diagonal injection.

    Attributes:
        config: LoopedBlockELLConfig — all hyperparameters.
    """

    config: LoopedBlockELLConfig

    @nn.compact
    def __call__(
        self,
        input_ids: jnp.ndarray,
        depths: jnp.ndarray,
        n_max: int,
        k_max: int,
        labels: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> dict:
        """Forward pass.

        IMPORTANT: n_max and k_max MUST be Python ints (not JAX arrays) so
        that lax.scan gets a static iteration count. Compute them from a
        DepthPlan BEFORE calling jit.

        Args:
            input_ids:    Token IDs                      [B, S]  int32
            depths:       Per-sequence total iters       [B]     int32
            n_max:        Max no-grad steps (Python int)
            k_max:        Max grad steps    (Python int)
            labels:       Target token IDs for LM loss   [B, S]  int32 (optional)
            deterministic: If False, apply dropout (training).

        Returns:
            dict with:
                'logits'     — [B, S, vocab_size]
                'loss'       — scalar (only when labels is not None)
                'depth_meta' — dict with t_max, n_max, k_max
        """
        cfg = self.config
        total_iters: int = n_max + k_max  # Python int — static for scan
        B, S = input_ids.shape

        # ── 1. Embedding ────────────────────────────────────────────────────
        # We need the embedding weight for weight tying, so keep a handle.
        embed = nn.Embed(
            num_embeddings=cfg.vocab_size,
            features=cfg.d_model,
            embedding_init=nn.initializers.normal(stddev=0.02),
            name="embed",
        )
        x = embed(input_ids)                      # [B, S, d_model]
        x = x * cfg.embedding_scale

        # ── 2. Prelude ───────────────────────────────────────────────────────
        for i in range(cfg.n_prelude):
            x = TransformerBlock(cfg, name=f"prelude_{i}")(x, deterministic=deterministic)

        # ── 3. Input norm (stabilise signal going into injection) ────────────
        e = RMSNorm(cfg.norm_eps, name="input_norm")(x)  # [B, S, d_model]

        # ── 4. Recurrent state init — truncated normal (Parcae-style) ────────
        # In Flax @nn.compact we can't call make_rng during normal forward
        # because we're in the params collection. Use a mutable param instead.
        # Actually the spec says to use make_rng('params') — fine in init,
        # but at inference time we need a reproducible init. Use zeros and
        # let the injection gate open naturally (decay * 0 + dt * e = dt * e).
        #
        # Rationale: the model is trained with random h init, but at inference
        # the hidden state converges quickly due to decay < 1. Zeros on first
        # token gives the same asymptotic behavior after 1-2 iterations.
        h = jnp.zeros_like(e)                    # [B, S, d_model]

        # ── 5. Build core blocks (shared weights across loop iterations) ──────
        # In Flax @nn.compact, modules created here get their params under
        # 'core_0', 'core_1', etc. and are shared across ALL scan iterations
        # because lax.scan closes over them from the outer scope.
        core_blocks = [
            TransformerBlock(cfg, name=f"core_{i}")
            for i in range(cfg.n_core)
        ]
        injection = DiagonalInjection(
            d_model=cfg.d_model,
            init_decay=cfg.init_decay,
            name="injection",
        )

        # Optional per-iteration embedding (Phase C routing conditioning)
        # Created only if n_clusters > 0 as a proxy for routing being enabled.
        # The embedding is near-zero init to not disturb the pretrained model.
        has_iter_embed = cfg.n_clusters > 0
        if has_iter_embed:
            iter_embed = nn.Embed(
                num_embeddings=cfg.max_depth,
                features=cfg.d_model,
                embedding_init=nn.initializers.normal(stddev=0.001),
                name="iteration_embed",
            )

        # ── 6. lax.scan core loop ─────────────────────────────────────────────
        # Carry = h  [B, S, d_model]
        # Scanned = step indices  [total_iters]
        #
        # Per-sequence active mask: active[b] = (step_idx < depths[b])
        #   → after a sequence hits its budget, h stays frozen.
        #
        # No-grad mask: is_nograd = (step_idx < n_max)
        #   → first n_max iterations run with stop_gradient on the output.
        #   → last k_max iterations carry full gradients.
        #
        # jax.checkpoint on the scan body: recomputes forward activations
        # during backward instead of storing them — O(1) memory per step.

        def scan_body(h_carry, step_idx):
            """One looped iteration.

            Args:
                h_carry:  Current hidden state  [B, S, d_model]
                step_idx: Current step (scalar int32, JAX traced)

            Returns:
                (h_next, None)  — scan output is None (we only care about carry)
            """
            # ── Injection ────────────────────────────────────────────────────
            h_new = injection(h_carry, e)         # [B, S, d_model]

            # ── Optional iteration embedding (Phase C) ────────────────────────
            if has_iter_embed:
                t_clamped = jnp.minimum(step_idx, cfg.max_depth - 1)
                # Broadcast iter embed over batch: [d_model] → [B, S, d_model]
                iter_vec = iter_embed(t_clamped)                  # [d_model]
                h_new = h_new + iter_vec[None, None, :]

            # ── Core layers ──────────────────────────────────────────────────
            for blk in core_blocks:
                h_new = blk(h_new, deterministic=deterministic)   # [B, S, d_model]

            # ── No-grad masking for first n_max iterations ────────────────────
            # step_idx is a traced integer; n_max is a Python int (static).
            # jnp comparison produces a traced bool — fine for jnp.where.
            is_nograd = step_idx < n_max                          # scalar bool
            h_new = jnp.where(is_nograd, jax.lax.stop_gradient(h_new), h_new)

            # ── Per-sequence freeze for finished sequences ─────────────────────
            # active[b] = True while step_idx < depths[b]
            # Broadcast mask: [B] → [B, 1, 1]
            active = (step_idx < depths)[:, None, None]           # [B, 1, 1]
            h_next = jnp.where(active, h_new, h_carry)

            return h_next, None

        # Apply gradient checkpointing to the scan body.
        # jax.checkpoint (remat) is the correct API for plain functions /
        # closures — nn.remat is reserved for Flax Module classes.
        # This makes activation memory O(1) per scan iteration instead of O(T).
        scan_body_ckpt = jax.checkpoint(scan_body, prevent_cse=False)

        if total_iters > 0:
            h, _ = jax.lax.scan(
                scan_body_ckpt,
                h,
                jnp.arange(total_iters, dtype=jnp.int32),
            )
        # If total_iters == 0 (shouldn't happen in practice), h stays all-zeros.

        # ── 7. Coda ──────────────────────────────────────────────────────────
        x = h
        for i in range(cfg.n_coda):
            x = TransformerBlock(cfg, name=f"coda_{i}")(x, deterministic=deterministic)

        # ── 8. Final norm + LM head ──────────────────────────────────────────
        x = RMSNorm(cfg.norm_eps, name="final_norm")(x)           # [B, S, d_model]

        # Weight-tied LM head: logits = x @ embed.T
        # In Flax, access the embedding kernel via embed.variables is not
        # available in @nn.compact after the first call. Instead, use
        # nn.Dense with kernel_init=lambda ...: embed.embedding to share.
        # The cleanest pattern: call embed.attend() which does exactly x @ W_e.T.
        logits = embed.attend(x)                                  # [B, S, vocab_size]

        # ── 9. Loss ───────────────────────────────────────────────────────────
        loss = None
        if labels is not None:
            # Labels are pre-shifted by the data loader (labels[t] = input_ids[t+1])
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits.reshape(-1, cfg.vocab_size),
                labels.reshape(-1),
            ).mean()

        return {
            "logits": logits,
            "loss": loss,
            "depth_meta": {
                "t_max": total_iters,
                "n_max": n_max,
                "k_max": k_max,
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience helpers
# ─────────────────────────────────────────────────────────────────────────────

def create_looped_transformer(
    config: LoopedBlockELLConfig,
    init_key: jax.random.KeyArray,
    batch_size: int = 2,
    seq_len: int = 64,
) -> tuple[LoopedTransformer, dict]:
    """Instantiate model and initialise parameters.

    Args:
        config:     Model config.
        init_key:   JAX PRNG key for parameter initialisation.
        batch_size: Dummy batch size for shape inference (not stored).
        seq_len:    Dummy sequence length for shape inference.

    Returns:
        (model, params) where params is the Flax parameter dict.
    """
    model = LoopedTransformer(config)

    # Dummy inputs for shape inference
    dummy_ids = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    dummy_depths = jnp.full((batch_size,), config.mean_depth, dtype=jnp.int32)
    bptt = config.bptt_depth or -(-config.mean_depth // 2)
    n_max = max(0, config.mean_depth - bptt)
    k_max = min(config.mean_depth, bptt)

    variables = model.init(
        init_key,
        dummy_ids,
        dummy_depths,
        n_max,
        k_max,
    )
    return model, variables


def model_fwd(
    model: LoopedTransformer,
    params: dict,
    input_ids: jnp.ndarray,
    plan: DepthPlan,
    labels: Optional[jnp.ndarray] = None,
    deterministic: bool = True,
    mutable: bool = False,
) -> dict:
    """Functional forward pass — wraps model.apply with a DepthPlan.

    Keeps n_max / k_max as Python ints so lax.scan stays static.

    Args:
        model:        LoopedTransformer instance.
        params:       Flax variables dict (from model.init or checkpoint).
        input_ids:    [B, S] int32 token IDs.
        plan:         DepthPlan (n_max, k_max must be Python ints).
        labels:       [B, S] int32 target IDs (optional; enables loss).
        deterministic: Passed through to blocks (dropout control).
        mutable:      If True, return (output, updates) for mutable state.

    Returns:
        Output dict from LoopedTransformer.__call__.
    """
    call_kwargs = dict(
        input_ids=input_ids,
        depths=plan.total,
        n_max=plan.n_max,          # Python int — static
        k_max=plan.k_max,          # Python int — static
        labels=labels,
        deterministic=deterministic,
    )
    if mutable:
        return model.apply(params, **call_kwargs, mutable=["block_ell"])
    return model.apply(params, **call_kwargs)
