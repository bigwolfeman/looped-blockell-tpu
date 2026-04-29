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
from ..layers.embeddings import LorentzEmbedding, HybridEmbedding
from .diagonal_injection import DiagonalInjection
from .hyper_connections import LoopBoundaryHC, init_streams, collapse_streams
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
        use_iter_embed: bool = False,
        outer_state: Optional[jnp.ndarray] = None,
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
            use_iter_embed: Enable per-iteration embedding (Phase C routing).
            outer_state:  Cross-sequence latent state     [B, S, d_model] (optional)
                          h_final from the previous sequence's core loop.
                          When provided and cfg.use_outer_ssm=True, injected
                          into the prelude output via a separate diagonal
                          injection before the core loop begins.

        Returns:
            dict with:
                'logits'     — [B, S, vocab_size]
                'loss'       — scalar (only when labels is not None)
                'depth_meta' — dict with t_max, n_max, k_max
                'outer_state_out' — [B, S, d_model] h_final for next sequence
        """
        cfg = self.config
        total_iters: int = n_max + k_max  # Python int — static for scan
        B, S = input_ids.shape

        # ── 1. Embedding ────────────────────────────────────────────────────
        if cfg.embed_geometry == "lorentz":
            embed = LorentzEmbedding(
                num_embeddings=cfg.vocab_size,
                features=cfg.d_model,
                name="embed",
            )
        elif cfg.embed_geometry == "hybrid":
            lorentz_dim = int(cfg.d_model * cfg.lorentz_dim_fraction)
            euclidean_dim = cfg.d_model - lorentz_dim
            embed = HybridEmbedding(
                num_embeddings=cfg.vocab_size,
                euclidean_dim=euclidean_dim,
                lorentz_dim=lorentz_dim,
                name="embed",
            )
        else:
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

        # ── 3b. Outer SSM injection (cross-sequence state) ──────────────────
        # Uses a separate DiagonalInjection so the outer loop has its own
        # learned decay rates, independent of the inner loop. Stability is
        # guaranteed by the same Parcae construction: decay ∈ (0,1).
        if cfg.use_outer_ssm and outer_state is not None:
            if cfg.outer_state_detach:
                outer_state = jax.lax.stop_gradient(outer_state)
            outer_injection = DiagonalInjection(
                d_model=cfg.d_model,
                init_decay=cfg.outer_init_decay,
                name="outer_injection",
            )
            e = outer_injection(outer_state, e)

        # ── 4. Recurrent state init — truncated normal (Parcae-style) ────────
        h = jax.random.normal(self.make_rng("state"), e.shape, dtype=jnp.float32) * 0.02
        h = h.astype(e.dtype)

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

        # Per-iteration embedding (Phase C routing conditioning).
        # Always created so params exist, but only USED when routing is active.
        # Near-zero init to avoid perturbing the pre-routing model.
        iter_embed = nn.Embed(
            num_embeddings=cfg.max_depth,
            features=cfg.d_model,
            embedding_init=nn.initializers.normal(stddev=0.001),
            name="iteration_embed",
        )

        # ── 6. Core loop (Python for-loop, unrolled by JIT) ──────────────────
        # We use a Python for-loop instead of lax.scan because:
        # 1. Flax nn.compact creates params as side effects — incompatible with
        #    lax.scan + jax.checkpoint (tracer leaks).
        # 2. On GPU (sm_120), lax.scan + remat creates a massive XLA compilation
        #    graph that takes 10+ minutes. A Python for-loop compiles faster
        #    because JIT unrolls it into a flat graph.
        # 3. Memory: with max_depth <= 12, the unrolled graph is manageable.
        #    For very deep loops (T>32), lax.scan would be needed — but Parcae
        #    shows T=8 is optimal, so this isn't a concern.
        #
        # The loop runs max(n_max + k_max) iterations. Per-sequence depths are
        # handled via jnp.where masking (frozen sequences waste FLOPs but
        # produce correct gradients, same as the Parcae paper).

        # Loop-boundary hyper-connections (Hyperloop-style)
        hc = None
        streams = None
        if cfg.use_loop_boundary_hc:
            hc = LoopBoundaryHC(
                d_model=cfg.d_model,
                n_streams=cfg.hc_n_streams,
                name="loop_hc",
            )
            streams = init_streams(h, cfg.hc_n_streams)  # [B, S, n, d]

        # Force Flax param materialization before jax.checkpoint.
        _ = injection(h, e)
        for blk in core_blocks:
            _ = blk(h, deterministic=deterministic)
        _ = iter_embed(jnp.array(0))
        if hc is not None:
            _ = hc(streams, mode="aggregate")
            _ = hc(streams, h, mode="distribute")

        @jax.checkpoint
        def _loop_body(h_in, t_idx, streams_in):
            # HC aggregate: mix streams → single input
            if hc is not None:
                h_agg = hc(streams_in, mode="aggregate")
            else:
                h_agg = h_in

            h_new = injection(h_agg, e)

            if use_iter_embed:
                t_clamped = jnp.minimum(t_idx, cfg.max_depth - 1)
                iter_vec = iter_embed(t_clamped)
                h_new = h_new + iter_vec[None, None, :]

            for blk in core_blocks:
                h_new = blk(h_new, deterministic=deterministic)

            is_nograd = jnp.array(t_idx) < n_max
            h_new = jnp.where(is_nograd, jax.lax.stop_gradient(h_new), h_new)

            active = (jnp.array(t_idx) < depths)[:, None, None]
            h_out = jnp.where(active, h_new, h_agg)

            # HC distribute: update streams with layer output
            if hc is not None:
                active_4d = active[:, :, :, None]
                streams_out = hc(streams_in, h_new, mode="distribute")
                streams_out = jnp.where(active_4d, streams_out, streams_in)
            else:
                streams_out = None

            return h_out, streams_out

        for t in range(total_iters):
            h, streams = _loop_body(h, t, streams)

        # Collapse streams back to single hidden state
        if hc is not None:
            h = collapse_streams(streams)

        # ── 7. Capture h_final for outer SSM before coda transforms it ────────
        h_final = h

        # ── 8. Coda ──────────────────────────────────────────────────────────
        x = h
        for i in range(cfg.n_coda):
            x = TransformerBlock(cfg, name=f"coda_{i}")(x, deterministic=deterministic)

        # ── 9. Final norm + LM head ──────────────────────────────────────────
        x = RMSNorm(cfg.norm_eps, name="final_norm")(x)           # [B, S, d_model]
        logits = embed.attend(x)                                  # [B, S, vocab_size]

        # ── 10. Loss ──────────────────────────────────────────────────────────
        loss = None
        if labels is not None:
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits.reshape(-1, cfg.vocab_size),
                labels.reshape(-1),
            ).mean()

        return {
            "logits": logits,
            "loss": loss,
            "outer_state_out": h_final,
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
    outer_state: Optional[jnp.ndarray] = None,
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
        outer_state:  [B, S, d_model] cross-sequence state from prev sequence.

    Returns:
        Output dict from LoopedTransformer.__call__.
        Includes 'outer_state_out' for carrying state to the next sequence.
    """
    call_kwargs = dict(
        input_ids=input_ids,
        depths=plan.total,
        n_max=plan.n_max,          # Python int — static
        k_max=plan.k_max,          # Python int — static
        labels=labels,
        deterministic=deterministic,
        outer_state=outer_state,
    )
    if mutable:
        return model.apply(params, **call_kwargs, mutable=["block_ell"])
    return model.apply(params, **call_kwargs)
