"""Tests for Phase 0 new components: outer SSM, Lorentz embeddings, sparse attention.

Run with: JAX_PLATFORM_NAME=cpu pytest tests/test_new_components.py -v
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_platform_name", "cpu")


# ─── Lorentz Embeddings ──────────────────────────────────────────────────────

class TestLorentzEmbeddings:
    def test_project_to_hyperboloid_constraint(self):
        from looped_blockell.layers.embeddings import project_to_hyperboloid, minkowski_dot
        space = jax.random.normal(jax.random.PRNGKey(0), (8, 64))
        hyp = project_to_hyperboloid(space)
        inner = minkowski_dot(hyp, hyp)
        # Should be -1 on the hyperboloid
        assert jnp.allclose(inner, -1.0, atol=1e-4), f"Minkowski inner: {inner.mean()}"

    def test_log_exp_roundtrip(self):
        from looped_blockell.layers.embeddings import (
            project_to_hyperboloid, log_map_origin, exp_map_origin,
        )
        space = jax.random.normal(jax.random.PRNGKey(1), (4, 32)) * 0.1
        hyp = project_to_hyperboloid(space)
        tangent = log_map_origin(hyp)
        hyp_back = exp_map_origin(tangent)
        assert jnp.allclose(hyp, hyp_back, atol=1e-3), "Log-Exp roundtrip failed"

    def test_lorentz_embedding_forward(self):
        from looped_blockell.layers.embeddings import LorentzEmbedding
        embed = LorentzEmbedding(num_embeddings=100, features=64)
        ids = jnp.array([[0, 1, 2, 3]])
        params = embed.init(jax.random.PRNGKey(2), ids)
        out = embed.apply(params, ids)
        assert out.shape == (1, 4, 64)

    def test_lorentz_attend(self):
        from looped_blockell.layers.embeddings import LorentzEmbedding
        embed = LorentzEmbedding(num_embeddings=100, features=64)
        ids = jnp.array([[0, 1, 2]])
        params = embed.init(jax.random.PRNGKey(3), ids)
        out = embed.apply(params, ids)
        logits = embed.apply(params, out, method=embed.attend)
        assert logits.shape == (1, 3, 100)

    def test_hybrid_embedding_forward(self):
        from looped_blockell.layers.embeddings import HybridEmbedding
        embed = HybridEmbedding(num_embeddings=100, euclidean_dim=32, lorentz_dim=32)
        ids = jnp.array([[0, 1, 2]])
        params = embed.init(jax.random.PRNGKey(4), ids)
        out = embed.apply(params, ids)
        assert out.shape == (1, 3, 64)

    def test_hybrid_attend(self):
        from looped_blockell.layers.embeddings import HybridEmbedding
        embed = HybridEmbedding(num_embeddings=100, euclidean_dim=32, lorentz_dim=32)
        ids = jnp.array([[0, 1]])
        params = embed.init(jax.random.PRNGKey(5), ids)
        out = embed.apply(params, ids)
        logits = embed.apply(params, out, method=embed.attend)
        assert logits.shape == (1, 2, 100)


# ─── Outer SSM Loop ──────────────────────────────────────────────────────────

class TestOuterSSM:
    """Tests for the outer SSM cross-sequence state.

    The LoopedTransformer uses lax.scan + jax.checkpoint internally.
    Flax nn.compact modules inside scan_body create params as side effects,
    which can leak tracers. The train.py handles this by providing the
    topology collection. For unit tests, we need to init with the right
    collections and pass rngs correctly.
    """

    def _make_config(self, use_outer=True, detach=True):
        from looped_blockell.config import LoopedBlockELLConfig
        return LoopedBlockELLConfig(
            d_model=64, n_heads=4, d_ff=256,
            n_layers=3, n_prelude=1, n_core=1, n_coda=1,
            vocab_size=128, max_seq_len=32,
            mean_depth=2, max_depth=4,
            use_outer_ssm=use_outer,
            outer_state_detach=detach,
        )

    def _init_and_apply(self, cfg, outer_state=None, labels=None, key_seed=0):
        from looped_blockell.looping.looped_model import LoopedTransformer
        model = LoopedTransformer(cfg)
        ids = jnp.ones((2, 32), dtype=jnp.int32)
        depths = jnp.full((2,), 2, dtype=jnp.int32)
        key = jax.random.PRNGKey(key_seed)
        k1, k2, k3 = jax.random.split(key, 3)

        init_kwargs = dict(input_ids=ids, depths=depths, n_max=1, k_max=1)
        if outer_state is not None:
            init_kwargs["outer_state"] = outer_state

        variables = model.init({"params": k1, "state": k2}, **init_kwargs)

        apply_kwargs = dict(
            input_ids=ids, depths=depths, n_max=1, k_max=1,
            rngs={"state": k3},
        )
        if outer_state is not None:
            apply_kwargs["outer_state"] = outer_state
        if labels is not None:
            apply_kwargs["labels"] = labels

        out = model.apply(variables, **apply_kwargs)
        return model, variables, out

    def test_outer_state_output_exists(self):
        cfg = self._make_config(use_outer=False)
        _, _, out = self._init_and_apply(cfg)
        assert "outer_state_out" in out
        assert out["outer_state_out"].shape == (2, 32, 64)

    def test_outer_state_injection_changes_output(self):
        cfg = self._make_config(use_outer=True, detach=True)
        zero_state = jnp.zeros((2, 32, 64))
        rand_state = jax.random.normal(jax.random.PRNGKey(99), (2, 32, 64)) * 0.1

        _, vars1, out1 = self._init_and_apply(cfg, outer_state=zero_state, key_seed=0)

        # Apply same model with different outer state
        from looped_blockell.looping.looped_model import LoopedTransformer
        out2 = LoopedTransformer(cfg).apply(
            vars1,
            input_ids=jnp.ones((2, 32), dtype=jnp.int32),
            depths=jnp.full((2,), 2, dtype=jnp.int32),
            n_max=1, k_max=1,
            outer_state=rand_state,
            rngs={"state": jax.random.PRNGKey(0)},
        )
        # Different outer state should produce different logits
        assert not jnp.allclose(out1["logits"], out2["logits"], atol=1e-3)

    def test_outer_injection_params_exist(self):
        cfg = self._make_config(use_outer=True)
        zero_state = jnp.zeros((2, 32, 64))
        _, variables, _ = self._init_and_apply(cfg, outer_state=zero_state)
        params = variables["params"]
        assert "outer_injection" in params, f"Missing outer_injection. Keys: {list(params.keys())}"
        assert "log_A" in params["outer_injection"]
        assert "log_dt" in params["outer_injection"]


# ─── DeepSeek Sparse Attention ────────────────────────────────────────────────

class TestDeepSeekSparseAttention:
    def test_short_sequence_fallback(self):
        """Short sequences should use full causal attention."""
        from looped_blockell.layers.sparse_attention import DeepSeekSparseAttention
        attn = DeepSeekSparseAttention(
            d_model=64, n_heads=4, max_seq_len=128,
            top_k=64, block_size=16,
        )
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 32, 64))
        params = attn.init(jax.random.PRNGKey(1), x)
        out = attn.apply(params, x)
        assert out.shape == (2, 32, 64)

    def test_output_shape(self):
        from looped_blockell.layers.sparse_attention import DeepSeekSparseAttention
        attn = DeepSeekSparseAttention(
            d_model=64, n_heads=4, max_seq_len=256,
            top_k=64, block_size=16,
        )
        x = jax.random.normal(jax.random.PRNGKey(0), (1, 128, 64))
        params = attn.init(jax.random.PRNGKey(1), x)
        out = attn.apply(params, x)
        assert out.shape == (1, 128, 64)


# ─── Config Validation ────────────────────────────────────────────────────────

class TestConfig:
    def test_default_config(self):
        from looped_blockell.config import LoopedBlockELLConfig
        cfg = LoopedBlockELLConfig()
        assert cfg.embed_geometry == "euclidean"
        assert cfg.use_sparse_attention is False
        assert cfg.use_outer_ssm is False

    def test_hybrid_config(self):
        from looped_blockell.config import LoopedBlockELLConfig
        cfg = LoopedBlockELLConfig(embed_geometry="hybrid", lorentz_dim_fraction=0.5)
        assert cfg.embed_geometry == "hybrid"

    def test_invalid_geometry_raises(self):
        from looped_blockell.config import LoopedBlockELLConfig
        with pytest.raises(AssertionError):
            LoopedBlockELLConfig(embed_geometry="invalid")
