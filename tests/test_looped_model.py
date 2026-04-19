"""Tests for the looped transformer model."""

import jax
import jax.numpy as jnp
import pytest

from looped_blockell.config import LoopedBlockELLConfig
from looped_blockell.looping import (
    LoopedTransformer,
    DiagonalInjection,
    sample_depth,
    sample_fixed,
)


@pytest.fixture
def small_config():
    return LoopedBlockELLConfig(
        d_model=64,
        n_heads=4,
        d_ff=256,
        n_layers=3,
        n_prelude=1,
        n_core=1,
        n_coda=1,
        vocab_size=256,
        max_seq_len=32,
        mean_depth=4,
        max_depth=8,
        tile_size=16,
        macro_tile_size=64,
    )


def test_diagonal_injection_stability():
    key = jax.random.PRNGKey(0)
    inj = DiagonalInjection(d_model=64)
    params = inj.init(key, jnp.zeros((2, 8, 64)), jnp.zeros((2, 8, 64)))

    h = jax.random.normal(key, (2, 8, 64))
    e = jax.random.normal(jax.random.PRNGKey(1), (2, 8, 64))

    h_new = inj.apply(params, h, e)
    assert h_new.shape == h.shape
    assert jnp.all(jnp.isfinite(h_new))


def test_depth_sampler_poisson():
    key = jax.random.PRNGKey(42)
    plan = sample_depth(key, batch_size=16, mean_depth=8, min_depth=1, max_depth=32, bptt_depth=4)
    assert plan.total.shape == (16,)
    assert plan.n_max >= 0
    assert plan.k_max >= 1
    assert plan.n_max + plan.k_max <= 32
    assert jnp.all(plan.total >= 1)
    assert jnp.all(plan.total <= 32)


def test_depth_sampler_fixed():
    plan = sample_fixed(batch_size=8, mean_depth=8, bptt_depth=4)
    assert plan.total.shape == (8,)
    assert jnp.all(plan.total == 8)
    assert plan.k_max == 4
    assert plan.n_max == 4


def test_looped_transformer_forward(small_config):
    cfg = small_config
    model = LoopedTransformer(cfg)
    key = jax.random.PRNGKey(0)

    input_ids = jax.random.randint(key, (2, 32), 0, cfg.vocab_size)
    depths = jnp.full((2,), 4, dtype=jnp.int32)

    variables = model.init(
        {"params": key},
        input_ids=input_ids,
        depths=depths,
        n_max=2,
        k_max=2,
        deterministic=True,
    )

    out = model.apply(
        variables,
        input_ids=input_ids,
        depths=depths,
        n_max=2,
        k_max=2,
        deterministic=True,
    )

    assert out["logits"].shape == (2, 32, cfg.vocab_size)


def test_looped_transformer_loss(small_config):
    cfg = small_config
    model = LoopedTransformer(cfg)
    key = jax.random.PRNGKey(0)

    input_ids = jax.random.randint(key, (2, 32), 0, cfg.vocab_size)
    labels = jax.random.randint(jax.random.PRNGKey(1), (2, 32), 0, cfg.vocab_size)
    depths = jnp.full((2,), 4, dtype=jnp.int32)

    variables = model.init(
        {"params": key},
        input_ids=input_ids,
        depths=depths,
        n_max=2,
        k_max=2,
        deterministic=True,
    )

    out = model.apply(
        variables,
        input_ids=input_ids,
        labels=labels,
        depths=depths,
        n_max=2,
        k_max=2,
        deterministic=True,
    )

    assert out["loss"] is not None
    assert out["loss"].shape == ()
    assert jnp.isfinite(out["loss"])


def test_looped_transformer_gradient(small_config):
    cfg = small_config
    model = LoopedTransformer(cfg)
    key = jax.random.PRNGKey(0)

    input_ids = jax.random.randint(key, (2, 32), 0, cfg.vocab_size)
    labels = jax.random.randint(jax.random.PRNGKey(1), (2, 32), 0, cfg.vocab_size)
    depths = jnp.full((2,), 4, dtype=jnp.int32)

    variables = model.init(
        {"params": key},
        input_ids=input_ids,
        depths=depths,
        n_max=2,
        k_max=2,
        deterministic=True,
    )

    def loss_fn(params):
        out = model.apply(
            {"params": params},
            input_ids=input_ids,
            labels=labels,
            depths=depths,
            n_max=2,
            k_max=2,
            deterministic=True,
        )
        return out["loss"]

    grads = jax.grad(loss_fn)(variables["params"])
    grad_norms = jax.tree_util.tree_map(lambda g: jnp.linalg.norm(g), grads)
    total_norm = sum(jax.tree_util.tree_leaves(grad_norms))
    assert total_norm > 0, "Gradients should be non-zero"
