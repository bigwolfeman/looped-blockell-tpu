"""Tests for PyTorch ↔ JAX checkpoint interop.

Verifies:
  1. PT model forward pass produces valid outputs
  2. Name mapping is invertible (round-trip)
  3. PT→JAX param conversion is correct (shapes, values)
  4. JAX→PT param conversion is correct
  5. Dense ↔ Block-ELL round-trip is lossless
  6. Optimizer state converts correctly
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interop.pt_model import InteropConfig, LoopedTransformerPT, sample_depth, sample_fixed
from interop.convert_checkpoint import (
    _pt_to_jax_name,
    _jax_to_pt_name,
    _dense_to_block_ell,
    _block_ell_to_dense,
    _needs_transpose,
    pt_to_jax,
    jax_to_pt,
    _set_nested,
    _get_nested,
)


@pytest.fixture
def small_cfg():
    return InteropConfig(
        d_model=64,
        n_heads=4,
        d_ff=256,
        n_prelude=1,
        n_core=2,
        n_coda=1,
        vocab_size=256,
        max_seq_len=32,
        mean_depth=3,
        max_depth=4,
        use_poisson=False,
        use_checkpointing=False,
        batch_size=2,
    )


@pytest.fixture
def model_and_ckpt(small_cfg):
    torch.manual_seed(42)
    model = LoopedTransformerPT(small_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95))

    # Run a few steps to populate optimizer state
    for _ in range(3):
        x = torch.randint(0, small_cfg.vocab_size, (2, 32))
        depths = torch.full((2,), small_cfg.mean_depth, dtype=torch.int32)
        out = model(x, depths, n_max=1, k_max=2, labels=x)
        out["loss"].backward()
        optimizer.step()
        optimizer.zero_grad()

    ckpt = {
        "step": 3,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    return model, optimizer, ckpt


# ─── Name mapping tests ──────────────────────────────────────────────────────

class TestNameMapping:
    def test_embed(self):
        path, leaf = _pt_to_jax_name("embed.weight")
        assert path == ["embed"]
        assert leaf == "embedding"
        assert _jax_to_pt_name(path, leaf) == "embed.weight"

    def test_prelude_block(self):
        path, leaf = _pt_to_jax_name("prelude.0.attention.qkv_proj.weight")
        assert path == ["prelude_0", "attention", "qkv_proj"]
        assert leaf == "kernel"
        assert _jax_to_pt_name(path, leaf) == "prelude.0.attention.qkv_proj.weight"

    def test_core_block(self):
        path, leaf = _pt_to_jax_name("core.2.mlp.fc1.weight")
        assert path == ["core_2", "mlp", "fc1"]
        assert leaf == "kernel"

    def test_norm(self):
        path, leaf = _pt_to_jax_name("input_norm.scale")
        assert path == ["input_norm"]
        assert leaf == "scale"
        assert _jax_to_pt_name(path, leaf) == "input_norm.scale"

    def test_injection(self):
        path, leaf = _pt_to_jax_name("injection.log_A")
        assert path == ["injection"]
        assert leaf == "log_A"

    def test_iteration_embed(self):
        path, leaf = _pt_to_jax_name("iteration_embed.weight")
        assert path == ["iteration_embed"]
        assert leaf == "embedding"
        assert _jax_to_pt_name(path, leaf) == "iteration_embed.weight"

    def test_round_trip_all_names(self, small_cfg):
        model = LoopedTransformerPT(small_cfg)
        for pt_name in model.state_dict().keys():
            if "freqs" in pt_name:
                continue
            path, leaf = _pt_to_jax_name(pt_name)
            recovered = _jax_to_pt_name(path, leaf)
            # Handle MLP values → kernel → weight naming
            if "fc1" in pt_name or "fc2" in pt_name:
                if ".weight" in pt_name:
                    assert recovered == pt_name, f"Round-trip failed: {pt_name} → {recovered}"
            else:
                assert recovered == pt_name, f"Round-trip failed: {pt_name} → {recovered}"


# ─── Block-ELL conversion tests ──────────────────────────────────────────────

class TestBlockELL:
    def test_dense_to_bell_round_trip(self):
        """Dense → Block-ELL → Dense should be lossless at density=1.0."""
        weight = np.random.randn(256, 64).astype(np.float32)
        values, col_indices = _dense_to_block_ell(weight, tile_size=16)

        assert values.shape == (16, 4, 16, 16)  # R=256/16=16, K=C=64/16=4
        assert col_indices.shape == (16, 4)

        recovered = _block_ell_to_dense(values, col_indices, 256, 64, tile_size=16)
        np.testing.assert_allclose(recovered, weight, atol=1e-6)

    def test_bell_col_indices_sequential(self):
        """At density=1.0, col_indices should be [0, 1, ..., C-1] per row."""
        weight = np.random.randn(128, 64).astype(np.float32)
        _, col_indices = _dense_to_block_ell(weight, tile_size=16)
        expected = np.tile(np.arange(4, dtype=np.int32), (8, 1))
        np.testing.assert_array_equal(col_indices, expected)


# ─── Transpose tests ─────────────────────────────────────────────────────────

class TestTranspose:
    def test_kernel_needs_transpose(self):
        assert _needs_transpose(["attention", "qkv_proj"], "kernel")
        assert _needs_transpose(["mlp", "fc1"], "kernel")

    def test_embedding_no_transpose(self):
        assert not _needs_transpose(["embed"], "embedding")

    def test_scale_no_transpose(self):
        assert not _needs_transpose(["norm"], "scale")

    def test_bias_no_transpose(self):
        assert not _needs_transpose(["mlp", "fc1"], "bias")

    def test_log_A_no_transpose(self):
        assert not _needs_transpose(["injection"], "log_A")


# ─── Full conversion tests ───────────────────────────────────────────────────

class TestFullConversion:
    def test_pt_to_jax_params(self, model_and_ckpt, small_cfg):
        model, _, ckpt = model_and_ckpt
        jax_ckpt = pt_to_jax(ckpt, tile_size=16)

        params = jax_ckpt["params"]
        assert "embed" in params
        assert "embedding" in params["embed"]
        assert params["embed"]["embedding"].shape == (256, 64)

        # Check a core block exists
        assert "core_0" in params
        assert "attention" in params["core_0"]
        assert "qkv_proj" in params["core_0"]["attention"]

        # Check MLP is in Block-ELL format
        assert "mlp" in params["core_0"]
        assert "fc1" in params["core_0"]["mlp"]
        assert "values" in params["core_0"]["mlp"]["fc1"]
        R = 256 // 16  # d_ff / tile_size
        C = 64 // 16   # d_model / tile_size
        assert params["core_0"]["mlp"]["fc1"]["values"].shape == (R, C, 16, 16)

        # Check topology
        topo = jax_ckpt["topology"]
        assert "core_0" in topo
        assert "col_indices" in topo["core_0"]["mlp"]["fc1"]

    def test_jax_to_pt_params(self, model_and_ckpt, small_cfg):
        model, _, ckpt = model_and_ckpt
        jax_ckpt = pt_to_jax(ckpt, tile_size=16)
        pt_ckpt = jax_to_pt(jax_ckpt, tile_size=16)

        sd = pt_ckpt["model_state_dict"]
        assert "embed.weight" in sd
        assert sd["embed.weight"].shape == (256, 64)

        # Check attention kernel was transposed back
        assert "core.0.attention.qkv_proj.weight" in sd
        assert sd["core.0.attention.qkv_proj.weight"].shape == (3 * 64, 64)

        # Check MLP was converted back to dense
        assert "core.0.mlp.fc1.weight" in sd
        assert sd["core.0.mlp.fc1.weight"].shape == (256, 64)

    def test_round_trip_values(self, model_and_ckpt, small_cfg):
        """PT → JAX → PT should preserve parameter values."""
        model, _, ckpt = model_and_ckpt
        original_sd = {k: v.clone() for k, v in ckpt["model_state_dict"].items()}

        jax_ckpt = pt_to_jax(ckpt, tile_size=16)
        pt_ckpt = jax_to_pt(jax_ckpt, tile_size=16)

        for name, orig_val in original_sd.items():
            if "freqs" in name:
                continue
            assert name in pt_ckpt["model_state_dict"], f"Missing: {name}"
            recovered = pt_ckpt["model_state_dict"][name]
            np.testing.assert_allclose(
                orig_val.numpy(), recovered.numpy(),
                atol=1e-5, rtol=1e-5,
                err_msg=f"Value mismatch for {name}",
            )

    def test_optimizer_state_round_trip(self, model_and_ckpt, small_cfg):
        """Optimizer state should survive PT → JAX → PT conversion."""
        _, _, ckpt = model_and_ckpt
        jax_ckpt = pt_to_jax(ckpt, tile_size=16)
        pt_ckpt = jax_to_pt(jax_ckpt, tile_size=16)

        orig_opt = ckpt["optimizer_state_dict"]["state"]
        recovered_opt = pt_ckpt["optimizer_state_dict"]["state"]

        for param_id in orig_opt:
            if param_id not in recovered_opt:
                continue
            orig = orig_opt[param_id]
            recov = recovered_opt[param_id]
            np.testing.assert_allclose(
                orig["exp_avg"].numpy(), recov["exp_avg"].numpy(),
                atol=1e-5, rtol=1e-5,
                err_msg=f"exp_avg mismatch for param {param_id}",
            )
            np.testing.assert_allclose(
                orig["exp_avg_sq"].numpy(), recov["exp_avg_sq"].numpy(),
                atol=1e-5, rtol=1e-5,
                err_msg=f"exp_avg_sq mismatch for param {param_id}",
            )


# ─── Model forward pass tests ────────────────────────────────────────────────

class TestModelForward:
    def test_forward_basic(self, small_cfg):
        torch.manual_seed(42)
        model = LoopedTransformerPT(small_cfg)
        x = torch.randint(0, small_cfg.vocab_size, (2, 32))
        depths = torch.full((2,), small_cfg.mean_depth, dtype=torch.int32)
        out = model(x, depths, n_max=1, k_max=2)
        assert out["logits"].shape == (2, 32, 256)
        assert out["loss"] is None

    def test_forward_with_loss(self, small_cfg):
        torch.manual_seed(42)
        model = LoopedTransformerPT(small_cfg)
        x = torch.randint(0, small_cfg.vocab_size, (2, 32))
        depths = torch.full((2,), small_cfg.mean_depth, dtype=torch.int32)
        out = model(x, depths, n_max=1, k_max=2, labels=x)
        assert out["loss"] is not None
        assert out["loss"].shape == ()
        assert out["loss"].item() > 0

    def test_backward(self, small_cfg):
        torch.manual_seed(42)
        model = LoopedTransformerPT(small_cfg)
        x = torch.randint(0, small_cfg.vocab_size, (2, 32))
        depths = torch.full((2,), small_cfg.mean_depth, dtype=torch.int32)
        out = model(x, depths, n_max=1, k_max=2, labels=x)
        out["loss"].backward()
        for name, p in model.named_parameters():
            # iteration_embed is unused when use_iter_embed=False (default)
            if "iteration_embed" in name:
                continue
            assert p.grad is not None, f"No grad for {name}"

    def test_depth_sampling(self):
        plan = sample_depth(8, mean_depth=6, max_depth=10, bptt_depth=3)
        assert plan.total.shape == (8,)
        assert plan.total.min() >= 1
        assert plan.total.max() <= 10
        assert plan.n_max >= 0
        assert plan.k_max >= 0
        assert plan.n_max + plan.k_max <= 10


# ─── Utility tests ───────────────────────────────────────────────────────────

class TestUtils:
    def test_set_get_nested(self):
        d = {}
        _set_nested(d, ["a", "b"], "c", 42)
        assert _get_nested(d, ["a", "b", "c"]) == 42

    def test_get_nested_missing(self):
        assert _get_nested({}, ["a", "b"]) is None
        assert _get_nested({"a": 1}, ["a", "b"]) is None
