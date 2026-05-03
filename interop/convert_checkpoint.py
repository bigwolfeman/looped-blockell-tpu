"""Bidirectional PyTorch ↔ JAX/Flax checkpoint converter.

Converts checkpoints between the PyTorch LoopedTransformerPT and the
JAX/Flax LoopedTransformer so training can resume seamlessly across
frameworks. wandb should show one continuous run.

Handles:
  - Parameter name mapping (dotted ↔ nested dict)
  - Tensor transpose (PyTorch [out, in] ↔ Flax [in, out] for Dense layers)
  - nn.Linear weights ↔ Block-ELL tile format (at density=1.0)
  - Optimizer state (AdamW ↔ optax chain(clip, adamw))
  - Training state (step, RNG key)

Usage:
  python interop/convert_checkpoint.py --direction pt2jax --input ckpt.pt --output ckpt.pkl
  python interop/convert_checkpoint.py --direction jax2pt --input ckpt.pkl --output ckpt.pt
"""

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ─── Name mapping ─────────────────────────────────────────────────────────────

def _pt_to_jax_name(pt_name: str) -> tuple[list[str], str]:
    """Convert PyTorch param name to JAX path + leaf name.

    Returns (path_parts, leaf_name) where path_parts are the nested dict
    keys and leaf_name is the Flax param name.

    Examples:
      'embed.weight'                    → (['embed'], 'embedding')
      'prelude.0.norm_attn.scale'       → (['prelude_0', 'norm_attn'], 'scale')
      'core.2.attention.qkv_proj.weight'→ (['core_2', 'attention', 'qkv_proj'], 'kernel')
      'core.0.mlp.fc1.weight'          → (['core_0', 'mlp', 'fc1'], 'kernel')
      'core.0.mlp.fc1.bias'            → (['core_0', 'mlp', 'fc1'], 'bias')
      'injection.log_A'                → (['injection'], 'log_A')
      'input_norm.scale'               → (['input_norm'], 'scale')
      'final_norm.scale'               → (['final_norm'], 'scale')
      'iteration_embed.weight'         → (['iteration_embed'], 'embedding')
      'loop_hc.alpha'                  → (['loop_hc'], 'alpha')
      'outer_injection.log_A'          → (['outer_injection'], 'log_A')
    """
    parts = pt_name.split(".")
    jax_parts = []
    i = 0
    while i < len(parts):
        p = parts[i]
        # ModuleList index: 'prelude.0' → 'prelude_0'
        if i + 1 < len(parts) and parts[i + 1].isdigit():
            jax_parts.append(f"{p}_{parts[i + 1]}")
            i += 2
        else:
            jax_parts.append(p)
            i += 1

    leaf = jax_parts[-1]
    path = jax_parts[:-1]

    # Rename leaf: weight→kernel/embedding, keep scale/bias/log_A/log_dt/alpha/beta_*
    if leaf == "weight":
        parent = path[-1] if path else ""
        if parent in ("embed", "iteration_embed"):
            leaf = "embedding"
        else:
            leaf = "kernel"

    return path, leaf


def _jax_to_pt_name(jax_path: list[str], leaf: str) -> str:
    """Reverse of _pt_to_jax_name."""
    pt_parts = []
    for p in jax_path:
        # 'prelude_0' → 'prelude', '0'
        if "_" in p:
            prefix, suffix = p.rsplit("_", 1)
            if suffix.isdigit():
                pt_parts.extend([prefix, suffix])
            else:
                pt_parts.append(p)
        else:
            pt_parts.append(p)

    # Rename leaf back
    if leaf == "kernel":
        leaf = "weight"
    elif leaf == "embedding":
        leaf = "weight"

    pt_parts.append(leaf)
    return ".".join(pt_parts)


# ─── Tensor conversion ───────────────────────────────────────────────────────

def _needs_transpose(jax_path: list[str], leaf: str) -> bool:
    """Whether this param needs transposing between PyTorch and JAX.

    Rule: 2D Dense layers (kernel) need transpose. Embeddings, norms,
    1D params (log_A, log_dt, bias) do not.
    """
    return leaf == "kernel"


def _dense_to_block_ell(weight: np.ndarray, tile_size: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Convert dense weight [out, in] to Block-ELL at density=1.0.

    Returns (values [R, K, B, B], col_indices [R, K]).
    The values follow Flax convention: values[r,k] is [B_in, B_out].
    """
    out_features, in_features = weight.shape
    B = tile_size
    R = out_features // B
    C = in_features // B
    K = C  # density=1.0

    # [out, in] → [R, B_out, C, B_in] → [R, C, B_in, B_out]
    # The einsum "bsrkd,rkdD->bsrD" contracts d (input sub-idx) with
    # x_gathered and keeps D (output sub-idx), so values must be [in, out].
    blocks = weight.reshape(R, B, C, B).transpose(0, 2, 3, 1)  # [R, C, B_in, B_out]
    col_indices = np.tile(np.arange(C, dtype=np.int32), (R, 1))  # [R, K]
    return blocks, col_indices


def _block_ell_to_dense(values: np.ndarray, col_indices: np.ndarray,
                         out_features: int, in_features: int,
                         tile_size: int = 16) -> np.ndarray:
    """Convert Block-ELL values [R, K, B_in, B_out] back to dense [out, in]."""
    B = tile_size
    R = out_features // B
    C = in_features // B

    dense = np.zeros((out_features, in_features), dtype=values.dtype)
    for r in range(R):
        for k in range(values.shape[1]):
            c = int(col_indices[r, k])
            # values[r,k] is [B_in, B_out], dense slice is [B_out, B_in]
            dense[r * B:(r + 1) * B, c * B:(c + 1) * B] = values[r, k].T
    return dense


# ─── PyTorch → JAX ───────────────────────────────────────────────────────────

def pt_to_jax(
    pt_ckpt: dict,
    tile_size: int = 16,
) -> dict:
    """Convert PyTorch checkpoint to JAX/Flax format.

    Args:
        pt_ckpt: PyTorch checkpoint dict with 'model_state_dict',
                 'optimizer_state_dict', 'step', etc.
        tile_size: Block-ELL tile size.

    Returns:
        JAX checkpoint dict ready for pickle + flax.serialization.
    """
    sd = pt_ckpt["model_state_dict"]
    params = {}

    # Build param name → index mapping for optimizer conversion
    param_names_ordered = list(sd.keys())

    for pt_name, tensor in sd.items():
        arr = tensor.detach().cpu().numpy()
        jax_path, leaf = _pt_to_jax_name(pt_name)

        # Skip RoPE buffers (computed, not learned)
        if "freqs" in pt_name:
            continue

        # Transpose Dense kernels
        if _needs_transpose(jax_path, leaf):
            parent = jax_path[-1] if jax_path else ""
            # MLP layers: convert to Block-ELL format
            if parent in ("fc1", "fc2"):
                # arr is [out, in] from PyTorch
                values, col_indices = _dense_to_block_ell(arr, tile_size)
                _set_nested(params, jax_path, "values", values)
                # col_indices stored separately (topology collection)
                # For now, embed them in a parallel structure
                continue
            else:
                arr = arr.T  # [out, in] → [in, out]

        _set_nested(params, jax_path, leaf, arr)

    # Handle MLP biases (already correct shape, just set them)
    # They were handled in the main loop above.

    # Handle MLP fc1/fc2 weights that became Block-ELL values
    # Re-iterate to pick up biases we may have missed
    for pt_name, tensor in sd.items():
        arr = tensor.detach().cpu().numpy()
        jax_path, leaf = _pt_to_jax_name(pt_name)
        parent = jax_path[-1] if jax_path else ""
        if parent in ("fc1", "fc2") and leaf == "bias":
            _set_nested(params, jax_path, "bias", arr)

    # Build topology dict (col_indices + alive_mask)
    topology = {}
    for pt_name, tensor in sd.items():
        jax_path, leaf = _pt_to_jax_name(pt_name)
        parent = jax_path[-1] if jax_path else ""
        if parent in ("fc1", "fc2") and leaf == "kernel":
            arr = tensor.detach().cpu().numpy()
            _, col_indices = _dense_to_block_ell(arr, tile_size)
            R, K = col_indices.shape
            alive_mask = np.ones((R, K), dtype=bool)
            _set_nested(topology, jax_path, "col_indices", col_indices)
            _set_nested(topology, jax_path, "alive_mask", alive_mask)

    # Convert optimizer state
    opt_jax = _convert_optimizer_pt_to_jax(pt_ckpt, sd, params, tile_size)

    # Build JAX checkpoint
    step = pt_ckpt.get("step", 0)
    jax_ckpt = {
        "step": step,
        "params": params,
        "opt_state": opt_jax,
        "topology": topology,
        "key": np.array([0, step], dtype=np.uint32),  # deterministic seed from step
    }

    if "outer_state" in pt_ckpt:
        jax_ckpt["outer_state"] = pt_ckpt["outer_state"].detach().cpu().numpy()

    return jax_ckpt


def _convert_optimizer_pt_to_jax(
    pt_ckpt: dict,
    sd: dict,
    jax_params: dict,
    tile_size: int,
) -> dict:
    """Convert PyTorch AdamW optimizer state to optax chain(clip, adamw) format.

    optax chain(clip_by_global_norm, adamw) state structure:
      (EmptyState(), (ScaleByAdamState(count, mu, nu), EmptyState()))

    mu and nu have the same pytree structure as params.
    """
    opt_sd = pt_ckpt.get("optimizer_state_dict")
    if opt_sd is None:
        return None

    states = opt_sd.get("state", {})
    param_names = [k for k in sd.keys() if "freqs" not in k]

    # Build mu and nu trees matching jax_params structure
    mu = _zero_like_nested(jax_params)
    nu = _zero_like_nested(jax_params)

    # Map param_id → param_name for optimizer state lookup
    param_groups = opt_sd.get("param_groups", [{}])
    param_ids = param_groups[0].get("params", list(range(len(param_names))))

    for i, pt_name in enumerate(param_names):
        param_id = param_ids[i] if i < len(param_ids) else i
        if param_id not in states:
            continue

        state = states[param_id]
        exp_avg = state["exp_avg"].detach().cpu().numpy()
        exp_avg_sq = state["exp_avg_sq"].detach().cpu().numpy()

        jax_path, leaf = _pt_to_jax_name(pt_name)
        parent = jax_path[-1] if jax_path else ""

        if parent in ("fc1", "fc2") and leaf == "kernel":
            # MLP weights: convert moments to Block-ELL format too
            exp_avg_bell, _ = _dense_to_block_ell(exp_avg, tile_size)
            exp_avg_sq_bell, _ = _dense_to_block_ell(exp_avg_sq, tile_size)
            _set_nested(mu, jax_path, "values", exp_avg_bell)
            _set_nested(nu, jax_path, "values", exp_avg_sq_bell)
        elif _needs_transpose(jax_path, leaf):
            _set_nested(mu, jax_path, leaf, exp_avg.T)
            _set_nested(nu, jax_path, leaf, exp_avg_sq.T)
        else:
            _set_nested(mu, jax_path, leaf, exp_avg)
            _set_nested(nu, jax_path, leaf, exp_avg_sq)

    # Get step count
    step = 0
    for s in states.values():
        if "step" in s:
            step = int(s["step"].item())
            break

    return {
        "count": np.array(step, dtype=np.int32),
        "mu": mu,
        "nu": nu,
    }


# ─── JAX → PyTorch ───────────────────────────────────────────────────────────

def jax_to_pt(
    jax_ckpt: dict,
    tile_size: int = 16,
    device: str = "cpu",
) -> dict:
    """Convert JAX/Flax checkpoint to PyTorch format.

    Args:
        jax_ckpt: JAX checkpoint dict with 'params', 'opt_state', 'step'.
        tile_size: Block-ELL tile size.
        device: Target torch device.

    Returns:
        PyTorch checkpoint dict with 'model_state_dict', 'optimizer_state_dict', 'step'.
    """
    params = jax_ckpt["params"]
    topology = jax_ckpt.get("topology", {})

    model_sd = {}

    def _walk_params(d: dict, prefix: list[str]):
        for key, val in d.items():
            path = prefix + [key]
            if isinstance(val, dict):
                _walk_params(val, path)
            elif isinstance(val, np.ndarray):
                jax_path = path[:-1]
                leaf = path[-1]

                # Skip alive_mask / col_indices (topology, not params)
                if leaf in ("col_indices", "alive_mask"):
                    continue

                pt_name = _jax_to_pt_name(jax_path, leaf)

                if leaf == "values":
                    # Block-ELL → dense nn.Linear weight
                    parent_key = jax_path[-1] if jax_path else ""
                    topo_node = _get_nested(topology, jax_path)
                    col_indices = topo_node.get("col_indices") if topo_node else None
                    if col_indices is None:
                        R, K = val.shape[:2]
                        col_indices = np.tile(np.arange(K, dtype=np.int32), (R, 1))

                    # Infer dimensions
                    R, K, B_in, B_out = val.shape
                    out_features = R * B_out
                    # Need to figure out C from col_indices
                    C = int(col_indices.max()) + 1
                    in_features = C * B_in

                    dense = _block_ell_to_dense(val, col_indices, out_features, in_features, tile_size)
                    # Fix pt_name: leaf was 'values', need it to be 'weight'
                    pt_name = _jax_to_pt_name(jax_path, "kernel")
                    model_sd[pt_name] = torch.tensor(dense, device=device)
                elif _needs_transpose(jax_path, leaf):
                    model_sd[pt_name] = torch.tensor(val.T, device=device)
                else:
                    model_sd[pt_name] = torch.tensor(val, device=device)

    _walk_params(params, [])

    # Convert optimizer state
    opt_sd = _convert_optimizer_jax_to_pt(jax_ckpt, model_sd, tile_size, device)

    pt_ckpt = {
        "step": jax_ckpt.get("step", 0),
        "model_state_dict": model_sd,
        "optimizer_state_dict": opt_sd,
    }

    if "outer_state" in jax_ckpt and jax_ckpt["outer_state"] is not None:
        pt_ckpt["outer_state"] = torch.tensor(jax_ckpt["outer_state"], device=device)

    return pt_ckpt


def _convert_optimizer_jax_to_pt(
    jax_ckpt: dict,
    model_sd: dict,
    tile_size: int,
    device: str,
) -> dict | None:
    """Convert optax optimizer state to PyTorch AdamW format."""
    opt = jax_ckpt.get("opt_state")
    if opt is None:
        return None

    mu_tree = opt.get("mu", {})
    nu_tree = opt.get("nu", {})
    count = int(opt.get("count", 0))

    # Flatten mu/nu trees to match model_sd ordering
    pt_state = {}
    param_list = list(model_sd.keys())

    for idx, pt_name in enumerate(param_list):
        parts = pt_name.split(".")
        leaf = parts[-1]

        # Reconstruct jax path
        jax_path, jax_leaf = _pt_to_jax_name(pt_name)
        parent = jax_path[-1] if jax_path else ""

        if parent in ("fc1", "fc2") and jax_leaf == "kernel":
            mu_val = _get_nested(mu_tree, jax_path + ["values"])
            nu_val = _get_nested(nu_tree, jax_path + ["values"])
        else:
            mu_val = _get_nested(mu_tree, jax_path + [jax_leaf])
            nu_val = _get_nested(nu_tree, jax_path + [jax_leaf])

        if mu_val is None or nu_val is None:
            continue

        # Convert moments back
        if parent in ("fc1", "fc2") and jax_leaf == "kernel":
            # Block-ELL → dense moments
            R, K, B_in, B_out = mu_val.shape
            out_f = R * B_out
            C = K  # at density=1.0
            in_f = C * B_in
            col_indices = np.tile(np.arange(C, dtype=np.int32), (R, 1))
            mu_dense = _block_ell_to_dense(mu_val, col_indices, out_f, in_f, tile_size)
            nu_dense = _block_ell_to_dense(nu_val, col_indices, out_f, in_f, tile_size)
            exp_avg = torch.tensor(mu_dense, device=device)
            exp_avg_sq = torch.tensor(nu_dense, device=device)
        elif _needs_transpose(jax_path, jax_leaf):
            exp_avg = torch.tensor(mu_val.T, device=device)
            exp_avg_sq = torch.tensor(nu_val.T, device=device)
        else:
            exp_avg = torch.tensor(mu_val, device=device)
            exp_avg_sq = torch.tensor(nu_val, device=device)

        pt_state[idx] = {
            "step": torch.tensor(float(count)),
            "exp_avg": exp_avg,
            "exp_avg_sq": exp_avg_sq,
        }

    return {
        "state": pt_state,
        "param_groups": [{
            "lr": 6e-4,
            "betas": (0.9, 0.95),
            "eps": 1e-8,
            "weight_decay": 0.1,
            "amsgrad": False,
            "maximize": False,
            "foreach": None,
            "capturable": False,
            "differentiable": False,
            "fused": None,
            "params": list(range(len(param_list))),
        }],
    }


# ─── Utilities ────────────────────────────────────────────────────────────────

def _set_nested(d: dict, path: list[str], leaf: str, value: Any):
    """Set d[path[0]][path[1]]...[leaf] = value, creating dicts as needed."""
    for p in path:
        if p not in d:
            d[p] = {}
        d = d[p]
    d[leaf] = value


def _get_nested(d: dict, path: list[str]) -> Any | None:
    """Get d[path[0]][path[1]]..., returning None if any key is missing."""
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


def _zero_like_nested(d: dict) -> dict:
    """Create a zero-valued copy of a nested param dict."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _zero_like_nested(v)
        elif isinstance(v, np.ndarray):
            out[k] = np.zeros_like(v)
    return out


# ─── Serialization helpers ────────────────────────────────────────────────────

def save_jax_checkpoint(jax_ckpt: dict, path: str | Path):
    """Save JAX checkpoint in the format run_ablation.py expects."""
    import flax.serialization as ser
    import jax.numpy as jnp

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serialized = {
        "step": jax_ckpt["step"],
        "params": ser.to_bytes(jax_ckpt["params"]),
        "key": jax_ckpt.get("key", np.array([0, 0], dtype=np.uint32)),
    }

    if jax_ckpt.get("opt_state") is not None:
        serialized["opt_state"] = ser.to_bytes(jax_ckpt["opt_state"])

    if jax_ckpt.get("outer_state") is not None:
        serialized["outer_state"] = jax_ckpt["outer_state"]

    with open(path, "wb") as f:
        pickle.dump(serialized, f)


def save_pt_checkpoint(pt_ckpt: dict, path: str | Path):
    """Save PyTorch checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pt_ckpt, path)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert checkpoints between PyTorch and JAX")
    parser.add_argument("--direction", required=True, choices=["pt2jax", "jax2pt"])
    parser.add_argument("--input", required=True, help="Input checkpoint path")
    parser.add_argument("--output", required=True, help="Output checkpoint path")
    parser.add_argument("--tile-size", type=int, default=16)
    args = parser.parse_args()

    if args.direction == "pt2jax":
        print(f"Converting PyTorch → JAX: {args.input} → {args.output}")
        pt_ckpt = torch.load(args.input, map_location="cpu", weights_only=False)
        jax_ckpt = pt_to_jax(pt_ckpt, tile_size=args.tile_size)
        save_jax_checkpoint(jax_ckpt, args.output)
        print(f"  Params: {sum(v.size for v in _flatten_dict(jax_ckpt['params'])):,}")
    elif args.direction == "jax2pt":
        print(f"Converting JAX → PyTorch: {args.input} → {args.output}")
        with open(args.input, "rb") as f:
            jax_ckpt = pickle.load(f)
        # Deserialize if needed
        if isinstance(jax_ckpt.get("params"), bytes):
            import flax.serialization as ser
            # Need template — construct from checkpoint structure
            raise NotImplementedError(
                "JAX→PT from serialized checkpoint requires a model template. "
                "Use jax_to_pt() directly with deserialized params."
            )
        pt_ckpt = jax_to_pt(jax_ckpt, tile_size=args.tile_size)
        save_pt_checkpoint(pt_ckpt, args.output)
        print(f"  Params: {sum(p.numel() for p in pt_ckpt['model_state_dict'].values()):,}")

    print("Done.")


def _flatten_dict(d: dict) -> list[np.ndarray]:
    """Flatten nested dict to list of arrays."""
    out = []
    for v in d.values():
        if isinstance(v, dict):
            out.extend(_flatten_dict(v))
        elif isinstance(v, np.ndarray):
            out.append(v)
    return out


if __name__ == "__main__":
    main()
