"""Neural Long-Term Memory (Titans paper, arXiv:2501.00663) — JAX/Flax port.

Paper-faithful implementation:
    - Memory M is a deep MLP whose WEIGHTS are the memory state
    - retrieve(x): forward pass through MLP (stop_gradient on MLP weights)
    - update(x):   compute associative loss, grad w.r.t. MLP weights, apply
                   momentum update via mutable 'memory_state' collection

Key equations (Section 3.1):
    Loss:   l(M; x) = ||M(k) - v||²                 (Eq. 12)
    S_t:    η_t * S_{t-1} - θ * ∇l(M; x)            (Eq. 13)
    M_t:    (1 - α_t) * M_{t-1} + S_t               (Eq. 14)

JAX-specific design decisions:
    - MLP weights live in the 'memory_state' variable collection (mutable).
    - Momentum buffer and surprise EMA also live in 'memory_state'.
    - `jax.value_and_grad` over a pure function with explicit params computes ∇l.
    - `jax.lax.stop_gradient` replaces torch.no_grad() / .detach().
    - update() guards writes with is_mutable_collection('memory_state') — safe
      in both init and mutable forward passes.
    - The MLP is a pure pytree (list of dicts), NOT a Flax sub-module, so
      jax.grad can differentiate through it cleanly.

Flax conventions:
    - NeuralMemory uses setup() for projection layers (nn.Dense) and exposes
      retrieve() and update() as regular methods (not @nn.compact).
    - The 'memory_state' variable is declared inside retrieve() / update()
      using self.variable(); this works because both are called from within
      a parent @nn.compact context (the LoopedTransformer.__call__).
    - __call__ is a thin wrapper that calls retrieve (used during model.init).

Memory integration in looped_model.py (residual mode):
    1. Before core loop: mem_readout = memory.retrieve(x)
    2. Inside scan body, after core blocks: h_new += mem_proj(mem_readout) * scale
    3. After core loop: memory.update(h_final, scale=memory_scale)

Warmup schedule (caller computes memory_scale):
    - 0.0 for first memory_warmup_steps
    - linear ramp 0→1 over memory_ramp_steps
    - 1.0 (full strength) thereafter
"""

from __future__ import annotations

import math
from typing import Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn


# ─────────────────────────────────────────────────────────────────────────────
# Pure MLP helpers (pytree-based, outside Flax module system)
# Operating on explicit param dicts lets jax.grad work cleanly.
# ─────────────────────────────────────────────────────────────────────────────

def _mlp_init(key: jax.random.KeyArray, d_in: int, d_hidden: int, d_out: int,
               n_layers: int) -> list[dict]:
    """Initialise MLP weights. Returns list of {w, b} dicts (one per layer).

    Architecture (paper recommendation: n_layers >= 2):
        - 1 layer:  Linear(d_in, d_out)
        - n layers: Linear(d_in, d_hidden) + SiLU + ... + Linear(d_hidden, d_out)
    """
    layers: list[dict] = []

    def _linear(key, fan_in, fan_out):
        w = jax.random.normal(key, (fan_in, fan_out), dtype=jnp.float32) * 0.02
        b = jnp.zeros((fan_out,), dtype=jnp.float32)
        return {"w": w, "b": b}

    if n_layers == 1:
        layers.append(_linear(key, d_in, d_out))
    else:
        keys = jax.random.split(key, n_layers)
        layers.append(_linear(keys[0], d_in, d_hidden))
        for i in range(1, n_layers - 1):
            layers.append(_linear(keys[i], d_hidden, d_hidden))
        layers.append(_linear(keys[-1], d_hidden, d_out))

    return layers


def _mlp_forward(params: list[dict], x: jnp.ndarray) -> jnp.ndarray:
    """Forward pass through MLP. Applies SiLU after all but last layer."""
    h = x
    for i, layer in enumerate(params):
        h = h @ layer["w"] + layer["b"]
        if i < len(params) - 1:
            h = jax.nn.silu(h)
    return h


def _mlp_flat(params: list[dict]) -> jnp.ndarray:
    """Flatten all MLP params into a 1-D float32 array."""
    leaves = jax.tree_util.tree_leaves(params)
    return jnp.concatenate([p.reshape(-1) for p in leaves])


def _mlp_n_params(params: list[dict]) -> int:
    """Total number of scalar parameters in the MLP."""
    return sum(p.size for p in jax.tree_util.tree_leaves(params))


# ─────────────────────────────────────────────────────────────────────────────
# NeuralMemory Flax module
# ─────────────────────────────────────────────────────────────────────────────

class NeuralMemory(nn.Module):
    """Titans-paper neural long-term memory (JAX/Flax).

    The memory IS the MLP weight tensor — retrieval is a forward pass,
    update is gradient descent on those weights.

    Uses setup() for projection layers (outer-optimiser trainables) and
    self.variable('memory_state', ...) inside retrieve()/update() for the
    inner MLP weights + momentum + surprise EMA.

    Attributes:
        d_model:         Input/output dim (= model hidden size).
        d_memory:        MLP internal dim. 0 → use d_model.
        n_memory_layers: Number of layers in memory MLP (>= 2 recommended).
        theta_lr:        Inner LR θ for memory weight updates.
        alpha_min:       Minimum forget rate.
        alpha_max:       Maximum forget rate (at maximum surprise).
        surprise_scale:  Steepness of sigmoid mapping surprise → forget rate.
        eta_fixed:       Fixed momentum decay η.
    """

    d_model: int
    d_memory: int = 0           # 0 → use d_model
    n_memory_layers: int = 4
    theta_lr: float = 0.01
    alpha_min: float = 0.0001
    alpha_max: float = 0.003
    surprise_scale: float = 3.0
    eta_fixed: float = 0.95

    def setup(self):
        _d_mem = self.d_memory if self.d_memory > 0 else self.d_model

        # Projection layers — trained by the outer optimiser (live in 'params')
        self.W_K = nn.Dense(_d_mem, use_bias=True,
                            kernel_init=nn.initializers.normal(0.02))
        self.W_V = nn.Dense(_d_mem, use_bias=True,
                            kernel_init=nn.initializers.normal(0.02))
        self.W_Q = nn.Dense(_d_mem, use_bias=True,
                            kernel_init=nn.initializers.normal(0.02))
        self.W_out = nn.Dense(self.d_model, use_bias=True,
                              kernel_init=nn.initializers.normal(0.02))

        # Query normalisation (non-learnable scale — frozen at 1.0)
        self.input_norm = nn.LayerNorm(epsilon=1e-5, use_bias=False,
                                       use_scale=False)

        # Memory state variable: MLP weights + momentum + surprise EMA.
        # Must be declared in setup() (not in a method) so Flax tracks it.
        self.mem_var = self.variable(
            "memory_state", "state", self._make_initial_state
        )

    @property
    def _d_mem(self) -> int:
        return self.d_memory if self.d_memory > 0 else self.d_model

    # ── memory_state helpers ──────────────────────────────────────────────────

    def _make_initial_state(self) -> dict:
        """Build zero-initialised memory state (called by self.variable init fn).

        Uses a hard-coded key — the actual initial MLP weights matter little
        since the memory starts empty and is filled at test-time via update().
        The small-std normal init (0.02) matches the PyTorch implementation.
        """
        _d_mem = self._d_mem
        key = jax.random.PRNGKey(42)  # deterministic init
        mlp_params = _mlp_init(key, _d_mem, _d_mem, _d_mem, self.n_memory_layers)
        n_p = _mlp_n_params(mlp_params)
        return {
            "mlp_params":   mlp_params,
            "momentum":     jnp.zeros((n_p,), dtype=jnp.float32),
            "surprise_ema": jnp.zeros((), dtype=jnp.float32),
            "ema_inited":   jnp.zeros((), dtype=jnp.bool_),
        }

    def _get_state_var(self):
        """Return the pre-declared memory_state variable (declared in setup)."""
        return self.mem_var

    # ── surprise-modulated forget rate (Python-land) ──────────────────────────

    def _compute_alpha(self, loss_val: float, state: dict) -> Tuple[float, dict]:
        """Map loss surprise to forget rate α; update running EMA.

        Runs in Python (not inside jit) so we use Python floats throughout.
        Returns (alpha, updated_state).
        """
        if math.isnan(loss_val) or math.isinf(loss_val) or loss_val < 0.0:
            return self.alpha_min, state

        ema_val = float(state["surprise_ema"])
        inited  = bool(state["ema_inited"])

        if not inited:
            ema_val = max(loss_val, 1e-8)
            new_state = {
                **state,
                "surprise_ema": jnp.array(ema_val, dtype=jnp.float32),
                "ema_inited":   jnp.array(True,    dtype=jnp.bool_),
            }
        else:
            ema_val = 0.99 * ema_val + 0.01 * loss_val
            new_state = {**state, "surprise_ema": jnp.array(ema_val, dtype=jnp.float32)}

        surprise_ratio = loss_val / max(ema_val, 1e-8)
        sig_in = max(-20.0, min(20.0, float(self.surprise_scale) * (surprise_ratio - 1.0)))
        sig_val = 1.0 / (1.0 + math.exp(-sig_in))
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * sig_val
        return alpha, new_state

    # ── public API ────────────────────────────────────────────────────────────

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Default call = retrieve. Forces param materialisation for W_K, W_V too.

        During model.init, Flax only initialises sub-modules that are actually
        called. We force W_K and W_V to exist by touching them here (discarding
        the result). This is the same pattern used in looped_model.py for scan
        body param materialisation.
        """
        # Force W_K, W_V params to be created (used in update(), not retrieve())
        _ = self.W_K(x)
        _ = self.W_V(x)
        return self.retrieve(x)

    def retrieve(self, x: jnp.ndarray) -> jnp.ndarray:
        """Retrieve from memory: y = W_out(M*(q(x))).

        MLP weights receive stop_gradient so outer-loop gradients don't flow
        through the memory MLP itself. W_Q and W_out ARE trained by the outer
        optimiser via the straight-through trick on the query vector.

        Args:
            x: [B, S, d_model]

        Returns:
            out: [B, S, d_model]
        """
        mem_var = self._get_state_var()

        q      = self.W_Q(x)              # [B, S, d_mem]
        q_norm = self.input_norm(q)       # [B, S, d_mem]

        # Stop gradient on memory weights — paper notation M* (Eq. 15)
        mlp_sg = jax.lax.stop_gradient(mem_var.value["mlp_params"])
        mem_out = _mlp_forward(mlp_sg, q_norm)    # [B, S, d_mem]

        # Straight-through estimator: adds zero to value but passes gradient to W_Q
        output = mem_out + (q_norm - jax.lax.stop_gradient(q_norm))

        return self.W_out(output)         # [B, S, d_model]

    def update(self, x: jnp.ndarray, scale: float = 1.0) -> None:
        """Update memory MLP weights using associative loss gradients.

        Implements Eqs. 12-14 from the Titans paper:
            k = W_K(x),  v = W_V(x)
            loss = ||M(k) - v||²                    (Eq. 12)
            S_t  = η * S_{t-1} - θ * ∇_W loss      (Eq. 13)
            W_t  = (1 - α) * W_{t-1} + S_t         (Eq. 14)

        Only writes if 'memory_state' is a mutable collection. Safe to call
        during a read-only apply (no-op) or mutable apply (performs update).

        Args:
            x:     [B, S, d_model] — hidden states to write into memory.
            scale: Warmup ramp multiplier (0.0 = off, 1.0 = full strength).
        """
        mem_var = self._get_state_var()

        if not self.is_mutable_collection("memory_state"):
            return  # read-only context (eval without mutable=['memory_state'])

        state      = mem_var.value
        mlp_params = state["mlp_params"]
        momentum   = state["momentum"]

        # Project to key / value space (outer-optimiser trained projections)
        k = self.W_K(x)   # [B, S, d_mem]
        v = self.W_V(x)   # [B, S, d_mem]

        # ── Associative loss and its gradient w.r.t. MLP weights ─────────────
        def _assoc_loss(params):
            pred = _mlp_forward(params, k)
            return jnp.mean((pred - v) ** 2)

        loss_arr, grads = jax.value_and_grad(_assoc_loss)(mlp_params)
        loss_val = float(loss_arr)

        # ── Gradient clipping ─────────────────────────────────────────────────
        flat_grad = _mlp_flat(grads)
        grad_norm = jnp.linalg.norm(flat_grad)
        flat_grad = jnp.where(grad_norm > 1.0, flat_grad / grad_norm, flat_grad)

        # NaN/Inf guard — skip this step silently
        if bool(jnp.any(jnp.isnan(flat_grad))) or bool(jnp.any(jnp.isinf(flat_grad))):
            return

        # Unflatten clipped gradient back into pytree matching mlp_params
        treedef = jax.tree_util.tree_structure(mlp_params)
        leaves  = jax.tree_util.tree_leaves(mlp_params)
        shapes  = [p.shape for p in leaves]
        clipped_leaves: list = []
        offset = 0
        for shape in shapes:
            n = math.prod(shape)
            clipped_leaves.append(flat_grad[offset:offset + n].reshape(shape))
            offset += n
        clipped_grads = jax.tree_util.tree_unflatten(treedef, clipped_leaves)

        # ── Surprise-modulated forget rate (Eq. 14 scalar α) ─────────────────
        alpha, state = self._compute_alpha(loss_val, state)

        # ── Momentum update: S_t = η * S_{t-1} - θ * clipped_grad  (Eq. 13) ──
        theta      = float(self.theta_lr) * scale
        flat_g     = _mlp_flat(clipped_grads)
        new_momentum = momentum * float(self.eta_fixed) - theta * flat_g

        # Unflatten new_momentum into pytree for per-weight update
        mom_leaves: list = []
        offset = 0
        for shape in shapes:
            n = math.prod(shape)
            mom_leaves.append(new_momentum[offset:offset + n].reshape(shape))
            offset += n
        new_mom_tree = jax.tree_util.tree_unflatten(treedef, mom_leaves)

        # ── Weight update: W_t = (1 - α) * W_{t-1} + S_t  (Eq. 14) ──────────
        new_mlp = jax.tree_util.tree_map(
            lambda w, s: (1.0 - alpha) * w + s,
            mlp_params,
            new_mom_tree,
        )

        # ── Write back to mutable variable ───────────────────────────────────
        mem_var.value = {
            **state,
            "mlp_params": new_mlp,
            "momentum":   new_momentum,
        }

    def reset_memory(self) -> None:
        """Reset momentum and surprise EMA (not MLP weights).

        Call at sequence boundaries during evaluation to prevent momentum
        bleed across unrelated sequences.
        """
        mem_var = self._get_state_var()
        if not self.is_mutable_collection("memory_state"):
            return
        state = mem_var.value
        n_p   = _mlp_n_params(state["mlp_params"])
        mem_var.value = {
            **state,
            "momentum":     jnp.zeros((n_p,), dtype=jnp.float32),
            "surprise_ema": jnp.zeros((), dtype=jnp.float32),
            "ema_inited":   jnp.zeros((), dtype=jnp.bool_),
        }
