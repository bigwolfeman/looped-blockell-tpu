"""Configuration for Looped Block-ELL Transformer."""

from dataclasses import dataclass, field


@dataclass
class LoopedBlockELLConfig:
    # Model dimensions
    d_model: int = 768
    n_heads: int = 12
    d_ff: int = 3072  # typically 4 * d_model
    n_layers: int = 6  # total unique layers
    vocab_size: int = 49152  # StarCoder2 tokenizer
    max_seq_len: int = 1024

    # Looping
    n_prelude: int = 1
    n_core: int = 4
    n_coda: int = 1
    mean_depth: int = 8
    max_depth: int = 32
    bptt_depth: int | None = None  # defaults to ceil(mean_depth/2)
    use_poisson: bool = True

    # Injection (SSM-style diagonal)
    init_decay: float = 0.447  # -log(2), spectral radius ~0.64

    # Outer SSM (cross-sequence state persistence)
    use_outer_ssm: bool = False
    outer_state_detach: bool = True  # stop_gradient on outer state by default
    outer_init_decay: float = 0.447  # separate decay for outer injection

    # Embeddings
    embed_geometry: str = "euclidean"  # euclidean | lorentz | hybrid
    lorentz_dim_fraction: float = 0.5  # fraction of d_model for Lorentz subspace (hybrid only)

    # XSA — Exclusive Self Attention (arXiv:2603.09078)
    use_xsa: bool = False
    # Attention Residuals (arXiv:2603.15031)
    use_attn_res: bool = False

    # Loop-boundary hyper-connections (Hyperloop arXiv:2604.21254)
    use_loop_boundary_hc: bool = False
    hc_n_streams: int = 4  # number of parallel residual streams

    # GQA (Grouped Query Attention)
    n_kv_heads: int | None = None  # None = same as n_heads (full MHA)

    # QK-Norm (Gemma 2 style — RMSNorm on Q and K per head_dim, before RoPE)
    use_qk_norm: bool = False

    # Attention
    use_sparse_attention: bool = False
    sparse_attn_type: str = "dsa"  # dsa (block-level) | csa (compressed sparse, V4-style)
    sparse_attn_top_k: int = 256
    sparse_attn_block_size: int = 32
    sparse_attn_n_indexer_heads: int = 4
    csa_compress_ratio: int = 8    # tokens per compressed KV entry
    csa_compress_stride: int = 4   # stride for overlapping compression windows
    csa_window_size: int = 128     # sliding window for recent uncompressed tokens

    # Block-ELL sparsity
    tile_size: int = 16
    macro_tile_size: int = 128  # 8×8 tiles for TPU MXU
    initial_density: float = 1.0

    # Pruning schedule
    prune_start: int = 0
    prune_end: int = 26000
    prune_frac: float = 0.10
    prune_interval: int = 2000
    score_interval: int = 10
    topology_interval: int = 100

    # Routing (Phase C)
    n_clusters: int = 16
    route_target_sparsity: float = 0.5
    route_warmup: int = 5000
    route_l1_weight: float = 0.01

    # Training
    lr: float = 6e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    total_steps: int = 50000
    batch_size: int = 8
    grad_clip: float = 1.0

    # MLP variant
    use_swiglu: bool = False  # SwiGLU (w_gate/w_up/w_down) vs GELU (fc1/fc2)
    # Note: when use_swiglu=True, d_ff is the intermediate dim for each of w_gate/w_up
    # (same as standard SwiGLU — total params per MLP: 3 * d_model * d_ff instead of 2)

    # Neural Long-Term Memory (Titans paper, arXiv:2501.00663)
    use_neural_memory: bool = False
    n_memory_layers: int = 4
    d_memory: int = 0            # 0 → use d_model
    memory_mode: str = "residual"  # residual | logit_bias (only residual implemented)
    memory_theta_lr: float = 0.01
    memory_alpha_min: float = 0.0001
    memory_alpha_max: float = 0.003
    memory_surprise_scale: float = 3.0
    memory_eta_fixed: float = 0.95
    memory_warmup_steps: int = 1000
    memory_ramp_steps: int = 4000
    memory_update_interval: int = 5   # update memory weights every N forward passes

    # Misc
    dropout: float = 0.0
    norm_eps: float = 1e-5
    tie_weights: bool = True
    embedding_scale: float = 1.0
    dtype: str = "bfloat16"

    def __post_init__(self):
        assert self.n_prelude + self.n_core + self.n_coda == self.n_layers
        assert self.d_model % self.n_heads == 0
        assert self.d_model % self.tile_size == 0
        assert self.d_ff % self.tile_size == 0
        assert self.macro_tile_size % self.tile_size == 0
        assert self.embed_geometry in ("euclidean", "lorentz", "hybrid")
        if self.embed_geometry == "hybrid":
            lorentz_dim = int(self.d_model * self.lorentz_dim_fraction)
            assert lorentz_dim > 0 and lorentz_dim < self.d_model
        if self.bptt_depth is None:
            self.bptt_depth = (self.mean_depth + 1) // 2
        if self.n_kv_heads is not None:
            assert self.n_heads % self.n_kv_heads == 0, \
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def tiles_per_macro(self) -> int:
        return self.macro_tile_size // self.tile_size

    @property
    def effective_depth(self) -> int:
        return self.n_prelude + self.n_core * self.mean_depth + self.n_coda
