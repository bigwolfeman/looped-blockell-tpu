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
        if self.bptt_depth is None:
            self.bptt_depth = (self.mean_depth + 1) // 2

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def tiles_per_macro(self) -> int:
        return self.macro_tile_size // self.tile_size

    @property
    def effective_depth(self) -> int:
        return self.n_prelude + self.n_core * self.mean_depth + self.n_coda
