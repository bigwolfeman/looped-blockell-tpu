"""PyTorch mirror of the JAX LoopedTransformer — identical architecture.

Every parameter name, shape, and computation matches the JAX/Flax model
so checkpoints can be converted losslessly between frameworks.

Architecture: Parcae-style looped transformer
  1. Embedding (weight-tied with LM head)
  2. Prelude blocks (non-looped)
  3. Input RMSNorm
  4. DiagonalInjection (SSM-style h = decay*h + dt*e)
  5. Core blocks × T iterations (looped, shared weights)
  6. Coda blocks (non-looped)
  7. Final RMSNorm + LM head (embed.weight.T)

Matches JAX param tree:
  embed.weight         ↔ params/embed/embedding
  prelude_i.*          ↔ params/prelude_i/*
  input_norm.scale     ↔ params/input_norm/scale
  injection.log_A/dt   ↔ params/injection/log_A, log_dt
  core_i.*             ↔ params/core_i/*
  coda_i.*             ↔ params/coda_i/*
  final_norm.scale     ↔ params/final_norm/scale
"""

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor
from torch.utils.checkpoint import checkpoint

# Add parent repo to path for titans_core imports
_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


@dataclass
class InteropConfig:
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    n_prelude: int = 4
    n_core: int = 8
    n_coda: int = 4
    vocab_size: int = 49152
    max_seq_len: int = 1024
    mean_depth: int = 6
    max_depth: int = 8
    bptt_depth: int | None = None
    use_poisson: bool = True
    init_decay: float = 0.447
    use_outer_ssm: bool = False
    outer_state_detach: bool = True
    outer_init_decay: float = 0.447
    embed_geometry: str = "euclidean"
    lorentz_dim_fraction: float = 0.5
    use_loop_boundary_hc: bool = False
    hc_type: str = "diagonal"  # diagonal | jpmhc
    hc_n_streams: int = 4
    # GQA (Grouped Query Attention)
    n_kv_heads: int | None = None  # None = same as n_heads (full MHA)
    # CLA (Cross-Layer Attention) — cache KV on loop iter 0, reuse for 1+
    use_cla: bool = False
    # QK-Norm (Gemma 2 style)
    use_qk_norm: bool = False
    # XSA — Exclusive Self Attention (arXiv:2603.09078)
    use_xsa: bool = False
    # Attention Residuals (arXiv:2603.15031)
    use_attn_res: bool = False
    attn_res_window: int = 0  # 0 = full gradient through all entries
    # Neural Memory (Titans, arXiv:2501.00663)
    use_neural_memory: bool = False
    n_memory_layers: int = 6
    d_memory: int = 1024  # hidden dim of memory MLP (~6M params with 6 layers at d=512→1024)
    memory_mode: str = "logit_bias"  # logit_bias | residual | append
    memory_theta_lr: float = 0.01
    memory_alpha_min: float = 0.0001
    memory_alpha_max: float = 0.003    # ~74% retained after 100 steps at max surprise
    memory_surprise_scale: float = 3.0
    memory_eta_fixed: float = 0.95
    use_sigreg: bool = False
    sigreg_lambda: float = 0.02
    use_differentiable_memory: bool = False
    memory_append_tokens: int = 8  # N for append mode
    memory_inner_steps: int = 1    # K inner gradient steps per forward pass
    memory_warmup_steps: int = 1000  # steps before memory activates
    memory_ramp_steps: int = 4000    # steps to linearly ramp memory from 0→1
    memory_update_interval: int = 1  # update MLP every N steps (retrieve every step)
    # SwiGLU activation (LLaMA-style)
    use_swiglu: bool = False
    # Multi-Token Prediction (arXiv:2404.19737)
    use_mtp: bool = False
    mtp_n_heads: int = 3    # predict t+1, t+2, t+3 (3 total)
    mtp_lambda: float = 0.1  # weight for auxiliary MTP losses
    # Block-ELL pruning (core blocks only)
    enable_pruning: bool = False
    tile_size: int = 16
    # ReMoE routing (after compaction)
    enable_routing: bool = False
    n_clusters: int = 16
    # CSA attention
    use_sparse_attention: bool = False
    sparse_attn_type: str = "csa"
    sparse_attn_top_k: int = 256
    sparse_attn_block_size: int = 32
    sparse_attn_n_indexer_heads: int = 4
    csa_compress_ratio: int = 8
    csa_compress_stride: int = 4
    csa_window_size: int = 128
    embedding_scale: float = 1.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_weights: bool = True
    use_checkpointing: bool = True
    # Training (stored in config for convenience)
    lr: float = 6e-4
    weight_decay: float = 0.1
    warmup_steps: int = 500
    total_steps: int = 15000
    batch_size: int = 20
    grad_clip: float = 1.0

    def __post_init__(self):
        if self.bptt_depth is None:
            self.bptt_depth = (self.mean_depth + 1) // 2

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def n_layers(self) -> int:
        return self.n_prelude + self.n_core + self.n_coda


# ─── Building blocks ──────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.scale


def _precompute_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> Tensor:
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)


def _apply_rope(x: Tensor, freqs: Tensor) -> Tensor:
    B, H, S, D = x.shape
    half = D // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = freqs[:S, :, 0].unsqueeze(0).unsqueeze(0)
    sin = freqs[:S, :, 1].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# ─── Lorentz embeddings ───────────────────────────────────────────────────────

EPS = 1e-6


def _project_to_hyperboloid(space: Tensor) -> Tensor:
    sq_norm = (space * space).sum(dim=-1, keepdim=True)
    x0 = torch.sqrt(torch.clamp(1.0 + sq_norm, min=EPS))
    return torch.cat([x0, space], dim=-1)


def _log_map_origin(x: Tensor) -> Tensor:
    x0 = x[..., :1]
    xs = x[..., 1:]
    alpha = torch.acosh(torch.clamp(x0, min=1.0 + EPS))
    denom = torch.sqrt(torch.clamp(x0 * x0 - 1.0, min=EPS))
    coeff = torch.where(denom < 1e-4, torch.ones_like(denom), alpha / denom)
    return coeff * xs


class LorentzEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, features: int):
        super().__init__()
        self.space_embed = nn.Embedding(num_embeddings, features)
        nn.init.normal_(self.space_embed.weight, std=0.005)
        self.features = features

    def forward(self, input_ids: Tensor) -> Tensor:
        space = self.space_embed(input_ids)
        hyp = _project_to_hyperboloid(space)
        return _log_map_origin(hyp)

    def attend(self, x: Tensor) -> Tensor:
        hyp = _project_to_hyperboloid(self.space_embed.weight)
        tangent_w = _log_map_origin(hyp)
        return x @ tangent_w.T


class HybridEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, euclidean_dim: int, lorentz_dim: int):
        super().__init__()
        self.euc_embed = nn.Embedding(num_embeddings, euclidean_dim)
        nn.init.normal_(self.euc_embed.weight, std=1.0 / math.sqrt(euclidean_dim))
        self.lor_embed = LorentzEmbedding(num_embeddings, lorentz_dim)

    def forward(self, input_ids: Tensor) -> Tensor:
        euc = self.euc_embed(input_ids)
        lor = self.lor_embed(input_ids)
        return torch.cat([euc, lor], dim=-1)

    def attend(self, x: Tensor) -> Tensor:
        euc_w = self.euc_embed.weight
        hyp = _project_to_hyperboloid(self.lor_embed.space_embed.weight)
        lor_w = _log_map_origin(hyp)
        full_w = torch.cat([euc_w, lor_w], dim=-1)
        return x @ full_w.T


# ─── CSA attention ────────────────────────────────────────────────────────────

class KVCompressor(nn.Module):
    def __init__(self, d_model: int, ratio: int = 8, stride: int = 4):
        super().__init__()
        self.ratio = ratio
        self.stride = stride
        self.gate_logits = nn.Parameter(torch.randn(ratio) * 0.02)

    def forward(self, x: Tensor) -> Tensor:
        B, L, D = x.shape
        n_windows = max(1, (L - self.ratio) // self.stride + 1)
        indices = (
            torch.arange(self.ratio, device=x.device).unsqueeze(0) +
            torch.arange(n_windows, device=x.device).unsqueeze(1) * self.stride
        )
        indices = indices.clamp(0, L - 1)
        windows = x[:, indices, :]  # [B, n_windows, ratio, D]
        gate_weights = F.softmax(self.gate_logits, dim=0)
        return torch.einsum("bwrd,r->bwd", windows, gate_weights)


class CSALightningIndexer(nn.Module):
    def __init__(self, d_model: int, n_indexer_heads: int = 4):
        super().__init__()
        self.n_indexer_heads = n_indexer_heads
        head_dim = d_model // n_indexer_heads
        self.wq = nn.Linear(d_model, n_indexer_heads * head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_indexer_heads * head_dim, bias=False)
        self.wg = nn.Linear(d_model, n_indexer_heads, bias=False)

    def forward(self, q: Tensor, k_compressed: Tensor) -> Tensor:
        head_dim = q.shape[-1] // self.n_indexer_heads
        B, Lq, _ = q.shape
        Lc = k_compressed.shape[1]
        qi = self.wq(q).reshape(B, Lq, self.n_indexer_heads, head_dim)
        ki = self.wk(k_compressed).reshape(B, Lc, self.n_indexer_heads, head_dim)
        w = self.wg(q)
        qk = torch.einsum("bqhd,bkhd->bqhk", qi, ki)
        qk = F.relu(qk)
        scores = torch.einsum("bqhk,bqh->bqk", qk, w)
        return scores


class CompressedSparseAttention(nn.Module):
    def __init__(self, cfg: InteropConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.d_model = cfg.d_model
        self.head_dim = cfg.head_dim
        self.top_k = cfg.sparse_attn_top_k
        self.window_size = cfg.csa_window_size

        self.wq = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.kv_compressor = KVCompressor(cfg.d_model, cfg.csa_compress_ratio, cfg.csa_compress_stride)
        self.wk_c = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wv_c = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.indexer = CSALightningIndexer(cfg.d_model, cfg.sparse_attn_n_indexer_heads)
        self.sink_logit = nn.Parameter(torch.zeros(cfg.n_heads, 1))

        self.register_buffer(
            "freqs", _precompute_freqs(cfg.head_dim, cfg.max_seq_len),
            persistent=False,
        )

    def forward(self, x: Tensor, deterministic: bool = True,
                memory_bias: Tensor | None = None) -> Tensor:
        B, L, D = x.shape
        hd = self.head_dim
        W = self.window_size

        q = self.wq(x).reshape(B, L, self.n_heads, hd).transpose(1, 2)
        k = self.wk(x).reshape(B, L, self.n_heads, hd).transpose(1, 2)
        v = self.wv(x).reshape(B, L, self.n_heads, hd).transpose(1, 2)

        q = _apply_rope(q.float(), self.freqs).to(x.dtype)
        k = _apply_rope(k.float(), self.freqs).to(x.dtype)

        n_compressed = max(1, (L - self.cfg.csa_compress_ratio) // self.cfg.csa_compress_stride + 1)
        if L <= W * 2 or n_compressed <= self.top_k:
            return self._full_causal(q, k, v, B, L)

        # Compress KV
        x_c = self.kv_compressor(x)
        Lc = x_c.shape[1]
        k_c = self.wk_c(x_c).reshape(B, Lc, self.n_heads, hd).transpose(1, 2)
        v_c = self.wv_c(x_c).reshape(B, Lc, self.n_heads, hd).transpose(1, 2)

        scale = math.sqrt(hd)

        # Compressed branch: all queries attend to causally-valid compressed entries
        # Each compressed entry j covers tokens [j*stride, j*stride+ratio).
        # Query i can only attend to entry j if j*stride + ratio - 1 <= i
        # (the entire source window must be at or before the query position).
        compressed_scores = torch.matmul(q, k_c.transpose(-2, -1)) / scale  # [B, H, L, Lc]
        stride = self.cfg.csa_compress_stride
        ratio = self.cfg.csa_compress_ratio
        entry_end = torch.arange(Lc, device=x.device) * stride + ratio - 1  # [Lc]
        query_pos = torch.arange(L, device=x.device)  # [L]
        causal_c = query_pos[:, None] >= entry_end[None, :]  # [L, Lc]
        compressed_scores = compressed_scores.masked_fill(~causal_c[None, None], float("-inf"))
        n_select = Lc

        # Sliding window via masked full attention (memory-efficient for L ≤ 4k)
        window_scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # [B, H, L, L]
        pos = torch.arange(L, device=x.device)
        # dist[i,j] = i - j: positive means key j is before query i (causal)
        dist = pos[:, None] - pos[None, :]  # [L, L]
        window_mask = (dist >= 0) & (dist < W)  # causal + window
        window_scores = window_scores.masked_fill(~window_mask[None, None], float("-inf"))

        # Attention sink
        sink_broadcast = self.sink_logit[None, :, None, :].expand(B, self.n_heads, L, 1)

        # Combined softmax over [compressed | window | sink]
        all_scores = torch.cat([
            compressed_scores.float(), window_scores.float(), sink_broadcast.float()
        ], dim=-1)  # [B, H, L, Lc + L + 1]
        attn_w = F.softmax(all_scores, dim=-1).to(x.dtype)

        w_compressed = attn_w[..., :n_select]
        w_window = attn_w[..., n_select:n_select + L]

        out_compressed = torch.matmul(w_compressed, v_c)
        out_window = torch.matmul(w_window, v)
        out = out_compressed + out_window

        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.wo(out)

    def _full_causal(self, q, k, v, B, L):
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        mask = torch.triu(torch.full((L, L), float("-inf"), device=q.device, dtype=q.dtype), diagonal=1)
        scores = scores + mask[None, None]
        w = F.softmax(scores.float(), dim=-1).to(q.dtype)
        out = torch.matmul(w, v)
        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.wo(out)


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: InteropConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads if cfg.n_kv_heads is not None else cfg.n_heads
        self.d_model = cfg.d_model
        self.head_dim = cfg.head_dim
        self.use_xsa = cfg.use_xsa
        self.kv_groups = self.n_heads // self.n_kv_heads

        if self.n_kv_heads == self.n_heads:
            self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        else:
            kv_dim = self.n_kv_heads * self.head_dim
            self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            self.kv_proj = nn.Linear(cfg.d_model, 2 * kv_dim, bias=False)

        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.register_buffer(
            "freqs", _precompute_freqs(cfg.head_dim, cfg.max_seq_len),
            persistent=False,
        )
        self.dropout = cfg.dropout

        self.use_qk_norm = getattr(cfg, 'use_qk_norm', False)
        if self.use_qk_norm:
            self.q_norm = RMSNorm(cfg.head_dim)
            self.k_norm = RMSNorm(cfg.head_dim)

        self.memory_mode = cfg.memory_mode if cfg.use_neural_memory else None

        # Memory logit bias: zero-init projection so it starts as no-op
        if cfg.use_neural_memory and cfg.memory_mode == "logit_bias":
            self.mem_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            nn.init.zeros_(self.mem_proj.weight)
        else:
            self.mem_proj = None

        # Append mode: KV projection for memory tokens
        if cfg.use_neural_memory and cfg.memory_mode == "append":
            self.mem_k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            self.mem_v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: Tensor, deterministic: bool = True,
                memory_bias: Tensor | None = None,
                cached_kv: tuple[Tensor, Tensor] | None = None) -> Tensor | tuple[Tensor, tuple[Tensor, Tensor]]:
        B, S, D = x.shape

        if self.n_kv_heads == self.n_heads:
            qkv = self.qkv_proj(x)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        else:
            q = self.q_proj(x).reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)
            kv = self.kv_proj(x)
            k, v = kv.chunk(2, dim=-1)
            k = k.reshape(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v = v.reshape(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = _apply_rope(q.float(), self.freqs).to(x.dtype)
        k = _apply_rope(k.float(), self.freqs).to(x.dtype)

        # CLA: use cached KV if provided (already GQA-expanded)
        if cached_kv is not None:
            k, v = cached_kv
        elif self.kv_groups > 1:
            k = k.repeat_interleave(self.kv_groups, dim=1)
            v = v.repeat_interleave(self.kv_groups, dim=1)

        # Memory logit bias: shift Q based on memory retrieval
        if self.mem_proj is not None and memory_bias is not None:
            mb = self.mem_proj(memory_bias)  # [B, S, d_model]
            mb = mb.reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)
            q = q + mb

        # Append mode: prepend memory tokens to KV (always visible)
        N_mem = 0
        if self.memory_mode == "append" and memory_bias is not None:
            mem_k = self.mem_k_proj(memory_bias)  # [B, S, D]
            mem_v = self.mem_v_proj(memory_bias)  # [B, S, D]
            N_mem = min(memory_bias.shape[1], 8)
            mem_k = mem_k[:, :N_mem].reshape(B, N_mem, self.n_heads, self.head_dim).transpose(1, 2)
            mem_v = mem_v[:, :N_mem].reshape(B, N_mem, self.n_heads, self.head_dim).transpose(1, 2)
            k = torch.cat([mem_k, k], dim=2)
            v = torch.cat([mem_v, v], dim=2)

        # Fast path: SDPA with fused kernel (no append tokens, handles GQA natively)
        if N_mem == 0 and not self.use_xsa:
            drop_p = self.dropout if (not deterministic and self.dropout > 0) else 0.0
            if self.kv_groups > 1 and cached_kv is None:
                # GQA: undo the repeat_interleave, let SDPA handle it
                k_gqa = k[:, ::self.kv_groups]
                v_gqa = v[:, ::self.kv_groups]
                out = F.scaled_dot_product_attention(
                    q, k_gqa, v_gqa, is_causal=True, dropout_p=drop_p,
                    enable_gqa=True,
                )
            else:
                out = F.scaled_dot_product_attention(
                    q, k, v, is_causal=True, dropout_p=drop_p,
                )
            out = out.transpose(1, 2).reshape(B, S, D)
            return self.out_proj(out)

        # Slow path: manual attention (append tokens or XSA)
        scale = math.sqrt(self.head_dim)
        S_kv = k.shape[2]
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale

        mask = torch.triu(
            torch.full((S, S_kv), float("-inf"), device=x.device, dtype=x.dtype),
            diagonal=1 + N_mem,
        )
        if N_mem > 0:
            mask[:, :N_mem] = 0.0
        attn = attn + mask.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(attn.float(), dim=-1).to(x.dtype)

        if self.dropout > 0.0 and not deterministic:
            attn = F.dropout(attn, p=self.dropout, training=True)

        out = torch.matmul(attn, v)

        if self.use_xsa:
            v_norm_sq = (v * v).sum(-1, keepdim=True).clamp(min=1e-8)
            proj = (out * v).sum(-1, keepdim=True) / v_norm_sq
            out = out - proj * v

        out = out.transpose(1, 2).reshape(B, S, D)
        return self.out_proj(out)


class PrunableLinear(nn.Module):
    """Linear layer with 16x16 tile-level pruning + Block-ELL compaction.

    Pre-compact: dense weight [out, in] with tile mask. O(out*in) compute.
    Post-compact: Block-ELL [R, K_new, B, B] with col_indices. O(R*K_new*B^2) compute.
    """

    def __init__(self, in_features: int, out_features: int, tile_size: int = 16, bias: bool = False):
        super().__init__()
        assert in_features % tile_size == 0 and out_features % tile_size == 0
        self.in_features = in_features
        self.out_features = out_features
        self.tile_size = tile_size
        self.R = out_features // tile_size
        self.C = in_features // tile_size

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self.register_buffer("tile_mask", torch.ones(self.R, self.C, dtype=torch.bool))
        self.register_buffer("score_ema", torch.zeros(self.R, self.C))
        self._compacted = False
        self._all_alive = True
        self._mask_expanded: Tensor | None = None

    def _expand_mask(self) -> Tensor:
        if self._mask_expanded is None:
            B = self.tile_size
            self._mask_expanded = (
                self.tile_mask.float()
                .unsqueeze(1).unsqueeze(3)
                .expand(self.R, B, self.C, B)
                .reshape(self.out_features, self.in_features)
            )
        return self._mask_expanded

    def forward(self, x: Tensor) -> Tensor:
        if self._compacted:
            return self._forward_bell(x)
        if self._all_alive:
            return F.linear(x, self.weight, self.bias)
        return F.linear(x, self.weight * self._expand_mask(), self.bias)

    def _forward_bell(self, x: Tensor) -> Tensor:
        squeezed = x.dim() == 2
        if squeezed:
            x = x.unsqueeze(1)
        Bsz, S, _ = x.shape
        B = self.tile_size
        x_blocks = x.reshape(Bsz, S, self.C, B)
        x_gathered = x_blocks[:, :, self.col_indices.long(), :]
        out = torch.einsum("bsrki,rkoi->bsrko", x_gathered, self.values)
        out = out.sum(dim=3).reshape(Bsz, S, self.out_features)
        if self.bias is not None:
            out = out + self.bias
        if squeezed:
            out = out.squeeze(1)
        return out

    def accumulate_scores(self):
        if self.weight.grad is None or self._compacted:
            return
        B = self.tile_size
        grad_tiles = self.weight.grad.detach().reshape(self.R, B, self.C, B)
        tile_norms = grad_tiles.pow(2).sum(dim=(1, 3)).sqrt()
        with torch.no_grad():
            self.score_ema.lerp_(tile_norms, 0.05)

    def prune_fraction(self, frac: float) -> tuple[int, int, float]:
        """Kill bottom frac of alive tiles. Returns (killed, was_alive, new_density)."""
        with torch.no_grad():
            alive = self.tile_mask
            n_alive = alive.sum().item()
            n_total = self.R * self.C
            if n_alive <= 1:
                return 0, n_alive, n_alive / n_total

            scores = self.score_ema.clone()
            scores[~alive] = float('inf')
            n_kill = max(1, int(n_alive * frac))
            n_kill = min(n_kill, n_alive - 1)

            threshold = scores[alive].topk(n_alive, largest=False).values[n_kill - 1]
            kill = alive & (scores <= threshold)
            actual_killed = kill.sum().item()
            if actual_killed > n_kill:
                indices = kill.nonzero()
                keep = indices[n_kill:]
                for idx in keep:
                    kill[idx[0], idx[1]] = False
                actual_killed = n_kill

            self.tile_mask[kill] = False
            self._all_alive = False
            self._mask_expanded = None

            mask_exp = self._expand_mask()
            self.weight.data *= mask_exp

            new_alive = self.tile_mask.sum().item()
            return actual_killed, n_alive, new_alive / n_total

    def rezero_dead(self):
        if self._compacted or self._all_alive:
            return
        with torch.no_grad():
            self.weight.data *= self._expand_mask()

    @property
    def density(self) -> float:
        if self._compacted:
            return self._compact_density
        return self.tile_mask.float().mean().item()

    def compact(self, n_clusters: int | None = None) -> int:
        """Rebuild as Block-ELL with compacted K. Returns K_new."""
        with torch.no_grad():
            B = self.tile_size
            alive = self.tile_mask
            alive_per_row = alive.sum(dim=1)
            K_new = max(1, int(alive_per_row.min().item()))

            scores = self.score_ema.clone()
            scores[~alive] = -float('inf')
            _, top_idx = scores.topk(K_new, dim=1, largest=True)

            col_indices = torch.zeros(self.R, K_new, dtype=torch.int32,
                                      device=self.weight.device)
            values = torch.zeros(self.R, K_new, B, B,
                                 device=self.weight.device, dtype=self.weight.dtype)
            w_tiles = self.weight.reshape(self.R, B, self.C, B).permute(0, 2, 1, 3)

            for r in range(self.R):
                for k_new, k_old in enumerate(top_idx[r]):
                    col_indices[r, k_new] = k_old.to(torch.int32)
                    values[r, k_new] = w_tiles[r, k_old]

            del self._parameters["weight"]
            self.values = nn.Parameter(values)
            self.register_buffer("col_indices", col_indices)
            self._compacted = True
            self._compact_density = K_new / self.C
            self.K = K_new

            if n_clusters is not None:
                rows_per = self.R // n_clusters
                remainder = self.R % n_clusters
                starts, ends = [], []
                row = 0
                for c in range(n_clusters):
                    starts.append(row)
                    row += rows_per + (1 if c < remainder else 0)
                    ends.append(row)
                self.n_clusters = n_clusters
                self.register_buffer("cluster_starts",
                                     torch.tensor(starts, dtype=torch.int32))
                self.register_buffer("cluster_ends",
                                     torch.tensor(ends, dtype=torch.int32))

            return K_new


class MLPBlock(nn.Module):
    def __init__(self, cfg: InteropConfig, prunable: bool = False):
        super().__init__()
        self.use_swiglu = getattr(cfg, 'use_swiglu', False)
        self.dropout = cfg.dropout
        self.prunable = prunable and cfg.enable_pruning

        if self.prunable and self.use_swiglu:
            B = cfg.tile_size
            self.w_gate = PrunableLinear(cfg.d_model, cfg.d_ff, B, bias=False)
            self.w_up = PrunableLinear(cfg.d_model, cfg.d_ff, B, bias=False)
            self.w_down = PrunableLinear(cfg.d_ff, cfg.d_model, B, bias=False)
        elif self.use_swiglu:
            self.w_gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
            self.w_up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
            self.w_down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        else:
            self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=True)
            self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=True)

    def forward(self, x: Tensor, deterministic: bool = True) -> Tensor:
        if self.use_swiglu:
            out = self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
        else:
            out = self.fc2(F.gelu(self.fc1(x)))
        if self.dropout > 0.0 and not deterministic:
            out = F.dropout(out, p=self.dropout, training=True)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, cfg: InteropConfig, prunable: bool = False):
        super().__init__()
        self.norm_attn = RMSNorm(cfg.d_model, cfg.norm_eps)
        if cfg.use_sparse_attention and cfg.sparse_attn_type == "csa":
            self.attention = CompressedSparseAttention(cfg)
        else:
            self.attention = MultiHeadAttention(cfg)
        self.norm_mlp = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = MLPBlock(cfg, prunable=prunable)

    def forward(self, x: Tensor, deterministic: bool = True,
                memory_bias: Tensor | None = None,
                cached_kv: tuple[Tensor, Tensor] | None = None) -> Tensor:
        if cached_kv is not None and isinstance(self.attention, MultiHeadAttention):
            x = x + self.attention(self.norm_attn(x), deterministic,
                                   memory_bias=memory_bias, cached_kv=cached_kv)
        else:
            x = x + self.attention(self.norm_attn(x), deterministic,
                                   memory_bias=memory_bias)
        x = x + self.mlp(self.norm_mlp(x), deterministic)
        return x


class DiagonalInjection(nn.Module):
    def __init__(self, d_model: int, init_decay: float = 0.447):
        super().__init__()
        self.log_A = nn.Parameter(torch.zeros(d_model))
        target_dt = -math.log(init_decay)
        init_log_dt = math.log(math.exp(target_dt) - 1.0)
        self.log_dt = nn.Parameter(torch.full((d_model,), init_log_dt))

    def forward(self, h: Tensor, e: Tensor) -> Tensor:
        A = -torch.exp(self.log_A)
        dt = F.softplus(self.log_dt)
        decay = torch.exp(dt * A)
        return decay * h + dt * e


class LoopBoundaryHC(nn.Module):
    """Original diagonal HC (kept for backwards compat)."""
    def __init__(self, d_model: int, n_streams: int = 4):
        super().__init__()
        self.n_streams = n_streams
        self.alpha = nn.Parameter(torch.ones(n_streams) / n_streams)
        self.beta_res = nn.Parameter(torch.ones(n_streams) * 0.9)
        self.beta_out = nn.Parameter(torch.ones(n_streams) * 0.1)

    def aggregate(self, streams: Tensor) -> Tensor:
        weights = F.softmax(self.alpha, dim=0)
        return (streams * weights[None, None, :, None]).sum(dim=2)

    def distribute(self, streams: Tensor, h_new: Tensor) -> Tensor:
        return (self.beta_res[None, None, :, None] * streams +
                self.beta_out[None, None, :, None] * h_new.unsqueeze(2))


def _cayley_iterative(H_raw: Tensor, alpha: float = 0.1, steps: int = 2) -> Tensor:
    """Iterative Cayley transform: unconstrained n×n → approximately orthogonal.

    Skew-symmetrize then fixed-point iterate. O(s * n^3) per call.
    For n=4, s=2: 128 FLOPs — negligible.
    """
    W = H_raw - H_raw.transpose(-2, -1)  # skew-symmetric
    I = torch.eye(W.shape[-1], device=W.device, dtype=W.dtype)
    # Broadcast I to match batch dims
    for _ in range(W.dim() - 2):
        I = I.unsqueeze(0)
    I = I.expand_as(W)
    Y = I + alpha * W
    for _ in range(steps):
        Y = I + (alpha / 2) * torch.matmul(W, I + Y)
    return Y


class JPmHC(nn.Module):
    """Cayley-orthogonal hyper-connections (arXiv:2602.18308).

    Full n×n cross-stream mixing with orthogonal residual (norm-preserving).
    Input-conditional: H_pre, H_post, H_res computed per-token from stream state.
    """
    def __init__(self, d_model: int, n_streams: int = 4, cayley_alpha: float = 0.1, cayley_steps: int = 2):
        super().__init__()
        self.n = n_streams
        self.d = d_model
        self.cayley_alpha = cayley_alpha
        self.cayley_steps = cayley_steps
        n = n_streams
        # Fused projection: stream state → 3 n×n matrices
        self.norm = RMSNorm(n * d_model)
        self.proj = nn.Linear(n * d_model, 3 * n * n, bias=True)
        # Small init so gradients flow from step 1; bias zeros → near-identity start
        nn.init.normal_(self.proj.weight, std=0.01)
        nn.init.zeros_(self.proj.bias)
        # Output gate scale (initialized small)
        self.out_scale = nn.Parameter(torch.ones(1) * 0.1)

    def _compute_matrices(self, streams: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute per-token H_pre, H_post, H_res from stream state."""
        B, S, n, d = streams.shape
        x_flat = streams.reshape(B, S, n * d)
        raw = self.proj(self.norm(x_flat))  # [B, S, 3*n*n]
        raw = raw.reshape(B, S, 3, self.n, self.n)

        H_pre_raw = raw[:, :, 0]   # [B, S, n, n]
        H_post_raw = raw[:, :, 1]
        H_res_raw = raw[:, :, 2]

        H_pre = F.softmax(H_pre_raw, dim=-1)     # row-stochastic
        H_post = F.softmax(H_post_raw, dim=-2)    # column-stochastic
        H_res = _cayley_iterative(H_res_raw, self.cayley_alpha, self.cayley_steps)

        return H_pre, H_post, H_res

    def aggregate(self, streams: Tensor) -> Tensor:
        """streams: [B, S, n, d] → single input [B, S, d]."""
        H_pre, _, _ = self._compute_matrices(streams)
        # H_pre @ streams: [B, S, n, n] @ [B, S, n, d] → [B, S, n, d]
        mixed = torch.matmul(H_pre, streams)
        return mixed.mean(dim=2)  # [B, S, d]

    def distribute(self, streams: Tensor, h_new: Tensor) -> Tensor:
        """Update streams with layer output. streams: [B, S, n, d], h_new: [B, S, d]."""
        _, H_post, H_res = self._compute_matrices(streams)
        # Orthogonal residual: preserves norm across loop iterations
        residual = torch.matmul(H_res, streams)  # [B, S, n, n] @ [B, S, n, d]
        # Distribute output to streams via column-stochastic H_post
        # H_post: [B, S, n, n], sum over columns = 1
        # Broadcast h_new to all streams then gate
        h_broadcast = h_new.unsqueeze(2).expand_as(streams)  # [B, S, n, d]
        output = torch.matmul(H_post, h_broadcast)  # [B, S, n, d]
        return residual + self.out_scale * output


# ─── Triton kernels for fused depth attention ─────────────────────────────────

@triton.jit
def _depth_attn_fwd_kernel(
    buf_ptr, scale_ptr, query_ptr, out_ptr,
    N,
    stride_buf_t, stride_buf_bs, stride_buf_d,
    stride_out_bs, stride_out_d,
    stride_scale, stride_query,
    D: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    q = tl.load(query_ptr + d_offs * stride_query, mask=d_mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + d_offs * stride_scale, mask=d_mask, other=1.0).to(tl.float32)

    m_prev = tl.full([], float('-inf'), dtype=tl.float32)
    l_prev = tl.zeros([], dtype=tl.float32)
    o_acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for t in range(N):
        base = t * stride_buf_t + pid * stride_buf_bs
        v = tl.load(buf_ptr + base + d_offs * stride_buf_d, mask=d_mask, other=0.0).to(tl.float32)
        rms = tl.sqrt(tl.sum(v * v) / D + 1e-6)
        k = v / rms * scale
        score = tl.sum(k * q)
        m_new = tl.maximum(m_prev, score)
        correction = tl.exp(m_prev - m_new)
        p = tl.exp(score - m_new)
        l_prev = correction * l_prev + p
        o_acc = correction * o_acc + p * v
        m_prev = m_new

    o_acc = o_acc / l_prev
    tl.store(out_ptr + pid * stride_out_bs + d_offs * stride_out_d, o_acc.to(tl.bfloat16), mask=d_mask)



@triton.jit
def _depth_attn_bwd_kernel(
    buf_ptr, scale_ptr, query_ptr, weights_ptr, grad_out_ptr,
    grad_buf_ptr, grad_scale_ptr, grad_query_ptr,
    N,
    stride_buf_t, stride_buf_bs, stride_buf_d,
    stride_go_bs, stride_go_d,
    stride_scale, stride_query,
    stride_w_t, stride_w_bs,
    D: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    go = tl.load(grad_out_ptr + pid * stride_go_bs + d_offs * stride_go_d,
                 mask=d_mask, other=0.0).to(tl.float32)
    q = tl.load(query_ptr + d_offs * stride_query, mask=d_mask, other=0.0).to(tl.float32)
    s = tl.load(scale_ptr + d_offs * stride_scale, mask=d_mask, other=1.0).to(tl.float32)

    # First pass: compute dw_t = dot(go, V_t) and go_dot_out = sum_t w_t * dw_t
    go_dot_out = tl.zeros([], dtype=tl.float32)
    for t in range(N):
        w_t = tl.load(weights_ptr + t * stride_w_t + pid * stride_w_bs).to(tl.float32)
        base = t * stride_buf_t + pid * stride_buf_bs
        v = tl.load(buf_ptr + base + d_offs * stride_buf_d, mask=d_mask, other=0.0).to(tl.float32)
        dw_t = tl.sum(go * v)
        go_dot_out += w_t * dw_t

    # Second pass: compute all gradients
    dq_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    ds_acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for t in range(N):
        w_t = tl.load(weights_ptr + t * stride_w_t + pid * stride_w_bs).to(tl.float32)
        base = t * stride_buf_t + pid * stride_buf_bs
        v = tl.load(buf_ptr + base + d_offs * stride_buf_d, mask=d_mask, other=0.0).to(tl.float32)

        # d_logit via softmax backward
        dw_t = tl.sum(go * v)
        d_logit = w_t * (dw_t - go_dot_out)

        # RMSNorm forward for this entry
        rms = tl.sqrt(tl.sum(v * v) / D + 1e-6)
        v_normed = v / rms
        k = v_normed * s

        dq_acc += d_logit * k
        ds_acc += d_logit * q * v_normed

        dv_sum = w_t * go
        dk = d_logit * q
        dk_s = dk * s
        inner = tl.sum(dk_s * v_normed) / D
        dv_norm = (dk_s - v_normed * inner) / rms

        dv_total = dv_sum + dv_norm
        tl.store(grad_buf_ptr + base + d_offs * stride_buf_d,
                 dv_total.to(tl.bfloat16), mask=d_mask)

    # Write partials to per-program buffers (no atomic contention)
    tl.store(grad_query_ptr + pid * stride_buf_bs + d_offs * stride_buf_d,
             dq_acc.to(tl.bfloat16), mask=d_mask)
    tl.store(grad_scale_ptr + pid * stride_buf_bs + d_offs * stride_buf_d,
             ds_acc.to(tl.bfloat16), mask=d_mask)


class _DepthAttnFn(torch.autograd.Function):
    """Custom autograd for depth attention with Triton forward kernel.

    Takes stacked values [N, BS, D] as input (with gradient), runs Triton
    kernel for forward, and computes analytical backward in PyTorch.
    """
    @staticmethod
    def forward(ctx, values_flat, norm_scale, query, N):
        """values_flat: [N, BS, D], norm_scale: [D], query: [D]"""
        BS, D = values_flat.shape[1], values_flat.shape[2]
        out = torch.empty(BS, D, device=values_flat.device, dtype=values_flat.dtype)
        BLOCK_D = triton.next_power_of_2(D)

        _depth_attn_fwd_kernel[(BS,)](
            values_flat, norm_scale, query, out,
            N,
            values_flat.stride(0), values_flat.stride(1), values_flat.stride(2),
            out.stride(0), out.stride(1),
            norm_scale.stride(0), query.stride(0),
            D=D, BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=1,
        )

        # Recompute weights in fp32 for backward (tiny cost for N≤10)
        with torch.no_grad():
            vf = values_flat[:N].float()
            rms = vf.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
            keys = vf / rms * norm_scale.float().unsqueeze(0).unsqueeze(0)
            logits = (keys * query.float().unsqueeze(0).unsqueeze(0)).sum(-1)  # [N, BS]
            weights = F.softmax(logits, dim=0)  # [N, BS]

        ctx.save_for_backward(values_flat, norm_scale, query, weights)
        ctx.N = N
        return out

    @staticmethod
    def backward(ctx, grad_out):
        values_flat, norm_scale, query, weights = ctx.saved_tensors
        N = ctx.N
        BS, D = grad_out.shape

        grad_values = torch.zeros_like(values_flat)
        # Partial buffers: each of BS programs writes its local dq/ds [D]
        grad_query_partial = torch.empty(BS, D, device=grad_out.device, dtype=torch.bfloat16)
        grad_scale_partial = torch.empty(BS, D, device=grad_out.device, dtype=torch.bfloat16)

        BLOCK_D = triton.next_power_of_2(D)
        _depth_attn_bwd_kernel[(BS,)](
            values_flat, norm_scale, query, weights, grad_out,
            grad_values, grad_scale_partial, grad_query_partial,
            N,
            values_flat.stride(0), values_flat.stride(1), values_flat.stride(2),
            grad_out.stride(0), grad_out.stride(1),
            norm_scale.stride(0), query.stride(0),
            weights.stride(0), weights.stride(1),
            D=D, BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=1,
        )

        # Reduce partials → [D] (single cuBLAS-level sum, microseconds)
        grad_query = grad_query_partial.float().sum(0)
        grad_scale = grad_scale_partial.float().sum(0)

        return grad_values, grad_scale.to(norm_scale.dtype), grad_query.to(query.dtype), None


class AttentionResidual(nn.Module):
    """Depth-wise attention over block outputs (arXiv:2603.15031).

    Triton fused forward + analytical backward via _DepthAttnFn.
    Single autograd node per call — no graph explosion from stacking entries.
    """
    def __init__(self, d_model: int, max_blocks: int):
        super().__init__()
        self.max_blocks = max_blocks
        self.proj = nn.Parameter(torch.zeros(max_blocks, d_model))
        self.norm = RMSNorm(d_model)
        self._entries: list[Tensor] = []

    def reset(self, B: int = 0, S: int = 0):
        self._entries = []

    def append(self, h: Tensor):
        self._entries.append(h.detach())

    def forward(self, block_idx: int, window: int = 0) -> Tensor:
        n = len(self._entries)
        if n <= 1:
            return self._entries[0]
        w = self.proj[min(block_idx, self.max_blocks - 1)]
        V = torch.stack(self._entries, dim=0)  # [n, B, S, D]
        N, Bsz, S, D = V.shape
        V_flat = V.reshape(N, Bsz * S, D).contiguous()
        out = _DepthAttnFn.apply(V_flat, self.norm.scale, w, N)
        return out.reshape(Bsz, S, D)


# ─── SIGReg (anti-collapse regularization for memory outputs) ────────────────

@torch.autocast("cuda", enabled=False)
def sigreg_loss(z: Tensor) -> Tensor:
    """SIGReg: penalizes representational collapse in memory outputs.

    Computes covariance of z, takes singular values, and penalizes
    them via -log(sigmoid(σ)). This pushes singular values to be large
    and spread out, preventing collapse to a low-rank subspace.
    """
    z_flat = z.reshape(-1, z.shape[-1]).float()
    z_centered = z_flat - z_flat.mean(0, keepdim=True)
    n = z_centered.shape[0]
    cov = z_centered.T @ z_centered / max(n - 1, 1)
    k = min(64, cov.shape[0])
    s = torch.linalg.svdvals(cov)[:k]
    return -torch.log(torch.sigmoid(s) + 1e-8).mean()


# ─── Multi-Token Prediction heads ─────────────────────────────────────────────

class MTPHead(nn.Module):
    """Single MTP prediction head: proj → norm → shared unembedding."""
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = RMSNorm(d_model)

    def forward(self, h: Tensor) -> Tensor:
        return self.norm(self.proj(h))


# ─── Main model ───────────────────────────────────────────────────────────────

class LoopedTransformerPT(nn.Module):
    """PyTorch mirror of JAX LoopedTransformer."""

    def __init__(self, cfg: InteropConfig):
        super().__init__()
        self.cfg = cfg

        # 1. Embedding
        self.embed_geometry = cfg.embed_geometry
        if cfg.embed_geometry == "lorentz":
            self.embed = LorentzEmbedding(cfg.vocab_size, cfg.d_model)
        elif cfg.embed_geometry == "hybrid":
            lorentz_dim = int(cfg.d_model * cfg.lorentz_dim_fraction)
            euclidean_dim = cfg.d_model - lorentz_dim
            self.embed = HybridEmbedding(cfg.vocab_size, euclidean_dim, lorentz_dim)
        else:
            self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
            nn.init.normal_(self.embed.weight, std=0.02)

        # 2. Prelude
        self.prelude = nn.ModuleList([
            TransformerBlock(cfg) for _ in range(cfg.n_prelude)
        ])

        # 3. Input norm
        self.input_norm = RMSNorm(cfg.d_model, cfg.norm_eps)

        # 4. Injection
        self.injection = DiagonalInjection(cfg.d_model, cfg.init_decay)

        # 4b. Outer injection (cross-sequence)
        if cfg.use_outer_ssm:
            self.outer_injection = DiagonalInjection(cfg.d_model, cfg.outer_init_decay)

        # 5. Core blocks (shared across loop iterations, prunable when enabled)
        self.core = nn.ModuleList([
            TransformerBlock(cfg, prunable=True) for _ in range(cfg.n_core)
        ])

        # 6. Iteration embedding (Phase C routing)
        self.iteration_embed = nn.Embedding(cfg.max_depth, cfg.d_model)
        nn.init.normal_(self.iteration_embed.weight, std=0.001)

        # 7. Loop-boundary hyper-connections
        if cfg.use_loop_boundary_hc:
            if cfg.hc_type == "jpmhc":
                self.loop_hc = JPmHC(cfg.d_model, cfg.hc_n_streams)
            else:
                self.loop_hc = LoopBoundaryHC(cfg.d_model, cfg.hc_n_streams)

        # 7b. Attention Residuals (depth-wise attention over all block outputs)
        if cfg.use_attn_res:
            max_blocks = 1 + cfg.max_depth + 1
            self.attn_res = AttentionResidual(cfg.d_model, max_blocks)

        # 7c. Neural Memory (Titans, arXiv:2501.00663)
        if cfg.use_neural_memory:
            from titans_core.memory.neural_memory import NeuralMemory
            self.neural_memory = NeuralMemory(
                d_model=cfg.d_model,
                d_memory=cfg.d_memory,
                n_memory_layers=cfg.n_memory_layers,
                theta_lr=cfg.memory_theta_lr,
                alpha_min=cfg.memory_alpha_min,
                alpha_max=cfg.memory_alpha_max,
                surprise_scale=cfg.memory_surprise_scale,
                eta_fixed=cfg.memory_eta_fixed,
            )
            # Project memory output back to d_model when d_memory != d_model
            d_mem = cfg.d_memory if cfg.d_memory != cfg.d_model else cfg.d_model
            if d_mem != cfg.d_model:
                self.mem_out_proj = nn.Linear(d_mem, cfg.d_model, bias=False)
            else:
                self.mem_out_proj = None

            # Residual mode: zero-init scale + projection
            if cfg.memory_mode in ("residual", "append_residual"):
                self.mem_res_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
                nn.init.zeros_(self.mem_res_proj.weight)
                self.mem_res_norm = RMSNorm(cfg.d_model)
                self.mem_res_scale = nn.Parameter(torch.zeros(1))

        # 8. Coda
        self.coda = nn.ModuleList([
            TransformerBlock(cfg) for _ in range(cfg.n_coda)
        ])

        # 9. Final norm (no separate lm_head — weight-tied)
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)

        # 10. MTP heads (share unembedding via embed.weight)
        if cfg.use_mtp:
            self.mtp_heads = nn.ModuleList([
                MTPHead(cfg.d_model) for _ in range(cfg.mtp_n_heads - 1)
            ])

    def forward(
        self,
        input_ids: Tensor,
        depths: Tensor,
        n_max: int,
        k_max: int,
        labels: Tensor | None = None,
        deterministic: bool = True,
        use_iter_embed: bool = False,
        outer_state: Tensor | None = None,
        step: int = 0,
    ) -> dict:
        cfg = self.cfg
        total_iters = n_max + k_max
        B, S = input_ids.shape

        # Memory warmup: off for first N steps, linear ramp after
        use_mem = cfg.use_neural_memory
        mem_scale = 1.0
        if use_mem:
            if step < cfg.memory_warmup_steps:
                use_mem = False
            elif step < cfg.memory_warmup_steps + cfg.memory_ramp_steps:
                mem_scale = (step - cfg.memory_warmup_steps) / max(cfg.memory_ramp_steps, 1)

        # 1. Embedding
        if self.embed_geometry in ("lorentz", "hybrid"):
            x = self.embed(input_ids) * cfg.embedding_scale
        else:
            x = self.embed(input_ids) * cfg.embedding_scale

        # 2. Prelude
        for blk in self.prelude:
            x = blk(x, deterministic)

        # 2b. Neural memory: retrieve ONCE before core loop
        mem_out = None
        if use_mem:
            mem_out = self.neural_memory.retrieve(x)  # [B, S, d_memory]
            if self.mem_out_proj is not None:
                mem_out = self.mem_out_proj(mem_out)   # [B, S, d_model]
            if mem_scale < 1.0:
                mem_out = mem_out * mem_scale

        # AttnRes: record prelude output as block 0
        use_ar = cfg.use_attn_res
        ar_idx = 1
        if use_ar:
            self.attn_res.reset(B, S)
            self.attn_res.append(x)

        # 3. Input norm
        e = self.input_norm(x)

        # 3b. Outer SSM injection
        if cfg.use_outer_ssm and outer_state is not None:
            os = outer_state
            if cfg.outer_state_detach:
                os = os.detach()
            e = self.outer_injection(os, e)

        # 4. Recurrent state init (truncated normal, Parcae-style)
        h = torch.randn_like(e) * 0.02

        # 5. Core loop
        use_hc = cfg.use_loop_boundary_hc
        if use_hc:
            streams = h.unsqueeze(2).expand(-1, -1, cfg.hc_n_streams, -1).clone()
        else:
            streams = None

        # Memory bias for attention (logit_bias, append, append_residual modes pass to core blocks)
        mem_bias_for_attn = mem_out if (use_mem and cfg.memory_mode in ("logit_bias", "append", "append_residual")) else None

        for t in range(total_iters):
            if use_hc:
                h_agg = self.loop_hc.aggregate(streams)
            else:
                h_agg = h

            h_new = self.injection(h_agg, e)

            # Residual mode: add memory output after injection
            if use_mem and cfg.memory_mode in ("residual", "append_residual") and mem_out is not None:
                h_new = h_new + self.mem_res_scale * self.mem_res_norm(self.mem_res_proj(mem_out))

            if use_iter_embed:
                t_clamped = min(t, cfg.max_depth - 1)
                iter_vec = self.iteration_embed(
                    torch.tensor(t_clamped, device=input_ids.device)
                )
                h_new = h_new + iter_vec

            if cfg.use_cla and t > 0 and hasattr(self, '_cla_cache'):
                # CLA: reuse KV from iteration 0, only recompute Q
                for i, blk in enumerate(self.core):
                    h_new = blk(h_new, deterministic, memory_bias=mem_bias_for_attn,
                                cached_kv=self._cla_cache[i])
            else:
                def _core_body(h_in, _mem_bias=mem_bias_for_attn):
                    for blk in self.core:
                        h_in = blk(h_in, deterministic, memory_bias=_mem_bias)
                    return h_in

                if cfg.use_checkpointing and self.training:
                    h_new = checkpoint(_core_body, h_new, use_reentrant=False)
                else:
                    h_new = _core_body(h_new)

                # CLA: cache KV from first iteration
                if cfg.use_cla and t == 0:
                    self._cla_cache = []
                    with torch.no_grad():
                        for blk in self.core:
                            attn = blk.attention
                            h_normed = blk.norm_attn(h_new)
                            if attn.n_kv_heads == attn.n_heads:
                                qkv = attn.qkv_proj(h_normed)
                                _, k_c, v_c = qkv.chunk(3, dim=-1)
                                k_c = k_c.reshape(B, S, attn.n_heads, attn.head_dim).transpose(1, 2)
                                v_c = v_c.reshape(B, S, attn.n_heads, attn.head_dim).transpose(1, 2)
                            else:
                                kv = attn.kv_proj(h_normed)
                                k_c, v_c = kv.chunk(2, dim=-1)
                                k_c = k_c.reshape(B, S, attn.n_kv_heads, attn.head_dim).transpose(1, 2)
                                v_c = v_c.reshape(B, S, attn.n_kv_heads, attn.head_dim).transpose(1, 2)
                            if attn.use_qk_norm:
                                k_c = attn.k_norm(k_c)
                            k_c = _apply_rope(k_c.float(), attn.freqs).to(h_new.dtype)
                            if attn.kv_groups > 1:
                                k_c = k_c.repeat_interleave(attn.kv_groups, dim=1)
                                v_c = v_c.repeat_interleave(attn.kv_groups, dim=1)
                            self._cla_cache.append((k_c, v_c))

            if use_ar:
                self.attn_res.append(h_new)
                ar_idx += 1

            if t < n_max:
                h_new = h_new.detach()

            active = (t < depths).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
            h = torch.where(active, h_new, h_agg)

            if use_hc:
                active_4d = active.unsqueeze(-1)  # [B, 1, 1, 1]
                streams_new = self.loop_hc.distribute(streams, h_new)
                streams = torch.where(active_4d, streams_new, streams)

        if use_hc:
            h = streams.mean(dim=2)

        # 7. Save h_final for outer SSM
        h_final = h

        # 7b. Neural memory: K inner gradient steps on h_final
        #     Retrieve every step, but update MLP weights every N steps
        memory_loss = None
        sigreg_val = None
        if use_mem:
            should_update = (step % cfg.memory_update_interval == 0)
            if should_update:
                h_for_mem = h_final.detach()
                for _ in range(cfg.memory_inner_steps):
                    memory_loss = self.neural_memory.update(
                        h_for_mem, return_stats=False,
                        differentiable=cfg.use_differentiable_memory,
                    )
            if cfg.use_sigreg:
                mem_check = self.neural_memory.retrieve(h_final)
                sigreg_val = sigreg_loss(mem_check)

        # 8. Coda
        if use_ar:
            x = self.attn_res(ar_idx, window=cfg.attn_res_window)
        else:
            x = h
        for blk in self.coda:
            x = blk(x, deterministic)

        # 9. Final norm + LM head (weight-tied)
        x = self.final_norm(x)

        # 10. Loss — chunked CE to avoid materializing full [B*S, V] logits
        loss = None
        mtp_loss = None
        logits = None
        if labels is not None:
            n_chunks = max(1, (B * S * cfg.vocab_size * 4) // (1 << 30))  # ~1GB per chunk
            if n_chunks <= 1:
                if self.embed_geometry in ("lorentz", "hybrid"):
                    logits = self.embed.attend(x)
                else:
                    logits = F.linear(x, self.embed.weight)
                task_loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), labels.reshape(-1))
            else:
                x_chunks = x.chunk(n_chunks, dim=1)
                l_chunks = labels.chunk(n_chunks, dim=1)
                ce_sum = torch.tensor(0.0, device=x.device, dtype=torch.float32)
                total_tokens = 0
                for xc, lc in zip(x_chunks, l_chunks):
                    if self.embed_geometry in ("lorentz", "hybrid"):
                        lgt = self.embed.attend(xc)
                    else:
                        lgt = F.linear(xc, self.embed.weight)
                    ce_sum = ce_sum + F.cross_entropy(
                        lgt.reshape(-1, cfg.vocab_size), lc.reshape(-1), reduction='sum')
                    total_tokens += lc.numel()
                task_loss = ce_sum / total_tokens
                logits = None  # not needed for training
            loss = task_loss
        else:
            if self.embed_geometry in ("lorentz", "hybrid"):
                logits = self.embed.attend(x)
            else:
                logits = F.linear(x, self.embed.weight)

            # MTP: auxiliary losses for predicting t+2, t+3, ...
            if cfg.use_mtp:
                # x_pre_norm is the hidden state before final_norm (reuse for MTP heads)
                # Each head k predicts token at position t+k+1 (head 0 = main = t+1)
                mtp_losses = []
                for k, head in enumerate(self.mtp_heads):
                    shift = k + 1  # head 0 predicts t+2, head 1 predicts t+3
                    if shift >= labels.shape[1]:
                        break
                    h_mtp = head(x)  # proj + norm on post-coda hidden states
                    if self.embed_geometry in ("lorentz", "hybrid"):
                        mtp_logits = self.embed.attend(h_mtp)
                    else:
                        mtp_logits = F.linear(h_mtp, self.embed.weight)
                    # Shift: predict labels[:, shift:] from logits[:, :-shift]
                    mtp_l = F.cross_entropy(
                        mtp_logits[:, :-shift].reshape(-1, cfg.vocab_size),
                        labels[:, shift:].reshape(-1),
                    )
                    mtp_losses.append(mtp_l)
                if mtp_losses:
                    mtp_loss = sum(mtp_losses) / len(mtp_losses)
                    loss = loss + cfg.mtp_lambda * mtp_loss

            if sigreg_val is not None:
                loss = loss + cfg.sigreg_lambda * sigreg_val

        return {
            "logits": logits,
            "loss": loss,
            "outer_state_out": h_final,
            "depth_meta": {"t_max": total_iters, "n_max": n_max, "k_max": k_max},
            "memory_loss": memory_loss,
            "mtp_loss": mtp_loss,
            "sigreg_loss": sigreg_val,
        }

    def get_prunable_modules(self) -> list[tuple[int, str, PrunableLinear]]:
        """Return (core_idx, name, module) for all PrunableLinear in core blocks."""
        mods = []
        for i, blk in enumerate(self.core):
            mlp = blk.mlp
            if hasattr(mlp, 'prunable') and mlp.prunable:
                for name in ("w_gate", "w_up", "w_down"):
                    m = getattr(mlp, name, None)
                    if isinstance(m, PrunableLinear):
                        mods.append((i, name, m))
        return mods


# ─── Depth sampling (mirrors JAX depth_sampler.py) ─────────────────────────

@dataclass
class DepthPlan:
    total: Tensor   # [B] int — per-sequence total iterations
    n_max: int      # Python int — max no-grad steps
    k_max: int      # Python int — max grad steps


def sample_depth(
    batch_size: int,
    mean_depth: int,
    max_depth: int = 32,
    bptt_depth: int | None = None,
    device: torch.device = torch.device("cpu"),
) -> DepthPlan:
    if bptt_depth is None:
        bptt_depth = (mean_depth + 1) // 2
    raw = torch.poisson(torch.full((batch_size,), float(mean_depth), device=device))
    total = raw.clamp(1, max_depth).to(torch.int32)
    k = torch.clamp(total, max=bptt_depth)
    n = total - k
    return DepthPlan(
        total=total,
        n_max=int(n.max().item()),
        k_max=int(k.max().item()),
    )


def sample_fixed(
    batch_size: int,
    mean_depth: int,
    bptt_depth: int | None = None,
    device: torch.device = torch.device("cpu"),
) -> DepthPlan:
    if bptt_depth is None:
        bptt_depth = (mean_depth + 1) // 2
    total = torch.full((batch_size,), mean_depth, dtype=torch.int32, device=device)
    k = min(mean_depth, bptt_depth)
    n = max(0, mean_depth - bptt_depth)
    return DepthPlan(total=total, n_max=n, k_max=k)
