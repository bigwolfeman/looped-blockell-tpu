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
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor
from torch.utils.checkpoint import checkpoint


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
    # XSA — Exclusive Self Attention (arXiv:2603.09078)
    use_xsa: bool = False
    # Attention Residuals (arXiv:2603.15031)
    use_attn_res: bool = False
    attn_res_window: int = 0  # 0 = full gradient through all entries
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

    def forward(self, x: Tensor, deterministic: bool = True) -> Tensor:
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
        self.d_model = cfg.d_model
        self.head_dim = cfg.head_dim
        self.use_xsa = cfg.use_xsa
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.register_buffer(
            "freqs", _precompute_freqs(cfg.head_dim, cfg.max_seq_len),
            persistent=False,
        )
        self.dropout = cfg.dropout

    def forward(self, x: Tensor, deterministic: bool = True) -> Tensor:
        B, S, D = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def reshape(t):
            return t.reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        q = _apply_rope(q.float(), self.freqs).to(x.dtype)
        k = _apply_rope(k.float(), self.freqs).to(x.dtype)

        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale

        mask = torch.triu(torch.full((S, S), float("-inf"), device=x.device, dtype=x.dtype), diagonal=1)
        attn = attn + mask.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(attn.float(), dim=-1).to(x.dtype)

        if self.dropout > 0.0 and not deterministic:
            attn = F.dropout(attn, p=self.dropout, training=True)

        out = torch.matmul(attn, v)  # [B, H, S, d_head]

        # XSA: subtract projection onto own value vector (arXiv:2603.09078)
        if self.use_xsa:
            v_norm_sq = (v * v).sum(-1, keepdim=True).clamp(min=1e-8)
            proj = (out * v).sum(-1, keepdim=True) / v_norm_sq
            out = out - proj * v

        out = out.transpose(1, 2).reshape(B, S, D)
        return self.out_proj(out)


class MLPBlock(nn.Module):
    def __init__(self, cfg: InteropConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=True)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=True)
        self.dropout = cfg.dropout

    def forward(self, x: Tensor, deterministic: bool = True) -> Tensor:
        h = F.gelu(self.fc1(x))
        out = self.fc2(h)
        if self.dropout > 0.0 and not deterministic:
            out = F.dropout(out, p=self.dropout, training=True)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, cfg: InteropConfig):
        super().__init__()
        self.norm_attn = RMSNorm(cfg.d_model, cfg.norm_eps)
        if cfg.use_sparse_attention and cfg.sparse_attn_type == "csa":
            self.attention = CompressedSparseAttention(cfg)
        else:
            self.attention = MultiHeadAttention(cfg)
        self.norm_mlp = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = MLPBlock(cfg)

    def forward(self, x: Tensor, deterministic: bool = True) -> Tensor:
        x = x + self.attention(self.norm_attn(x), deterministic)
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

    Triton fused forward + analytical backward via custom autograd.Function.
    Zero-init queries for uniform initial attention (paper spec).
    Single autograd node per call — eliminates graph traversal overhead.
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
        self._entries.append(h)

    def forward(self, block_idx: int, window: int = 0) -> Tensor:
        n = len(self._entries)
        if n <= 1:
            return self._entries[0]
        w = self.proj[min(block_idx, self.max_blocks - 1)]
        V = torch.stack(self._entries, dim=0)  # [n, B, S, D]
        K = self.norm(V)
        logits = torch.einsum("d, nbsd -> nbs", w, K)
        weights = F.softmax(logits, dim=0)
        return torch.einsum("nbs, nbsd -> bsd", weights, V)


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

        # 5. Core blocks (shared across loop iterations)
        self.core = nn.ModuleList([
            TransformerBlock(cfg) for _ in range(cfg.n_core)
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
            # prelude=1 block + max_depth iterations + coda=1 block
            max_blocks = 1 + cfg.max_depth + 1
            self.attn_res = AttentionResidual(cfg.d_model, max_blocks)

        # 8. Coda
        self.coda = nn.ModuleList([
            TransformerBlock(cfg) for _ in range(cfg.n_coda)
        ])

        # 9. Final norm (no separate lm_head — weight-tied)
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)

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
    ) -> dict:
        cfg = self.cfg
        total_iters = n_max + k_max
        B, S = input_ids.shape

        # 1. Embedding
        if self.embed_geometry in ("lorentz", "hybrid"):
            x = self.embed(input_ids) * cfg.embedding_scale
        else:
            x = self.embed(input_ids) * cfg.embedding_scale

        # 2. Prelude
        for blk in self.prelude:
            x = blk(x, deterministic)

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

        for t in range(total_iters):
            if use_hc:
                h_agg = self.loop_hc.aggregate(streams)
            elif use_ar:
                h_agg = self.attn_res(ar_idx, window=cfg.attn_res_window)
            else:
                h_agg = h

            h_new = self.injection(h_agg, e)

            if use_iter_embed:
                t_clamped = min(t, cfg.max_depth - 1)
                iter_vec = self.iteration_embed(
                    torch.tensor(t_clamped, device=input_ids.device)
                )
                h_new = h_new + iter_vec

            def _core_body(h_in):
                for blk in self.core:
                    h_in = blk(h_in, deterministic)
                return h_in

            if cfg.use_checkpointing and self.training:
                h_new = checkpoint(_core_body, h_new, use_reentrant=False)
            else:
                h_new = _core_body(h_new)

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

        # 8. Coda
        if use_ar:
            x = self.attn_res(ar_idx, window=cfg.attn_res_window)
        else:
            x = h
        for blk in self.coda:
            x = blk(x, deterministic)

        # 9. Final norm + LM head (weight-tied)
        x = self.final_norm(x)
        if self.embed_geometry in ("lorentz", "hybrid"):
            logits = self.embed.attend(x)
        else:
            logits = F.linear(x, self.embed.weight)  # [B, S, vocab_size]

        # 10. Loss
        loss = None
        if labels is not None:
            task_loss = F.cross_entropy(
                logits.reshape(-1, cfg.vocab_size),
                labels.reshape(-1),
            )
            loss = task_loss

        return {
            "logits": logits,
            "loss": loss,
            "outer_state_out": h_final,
            "depth_meta": {"t_max": total_iters, "n_max": n_max, "k_max": k_max},
        }


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
