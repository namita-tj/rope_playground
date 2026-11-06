import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from positional_encodings import (
    apply_rope,
    get_alibi_bias,
    get_sinusoidal_encoding,
    precompute_rope_frequencies,
    rope_slice,
    scaled_rope_frequencies,
)

POS_TYPES = ("none", "sinusoidal", "rope", "alibi")


class CustomMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention built from scratch.
    Supports Sinusoidal PE, RoPE (Rotary Position Embeddings), ALiBi (linear distance
    penalties) or no position information at all.
    """
    def __init__(self, d_model: int, n_heads: int, pos_type: str = "rope", max_seq_len: int = 2048):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        assert pos_type in POS_TYPES, f"pos_type must be one of {POS_TYPES}"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.pos_type = pos_type

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Precompute RoPE rotation buffers if RoPE is active
        if self.pos_type == "rope":
            cos, sin = precompute_rope_frequencies(self.head_dim, max_seq_len)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def set_rope_scaling(self, method: str | None = None, factor: float = 1.0, original_max_len: int = 256) -> None:
        """Swaps in stretched RoPE gears (PI, NTK or YaRN); method=None restores plain RoPE."""
        if self.pos_type != "rope":
            raise ValueError(f"RoPE scaling only applies to pos_type='rope', not {self.pos_type!r}")
        cos, sin = scaled_rope_frequencies(self.head_dim, self.rope_cos.shape[0], method, factor, original_max_len)
        self.rope_cos = cos.to(self.rope_cos.device)
        self.rope_sin = sin.to(self.rope_sin.device)

    def forward(
        self, x: torch.Tensor, causal: bool = False, offset: int = 0, need_weights: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass for Multi-Head Attention.
        Input x shape: (Batch_Size, Sequence_Length, Hidden_Dimension)
        `offset` is the position of the first token (0 unless continuing a sequence).
        `need_weights=False` skips building the attention map and uses PyTorch's fused
        attention kernel instead: same result, much faster for training.
        Returns: (Output Tensor, Attention Weights Map or None)
        """
        B, T, C = x.shape
        if C != self.d_model:
            raise ValueError(f"expected hidden size {self.d_model}, got {C}")

        # If Sinusoidal, add positional waves directly to input embeddings BEFORE projection
        if self.pos_type == "sinusoidal":
            pe = get_sinusoidal_encoding(offset + T, C)[offset:].to(x.device)
            x = x + pe

        # Project -> Reshape (B, T, n_heads, head_dim) -> Transpose (B, n_heads, T, head_dim)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # If RoPE, apply vector rotation to Q and K AFTER projection
        if self.pos_type == "rope":
            # Only the gears for these T positions, not the whole precomputed table
            cos, sin = rope_slice(self.rope_cos, self.rope_sin, T, offset)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        # Everything added to the scores before softmax: ALiBi's distance penalty and the causal mask
        bias = None
        if self.pos_type == "alibi":
            # Only distances matter, so the offset changes nothing here
            bias = get_alibi_bias(self.n_heads, T, device=x.device)
        if causal:
            mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
            bias = mask if bias is None else bias + mask

        if not need_weights:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
            return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, C)), None

        # Multiply Q (B, h, T, d_k) by K^T (B, h, d_k, T) -> Scores (B, h, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if bias is not None:
            scores = scores + bias

        attn_weights = F.softmax(scores, dim=-1)

        # Multiply Attention Weights (B, h, T, T) by V (B, h, T, d_v) -> (B, h, T, d_v)
        out = torch.matmul(attn_weights, v)

        # Transpose back (B, T, h, d_v) and flatten back to original embedding shape (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Apply final output linear projection
        return self.out_proj(out), attn_weights
