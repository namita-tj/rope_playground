import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from positional_encodings import get_sinusoidal_encoding, precompute_rope_frequencies, apply_rope


class CustomMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention built from scratch.
    Supports both Sinusoidal PE and RoPE (Rotary Position Embeddings).
    """
    def __init__(self, d_model: int, n_heads: int, pos_type: str = "rope", max_seq_len: int = 2048):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        assert pos_type in ["sinusoidal", "rope", "none"], "pos_type must be 'sinusoidal', 'rope', or 'none'"
        
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

    def forward(self, x: torch.Tensor, causal: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Multi-Head Attention.
        Input x shape: (Batch_Size, Sequence_Length, Hidden_Dimension)
        Returns: (Output Tensor, Attention Weights Map)
        """
        B, T, C = x.shape
        
        # If Sinusoidal, add positional waves directly to input embeddings BEFORE projection
        if self.pos_type == "sinusoidal":
            pe = get_sinusoidal_encoding(T, C).to(x.device)
            x = x + pe
            
        # Project -> Reshape (B, T, n_heads, head_dim) -> Transpose (B, n_heads, T, head_dim)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # If RoPE, apply vector rotation to Q and K AFTER projection
        if self.pos_type == "rope":
            q = apply_rope(q, self.rope_cos, self.rope_sin)
            k = apply_rope(k, self.rope_cos, self.rope_sin)
            
        # Multiply Q (B, h, T, d_k) by K^T (B, h, d_k, T) -> Scores (B, h, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply Causal Mask if autoregressive generation is requested
        if causal:
            mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
            scores = scores + mask
            
        attn_weights = F.softmax(scores, dim=-1)
        
        # Multiply Attention Weights (B, h, T, T) by V (B, h, T, d_v) -> (B, h, T, d_v)
        out = torch.matmul(attn_weights, v)
        
        # Transpose back (B, T, h, d_v) and flatten back to original embedding shape (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # Apply final output linear projection
        return self.out_proj(out), attn_weights