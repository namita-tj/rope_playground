import torch
import math

def get_sinusoidal_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """
    Generates Vaswani Sinusoidal Positional Encodings.
    
    Returns:
        Tensor of shape (1, seq_len, d_model)
    """
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe.unsqueeze(0)


def precompute_rope_frequencies(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precomputes Cosine and Sine rotation angles for RoPE.
    
    Returns:
        (cos, sin) tuple of Tensors, each of shape (max_seq_len, head_dim)
    """
    assert head_dim % 2 == 0, "Head dimension must be an even number."
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float)
    angles = torch.outer(positions, freqs)
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies Rotary Position Embedding (RoPE) to a Query or Key tensor.
    
    Input shape x: (batch_size, n_heads, seq_len, head_dim)
    Returns: Rotated tensor of exact same shape as x.
    """
    seq_len = x.shape[2]
    head_dim = x.shape[3]
    
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(1)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(1)
    
    x1 = x[..., :head_dim // 2]
    x2 = x[..., head_dim // 2:]
    
    x_tilde = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (x_tilde * sin)