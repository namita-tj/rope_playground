import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from positional_encodings import get_sinusoidal_encoding, precompute_rope_frequencies, apply_rope

def main():
    print("\n>_ running sanity checks on encodings...\n")
    
    # 1. Test Sinusoidal Encodings Shape
    seq_len = 10
    d_model = 64
    sin_pe = get_sinusoidal_encoding(seq_len, d_model)
    print(f"[v] Sinusoidal Canvas Shape: {sin_pe.shape} (Expected: [10, 64])")
    
    # 2. Test RoPE Frequency Precomputation Shape
    head_dim = 32
    max_seq_len = 100
    cos, sin = precompute_rope_frequencies(head_dim, max_seq_len)
    
    # Note: head_dim is 32, so we expect 16 pairs!
    print(f"[v] RoPE Gears (Cos/Sin) Shape: {cos.shape} (Expected: [100, 16])")
    
    # 3. Test RoPE Application & Norm Preservation
    # Create a dummy Query tensor: batch=1, heads=2, seq_len=5, head_dim=32
    q = torch.randn(1, 2, 5, head_dim)
    
    # Slice our precomputed gears to match the sequence length of 5
    cos_sliced = cos[:5, :]
    sin_sliced = sin[:5, :]
    
    q_rotated = apply_rope(q, cos_sliced, sin_sliced)
    print(f"[v] Rotated Queries Shape: {q_rotated.shape} (Expected: [1, 2, 5, 32])")
    
    # Check that magnitude (norm) is preserved across all vectors
    norm_before = torch.norm(q, dim=-1)
    norm_after = torch.norm(q_rotated, dim=-1)
    
    print("\n(0_o) checking if the carousel broke any words...")
    
    # allclose checks if every single number matches perfectly (within a tiny 1e-5 margin of error)
    if torch.allclose(norm_before, norm_after, atol=1e-5):
        print("🎉 SUCCESS: RoPE perfectly preserved vector lengths (meaning is safe)!\n")
    else:
        print("❌ ERROR: Vector lengths changed during rotation. The words lost their meaning.\n")

if __name__ == "__main__":
    main()