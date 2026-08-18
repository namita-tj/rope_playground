import torch
from positional_encodings import get_sinusoidal_encoding, precompute_rope_frequencies, apply_rope


def main():
    print("--- Testing Step 1: Encodings ---")
    
    # 1. Test Sinusoidal Encodings Shape
    seq_len = 10
    d_model = 64
    sin_pe = get_sinusoidal_encoding(seq_len, d_model)
    print(f"✅ Sinusoidal PE Shape: {sin_pe.shape} (Expected: [1, 10, 64])")
    
    # 2. Test RoPE Frequency Precomputation Shape
    head_dim = 32
    max_seq_len = 100
    cos, sin = precompute_rope_frequencies(head_dim, max_seq_len)
    print(f"✅ RoPE Cos Shape: {cos.shape} (Expected: [100, 32])")
    
    # 3. Test RoPE Application & Norm Preservation
    # Create a dummy Query tensor: batch=1, heads=2, seq_len=5, head_dim=32
    q = torch.randn(1, 2, 5, head_dim)
    q_rotated = apply_rope(q, cos, sin)
    
    print(f"✅ Rotated Q Shape: {q_rotated.shape} (Expected: [1, 2, 5, 32])")
    
    # Check that magnitude (norm) is preserved across all vectors
    norm_before = torch.norm(q, dim=-1)
    norm_after = torch.norm(q_rotated, dim=-1)
    
    if torch.allclose(norm_before, norm_after, atol=1e-5):
        print("🎉 SUCCESS: RoPE perfectly preserved vector magnitudes (lengths)!")
    else:
        print("❌ ERROR: Vector magnitudes changed during rotation.")

if __name__ == "__main__":
    main()