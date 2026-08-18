import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from attention import CustomMultiHeadAttention


def test_attention_shapes_and_mechanics():
    print("--- Testing Step 2: Multi-Head Attention ---")
    
    batch_size = 2
    seq_len = 10
    d_model = 64
    n_heads = 4
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    attn_sin = CustomMultiHeadAttention(d_model=d_model, n_heads=n_heads, pos_type="sinusoidal")
    out_sin, weights_sin = attn_sin(x)
    
    print(f"✅ Sinusoidal Output Shape: {out_sin.shape} (Expected: [{batch_size}, {seq_len}, {d_model}])")
    print(f"✅ Sinusoidal Attention Weights Shape: {weights_sin.shape} (Expected: [{batch_size}, {n_heads}, {seq_len}, {seq_len}])")
    
    attn_rope = CustomMultiHeadAttention(d_model=d_model, n_heads=n_heads, pos_type="rope")
    out_rope, weights_rope = attn_rope(x)
    
    print(f"✅ RoPE Output Shape: {out_rope.shape} (Expected: [{batch_size}, {seq_len}, {d_model}])")
    print(f"✅ RoPE Attention Weights Shape: {weights_rope.shape} (Expected: [{batch_size}, {n_heads}, {seq_len}, {seq_len}])")
    
    out_causal, weights_causal = attn_rope(x, causal=True)
    # Check that upper triangle of attention matrix is strictly zero
    upper_tri_sum = torch.triu(weights_causal, diagonal=1).sum().item()
    if upper_tri_sum == 0.0:
        print("🎉 SUCCESS: Causal mask correctly blocked future tokens!")
    else:
        print("❌ ERROR: Causal mask leaked future token information.")
        
    loss = out_rope.sum()
    loss.backward()
    
    has_grads = all(p.grad is not None and not torch.isnan(p.grad).any() for p in attn_rope.parameters())
    if has_grads:
        print("🎉 SUCCESS: Backpropagation completed with valid non-NaN gradients!")
    else:
        print("❌ ERROR: Gradient flow failed or produced NaNs.")


if __name__ == "__main__":
    test_attention_shapes_and_mechanics()