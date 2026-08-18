import torch
import math
from positional_encodings import get_sinusoidal_encoding, precompute_rope_frequencies, apply_rope

def experiment_1_sinusoidal_waves():
    print("\n" + "="*60)
    print("🔬 EXPERIMENT 1: Early vs. Late Sinusoidal Dimensions")
    print("="*60)
    
    seq_len = 5
    d_model = 16
    pe = get_sinusoidal_encoding(seq_len, d_model).squeeze(0)  # Shape: (5, 16)
    
    print("\nNotice how Early Dim (0) oscillates rapidly across positions,")
    print("while Late Dim (14) changes very slowly:\n")
    print(f"{'Pos':<6} | {'Dim 0 (Fast Sine)':<20} | {'Dim 14 (Slow Sine)':<20}")
    print("-" * 52)
    for pos in range(seq_len):
        fast_val = pe[pos, 0].item()
        slow_val = pe[pos, 14].item()
        print(f"Pos {pos:<2} | {fast_val:^+20.4f} | {slow_val:^+20.4f}")


def experiment_2_rope_rotation_in_2d():
    print("\n" + "="*60)
    print("🌀 EXPERIMENT 2: Watching RoPE Rotate a 2D Vector")
    print("="*60)
    
    head_dim = 2  # Simple 2D arrow [x, y]
    max_seq_len = 5
    cos, sin = precompute_rope_frequencies(head_dim, max_seq_len)
    
    # Create a unit Query vector pointing along the X-axis: [1.0, 0.0]
    # Shape: (batch=1, heads=1, seq_len=1, head_dim=2)
    q_base = torch.tensor([[[[1.0, 0.0]]]])
    
    print("\nStarting Vector: [1.0, 0.0] (Length = 1.000)")
    print("Watch the vector spin as position increases:\n")
    print(f"{'Position':<8} | {'Rotated [x, y] Vector':<25} | {'Vector Norm (Length)':<20}")
    print("-" * 60)
    
    for pos in range(max_seq_len):
        # Apply RoPE for this specific position
        cos_pos = cos[pos:pos+1, :]
        sin_pos = sin[pos:pos+1, :]
        q_rotated = apply_rope(q_base, cos_pos, sin_pos).squeeze()
        
        x, y = q_rotated[0].item(), q_rotated[1].item()
        norm = torch.norm(q_rotated).item()
        print(f"Pos {pos:<5} | [{x:^+8.4f}, {y:^+8.4f}]           | {norm:<20.4f}")


def experiment_3_relative_distance_proof():
    print("\n" + "="*60)
    print("🎯 EXPERIMENT 3: The Relative Distance Magic (Dot Product)")
    print("="*60)
    
    head_dim = 8
    max_seq_len = 200
    cos, sin = precompute_rope_frequencies(head_dim, max_seq_len)
    
    # Create a random Query and Key vector
    torch.manual_seed(42)
    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)
    
    # Pair 1: Position 2 and Position 5 (Distance = 3 steps)
    q_pos2 = apply_rope(q, cos[2:3, :], sin[2:3, :])
    k_pos5 = apply_rope(k, cos[5:6, :], sin[5:6, :])
    score_near = torch.matmul(q_pos2, k_pos5.transpose(-2, -1)).item()
    
    # Pair 2: Position 102 and Position 105 (Distance STILL = 3 steps!)
    q_pos102 = apply_rope(q, cos[102:103, :], sin[102:103, :])
    k_pos105 = apply_rope(k, cos[105:106, :], sin[105:106, :])
    score_far = torch.matmul(q_pos102, k_pos105.transpose(-2, -1)).item()
    
    print("\nComparing Attention Match Scores for tokens 3 steps apart:")
    print(f"• Tokens at (Pos 2, Pos 5)   -> Dot Product Score: {score_near:.6f}")
    print(f"• Tokens at (Pos 102, Pos 105) -> Dot Product Score: {score_far:.6f}")
    
    if math.isclose(score_near, score_far, rel_tol=1e-5):
        print("\n🎉 MAGICAL PROOF: The attention match score depends ONLY on relative distance (3 steps)!")
        print("It does NOT matter where they sit in the document!")


if __name__ == "__main__":
    experiment_1_sinusoidal_waves()
    experiment_2_rope_rotation_in_2d()
    experiment_3_relative_distance_proof()