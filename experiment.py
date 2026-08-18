import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from positional_encodings import get_sinusoidal_encoding, precompute_rope_frequencies, apply_rope

def run_distance_decay_experiment(seq_len: int = 128, d_model: int = 64):
    """
    Measures how attention scores decay over relative token distance |i - j|
    comparing: No Positional Encoding, Sinusoidal PE, and RoPE.
    """
    torch.manual_seed(42)
    
    # Random Query and Key vectors [1, seq_len, d_model]
    Q = torch.randn(1, seq_len, d_model)
    K = torch.randn(1, seq_len, d_model)
    
    # 1. No Positional Encoding
    scores_none = torch.matmul(Q, K.transpose(-2, -1)) / (d_model ** 0.5)
    attn_none = F.softmax(scores_none, dim=-1).squeeze(0).detach().numpy()
    
    # 2. Sinusoidal Positional Encoding (Additive)
    PE = get_sinusoidal_encoding(seq_len, d_model).unsqueeze(0)  # [1, seq_len, d_model]
    Q_sin = Q + PE
    K_sin = K + PE
    scores_sin = torch.matmul(Q_sin, K_sin.transpose(-2, -1)) / (d_model ** 0.5)
    attn_sin = F.softmax(scores_sin, dim=-1).squeeze(0).detach().numpy()
    
    # 3. Rotary Position Embeddings (RoPE)
    cos, sin = precompute_rope_frequencies(seq_len, d_model)
    Q_rope = apply_rope(Q, cos, sin)
    K_rope = apply_rope(K, cos, sin)
    scores_rope = torch.matmul(Q_rope, K_rope.transpose(-2, -1)) / (d_model ** 0.5)
    attn_rope = F.softmax(scores_rope, dim=-1).squeeze(0).detach().numpy()
    
    # Compute average attention weight per relative distance |i - j|
    distances = list(range(seq_len))
    avg_none, avg_sin, avg_rope = [], [], []
    
    for d in distances:
        # Create boolean mask for token index pairs where |i - j| == d
        i_indices, j_indices = np.indices((seq_len, seq_len))
        mask = np.abs(i_indices - j_indices) == d
        
        avg_none.append(attn_none[mask].mean())
        avg_sin.append(attn_sin[mask].mean())
        avg_rope.append(attn_rope[mask].mean())
        
    # Generate visualization plot
    plt.figure(figsize=(10, 6))
    plt.plot(distances, avg_none, label="No Positional Encoding", linestyle="--", color="gray")
    plt.plot(distances, avg_sin, label="Sinusoidal PE (Additive)", color="blue")
    plt.plot(distances, avg_rope, label="RoPE (Rotary Rotation)", color="red", linewidth=2)
    
    plt.title("Relative Distance vs. Attention Weight Decay", fontsize=14)
    plt.xlabel("Relative Distance |pos_i - pos_j|", fontsize=12)
    plt.ylabel("Average Attention Weight", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    
    output_plot = "decay_plot.png"
    plt.savefig(output_plot, dpi=300, bbox_inches="tight")
    print(f"✅ Benchmark complete! Plot successfully generated and saved to '{output_plot}'.")

if __name__ == "__main__":
    run_distance_decay_experiment()