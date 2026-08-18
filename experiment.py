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
    
    # 3D base tensors: [batch_size=1, seq_len, d_model]
    Q = torch.randn(1, seq_len, d_model)
    K = torch.randn(1, seq_len, d_model)
    
    # 1. No Positional Encoding
    scores_none = torch.matmul(Q, K.transpose(-2, -1)) / (d_model ** 0.5)
    # Ensure 2D matrix shape [seq_len, seq_len]
    attn_none = F.softmax(scores_none, dim=-1).view(seq_len, seq_len).detach().numpy()
    
    # 2. Sinusoidal Positional Encoding (Additive)
    PE = get_sinusoidal_encoding(seq_len, d_model).unsqueeze(0)  # [1, seq_len, d_model]
    Q_sin = Q + PE
    K_sin = K + PE
    scores_sin = torch.matmul(Q_sin, K_sin.transpose(-2, -1)) / (d_model ** 0.5)
    # Ensure 2D matrix shape [seq_len, seq_len]
    attn_sin = F.softmax(scores_sin, dim=-1).view(seq_len, seq_len).detach().numpy()
    
    # 3. Rotary Position Embeddings (RoPE)
    # Convert to 4D: [batch_size=1, num_heads=1, seq_len, head_dim=d_model]
    Q_4d = Q.unsqueeze(1)
    K_4d = K.unsqueeze(1)
    
    cos, sin = precompute_rope_frequencies(d_model, seq_len)
    
    # Align frequency matrix shape to [seq_len, d_model]
    if cos.shape == (d_model, seq_len):
        cos, sin = cos.T, sin.T
    elif cos.squeeze().shape == (d_model, seq_len):
        cos, sin = cos.squeeze().T, sin.squeeze().T
        
    Q_rope_4d = apply_rope(Q_4d, cos, sin)
    K_rope_4d = apply_rope(K_4d, cos, sin)
    
    scores_rope = torch.matmul(Q_rope_4d, K_rope_4d.transpose(-2, -1)) / (d_model ** 0.5)
    # Ensure 2D matrix shape [seq_len, seq_len]
    attn_rope = F.softmax(scores_rope, dim=-1).view(seq_len, seq_len).detach().numpy()
    
    # Compute average attention weight per relative distance |i - j|
    distances = list(range(seq_len))
    avg_none, avg_sin, avg_rope = [], [], []
    
    i_indices, j_indices = np.indices((seq_len, seq_len))
    
    for d in distances:
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