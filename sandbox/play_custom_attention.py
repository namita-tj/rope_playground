import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Setup minimal hyperparameters
batch_size = 1
seq_len = 4          # Words: ["AI", "learns", "very", "fast"]
d_model = 8          # Embedding size per token
num_heads = 2        # Split into 2 heads
d_k = d_model // num_heads  # 4 dimensions per head

torch.manual_seed(42)
X = torch.randn(batch_size, seq_len, d_model)  # Input tensor [1, 4, 8]

print("=== STEP 1: LINEAR PROJECTIONS ===")
W_q = nn.Linear(d_model, d_model, bias=False)
W_k = nn.Linear(d_model, d_model, bias=False)
W_v = nn.Linear(d_model, d_model, bias=False)

Q = W_q(X)  # Shape: [1, 4, 8]
K = W_k(X)  # Shape: [1, 4, 8]
V = W_v(X)  # Shape: [1, 4, 8]
print(f"Input X shape: {X.shape}")
print(f"Projected Q shape: {Q.shape}")

print("\n=== STEP 2: HEAD SPLITTING & TRANSPOSITION ===")
# Reshape: [1, 4, 8] -> [1, 4, 2, 4] -> Transpose: [1, 2, 4, 4]
Q_split = Q.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
K_split = K.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
V_split = V.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
print(f"Split Q shape: {Q_split.shape} (Batch, Heads, Seq_Len, Head_Dim)")

print("\n=== STEP 3: SCALED DOT-PRODUCT & CAUSAL MASKING ===")
# Raw scores = (Q * K^T) / sqrt(d_k)
scores = torch.matmul(Q_split, K_split.transpose(-2, -1)) / (d_k ** 0.5)

# Build Causal Mask (Lower Triangular Matrix)
causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
scores_masked = scores.masked_fill(~causal_mask, float('-inf'))
attn_weights = F.softmax(scores_masked, dim=-1)

print("Head 1 Attention Matrix (Causal Masked):")
print(torch.round(attn_weights[0, 0], decimals=3))

print("\n=== STEP 4: RECOMBINATION & OUTPUT PROJECTION ===")
# Multiply weights by Values: [1, 2, 4, 4] x [1, 2, 4, 4] -> [1, 2, 4, 4]
context_heads = torch.matmul(attn_weights, V_split)

# Concatenate heads back: [1, 2, 4, 4] -> Transpose -> [1, 4, 8]
context_concat = context_heads.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
W_o = nn.Linear(d_model, d_model, bias=False)
output = W_o(context_concat)

print(f"Recombined output shape: {output.shape} (Matches original input shape!)")