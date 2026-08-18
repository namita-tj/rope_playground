# Sinusoidal vs. RoPE Attention Playground

An isolated, zero-dependency PyTorch playground comparing Vaswani Sinusoidal Absolute Positional Encodings against Rotary Position Embeddings (RoPE).

## 📌 Project Overview

This project explores how Transformers handle sequence order from scratch using PyTorch:

* **Scaled Dot-Product Attention:** $(\text{softmax}(QK^T / \sqrt{d_k})V)$
* **Multi-Head Attention (MHA):** Head splitting, parallel matrix projections, and causal masking
* **Vaswani Sinusoidal Encodings:** Static additive sine/cosine waves
* **Rotary Position Embeddings (RoPE):** 2D vector rotations in complex representation space
* **Attention Decay Benchmarks:** Relative distance analysis across sequence lengths

---

## 📁 Repository Structure

```text
rope_playground/
├── LICENSE                     # Open-source license
├── README.md                  # Project overview and setup guide
├── attention.py               # Custom Multi-Head Attention module from scratch
├── positional_encodings.py    # Sinusoidal PE and RoPE generator logic
├── sandbox/
│   ├── play_custom_attention.py  # Step-by-step MHA tensor inspection
│   └── play_encodings.py         # Interactive RoPE vs. Sinusoidal playground
└── tests/
    ├── test_attention_step2.py   # PyTorch unit tests for MHA and causal masking
    └── test_encodings_step1.py   # Unit tests for encoding shapes and vector norms