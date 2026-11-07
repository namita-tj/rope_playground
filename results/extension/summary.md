# Stretching RoPE: validation perplexity (lower is better), trained on 256-character windows

## Zero-shot (no extra training, stretch = length / training length)

| Method | 256 (1x) | 512 (2x) | 1024 (4x) | 2048 (8x) | RoPE breakdown undone at 8x |
| --- | ---: | ---: | ---: | ---: | ---: |
| RoPE (no stretch) | 4.39 ± 0.03 | 7.36 ± 0.85 | 17.09 ± 2.14 | 30.56 ± 3.40 | - |
| Position Interpolation | 4.39 ± 0.03 | 17.02 ± 1.65 | 41.39 ± 7.05 | 53.50 ± 10.45 | -88% |
| NTK scaling | 4.39 ± 0.03 | 4.95 ± 0.03 | 7.10 ± 0.57 | 11.84 ± 1.24 | 72% |
| YaRN | 4.39 ± 0.03 | 4.89 ± 0.07 | 5.87 ± 0.30 | 7.66 ± 0.54 | 88% |

## Fine-tuned (short extra training on 4x longer text, stretch 4)

| Method | 256 (1x) | 512 (2x) | 1024 (4x) | 2048 (8x) | RoPE breakdown undone at 8x |
| --- | ---: | ---: | ---: | ---: | ---: |
| RoPE (no stretch) | 4.45 ± 0.05 | 4.41 ± 0.04 | 4.39 ± 0.04 | 5.21 ± 0.12 | 97% |
| Position Interpolation | 4.58 ± 0.06 | 4.54 ± 0.06 | 4.52 ± 0.06 | 10.74 ± 0.65 | 76% |
| NTK scaling | 4.44 ± 0.04 | 4.40 ± 0.04 | 4.38 ± 0.03 | 7.75 ± 0.71 | 87% |
| YaRN | 4.44 ± 0.04 | 4.41 ± 0.04 | 4.38 ± 0.04 | 8.96 ± 0.81 | 83% |

ALiBi from session 2 for reference: 256: 4.47, 512: 4.43, 1024: 4.40, 2048: 4.39

Mean ± standard deviation over 3 seed(s).
