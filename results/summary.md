# Results: validation perplexity (lower is better), trained on 256-character windows

| Method | 256 (1x) | 512 (2x) | 1024 (4x) | 2048 (8x) | Training speed |
| --- | ---: | ---: | ---: | ---: | ---: |
| none | 5.49 ± 0.04 | 6.88 ± 0.39 | 11.91 ± 2.33 | 19.58 ± 6.10 | 270,433 chars/s |
| sinusoidal | 4.73 ± 0.04 | 10.17 ± 0.49 | 19.09 ± 1.74 | 26.96 ± 2.97 | 262,141 chars/s |
| rope | 4.39 ± 0.03 | 7.36 ± 0.85 | 17.09 ± 2.14 | 30.56 ± 3.40 | 232,266 chars/s |
| alibi | 4.47 ± 0.03 | 4.43 ± 0.04 | 4.40 ± 0.04 | 4.39 ± 0.04 | 258,690 chars/s |

## Degradation: perplexity at each length divided by perplexity at the training length

| Method | 1x | 2x | 4x | 8x |
| --- | ---: | ---: | ---: | ---: |
| none | 1.00 | 1.25 | 2.17 | 3.57 |
| sinusoidal | 1.00 | 2.15 | 4.03 | 5.70 |
| rope | 1.00 | 1.68 | 3.89 | 6.96 |
| alibi | 1.00 | 0.99 | 0.98 | 0.98 |

Mean ± standard deviation over 3 seed(s) per method.
