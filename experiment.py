"""
THE DECAY EXPERIMENT: does RoPE make far-away words matter less?

The RoPE paper (Su et al., 2021) claims "long-term decay": the further apart two
tokens are, the smaller their attention score tends to be. But that only holds for
tokens whose content is *related*. Two completely random, unrelated vectors already
have an average score of zero at every distance, so there is nothing to decay.

So we measure two things, averaged over many random vectors:

  1. SAME CONTENT:  the score between a word and *itself* placed d positions away.
                    Without position info this is always 1.0. With RoPE it should decay.
  2. UNRELATED:     the score between two independent random words d positions apart.
                    This stays around 0 with or without RoPE (the trap the first version
                    of this experiment fell into).

Scores are cosine-style: divided by the score at distance 0, so 1.0 = "as related as
standing right next to each other".
"""

from dataclasses import dataclass

import numpy as np
import torch

from positional_encodings import apply_rope, precompute_rope_frequencies


@dataclass
class DecayResult:
    distances: np.ndarray
    same_mean: np.ndarray       # RoPE, same content
    same_std: np.ndarray
    unrelated_mean: np.ndarray  # RoPE, unrelated content
    unrelated_std: np.ndarray


def rope_scores_vs_distance(
    head_dim: int = 64, max_distance: int = 512, n_samples: int = 1000, seed: int = 42
) -> DecayResult:
    """Normalised RoPE attention scores between position 0 and position d."""
    gen = torch.Generator().manual_seed(seed)
    cos, sin = precompute_rope_frequencies(head_dim, max_distance + 1)

    # n_samples random "words", unit length, shaped [n, 1 head, 1 position, head_dim]
    q = torch.randn(n_samples, 1, 1, head_dim, generator=gen)
    q = q / q.norm(dim=-1, keepdim=True)
    other = torch.randn(n_samples, 1, 1, head_dim, generator=gen)
    other = other / other.norm(dim=-1, keepdim=True)

    q_at_0 = apply_rope(q, cos[:1], sin[:1])          # the query stays at position 0
    same_at_d = apply_rope(q, cos, sin)               # the same word at every position d
    other_at_d = apply_rope(other, cos, sin)          # an unrelated word at every position d

    # Dot products -> [n_samples, max_distance + 1]
    same = (q_at_0 * same_at_d).sum(-1).squeeze(1)
    unrelated = (q_at_0 * other_at_d).sum(-1).squeeze(1)

    return DecayResult(
        distances=np.arange(max_distance + 1),
        same_mean=same.mean(0).numpy(),
        same_std=same.std(0).numpy(),
        unrelated_mean=unrelated.mean(0).numpy(),
        unrelated_std=unrelated.std(0).numpy(),
    )


def plot(result: DecayResult, path: str = "assets/decay_plot.png") -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = result.distances
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(1.0, color="gray", linestyle="--", label="No position info, same word (always 1.0)")
    ax.plot(d, result.same_mean, color="red", linewidth=2, label="RoPE, same word")
    ax.fill_between(d, result.same_mean - result.same_std, result.same_mean + result.same_std,
                    color="red", alpha=0.12)
    ax.plot(d, result.unrelated_mean, color="steelblue", label="RoPE, unrelated words")
    ax.fill_between(d, result.unrelated_mean - result.unrelated_std,
                    result.unrelated_mean + result.unrelated_std, color="steelblue", alpha=0.12)
    ax.set_title("RoPE long-term decay: related words matter less the further apart they are", fontsize=13)
    ax.set_xlabel("Relative distance |pos_i - pos_j| (tokens)", fontsize=12)
    ax.set_ylabel("Attention score (1.0 = same word, same position)", fontsize=12)
    ax.legend(fontsize=10, loc="center right")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def run_distance_decay_experiment(head_dim: int = 64, max_distance: int = 512) -> DecayResult:
    result = rope_scores_vs_distance(head_dim=head_dim, max_distance=max_distance)
    path = plot(result)
    near, far = result.same_mean[:8].mean(), result.same_mean[-64:].mean()
    print(f"same word, distance 0-7:      average score {near:.3f}")
    print(f"same word, last 64 distances: average score {far:.3f}")
    print(f"unrelated words, all distances: average score {result.unrelated_mean.mean():+.3f}")
    print(f"✅ plot saved to '{path}'")
    return result


if __name__ == "__main__":
    run_distance_decay_experiment()
