"""Checks the decay experiment's numbers, not just that it runs."""

import numpy as np

from experiment import plot, rope_scores_vs_distance


def test_same_word_starts_at_one_and_decays():
    r = rope_scores_vs_distance(head_dim=64, max_distance=512, n_samples=500)
    assert r.same_mean[0] == np.float32(1.0) or abs(r.same_mean[0] - 1.0) < 1e-5
    assert r.same_mean[:8].mean() > 0.7
    assert r.same_mean[-64:].mean() < 0.4
    assert r.same_mean[:8].mean() > 2 * r.same_mean[-64:].mean()


def test_unrelated_words_average_zero_at_every_distance():
    r = rope_scores_vs_distance(head_dim=64, max_distance=256, n_samples=2000)
    assert abs(r.unrelated_mean).max() < 0.02


def test_reproducible_with_seed():
    a = rope_scores_vs_distance(max_distance=32, n_samples=50, seed=1)
    b = rope_scores_vs_distance(max_distance=32, n_samples=50, seed=1)
    assert np.array_equal(a.same_mean, b.same_mean)


def test_plot_is_written(tmp_path):
    r = rope_scores_vs_distance(max_distance=16, n_samples=20)
    path = plot(r, str(tmp_path / "decay.png"))
    assert (tmp_path / "decay.png").stat().st_size > 1000
    assert path.endswith("decay.png")
