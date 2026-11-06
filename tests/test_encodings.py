"""Checks that the positional encodings do what the papers say they do."""

import math

import pytest
import torch

from positional_encodings import apply_rope, get_sinusoidal_encoding, precompute_rope_frequencies, rope_slice

torch.manual_seed(0)


# ── Sinusoidal ────────────────────────────────────────────────────────────────
def test_sinusoidal_shape_and_range():
    pe = get_sinusoidal_encoding(50, 64)
    assert pe.shape == (50, 64)
    assert pe.abs().max() <= 1.0


def test_sinusoidal_known_values():
    pe = get_sinusoidal_encoding(4, 8)
    # Position 0: every sin is 0 and every cos is 1
    assert torch.allclose(pe[0, 0::2], torch.zeros(4))
    assert torch.allclose(pe[0, 1::2], torch.ones(4))
    # Position 1, fastest wave: sin(1) and cos(1)
    assert pe[1, 0].item() == pytest.approx(math.sin(1.0), abs=1e-6)
    assert pe[1, 1].item() == pytest.approx(math.cos(1.0), abs=1e-6)


# ── RoPE tables ───────────────────────────────────────────────────────────────
def test_rope_table_shape():
    cos, sin = precompute_rope_frequencies(32, 100)
    assert cos.shape == sin.shape == (100, 16)


def test_rope_rejects_odd_dimension():
    with pytest.raises(ValueError):
        precompute_rope_frequencies(31, 10)


def test_rope_slice_bounds():
    cos, sin = precompute_rope_frequencies(8, 10)
    c, s = rope_slice(cos, sin, seq_len=4, offset=3)
    assert torch.equal(c, cos[3:7]) and torch.equal(s, sin[3:7])
    with pytest.raises(ValueError):
        rope_slice(cos, sin, seq_len=4, offset=7)


# ── RoPE behaviour ────────────────────────────────────────────────────────────
def _rotate_at(x: torch.Tensor, position: int, dim: int, table_len: int = 4096) -> torch.Tensor:
    """Applies RoPE to a single vector as if it stood at `position`."""
    cos, sin = precompute_rope_frequencies(dim, table_len)
    c, s = rope_slice(cos, sin, 1, position)
    return apply_rope(x.view(1, 1, 1, dim), c, s).view(dim)


def test_rope_preserves_length():
    cos, sin = precompute_rope_frequencies(32, 20)
    q = torch.randn(2, 3, 20, 32)
    rotated = apply_rope(q, cos, sin)
    assert rotated.shape == q.shape
    assert torch.allclose(rotated.norm(dim=-1), q.norm(dim=-1), atol=1e-5)


def test_rope_position_zero_is_identity():
    q = torch.randn(64)
    assert torch.allclose(_rotate_at(q, 0, 64), q)


@pytest.mark.parametrize(("m", "n", "shift"), [(2, 5, 100), (0, 37, 1000), (500, 3, 7)])
def test_rope_score_depends_only_on_relative_distance(m, n, shift):
    """RoPE's defining property: q at m vs k at n scores the same as at m+s vs n+s."""
    dim = 64
    q, k = torch.randn(dim), torch.randn(dim)
    original = _rotate_at(q, m, dim) @ _rotate_at(k, n, dim)
    shifted = _rotate_at(q, m + shift, dim) @ _rotate_at(k, n + shift, dim)
    assert original.item() == pytest.approx(shifted.item(), rel=1e-4, abs=1e-4)


def test_rope_matches_complex_number_reference():
    """Independent check: RoPE is multiplication by e^(i * position * theta).

    Pairs dimension j with dimension j + dim/2 (the "rotate half" layout that
    Hugging Face's Llama uses), turns each pair into a complex number, rotates it
    and turns it back.
    """
    dim, seq_len = 16, 12
    x = torch.randn(1, 2, seq_len, dim, dtype=torch.float64)
    cos, sin = precompute_rope_frequencies(dim, seq_len)
    ours = apply_rope(x, cos.double(), sin.double())

    theta = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    angles = torch.outer(torch.arange(seq_len, dtype=torch.float64), theta)
    z = torch.complex(x[..., : dim // 2], x[..., dim // 2 :]) * torch.polar(torch.ones_like(angles), angles)
    reference = torch.cat([z.real, z.imag], dim=-1)

    assert torch.allclose(ours, reference, atol=1e-10)


def test_rope_matches_huggingface_llama_formula():
    """Same computation as transformers' LlamaRotaryEmbedding + apply_rotary_pos_emb,
    written out here so the test does not need the transformers package."""
    dim, seq_len = 32, 20
    q = torch.randn(1, 4, seq_len, dim)

    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    freqs = torch.outer(torch.arange(seq_len).float(), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)

    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    hf = q * emb.cos() + rotate_half(q) * emb.sin()

    cos, sin = precompute_rope_frequencies(dim, seq_len)
    assert torch.allclose(apply_rope(q, cos, sin), hf, atol=1e-6)
