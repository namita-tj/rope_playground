"""Checks the from-scratch attention against PyTorch's built-in implementation."""

import pytest
import torch
import torch.nn.functional as F

from attention import CustomMultiHeadAttention
from positional_encodings import apply_rope, rope_slice

D_MODEL, N_HEADS = 32, 4
POS_TYPES = ["none", "sinusoidal", "rope"]


def _module(pos_type: str, max_seq_len: int = 256) -> CustomMultiHeadAttention:
    torch.manual_seed(0)
    return CustomMultiHeadAttention(D_MODEL, N_HEADS, pos_type=pos_type, max_seq_len=max_seq_len).eval()


@pytest.mark.parametrize("pos_type", POS_TYPES)
@pytest.mark.parametrize("seq_len", [1, 7, 33, 256])
def test_any_sequence_length_works(pos_type, seq_len):
    """Regression test: RoPE used to crash unless seq_len == max_seq_len."""
    out, weights = _module(pos_type)(torch.randn(2, seq_len, D_MODEL))
    assert out.shape == (2, seq_len, D_MODEL)
    assert weights.shape == (2, N_HEADS, seq_len, seq_len)


def test_too_long_sequence_gives_clear_error():
    with pytest.raises(ValueError, match="outside the precomputed table"):
        _module("rope", max_seq_len=8)(torch.randn(1, 9, D_MODEL))


def test_wrong_hidden_size_gives_clear_error():
    with pytest.raises(ValueError, match="hidden size"):
        _module("none")(torch.randn(1, 4, D_MODEL + 1))


def test_heads_must_divide_model_size():
    with pytest.raises(AssertionError):
        CustomMultiHeadAttention(30, 4)


@pytest.mark.parametrize("pos_type", POS_TYPES)
@pytest.mark.parametrize("causal", [False, True])
def test_attention_weights_are_probabilities(pos_type, causal):
    _, weights = _module(pos_type)(torch.randn(2, 10, D_MODEL), causal=causal)
    assert torch.allclose(weights.sum(-1), torch.ones(2, N_HEADS, 10), atol=1e-6)
    assert (weights >= 0).all()
    if causal:
        future = torch.triu(torch.ones(10, 10, dtype=torch.bool), diagonal=1)
        assert (weights[..., future] == 0).all()


@pytest.mark.parametrize("pos_type", ["none", "rope"])
@pytest.mark.parametrize("causal", [False, True])
def test_matches_pytorch_scaled_dot_product_attention(pos_type, causal):
    """Same projections, then PyTorch's own attention kernel: outputs must agree."""
    m = _module(pos_type)
    x = torch.randn(2, 12, D_MODEL)
    B, T, C = x.shape
    head_dim = C // N_HEADS

    def heads(t):
        return t.view(B, T, N_HEADS, head_dim).transpose(1, 2)

    q, k, v = heads(m.q_proj(x)), heads(m.k_proj(x)), heads(m.v_proj(x))
    if pos_type == "rope":
        cos, sin = rope_slice(m.rope_cos, m.rope_sin, T)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    reference = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    reference = m.out_proj(reference.transpose(1, 2).reshape(B, T, C))

    ours, _ = m(x, causal=causal)
    assert torch.allclose(ours, reference, atol=1e-5)


@pytest.mark.parametrize("pos_type", POS_TYPES)
def test_causal_mask_hides_the_future(pos_type):
    """Changing the last token must not change any earlier output."""
    m = _module(pos_type)
    x = torch.randn(1, 8, D_MODEL)
    changed = x.clone()
    changed[:, -1] = torch.randn(D_MODEL)
    out_a, _ = m(x, causal=True)
    out_b, _ = m(changed, causal=True)
    assert torch.allclose(out_a[:, :-1], out_b[:, :-1], atol=1e-6)
    assert not torch.allclose(out_a[:, -1], out_b[:, -1])


def test_without_position_info_order_does_not_matter():
    """No positional encoding = attention is permutation-equivariant: shuffle in, shuffle out."""
    m = _module("none")
    x = torch.randn(1, 6, D_MODEL)
    perm = torch.randperm(6)
    out, _ = m(x)
    out_perm, _ = m(x[:, perm])
    assert torch.allclose(out[:, perm], out_perm, atol=1e-6)


@pytest.mark.parametrize("pos_type", ["sinusoidal", "rope"])
def test_with_position_info_order_matters(pos_type):
    m = _module(pos_type)
    x = torch.randn(1, 6, D_MODEL)
    perm = torch.tensor([5, 4, 3, 2, 1, 0])
    out, _ = m(x)
    out_perm, _ = m(x[:, perm])
    assert not torch.allclose(out[:, perm], out_perm, atol=1e-4)


def test_rope_attention_is_shift_invariant():
    """With RoPE, the same sentence starting at a later position gives identical attention."""
    m = _module("rope")
    x = torch.randn(1, 10, D_MODEL)
    _, w_start = m(x, offset=0)
    _, w_later = m(x, offset=100)
    assert torch.allclose(w_start, w_later, atol=1e-5)
