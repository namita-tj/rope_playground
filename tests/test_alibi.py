"""ALiBi: slopes from the paper, the distance penalty, and attention with it."""

import pytest
import torch
import torch.nn.functional as F

from attention import CustomMultiHeadAttention
from positional_encodings import get_alibi_bias, get_alibi_slopes


def test_slopes_for_8_heads_are_halving():
    expected = torch.tensor([1 / 2**i for i in range(1, 9)])
    assert torch.allclose(get_alibi_slopes(8), expected)


def test_slopes_for_non_power_of_two_follow_the_official_code():
    # 6 heads: the 4-head slopes, then every second 8-head slope (0.5, 0.125)
    expected = torch.tensor([0.25, 0.0625, 0.015625, 0.00390625, 0.5, 0.125])
    assert torch.allclose(get_alibi_slopes(6), expected)


def test_bias_is_minus_slope_times_distance():
    bias = get_alibi_bias(4, 6)
    slopes = get_alibi_slopes(4)
    assert bias.shape == (4, 6, 6)
    assert torch.all(bias.diagonal(dim1=-2, dim2=-1) == 0)
    assert bias[2, 5, 1].item() == pytest.approx(-slopes[2].item() * 4)
    assert torch.allclose(bias, bias.transpose(-2, -1))  # only the distance matters


def _module(**kw):
    torch.manual_seed(0)
    return CustomMultiHeadAttention(32, 4, pos_type="alibi", **kw).eval()


@pytest.mark.parametrize("causal", [False, True])
def test_alibi_attention_matches_pytorch_with_the_same_bias(causal):
    m = _module()
    x = torch.randn(2, 9, 32)
    B, T, C = x.shape

    def heads(t):
        return t.view(B, T, 4, 8).transpose(1, 2)

    bias = get_alibi_bias(4, T)
    if causal:
        bias = bias + torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
    ref = F.scaled_dot_product_attention(heads(m.q_proj(x)), heads(m.k_proj(x)), heads(m.v_proj(x)), attn_mask=bias)
    ref = m.out_proj(ref.transpose(1, 2).reshape(B, T, C))
    ours, _ = m(x, causal=causal)
    assert torch.allclose(ours, ref, atol=1e-5)


def test_alibi_works_at_any_length_without_a_table():
    out, w = _module()(torch.randn(1, 5000, 32)[:, :3000], causal=True, need_weights=False)
    assert out.shape == (1, 3000, 32) and w is None


def test_alibi_is_shift_invariant():
    m = _module()
    x = torch.randn(1, 10, 32)
    assert torch.allclose(m(x, offset=0)[1], m(x, offset=500)[1])


@pytest.mark.parametrize("pos_type", ["none", "sinusoidal", "rope", "alibi"])
@pytest.mark.parametrize("causal", [False, True])
def test_fast_path_equals_explicit_attention(pos_type, causal):
    """need_weights=False uses PyTorch's fused kernel; the result must not change."""
    torch.manual_seed(0)
    m = CustomMultiHeadAttention(32, 4, pos_type=pos_type, max_seq_len=64).eval()
    x = torch.randn(2, 17, 32)
    slow, weights = m(x, causal=causal, need_weights=True)
    fast, none = m(x, causal=causal, need_weights=False)
    assert weights is not None and none is None
    assert torch.allclose(slow, fast, atol=1e-5)


def test_alibi_attends_less_to_far_tokens_with_identical_content():
    """With identical tokens, content can't tell them apart, so attention must decay with distance."""
    m = _module()
    x = torch.ones(1, 12, 32)
    _, w = m(x, causal=True)
    last_row = w[0, 0, -1]  # steepest head, last query
    assert torch.all(last_row[1:] >= last_row[:-1])  # closer keys get more weight
