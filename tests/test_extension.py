"""Session 3: Position Interpolation, NTK scaling and YaRN, checked against their papers' formulas."""

import json
import math

import pytest
import torch

import extend
from model import GPTConfig, TinyGPT
from positional_encodings import (
    apply_rope,
    precompute_rope_frequencies,
    rope_slice,
    scaled_rope_frequencies,
)

DIM, LEN, TRAIN = 32, 2048, 256


def _angles(cos, sin):
    return torch.atan2(sin, cos)


@pytest.mark.parametrize("method", [None, "pi", "ntk", "yarn"])
def test_no_stretch_is_plain_rope(method):
    plain = precompute_rope_frequencies(DIM, LEN)
    stretched = scaled_rope_frequencies(DIM, LEN, method, 1.0, TRAIN)
    assert torch.allclose(stretched[0], plain[0]) and torch.allclose(stretched[1], plain[1])


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="unknown RoPE scaling"):
        scaled_rope_frequencies(DIM, LEN, "longrope", 4.0, TRAIN)


def test_position_interpolation_squeezes_positions():
    """PI with factor 4: position 4m looks exactly like position m did in training."""
    plain_cos, plain_sin = precompute_rope_frequencies(DIM, LEN)
    pi_cos, pi_sin = scaled_rope_frequencies(DIM, LEN, "pi", 4.0, TRAIN)
    for m in (1, 37, 255):
        assert torch.allclose(pi_cos[4 * m], plain_cos[m], atol=1e-5)
        assert torch.allclose(pi_sin[4 * m], plain_sin[m], atol=1e-5)


def test_ntk_keeps_fast_hands_and_slows_the_slowest_by_the_factor():
    plain_cos, plain_sin = precompute_rope_frequencies(DIM, LEN)
    ntk_cos, ntk_sin = scaled_rope_frequencies(DIM, LEN, "ntk", 4.0, TRAIN)
    # fastest pair: unchanged
    assert torch.allclose(ntk_cos[:, 0], plain_cos[:, 0]) and torch.allclose(ntk_sin[:, 0], plain_sin[:, 0])
    # slowest pair: exactly 4x slower (base * s^(d/(d-2)) makes the last frequency theta/s)
    plain_slow = _angles(plain_cos[2000, -1], plain_sin[2000, -1]).item()
    ntk_slow = _angles(ntk_cos[2000, -1], ntk_sin[2000, -1]).item()
    assert ntk_slow / plain_slow == pytest.approx(0.25, rel=1e-4)
    # in between: slowed, but by less than the factor
    plain_angle = _angles(plain_cos[1, DIM // 4], plain_sin[1, DIM // 4]).item()
    ntk_angle = _angles(ntk_cos[1, DIM // 4], ntk_sin[1, DIM // 4]).item()
    assert plain_angle / 4 < ntk_angle < plain_angle


def test_yarn_keeps_fast_hands_interpolates_slow_ones_and_sharpens_attention():
    factor = 4.0
    mscale = 0.1 * math.log(factor) + 1.0
    plain_cos, plain_sin = precompute_rope_frequencies(DIM, LEN)
    pi_cos, pi_sin = scaled_rope_frequencies(DIM, LEN, "pi", factor, TRAIN)
    y_cos, y_sin = scaled_rope_frequencies(DIM, LEN, "yarn", factor, TRAIN)

    inv_freq = 1.0 / (10000 ** (torch.arange(0, DIM, 2).float() / DIM))
    turns = TRAIN / (2 * math.pi / inv_freq)
    fast, slow = turns > 32, turns < 1
    assert fast.any() and slow.any(), "test setup should have both kinds of hands"
    assert torch.allclose(y_cos[:, fast], plain_cos[:, fast] * mscale, atol=1e-5)
    assert torch.allclose(y_cos[:, slow], pi_cos[:, slow] * mscale, atol=1e-5)
    assert torch.allclose(y_sin[:, slow], pi_sin[:, slow] * mscale, atol=1e-5)
    # the attention temperature: every pair is scaled by the same factor
    assert torch.allclose((y_cos**2 + y_sin**2), torch.full_like(y_cos, mscale**2), atol=1e-5)


@pytest.mark.parametrize("method", ["pi", "ntk", "yarn"])
def test_stretched_rope_still_depends_only_on_relative_distance(method):
    cos, sin = scaled_rope_frequencies(DIM, LEN, method, 8.0, TRAIN)
    q, k = torch.randn(DIM), torch.randn(DIM)

    def at(x, pos):
        c, s = rope_slice(cos, sin, 1, pos)
        return apply_rope(x.view(1, 1, 1, DIM), c, s).view(DIM)

    assert (at(q, 3) @ at(k, 40)).item() == pytest.approx((at(q, 1003) @ at(k, 1040)).item(), rel=1e-4, abs=1e-4)


def _gpt(pos_type="rope"):
    torch.manual_seed(0)
    return TinyGPT(GPTConfig(vocab_size=20, pos_type=pos_type, n_layer=2, n_head=2, d_model=32,
                             dropout=0.0, max_seq_len=512)).eval()


def test_model_scaling_changes_long_text_and_none_restores_plain_rope():
    m = _gpt()
    idx = torch.randint(0, 20, (1, 300))
    plain, _, _ = m(idx)
    m.set_rope_scaling("yarn", 4.0, 64)
    stretched, _, _ = m(idx)
    assert not torch.allclose(plain, stretched)
    m.set_rope_scaling(None)
    again, _, _ = m(idx)
    assert torch.allclose(plain, again)


@pytest.mark.parametrize("pos_type", ["none", "sinusoidal", "alibi"])
def test_scaling_a_non_rope_model_is_an_error(pos_type):
    with pytest.raises(ValueError, match="needs a RoPE model"):
        _gpt(pos_type).set_rope_scaling("pi", 2.0, 64)


def test_extension_study_end_to_end(tmp_path):
    text = tmp_path / "text.txt"
    text.write_text("to be or not to be, that is the question. " * 200)  # validation split must fit a 512 window
    out = tmp_path / "ext"
    assert extend.main(["--quick", "--steps", "20", "--ft-steps", "3", "--seeds", "0", "1",
                        "--text-file", str(text), "--out", str(out)]) == 0
    s = json.loads((out / "summary.json").read_text())
    assert s["lengths"] == [64, 128, 256, 512] and s["seeds"] == 2
    for setting in ("zero_shot", "fine_tuned"):
        for method in extend.METHODS:
            block = s["settings"][setting][method]
            assert block["512"]["mean"] > 0 and "recovered" in block
    # zero-shot: no stretch at the training length, so every method gives the same perplexity there
    zs = s["settings"]["zero_shot"]
    assert len({zs[m]["64"]["mean"] for m in extend.METHODS}) == 1
    assert (out / "extension.png").stat().st_size > 1000
    assert "Position Interpolation" in (out / "summary.md").read_text()
