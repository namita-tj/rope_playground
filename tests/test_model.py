"""The tiny GPT: shapes, causality, and behaviour per position method."""

import math

import pytest
import torch

from model import GPTConfig, TinyGPT

POS_TYPES = ["none", "sinusoidal", "rope", "alibi"]
VOCAB = 20


def _model(pos_type: str, **kw) -> TinyGPT:
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=VOCAB, pos_type=pos_type, n_layer=2, n_head=2, d_model=32, dropout=0.0,
                    max_seq_len=kw.pop("max_seq_len", 256), **kw)
    return TinyGPT(cfg).eval()


@pytest.mark.parametrize("pos_type", POS_TYPES)
def test_shapes_and_initial_loss(pos_type):
    m = _model(pos_type)
    idx = torch.randint(0, VOCAB, (3, 16))
    targets = torch.randint(0, VOCAB, (3, 16))  # independent of the input (tied weights favour repeating it)
    logits, loss, maps = m(idx, targets)
    assert logits.shape == (3, 16, VOCAB) and maps is None
    # An untrained model should be about as unsure as random guessing: loss ~ ln(vocab)
    assert loss.item() == pytest.approx(math.log(VOCAB), abs=0.3)


@pytest.mark.parametrize("pos_type", POS_TYPES)
def test_model_is_causal(pos_type):
    """Predictions for position t may only use characters up to t."""
    m = _model(pos_type)
    idx = torch.randint(0, VOCAB, (1, 12))
    changed = idx.clone()
    changed[0, -1] = (idx[0, -1] + 1) % VOCAB
    a, _, _ = m(idx)
    b, _, _ = m(changed)
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5)


@pytest.mark.parametrize("pos_type", POS_TYPES)
def test_attention_maps_on_request(pos_type):
    _, _, maps = _model(pos_type)(torch.randint(0, VOCAB, (1, 8)), need_weights=True)
    assert len(maps) == 2 and maps[0].shape == (1, 2, 8, 8)


def test_sinusoidal_is_added_once_at_the_input_not_in_every_layer():
    m = _model("sinusoidal")
    assert all(block.attn.pos_type == "none" for block in m.blocks)
    assert all(block.attn.pos_type == "alibi" for block in _model("alibi").blocks)


def test_head_shares_weights_with_the_embedding():
    m = _model("rope")
    assert m.head.weight.data_ptr() == m.tok_emb.weight.data_ptr()


@pytest.mark.parametrize("pos_type", POS_TYPES)
def test_runs_beyond_training_length(pos_type):
    """Evaluation at 8x a training length of 32 must work for every method."""
    logits, _, _ = _model(pos_type)(torch.randint(0, VOCAB, (1, 256)))
    assert torch.isfinite(logits).all()


def test_rope_beyond_its_table_gives_a_clear_error():
    with pytest.raises(ValueError, match="outside the precomputed table"):
        _model("rope", max_seq_len=16)(torch.randint(0, VOCAB, (1, 17)))


def test_generate_appends_characters():
    out = _model("alibi").generate(torch.zeros(1, 3, dtype=torch.long), 10, generator=torch.Generator().manual_seed(0))
    assert out.shape == (1, 13) and out.max() < VOCAB


def test_unknown_position_method_is_rejected():
    with pytest.raises(AssertionError):
        _model("learned")
