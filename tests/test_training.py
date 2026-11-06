"""Data handling, the learning-rate schedule, training, evaluation and the study pipeline."""

import json

import pytest
import torch

import study
from data import CharDataset, eval_windows, get_batch
from train import TrainConfig, evaluate_loss, lr_at, train

TEXT = "to be or not to be, that is the question. " * 60


# ── data ──────────────────────────────────────────────────────────────────────
def test_encode_decode_roundtrip_and_split():
    ds = CharDataset(TEXT)
    assert ds.decode(ds.encode("to be")) == "to be"
    assert len(ds.train) + len(ds.val) == len(TEXT)
    assert len(ds.val) == pytest.approx(0.1 * len(TEXT), abs=1)
    assert ds.vocab_size == len(set(TEXT))


def test_batch_targets_are_inputs_shifted_by_one():
    ds = CharDataset(TEXT)
    x, y = get_batch(ds.train, 4, 16, torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (4, 16)
    for row_x, row_y in zip(x, y, strict=True):
        start = next(i for i in range(len(ds.train)) if torch.equal(ds.train[i : i + 16], row_x))
        assert torch.equal(ds.train[start + 1 : start + 17], row_y)


def test_eval_windows_do_not_overlap_and_cover_the_data():
    data = torch.arange(100)
    x, y = eval_windows(data, 32)
    assert x.shape == (3, 32)
    assert torch.equal(x.flatten(), torch.arange(96)) and torch.equal(y.flatten(), torch.arange(1, 97))
    with pytest.raises(ValueError):
        eval_windows(data, 200)


# ── schedule ──────────────────────────────────────────────────────────────────
def test_learning_rate_warms_up_then_decays_to_min():
    cfg = TrainConfig(steps=1000, warmup=100, lr=1e-3, min_lr=1e-4)
    assert lr_at(0, cfg) == pytest.approx(1e-5)
    assert lr_at(99, cfg) == pytest.approx(1e-3)
    assert lr_at(550, cfg) == pytest.approx((1e-3 + 1e-4) / 2, rel=1e-3)
    assert lr_at(999, cfg) == pytest.approx(1e-4, rel=1e-2)
    assert all(lr_at(s, cfg) >= lr_at(s + 1, cfg) for s in range(100, 999))


# ── training ──────────────────────────────────────────────────────────────────
TINY = {"steps": 80, "context": 32, "batch_size": 16, "n_layer": 1, "n_head": 2, "d_model": 32,
        "dropout": 0.0, "warmup": 10, "log_every": 1000, "device": "cpu"}


@pytest.mark.parametrize("pos_type", ["none", "sinusoidal", "rope", "alibi"])
def test_training_reduces_the_loss(pos_type):
    ds = CharDataset(TEXT)
    model, history = train(TrainConfig(pos_type=pos_type, **TINY), ds, max_seq_len=128)
    assert history["train_loss"][-1] < 0.7 * history["train_loss"][0]
    assert history["tokens_per_second"] > 0 and history["n_params"] > 0
    # Evaluation works at the training length and beyond it
    assert evaluate_loss(model, ds.val, 32) < history["train_loss"][0]
    assert evaluate_loss(model, ds.val, 64) > 0


def test_same_seed_same_model():
    ds = CharDataset(TEXT)
    _, a = train(TrainConfig(pos_type="rope", seed=3, **TINY), ds, max_seq_len=64)
    _, b = train(TrainConfig(pos_type="rope", seed=3, **TINY), ds, max_seq_len=64)
    assert a["train_loss"] == b["train_loss"]


# ── study ─────────────────────────────────────────────────────────────────────
def test_study_end_to_end(tmp_path):
    text_file = tmp_path / "text.txt"
    text_file.write_text(TEXT)
    out = tmp_path / "runs"
    code = study.main(["--quick", "--steps", "20", "--context", "16", "--pos", "rope", "alibi",
                       "--seeds", "0", "1", "--text-file", str(text_file), "--out", str(out)])
    assert code == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["lengths"] == [16, 32, 64, 128]
    for pos in ("rope", "alibi"):
        m = summary["methods"][pos]
        assert m["seeds"] == 2
        assert m["ratio_vs_train_length"]["16"] == 1.0
        assert m["perplexity"]["128"]["std"] >= 0
    assert (out / "rope_seed1.json").exists()
    assert "| rope |" in (out / "summary.md").read_text()
    assert (out / "length_generalization.png").stat().st_size > 1000
