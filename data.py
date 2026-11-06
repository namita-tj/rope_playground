"""
TINY SHAKESPEARE: ~1 MB of Shakespeare plays, read one character at a time.

The classic dataset for small language-model experiments (from Karpathy's char-rnn).
Downloaded once to data/ (git-ignored). 90% is used for training, the last 10% only
for measuring how well each model predicts text it never saw.
"""

import urllib.request
from pathlib import Path

import torch

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = Path(__file__).parent / "data"


def load_text(path: Path | None = None) -> str:
    """The Tiny Shakespeare text, downloaded on first use."""
    path = path or DATA_DIR / "tinyshakespeare.txt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading Tiny Shakespeare to {path} ...")
        urllib.request.urlretrieve(URL, path)
    return path.read_text(encoding="utf-8")


class CharDataset:
    """Character vocabulary plus the encoded train and validation splits."""

    def __init__(self, text: str, val_share: float = 0.1):
        if len(text) < 10:
            raise ValueError("text is too short to split into train and validation")
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        data = torch.tensor(self.encode(text), dtype=torch.long)
        split = int(len(data) * (1 - val_share))
        self.train, self.val = data[:split], data[split:]

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, ids) -> str:
        return "".join(self.chars[int(i)] for i in ids)


def get_batch(data: torch.Tensor, batch_size: int, context: int, generator: torch.Generator,
              device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Random windows of `context` characters; the targets are the same windows shifted by one."""
    if len(data) <= context:
        raise ValueError(f"data ({len(data)} chars) is shorter than the context ({context})")
    starts = torch.randint(0, len(data) - context, (batch_size,), generator=generator)
    x = torch.stack([data[s : s + context] for s in starts])
    y = torch.stack([data[s + 1 : s + 1 + context] for s in starts])
    return x.to(device), y.to(device)


def eval_windows(data: torch.Tensor, context: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Non-overlapping windows covering the data once, for a deterministic evaluation."""
    n = (len(data) - 1) // context
    if n < 1:
        raise ValueError(f"data ({len(data)} chars) is too short for windows of {context}")
    x = data[: n * context].view(n, context)
    y = data[1 : n * context + 1].view(n, context)
    return x, y
