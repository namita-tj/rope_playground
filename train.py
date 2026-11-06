"""
TRAINING AND EVALUATION for the tiny GPT.

Every run uses the same recipe; only the position method and the random seed change.
Evaluation reports the loss on the validation text cut into non-overlapping windows,
so every model is judged on exactly the same characters.
"""

import math
import time
from dataclasses import asdict, dataclass

import torch

from data import CharDataset, eval_windows, get_batch
from model import GPTConfig, TinyGPT


@dataclass
class TrainConfig:
    pos_type: str = "rope"
    seed: int = 0
    context: int = 256        # characters the model sees during training
    batch_size: int = 64
    steps: int = 3000
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    dropout: float = 0.1
    log_every: int = 250
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self) -> dict:
        return asdict(self)


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warm-up, then cosine decay from lr to min_lr."""
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    progress = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * min(progress, 1.0)))


def set_seed(seed: int) -> torch.Generator:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return torch.Generator().manual_seed(seed)


def train(cfg: TrainConfig, dataset: CharDataset, max_seq_len: int = 4096) -> tuple[TinyGPT, dict]:
    """Trains one model. Returns it with its loss curve and training speed."""
    gen = set_seed(cfg.seed)
    model = TinyGPT(GPTConfig(
        vocab_size=dataset.vocab_size, pos_type=cfg.pos_type, n_layer=cfg.n_layer, n_head=cfg.n_head,
        d_model=cfg.d_model, dropout=cfg.dropout, max_seq_len=max_seq_len,
    )).to(cfg.device)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr, betas=(0.9, 0.99),
    )

    history = {"step": [], "train_loss": []}
    model.train()
    start = time.perf_counter()
    for step in range(cfg.steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, cfg)
        x, y = get_batch(dataset.train, cfg.batch_size, cfg.context, gen, cfg.device)
        _, loss, _ = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            history["step"].append(step)
            history["train_loss"].append(round(loss.item(), 4))
            print(f"  [{cfg.pos_type} seed {cfg.seed}] step {step:>5}  loss {loss.item():.3f}")
    if cfg.device == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    history["train_seconds"] = round(seconds, 1)
    history["tokens_per_second"] = round(cfg.steps * cfg.batch_size * cfg.context / seconds)
    history["n_params"] = model.num_params()
    return model, history


@torch.no_grad()
def evaluate_loss(model: TinyGPT, data: torch.Tensor, length: int, device: str = "cpu",
                  tokens_per_batch: int = 16384) -> float:
    """Average next-character loss (nats) over non-overlapping windows of `length` characters."""
    model.eval()
    x, y = eval_windows(data, length)
    batch = max(1, tokens_per_batch // length)
    total, count = 0.0, 0
    for i in range(0, len(x), batch):
        xb, yb = x[i : i + batch].to(device), y[i : i + batch].to(device)
        _, loss, _ = model(xb, yb)
        total += loss.item() * xb.numel()
        count += xb.numel()
    model.train()
    return total / count


def perplexity(loss: float) -> float:
    """How many characters the model is 'choosing between' on average: e^loss."""
    return math.exp(loss)
