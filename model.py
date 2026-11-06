"""
A TINY GPT: the smallest language model that is still a real transformer.

Reads text one character at a time and learns to predict the next character.
Built from the attention in attention.py, so the only thing that changes between
experiments is HOW the model knows where each character sits:

  none        no position info at all (only the causal mask hints at order)
  sinusoidal  the 2017 waves, added once to the input (as in "Attention Is All You Need")
  rope        rotations inside every attention layer (Llama, Qwen, Mistral)
  alibi       distance penalties inside every attention layer (BLOOM, MPT)
"""

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import POS_TYPES, CustomMultiHeadAttention
from positional_encodings import get_sinusoidal_encoding

# Sinusoidal entries have RMS 1/sqrt(2); this brings them to RMS 0.02, the init scale of every embedding here.
SINUSOIDAL_SCALE = 0.02 * math.sqrt(2)


@dataclass
class GPTConfig:
    vocab_size: int
    pos_type: str = "rope"
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    dropout: float = 0.1
    max_seq_len: int = 4096  # RoPE table size; evaluation can go up to this length

    def to_dict(self) -> dict:
        return asdict(self)


class Block(nn.Module):
    """One transformer layer: attention, then a small feed-forward network (pre-norm)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        # Sinusoidal positions are added once at the input, so attention itself gets none.
        attn_pos = "none" if cfg.pos_type == "sinusoidal" else cfg.pos_type
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CustomMultiHeadAttention(cfg.d_model, cfg.n_head, pos_type=attn_pos, max_seq_len=cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, need_weights: bool = False):
        attn_out, weights = self.attn(self.ln1(x), causal=True, need_weights=need_weights)
        x = x + self.drop(attn_out)
        x = x + self.mlp(self.ln2(x))
        return x, weights


class TinyGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.pos_type in POS_TYPES, f"pos_type must be one of {POS_TYPES}"
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying, as in GPT-2
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, need_weights: bool = False):
        """idx: [batch, seq_len] character ids. Returns (logits, loss or None, attention maps or None)."""
        B, T = idx.shape
        x = self.tok_emb(idx)
        if self.cfg.pos_type == "sinusoidal":
            # The waves have amplitude 1 but the word vectors start at std 0.02, so unscaled waves
            # drown the words (perplexity ~28 instead of ~11 in a 600-step check). Scaling them to
            # the size of GPT-2's learned position embeddings (std 0.02) keeps both audible.
            x = x + SINUSOIDAL_SCALE * get_sinusoidal_encoding(T, self.cfg.d_model).to(idx.device)
        x = self.drop(x)
        maps = []
        for block in self.blocks:
            x, weights = block(x, need_weights=need_weights)
            maps.append(weights)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, (maps if need_weights else None)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 generator: torch.Generator | None = None) -> torch.Tensor:
        """Writes `max_new_tokens` characters after `idx`, one at a time."""
        for _ in range(max_new_tokens):
            logits, _, _ = self(idx[:, -self.cfg.max_seq_len:])
            probs = F.softmax(logits[:, -1, :] / max(temperature, 1e-6), dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1, generator=generator)], dim=1)
        return idx
