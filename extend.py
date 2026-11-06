"""
SESSION 3: can we stretch RoPE to read longer texts?

Session 2 showed RoPE is the best reader at its training length (256 characters)
and the worst one beyond it. This study trains the same RoPE GPT again and tries the
three tricks real LLMs use to stretch it (see positional_encodings.py):

  pi    Position Interpolation: slow every clock hand down
  ntk   NTK-aware scaling: stretch the base, keep the fast hands
  yarn  YaRN: per hand, plus slightly sharper attention

Two settings for each:
  1. zero-shot   no extra training; the stretch factor is set to (text length / 256)
  2. fine-tuned  a short extra training run on 1024-character text with the stretch
                 switched on (factor 4), which is how long-context LLMs are made

    python extend.py           # full study, about 25-30 min on a Colab T4
    python extend.py --quick   # tiny CPU version to check the pipeline

Writes runs_extension/summary.json, summary.md and extension.png.
"""

import argparse
import copy
import json
import math
import statistics
from pathlib import Path

import torch

from data import CharDataset, get_batch, load_text
from model import TinyGPT
from train import TrainConfig, evaluate_loss, perplexity, set_seed, train

METHODS = ("rope", "pi", "ntk", "yarn")
LABELS = {"rope": "RoPE (no stretch)", "pi": "Position Interpolation", "ntk": "NTK scaling", "yarn": "YaRN"}
QUICK = {"steps": 150, "context": 64, "batch_size": 32, "n_layer": 2, "d_model": 64, "log_every": 50}


def set_method(model: TinyGPT, method: str, factor: float, train_len: int) -> None:
    model.set_rope_scaling(None if method == "rope" else method, factor, train_len)


@torch.no_grad()
def evaluate_methods(model: TinyGPT, dataset: CharDataset, lengths: list[int], train_len: int, device: str,
                     methods=METHODS, fixed_factor: float | None = None) -> dict:
    """Perplexity per method and length. Without fixed_factor the stretch matches each length."""
    out: dict = {}
    for method in methods:
        out[method] = {}
        for n in lengths:
            factor = fixed_factor if (fixed_factor and method != "rope") else max(1.0, n / train_len)
            set_method(model, method, factor, train_len)
            out[method][str(n)] = round(perplexity(evaluate_loss(model, dataset.val, n, device)), 4)
    model.set_rope_scaling(None)
    return out


def finetune(model: TinyGPT, dataset: CharDataset, method: str, factor: float, train_len: int, context: int,
             steps: int, batch_size: int, lr: float, device: str, seed: int) -> TinyGPT:
    """A short extra training run on longer text with the stretch switched on."""
    tuned = copy.deepcopy(model)
    set_method(tuned, method, factor if method != "rope" else 1.0, train_len)
    gen = set_seed(seed + 1000)
    optimizer = torch.optim.AdamW(tuned.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.1)
    tuned.train()
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = lr * min(1.0, (step + 1) / max(1, steps // 10))
        x, y = get_batch(dataset.train, batch_size, context, gen, device)
        _, loss, _ = tuned(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tuned.parameters(), 1.0)
        optimizer.step()
    tuned.eval()
    return tuned


def summarize(runs: list[dict], lengths: list[int], train_len: int, alibi: dict | None) -> dict:
    summary: dict = {"train_length": train_len, "lengths": lengths, "seeds": len(runs), "settings": {}}
    for setting in ("zero_shot", "fine_tuned"):
        block = {}
        for method in METHODS:
            stats = {}
            for n in lengths:
                values = [r[setting][method][str(n)] for r in runs]
                stats[str(n)] = {"mean": round(statistics.mean(values), 3),
                                 "std": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0}
            block[method] = stats
        summary["settings"][setting] = block
    # Share of plain RoPE's breakdown (zero-shot, no stretch) that each method undoes, in both settings:
    # 100% = as good as at the training length, 0% = no better than plain RoPE, negative = worse.
    zs = summary["settings"]["zero_shot"]["rope"]
    base = zs[str(train_len)]["mean"]
    for block in summary["settings"].values():
        for method in METHODS:
            block[method]["recovered"] = {}
            for n in lengths[1:]:
                damage = zs[str(n)]["mean"] - base
                gain = zs[str(n)]["mean"] - block[method][str(n)]["mean"]
                block[method]["recovered"][str(n)] = round(gain / damage, 3) if damage > 0 else None
    summary["alibi_reference"] = alibi
    return summary


def to_markdown(summary: dict) -> str:
    lengths, base = summary["lengths"], summary["train_length"]
    lines = [f"# Stretching RoPE: validation perplexity (lower is better), trained on {base}-character windows", ""]
    titles = {"zero_shot": "Zero-shot (no extra training, stretch = length / training length)",
              "fine_tuned": "Fine-tuned (short extra training on 4x longer text, stretch 4)"}
    for setting, title in titles.items():
        block = summary["settings"][setting]
        lines += [f"## {title}", "",
                  "| Method | " + " | ".join(f"{n} ({n // base}x)" for n in lengths)
                  + " | RoPE breakdown undone at 8x |",
                  "| --- | " + " | ".join("---:" for _ in lengths) + " | ---: |"]
        for method in METHODS:
            cells = [f"{block[method][str(n)]['mean']:.2f} ± {block[method][str(n)]['std']:.2f}" for n in lengths]
            rec = block[method]["recovered"].get(str(lengths[-1]))
            skip = rec is None or (method == "rope" and setting == "zero_shot")
            lines.append(f"| {LABELS[method]} | " + " | ".join(cells) + f" | {'-' if skip else f'{rec:.0%}'} |")
        lines.append("")
    if summary.get("alibi_reference"):
        a = summary["alibi_reference"]
        ref = ", ".join(f"{n}: {a[str(n)]:.2f}" for n in lengths if str(n) in a)
        lines += [f"ALiBi from session 2 for reference: {ref}", ""]
    lines.append(f"Mean ± standard deviation over {summary['seeds']} seed(s).")
    return "\n".join(lines) + "\n"


def plot(summary: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lengths, base = summary["lengths"], summary["train_length"]
    colors = {"rope": "tab:red", "pi": "tab:orange", "ntk": "tab:purple", "yarn": "tab:blue"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, (setting, title) in zip(axes, [("zero_shot", "Zero-shot: stretch at test time only"),
                                           ("fine_tuned", "After a short fine-tune on 4x longer text")], strict=True):
        block = summary["settings"][setting]
        for method in METHODS:
            means = [block[method][str(n)]["mean"] for n in lengths]
            stds = [block[method][str(n)]["std"] for n in lengths]
            ax.errorbar(lengths, means, yerr=stds, marker="o", capsize=4, label=LABELS[method], color=colors[method])
        a = summary.get("alibi_reference")
        if a:
            ax.plot(lengths, [a.get(str(n), math.nan) for n in lengths], "g--", label="ALiBi (session 2)")
        ax.axvline(base, color="black", linestyle=":", alpha=0.6)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(lengths, [str(n) for n in lengths])
        ax.set_xlabel("Text length at test time (characters)")
        ax.set_title(title)
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
    axes[0].set_ylabel("Validation perplexity (log scale, lower is better)")
    axes[0].legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_alibi_reference(lengths: list[int]) -> dict | None:
    path = Path(__file__).parent / "results" / "summary.json"
    try:
        s = json.loads(path.read_text())
        means = s["methods"]["alibi"]["perplexity"]
        return {str(n): means[str(n)]["mean"] for n in lengths if str(n) in means}
    except (OSError, KeyError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stretching RoPE: PI, NTK scaling and YaRN")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--multipliers", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--ft-multiplier", type=int, default=4, help="fine-tune on this many times the training length")
    parser.add_argument("--ft-steps", type=int, default=300)
    parser.add_argument("--ft-lr", type=float, default=3e-4)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--out", type=Path, default=Path("runs_extension"))
    args = parser.parse_args(argv)

    overrides = dict(QUICK) if args.quick else {}
    if args.steps:
        overrides["steps"] = args.steps
    if args.quick:
        args.ft_steps = min(args.ft_steps, 30)
    train_len = overrides.get("context", TrainConfig.context)
    lengths = [train_len * m for m in args.multipliers]
    ft_context = train_len * args.ft_multiplier
    ft_batch = max(1, TrainConfig.batch_size * train_len // ft_context) if not args.quick else 8

    dataset = CharDataset(args.text_file.read_text(encoding="utf-8") if args.text_file else load_text())
    args.out.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in args.seeds:
        cfg = TrainConfig(pos_type="rope", seed=seed, **overrides)
        print(f"\n== seed {seed}: training the RoPE model ({cfg.steps} steps)")
        model, history = train(cfg, dataset, max_seq_len=max(lengths) + 1)
        model.eval()
        run = {"seed": seed, "history": history,
               "zero_shot": evaluate_methods(model, dataset, lengths, train_len, cfg.device), "fine_tuned": {}}
        print("  zero-shot:", {m: run["zero_shot"][m] for m in METHODS})
        for method in METHODS:
            print(f"  fine-tuning with {method} on {ft_context}-character text ({args.ft_steps} steps)")
            tuned = finetune(model, dataset, method, args.ft_multiplier, train_len, ft_context, args.ft_steps,
                             ft_batch, args.ft_lr, cfg.device, seed)
            run["fine_tuned"][method] = evaluate_methods(tuned, dataset, lengths, train_len, cfg.device,
                                                         methods=(method,), fixed_factor=args.ft_multiplier)[method]
            del tuned
        print("  fine-tuned:", run["fine_tuned"])
        (args.out / f"rope_seed{seed}.json").write_text(json.dumps(run, indent=2))
        runs.append(run)

    alibi = None if args.quick or args.text_file else load_alibi_reference(lengths)
    summary = summarize(runs, lengths, train_len, alibi)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    markdown = to_markdown(summary)
    (args.out / "summary.md").write_text(markdown)
    plot(summary, args.out / "extension.png")
    print("\n" + markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
