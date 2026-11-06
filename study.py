"""
THE STUDY: does the way a transformer encodes position change how well it reads,
and what happens when the text is longer than anything it saw in training?

Trains the same tiny GPT with each position method (none, sinusoidal, RoPE, ALiBi)
and several random seeds, then measures validation perplexity at 1x, 2x, 4x and 8x
the training length.

    python study.py                 # the full study (GPU recommended, ~20-30 min on a Colab T4)
    python study.py --quick         # tiny version on CPU in about a minute, for checking the pipeline
    python study.py --mlflow        # also log every run to MLflow

Writes runs/<method>_seed<k>.json, runs/summary.json, runs/summary.md and a
perplexity-vs-length plot.
"""

import argparse
import json
import statistics
from pathlib import Path

from attention import POS_TYPES
from data import CharDataset, load_text
from train import TrainConfig, evaluate_loss, perplexity, train

QUICK = {"steps": 150, "context": 64, "batch_size": 32, "n_layer": 2, "d_model": 64, "log_every": 50}


def run_one(cfg: TrainConfig, dataset: CharDataset, lengths: list[int], out_dir: Path, use_mlflow: bool) -> dict:
    model, history = train(cfg, dataset, max_seq_len=max(max(lengths), cfg.context))
    losses = {str(n): round(evaluate_loss(model, dataset.val, n, cfg.device), 4) for n in lengths}
    result = {
        "config": cfg.to_dict(),
        "history": history,
        "val_loss": losses,
        "val_perplexity": {n: round(perplexity(v), 4) for n, v in losses.items()},
        "sample": dataset.decode(model.generate(
            dataset.val[:1].view(1, 1).to(cfg.device), 200, temperature=0.8,
        )[0].tolist()),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{cfg.pos_type}_seed{cfg.seed}.json").write_text(json.dumps(result, indent=2))
    if use_mlflow:
        import mlflow

        with mlflow.start_run(run_name=f"{cfg.pos_type}-seed{cfg.seed}"):
            mlflow.log_params(cfg.to_dict())
            mlflow.log_metrics({f"val_ppl_{n}": v for n, v in result["val_perplexity"].items()})
            mlflow.log_metric("tokens_per_second", history["tokens_per_second"])
    ppl = ", ".join(f"{n}: {v:.2f}" for n, v in result["val_perplexity"].items())
    print(f"  -> {cfg.pos_type} seed {cfg.seed}: perplexity by length {{{ppl}}}")
    return result


def summarize(results: list[dict], lengths: list[int], train_length: int) -> dict:
    """Mean and spread over seeds for every method and length, plus degradation vs the training length."""
    summary: dict = {"train_length": train_length, "lengths": lengths, "methods": {}}
    for pos in dict.fromkeys(r["config"]["pos_type"] for r in results):
        runs = [r for r in results if r["config"]["pos_type"] == pos]
        stats = {}
        for n in lengths:
            values = [r["val_perplexity"][str(n)] for r in runs]
            stats[str(n)] = {
                "mean": round(statistics.mean(values), 3),
                "std": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
            }
        base = stats[str(train_length)]["mean"]
        summary["methods"][pos] = {
            "seeds": len(runs),
            "perplexity": stats,
            "ratio_vs_train_length": {str(n): round(stats[str(n)]["mean"] / base, 3) for n in lengths},
            "tokens_per_second": round(statistics.mean(r["history"]["tokens_per_second"] for r in runs)),
            "n_params": runs[0]["history"]["n_params"],
        }
    return summary


def to_markdown(summary: dict) -> str:
    lengths, base = summary["lengths"], summary["train_length"]
    lines = [
        f"# Results: validation perplexity (lower is better), trained on {base}-character windows",
        "",
        "| Method | " + " | ".join(f"{n} ({n // base}x)" for n in lengths) + " | Training speed |",
        "| --- | " + " | ".join("---:" for _ in lengths) + " | ---: |",
    ]
    for pos, m in summary["methods"].items():
        cells = []
        for n in lengths:
            s = m["perplexity"][str(n)]
            cells.append(f"{s['mean']:.2f} ± {s['std']:.2f}")
        lines.append(f"| {pos} | " + " | ".join(cells) + f" | {m['tokens_per_second']:,} chars/s |")
    lines += [
        "",
        "## Degradation: perplexity at each length divided by perplexity at the training length",
        "",
        "| Method | " + " | ".join(f"{n // base}x" for n in lengths) + " |",
        "| --- | " + " | ".join("---:" for _ in lengths) + " |",
    ]
    for pos, m in summary["methods"].items():
        lines.append(f"| {pos} | " + " | ".join(f"{m['ratio_vs_train_length'][str(n)]:.2f}" for n in lengths) + " |")
    seeds = {m["seeds"] for m in summary["methods"].values()}
    lines += ["", f"Mean ± standard deviation over {', '.join(map(str, sorted(seeds)))} seed(s) per method."]
    return "\n".join(lines) + "\n"


def plot(summary: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lengths, base = summary["lengths"], summary["train_length"]
    colors = {"none": "gray", "sinusoidal": "tab:blue", "rope": "tab:red", "alibi": "tab:green"}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for pos, m in summary["methods"].items():
        means = [m["perplexity"][str(n)]["mean"] for n in lengths]
        stds = [m["perplexity"][str(n)]["std"] for n in lengths]
        ax.errorbar(lengths, means, yerr=stds, marker="o", capsize=4, label=pos, color=colors.get(pos))
    ax.axvline(base, color="black", linestyle=":", alpha=0.6)
    ax.text(base, ax.get_ylim()[1], " trained up to here", va="top", fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(lengths, [str(n) for n in lengths])
    ax.set_xlabel("Text length at test time (characters)")
    ax.set_ylabel("Validation perplexity (log scale, lower is better)")
    ax.set_title("What happens beyond the training length?")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument("--pos", nargs="+", default=list(POS_TYPES), choices=POS_TYPES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int)
    parser.add_argument("--context", type=int)
    parser.add_argument("--multipliers", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--quick", action="store_true", help="tiny CPU-sized run to check the pipeline")
    parser.add_argument("--text-file", type=Path, help="train on this text instead of Tiny Shakespeare")
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--plot", type=Path, help="default: <out>/length_generalization.png")
    parser.add_argument("--mlflow", action="store_true")
    args = parser.parse_args(argv)

    overrides = dict(QUICK) if args.quick else {}
    if args.steps:
        overrides["steps"] = args.steps
    if args.context:
        overrides["context"] = args.context
    context = overrides.get("context", TrainConfig.context)
    lengths = [context * m for m in args.multipliers]

    dataset = CharDataset(args.text_file.read_text(encoding="utf-8") if args.text_file else load_text())
    print(f"vocabulary {dataset.vocab_size} characters | train {len(dataset.train):,} | val {len(dataset.val):,}")

    results = []
    for pos in args.pos:
        for seed in args.seeds:
            cfg = TrainConfig(pos_type=pos, seed=seed, **overrides)
            results.append(run_one(cfg, dataset, lengths, args.out, args.mlflow))

    summary = summarize(results, lengths, context)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    markdown = to_markdown(summary)
    (args.out / "summary.md").write_text(markdown)
    plot(summary, args.plot or args.out / "length_generalization.png")
    print("\n" + markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
