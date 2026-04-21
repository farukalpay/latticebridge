from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _render_dataset_bars(summary: dict, output_dir: Path) -> list[str]:
    figure_paths: list[str] = []
    for dataset_name, methods in summary.items():
        method_names = list(methods.keys())
        success = [methods[name]["success_rate"] for name in method_names]
        rouge = [methods[name]["rouge_l"] for name in method_names]
        runtime = [methods[name]["runtime_seconds"] for name in method_names]

        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        axes[0].bar(method_names, success, color="#2A6F97")
        axes[0].set_title(f"{dataset_name}: success")
        axes[0].set_ylim(0, 1.0)
        axes[1].bar(method_names, rouge, color="#A44A3F")
        axes[1].set_title(f"{dataset_name}: ROUGE-L")
        axes[1].set_ylim(0, max(0.2, max(rouge) * 1.15))
        axes[2].bar(method_names, runtime, color="#6C8E5E")
        axes[2].set_title(f"{dataset_name}: runtime")
        for axis in axes:
            axis.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        path = output_dir / f"{dataset_name}_benchmark.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        figure_paths.append(str(path))
    return figure_paths


def _render_frontier(summary: dict, output_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    dataset_colors = {
        "common_gen": "#2A6F97",
        "e2e_nlg": "#C16630",
    }
    method_markers = {
        "greedy": "o",
        "beam_filter": "s",
        "ancestral_best_of_k": "^",
        "twisted_smc": "D",
    }
    for dataset_name, methods in summary.items():
        color = dataset_colors.get(dataset_name, "#444444")
        for method_name, metrics in methods.items():
            ax.scatter(
                metrics["runtime_seconds"],
                metrics["coverage"],
                s=105,
                color=color,
                marker=method_markers.get(method_name, "o"),
                alpha=0.9,
                edgecolor="white",
                linewidth=0.8,
                label=f"{dataset_name} / {method_name}",
            )
    ax.set_xlabel("Runtime per example (s)")
    ax.set_ylabel("Anchor coverage")
    ax.set_title("Coverage-runtime frontier")
    ax.set_xscale("log")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    fig.tight_layout()
    path = output_dir / "coverage_runtime_frontier.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _render_training_curve(train_report_path: Path, output_dir: Path) -> str:
    report = _load_json(train_report_path)
    history = report["history"]
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(epochs, train_loss, marker="o", linewidth=2.2, color="#2A6F97", label="train")
    ax.plot(epochs, val_loss, marker="s", linewidth=2.2, color="#C16630", label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Prefix model training dynamics")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = output_dir / "training_curve.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def render_summary_figures(
    summary_path: Path,
    output_dir: Path,
    train_report_path: Path | None = None,
) -> list[str]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = _render_dataset_bars(summary, output_dir)
    figure_paths.append(_render_frontier(summary, output_dir))
    if train_report_path is not None and train_report_path.exists():
        figure_paths.append(_render_training_curve(train_report_path, output_dir))
    return figure_paths
