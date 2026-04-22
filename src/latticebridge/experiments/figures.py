from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


METHOD_ORDER = ["greedy", "beam_filter", "ancestral_best_of_k", "twisted_smc"]
METHOD_LABELS = {
    "greedy": "Greedy",
    "beam_filter": "Beam",
    "ancestral_best_of_k": "Best-16",
    "twisted_smc": "SMC",
}
DATASET_LABELS = {
    "common_gen": "CommonGen",
    "e2e_nlg": "E2E NLG",
    "wiki_bio": "WikiBio",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ordered_methods(methods: dict) -> list[str]:
    ordered = [name for name in METHOD_ORDER if name in methods]
    ordered.extend(name for name in methods if name not in ordered)
    return ordered


def _render_dataset_bars(summary: dict, output_dir: Path) -> list[str]:
    figure_paths: list[str] = []
    for dataset_name, methods in summary.items():
        method_names = _ordered_methods(methods)
        labels = [METHOD_LABELS.get(name, name) for name in method_names]
        success = [methods[name]["success_rate"] for name in method_names]
        success_se = [methods[name].get("success_rate_se", 0.0) for name in method_names]
        coverage = [methods[name]["coverage"] for name in method_names]
        coverage_se = [methods[name].get("coverage_se", 0.0) for name in method_names]
        rouge = [methods[name]["rouge_l"] for name in method_names]
        rouge_se = [methods[name].get("rouge_l_se", 0.0) for name in method_names]
        runtime = [methods[name]["runtime_seconds"] for name in method_names]
        runtime_se = [methods[name].get("runtime_seconds_se", 0.0) for name in method_names]

        fig, axes_grid = plt.subplots(2, 2, figsize=(8.0, 6.0))
        axes = axes_grid.reshape(-1)
        dataset_label = DATASET_LABELS.get(dataset_name, dataset_name)
        axes[0].bar(labels, success, color="#2A6F97", yerr=success_se, capsize=4, ecolor="#17354C")
        axes[0].set_title("Success")
        axes[0].set_ylim(0, 1.0)
        axes[1].bar(labels, coverage, color="#3D8B5F", yerr=coverage_se, capsize=4, ecolor="#254B33")
        axes[1].set_title("Coverage")
        axes[1].set_ylim(0, 1.0)
        axes[2].bar(labels, rouge, color="#A44A3F", yerr=rouge_se, capsize=4, ecolor="#682820")
        axes[2].set_title("ROUGE-L")
        axes[2].set_ylim(0, max(0.2, max(rouge) * 1.15))
        axes[3].bar(labels, runtime, color="#6C8E5E", yerr=runtime_se, capsize=4, ecolor="#4A6340")
        axes[3].set_title("Runtime (s)")
        for axis in axes:
            axis.tick_params(axis="x", rotation=18)
            axis.grid(axis="y", alpha=0.2, linestyle="--")
        fig.suptitle(f"{dataset_label} validation benchmark", y=1.03, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        path = output_dir / f"{dataset_name}_benchmark.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        figure_paths.append(str(path))
    return figure_paths


def _render_frontier(summary: dict, output_dir: Path) -> str:
    dataset_names = list(summary.keys())
    fig, axes = plt.subplots(1, len(dataset_names), figsize=(4.9 * len(dataset_names), 4.3), sharey=True)
    if len(dataset_names) == 1:
        axes = [axes]
    method_styles = {
        "greedy": ("o", "#496D89"),
        "beam_filter": ("s", "#6D8F3E"),
        "ancestral_best_of_k": ("^", "#C16630"),
        "twisted_smc": ("D", "#8E4B63"),
    }
    legend_handles = []
    for ax, dataset_name in zip(axes, dataset_names):
        methods = summary[dataset_name]
        runtimes = [methods[name]["runtime_seconds"] for name in _ordered_methods(methods)]
        method_names = _ordered_methods(methods)
        for method_name in method_names:
            metrics = methods[method_name]
            marker, color = method_styles.get(method_name, ("o", "#444444"))
            ax.scatter(
                metrics["runtime_seconds"],
                metrics["coverage"],
                s=118,
                color=color,
                marker=marker,
                alpha=0.9,
                edgecolor="white",
                linewidth=0.95,
            )
            ax.errorbar(
                metrics["runtime_seconds"],
                metrics["coverage"],
                xerr=metrics.get("runtime_seconds_se", 0.0),
                yerr=metrics.get("coverage_se", 0.0),
                fmt="none",
                ecolor=color,
                elinewidth=1.0,
                alpha=0.4,
                capsize=3,
            )
        if not legend_handles:
            for method_name in method_names:
                marker, color = method_styles.get(method_name, ("o", "#444444"))
                legend_handles.append(
                    ax.scatter([], [], s=92, color=color, marker=marker, label=METHOD_LABELS.get(method_name, method_name))
                )
        ax.set_xlabel("Runtime per example (s)")
        ax.set_title(DATASET_LABELS.get(dataset_name, dataset_name), fontsize=12)
        ax.set_xscale("log")
        ax.set_xlim(min(runtimes) * 0.75, max(runtimes) * 1.85)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.22, linestyle="--", linewidth=0.8)
        ax.tick_params(labelsize=10)
    axes[0].set_ylabel("Anchor coverage")
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.08),
            ncol=min(4, len(legend_handles)),
            frameon=False,
            fontsize=10,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
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
    fig.tight_layout(rect=(0, 0, 1, 0.97))
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
