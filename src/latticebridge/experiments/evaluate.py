from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path

import torch

from latticebridge.benchmarks.generation import (
    ancestral_best_of_k,
    beam_filter_decode,
    build_benchmark_tasks,
    greedy_decode,
    twisted_smc_decode,
)
from latticebridge.data.records import read_jsonl
from latticebridge.models.tokenizer import LatticeTokenizer


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _standard_error(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = _mean(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def _method_summary(rows) -> dict[str, float]:
    successes = [1.0 if row.success else 0.0 for row in rows]
    coverage = [row.coverage for row in rows]
    rouge_l = [row.rouge_l for row in rows]
    token_f1 = [row.token_f1 for row in rows]
    runtime = [row.runtime_seconds for row in rows]
    summary = {
        "count": float(len(rows)),
        "success_rate": _mean(successes),
        "success_rate_se": _standard_error(successes),
        "coverage": _mean(coverage),
        "coverage_se": _standard_error(coverage),
        "rouge_l": _mean(rouge_l),
        "rouge_l_se": _standard_error(rouge_l),
        "token_f1": _mean(token_f1),
        "token_f1_se": _standard_error(token_f1),
        "runtime_seconds": _mean(runtime),
        "runtime_seconds_se": _standard_error(runtime),
    }
    metadata_keys = sorted({key for row in rows for key in row.metadata if isinstance(row.metadata[key], (int, float))})
    for key in metadata_keys:
        values = [float(row.metadata[key]) for row in rows if isinstance(row.metadata.get(key), (int, float))]
        if values:
            summary[f"metadata_{key}"] = _mean(values)
            summary[f"metadata_{key}_se"] = _standard_error(values)
    return summary


def _write_snapshot(
    *,
    output_dir: Path,
    split: str,
    results: list[dict[str, object]],
    summary: dict[str, dict[str, dict[str, float]]],
    tasks: dict[str, dict[str, float]],
    config: dict[str, object],
    suffix: str = "",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{split}_results{suffix}.json"
    (output_dir / stem).write_text(json.dumps(results, indent=2), encoding="utf-8")
    stem = f"{split}_summary{suffix}.json"
    (output_dir / stem).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    stem = f"{split}_tasks{suffix}.json"
    (output_dir / stem).write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    stem = f"{split}_config{suffix}.json"
    (output_dir / stem).write_text(json.dumps(config, indent=2), encoding="utf-8")


def run_benchmarks(
    *,
    processed_root: Path,
    tokenizer_path: Path,
    model,
    output_dir: Path,
    device: torch.device,
    split: str = "test",
    per_dataset_limit: int = 120,
    max_new_tokens: int = 72,
    max_anchors: int = 3,
    min_anchors: int = 2,
    beam_size: int = 6,
    num_samples: int = 16,
    particles: int = 96,
    lambda_weight: float = 2.0,
    twist_scale: float = 2.0,
    sample_temperature: float = 0.95,
    smc_temperature: float = 0.9,
    ess_threshold: float = 0.5,
    split_interval: int = 12,
    elite_fraction: float = 0.2,
    support_scale: float = 0.4,
    lookahead_weight: float = 0.0,
    lookahead_depth: int = 0,
    lookahead_interval: int = 0,
    random_seed: int | None = None,
    log_interval: int = 0,
) -> dict[str, object]:
    tokenizer = LatticeTokenizer.load(tokenizer_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    records = read_jsonl(processed_root / f"{split}.jsonl")
    grouped = defaultdict(list)
    for record in records:
        grouped[record.dataset_name].append(record)

    config = {
        "split": split,
        "per_dataset_limit": per_dataset_limit,
        "max_new_tokens": max_new_tokens,
        "max_anchors": max_anchors,
        "min_anchors": min_anchors,
        "beam_size": beam_size,
        "num_samples": num_samples,
        "particles": particles,
        "lambda_weight": lambda_weight,
        "twist_scale": twist_scale,
        "sample_temperature": sample_temperature,
        "smc_temperature": smc_temperature,
        "ess_threshold": ess_threshold,
        "split_interval": split_interval,
        "elite_fraction": elite_fraction,
        "support_scale": support_scale,
        "lookahead_weight": lookahead_weight,
        "lookahead_depth": lookahead_depth,
        "lookahead_interval": lookahead_interval,
        "random_seed": random_seed,
        "log_interval": log_interval,
        "device": device.type,
        "constraint_selection": "empirical_source_idf_attested_in_reference",
        "candidate_selection": "accepting_then_required_coverage_then_source_coverage_then_source_intrusion_then_log_score_then_rouge_l",
    }

    results: list[dict[str, object]] = []
    summary: dict[str, dict[str, dict[str, float]]] = {}
    task_summary: dict[str, dict[str, float]] = {}
    for dataset_name, dataset_records in grouped.items():
        selected_records = dataset_records[:per_dataset_limit]
        tasks = build_benchmark_tasks(
            selected_records,
            tokenizer,
            max_anchors=max_anchors,
            min_anchors=min_anchors,
        )
        task_summary[dataset_name] = {
            "candidate_records": float(len(selected_records)),
            "benchmark_tasks": float(len(tasks)),
            "mean_required_phrases": _mean([float(len(task.required_phrases)) for task in tasks]),
            "max_anchors": float(max_anchors),
            "min_anchors": float(min_anchors),
        }
        dataset_results = []
        for task_index, task in enumerate(tasks, start=1):
            dataset_results.extend(
                [
                    greedy_decode(model, tokenizer, task, max_new_tokens=max_new_tokens, device=device),
                    beam_filter_decode(model, tokenizer, task, max_new_tokens=max_new_tokens, beam_size=beam_size, device=device),
                    ancestral_best_of_k(
                        model,
                        tokenizer,
                        task,
                        max_new_tokens=max_new_tokens,
                        num_samples=num_samples,
                        temperature=sample_temperature,
                        device=device,
                    ),
                    twisted_smc_decode(
                        model,
                        tokenizer,
                        task,
                        max_new_tokens=max_new_tokens,
                        particles=particles,
                        lambda_weight=lambda_weight,
                        twist_scale=twist_scale,
                        ess_threshold=ess_threshold,
                        split_interval=split_interval,
                        elite_fraction=elite_fraction,
                        temperature=smc_temperature,
                        support_scale=support_scale,
                        lookahead_weight=lookahead_weight,
                        lookahead_depth=lookahead_depth,
                        lookahead_interval=lookahead_interval,
                        device=device,
                    ),
                ]
            )
            if log_interval > 0 and (task_index % log_interval == 0 or task_index == len(tasks)):
                partial_summary = dict(summary)
                partial_by_method = defaultdict(list)
                for result in dataset_results:
                    partial_by_method[result.method].append(result)
                partial_summary[dataset_name] = {
                    method: _method_summary(rows)
                    for method, rows in partial_by_method.items()
                }
                partial_results = results + [result.to_dict() for result in dataset_results]
                _write_snapshot(
                    output_dir=output_dir,
                    split=split,
                    results=partial_results,
                    summary=partial_summary,
                    tasks=task_summary,
                    config=config,
                    suffix="_partial",
                )
                print(f"[benchmark] {dataset_name}: {task_index}/{len(tasks)} tasks", flush=True)

        summary[dataset_name] = {}
        by_method = defaultdict(list)
        for result in dataset_results:
            results.append(result.to_dict())
            by_method[result.method].append(result)
        for method, rows in by_method.items():
            summary[dataset_name][method] = _method_summary(rows)
        _write_snapshot(
            output_dir=output_dir,
            split=split,
            results=results,
            summary=summary,
            tasks=task_summary,
            config=config,
            suffix="_partial",
        )

    _write_snapshot(
        output_dir=output_dir,
        split=split,
        results=results,
        summary=summary,
        tasks=task_summary,
        config=config,
    )
    return {"results": results, "summary": summary, "tasks": task_summary}
