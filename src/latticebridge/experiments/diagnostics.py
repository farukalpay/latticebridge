from __future__ import annotations

from collections import defaultdict
import json
import re
import unicodedata
from pathlib import Path

from latticebridge.benchmarks.generation import build_benchmark_tasks
from latticebridge.data.records import read_jsonl
from latticebridge.models.tokenizer import LatticeTokenizer


DATASET_LABELS = {
    "common_gen": "CommonGen",
    "e2e_nlg": "E2E NLG",
    "wiki_bio": "WikiBio",
}

SCENARIO_LABELS = {
    "exact_lift": "exact lift",
    "coverage_lift": "coverage lift",
    "low_mass_success": "low-mass success",
    "near_miss": "near miss",
}


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _clean_text(text: str) -> str:
    compact = _normalize_ws(text)
    if not compact:
        return "-"
    return unicodedata.normalize("NFKD", compact).encode("ascii", "ignore").decode("ascii")


def _load_task_lookup(
    *,
    processed_root: Path,
    tokenizer_path: Path,
    split: str,
    per_dataset_limit: int,
    max_anchors: int,
    min_anchors: int,
) -> dict[tuple[str, str], dict[str, object]]:
    tokenizer = LatticeTokenizer.load(tokenizer_path)
    records = read_jsonl(processed_root / f"{split}.jsonl")
    grouped = defaultdict(list)
    for record in records:
        grouped[record.dataset_name].append(record)

    task_lookup: dict[tuple[str, str], dict[str, object]] = {}
    for dataset_name, dataset_records in grouped.items():
        selected_records = dataset_records[:per_dataset_limit]
        tasks = build_benchmark_tasks(
            selected_records,
            tokenizer,
            max_anchors=max_anchors,
            min_anchors=min_anchors,
        )
        for task in tasks:
            task_lookup[(task.dataset_name, task.example_id)] = {
                "source_text": task.source_text,
                "required_phrases": task.required_phrases,
                "references": task.references,
            }
    return task_lookup


def _best_baseline(method_rows: dict[str, dict[str, object]]) -> dict[str, object]:
    baseline_candidates = [
        row
        for method_name, row in method_rows.items()
        if method_name != "twisted_smc"
    ]
    return max(
        baseline_candidates,
        key=lambda row: (
            int(bool(row["success"])),
            float(row["coverage"]),
            float(row["rouge_l"]),
            float(row["token_f1"]),
        ),
    )


def _scenario_bucket(record: dict[str, object]) -> str:
    smc_success = bool(record["smc_success"])
    baseline_success = bool(record["baseline_success"])
    coverage_gain = float(record["coverage_gain"])
    acceptance_mass = float(record["acceptance_mass"])
    if smc_success and not baseline_success:
        return "exact_lift"
    if smc_success and acceptance_mass < 0.12:
        return "low_mass_success"
    if coverage_gain > 0.0:
        return "coverage_lift"
    return "near_miss"


def _quality_score(record: dict[str, object]) -> float:
    return 0.5 * (float(record["smc_rouge_l"]) + float(record["smc_token_f1"]))


def _scenario_rank(record: dict[str, object]) -> tuple[float, ...]:
    scenario = str(record["scenario"])
    if scenario == "exact_lift":
        return (
            _quality_score(record),
            float(record["smc_coverage"]),
            float(record["coverage_gain"]),
            float(record["acceptance_mass"]),
        )
    if scenario == "coverage_lift":
        return (
            _quality_score(record),
            float(record["coverage_gain"]),
            float(record["smc_coverage"]),
            float(record["acceptance_mass"]),
        )
    if scenario == "low_mass_success":
        return (
            _quality_score(record),
            -float(record["acceptance_mass"]),
            float(record["coverage_gain"]),
        )
    return (
        _quality_score(record),
        float(record["coverage_gain"]),
        -float(record["acceptance_mass"]),
    )


def _select_examples(records: list[dict[str, object]], per_dataset_examples: int) -> list[dict[str, object]]:
    scenario_limits = {
        "exact_lift": 3,
        "coverage_lift": 2,
        "low_mass_success": 1,
        "near_miss": 1,
    }
    selected: list[dict[str, object]] = []
    seen_keys: set[tuple[str, str]] = set()
    seen_anchor_sets: set[tuple[str, ...]] = set()
    for scenario_name in ("exact_lift", "coverage_lift", "low_mass_success", "near_miss"):
        candidates = [record for record in records if record["scenario"] == scenario_name]
        candidates.sort(key=_scenario_rank, reverse=True)
        for record in candidates:
            key = (str(record["dataset_name"]), str(record["example_id"]))
            anchor_signature = tuple(sorted(str(anchor).lower() for anchor in record["anchors"]))
            if key in seen_keys:
                continue
            if anchor_signature in seen_anchor_sets:
                continue
            if sum(1 for row in selected if row["scenario"] == scenario_name) >= scenario_limits[scenario_name]:
                break
            selected.append(record)
            seen_keys.add(key)
            seen_anchor_sets.add(anchor_signature)
    if len(selected) < per_dataset_examples:
        remaining = [record for record in records if (record["dataset_name"], record["example_id"]) not in seen_keys]
        remaining.sort(
            key=lambda record: (
                _quality_score(record),
                int(bool(record["smc_success"])) - int(bool(record["baseline_success"])),
                float(record["smc_coverage"]),
                float(record["acceptance_mass"]),
            ),
            reverse=True,
        )
        for record in remaining:
            anchor_signature = tuple(sorted(str(anchor).lower() for anchor in record["anchors"]))
            if anchor_signature in seen_anchor_sets:
                continue
            selected.append(record)
            seen_anchor_sets.add(anchor_signature)
            if len(selected) >= per_dataset_examples:
                break
    if len(selected) < per_dataset_examples:
        remaining = [record for record in records if (record["dataset_name"], record["example_id"]) not in seen_keys]
        remaining.sort(
            key=lambda record: (
                _quality_score(record),
                int(bool(record["smc_success"])) - int(bool(record["baseline_success"])),
                float(record["smc_coverage"]),
                float(record["acceptance_mass"]),
            ),
            reverse=True,
        )
        for record in remaining:
            selected.append(record)
            if len(selected) >= per_dataset_examples:
                break
    return selected[:per_dataset_examples]


def _record_for_example(
    *,
    dataset_name: str,
    example_id: str,
    methods: dict[str, dict[str, object]],
    task_info: dict[str, object],
) -> dict[str, object]:
    smc = methods["twisted_smc"]
    baseline = _best_baseline(methods)
    acceptance_mass = float(smc.get("metadata", {}).get("acceptance_mass", 0.0))
    mean_ess = float(smc.get("metadata", {}).get("mean_ess", 0.0))
    record = {
        "dataset_name": dataset_name,
        "example_id": example_id,
        "anchors": list(task_info["required_phrases"]),
        "source_text": str(task_info["source_text"]),
        "baseline_method": str(baseline["method"]),
        "baseline_text": str(baseline["text"]),
        "baseline_success": bool(baseline["success"]),
        "baseline_coverage": float(baseline["coverage"]),
        "baseline_rouge_l": float(baseline["rouge_l"]),
        "baseline_token_f1": float(baseline["token_f1"]),
        "smc_text": str(smc["text"]),
        "smc_success": bool(smc["success"]),
        "smc_coverage": float(smc["coverage"]),
        "smc_rouge_l": float(smc["rouge_l"]),
        "smc_token_f1": float(smc["token_f1"]),
        "coverage_gain": float(smc["coverage"]) - float(baseline["coverage"]),
        "acceptance_mass": acceptance_mass,
        "mean_ess": mean_ess,
    }
    record["scenario"] = _scenario_bucket(record)
    return record


def _render_latex(records_by_dataset: dict[str, list[dict[str, object]]]) -> str:
    lines = [
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\setlength{\LTleft}{0pt}",
        r"\setlength{\LTright}{0pt}",
        r"\renewcommand{\arraystretch}{1.08}",
    ]
    for dataset_name, rows in records_by_dataset.items():
        label = _latex_escape(DATASET_LABELS.get(dataset_name, dataset_name))
        lines.extend(
            [
                "",
                rf"\subsection*{{{label}}}",
                r"\begin{longtable}{>{\raggedright\arraybackslash}p{0.09\textwidth}>{\raggedright\arraybackslash}p{0.15\textwidth}>{\raggedright\arraybackslash}p{0.13\textwidth}>{\raggedright\arraybackslash}p{0.27\textwidth}>{\raggedright\arraybackslash}p{0.27\textwidth}}",
                r"\toprule",
                r"Case & Anchors & Metrics & Baseline & LatticeBridge \\",
                r"\midrule",
                r"\endfirsthead",
                r"\toprule",
                r"Case & Anchors & Metrics & Baseline & LatticeBridge \\",
                r"\midrule",
                r"\endhead",
            ]
        )
        for row in rows:
            case_label = _latex_escape(SCENARIO_LABELS.get(str(row["scenario"]), str(row["scenario"])))
            anchors = _latex_escape(", ".join(str(value) for value in row["anchors"]))
            metrics = _latex_escape(
                f"cov {row['baseline_coverage']:.2f} to {row['smc_coverage']:.2f}; "
                f"RL {row['baseline_rouge_l']:.2f} to {row['smc_rouge_l']:.2f}; "
                f"mass {row['acceptance_mass']:.2f}"
            )
            baseline_text = _latex_escape(_clean_text(str(row["baseline_text"])))
            smc_text = _latex_escape(_clean_text(str(row["smc_text"])))
            lines.append(rf"{case_label} & {anchors} & {metrics} & {baseline_text} & {smc_text} \\")
            lines.append(r"\midrule")
        lines.extend([r"\bottomrule", r"\end{longtable}"])
    lines.append(r"\endgroup")
    return "\n".join(lines) + "\n"


def build_example_diagnostics(
    *,
    results_path: Path,
    config_path: Path,
    processed_root: Path,
    tokenizer_path: Path,
    split: str,
    output_json_path: Path,
    output_tex_path: Path,
    per_dataset_examples: int = 8,
) -> dict[str, list[dict[str, object]]]:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    task_lookup = _load_task_lookup(
        processed_root=processed_root,
        tokenizer_path=tokenizer_path,
        split=split,
        per_dataset_limit=int(config["per_dataset_limit"]),
        max_anchors=int(config["max_anchors"]),
        min_anchors=int(config["min_anchors"]),
    )

    grouped = defaultdict(dict)
    for row in results:
        key = (str(row["dataset_name"]), str(row["example_id"]))
        grouped[key][str(row["method"])] = row

    records_by_dataset: dict[str, list[dict[str, object]]] = defaultdict(list)
    for (dataset_name, example_id), methods in grouped.items():
        if "twisted_smc" not in methods:
            continue
        task_info = task_lookup.get((dataset_name, example_id))
        if task_info is None:
            continue
        record = _record_for_example(
            dataset_name=dataset_name,
            example_id=example_id,
            methods=methods,
            task_info=task_info,
        )
        records_by_dataset[dataset_name].append(record)

    curated: dict[str, list[dict[str, object]]] = {}
    for dataset_name, rows in records_by_dataset.items():
        curated[dataset_name] = _select_examples(rows, per_dataset_examples)

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_tex_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(curated, indent=2), encoding="utf-8")
    output_tex_path.write_text(_render_latex(curated), encoding="utf-8")
    return curated
