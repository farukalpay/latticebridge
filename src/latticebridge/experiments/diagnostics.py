from __future__ import annotations

from collections import Counter, defaultdict
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

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
CONTENT_ALPHA_MIN_CHARS = 3
CORPUS_COMMON_FRACTION = 0.01
MIN_CORPUS_COMMON_DOCUMENTS = 5
QUALITY_ROUGE_WEIGHT = 0.35
QUALITY_TOKEN_F1_WEIGHT = 0.25
QUALITY_REQUIRED_COVERAGE_WEIGHT = 0.30
QUALITY_SOURCE_COVERAGE_WEIGHT = 0.35
QUALITY_COVERAGE_GAIN_WEIGHT = 0.15
QUALITY_SURFACE_NOISE_WEIGHT = 2.00
QUALITY_INTRUSION_WEIGHT = 0.20


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


def _word_set(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return {match.group(0).lower() for match in WORD_RE.finditer(normalized)}


def _content_words(text: str) -> set[str]:
    return {
        word
        for word in _word_set(text)
        if any(char.isdigit() for char in word) or len(word) >= CONTENT_ALPHA_MIN_CHARS
    }


def _normalize_for_match(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()
    return _normalize_ws(normalized)


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_text = _normalize_for_match(text)
    normalized_phrase = _normalize_for_match(phrase)
    if not normalized_phrase:
        return False
    pattern = re.escape(normalized_phrase)
    if normalized_phrase[0].isalnum():
        pattern = r"(?<![a-z0-9])" + pattern
    if normalized_phrase[-1].isalnum():
        pattern += r"(?![a-z0-9])"
    return re.search(pattern, normalized_text) is not None


def _surface_noise(text: str, evidence_words: set[str]) -> float:
    words = list(_content_words(text))
    if not words:
        return 0.0
    unsupported = [word for word in words if word not in evidence_words]
    return len(unsupported) / len(words)


def _phrase_intrusions(text: str, phrase_index: dict[str, list[str]]) -> list[str]:
    candidate_phrases: set[str] = set()
    for word in _content_words(text):
        candidate_phrases.update(phrase_index.get(word, []))
    return sorted(phrase for phrase in candidate_phrases if _contains_phrase(text, phrase))


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
    corpus_document_frequency: Counter[str] = Counter()
    phrase_inventory_by_dataset: dict[str, set[str]] = defaultdict(set)
    phrase_term_frequency_by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        evidence_surface = " ".join([record.source_text, record.target_text, *record.references])
        corpus_document_frequency.update(_content_words(evidence_surface))
        for phrase in record.candidate_phrases:
            normalized_phrase = _normalize_for_match(phrase)
            if normalized_phrase:
                phrase_inventory_by_dataset[record.dataset_name].add(phrase)
                phrase_term_frequency_by_dataset[record.dataset_name].update(_content_words(phrase))
                corpus_document_frequency.update(_content_words(phrase))
    corpus_common_words = {
        word
        for word, count in corpus_document_frequency.items()
        if count >= max(MIN_CORPUS_COMMON_DOCUMENTS, int(CORPUS_COMMON_FRACTION * max(1, len(records))))
    }
    grouped = defaultdict(list)
    for record in records:
        grouped[record.dataset_name].append(record)

    task_lookup: dict[tuple[str, str], dict[str, object]] = {}
    for dataset_name, dataset_records in grouped.items():
        selected_records = dataset_records[:per_dataset_limit]
        record_lookup = {record.example_id: record for record in selected_records}
        tasks = build_benchmark_tasks(
            selected_records,
            tokenizer,
            max_anchors=max_anchors,
            min_anchors=min_anchors,
        )
        for task in tasks:
            source_phrase_keys = {_normalize_for_match(phrase) for phrase in task.source_phrases}
            contrast_phrases = sorted(
                phrase
                for phrase in phrase_inventory_by_dataset[dataset_name]
                if _normalize_for_match(phrase) not in source_phrase_keys
            )
            contrast_phrase_index: dict[str, list[str]] = defaultdict(list)
            term_frequency = phrase_term_frequency_by_dataset[dataset_name]
            for phrase in contrast_phrases:
                phrase_terms = _content_words(phrase)
                if not phrase_terms:
                    continue
                index_term = min(phrase_terms, key=lambda term: (term_frequency.get(term, 0), term))
                contrast_phrase_index[index_term].append(phrase)
            record = record_lookup[task.example_id]
            task_evidence = " ".join(
                [
                    task.source_text,
                    task.target_text,
                    *task.references,
                    *task.source_phrases,
                    *record.candidate_phrases,
                ]
            )
            task_lookup[(task.dataset_name, task.example_id)] = {
                "source_text": task.source_text,
                "required_phrases": task.required_phrases,
                "references": task.references,
                "evidence_words": _content_words(task_evidence) | corpus_common_words,
                "contrast_phrase_index": dict(contrast_phrase_index),
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
    return (
        QUALITY_ROUGE_WEIGHT * float(record["smc_rouge_l"])
        + QUALITY_TOKEN_F1_WEIGHT * float(record["smc_token_f1"])
        + QUALITY_REQUIRED_COVERAGE_WEIGHT * float(record["smc_coverage"])
        + QUALITY_SOURCE_COVERAGE_WEIGHT * float(record["smc_source_coverage"])
        + QUALITY_COVERAGE_GAIN_WEIGHT * max(0.0, float(record["coverage_gain"]))
        - QUALITY_SURFACE_NOISE_WEIGHT * float(record["smc_surface_noise"])
        - QUALITY_INTRUSION_WEIGHT * float(record["smc_source_intrusion_count"])
    )


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
    selected: list[dict[str, object]] = []
    seen_keys: set[tuple[str, str]] = set()
    seen_anchor_sets: set[tuple[str, ...]] = set()
    scenario_cap = max(2, per_dataset_examples // 2)
    scenario_counts: dict[str, int] = defaultdict(int)
    candidates = sorted(records, key=_scenario_rank, reverse=True)
    for record in candidates:
        key = (str(record["dataset_name"]), str(record["example_id"]))
        anchor_signature = tuple(sorted(str(anchor).lower() for anchor in record["anchors"]))
        scenario = str(record["scenario"])
        if key in seen_keys or anchor_signature in seen_anchor_sets:
            continue
        if scenario_counts[scenario] >= scenario_cap:
            continue
        selected.append(record)
        seen_keys.add(key)
        seen_anchor_sets.add(anchor_signature)
        scenario_counts[scenario] += 1
        if len(selected) >= per_dataset_examples:
            break
    if len(selected) < per_dataset_examples:
        for record in candidates:
            key = (str(record["dataset_name"]), str(record["example_id"]))
            anchor_signature = tuple(sorted(str(anchor).lower() for anchor in record["anchors"]))
            if key in seen_keys or anchor_signature in seen_anchor_sets:
                continue
            selected.append(record)
            seen_keys.add(key)
            seen_anchor_sets.add(anchor_signature)
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
    smc_surface_noise = _surface_noise(str(smc["text"]), set(task_info["evidence_words"]))
    baseline_surface_noise = _surface_noise(str(baseline["text"]), set(task_info["evidence_words"]))
    phrase_index = dict(task_info["contrast_phrase_index"])
    smc_intrusions = _phrase_intrusions(str(smc["text"]), phrase_index)
    baseline_intrusions = _phrase_intrusions(str(baseline["text"]), phrase_index)
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
        "baseline_source_coverage": float(baseline.get("metadata", {}).get("source_coverage", baseline["coverage"])),
        "baseline_surface_noise": baseline_surface_noise,
        "baseline_source_intrusions": baseline_intrusions,
        "baseline_source_intrusion_count": float(len(baseline_intrusions)),
        "smc_text": str(smc["text"]),
        "smc_success": bool(smc["success"]),
        "smc_coverage": float(smc["coverage"]),
        "smc_rouge_l": float(smc["rouge_l"]),
        "smc_token_f1": float(smc["token_f1"]),
        "smc_source_coverage": float(smc.get("metadata", {}).get("source_coverage", smc["coverage"])),
        "smc_surface_noise": smc_surface_noise,
        "smc_source_intrusions": smc_intrusions,
        "smc_source_intrusion_count": float(len(smc_intrusions)),
        "coverage_gain": float(smc["coverage"]) - float(baseline["coverage"]),
        "source_coverage_gain": float(smc.get("metadata", {}).get("source_coverage", smc["coverage"]))
        - float(baseline.get("metadata", {}).get("source_coverage", baseline["coverage"])),
        "acceptance_mass": acceptance_mass,
        "mean_ess": mean_ess,
    }
    record["scenario"] = _scenario_bucket(record)
    return record


def _render_latex(records_by_dataset: dict[str, list[dict[str, object]]]) -> str:
    lines = [
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
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
                r"\begin{longtable}{>{\raggedright\arraybackslash}p{0.10\textwidth}>{\raggedright\arraybackslash}p{0.24\textwidth}>{\raggedright\arraybackslash}p{0.22\textwidth}>{\raggedright\arraybackslash}p{0.22\textwidth}>{\raggedright\arraybackslash}p{0.11\textwidth}}",
                r"\toprule",
                r"Case & Anchors & Best baseline & LatticeBridge & Particle state \\",
                r"\midrule",
                r"\endfirsthead",
                r"\toprule",
                r"Case & Anchors & Best baseline & LatticeBridge & Particle state \\",
                r"\midrule",
                r"\endhead",
            ]
        )
        for row in rows:
            case_label = _latex_escape(SCENARIO_LABELS.get(str(row["scenario"]), str(row["scenario"])))
            anchors = _latex_escape(", ".join(str(value) for value in row["anchors"]))
            baseline_metrics = _latex_escape(
                f"{row['baseline_method']}; "
                f"req {row['baseline_coverage']:.2f}; "
                f"src {row['baseline_source_coverage']:.2f}; "
                f"intr {row['baseline_source_intrusion_count']:.0f}; "
                f"RL {row['baseline_rouge_l']:.2f}"
            )
            bridge_metrics = _latex_escape(
                f"req {row['smc_coverage']:.2f}; "
                f"src {row['smc_source_coverage']:.2f}; "
                f"intr {row['smc_source_intrusion_count']:.0f}; "
                f"RL {row['smc_rouge_l']:.2f}"
            )
            particle_state = _latex_escape(
                f"mass {row['acceptance_mass']:.2f}"
                f"; ESS {row['mean_ess']:.1f}"
            )
            lines.append(rf"{case_label} & {anchors} & {baseline_metrics} & {bridge_metrics} & {particle_state} \\")
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
