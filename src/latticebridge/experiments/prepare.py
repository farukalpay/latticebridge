from __future__ import annotations

from pathlib import Path
import json

from latticebridge.data.adapters import build_records
from latticebridge.data.download import download_dataset_assets, load_manifest
from latticebridge.data.records import DatasetRecord, write_jsonl


def prepare_workspace(
    *,
    manifest_path: Path,
    cache_root: Path,
    processed_root: Path,
    tokenizer_corpus_path: Path,
    hf_token: str = "",
) -> dict[str, int]:
    manifest = load_manifest(manifest_path)
    download_dataset_assets(manifest, cache_root, token=hf_token)
    summary: dict[str, int] = {}

    combined: dict[str, list[DatasetRecord]] = {"train": [], "validation": [], "test": []}
    dataset_configs = manifest["datasets"]
    assert isinstance(dataset_configs, dict)

    for dataset_name, config in dataset_configs.items():
        assert isinstance(config, dict)
        for split in ("train", "validation", "test"):
            limit_key = f"{split}_limit"
            limit = int(config.get(limit_key, 0)) or None
            records = build_records(dataset_name, cache_root, split, limit=limit)
            combined[split].extend(records)
            summary[f"{dataset_name}.{split}"] = len(records)
            write_jsonl(processed_root / dataset_name / f"{split}.jsonl", records)

    for split, records in combined.items():
        write_jsonl(processed_root / f"{split}.jsonl", records)

    tokenizer_corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with tokenizer_corpus_path.open("w", encoding="utf-8") as handle:
        for split in ("train", "validation"):
            for record in combined[split]:
                handle.write(record.source_text + "\n")
                handle.write(record.target_text + "\n")

    (processed_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
