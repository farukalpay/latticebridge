from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from .records import DatasetRecord


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _common_gen_records(root: Path, split: str, limit: int | None = None) -> list[DatasetRecord]:
    filename = {
        "train": "commongen.train.jsonl",
        "validation": "commongen.dev.jsonl",
        "test": "commongen.test_noref.jsonl",
    }[split]
    path = root / "common_gen" / "bundle" / filename
    records: list[DatasetRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and len(records) >= limit:
                break
            raw = json.loads(line.replace(", }", "}"))
            concepts = raw["concept_set"].split("#")
            if split == "train":
                scenes = raw["scene"]
                for scene_idx, scene in enumerate(scenes):
                    records.append(
                        DatasetRecord(
                            dataset_name="common_gen",
                            split=split,
                            example_id=f"common_gen-{idx}-{scene_idx}",
                            source_text="concepts: " + " | ".join(concepts),
                            target_text=_normalize_ws(scene),
                            candidate_phrases=[_normalize_ws(value) for value in concepts],
                            metadata={"concepts": concepts},
                        )
                    )
                    if limit is not None and len(records) >= limit:
                        break
            else:
                refs = [_normalize_ws(scene) for scene in raw.get("scene", [])]
                records.append(
                    DatasetRecord(
                        dataset_name="common_gen",
                        split=split,
                        example_id=f"common_gen-{idx}",
                        source_text="concepts: " + " | ".join(concepts),
                        target_text=refs[0] if refs else "",
                        candidate_phrases=[_normalize_ws(value) for value in concepts],
                        references=refs,
                        metadata={"concepts": concepts},
                    )
                )
    return records


def _parse_e2e_mr(raw: str) -> list[tuple[str, str]]:
    fields: list[tuple[str, str]] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if "[" not in piece or not piece.endswith("]"):
            continue
        key, value = piece.split("[", 1)
        fields.append((key.strip(), value[:-1].strip()))
    return fields


def _serialize_e2e(fields: list[tuple[str, str]]) -> str:
    return " ; ".join(f"{key} = {value}" for key, value in fields)


def _e2e_records(root: Path, split: str, limit: int | None = None) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    if split == "train":
        path = root / "e2e_nlg" / "train-fixed.no-ol.csv"
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for idx, row in enumerate(reader):
                if limit is not None and idx >= limit:
                    break
                fields = _parse_e2e_mr(row["mr"])
                records.append(
                    DatasetRecord(
                        dataset_name="e2e_nlg",
                        split=split,
                        example_id=f"e2e_nlg-{idx}",
                        source_text=_serialize_e2e(fields),
                        target_text=_normalize_ws(row["ref"]),
                        candidate_phrases=[_normalize_ws(value) for _, value in fields],
                        metadata={"fields": fields},
                    )
                )
        return records

    path = root / "e2e_nlg" / ("validation.json" if split == "validation" else "test.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    examples = list(payload.values())[0] if isinstance(payload, dict) else payload
    for idx, example in enumerate(examples):
        if limit is not None and idx >= limit:
            break
        fields = _parse_e2e_mr(example["meaning_representation"])
        refs = [_normalize_ws(value) for value in example["references"]]
        records.append(
            DatasetRecord(
                dataset_name="e2e_nlg",
                split=split,
                example_id=f"e2e_nlg-{idx}",
                source_text=_serialize_e2e(fields),
                target_text=refs[0] if refs else "",
                candidate_phrases=[_normalize_ws(value) for _, value in fields],
                references=refs,
                metadata={"fields": fields},
            )
        )
    return records


def _wiki_bio_records(root: Path, split: str, limit: int | None = None) -> list[DatasetRecord]:
    split_map = {"train": "train", "validation": "valid", "test": "test"}
    prefix = split_map[split]
    data_root = root / "wiki_bio" / "bundle" / "wikipedia-biography-dataset" / prefix
    sent_file = data_root / f"{prefix}.sent"
    title_file = data_root / f"{prefix}.title"
    box_file = data_root / f"{prefix}.box"
    nb_file = data_root / f"{prefix}.nb"

    records: list[DatasetRecord] = []
    with (
        sent_file.open("r", encoding="utf-8") as sent_handle,
        title_file.open("r", encoding="utf-8") as title_handle,
        box_file.open("r", encoding="utf-8") as box_handle,
        nb_file.open("r", encoding="utf-8") as nb_handle,
    ):
        for idx, (title, infobox, nb_lines) in enumerate(zip(title_handle, box_handle, nb_handle)):
            if limit is not None and idx >= limit:
                break
            lines = [sent_handle.readline().strip() for _ in range(int(nb_lines.strip()))]
            target_text = _normalize_ws(" ".join(lines))
            field_pairs: list[tuple[str, str]] = []
            groups: dict[str, dict[int, str]] = {}
            for cell in infobox.strip().split("\t"):
                if "<none>" in cell or ":" not in cell or "_" not in cell.split(":", 1)[0]:
                    continue
                raw_key, raw_value = cell.split(":", 1)
                field_name, field_index = raw_key.rsplit("_", 1)
                if not field_index.isdigit():
                    continue
                groups.setdefault(field_name, {})[int(field_index)] = raw_value
            for field_name, pieces in groups.items():
                merged = " ".join(value for _, value in sorted(pieces.items()))
                field_pairs.append((field_name, _normalize_ws(merged)))
            source_text = "title = " + _normalize_ws(title) + " ; " + " ; ".join(
                f"{field} = {value}" for field, value in field_pairs
            )
            records.append(
                DatasetRecord(
                    dataset_name="wiki_bio",
                    split=split,
                    example_id=f"wiki_bio-{idx}",
                    source_text=source_text,
                    target_text=target_text,
                    candidate_phrases=[_normalize_ws(title)] + [value for _, value in field_pairs],
                    metadata={"title": _normalize_ws(title), "field_count": len(field_pairs)},
                )
            )
    return records


ADAPTERS = {
    "common_gen": _common_gen_records,
    "e2e_nlg": _e2e_records,
    "wiki_bio": _wiki_bio_records,
}


def build_records(dataset_name: str, cache_root: Path, split: str, limit: int | None = None) -> list[DatasetRecord]:
    return ADAPTERS[dataset_name](cache_root, split, limit=limit)
