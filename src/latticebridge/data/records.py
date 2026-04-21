from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json


@dataclass
class DatasetRecord:
    dataset_name: str
    split: str
    example_id: str
    source_text: str
    target_text: str
    candidate_phrases: list[str]
    references: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "DatasetRecord":
        return cls(**json.loads(raw))


def write_jsonl(path: Path, records: list[DatasetRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.to_json())
            handle.write("\n")


def read_jsonl(path: Path) -> list[DatasetRecord]:
    return [
        DatasetRecord.from_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
