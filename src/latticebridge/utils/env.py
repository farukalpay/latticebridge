from __future__ import annotations

import os
from pathlib import Path


def parse_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            values[key] = value
    return values


def inject_env_from_dotenv(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    values = parse_dotenv(Path(path))
    for key, value in values.items():
        os.environ.setdefault(key, value)
    return values


def resolve_hf_token(extra_dotenv: str | Path | None = None) -> str:
    if extra_dotenv is not None:
        inject_env_from_dotenv(extra_dotenv)
    for key in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return ""
