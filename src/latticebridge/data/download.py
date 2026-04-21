from __future__ import annotations

import json
import tarfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests


def load_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _download(url: str, target: Path, token: str = "") -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target
    headers = {}
    hostname = urlparse(url).hostname or ""
    if token and ("huggingface.co" in hostname or "hf.co" in hostname):
        headers["Authorization"] = f"Bearer {token}"
    with requests.get(url, headers=headers, stream=True, timeout=120) as response:
        response.raise_for_status()
        with target.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)
    return target


def download_dataset_assets(manifest: dict[str, object], root: Path, token: str = "") -> dict[str, dict[str, Path]]:
    outputs: dict[str, dict[str, Path]] = {}
    datasets = manifest["datasets"]
    assert isinstance(datasets, dict)
    for dataset_name, config in datasets.items():
        assert isinstance(config, dict)
        files = config["files"]
        assert isinstance(files, dict)
        dataset_root = root / dataset_name
        dataset_root.mkdir(parents=True, exist_ok=True)
        local_files: dict[str, Path] = {}
        for alias, file_config in files.items():
            assert isinstance(file_config, dict)
            url = str(file_config["url"])
            filename = str(file_config["filename"])
            archive_path = _download(url, dataset_root / filename, token=token)
            local_files[alias] = archive_path
            if file_config.get("extract") == "zip":
                extract_root = dataset_root / alias
                if not extract_root.exists():
                    with zipfile.ZipFile(archive_path) as handle:
                        handle.extractall(extract_root)
                local_files[f"{alias}_dir"] = extract_root
            if file_config.get("extract") == "tar.gz":
                extract_root = dataset_root / alias
                if not extract_root.exists():
                    extract_root.mkdir(parents=True, exist_ok=True)
                    with tarfile.open(archive_path, "r:gz") as handle:
                        handle.extractall(extract_root)
                local_files[f"{alias}_dir"] = extract_root
        outputs[dataset_name] = local_files
    return outputs
