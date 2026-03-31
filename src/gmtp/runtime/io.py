from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class RunPaths:
    root: Path
    config_path: Path
    summary_path: Path
    checkpoints_dir: Path
    debug_dir: Path
    videos_dir: Path


def sanitize_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return sanitized or "run"


def build_run_paths(output_root: str | Path, category: str, name: str) -> RunPaths:
    root = Path(output_root).expanduser().resolve() / category / f"{datetime.now():%Y%m%d_%H%M%S}_{sanitize_name(name)}"
    root.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = root / "checkpoints"
    debug_dir = root / "debug"
    videos_dir = root / "videos"
    checkpoints_dir.mkdir(exist_ok=True)
    debug_dir.mkdir(exist_ok=True)
    videos_dir.mkdir(exist_ok=True)
    return RunPaths(
        root=root,
        config_path=root / "config.json",
        summary_path=root / "summary.json",
        checkpoints_dir=checkpoints_dir,
        debug_dir=debug_dir,
        videos_dir=videos_dir,
    )


def jsonify(value: Any) -> Any:
    if is_dataclass(value):
        return jsonify(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonify(item) for item in value]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def write_json(path: str | Path, payload: Any) -> Path:
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(jsonify(payload), handle, indent=2, sort_keys=True)
    return json_path
