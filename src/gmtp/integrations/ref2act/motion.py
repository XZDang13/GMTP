from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
MOTION_ASSET_DIR = PROJECT_ROOT / "env" / "assests"

DEFAULT_EXPERIMENT_MOTION_FILES = (
    #"env/assests/05_05_stageii.npz",
    "env/assests/06_13_stageii.npz",
    #"env/assests/55_02_stageii.npz",
    #"env/assests/63_25_stageii.npz",
    #"env/assests/61_10_stageii.npz",
    #"env/assests/115_06_stageii.npz",
    #"env/assests/115_02_stageii.npz",
    #"env/assests/120_01_stageii.npz",
)


def normalize_motion_files(motion_files: str | Sequence[str] | None) -> list[str]:
    if motion_files is None:
        return list(DEFAULT_EXPERIMENT_MOTION_FILES)

    if isinstance(motion_files, str):
        candidates = [motion_files]
    else:
        candidates = list(motion_files)

    normalized: list[str] = []
    for candidate in candidates:
        if candidate is None:
            continue
        parts = [part.strip() for part in str(candidate).split(",") if part.strip()]
        normalized.extend(parts or [str(candidate)])

    if not normalized:
        raise ValueError("At least one motion file must be provided.")
    return normalized


def resolve_motion_file(motion_file: str) -> str:
    path = Path(motion_file).expanduser()
    if not path.suffix:
        path = MOTION_ASSET_DIR / f"{path.name}.npz"
    elif not path.is_absolute():
        path = PROJECT_ROOT / path

    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Motion file does not exist: {path}")
    return str(path)


def resolve_motion_files(motion_files: str | Sequence[str] | None) -> list[str]:
    return [resolve_motion_file(motion_file) for motion_file in normalize_motion_files(motion_files)]


def motion_names(motion_files: str | Sequence[str] | None) -> list[str]:
    return [Path(motion_file).stem for motion_file in normalize_motion_files(motion_files)]


def motion_label(motion_files: str | Sequence[str] | None) -> str:
    return "_".join(motion_names(motion_files))


def infer_motion_files_from_checkpoint(
    checkpoint_path: str | Path,
    actor_type: str,
    checkpoint_env: dict,
    default_motion_files: str | Sequence[str] | None = None,
) -> list[str]:
    checkpoint_motion_files = checkpoint_env.get("motion_files")
    if checkpoint_motion_files:
        try:
            return resolve_motion_files(checkpoint_motion_files)
        except FileNotFoundError:
            pass

    checkpoint_motion_names = checkpoint_env.get("motion_names")
    if checkpoint_motion_names:
        try:
            return resolve_motion_files(checkpoint_motion_names)
        except FileNotFoundError:
            pass

    marker = f"_{actor_type}_"
    checkpoint_stem = Path(checkpoint_path).stem
    if marker in checkpoint_stem:
        motion_name = checkpoint_stem.rsplit(marker, 1)[0]
        candidate = PROJECT_ROOT / "env" / "assests" / f"{motion_name}.npz"
        if candidate.exists():
            return [str(candidate.resolve())]

    return resolve_motion_files(default_motion_files)
