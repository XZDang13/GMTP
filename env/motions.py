from collections.abc import Sequence
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MOTION_ASSET_DIR = PROJECT_ROOT / "env" / "assests"

DEFAULT_EXPERIMENT_MOTION_FILES = (
    "env/assests/walk.npz",
    "env/assests/runing.npz",
    "env/assests/handshake.npz",
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
