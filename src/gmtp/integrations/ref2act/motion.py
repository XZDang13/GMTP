from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DESKTOP_ROOT = PROJECT_ROOT.parent
MOTION_ASSET_DIR = PROJECT_ROOT / "env" / "assests"
MOCAP_DATA_DIR = DESKTOP_ROOT / "mocap_data"
DEFAULT_MOCAP_DATASET_NAMES = ("CMU", "OMOMO")
_MAX_VERBOSE_MOTION_LABEL_NAMES = 12


def _discover_default_experiment_motion_files() -> tuple[str, ...]:
    dataset_dirs = tuple((MOCAP_DATA_DIR / name).resolve() for name in DEFAULT_MOCAP_DATASET_NAMES)
    missing_dirs = [path for path in dataset_dirs if not path.is_dir()]
    if missing_dirs:
        raise FileNotFoundError(
            "Default mocap dataset directories were not found: "
            f"{', '.join(str(path) for path in missing_dirs)}. "
            f"Expected CMU and OMOMO data under {MOCAP_DATA_DIR}."
        )

    empty_dirs = [path for path in dataset_dirs if not any(path.rglob("*.npz"))]
    if empty_dirs:
        raise FileNotFoundError(
            "Default mocap dataset directories contain no .npz motion files: "
            f"{', '.join(str(path) for path in empty_dirs)}."
        )

    return tuple(str(path) for path in dataset_dirs)


DEFAULT_EXPERIMENT_MOTION_FILES = _discover_default_experiment_motion_files()


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


def _looks_like_explicit_path(candidate: str) -> bool:
    text = str(candidate).strip()
    return text.startswith(("~", ".", "/")) or "/" in text or "\\" in text


def _normalized_dataset_name(name: str) -> str | None:
    normalized = str(name).strip().lower()
    for dataset_name in DEFAULT_MOCAP_DATASET_NAMES:
        if normalized == dataset_name.lower():
            return dataset_name
    return None


def _path_under_mocap_dataset(path: Path) -> str | None:
    try:
        relative_path = path.expanduser().resolve().relative_to(MOCAP_DATA_DIR.resolve())
    except ValueError:
        return None
    if not relative_path.parts:
        return None
    return _normalized_dataset_name(relative_path.parts[0])


@lru_cache(maxsize=None)
def _find_mocap_motion_file_by_name(file_name: str) -> tuple[Path, ...]:
    if not MOCAP_DATA_DIR.is_dir():
        return ()
    return tuple(sorted(path.resolve() for path in MOCAP_DATA_DIR.rglob(file_name) if path.is_file()))


@lru_cache(maxsize=None)
def _find_motion_files_under_directory(directory: str) -> tuple[str, ...]:
    path = Path(directory).expanduser().resolve()
    return tuple(str(item.resolve()) for item in sorted(path.rglob("*.npz")) if item.is_file())


def _resolve_motion_path_candidate(motion_file: str) -> Path:
    path = Path(motion_file).expanduser()
    if path.is_absolute():
        return path.resolve()

    project_path = (PROJECT_ROOT / path).resolve()
    if project_path.exists():
        return project_path

    desktop_path = (DESKTOP_ROOT / path).resolve()
    if desktop_path.exists():
        return desktop_path

    mocap_path = (MOCAP_DATA_DIR / path).resolve()
    if mocap_path.exists():
        return mocap_path

    if path.parts:
        dataset_name = _normalized_dataset_name(path.parts[0])
        if dataset_name is not None:
            return (MOCAP_DATA_DIR / dataset_name / Path(*path.parts[1:])).resolve()
        if str(path.parts[0]).lower() == "mocap_data":
            return (DESKTOP_ROOT / path).resolve()

    asset_file_name = path.name if path.suffix else f"{path.name}.npz"
    asset_candidate = (MOTION_ASSET_DIR / asset_file_name).resolve()
    if not path.suffix and asset_candidate.exists():
        return asset_candidate

    if not _looks_like_explicit_path(motion_file):
        file_name = path.name if path.suffix == ".npz" else f"{path.name}.npz"
        mocap_matches = _find_mocap_motion_file_by_name(file_name)
        if len(mocap_matches) == 1:
            return mocap_matches[0]
        if len(mocap_matches) > 1:
            raise FileNotFoundError(
                f"Motion file name is ambiguous under {MOCAP_DATA_DIR}: {file_name}. "
                "Pass a dataset-relative or absolute path instead."
            )

    if asset_candidate.exists():
        return asset_candidate

    if path.suffix or _looks_like_explicit_path(motion_file):
        return project_path

    return asset_candidate


def resolve_motion_file(motion_file: str) -> str:
    path = _resolve_motion_path_candidate(motion_file)
    if not path.exists():
        raise FileNotFoundError(f"Motion file does not exist: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"Motion file resolves to a directory, not a file: {path}")
    return str(path)


def resolve_motion_files(motion_files: str | Sequence[str] | None) -> list[str]:
    resolved: list[str] = []
    for motion_file in normalize_motion_files(motion_files):
        path = _resolve_motion_path_candidate(motion_file)
        if not path.exists():
            raise FileNotFoundError(f"Motion file does not exist: {path}")
        if path.is_dir():
            nested_motion_files = _find_motion_files_under_directory(str(path))
            if not nested_motion_files:
                raise FileNotFoundError(f"No .npz motion files found under directory: {path}")
            resolved.extend(nested_motion_files)
            continue
        resolved.append(str(path))
    return resolved


def motion_names(motion_files: str | Sequence[str] | None) -> list[str]:
    return [Path(motion_file).stem for motion_file in normalize_motion_files(motion_files)]


def motion_label(motion_files: str | Sequence[str] | None) -> str:
    normalized = normalize_motion_files(motion_files)
    compact_dataset_label = _compact_mocap_dataset_label(normalized)
    if compact_dataset_label is not None:
        return compact_dataset_label
    return "_".join(Path(motion_file).stem for motion_file in normalized)


def _compact_mocap_dataset_label(motion_files: Sequence[str]) -> str | None:
    if len(motion_files) <= _MAX_VERBOSE_MOTION_LABEL_NAMES:
        return None

    dataset_names: set[str] = set()
    for motion_file in motion_files:
        dataset_name = _path_under_mocap_dataset(Path(motion_file))
        if dataset_name is None:
            return None
        dataset_names.add(dataset_name)

    if not dataset_names:
        return None
    return "_".join(name for name in DEFAULT_MOCAP_DATASET_NAMES if name in dataset_names)


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
        candidate = _resolve_motion_path_candidate(motion_name)
        if candidate.exists():
            return [str(candidate.resolve())]

    return resolve_motion_files(default_motion_files)
