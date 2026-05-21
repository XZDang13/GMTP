from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DESKTOP_ROOT = PROJECT_ROOT.parent
MOTION_ASSET_DIR = PROJECT_ROOT / "env" / "assests"
MOCAP_DATA_DIR = DESKTOP_ROOT / "mocap_data"
DEFAULT_MOCAP_DATASET_NAMES = ("CMU", "OMOMO")
MOCAP_TRAINING_DATA_DIR = MOCAP_DATA_DIR / "training_data"
MOCAP_DATASET_ALIASES = {
    "CMU": (
        "CMU",
        "CMU_data",
        "selected_1000_each_longer_than_2s/CMU_data",
    ),
    "OMOMO": (
        "OMOMO",
        "OMOMO_data",
        "selected_1000_each_longer_than_2s/OMOMO_data",
    ),
}
MOCAP_FLAT_DATASET_PREFIXES = {
    "CMU": "CMU_data__",
    "OMOMO": "OMOMO_data__",
}
_MAX_VERBOSE_MOTION_LABEL_NAMES = 12


def _directory_has_motion_files(path: Path) -> bool:
    return path.is_dir() and any(path.rglob("*.npz"))


@lru_cache(maxsize=None)
def _resolve_mocap_dataset_dir(dataset_name: str) -> Path | None:
    normalized_name = _normalized_dataset_name(dataset_name)
    if normalized_name is None:
        return None
    for relative_path in MOCAP_DATASET_ALIASES[normalized_name]:
        candidate = (MOCAP_DATA_DIR / relative_path).resolve()
        if _directory_has_motion_files(candidate):
            return candidate
    return None


@lru_cache(maxsize=None)
def _find_flattened_mocap_dataset_files(dataset_name: str) -> tuple[str, ...]:
    normalized_name = _normalized_dataset_name(dataset_name)
    if normalized_name is None or not MOCAP_TRAINING_DATA_DIR.is_dir():
        return ()
    prefix = MOCAP_FLAT_DATASET_PREFIXES.get(normalized_name)
    if prefix is None:
        return ()
    return tuple(
        str(path.resolve())
        for path in sorted(MOCAP_TRAINING_DATA_DIR.glob(f"{prefix}*.npz"))
        if path.is_file()
    )


def _resolve_mocap_dataset_sources(dataset_name: str) -> tuple[str, ...]:
    dataset_dir = _resolve_mocap_dataset_dir(dataset_name)
    if dataset_dir is not None:
        return (str(dataset_dir),)
    return _find_flattened_mocap_dataset_files(dataset_name)


def _discover_default_experiment_motion_files() -> tuple[str, ...]:
    sources: list[str] = []
    for dataset_name in DEFAULT_MOCAP_DATASET_NAMES:
        sources.extend(_resolve_mocap_dataset_sources(dataset_name))

    if sources:
        return tuple(sources)

    asset_motion_files = tuple(
        str(path.resolve())
        for path in sorted(MOTION_ASSET_DIR.glob("*.npz"))
        if path.is_file()
    )
    return asset_motion_files


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
        for alias in MOCAP_DATASET_ALIASES[dataset_name]:
            if normalized == Path(alias).name.lower():
                return dataset_name
    return None


DEFAULT_EXPERIMENT_MOTION_FILES = _discover_default_experiment_motion_files()


def _dataset_name_from_flattened_file_name(file_name: str) -> str | None:
    for dataset_name, prefix in MOCAP_FLAT_DATASET_PREFIXES.items():
        if str(file_name).startswith(prefix):
            return dataset_name
    return None


def _path_under_mocap_dataset(path: Path) -> str | None:
    try:
        relative_path = path.expanduser().resolve().relative_to(MOCAP_DATA_DIR.resolve())
    except ValueError:
        return None
    if not relative_path.parts:
        return None
    if len(relative_path.parts) >= 2 and relative_path.parts[0] == MOCAP_TRAINING_DATA_DIR.name:
        flattened_dataset_name = _dataset_name_from_flattened_file_name(relative_path.parts[1])
        if flattened_dataset_name is not None:
            return flattened_dataset_name
    for part in relative_path.parts:
        dataset_name = _normalized_dataset_name(part)
        if dataset_name is not None:
            return dataset_name
    return None


@lru_cache(maxsize=None)
def _find_mocap_motion_file_by_name(file_name: str) -> tuple[Path, ...]:
    if not MOCAP_DATA_DIR.is_dir():
        return ()
    matches = {path.resolve() for path in MOCAP_DATA_DIR.rglob(file_name) if path.is_file()}
    if MOCAP_TRAINING_DATA_DIR.is_dir():
        matches.update(
            path.resolve()
            for path in MOCAP_TRAINING_DATA_DIR.glob(f"*__{file_name}")
            if path.is_file()
        )
    return tuple(sorted(matches))


@lru_cache(maxsize=None)
def _find_mocap_motion_file_by_dataset_and_name(dataset_name: str, file_name: str) -> tuple[Path, ...]:
    normalized_name = _normalized_dataset_name(dataset_name)
    if normalized_name is None:
        return ()
    matches = []
    dataset_dir = _resolve_mocap_dataset_dir(normalized_name)
    if dataset_dir is not None:
        matches.extend(path.resolve() for path in dataset_dir.rglob(file_name) if path.is_file())
    if MOCAP_TRAINING_DATA_DIR.is_dir():
        prefix = MOCAP_FLAT_DATASET_PREFIXES[normalized_name]
        matches.extend(
            path.resolve()
            for path in MOCAP_TRAINING_DATA_DIR.glob(f"{prefix}*__{file_name}")
            if path.is_file()
        )
    return tuple(sorted(set(matches)))


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
            dataset_dir = _resolve_mocap_dataset_dir(dataset_name)
            if dataset_dir is not None:
                return (dataset_dir / Path(*path.parts[1:])).resolve()
            if len(path.parts) > 1:
                dataset_matches = _find_mocap_motion_file_by_dataset_and_name(dataset_name, path.parts[-1])
                if len(dataset_matches) == 1:
                    return dataset_matches[0]
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
        dataset_name = _normalized_dataset_name(str(motion_file))
        if dataset_name is not None and not _looks_like_explicit_path(str(motion_file)):
            dataset_sources = _resolve_mocap_dataset_sources(dataset_name)
            if dataset_sources:
                for source in dataset_sources:
                    path = Path(source)
                    if path.is_dir():
                        nested_motion_files = _find_motion_files_under_directory(str(path))
                        if not nested_motion_files:
                            raise FileNotFoundError(f"No .npz motion files found under directory: {path}")
                        resolved.extend(nested_motion_files)
                    else:
                        resolved.append(str(path))
                continue

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
