from pathlib import Path

from gmtp.integrations.ref2act.motion import (
    DEFAULT_EXPERIMENT_MOTION_FILES,
    DEFAULT_MOCAP_DATASET_NAMES,
    MOCAP_DATA_DIR,
    infer_motion_files_from_checkpoint,
    motion_label,
    motion_names,
    normalize_motion_files,
    resolve_motion_file,
    resolve_motion_files,
)


def test_default_experiment_motion_files_use_cmu_and_omomo_datasets():
    assert DEFAULT_EXPERIMENT_MOTION_FILES
    assert all(Path(path).exists() for path in DEFAULT_EXPERIMENT_MOTION_FILES)

    resolved = resolve_motion_files(DEFAULT_EXPERIMENT_MOTION_FILES)

    assert resolved
    assert all(Path(path).is_file() for path in resolved)
    if len(resolved) > 12:
        assert motion_label(resolved) in {"CMU_OMOMO", "CMU", "OMOMO"}


def test_motion_helpers_normalize_resolve_and_label():
    normalized = normalize_motion_files(["02_04_stageii", "env/assests/05_04_stageii.npz"])
    assert normalized == ["02_04_stageii", "env/assests/05_04_stageii.npz"]

    resolved = resolve_motion_files(normalized)
    assert all(Path(path).exists() for path in resolved)
    assert motion_names(normalized) == ["02_04_stageii", "05_04_stageii"]
    assert motion_label(normalized) == "02_04_stageii_05_04_stageii"
    assert resolve_motion_file("02_04_stageii").endswith("02_04_stageii.npz")


def test_motion_helpers_resolve_mocap_dataset_aliases_and_motion_names():
    resolved = resolve_motion_files(["CMU", "OMOMO"])

    assert any("CMU" in Path(path).name or "/CMU" in path for path in resolved)
    assert any("OMOMO" in Path(path).name or "/OMOMO" in path for path in resolved)
    assert motion_label(resolved) == "CMU_OMOMO"
    assert resolve_motion_file("11_01_stageii").endswith("11_01_stageii.npz")


def test_infer_motion_files_from_checkpoint_prefers_checkpoint_metadata():
    motion_files = infer_motion_files_from_checkpoint(
        "foo/bar/model.pth",
        "vanila",
        {"motion_files": ["env/assests/02_04_stageii.npz"]},
    )
    assert len(motion_files) == 1
    assert motion_files[0].endswith("02_04_stageii.npz")


def test_resolve_motion_files_expands_directories_recursively(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    first_dir = dataset_dir / "01"
    second_dir = dataset_dir / "02"
    first_dir.mkdir(parents=True)
    second_dir.mkdir()

    first_file = first_dir / "01_01_stageii.npz"
    second_file = second_dir / "02_03_stageii.npz"
    first_file.touch()
    second_file.touch()
    (second_dir / "ignore.txt").write_text("ignore", encoding="utf-8")

    resolved = resolve_motion_files([str(dataset_dir)])

    assert resolved == [str(first_file.resolve()), str(second_file.resolve())]
