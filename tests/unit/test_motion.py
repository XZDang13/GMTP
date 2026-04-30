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
    expected = tuple(str((MOCAP_DATA_DIR / name).resolve()) for name in DEFAULT_MOCAP_DATASET_NAMES)

    assert DEFAULT_EXPERIMENT_MOTION_FILES == expected
    assert {Path(path).name for path in DEFAULT_EXPERIMENT_MOTION_FILES} == {"CMU", "OMOMO"}
    assert all(Path(path).is_dir() for path in DEFAULT_EXPERIMENT_MOTION_FILES)


def test_motion_helpers_normalize_resolve_and_label():
    normalized = normalize_motion_files(["jump_anchor", "env/assests/walk_anchor.npz"])
    assert normalized == ["jump_anchor", "env/assests/walk_anchor.npz"]

    resolved = resolve_motion_files(normalized)
    assert all(Path(path).exists() for path in resolved)
    assert motion_names(normalized) == ["jump_anchor", "walk_anchor"]
    assert motion_label(normalized) == "jump_anchor_walk_anchor"
    assert resolve_motion_file("jump_anchor").endswith("jump_anchor.npz")


def test_motion_helpers_resolve_mocap_dataset_aliases_and_motion_names():
    resolved = resolve_motion_files(["CMU", "OMOMO"])

    assert any("/CMU/" in path for path in resolved)
    assert any("/OMOMO/" in path for path in resolved)
    assert motion_label(resolved) == "CMU_OMOMO"
    assert resolve_motion_file("11_01_stageii").endswith("CMU/11/11_01_stageii.npz")


def test_infer_motion_files_from_checkpoint_prefers_checkpoint_metadata():
    motion_files = infer_motion_files_from_checkpoint(
        "foo/bar/model.pth",
        "vanila",
        {"motion_files": ["env/assests/jump_anchor.npz"]},
    )
    assert len(motion_files) == 1
    assert motion_files[0].endswith("jump_anchor.npz")


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
