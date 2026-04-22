from pathlib import Path

from gmtp.integrations.ref2act.motion import (
    DEFAULT_EXPERIMENT_MOTION_FILES,
    infer_motion_files_from_checkpoint,
    motion_label,
    motion_names,
    normalize_motion_files,
    resolve_motion_file,
    resolve_motion_files,
)


def test_default_experiment_motion_files_use_shipped_anchor_assets():
    expected = tuple(f"env/assests/{path.name}" for path in sorted(Path("env/assests").glob("*_anchor.npz")))

    assert DEFAULT_EXPERIMENT_MOTION_FILES == expected
    assert expected
    assert all(path.endswith("_anchor.npz") for path in DEFAULT_EXPERIMENT_MOTION_FILES)


def test_motion_helpers_normalize_resolve_and_label():
    normalized = normalize_motion_files(["jump_anchor", "env/assests/walk_anchor.npz"])
    assert normalized == ["jump_anchor", "env/assests/walk_anchor.npz"]

    resolved = resolve_motion_files(normalized)
    assert all(Path(path).exists() for path in resolved)
    assert motion_names(normalized) == ["jump_anchor", "walk_anchor"]
    assert motion_label(normalized) == "jump_anchor_walk_anchor"
    assert resolve_motion_file("jump_anchor").endswith("jump_anchor.npz")


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
