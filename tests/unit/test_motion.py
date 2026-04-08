from pathlib import Path

from gmtp.integrations.ref2act.motion import (
    infer_motion_files_from_checkpoint,
    motion_label,
    motion_names,
    normalize_motion_files,
    resolve_motion_file,
    resolve_motion_files,
)


def test_motion_helpers_normalize_resolve_and_label():
    normalized = normalize_motion_files(["115_06_stageii", "env/assests/120_01_stageii.npz"])
    assert normalized == ["115_06_stageii", "env/assests/120_01_stageii.npz"]

    resolved = resolve_motion_files(normalized)
    assert all(Path(path).exists() for path in resolved)
    assert motion_names(normalized) == ["115_06_stageii", "120_01_stageii"]
    assert motion_label(normalized) == "115_06_stageii_120_01_stageii"
    assert resolve_motion_file("115_06_stageii").endswith("115_06_stageii.npz")


def test_infer_motion_files_from_checkpoint_prefers_checkpoint_metadata():
    motion_files = infer_motion_files_from_checkpoint(
        "foo/bar/model.pth",
        "vanila",
        {"motion_files": ["env/assests/115_06_stageii.npz"]},
    )
    assert len(motion_files) == 1
    assert motion_files[0].endswith("115_06_stageii.npz")


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
