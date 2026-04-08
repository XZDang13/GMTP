import types

import torch

import gmtp.runtime.motion_mae_pretrain as motion_mae_pretrain
from gmtp.runtime.motion_mae_pretrain import MotionMAEPretrainRunner, _format_batch_loss_log, _format_epoch_metrics_log


def test_format_batch_loss_log_uses_available_slice_losses_only():
    losses = {
        "loss": torch.tensor(1.25),
        "root_loss": torch.tensor(0.25),
        "joint_pos_loss": torch.tensor(0.4),
        "joint_vel_loss": torch.tensor(0.6),
        "joint_pos_error": torch.tensor(0.4),
        "joint_vel_error": torch.tensor(0.6),
    }

    formatted = _format_batch_loss_log(losses, loss_names=("root", "joint", "end_effector"))

    assert formatted == (
        "loss=1.250000 root=0.250000 joint_pos=0.400000 joint_vel=0.600000 "
        "joint_pos_error=0.400000 joint_vel_error=0.600000"
    )


def test_progress_bar_is_tty_aware_and_sets_expected_kwargs(monkeypatch):
    captured: dict[str, object] = {}

    class DummyTqdm:
        def __init__(self, iterable, **kwargs):
            captured["iterable"] = list(iterable)
            captured["kwargs"] = kwargs

        @staticmethod
        def write(message: str) -> None:
            captured["message"] = message

    monkeypatch.setattr(motion_mae_pretrain, "tqdm", DummyTqdm)
    monkeypatch.setattr(motion_mae_pretrain.sys.stderr, "isatty", lambda: False)

    motion_mae_pretrain._progress_bar(range(3), total=3, desc="epochs", leave=False)

    assert captured["iterable"] == [0, 1, 2]
    assert captured["kwargs"] == {
        "total": 3,
        "desc": "epochs",
        "leave": False,
        "dynamic_ncols": True,
        "disable": True,
    }


def test_format_epoch_metrics_log_includes_individual_loss_terms():
    metrics = {
        "loss": 1.25,
        "reconstruction_loss": 1.25,
        "root_loss": 0.25,
        "root_weighted_loss": 0.5,
        "joint_pos_loss": 0.4,
        "joint_pos_weighted_loss": 0.3,
        "joint_vel_loss": 0.6,
        "joint_vel_weighted_loss": 0.45,
        "joint_pos_error": 0.4,
        "joint_vel_error": 0.6,
    }

    formatted = _format_epoch_metrics_log(metrics, prefix="train", loss_names=("root", "joint"))

    assert formatted == (
        "train_loss=1.250000 "
        "train_reconstruction_loss=1.250000 "
        "train_root_loss=0.250000 "
        "train_root_weighted_loss=0.500000 "
        "train_joint_pos_loss=0.400000 "
        "train_joint_pos_weighted_loss=0.300000 "
        "train_joint_vel_loss=0.600000 "
        "train_joint_vel_weighted_loss=0.450000 "
        "train_joint_pos_error=0.400000 "
        "train_joint_vel_error=0.600000"
    )


def test_build_summary_includes_best_and_epoch_loss_history():
    fake_runner = types.SimpleNamespace(
        motion_files=["motion_a.npz"],
        motion_name="motion_a",
        data_bundle=types.SimpleNamespace(
            train_motion_names=("motion_a",),
            val_motion_names=("motion_b",),
            train_window_count=12,
            val_window_count=3,
        ),
        schema=types.SimpleNamespace(to_dict=lambda: {"d_ref": 7, "d_target": 7}),
        run_paths=types.SimpleNamespace(root="/tmp/fake-run"),
    )

    summary = MotionMAEPretrainRunner._build_summary(
        fake_runner,
        best_epoch=2,
        best_metric=0.125,
        best_train_metrics={"loss": 0.2, "root_loss": 0.05},
        best_val_metrics={"loss": 0.125, "root_loss": 0.025},
        final_train_metrics={"loss": 0.15, "root_loss": 0.04},
        final_val_metrics={"loss": 0.13, "root_loss": 0.03},
        epoch_history=[
            {
                "epoch": 1,
                "is_best": False,
                "best_metric_so_far": 0.2,
                "train_metrics": {"loss": 0.3, "root_loss": 0.1},
                "val_metrics": {"loss": 0.2, "root_loss": 0.08},
            },
            {
                "epoch": 2,
                "is_best": True,
                "best_metric_so_far": 0.125,
                "train_metrics": {"loss": 0.2, "root_loss": 0.05},
                "val_metrics": {"loss": 0.125, "root_loss": 0.025},
            },
        ],
        best_paths={"best_motion_mae": "/tmp/best.pth"},
        final_paths={"final_motion_mae": "/tmp/final.pth"},
        completed_epochs=2,
        status="completed",
    )

    assert summary["status"] == "completed"
    assert summary["completed_epochs"] == 2
    assert summary["best_epoch"] == 2
    assert summary["best_train_metrics"]["root_loss"] == 0.05
    assert summary["best_val_metrics"]["root_loss"] == 0.025
    assert len(summary["epoch_history"]) == 2
    assert summary["epoch_history"][1]["is_best"] is True
