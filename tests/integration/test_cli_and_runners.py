import json
import sys
import types
from pathlib import Path

import pytest
import torch

from gmtp.cli.main import build_parser
from gmtp.integrations.ref2act.observation_history import (
    build_motion_policy_window_lengths,
    build_robot_policy_window_lengths,
)
from gmtp.models import Critic, FiLMResActor
from gmtp.motion_mae import (
    FeatureSliceSpec,
    MotionFeatureSchema,
    MotionMAEDataConfig,
    MotionMAEFeatureConfig,
    MotionMAEModelConfig,
    MotionMAEPretrainConfig,
    ReferenceMotionMAE,
    build_motion_mae_encoder_checkpoint,
    save_motion_mae_encoder_checkpoint,
)
from gmtp.runtime.checkpoints import build_training_checkpoint, load_checkpoint_v2, save_checkpoint_v2
from gmtp.runtime.config import IsaacEvalConfig, RunConfig, Sim2SimEvalConfig
from gmtp.runtime.eval_isaac import IsaacEvalRunner
from gmtp.runtime.train_runner import TrainRunner
from gmtp.models.motion_encoder import build_motion_window_layout

DEFAULT_TEST_MOTION_FILE = "env/assests/jump_anchor.npz"


def _motion_obs_dim(motion_window_length: int = 1) -> int:
    return build_motion_window_layout(2, motion_window_length).motion_obs_dim


def _motion_step_dim(motion_window_length: int = 1) -> int:
    return build_motion_window_layout(2, motion_window_length).motion_step_dim


class _DummyTrainEnv:
    def __init__(self, batch_size: int, *, robot_window_length: int = 4, motion_window_length: int = 1):
        self.unwrapped = self
        self.device = "cpu"
        self.batch_size = batch_size
        self.robot_window_length = int(robot_window_length)
        self.motion_window_length = int(motion_window_length)
        self.motion_obs_dim = _motion_obs_dim(self.motion_window_length)
        self.robot_obs_dim = 12 * self.robot_window_length
        self.motion_lib = types.SimpleNamespace(body_names=("pelvis", "left_hand", "right_hand"))
        self.extras = {}
        self.step_count = 0
        self.reference_motion = self._build_reference_motion()

    def _build_reference_motion(self):
        return types.SimpleNamespace(
            joint_pos=torch.tensor([[0.1, 0.2]], dtype=torch.float32).repeat(self.batch_size, 1),
            joint_vel=torch.tensor([[0.3, 0.4]], dtype=torch.float32).repeat(self.batch_size, 1),
            body_positions=torch.tensor(
                [[[0.0, 0.0, 0.0], [0.2, 0.0, 0.1], [0.0, 0.3, 0.2]]],
                dtype=torch.float32,
            ).repeat(self.batch_size, 1, 1),
            body_quaternions=torch.tensor(
                [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]],
                dtype=torch.float32,
            ).repeat(self.batch_size, 1, 1),
            anchor_body_index=0,
        )

    def reset(self):
        self.step_count = 0
        self.reference_motion = self._build_reference_motion()
        obs = {
            "motion": torch.zeros(self.batch_size, self.motion_obs_dim),
            "robot": torch.zeros(self.batch_size, self.robot_obs_dim),
            "privilege": torch.zeros(self.batch_size, 5),
        }
        return obs, {}

    def step(self, action):
        self.step_count += 1
        self.reference_motion = self._build_reference_motion()
        obs = {
            "motion": torch.full((self.batch_size, self.motion_obs_dim), float(self.step_count)),
            "robot": torch.full((self.batch_size, self.robot_obs_dim), float(self.step_count)),
            "privilege": torch.zeros(self.batch_size, 5),
        }
        reward = torch.ones(self.batch_size)
        terminate = torch.zeros(self.batch_size, dtype=torch.bool)
        timeout = torch.zeros(self.batch_size, dtype=torch.bool)
        return obs, reward, terminate, timeout, {}

    def get_joint_params(self):
        return {
            "joint_names": ["j0", "j1"],
            "joint_effort_limits": torch.ones(2),
            "joint_pos_limits": torch.tensor([[-1.0, 1.0], [-1.0, 1.0]]),
            "joint_stiffness": torch.ones(2),
            "joint_damping": torch.full((2,), 0.1),
            "action_offset": torch.zeros(2),
            "action_scale": torch.ones(2),
            "action_mode": "offset",
        }

    def close(self):
        return None


def _recovery_metrics_for_step(step_count: int) -> dict[str, float]:
    if step_count == 1:
        return {
            "active_rate": 0.2,
            "entry_rate": 0.1,
            "exit_rate": 0.05,
            "timeout_rate": 0.0,
            "reference_time_scale_mean": 0.9,
            "score_mean": 1.5,
        }
    return {
        "active_rate": 0.4,
        "entry_rate": 0.3,
        "exit_rate": 0.15,
        "timeout_rate": 0.05,
        "reference_time_scale_mean": 0.8,
        "score_mean": 1.0,
    }


class _RecoveryInfoTrainEnv(_DummyTrainEnv):
    def step(self, action):
        obs, reward, terminate, timeout, _ = super().step(action)
        metrics = _recovery_metrics_for_step(self.step_count)
        info = {
            "env": {
                "fall_recovery": {
                    "active_rate": torch.tensor([metrics["active_rate"]], dtype=torch.float32),
                    "entry_rate": torch.tensor(metrics["entry_rate"], dtype=torch.float32),
                    "exit_rate": torch.tensor(metrics["exit_rate"], dtype=torch.float32),
                    "timeout_rate": torch.tensor(metrics["timeout_rate"], dtype=torch.float32),
                    "reference_time_scale_mean": torch.tensor(metrics["reference_time_scale_mean"], dtype=torch.float32),
                },
                "tracking_quality": {
                    "score_mean": torch.tensor(metrics["score_mean"], dtype=torch.float32),
                },
            }
        }
        return obs, reward, terminate, timeout, info


class _RecoveryExtrasTrainEnv(_DummyTrainEnv):
    def step(self, action):
        obs, reward, terminate, timeout, _ = super().step(action)
        metrics = _recovery_metrics_for_step(self.step_count)
        self.extras = {
            "fall_recovery/active_rate": torch.tensor(metrics["active_rate"], dtype=torch.float32),
            "fall_recovery/entry_rate": torch.tensor(metrics["entry_rate"], dtype=torch.float32),
            "fall_recovery/exit_rate": torch.tensor(metrics["exit_rate"], dtype=torch.float32),
            "fall_recovery/timeout_rate": torch.tensor(metrics["timeout_rate"], dtype=torch.float32),
            "fall_recovery/reference_time_scale_mean": torch.tensor(metrics["reference_time_scale_mean"], dtype=torch.float32),
            "tracking_quality/score_mean": torch.tensor(metrics["score_mean"], dtype=torch.float32),
        }
        return obs, reward, terminate, timeout, {}


def _window_length(window_lengths, prefix: str, default: int) -> int:
    if not window_lengths:
        return default
    return int(next((value for key, value in window_lengths.items() if str(key).startswith(prefix)), default))


def _fake_train_module(env_cls=_DummyTrainEnv):
    def make_training_env(
        window_lengths=None,
        motion_files=None,
    ):
        cfg = types.SimpleNamespace(
            scene=types.SimpleNamespace(num_envs=2),
            action_space=2,
            expert_motion_file=list(motion_files or [DEFAULT_TEST_MOTION_FILE]),
            action=types.SimpleNamespace(mode="offset"),
            action_mod="Offset",
            root_link_name="torso_link",
            anchor_body_name="torso_link",
            segment_source="Anchor",
            sampling_strategy="FailureWeighted",
            failure_temperature=1.0,
        )
        return (
            env_cls(
                batch_size=2,
                robot_window_length=_window_length(window_lengths, "projected_gravity", 4),
                motion_window_length=_window_length(window_lengths, "target_projected_gravity", 1),
            ),
            cfg,
        )

    return types.SimpleNamespace(
        make_training_env=make_training_env,
    )


def _fake_eval_module():
    cfg = types.SimpleNamespace(scene=types.SimpleNamespace(num_envs=1), action_space=2)
    return types.SimpleNamespace(
        make_eval_env=lambda motion_files, show_reference_motion=False, window_lengths=None, render_mode=None: (
            _DummyTrainEnv(
                batch_size=1,
                robot_window_length=_window_length(window_lengths, "projected_gravity", 1),
                motion_window_length=_window_length(window_lengths, "target_projected_gravity", 1),
            ),
            cfg,
        )
    )


def _write_checkpoint(
    tmp_path,
    *,
    motion_files: list[str] | None = None,
    robot_window_length: int = 1,
    motion_window_length: int = 1,
    motion_encoder_type: str = "transformer",
    motion_obs_dim: int | None = None,
    motion_mae_encoder_checkpoint: str | None = None,
):
    motion_obs_dim = motion_obs_dim if motion_obs_dim is not None else _motion_obs_dim(motion_window_length)
    robot_obs_dim = 12 * robot_window_length
    actor = FiLMResActor(
        robot_obs_dim=robot_obs_dim,
        motion_obs_dim=motion_obs_dim,
        action_dim=2,
        num_blocks=4,
        robot_window_length=robot_window_length,
        motion_window_length=motion_window_length,
        motion_encoder_type=motion_encoder_type,
        motion_mae_encoder_checkpoint=motion_mae_encoder_checkpoint,
    )
    critic = Critic(obs_dim=5)
    observation_window_lengths = {}
    if robot_window_length > 1:
        observation_window_lengths.update(build_robot_policy_window_lengths(robot_window_length))
    if motion_window_length > 1:
        observation_window_lengths.update(build_motion_policy_window_lengths(motion_window_length))
    checkpoint = build_training_checkpoint(
        actor=actor,
        critic=critic,
        motion_files=motion_files or [DEFAULT_TEST_MOTION_FILE],
        joint_params={
            "joint_names": ["j0", "j1"],
            "joint_effort_limits": torch.ones(2),
            "joint_pos_limits": torch.tensor([[-1.0, 1.0], [-1.0, 1.0]]),
            "joint_stiffness": torch.ones(2),
            "joint_damping": torch.full((2,), 0.1),
            "action_offset": torch.zeros(2),
            "action_scale": torch.ones(2),
        },
        action_mode="offset",
        root_name="torso_link",
        anchor_body_name="torso_link",
        segment_source="anchor",
        sampling_strategy="failure_weighted",
        motion_mae_encoder_checkpoint=motion_mae_encoder_checkpoint,
        observation_window_lengths=observation_window_lengths or None,
    )
    return save_checkpoint_v2(checkpoint, tmp_path / "model_v2.pth")


def _write_isaac_checkpoint(tmp_path):
    return _write_checkpoint(tmp_path)


def _write_sim2sim_checkpoint(tmp_path):
    return _write_checkpoint(tmp_path)


def _motion_mae_schema() -> MotionFeatureSchema:
    motion_step_dim = _motion_step_dim()
    return MotionFeatureSchema(
        d_ref=motion_step_dim,
        d_target=motion_step_dim,
        full_feature_dim=motion_step_dim,
        base_slices=(
            FeatureSliceSpec("root", 0, 3),
            FeatureSliceSpec("joint", 3, motion_step_dim),
        ),
        reference_slices=(
            FeatureSliceSpec("root", 0, 3),
            FeatureSliceSpec("joint", 3, motion_step_dim),
        ),
        target_slices=(
            FeatureSliceSpec("root", 0, 3),
            FeatureSliceSpec("joint", 3, motion_step_dim),
        ),
        policy_motion_slice=FeatureSliceSpec("policy_motion", 0, motion_step_dim),
        anchor_body_name="pelvis",
        end_effector_body_names=("left_hand", "right_hand"),
        reference_feature_names=("root", "joint"),
        target_feature_names=("root", "joint"),
        policy_feature_names=("root", "joint"),
        joint_names=("j0", "j1"),
        body_names=("pelvis", "left_hand", "right_hand"),
        reference_mean=tuple(0.0 for _ in range(motion_step_dim)),
        reference_std=tuple(1.0 for _ in range(motion_step_dim)),
        target_mean=tuple(0.0 for _ in range(motion_step_dim)),
        target_std=tuple(1.0 for _ in range(motion_step_dim)),
    )


def _write_motion_mae_encoder_checkpoint(tmp_path) -> Path:
    model = ReferenceMotionMAE(
        input_dim=_motion_step_dim(),
        target_dim=_motion_step_dim(),
        past_frames=4,
        future_frames=2,
        latent_dim=6,
        d_model=16,
        encoder_layers=2,
        decoder_layers=1,
        nhead=4,
        dim_feedforward=32,
    )
    checkpoint = build_motion_mae_encoder_checkpoint(
        model=model,
        schema=_motion_mae_schema(),
        config=MotionMAEPretrainConfig(
            data=MotionMAEDataConfig(
                motion_files=(DEFAULT_TEST_MOTION_FILE,),
                past_frames=4,
                future_frames=2,
                split_mode="by_window",
                val_ratio=0.5,
            ),
            feature=MotionMAEFeatureConfig(
                reference_feature_names=("root", "joint"),
                target_feature_names=("root", "joint"),
                policy_feature_names=("root", "joint"),
                end_effector_body_names=("left_hand", "right_hand"),
            ),
            model=MotionMAEModelConfig(
                d_model=16,
                latent_dim=6,
                encoder_layers=2,
                decoder_layers=1,
                nhead=4,
                dim_feedforward=32,
            ),
        ),
        epoch=1,
        best_metric=0.7,
        artifacts={"run_dir": "runs/pretrain/demo"},
    )
    return save_motion_mae_encoder_checkpoint(checkpoint, tmp_path / "motion_mae_encoder.pth")


def test_cli_parser_builds_train_and_eval_commands():
    parser = build_parser()
    args = parser.parse_args(["train", "--num-updates", "5"])
    assert args.command == "train"
    assert args.num_updates == 5
    assert args.num_blocks == 4
    assert args.robot_window_length == 4
    assert args.robot_encoder_type == "transformer"
    assert args.motion_window_length == 1
    assert args.motion_encoder_type == "transformer"
    assert args.actor_fusion_type == "film"
    assert args.disable_amp is False
    assert args.disable_wandb is False
    assert args.motion_mae_encoder_checkpoint is None
    assert args.motion_files is None
    assert args.resume_checkpoint_path is None
    assert args.anchor_log_interval == 100
    assert args.anchor_heatmap_bins == 128

    with pytest.raises(SystemExit):
        parser.parse_args(["train", "--attn-block-size", "5"])

    args = parser.parse_args(["train", "--disable-amp"])
    assert args.disable_amp is True

    args = parser.parse_args(["train", "--anchor-log-interval", "25", "--anchor-heatmap-bins", "64"])
    assert args.anchor_log_interval == 25
    assert args.anchor_heatmap_bins == 64

    args = parser.parse_args(["train", "--actor-fusion-type", "motion_residual"])
    assert args.actor_fusion_type == "motion_residual"

    args = parser.parse_args(["train", "--motion-files", "CMU/11/11_01_stageii.npz", "env/assests/jump_anchor.npz"])
    assert args.motion_files == ["CMU/11/11_01_stageii.npz", "env/assests/jump_anchor.npz"]

    args = parser.parse_args(["train", "--resume-checkpoint", "path/to/model_v2.pth"])
    assert args.resume_checkpoint_path == "path/to/model_v2.pth"

    args = parser.parse_args(["eval", "isaac", "--checkpoint", "foo.pth", "--disable-amp"])
    assert args.command == "eval"
    assert args.eval_target == "isaac"
    assert args.disable_amp is True
    assert args.motion_window_length is None
    assert args.motion_encoder_type is None
    assert args.motion_mae_encoder_checkpoint is None

    args = parser.parse_args(
        [
            "eval",
            "sim2sim",
            "--checkpoint",
            "foo.pth",
            "--motion-files",
            "foo",
            "bar",
            "--action-mode",
            "residual",
            "--num-steps",
            "12",
            "--disable-amp",
        ]
    )
    assert args.command == "eval"
    assert args.eval_target == "sim2sim"
    assert args.motion_files == ["foo", "bar"]
    assert args.action_mode == "residual"
    assert args.num_steps == 12
    assert args.robot_window_length is None
    assert args.motion_window_length is None
    assert args.motion_encoder_type is None
    assert args.disable_amp is True
    assert args.save_video is False
    assert not hasattr(args, "attn_block_size")

    args = parser.parse_args(["eval", "sim2sim", "--checkpoint", "foo.pth", "--save-video"])
    assert args.save_video is True

    args = parser.parse_args(["eval", "sim2sim", "--checkpoint", "foo.pth", "--allow-unstable-init"])
    assert args.allow_unstable_init is True

    args = parser.parse_args(["pretrain", "motion-mae", "--config", "foo.json"])
    assert args.command == "pretrain"
    assert args.pretrain_target == "motion-mae"
    assert args.config == "foo.json"

    args = parser.parse_args(["pretrain", "motion-mae-latents", "--checkpoint", "encoder.pth", "--config", "foo.json"])
    assert args.command == "pretrain"
    assert args.pretrain_target == "motion-mae-latents"
    assert args.checkpoint == "encoder.pth"

    args = parser.parse_args(
        [
            "pretrain",
            "motion-mae-visualize",
            "--checkpoint",
            "motion_mae.pth",
            "--config",
            "foo.json",
            "--split",
            "train",
            "--motion-name",
            "demo",
            "--sample-index",
            "3",
            "--whole-motion",
            "--future-frame-index",
            "1",
            "--fps",
            "50",
        ]
    )
    assert args.command == "pretrain"
    assert args.pretrain_target == "motion-mae-visualize"
    assert args.checkpoint == "motion_mae.pth"
    assert args.config == "foo.json"
    assert args.split == "train"
    assert args.motion_name == "demo"
    assert args.sample_index == 3
    assert args.whole_motion is True
    assert args.future_frame_index == 1
    assert args.fps == 50


def test_cli_parser_rejects_removed_migrate_checkpoint_command():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["migrate-checkpoint", "--checkpoint", "legacy.pth"])


def test_train_runner_dry_construction(monkeypatch):
    monkeypatch.setitem(sys.modules, "gmtp.integrations.ref2act.isaac_env", _fake_train_module())
    runner = TrainRunner(RunConfig(use_wandb=False))
    assert runner.actor_type.value == "film_res"
    assert runner.requested_amp is True
    assert runner.use_amp is False
    assert runner.observation_window_lengths == {
        **build_robot_policy_window_lengths(4),
        **build_motion_policy_window_lengths(1),
    }
    assert runner.motion_files[0].endswith("jump_anchor.npz")
    config_payload = json.loads(runner.run_paths.config_path.read_text(encoding="utf-8"))
    assert config_payload["config"]["use_amp"] is True
    runner.env.close()


def test_train_runner_passes_motion_file_override_to_training_env(monkeypatch):
    monkeypatch.setitem(sys.modules, "gmtp.integrations.ref2act.isaac_env", _fake_train_module())
    motion_files = ["env/assests/walk_anchor.npz"]

    runner = TrainRunner(RunConfig(use_wandb=False, motion_files=motion_files))

    assert runner.motion_files == motion_files
    config_payload = json.loads(runner.run_paths.config_path.read_text(encoding="utf-8"))
    assert config_payload["config"]["motion_files"] == motion_files
    runner.env.close()


def test_train_runner_constructs_film_res_actor(monkeypatch):
    monkeypatch.setitem(sys.modules, "gmtp.integrations.ref2act.isaac_env", _fake_train_module())
    runner = TrainRunner(
        RunConfig(
            num_blocks=4,
            robot_encoder_type="cnn",
            actor_fusion_type="motion_residual",
            use_wandb=False,
        )
    )
    assert runner.actor_type.value == "film_res"
    assert runner.actor_kwargs == {
        "num_blocks": 4,
        "robot_window_length": 4,
        "robot_encoder_type": "cnn",
        "motion_window_length": 1,
        "motion_encoder_type": "mlp",
        "actor_fusion_type": "motion_residual",
    }
    assert runner.observation_window_lengths == {
        **build_robot_policy_window_lengths(4),
        **build_motion_policy_window_lengths(1),
    }
    assert isinstance(runner.actor, FiLMResActor)
    runner.env.close()


def test_train_runner_rollout_updates_actor_statistics(monkeypatch):
    monkeypatch.setitem(sys.modules, "gmtp.integrations.ref2act.isaac_env", _fake_train_module())
    runner = TrainRunner(
        RunConfig(
            use_wandb=False,
            rollout_steps=1,
            num_updates=1,
        )
    )

    returned_obs = runner.rollout(runner.initial_obs)

    assert returned_obs["motion"].shape == (2, _motion_step_dim())
    assert returned_obs["robot"].shape == (2, 4, 12)
    assert runner.rollout_buffer.data["motion_observations"].shape[-1] == _motion_step_dim()
    assert runner.rollout_buffer.data["robot_observations"].shape[-2:] == (4, 12)
    runner.env.close()


def test_train_runner_logs_rollout_recovery_metrics_from_info(monkeypatch):
    monkeypatch.setitem(sys.modules, "gmtp.integrations.ref2act.isaac_env", _fake_train_module(_RecoveryInfoTrainEnv))
    runner = TrainRunner(
        RunConfig(
            use_wandb=False,
            rollout_steps=2,
            num_updates=1,
        )
    )
    logged_payloads = []
    runner._log_metrics = lambda payload: logged_payloads.append(dict(payload))

    runner.rollout(runner.initial_obs)

    assert len(logged_payloads) == 1
    assert logged_payloads[0] == pytest.approx(
        {
            "recovery/active_rate": 0.3,
            "recovery/entry_rate": 0.2,
            "recovery/exit_rate": 0.1,
            "recovery/timeout_rate": 0.025,
            "recovery/reference_time_scale_mean": 0.85,
            "recovery/tracking_score_mean": 1.25,
            "recovery/exit_to_entry_ratio": 0.5,
            "recovery/timeout_to_entry_ratio": 0.125,
        }
    )
    runner.env.close()


def test_train_runner_logs_rollout_recovery_metrics_from_extras_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "gmtp.integrations.ref2act.isaac_env", _fake_train_module(_RecoveryExtrasTrainEnv))
    runner = TrainRunner(
        RunConfig(
            use_wandb=False,
            rollout_steps=2,
            num_updates=1,
        )
    )
    logged_payloads = []
    runner._log_metrics = lambda payload: logged_payloads.append(dict(payload))

    runner.rollout(runner.initial_obs)

    assert len(logged_payloads) == 1
    assert logged_payloads[0]["recovery/active_rate"] == pytest.approx(0.3)
    assert logged_payloads[0]["recovery/exit_to_entry_ratio"] == pytest.approx(0.5)
    runner.env.close()


def test_train_runner_skips_recovery_metric_log_when_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "gmtp.integrations.ref2act.isaac_env", _fake_train_module())
    runner = TrainRunner(
        RunConfig(
            use_wandb=False,
            rollout_steps=1,
            num_updates=1,
        )
    )
    logged_payloads = []
    runner._log_metrics = lambda payload: logged_payloads.append(dict(payload))

    runner.rollout(runner.initial_obs)

    assert logged_payloads == []
    runner.env.close()


def test_train_runner_supports_actor_integrated_motion_mae(tmp_path, monkeypatch):
    motion_mae_checkpoint = _write_motion_mae_encoder_checkpoint(tmp_path)
    monkeypatch.setitem(sys.modules, "gmtp.integrations.ref2act.isaac_env", _fake_train_module())
    runner = TrainRunner(
        RunConfig(
            use_wandb=False,
            rollout_steps=1,
            num_updates=1,
            motion_window_length=4,
            motion_encoder_type="mae",
            motion_mae_encoder_checkpoint=str(motion_mae_checkpoint),
        )
    )

    assert runner.motion_mae_encoder_checkpoint == str(motion_mae_checkpoint.resolve())
    assert runner.obs_dims["motion"] == _motion_obs_dim(4)
    returned_obs = runner.rollout(runner.initial_obs)
    assert returned_obs["motion"].shape == (2, 4, _motion_step_dim())
    assert runner.rollout_buffer.data["motion_observations"].shape[-2:] == (4, _motion_step_dim())
    runner.env.close()


def test_train_runner_update_smoke_uses_cpu_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "gmtp.integrations.ref2act.isaac_env", _fake_train_module())
    runner = TrainRunner(
        RunConfig(
            use_wandb=False,
            rollout_steps=1,
            num_updates=1,
        )
    )

    runner.rollout(runner.initial_obs)
    runner.update()

    assert runner.requested_amp is True
    assert runner.use_amp is False
    assert hasattr(runner, "actor_optimizer")
    assert hasattr(runner, "critic_optimizer")
    assert runner.actor_optimizer is not runner.critic_optimizer
    assert runner.lr_scheduler.optimizer is runner.actor_optimizer
    runner.env.close()


def test_train_runner_warm_starts_from_checkpoint_without_trainer_state(tmp_path, monkeypatch, capsys):
    checkpoint_path = _write_checkpoint(tmp_path, robot_window_length=1)
    checkpoint = load_checkpoint_v2(checkpoint_path)
    monkeypatch.setitem(sys.modules, "gmtp.integrations.ref2act.isaac_env", _fake_train_module())

    runner = TrainRunner(
        RunConfig(
            use_wandb=False,
            resume_checkpoint_path=str(checkpoint_path),
            output_root=str(tmp_path / "runs"),
        )
    )
    output = capsys.readouterr().out

    assert "warm-starting actor/critic" in output
    assert runner.resume_mode == "warm_start"
    assert runner.resume_trainer_state_restored is False
    assert runner.update_count == 0
    assert runner.global_step == 0
    assert runner.actor_kwargs == checkpoint.actor_kwargs
    assert runner.motion_files == checkpoint.motion_files
    for key, value in checkpoint.model["actor"].items():
        assert torch.equal(runner.actor.state_dict()[key].detach().cpu(), value)
    for key, value in checkpoint.model["critic"].items():
        assert torch.equal(runner.critic.state_dict()[key].detach().cpu(), value)
    runner.env.close()


def test_train_runner_restores_full_trainer_state_and_runs_additional_updates(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "gmtp.integrations.ref2act.isaac_env", _fake_train_module())
    source_runner = TrainRunner(
        RunConfig(
            use_wandb=False,
            rollout_steps=1,
            num_updates=1,
            output_root=str(tmp_path / "source_runs"),
        )
    )
    source_runner.rollout(source_runner.initial_obs)
    source_runner.update()
    checkpoint_path = source_runner.save_checkpoint(str(source_runner.update_count))
    source_runner.env.close()
    checkpoint = load_checkpoint_v2(checkpoint_path)

    runner = TrainRunner(
        RunConfig(
            use_wandb=False,
            rollout_steps=1,
            num_updates=2,
            checkpoint_interval=1000,
            resume_checkpoint_path=str(checkpoint_path),
            output_root=str(tmp_path / "resume_runs"),
        )
    )

    assert runner.resume_mode == "full_state"
    assert runner.resume_trainer_state_restored is True
    assert runner.start_update_count == 1
    assert runner.start_global_step == 1
    assert runner.update_count == checkpoint.training["update_count"]
    assert runner.global_step == checkpoint.training["global_step"]
    assert runner.lr_scheduler.state_dict() == checkpoint.training["lr_scheduler"]
    assert runner.grad_scaler.state_dict() == checkpoint.training["grad_scaler"]

    restored_actor_optimizer = runner.actor_optimizer.state_dict()["optimizers"][0]["state"]
    checkpoint_actor_optimizer = checkpoint.training["actor_optimizer"]["optimizers"][0]["state"]
    assert restored_actor_optimizer.keys() == checkpoint_actor_optimizer.keys()
    first_state_key = next(iter(checkpoint_actor_optimizer))
    for state_name, checkpoint_state_value in checkpoint_actor_optimizer[first_state_key].items():
        if torch.is_tensor(checkpoint_state_value):
            assert torch.equal(
                restored_actor_optimizer[first_state_key][state_name].detach().cpu(),
                checkpoint_state_value.detach().cpu(),
            )
            break
    else:
        raise AssertionError("Expected at least one tensor optimizer state value.")

    summary = runner.train()
    final_checkpoint = load_checkpoint_v2(summary["final_checkpoint"])

    assert summary["resume_checkpoint"] == str(checkpoint_path.resolve())
    assert summary["resume_mode"] == "full_state"
    assert summary["resume_trainer_state_restored"] is True
    assert summary["num_updates"] == 2
    assert summary["start_update_count"] == 1
    assert summary["final_update_count"] == 3
    assert summary["start_global_step"] == 1
    assert summary["final_global_step"] == 3
    assert final_checkpoint.training["update_count"] == 3
    assert final_checkpoint.training["global_step"] == 3


def test_train_runner_logs_anchor_probability_summary_and_heatmap_every_configured_interval(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.setitem(sys.modules, "gmtp.integrations.ref2act.isaac_env", _fake_train_module())
    runner = TrainRunner(
        RunConfig(
            use_wandb=False,
            rollout_steps=1,
            num_updates=1,
            output_root=str(tmp_path / "runs"),
        )
    )
    runner._collect_anchor_reset_probabilities = lambda: [
        {"motion_name": "jump_anchor", "anchor_index": 0, "anchor_time": 0.0, "probability": 1.0}
    ]
    logged_payloads = []
    runner._log_metrics = logged_payloads.append

    capsys.readouterr()
    runner.rollout(runner.initial_obs)
    runner.update()
    assert "sampler snapshot after update" not in capsys.readouterr().out
    assert not any(
        "sampler/reset_distribution/anchors/jump_anchor/A000" in payload for payload in logged_payloads
    )

    runner.update_count = 99
    runner.rollout(runner.initial_obs)
    runner.update()
    output = capsys.readouterr().out

    assert "sampler snapshot after update 100" in output
    assert "max_p=1.000000" in output
    assert "reset anchors:" in output
    assert "reset motions:" in output
    assert "jump_anchor A0 t=0.000s p=1.000000" in output
    assert any(
        "sampler/reset_distribution/anchors/jump_anchor/A000" not in payload
        and payload.get("sampler/reset_distribution/anchors/max_probability") == 1.0
        and payload.get("sampler/reset_distribution/anchors/top1_mass") == 1.0
        for payload in logged_payloads
    )
    anchor_debug_dir = runner.run_paths.debug_dir / "anchor_reset_probabilities"
    assert (anchor_debug_dir / "latest_heatmap.png").exists()
    assert (anchor_debug_dir / "update_000100_heatmap.png").exists()
    assert (anchor_debug_dir / "update_000100.npz").exists()
    runner.env.close()


def test_train_runner_summary_and_checkpoint_record_anchor_sampler_metadata(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "gmtp.integrations.ref2act.isaac_env", _fake_train_module())
    runner = TrainRunner(
        RunConfig(
            use_wandb=False,
            rollout_steps=1,
            num_updates=1,
            checkpoint_interval=1000,
            output_root=str(tmp_path / "runs"),
        )
    )

    summary = runner.train()
    checkpoint = load_checkpoint_v2(summary["final_checkpoint"])

    assert summary["segment_source"] == "anchor"
    assert summary["sampling_strategy"] == "failure_weighted"
    assert checkpoint.env["segment_source"] == "anchor"
    assert checkpoint.env["sampling_strategy"] == "failure_weighted"
    assert checkpoint.motion_files[0].endswith("jump_anchor.npz")


def test_train_runner_uses_failure_weighted_sampling_from_start(monkeypatch):
    monkeypatch.setitem(sys.modules, "gmtp.integrations.ref2act.isaac_env", _fake_train_module())
    runner = TrainRunner(RunConfig(use_wandb=False))

    assert runner.sampling_strategy == "failure_weighted"
    assert runner.cfg.sampling_strategy == "FailureWeighted"
    runner.env.close()


def test_isaac_eval_runner_dry_construction(tmp_path, monkeypatch):
    checkpoint_path = _write_isaac_checkpoint(tmp_path)
    monkeypatch.setitem(sys.modules, "gmtp.integrations.ref2act.isaac_env", _fake_eval_module())
    runner = IsaacEvalRunner(
        IsaacEvalConfig(
            checkpoint_path=str(checkpoint_path),
            num_steps=1,
            progress_interval=0,
            output_root=str(tmp_path / "runs"),
        )
    )
    summary = runner.evaluate()

    assert runner.motion_files[0].endswith("jump_anchor.npz")
    assert summary["amp_requested"] is True
    assert summary["amp_enabled"] is False
    assert summary["amp_dtype"] == "float16"
    config_payload = json.loads(runner.run_paths.config_path.read_text(encoding="utf-8"))
    assert config_payload["config"]["use_amp"] is True


def test_isaac_eval_runner_supports_actor_integrated_motion_mae(tmp_path, monkeypatch):
    motion_mae_checkpoint = _write_motion_mae_encoder_checkpoint(tmp_path)
    checkpoint_path = _write_checkpoint(
        tmp_path,
        motion_window_length=4,
        motion_encoder_type="mae",
        motion_mae_encoder_checkpoint=str(motion_mae_checkpoint),
    )
    monkeypatch.setitem(sys.modules, "gmtp.integrations.ref2act.isaac_env", _fake_eval_module())
    runner = IsaacEvalRunner(
        IsaacEvalConfig(
            checkpoint_path=str(checkpoint_path),
            motion_window_length=4,
            motion_encoder_type="mae",
            motion_mae_encoder_checkpoint=str(motion_mae_checkpoint),
            num_steps=1,
            progress_interval=0,
            output_root=str(tmp_path / "runs-mae"),
        )
    )
    summary = runner.evaluate()

    assert summary["motion_mae_encoder_checkpoint"] == str(motion_mae_checkpoint.resolve())
    assert runner.obs_dims["motion"] == _motion_obs_dim(4)


@pytest.mark.skipif(pytest.importorskip("mujoco", reason="mujoco required") is None, reason="mujoco required")
def test_sim2sim_runner_accepts_v2_checkpoint(tmp_path):
    from gmtp.runtime.eval_sim2sim import Sim2SimEvalRunner

    checkpoint_path = _write_sim2sim_checkpoint(tmp_path)

    runner = Sim2SimEvalRunner(
        Sim2SimEvalConfig(
            checkpoint_path=str(checkpoint_path),
            num_steps=1,
            output_root=str(tmp_path / "runs"),
        )
    )

    assert runner.action_mode == "offset"
    assert runner.requested_amp is True
    assert runner.use_amp is False
    assert runner.checkpoint.motion_files[0].endswith("jump_anchor.npz")
