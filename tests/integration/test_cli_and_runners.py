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
from gmtp.runtime.checkpoints import build_training_checkpoint, save_checkpoint_v2
from gmtp.runtime.config import IsaacEvalConfig, RunConfig, Sim2SimEvalConfig
from gmtp.runtime.eval_isaac import IsaacEvalRunner
from gmtp.runtime.train_runner import TrainRunner


class _DummyTrainEnv:
    def __init__(self, batch_size: int, *, robot_window_length: int = 4, motion_window_length: int = 1):
        self.unwrapped = self
        self.device = "cpu"
        self.batch_size = batch_size
        self.robot_window_length = int(robot_window_length)
        self.motion_window_length = int(motion_window_length)
        self.motion_obs_dim = 7 * self.motion_window_length
        self.robot_obs_dim = 12 * self.robot_window_length
        self.motion_lib = types.SimpleNamespace(body_names=("pelvis", "left_hand", "right_hand"))
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


def _window_length(window_lengths, prefix: str, default: int) -> int:
    if not window_lengths:
        return default
    return int(next((value for key, value in window_lengths.items() if str(key).startswith(prefix)), default))


def _fake_train_module():
    cfg = types.SimpleNamespace(
        scene=types.SimpleNamespace(num_envs=2),
        action_space=2,
        expert_motion_file=["env/assests/115_06_stageii.npz"],
        action=types.SimpleNamespace(mode="offset"),
        action_mod="Offset",
        root_link_name="torso_link",
        anchor_body_name="torso_link",
    )
    return types.SimpleNamespace(
        make_training_env=lambda window_lengths=None: (
            _DummyTrainEnv(
                batch_size=2,
                robot_window_length=_window_length(window_lengths, "projected_gravity", 4),
                motion_window_length=_window_length(window_lengths, "target_projected_gravity", 1),
            ),
            cfg,
        )
    )


def _fake_eval_module():
    cfg = types.SimpleNamespace(scene=types.SimpleNamespace(num_envs=1), action_space=2)
    return types.SimpleNamespace(
        make_eval_env=lambda motion_files, show_reference_motion=False, window_lengths=None: (
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
    motion_obs_dim = motion_obs_dim if motion_obs_dim is not None else 7 * motion_window_length
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
        motion_files=motion_files or ["env/assests/115_06_stageii.npz"],
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
        motion_mae_encoder_checkpoint=motion_mae_encoder_checkpoint,
        observation_window_lengths=observation_window_lengths or None,
    )
    return save_checkpoint_v2(checkpoint, tmp_path / "model_v2.pth")


def _write_isaac_checkpoint(tmp_path):
    return _write_checkpoint(tmp_path)


def _write_sim2sim_checkpoint(tmp_path):
    return _write_checkpoint(tmp_path)


def _motion_mae_schema() -> MotionFeatureSchema:
    return MotionFeatureSchema(
        d_ref=7,
        d_target=7,
        full_feature_dim=7,
        base_slices=(
            FeatureSliceSpec("root", 0, 3),
            FeatureSliceSpec("joint", 3, 7),
        ),
        reference_slices=(
            FeatureSliceSpec("root", 0, 3),
            FeatureSliceSpec("joint", 3, 7),
        ),
        target_slices=(
            FeatureSliceSpec("root", 0, 3),
            FeatureSliceSpec("joint", 3, 7),
        ),
        policy_motion_slice=FeatureSliceSpec("policy_motion", 0, 7),
        anchor_body_name="pelvis",
        end_effector_body_names=("left_hand", "right_hand"),
        reference_feature_names=("root", "joint"),
        target_feature_names=("root", "joint"),
        policy_feature_names=("root", "joint"),
        joint_names=("j0", "j1"),
        body_names=("pelvis", "left_hand", "right_hand"),
        reference_mean=tuple(0.0 for _ in range(7)),
        reference_std=tuple(1.0 for _ in range(7)),
        target_mean=tuple(0.0 for _ in range(7)),
        target_std=tuple(1.0 for _ in range(7)),
    )


def _write_motion_mae_encoder_checkpoint(tmp_path) -> Path:
    model = ReferenceMotionMAE(
        input_dim=7,
        target_dim=7,
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
                motion_files=("env/assests/115_02_stageii.npz",),
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
    assert args.disable_amp is False
    assert args.disable_wandb is False
    assert args.motion_mae_encoder_checkpoint is None

    with pytest.raises(SystemExit):
        parser.parse_args(["train", "--attn-block-size", "5"])

    args = parser.parse_args(["train", "--disable-amp"])
    assert args.disable_amp is True

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
    assert runner.motion_files[0].endswith("115_06_stageii.npz")
    config_payload = json.loads(runner.run_paths.config_path.read_text(encoding="utf-8"))
    assert config_payload["config"]["use_amp"] is True
    runner.env.close()


def test_train_runner_constructs_film_res_actor(monkeypatch):
    monkeypatch.setitem(sys.modules, "gmtp.integrations.ref2act.isaac_env", _fake_train_module())
    runner = TrainRunner(
        RunConfig(
            num_blocks=4,
            robot_encoder_type="cnn",
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

    assert returned_obs["motion"].shape == (2, 7)
    assert returned_obs["robot"].shape == (2, 4, 12)
    assert runner.rollout_buffer.data["motion_observations"].shape[-1] == 7
    assert runner.rollout_buffer.data["robot_observations"].shape[-2:] == (4, 12)
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
    assert runner.obs_dims["motion"] == 28
    returned_obs = runner.rollout(runner.initial_obs)
    assert returned_obs["motion"].shape == (2, 4, 7)
    assert runner.rollout_buffer.data["motion_observations"].shape[-2:] == (4, 7)
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

    assert runner.motion_files[0].endswith("115_06_stageii.npz")
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
    assert runner.obs_dims["motion"] == 28


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
    assert runner.motion_files[0].endswith("115_06_stageii.npz")
