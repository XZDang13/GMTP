import sys
import types

import pytest
import torch

from gmtp.cli.main import build_parser
from gmtp.models import Critic, FiLMAttnResActor
from gmtp.runtime.checkpoints import build_training_checkpoint, save_checkpoint_v2
from gmtp.runtime.config import IsaacEvalConfig, RunConfig, Sim2SimEvalConfig
from gmtp.runtime.eval_isaac import IsaacEvalRunner
from gmtp.runtime.train_runner import TrainRunner


class _DummyTrainEnv:
    def __init__(self):
        self.unwrapped = self
        self.device = torch.device("cpu")

    def reset(self):
        obs = {
            "motion": torch.zeros(2, 3),
            "robot": torch.zeros(2, 4),
            "privilege": torch.zeros(2, 5),
        }
        return obs, {}

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
    return types.SimpleNamespace(make_training_env=lambda: (_DummyTrainEnv(), cfg))


def _fake_eval_module():
    cfg = types.SimpleNamespace(scene=types.SimpleNamespace(num_envs=1), action_space=2)
    env = _DummyTrainEnv()
    env.reset = lambda: (
        {
            "motion": torch.zeros(1, 3),
            "robot": torch.zeros(1, 4),
            "privilege": torch.zeros(1, 5),
        },
        {},
    )
    return types.SimpleNamespace(make_eval_env=lambda motion_files, show_reference_motion=False: (env, cfg))


def _write_checkpoint(
    tmp_path,
    *,
    motion_obs_dim: int,
    robot_obs_dim: int,
    motion_files: list[str] | None = None,
):
    actor = FiLMAttnResActor(
        robot_obs_dim=robot_obs_dim,
        motion_obs_dim=motion_obs_dim,
        action_dim=2,
        num_blocks=4,
        attn_block_size=2,
    )
    critic = Critic(obs_dim=5)
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
    )
    return save_checkpoint_v2(checkpoint, tmp_path / "model_v2.pth")


def _write_isaac_checkpoint(tmp_path):
    return _write_checkpoint(tmp_path, motion_obs_dim=3, robot_obs_dim=4)


def _write_sim2sim_checkpoint(tmp_path):
    return _write_checkpoint(tmp_path, motion_obs_dim=7, robot_obs_dim=12)


def test_cli_parser_builds_train_and_eval_commands():
    parser = build_parser()
    args = parser.parse_args(["train", "--num-updates", "5"])
    assert args.command == "train"
    assert args.num_updates == 5
    assert args.num_blocks == 6
    assert args.attn_block_size == 4

    args = parser.parse_args(["train", "--adain-res-blocks", "7"])
    assert args.num_blocks == 7

    args = parser.parse_args(["train", "--attn-block-size", "5"])
    assert args.attn_block_size == 5

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
        ]
    )
    assert args.command == "eval"
    assert args.eval_target == "sim2sim"
    assert args.motion_files == ["foo", "bar"]
    assert args.action_mode == "residual"
    assert args.num_steps == 12
    assert args.save_video is False
    assert args.attn_block_size is None

    args = parser.parse_args(["eval", "sim2sim", "--checkpoint", "foo.pth", "--save-video"])
    assert args.save_video is True


def test_cli_parser_rejects_removed_migrate_checkpoint_command():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["migrate-checkpoint", "--checkpoint", "legacy.pth"])


def test_train_runner_dry_construction(monkeypatch):
    monkeypatch.setitem(sys.modules, "gmtp.integrations.ref2act.isaac_env", _fake_train_module())
    runner = TrainRunner(RunConfig(use_wandb=False))
    assert runner.actor_type.value == "film_attn_res"
    assert runner.motion_files[0].endswith("115_06_stageii.npz")
    runner.env.close()


def test_train_runner_constructs_film_attn_res_actor(monkeypatch):
    monkeypatch.setitem(sys.modules, "gmtp.integrations.ref2act.isaac_env", _fake_train_module())
    runner = TrainRunner(
        RunConfig(
            num_blocks=4,
            attn_block_size=2,
            use_wandb=False,
        )
    )
    assert runner.actor_type.value == "film_attn_res"
    assert runner.actor_kwargs == {"num_blocks": 4, "attn_block_size": 2}
    assert isinstance(runner.actor, FiLMAttnResActor)
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
    assert runner.motion_files[0].endswith("115_06_stageii.npz")
    runner.env.close()


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
    assert runner.motion_files[0].endswith("115_06_stageii.npz")
