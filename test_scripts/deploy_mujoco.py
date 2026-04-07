from __future__ import annotations

import argparse
import inspect
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch

from gmtp.integrations.ref2act import DEFAULT_EXPERIMENT_MOTION_FILES, infer_motion_files_from_checkpoint, resolve_motion_files
from gmtp.integrations.ref2act.mujoco import (
    DEFAULT_ANCHOR_BODY_NAME,
    DEFAULT_ROOT_NAME,
    get_mujoco_symbols,
    resolve_action_mode,
    resolve_name_override,
)
from gmtp.integrations.ref2act.observation_history import (
    build_gmtp_policy_observation_spec,
    resolve_observation_window_lengths,
)
from gmtp.models import get_actor_observation
from gmtp.runtime.checkpoints import load_checkpoint_v2
from gmtp.runtime.observations import (
    extract_sim2sim_actor_obs_from_mapping,
    infer_actor_observation_dims_from_state_dict,
    infer_sim2sim_observation_dims,
    parse_sim2sim_obs,
    replace_sim2sim_group_latest_terms,
    split_sim2sim_group_observations,
)
from gmtp.runtime.policy import build_motion_mae_adapter, load_actor_from_checkpoint, resolve_motion_mae_checkpoint_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone MuJoCo smoke-test runner for a GMTP checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--motion-file", default=None)
    parser.add_argument("--num-blocks", type=int, default=None)
    parser.add_argument("--robot-window-length", type=int, default=None)
    parser.add_argument("--motion-mae-encoder-checkpoint", default=None)
    parser.add_argument("--num-steps", type=int, default=2000)
    parser.add_argument("--simulation-dt", type=float, default=1 / 200)
    parser.add_argument("--decimation", type=int, default=4)
    parser.add_argument("--action-mode", default=None)
    parser.add_argument("--root-name", default=None)
    parser.add_argument("--anchor-body-name", default=None)
    parser.add_argument("--headless", action="store_true")
    return parser


def _infer_action_dim(checkpoint_env: Mapping[str, Any]) -> int:
    joint_names = checkpoint_env.get("joint_names")
    if joint_names:
        return len(joint_names)

    for key in (
        "action_offset",
        "action_scale",
        "joint_effort_limits",
        "joint_stiffness",
        "joint_damping",
        "joint_pos_limits",
    ):
        value = checkpoint_env.get(key)
        if value is None:
            continue
        tensor = torch.as_tensor(value)
        if tensor.ndim == 0:
            continue
        return int(tensor.shape[0])

    raise KeyError("Could not infer action dimension from checkpoint env payload.")


def _resolve_motion_file(
    *,
    checkpoint_path: Path,
    checkpoint_actor_type: str,
    checkpoint_env: dict[str, Any],
    explicit_motion_file: str | None,
) -> str:
    if explicit_motion_file is not None:
        return resolve_motion_files([explicit_motion_file])[0]

    motion_files = infer_motion_files_from_checkpoint(
        checkpoint_path,
        checkpoint_actor_type,
        checkpoint_env,
        default_motion_files=DEFAULT_EXPERIMENT_MOTION_FILES,
    )
    return motion_files[0]


def _get_env_obs_dict(env: Any) -> Mapping[str, Any] | None:
    getter = getattr(env, "get_obs_dict", None)
    if not callable(getter):
        return None

    try:
        parameters = inspect.signature(getter).parameters
    except (TypeError, ValueError):
        parameters = {}

    obs = getter(advance_time=False) if "advance_time" in parameters else getter()
    if not isinstance(obs, Mapping):
        raise ValueError(f"Expected env.get_obs_dict() to return a mapping, got {type(obs).__name__}.")
    return obs


def _get_gravity_orientation(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = torch.as_tensor(quaternion, dtype=torch.float32).reshape(-1)
    if quaternion.numel() != 4:
        raise ValueError(f"Expected free-joint quaternion with 4 values, got shape {tuple(quaternion.shape)}.")

    qw, qx, qy, qz = quaternion
    return torch.stack(
        (
            2.0 * (-qz * qx + qw * qy),
            -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz),
        )
    )


def _override_robot_terms_from_mujoco_state(
    env: Any,
    obs_parts: dict[str, torch.Tensor],
    *,
    action_dim: int,
    observation_window_lengths: Mapping[str, int] | None,
) -> dict[str, torch.Tensor]:
    mj_data = getattr(env, "mj_data", None)
    if mj_data is None:
        return obs_parts

    qpos = getattr(mj_data, "qpos", None)
    qvel = getattr(mj_data, "qvel", None)
    if qpos is None or qvel is None:
        return obs_parts

    qpos = torch.as_tensor(qpos, dtype=torch.float32).reshape(-1)
    qvel = torch.as_tensor(qvel, dtype=torch.float32).reshape(-1)
    if qpos.numel() < 7 or qvel.numel() < 6:
        return obs_parts

    robot_projected_gravity = _get_gravity_orientation(qpos[3:7])
    anchor_ang_vel = qvel[3:6].clone()

    obs_parts = dict(obs_parts)
    obs_parts["robot_projected_gravity"] = robot_projected_gravity
    obs_parts["anchor_ang_vel"] = anchor_ang_vel
    obs_parts["robot"] = replace_sim2sim_group_latest_terms(
        obs_parts["robot"],
        group_name="robot",
        action_dim=action_dim,
        latest_terms={
            "projected_gravity": robot_projected_gravity,
            "anchor_ang_vel_b": anchor_ang_vel,
        },
        observation_window_lengths=observation_window_lengths,
    )
    obs_parts["robot_obs"] = obs_parts["robot"]
    return obs_parts


def _coerce_flat_obs(obs: Any) -> torch.Tensor:
    tensor = torch.as_tensor(obs, dtype=torch.float32).reshape(-1)
    return tensor


def _extract_obs_parts(
    env: Any,
    obs: Any,
    action_dim: int,
    observation_window_lengths: Mapping[str, int] | None,
) -> dict[str, torch.Tensor]:
    structured_obs = None
    if isinstance(obs, Mapping):
        structured_obs = obs
    else:
        try:
            structured_obs = _get_env_obs_dict(env)
        except (AttributeError, TypeError, ValueError, KeyError):
            structured_obs = None

    if structured_obs is not None:
        structured_actor_obs = extract_sim2sim_actor_obs_from_mapping(
            structured_obs,
            action_dim=action_dim,
            observation_window_lengths=observation_window_lengths,
        )
        if structured_actor_obs is None:
            raise KeyError("Expected structured sim2sim observation mapping to include motion/robot entries.")
        obs_parts = split_sim2sim_group_observations(
            structured_actor_obs["motion"],
            structured_actor_obs["robot"],
            action_dim,
            observation_window_lengths=observation_window_lengths,
        )
    else:
        obs_parts = parse_sim2sim_obs(
            _coerce_flat_obs(obs),
            action_dim,
            observation_window_lengths=observation_window_lengths,
        )
    return _override_robot_terms_from_mujoco_state(
        env,
        obs_parts,
        action_dim=action_dim,
        observation_window_lengths=observation_window_lengths,
    )


def _tensor_dict_to_batch(obs_parts: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "motion": obs_parts["motion"].unsqueeze(0),
        "robot": obs_parts["robot"].unsqueeze(0),
    }


def _viewer_is_running(env: Any, *, render: bool) -> bool:
    if not render:
        return True
    viewer = getattr(env, "mj_viewer", None)
    if viewer is None:
        return False
    return bool(getattr(viewer, "is_alive", True))


def _build_env(
    *,
    checkpoint_env: dict[str, Any],
    motion_file: str,
    simulation_dt: float,
    decimation: int,
    action_mode: str,
    root_name: str,
    anchor_body_name: str,
    render: bool,
    observation_window_lengths: Mapping[str, int] | None,
) -> Any:
    symbols = get_mujoco_symbols()
    env_kwargs = {
        "simulation_dt": simulation_dt,
        "decimation": decimation,
        "kp": torch.as_tensor(checkpoint_env["joint_stiffness"], dtype=torch.float32),
        "kd": torch.as_tensor(checkpoint_env["joint_damping"], dtype=torch.float32),
        "effort_limits": torch.as_tensor(checkpoint_env["joint_effort_limits"], dtype=torch.float32),
        "joint_pos_limits": torch.as_tensor(checkpoint_env["joint_pos_limits"], dtype=torch.float32),
        "action_offset": torch.as_tensor(checkpoint_env["action_offset"], dtype=torch.float32),
        "action_scale": torch.as_tensor(checkpoint_env["action_scale"], dtype=torch.float32),
        "expert_motion_file": motion_file,
        "root_link_name": root_name,
        "anchor_body_name": anchor_body_name,
        "render": render,
        "action_mode": action_mode,
    }
    observation_builder_cls = getattr(symbols, "IsaacLabMujocoObservation", None)
    if observation_builder_cls is not None:
        try:
            init_parameters = inspect.signature(symbols.MujocoEnv).parameters
        except (TypeError, ValueError):
            init_parameters = {}
        if "observation_builder" in init_parameters:
            env_kwargs["observation_builder"] = observation_builder_cls(
                spec=build_gmtp_policy_observation_spec(
                    add_noise=False,
                    window_lengths=observation_window_lengths or None,
                )
            )
    return symbols.MujocoEnv(**env_kwargs)


@torch.no_grad()
def run(args: argparse.Namespace) -> int:
    if args.num_steps < 0:
        raise ValueError(f"--num-steps must be non-negative, got {args.num_steps}.")
    if args.decimation < 1:
        raise ValueError(f"--decimation must be positive, got {args.decimation}.")
    if args.simulation_dt <= 0:
        raise ValueError(f"--simulation-dt must be positive, got {args.simulation_dt}.")

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = load_checkpoint_v2(checkpoint_path)
    checkpoint_env = checkpoint.env
    action_dim = _infer_action_dim(checkpoint_env)
    observation_window_lengths = resolve_observation_window_lengths(
        robot_window_length=args.robot_window_length,
        checkpoint_env=checkpoint_env,
    )
    motion_file = _resolve_motion_file(
        checkpoint_path=checkpoint_path,
        checkpoint_actor_type=checkpoint.actor_type,
        checkpoint_env=checkpoint_env,
        explicit_motion_file=args.motion_file,
    )
    action_mode, action_mode_source = resolve_action_mode(
        checkpoint_env,
        args.action_mode,
        torch.as_tensor(checkpoint_env["action_offset"], dtype=torch.float32),
        torch.as_tensor(checkpoint_env["action_scale"], dtype=torch.float32),
        torch.as_tensor(checkpoint_env["joint_pos_limits"], dtype=torch.float32),
    )
    root_name = resolve_name_override(
        args.root_name,
        checkpoint_env,
        ("root_name", "root_link_name"),
        DEFAULT_ROOT_NAME,
    )
    anchor_body_name = resolve_name_override(
        args.anchor_body_name,
        checkpoint_env,
        ("anchor_body_name",),
        DEFAULT_ANCHOR_BODY_NAME,
    )
    resolved_motion_mae_checkpoint = resolve_motion_mae_checkpoint_path(
        checkpoint,
        override=args.motion_mae_encoder_checkpoint,
    )
    motion_mae_encoder_checkpoint = (
        None if resolved_motion_mae_checkpoint is None else str(resolved_motion_mae_checkpoint)
    )
    motion_mae_adapter = build_motion_mae_adapter(
        motion_mae_encoder_checkpoint,
        device=torch.device("cpu"),
    )
    raw_obs_dims = infer_sim2sim_observation_dims(
        action_dim,
        observation_window_lengths=observation_window_lengths,
    )
    obs_dims = motion_mae_adapter.augment_observation_dims(raw_obs_dims) if motion_mae_adapter is not None else raw_obs_dims
    checkpoint_obs_dims = infer_actor_observation_dims_from_state_dict(
        checkpoint.model["actor"],
        checkpoint.actor_type,
    )
    if checkpoint_obs_dims["motion"] != obs_dims["motion"] or checkpoint_obs_dims["robot"] != obs_dims["robot"]:
        raise ValueError(
            "Checkpoint actor observation dims do not match runtime env dims: "
            f"checkpoint={checkpoint_obs_dims}, runtime={obs_dims}."
        )
    actor, actor_type, actor_kwargs = load_actor_from_checkpoint(
        checkpoint,
        obs_dims=obs_dims,
        action_dim=action_dim,
        device=torch.device("cpu"),
        num_blocks=args.num_blocks,
    )
    render = not args.headless
    env = _build_env(
        checkpoint_env=checkpoint_env,
        motion_file=motion_file,
        simulation_dt=args.simulation_dt,
        decimation=args.decimation,
        action_mode=action_mode,
        root_name=root_name,
        anchor_body_name=anchor_body_name,
        render=render,
        observation_window_lengths=observation_window_lengths,
    )

    print(
        "Starting MuJoCo deploy smoke test "
        f"(checkpoint={checkpoint_path}, motion={motion_file}, actor_type={actor_type.value}, "
        f"action_mode={action_mode} [{action_mode_source}], render={render})"
    )

    steps_executed = 0
    try:
        obs_parts = _extract_obs_parts(
            env,
            env.reset(),
            action_dim,
            observation_window_lengths,
        )
        if motion_mae_adapter is not None:
            motion_mae_adapter.initialize_history(env)
        while steps_executed < args.num_steps and _viewer_is_running(env, render=render):
            actor_env_obs = _tensor_dict_to_batch(obs_parts)
            actor_obs = get_actor_observation(actor_env_obs, actor_type)
            if motion_mae_adapter is not None:
                actor_obs = motion_mae_adapter.augment_actor_observation(actor_obs)
            actor_step = actor(actor_obs)
            action = actor_step.mean.squeeze(0).detach().to(device="cpu", dtype=torch.float32)
            if not torch.isfinite(action).all():
                raise RuntimeError(
                    f"Non-finite action detected: min={float(action.min().item()):.6f} "
                    f"max={float(action.max().item()):.6f}"
                )

            obs_parts = _extract_obs_parts(
                env,
                env.step(action),
                action_dim,
                observation_window_lengths,
            )
            if motion_mae_adapter is not None:
                motion_mae_adapter.update_history(env)
            steps_executed += 1
    finally:
        env.close()

    print(
        "Finished MuJoCo deploy smoke test "
        f"(steps={steps_executed}/{args.num_steps}, motion={motion_file}, actor_kwargs={actor_kwargs})"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
