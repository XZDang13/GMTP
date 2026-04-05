from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from gmtp.models import ActorType, normalize_actor_type


def infer_env_observation_dims(obs: dict[str, torch.Tensor]) -> dict[str, int]:
    required_keys = ("motion", "robot", "privilege")
    missing_keys = [key for key in required_keys if key not in obs]
    if missing_keys:
        raise KeyError(f"Environment observation is missing required keys: {missing_keys}.")

    motion_dim = obs["motion"].shape[-1]
    robot_dim = obs["robot"].shape[-1]
    critic_dim = obs["privilege"].shape[-1]
    return {
        "motion": motion_dim,
        "robot": robot_dim,
        "critic": critic_dim,
        "policy": motion_dim + robot_dim,
    }


def infer_actor_observation_dims_from_state_dict(
    actor_weights: dict[str, torch.Tensor],
    actor_type: ActorType | str,
) -> dict[str, int]:
    normalize_actor_type(str(actor_type))
    motion_dim = actor_weights["motion_obs_normlizer.mean"].shape[0]
    robot_dim = actor_weights["robot_obs_normlizer.mean"].shape[0]
    return {
        "motion": motion_dim,
        "robot": robot_dim,
        "policy": motion_dim + robot_dim,
    }


def infer_sim2sim_observation_dims(action_dim: int) -> dict[str, int]:
    return {
        "motion": action_dim * 2 + 3,
        "robot": action_dim * 3 + 6,
        "policy": action_dim * 5 + 9,
    }


def parse_sim2sim_obs(flat_obs: torch.Tensor, action_dim: int) -> dict[str, torch.Tensor]:
    expected_dim = infer_sim2sim_observation_dims(action_dim)["policy"]
    if flat_obs.ndim != 1:
        raise ValueError(f"Expected a flat sim2sim observation, got shape {tuple(flat_obs.shape)}.")
    if flat_obs.numel() != expected_dim:
        raise ValueError(f"Expected sim2sim observation dim {expected_dim}, got {flat_obs.numel()}.")

    offset = 0
    target_projected_gravity = flat_obs[offset : offset + 3]
    offset += 3
    target_joint_pos = flat_obs[offset : offset + action_dim]
    offset += action_dim
    target_joint_vel = flat_obs[offset : offset + action_dim]
    offset += action_dim
    robot_projected_gravity = flat_obs[offset : offset + 3]
    offset += 3
    anchor_ang_vel = flat_obs[offset : offset + 3]
    offset += 3
    robot_joint_pos = flat_obs[offset : offset + action_dim]
    offset += action_dim
    robot_joint_vel = flat_obs[offset : offset + action_dim]
    offset += action_dim
    previous_action = flat_obs[offset : offset + action_dim]

    motion_obs = torch.cat((target_projected_gravity, target_joint_pos, target_joint_vel), dim=-1)
    robot_obs = torch.cat(
        (
            robot_projected_gravity,
            anchor_ang_vel,
            robot_joint_pos,
            robot_joint_vel,
            previous_action,
        ),
        dim=-1,
    )

    return {
        "motion": motion_obs,
        "robot": robot_obs,
        "motion_obs": motion_obs,
        "robot_obs": robot_obs,
        "target_projected_gravity": target_projected_gravity,
        "target_joint_pos": target_joint_pos,
        "target_joint_vel": target_joint_vel,
        "robot_projected_gravity": robot_projected_gravity,
        "anchor_ang_vel": anchor_ang_vel,
        "robot_joint_pos": robot_joint_pos,
        "robot_joint_vel": robot_joint_vel,
        "previous_action": previous_action,
    }


def _coerce_sim2sim_vector(value: Any, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device="cpu").reshape(-1)
    if tensor.ndim != 1:
        raise ValueError(f"Expected sim2sim {name} rank 1, got shape {tuple(tensor.shape)}.")
    return tensor


def extract_sim2sim_actor_obs_from_mapping(obs: Mapping[str, Any]) -> dict[str, torch.Tensor] | None:
    motion_value = obs.get("motion", obs.get("motion_obs"))
    robot_value = obs.get("robot", obs.get("robot_obs"))
    if motion_value is None or robot_value is None:
        return None

    motion_obs = _coerce_sim2sim_vector(motion_value, name="motion")
    robot_obs = _coerce_sim2sim_vector(robot_value, name="robot")
    return {
        "motion": motion_obs,
        "robot": robot_obs,
        "motion_obs": motion_obs,
        "robot_obs": robot_obs,
    }


def split_sim2sim_group_observations(
    motion_obs: torch.Tensor,
    robot_obs: torch.Tensor,
    action_dim: int,
) -> dict[str, torch.Tensor]:
    motion_obs = _coerce_sim2sim_vector(motion_obs, name="motion")
    robot_obs = _coerce_sim2sim_vector(robot_obs, name="robot")
    expected_dims = infer_sim2sim_observation_dims(action_dim)
    if motion_obs.numel() != expected_dims["motion"]:
        raise ValueError(f"Expected structured sim2sim motion dim {expected_dims['motion']}, got {motion_obs.numel()}.")
    if robot_obs.numel() != expected_dims["robot"]:
        raise ValueError(f"Expected structured sim2sim robot dim {expected_dims['robot']}, got {robot_obs.numel()}.")

    motion_offset = 0
    target_projected_gravity = motion_obs[motion_offset : motion_offset + 3]
    motion_offset += 3
    target_joint_pos = motion_obs[motion_offset : motion_offset + action_dim]
    motion_offset += action_dim
    target_joint_vel = motion_obs[motion_offset : motion_offset + action_dim]

    robot_offset = 0
    robot_projected_gravity = robot_obs[robot_offset : robot_offset + 3]
    robot_offset += 3
    anchor_ang_vel = robot_obs[robot_offset : robot_offset + 3]
    robot_offset += 3
    robot_joint_pos = robot_obs[robot_offset : robot_offset + action_dim]
    robot_offset += action_dim
    robot_joint_vel = robot_obs[robot_offset : robot_offset + action_dim]
    robot_offset += action_dim
    previous_action = robot_obs[robot_offset : robot_offset + action_dim]

    return {
        "motion": motion_obs,
        "robot": robot_obs,
        "motion_obs": motion_obs,
        "robot_obs": robot_obs,
        "target_projected_gravity": target_projected_gravity,
        "target_joint_pos": target_joint_pos,
        "target_joint_vel": target_joint_vel,
        "robot_projected_gravity": robot_projected_gravity,
        "anchor_ang_vel": anchor_ang_vel,
        "robot_joint_pos": robot_joint_pos,
        "robot_joint_vel": robot_joint_vel,
        "previous_action": previous_action,
    }


def build_sim2sim_obs_parts_from_context(context: Any) -> dict[str, torch.Tensor]:
    target_projected_gravity = _coerce_sim2sim_vector(context.target_projected_gravity, name="target_projected_gravity")
    target_joint_pos = _coerce_sim2sim_vector(context.target_joint_pos, name="target_joint_pos")
    target_joint_vel = _coerce_sim2sim_vector(context.target_joint_vel, name="target_joint_vel")
    robot_projected_gravity = _coerce_sim2sim_vector(context.projected_gravity, name="projected_gravity")
    anchor_ang_vel = _coerce_sim2sim_vector(context.anchor_ang_vel_b, name="anchor_ang_vel_b")
    robot_joint_pos = _coerce_sim2sim_vector(context.joint_pos, name="joint_pos")
    robot_joint_vel = _coerce_sim2sim_vector(context.joint_vel, name="joint_vel")
    previous_action = _coerce_sim2sim_vector(context.previous_action, name="previous_action")

    motion_obs = torch.cat((target_projected_gravity, target_joint_pos, target_joint_vel), dim=-1)
    robot_obs = torch.cat(
        (
            robot_projected_gravity,
            anchor_ang_vel,
            robot_joint_pos,
            robot_joint_vel,
            previous_action,
        ),
        dim=-1,
    )
    return {
        "motion": motion_obs,
        "robot": robot_obs,
        "motion_obs": motion_obs,
        "robot_obs": robot_obs,
        "target_projected_gravity": target_projected_gravity,
        "target_joint_pos": target_joint_pos,
        "target_joint_vel": target_joint_vel,
        "robot_projected_gravity": robot_projected_gravity,
        "anchor_ang_vel": anchor_ang_vel,
        "robot_joint_pos": robot_joint_pos,
        "robot_joint_vel": robot_joint_vel,
        "previous_action": previous_action,
    }


def extract_sim2sim_metrics_from_parts(obs_parts: Mapping[str, torch.Tensor]) -> dict[str, float]:
    required_keys = (
        "target_joint_pos",
        "robot_joint_pos",
        "target_joint_vel",
        "robot_joint_vel",
        "target_projected_gravity",
        "robot_projected_gravity",
    )
    missing_keys = [key for key in required_keys if key not in obs_parts]
    if missing_keys:
        raise KeyError(f"Sim2sim observation parts are missing required keys: {missing_keys}.")

    return {
        "joint_pos_mae": torch.mean(torch.abs(obs_parts["target_joint_pos"] - obs_parts["robot_joint_pos"])).item(),
        "joint_vel_mae": torch.mean(torch.abs(obs_parts["target_joint_vel"] - obs_parts["robot_joint_vel"])).item(),
        "gravity_mae": torch.mean(
            torch.abs(obs_parts["target_projected_gravity"] - obs_parts["robot_projected_gravity"])
        ).item(),
    }


def extract_sim2sim_metrics(flat_obs: torch.Tensor, action_dim: int) -> dict[str, float]:
    return extract_sim2sim_metrics_from_parts(parse_sim2sim_obs(flat_obs, action_dim))


def build_actor_obs_log_fields(actor_obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {f"actor_obs_{key}": value for key, value in actor_obs.items()}
