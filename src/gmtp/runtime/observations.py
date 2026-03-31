from __future__ import annotations

import torch

from gmtp.models import ActorType, is_concat_actor, normalize_actor_type


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
    normalized = normalize_actor_type(str(actor_type))
    if is_concat_actor(normalized):
        policy_dim = actor_weights["normlizer.mean"].shape[0]
        return {"policy": policy_dim}

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


def extract_sim2sim_metrics(flat_obs: torch.Tensor, action_dim: int) -> dict[str, float]:
    obs_parts = parse_sim2sim_obs(flat_obs, action_dim)
    return {
        "joint_pos_mae": torch.mean(torch.abs(obs_parts["target_joint_pos"] - obs_parts["robot_joint_pos"])).item(),
        "joint_vel_mae": torch.mean(torch.abs(obs_parts["target_joint_vel"] - obs_parts["robot_joint_vel"])).item(),
        "gravity_mae": torch.mean(
            torch.abs(obs_parts["target_projected_gravity"] - obs_parts["robot_projected_gravity"])
        ).item(),
    }


def build_actor_obs_log_fields(actor_obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {f"actor_obs_{key}": value for key, value in actor_obs.items()}


def get_actor_state_log_fields(actor_state: torch.Tensor | None) -> dict[str, float]:
    if actor_state is None:
        return {}
    actor_state = actor_state.detach()
    return {
        "actor_state_l2": float(actor_state.norm().item()),
        "actor_state_max_abs": float(actor_state.abs().max().item()),
    }
