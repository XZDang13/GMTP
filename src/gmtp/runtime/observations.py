from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import Any

import torch

from gmtp.integrations.ref2act.compat import _import_module
from gmtp.integrations.ref2act.observation_history import (
    MOTION_POLICY_OBSERVATION_TERM_IDS,
    ROBOT_POLICY_OBSERVATION_TERM_IDS,
    build_gmtp_policy_observation_spec,
    normalize_observation_window_lengths,
)
from gmtp.models import ActorType, normalize_actor_type
from gmtp.models.motion_encoder import (
    build_motion_window_layout,
    flatten_motion_history,
    reshape_motion_history,
)
from gmtp.models.robot_encoder import (
    build_robot_window_layout,
    flatten_robot_history,
    reshape_robot_history,
)

_OBSERVATION_SPEC = _import_module("ref2act.common.observation_spec")
DEFAULT_OBSERVATION_TERM_REGISTRY = _OBSERVATION_SPEC.DEFAULT_OBSERVATION_TERM_REGISTRY
ObservationLayout = _OBSERVATION_SPEC.ObservationLayout

_SIM2SIM_TERM_OUTPUT_NAMES = {
    "target_projected_gravity": "target_projected_gravity",
    "target_joint_pos": "target_joint_pos",
    "projected_gravity": "robot_projected_gravity",
    "anchor_ang_vel_b": "anchor_ang_vel",
    "joint_pos": "robot_joint_pos",
    "joint_vel": "robot_joint_vel",
    "previous_action": "previous_action",
}


def _trailing_numel(value: torch.Tensor) -> int:
    if value.ndim < 1:
        raise ValueError(f"Expected observation tensor rank >= 1, got shape {tuple(value.shape)}.")
    if value.ndim == 1:
        return int(value.numel())
    return int(value[0].numel())


def _resolve_robot_window_length(
    observation_window_lengths: Mapping[str, int] | None,
) -> int:
    normalized = normalize_observation_window_lengths(observation_window_lengths)
    lengths = {int(normalized.get(term_id, 1)) for term_id in ROBOT_POLICY_OBSERVATION_TERM_IDS}
    if len(lengths) != 1:
        raise ValueError(
            "Robot policy observation window lengths must match across "
            f"{ROBOT_POLICY_OBSERVATION_TERM_IDS}, got {sorted(lengths)}."
        )
    return lengths.pop()


def _resolve_motion_window_length(
    observation_window_lengths: Mapping[str, int] | None,
) -> int:
    normalized = normalize_observation_window_lengths(observation_window_lengths)
    lengths = {int(normalized.get(term_id, 1)) for term_id in MOTION_POLICY_OBSERVATION_TERM_IDS}
    if len(lengths) != 1:
        raise ValueError(
            "Motion policy observation window lengths must match across "
            f"{MOTION_POLICY_OBSERVATION_TERM_IDS}, got {sorted(lengths)}."
        )
    return lengths.pop()


def structure_motion_observation(
    motion_obs: Any,
    *,
    action_dim: int,
    observation_window_lengths: Mapping[str, int] | None = None,
) -> torch.Tensor:
    tensor = torch.as_tensor(motion_obs)
    motion_window_length = _resolve_motion_window_length(observation_window_lengths)
    if motion_window_length == 1:
        if tensor.ndim == 3 and tensor.shape[1] == 1:
            tensor = tensor.squeeze(1)
        if tensor.ndim not in {1, 2}:
            raise ValueError(
                f"Expected single-frame motion observation rank 1 or 2, got shape {tuple(tensor.shape)}."
            )
        return tensor

    layout = build_motion_window_layout(action_dim, motion_window_length)
    structured_shape = (layout.window_length, layout.motion_step_dim)
    if tensor.ndim in {1, 2} and tensor.shape[-1] == layout.motion_obs_dim:
        return reshape_motion_history(tensor, layout)
    if tensor.ndim in {2, 3} and tuple(tensor.shape[-2:]) == structured_shape:
        return tensor
    raise ValueError(
        "Expected windowed motion observation to be either flat with dim "
        f"{layout.motion_obs_dim} or structured with trailing shape {structured_shape}, "
        f"got {tuple(tensor.shape)}."
    )


def structure_robot_observation(
    robot_obs: Any,
    *,
    action_dim: int,
    observation_window_lengths: Mapping[str, int] | None = None,
) -> torch.Tensor:
    tensor = torch.as_tensor(robot_obs)
    robot_window_length = _resolve_robot_window_length(observation_window_lengths)
    if robot_window_length == 1:
        if tensor.ndim == 3 and tensor.shape[1] == 1:
            tensor = tensor.squeeze(1)
        if tensor.ndim not in {1, 2}:
            raise ValueError(
                f"Expected single-frame robot observation rank 1 or 2, got shape {tuple(tensor.shape)}."
            )
        return tensor

    layout = build_robot_window_layout(action_dim, robot_window_length)
    structured_shape = (layout.window_length, layout.robot_step_dim)
    if tensor.ndim in {1, 2} and tensor.shape[-1] == layout.robot_obs_dim:
        return reshape_robot_history(tensor, layout)
    if tensor.ndim in {2, 3} and tuple(tensor.shape[-2:]) == structured_shape:
        return tensor
    raise ValueError(
        "Expected windowed robot observation to be either flat with dim "
        f"{layout.robot_obs_dim} or structured with trailing shape {structured_shape}, "
        f"got {tuple(tensor.shape)}."
    )


def structure_env_observation(
    obs: Mapping[str, Any],
    *,
    action_dim: int,
    observation_window_lengths: Mapping[str, int] | None = None,
) -> dict[str, torch.Tensor]:
    structured_obs = dict(obs)
    if "motion" in structured_obs:
        structured_obs["motion"] = structure_motion_observation(
            structured_obs["motion"],
            action_dim=action_dim,
            observation_window_lengths=observation_window_lengths,
        )
    if "robot" not in structured_obs:
        return structured_obs
    structured_obs["robot"] = structure_robot_observation(
        structured_obs["robot"],
        action_dim=action_dim,
        observation_window_lengths=observation_window_lengths,
    )
    return structured_obs


def infer_env_observation_dims(obs: dict[str, torch.Tensor]) -> dict[str, int]:
    required_keys = ("motion", "robot", "privilege")
    missing_keys = [key for key in required_keys if key not in obs]
    if missing_keys:
        raise KeyError(f"Environment observation is missing required keys: {missing_keys}.")

    motion_dim = _trailing_numel(torch.as_tensor(obs["motion"]))
    robot_dim = _trailing_numel(torch.as_tensor(obs["robot"]))
    critic_dim = _trailing_numel(torch.as_tensor(obs["privilege"]))
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
    motion_dim = int(actor_weights["motion_obs_normlizer.mean"].numel())
    robot_dim = int(actor_weights["robot_obs_normlizer.mean"].numel())
    return {
        "motion": motion_dim,
        "robot": robot_dim,
        "policy": motion_dim + robot_dim,
    }


def _window_lengths_cache_key(
    observation_window_lengths: Mapping[str, int] | None,
) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(normalize_observation_window_lengths(observation_window_lengths).items()))


@lru_cache(maxsize=None)
def _get_sim2sim_policy_spec_details(
    action_dim: int,
    window_lengths_key: tuple[tuple[str, int], ...],
) -> tuple[Any, Any, dict[str, Any], dict[str, int]]:
    observation_window_lengths = dict(window_lengths_key)
    spec = build_gmtp_policy_observation_spec(
        add_noise=False,
        window_lengths=observation_window_lengths or None,
    )
    layout = ObservationLayout(joint_dim=action_dim, action_dim=action_dim, key_body_count=0)
    group_specs = {group.name: group for group in spec.enabled_groups()}
    group_dims = {name: int(dim) for name, dim in spec.describe(layout).group_dims.items()}
    return spec, layout, group_specs, group_dims


def _term_dim(term_spec: Any, layout: Any) -> int:
    return int(DEFAULT_OBSERVATION_TERM_REGISTRY[term_spec.type].dimension(layout, term_spec))


def infer_sim2sim_observation_dims(
    action_dim: int,
    observation_window_lengths: Mapping[str, int] | None = None,
) -> dict[str, int]:
    _, _, _, group_dims = _get_sim2sim_policy_spec_details(
        action_dim,
        _window_lengths_cache_key(observation_window_lengths),
    )
    motion_dim = int(group_dims.get("motion", 0))
    robot_dim = int(group_dims.get("robot", 0))
    return {
        "motion": motion_dim,
        "robot": robot_dim,
        "policy": motion_dim + robot_dim,
    }


def _extract_latest_term_value(term_flat: torch.Tensor, *, term_spec: Any, term_dim: int) -> torch.Tensor:
    if term_spec.window_length > 1 and term_spec.flatten:
        return term_flat.reshape(term_spec.window_length, term_dim)[-1]
    return term_flat


def _coerce_sim2sim_vector(value: Any, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device="cpu").reshape(-1)
    if tensor.ndim != 1:
        raise ValueError(f"Expected sim2sim {name} rank 1, got shape {tuple(tensor.shape)}.")
    return tensor


def _split_group_observation(
    group_name: str,
    group_obs: torch.Tensor,
    *,
    action_dim: int,
    observation_window_lengths: Mapping[str, int] | None = None,
) -> dict[str, torch.Tensor]:
    _, layout, group_specs, group_dims = _get_sim2sim_policy_spec_details(
        action_dim,
        _window_lengths_cache_key(observation_window_lengths),
    )
    group_spec = group_specs[group_name]
    expected_dim = int(group_dims[group_name])
    if group_obs.numel() != expected_dim:
        raise ValueError(f"Expected structured sim2sim {group_name} dim {expected_dim}, got {group_obs.numel()}.")

    offset = 0
    parsed_terms: dict[str, torch.Tensor] = {}
    for term_spec in group_spec.terms:
        if not term_spec.enabled:
            continue
        term_dim = _term_dim(term_spec, layout)
        flattened_dim = term_dim * term_spec.window_length if term_spec.flatten else term_dim
        term_flat = group_obs[offset : offset + flattened_dim]
        parsed_terms[_SIM2SIM_TERM_OUTPUT_NAMES[term_spec.id]] = _extract_latest_term_value(
            term_flat,
            term_spec=term_spec,
            term_dim=term_dim,
        )
        offset += flattened_dim

    return parsed_terms


def replace_sim2sim_group_latest_terms(
    group_obs: torch.Tensor,
    *,
    group_name: str,
    action_dim: int,
    latest_terms: Mapping[str, Any],
    observation_window_lengths: Mapping[str, int] | None = None,
) -> torch.Tensor:
    if group_name == "motion" and _resolve_motion_window_length(observation_window_lengths) > 1:
        layout = build_motion_window_layout(action_dim, _resolve_motion_window_length(observation_window_lengths))
        structured = structure_motion_observation(
            group_obs,
            action_dim=action_dim,
            observation_window_lengths=observation_window_lengths,
        )
        if structured.ndim != 2:
            raise ValueError(
                f"Expected structured sim2sim motion observation rank 2, got shape {tuple(structured.shape)}."
            )

        updated = structured.clone()
        term_map = {term_layout.term_id: term_layout for term_layout in layout.terms}
        for term_id, replacement_value in latest_terms.items():
            if term_id not in term_map:
                continue
            term_layout = term_map[term_id]
            replacement = _coerce_sim2sim_vector(replacement_value, name=f"{group_name}.{term_id}")
            if replacement.numel() != term_layout.term_dim:
                raise ValueError(
                    f"Expected replacement for {group_name}.{term_id} dim {term_layout.term_dim}, got {replacement.numel()}."
                )
            updated[-1, term_layout.step_slice] = replacement

        input_tensor = torch.as_tensor(group_obs)
        return flatten_motion_history(updated, layout) if input_tensor.ndim == 1 else updated

    if group_name == "robot" and _resolve_robot_window_length(observation_window_lengths) > 1:
        layout = build_robot_window_layout(action_dim, _resolve_robot_window_length(observation_window_lengths))
        structured = structure_robot_observation(
            group_obs,
            action_dim=action_dim,
            observation_window_lengths=observation_window_lengths,
        )
        if structured.ndim != 2:
            raise ValueError(
                f"Expected structured sim2sim robot observation rank 2, got shape {tuple(structured.shape)}."
            )

        updated = structured.clone()
        term_map = {term_layout.term_id: term_layout for term_layout in layout.terms}
        for term_id, replacement_value in latest_terms.items():
            if term_id not in term_map:
                continue
            term_layout = term_map[term_id]
            replacement = _coerce_sim2sim_vector(replacement_value, name=f"{group_name}.{term_id}")
            if replacement.numel() != term_layout.term_dim:
                raise ValueError(
                    f"Expected replacement for {group_name}.{term_id} dim {term_layout.term_dim}, got {replacement.numel()}."
                )
            updated[-1, term_layout.step_slice] = replacement

        input_tensor = torch.as_tensor(group_obs)
        return flatten_robot_history(updated, layout) if input_tensor.ndim == 1 else updated

    group_obs = _coerce_sim2sim_vector(group_obs, name=group_name).clone()
    _, layout, group_specs, group_dims = _get_sim2sim_policy_spec_details(
        action_dim,
        _window_lengths_cache_key(observation_window_lengths),
    )
    group_spec = group_specs[group_name]
    expected_dim = int(group_dims[group_name])
    if group_obs.numel() != expected_dim:
        raise ValueError(f"Expected structured sim2sim {group_name} dim {expected_dim}, got {group_obs.numel()}.")

    offset = 0
    for term_spec in group_spec.terms:
        if not term_spec.enabled:
            continue
        term_dim = _term_dim(term_spec, layout)
        flattened_dim = term_dim * term_spec.window_length if term_spec.flatten else term_dim
        if term_spec.id in latest_terms:
            replacement = _coerce_sim2sim_vector(latest_terms[term_spec.id], name=f"{group_name}.{term_spec.id}")
            if replacement.numel() != term_dim:
                raise ValueError(
                    f"Expected replacement for {group_name}.{term_spec.id} dim {term_dim}, got {replacement.numel()}."
                )
            if term_spec.window_length > 1 and term_spec.flatten:
                group_obs[offset + flattened_dim - term_dim : offset + flattened_dim] = replacement
            else:
                group_obs[offset : offset + flattened_dim] = replacement
        offset += flattened_dim

    return group_obs


def parse_sim2sim_obs(
    flat_obs: torch.Tensor,
    action_dim: int,
    observation_window_lengths: Mapping[str, int] | None = None,
) -> dict[str, torch.Tensor]:
    expected_dims = infer_sim2sim_observation_dims(action_dim, observation_window_lengths)
    expected_dim = expected_dims["policy"]
    if flat_obs.ndim != 1:
        raise ValueError(f"Expected a flat sim2sim observation, got shape {tuple(flat_obs.shape)}.")
    if flat_obs.numel() != expected_dim:
        raise ValueError(f"Expected sim2sim observation dim {expected_dim}, got {flat_obs.numel()}.")

    motion_dim = expected_dims["motion"]
    motion_obs = flat_obs[:motion_dim]
    robot_obs = flat_obs[motion_dim:]
    return split_sim2sim_group_observations(
        motion_obs,
        robot_obs,
        action_dim,
        observation_window_lengths=observation_window_lengths,
    )


def extract_sim2sim_actor_obs_from_mapping(
    obs: Mapping[str, Any],
    *,
    action_dim: int,
    observation_window_lengths: Mapping[str, int] | None = None,
) -> dict[str, torch.Tensor] | None:
    motion_value = obs.get("motion", obs.get("motion_obs"))
    robot_value = obs.get("robot", obs.get("robot_obs"))
    if motion_value is None or robot_value is None:
        return None

    motion_obs = structure_motion_observation(
        motion_value,
        action_dim=action_dim,
        observation_window_lengths=observation_window_lengths,
    )
    robot_obs = structure_robot_observation(
        robot_value,
        action_dim=action_dim,
        observation_window_lengths=observation_window_lengths,
    )
    obs_parts = {
        "motion": motion_obs,
        "robot": robot_obs,
        "motion_obs": motion_obs,
        "robot_obs": robot_obs,
    }
    return obs_parts


def split_sim2sim_group_observations(
    motion_obs: torch.Tensor,
    robot_obs: torch.Tensor,
    action_dim: int,
    observation_window_lengths: Mapping[str, int] | None = None,
) -> dict[str, torch.Tensor]:
    motion_obs = structure_motion_observation(
        motion_obs,
        action_dim=action_dim,
        observation_window_lengths=observation_window_lengths,
    )
    robot_obs = structure_robot_observation(
        robot_obs,
        action_dim=action_dim,
        observation_window_lengths=observation_window_lengths,
    )
    expected_dims = infer_sim2sim_observation_dims(action_dim, observation_window_lengths)
    if motion_obs.numel() != expected_dims["motion"]:
        raise ValueError(f"Expected structured sim2sim motion dim {expected_dims['motion']}, got {motion_obs.numel()}.")
    if int(robot_obs.numel()) != expected_dims["robot"]:
        raise ValueError(f"Expected structured sim2sim robot dim {expected_dims['robot']}, got {robot_obs.numel()}.")

    parsed_terms = {}
    parsed_terms.update(
        _split_group_observation(
            "motion",
            flatten_motion_history(
                motion_obs,
                build_motion_window_layout(action_dim, _resolve_motion_window_length(observation_window_lengths)),
            )
            if motion_obs.ndim == 2
            else motion_obs,
            action_dim=action_dim,
            observation_window_lengths=observation_window_lengths,
        )
    )
    parsed_terms.update(
        _split_group_observation(
            "robot",
            flatten_robot_history(
                robot_obs,
                build_robot_window_layout(action_dim, _resolve_robot_window_length(observation_window_lengths)),
            )
            if robot_obs.ndim == 2
            else robot_obs,
            action_dim=action_dim,
            observation_window_lengths=observation_window_lengths,
        )
    )

    return {
        "motion": motion_obs,
        "robot": robot_obs,
        "motion_obs": motion_obs,
        "robot_obs": robot_obs,
        **parsed_terms,
    }


def build_sim2sim_obs_parts_from_context(context: Any) -> dict[str, torch.Tensor]:
    target_projected_gravity = _coerce_sim2sim_vector(context.target_projected_gravity, name="target_projected_gravity")
    target_joint_pos = _coerce_sim2sim_vector(context.target_joint_pos, name="target_joint_pos")
    target_joint_vel_raw = getattr(context, "target_joint_vel", None)
    target_joint_vel = (
        None
        if target_joint_vel_raw is None
        else _coerce_sim2sim_vector(target_joint_vel_raw, name="target_joint_vel")
    )
    robot_projected_gravity = _coerce_sim2sim_vector(context.projected_gravity, name="projected_gravity")
    anchor_ang_vel = _coerce_sim2sim_vector(context.anchor_ang_vel_b, name="anchor_ang_vel_b")
    robot_joint_pos = _coerce_sim2sim_vector(context.joint_pos, name="joint_pos")
    robot_joint_vel = _coerce_sim2sim_vector(context.joint_vel, name="joint_vel")
    previous_action = _coerce_sim2sim_vector(context.previous_action, name="previous_action")

    motion_obs = torch.cat((target_projected_gravity, target_joint_pos), dim=-1)
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
    obs_parts = {
        "motion": motion_obs,
        "robot": robot_obs,
        "motion_obs": motion_obs,
        "robot_obs": robot_obs,
        "target_projected_gravity": target_projected_gravity,
        "target_joint_pos": target_joint_pos,
        "robot_projected_gravity": robot_projected_gravity,
        "anchor_ang_vel": anchor_ang_vel,
        "robot_joint_pos": robot_joint_pos,
        "robot_joint_vel": robot_joint_vel,
        "previous_action": previous_action,
    }
    if target_joint_vel is not None:
        obs_parts["target_joint_vel"] = target_joint_vel
    return obs_parts


def extract_sim2sim_metrics_from_parts(obs_parts: Mapping[str, torch.Tensor]) -> dict[str, float]:
    required_keys = (
        "target_joint_pos",
        "robot_joint_pos",
        "target_projected_gravity",
        "robot_projected_gravity",
    )
    missing_keys = [key for key in required_keys if key not in obs_parts]
    if missing_keys:
        raise KeyError(f"Sim2sim observation parts are missing required keys: {missing_keys}.")

    return {
        "joint_pos_mae": torch.mean(torch.abs(obs_parts["target_joint_pos"] - obs_parts["robot_joint_pos"])).item(),
        "gravity_mae": torch.mean(
            torch.abs(obs_parts["target_projected_gravity"] - obs_parts["robot_projected_gravity"])
        ).item(),
    }


def extract_sim2sim_metrics(
    flat_obs: torch.Tensor,
    action_dim: int,
    observation_window_lengths: Mapping[str, int] | None = None,
) -> dict[str, float]:
    return extract_sim2sim_metrics_from_parts(
        parse_sim2sim_obs(
            flat_obs,
            action_dim,
            observation_window_lengths=observation_window_lengths,
        )
    )


def build_actor_obs_log_fields(actor_obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {f"actor_obs_{key}": value for key, value in actor_obs.items()}
