from __future__ import annotations

from collections.abc import Mapping

from .compat import _import_module

_OBSERVATION_SPEC = _import_module("ref2act.common.observation_spec")
ObservationGroupSpec = _OBSERVATION_SPEC.ObservationGroupSpec
ObservationNoiseSpec = _OBSERVATION_SPEC.ObservationNoiseSpec
ObservationSpec = _OBSERVATION_SPEC.ObservationSpec
ObservationTermSpec = _OBSERVATION_SPEC.ObservationTermSpec

DEFAULT_ROBOT_WINDOW_LENGTH = 4
DEFAULT_MOTION_WINDOW_LENGTH = 1
MOTION_POLICY_OBSERVATION_TERM_IDS = (
    "target_projected_gravity",
    "target_joint_pos",
    "target_joint_vel",
)
ROBOT_POLICY_OBSERVATION_TERM_IDS = (
    "projected_gravity",
    "anchor_ang_vel_b",
    "joint_pos",
    "joint_vel",
    "previous_action",
)

# Keep Isaac and MuJoCo observation layouts aligned in one place.
DEFAULT_OBSERVATION_WINDOW_LENGTHS: dict[str, int] = {
    "target_projected_gravity": 1,
    "target_joint_pos": 1,
    "target_joint_vel": 1,
    "projected_gravity": 1,
    "anchor_ang_vel_b": 1,
    "joint_pos": 1,
    "joint_vel": 1,
    "previous_action": 1,
    "priv_target_joint_pos": 1,
    "priv_target_joint_vel": 1,
    "relative_anchor_pos": 1,
    "relative_anchor_tangent_and_normal": 1,
    "relative_key_pos": 1,
    "relative_key_tangent_and_normal": 1,
    "anchor_lin_vel": 1,
    "priv_anchor_ang_vel_b": 1,
    "priv_joint_pos": 1,
    "priv_joint_vel": 1,
    "priv_previous_action": 1,
}


def _window_length(term_id: str, overrides: Mapping[str, int] | None) -> int:
    value = (overrides or {}).get(term_id, DEFAULT_OBSERVATION_WINDOW_LENGTHS.get(term_id, 1))
    value = int(value)
    if value < 1:
        raise ValueError(f"Observation window length for '{term_id}' must be positive, got {value}.")
    return value


def build_robot_policy_window_lengths(robot_window_length: int) -> dict[str, int]:
    value = int(robot_window_length)
    if value < 1:
        raise ValueError(f"Robot observation window length must be positive, got {value}.")
    return {term_id: value for term_id in ROBOT_POLICY_OBSERVATION_TERM_IDS}


def build_motion_policy_window_lengths(motion_window_length: int) -> dict[str, int]:
    value = int(motion_window_length)
    if value < 1:
        raise ValueError(f"Motion observation window length must be positive, got {value}.")
    return {term_id: value for term_id in MOTION_POLICY_OBSERVATION_TERM_IDS}


def _validate_uniform_group_lengths(
    normalized: Mapping[str, int],
    *,
    term_ids: tuple[str, ...],
    label: str,
) -> None:
    group_lengths = {term_id: normalized.get(term_id, 1) for term_id in term_ids}
    if any(term_id in normalized for term_id in term_ids):
        unique_lengths = set(group_lengths.values())
        if len(unique_lengths) != 1:
            raise ValueError(f"{label} window lengths must match across {term_ids}, got {group_lengths}.")


def normalize_observation_window_lengths(window_lengths: Mapping[str, int] | None) -> dict[str, int]:
    if window_lengths is None:
        return {}

    normalized = {}
    for term_id, value in window_lengths.items():
        normalized[str(term_id)] = _window_length(str(term_id), {str(term_id): int(value)})

    _validate_uniform_group_lengths(
        normalized,
        term_ids=ROBOT_POLICY_OBSERVATION_TERM_IDS,
        label="Robot policy observation",
    )
    _validate_uniform_group_lengths(
        normalized,
        term_ids=MOTION_POLICY_OBSERVATION_TERM_IDS,
        label="Motion policy observation",
    )

    return normalized


def resolve_observation_window_lengths(
    *,
    robot_window_length: int | None = None,
    motion_window_length: int | None = None,
    checkpoint_env: Mapping[str, object] | None = None,
) -> dict[str, int]:
    resolved_window_lengths: dict[str, int] = {}

    if checkpoint_env is not None:
        raw_window_lengths = checkpoint_env.get("observation_window_lengths")
        if raw_window_lengths is not None:
            if not isinstance(raw_window_lengths, Mapping):
                raise ValueError(
                    "Checkpoint observation_window_lengths must be a mapping, "
                    f"got {type(raw_window_lengths).__name__}."
                )
            resolved_window_lengths.update(normalize_observation_window_lengths(raw_window_lengths))

    if robot_window_length is not None:
        resolved_window_lengths.update(build_robot_policy_window_lengths(robot_window_length))
    if motion_window_length is not None:
        resolved_window_lengths.update(build_motion_policy_window_lengths(motion_window_length))

    return normalize_observation_window_lengths(resolved_window_lengths)


def build_gmtp_observation_spec(
    *,
    add_noise: bool,
    window_lengths: Mapping[str, int] | None = None,
) -> ObservationSpec:
    robot_terms = (
        ObservationTermSpec(
            id="projected_gravity",
            type="projected_gravity",
            window_length=_window_length("projected_gravity", window_lengths),
            noise=ObservationNoiseSpec(-0.05, 0.05) if add_noise else None,
        ),
        ObservationTermSpec(
            id="anchor_ang_vel_b",
            type="anchor_ang_vel_b",
            window_length=_window_length("anchor_ang_vel_b", window_lengths),
            noise=ObservationNoiseSpec(-0.3, 0.3) if add_noise else None,
        ),
        ObservationTermSpec(
            id="joint_pos",
            type="joint_pos",
            window_length=_window_length("joint_pos", window_lengths),
            noise=ObservationNoiseSpec(-0.01, 0.01) if add_noise else None,
        ),
        ObservationTermSpec(
            id="joint_vel",
            type="joint_vel",
            window_length=_window_length("joint_vel", window_lengths),
            noise=ObservationNoiseSpec(-0.5, 0.5) if add_noise else None,
        ),
        ObservationTermSpec(
            id="previous_action",
            type="previous_action",
            window_length=_window_length("previous_action", window_lengths),
        ),
    )
    return ObservationSpec(
        groups=(
            ObservationGroupSpec(
                name="motion",
                terms=(
                    ObservationTermSpec(
                        id="target_projected_gravity",
                        type="target_projected_gravity",
                        window_length=_window_length("target_projected_gravity", window_lengths),
                    ),
                    ObservationTermSpec(
                        id="target_joint_pos",
                        type="target_joint_pos",
                        window_length=_window_length("target_joint_pos", window_lengths),
                    ),
                    ObservationTermSpec(
                        id="target_joint_vel",
                        type="target_joint_vel",
                        window_length=_window_length("target_joint_vel", window_lengths),
                    ),
                ),
            ),
            ObservationGroupSpec(name="robot", terms=robot_terms),
            ObservationGroupSpec(
                name="privilege",
                terms=(
                    ObservationTermSpec(
                        id="priv_target_joint_pos",
                        type="target_joint_pos",
                        window_length=_window_length("priv_target_joint_pos", window_lengths),
                    ),
                    ObservationTermSpec(
                        id="priv_target_joint_vel",
                        type="target_joint_vel",
                        window_length=_window_length("priv_target_joint_vel", window_lengths),
                    ),
                    ObservationTermSpec(
                        id="relative_anchor_pos",
                        type="relative_anchor_pos",
                        window_length=_window_length("relative_anchor_pos", window_lengths),
                    ),
                    ObservationTermSpec(
                        id="relative_anchor_tangent_and_normal",
                        type="relative_anchor_tangent_and_normal",
                        window_length=_window_length("relative_anchor_tangent_and_normal", window_lengths),
                    ),
                    ObservationTermSpec(
                        id="relative_key_pos",
                        type="relative_key_pos",
                        window_length=_window_length("relative_key_pos", window_lengths),
                    ),
                    ObservationTermSpec(
                        id="relative_key_tangent_and_normal",
                        type="relative_key_tangent_and_normal",
                        window_length=_window_length("relative_key_tangent_and_normal", window_lengths),
                    ),
                    ObservationTermSpec(
                        id="anchor_lin_vel",
                        type="anchor_lin_vel",
                        window_length=_window_length("anchor_lin_vel", window_lengths),
                    ),
                    ObservationTermSpec(
                        id="priv_anchor_ang_vel_b",
                        type="anchor_ang_vel_b",
                        window_length=_window_length("priv_anchor_ang_vel_b", window_lengths),
                    ),
                    ObservationTermSpec(
                        id="priv_joint_pos",
                        type="joint_pos",
                        window_length=_window_length("priv_joint_pos", window_lengths),
                    ),
                    ObservationTermSpec(
                        id="priv_joint_vel",
                        type="joint_vel",
                        window_length=_window_length("priv_joint_vel", window_lengths),
                    ),
                    ObservationTermSpec(
                        id="priv_previous_action",
                        type="previous_action",
                        window_length=_window_length("priv_previous_action", window_lengths),
                    ),
                ),
            ),
        )
    )


def build_gmtp_policy_observation_spec(
    *,
    add_noise: bool,
    window_lengths: Mapping[str, int] | None = None,
) -> ObservationSpec:
    spec = build_gmtp_observation_spec(add_noise=add_noise, window_lengths=window_lengths)
    return ObservationSpec(groups=tuple(group for group in spec.groups if group.name in {"motion", "robot"}))


__all__ = [
    "DEFAULT_MOTION_WINDOW_LENGTH",
    "DEFAULT_ROBOT_WINDOW_LENGTH",
    "DEFAULT_OBSERVATION_WINDOW_LENGTHS",
    "MOTION_POLICY_OBSERVATION_TERM_IDS",
    "build_motion_policy_window_lengths",
    "ROBOT_POLICY_OBSERVATION_TERM_IDS",
    "build_robot_policy_window_lengths",
    "build_gmtp_observation_spec",
    "build_gmtp_policy_observation_spec",
    "normalize_observation_window_lengths",
    "resolve_observation_window_lengths",
]
