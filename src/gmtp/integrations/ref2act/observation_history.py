from __future__ import annotations

from collections.abc import Mapping

from .compat import _import_module

_OBSERVATION_SPEC = _import_module("ref2act.common.observation_spec")
ObservationGroupSpec = _OBSERVATION_SPEC.ObservationGroupSpec
ObservationNoiseSpec = _OBSERVATION_SPEC.ObservationNoiseSpec
ObservationSpec = _OBSERVATION_SPEC.ObservationSpec
ObservationTermSpec = _OBSERVATION_SPEC.ObservationTermSpec

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


__all__ = [
    "DEFAULT_OBSERVATION_WINDOW_LENGTHS",
    "build_gmtp_observation_spec",
]
