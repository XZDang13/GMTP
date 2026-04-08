from gmtp.integrations.ref2act.compat import _import_module
from gmtp.integrations.ref2act.observation_history import (
    build_gmtp_observation_spec,
    build_motion_policy_window_lengths,
    build_robot_policy_window_lengths,
    resolve_observation_window_lengths,
)


def test_build_gmtp_observation_spec_matches_default_dims():
    observation_spec_mod = _import_module("ref2act.common.observation_spec")
    spec = build_gmtp_observation_spec(add_noise=True)
    layout = observation_spec_mod.ObservationLayout(joint_dim=2, action_dim=2, key_body_count=4)

    assert spec.describe(layout).group_dims == {
        "motion": 7,
        "robot": 12,
        "privilege": 61,
    }


def test_build_gmtp_observation_spec_disables_noise_for_eval():
    spec = build_gmtp_observation_spec(add_noise=False)
    assert all(term.noise is None for group in spec.groups for term in group.terms)


def test_build_gmtp_observation_spec_applies_window_length_overrides():
    observation_spec_mod = _import_module("ref2act.common.observation_spec")
    spec = build_gmtp_observation_spec(
        add_noise=False,
        window_lengths={
            "target_joint_pos": 2,
            "joint_pos": 3,
        },
    )
    layout = observation_spec_mod.ObservationLayout(joint_dim=2, action_dim=2, key_body_count=1)

    assert spec.describe(layout).group_dims == {
        "motion": 9,
        "robot": 16,
        "privilege": 34,
    }


def test_build_robot_policy_window_lengths_applies_uniform_robot_window():
    observation_spec_mod = _import_module("ref2act.common.observation_spec")
    spec = build_gmtp_observation_spec(
        add_noise=False,
        window_lengths=build_robot_policy_window_lengths(4),
    )
    layout = observation_spec_mod.ObservationLayout(joint_dim=2, action_dim=2, key_body_count=4)

    assert spec.describe(layout).group_dims == {
        "motion": 7,
        "robot": 48,
        "privilege": 61,
    }


def test_resolve_observation_window_lengths_prefers_explicit_override():
    assert resolve_observation_window_lengths(
        robot_window_length=4,
        checkpoint_env={
            "observation_window_lengths": {
                "projected_gravity": 2,
                "anchor_ang_vel_b": 2,
                "joint_pos": 2,
                "joint_vel": 2,
                "previous_action": 2,
            }
        },
    ) == build_robot_policy_window_lengths(4)


def test_build_motion_policy_window_lengths_applies_uniform_motion_window():
    observation_spec_mod = _import_module("ref2act.common.observation_spec")
    spec = build_gmtp_observation_spec(
        add_noise=False,
        window_lengths=build_motion_policy_window_lengths(3),
    )
    layout = observation_spec_mod.ObservationLayout(joint_dim=2, action_dim=2, key_body_count=4)

    assert spec.describe(layout).group_dims == {
        "motion": 21,
        "robot": 12,
        "privilege": 61,
    }


def test_resolve_observation_window_lengths_merges_group_overrides_with_checkpoint_env():
    assert resolve_observation_window_lengths(
        robot_window_length=4,
        motion_window_length=3,
        checkpoint_env={
            "observation_window_lengths": {
                "priv_joint_pos": 2,
                "target_projected_gravity": 2,
                "target_joint_pos": 2,
                "target_joint_vel": 2,
            }
        },
    ) == {
        **build_robot_policy_window_lengths(4),
        **build_motion_policy_window_lengths(3),
        "priv_joint_pos": 2,
    }
