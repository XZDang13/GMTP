from gmtp.integrations.ref2act.compat import _import_module
from gmtp.integrations.ref2act.observation_history import build_gmtp_observation_spec


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
