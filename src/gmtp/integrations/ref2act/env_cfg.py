from dataclasses import replace

from isaaclab.utils import configclass

from .compat import load_env_cfg_symbols
from .motion import DEFAULT_EXPERIMENT_MOTION_FILES, resolve_motion_files
from .observation_history import build_gmtp_observation_spec

TRACKING_RECOVERY_PENALTY_WEIGHT = -2.0
TRAINING_NUM_ENVS = 6000

_REF2ACT = load_env_cfg_symbols()
G1MotionTrackingEnvCfg = _REF2ACT.G1MotionTrackingEnvCfg
G1TrainingEventCfg = _REF2ACT.G1TrainingEventCfg
SamplingStrategy = _REF2ACT.SamplingStrategy
SegmentSource = _REF2ACT.SegmentSource
_BASE_G1_ENV_CFG = G1MotionTrackingEnvCfg()


def _set_enabled_cfg(cfg, enabled: bool):
    if cfg is None or not hasattr(cfg, "enabled"):
        return cfg
    return replace(cfg, enabled=enabled)


def _configure_robust_tracking_without_fall_recovery(robust_tracking_cfg):
    updates = {}
    if hasattr(robust_tracking_cfg, "enabled"):
        updates["enabled"] = True

    quality_gate_cfg = getattr(robust_tracking_cfg, "quality_gate", None)
    disabled_quality_gate_cfg = _set_enabled_cfg(quality_gate_cfg, False)
    if disabled_quality_gate_cfg is not quality_gate_cfg:
        updates["quality_gate"] = disabled_quality_gate_cfg

    fall_recovery_cfg = getattr(robust_tracking_cfg, "fall_recovery", None)
    disabled_fall_recovery_cfg = _set_enabled_cfg(fall_recovery_cfg, False)
    if disabled_fall_recovery_cfg is not fall_recovery_cfg:
        updates["fall_recovery"] = disabled_fall_recovery_cfg

    fall_guard_cfg = getattr(robust_tracking_cfg, "fall_guard", None)
    disabled_fall_guard_cfg = _set_enabled_cfg(fall_guard_cfg, False)
    if disabled_fall_guard_cfg is not fall_guard_cfg:
        updates["fall_guard"] = disabled_fall_guard_cfg

    if not updates:
        return robust_tracking_cfg
    return replace(robust_tracking_cfg, **updates)


def _set_tracking_recovery_penalty(rewards_cfg, weight: float):
    terms = getattr(rewards_cfg, "terms", None)
    if terms is None:
        return rewards_cfg

    updated_terms = []
    changed = False
    for term in terms:
        is_tracking_recovery_penalty = (
            getattr(term, "id", None) == "tracking_recovery_penalty"
            or getattr(term, "type", None) == "tracking_recovery_penalty"
        )
        if is_tracking_recovery_penalty:
            if hasattr(term, "weight"):
                updated_terms.append(replace(term, weight=weight))
                changed = True
                continue
        updated_terms.append(term)

    if not changed:
        return rewards_cfg
    return replace(rewards_cfg, terms=tuple(updated_terms))


@configclass
class G1MultiMotionEnv(G1MotionTrackingEnvCfg):
    expert_motion_file = resolve_motion_files(DEFAULT_EXPERIMENT_MOTION_FILES)
    episode_length_s = 20
    observation = build_gmtp_observation_spec(add_noise=True)
    action = replace(_BASE_G1_ENV_CFG.action, mode="offset")
    random_start = False
    events = None
    root_link_name = "pelvis"
    anchor_body_name = "pelvis"
    if hasattr(_BASE_G1_ENV_CFG, "recovery"):
        recovery = _set_enabled_cfg(_BASE_G1_ENV_CFG.recovery, False)
    if hasattr(_BASE_G1_ENV_CFG, "robust_tracking"):
        robust_tracking = _configure_robust_tracking_without_fall_recovery(_BASE_G1_ENV_CFG.robust_tracking)
    if hasattr(_BASE_G1_ENV_CFG, "rewards"):
        rewards = _set_tracking_recovery_penalty(
            _BASE_G1_ENV_CFG.rewards,
            TRACKING_RECOVERY_PENALTY_WEIGHT,
        )


@configclass
class G1MultiMotionTrainingEnv(G1MultiMotionEnv):
    if hasattr(_BASE_G1_ENV_CFG, "scene"):
        scene = replace(_BASE_G1_ENV_CFG.scene, num_envs=TRAINING_NUM_ENVS)
    sampling_strategy = SamplingStrategy.FailureWeighted
    segment_source = SegmentSource.Anchor
    init_failure_bins = True
    failure_decay = 0.999
    failure_weight_uniform_mix = 0.05
    failure_weight_max_uniform_ratio = 8.0
    failure_weight_exploration_bonus = 0.10
    failure_temperature = 0.75
    events = G1TrainingEventCfg()
