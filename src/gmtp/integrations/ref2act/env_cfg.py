from dataclasses import replace

from isaaclab.utils import configclass

from .compat import load_env_cfg_symbols
from .motion import DEFAULT_EXPERIMENT_MOTION_FILES, resolve_motion_files
from .observation_history import build_gmtp_observation_spec

TRACKING_QUALITY_SOFT_THRESHOLD = 1.25
TRAINING_NUM_ENVS = 4096

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


def set_robust_tracking_quality_gate_enabled(robust_tracking_cfg, enabled: bool):
    quality_gate_cfg = getattr(robust_tracking_cfg, "quality_gate", None)
    updated_quality_gate_cfg = _set_enabled_cfg(quality_gate_cfg, enabled)
    if updated_quality_gate_cfg is quality_gate_cfg:
        return robust_tracking_cfg
    return replace(robust_tracking_cfg, quality_gate=updated_quality_gate_cfg)


def _enable_terminal_quality_gate(cfg):
    if cfg is None or not hasattr(cfg, "enabled"):
        return cfg

    updates = {"enabled": True}
    if hasattr(cfg, "soft_threshold"):
        updates["soft_threshold"] = TRACKING_QUALITY_SOFT_THRESHOLD
    recovery_enter_threshold = getattr(cfg, "recovery_enter_threshold", None)
    hard_tracking_threshold = getattr(cfg, "hard_tracking_threshold", None)
    if recovery_enter_threshold is not None and hasattr(cfg, "hard_tracking_threshold"):
        if hard_tracking_threshold is None or float(hard_tracking_threshold) > float(recovery_enter_threshold):
            updates["hard_tracking_threshold"] = recovery_enter_threshold
    if hasattr(cfg, "record_soft_violations"):
        updates["record_soft_violations"] = True

    return replace(cfg, **updates)


def _configure_robust_tracking_without_fall_recovery(robust_tracking_cfg):
    updates = {}
    if hasattr(robust_tracking_cfg, "enabled"):
        updates["enabled"] = True

    quality_gate_cfg = getattr(robust_tracking_cfg, "quality_gate", None)
    enabled_quality_gate_cfg = _enable_terminal_quality_gate(quality_gate_cfg)
    if enabled_quality_gate_cfg is not quality_gate_cfg:
        updates["quality_gate"] = enabled_quality_gate_cfg

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


@configclass
class G1MultiMotionTrainingEnv(G1MultiMotionEnv):
    if hasattr(_BASE_G1_ENV_CFG, "scene"):
        scene = replace(_BASE_G1_ENV_CFG.scene, num_envs=TRAINING_NUM_ENVS)
    sampling_strategy = SamplingStrategy.FailureWeighted
    segment_source = SegmentSource.Anchor
    init_failure_bins = True
    failure_decay = 0.99
    failure_weight_uniform_mix = 0.35
    failure_weight_max_uniform_ratio = 10.0
    failure_weight_exploration_bonus = 0.10
    failure_temperature = 1.5
    events = G1TrainingEventCfg()
