from dataclasses import replace

from isaaclab.utils import configclass

from .compat import load_env_cfg_symbols
from .motion import DEFAULT_EXPERIMENT_MOTION_FILES, resolve_motion_files
from .observation_history import build_gmtp_observation_spec

_REF2ACT = load_env_cfg_symbols()
G1MotionTrackingEnvCfg = _REF2ACT.G1MotionTrackingEnvCfg
G1TrainingEventCfg = _REF2ACT.G1TrainingEventCfg
SamplingStrategy = _REF2ACT.SamplingStrategy
SegmentSource = _REF2ACT.SegmentSource
AdaptiveSamplerCfg = _REF2ACT.AdaptiveSamplerCfg
_BASE_G1_ENV_CFG = G1MotionTrackingEnvCfg()


def _set_enabled_cfg(cfg, enabled: bool):
    if cfg is None or not hasattr(cfg, "enabled"):
        return cfg
    return replace(cfg, enabled=enabled)


def _enable_terminal_quality_gate(cfg):
    if cfg is None or not hasattr(cfg, "enabled"):
        return cfg

    updates = {"enabled": True}
    recovery_enter_threshold = getattr(cfg, "recovery_enter_threshold", None)
    hard_tracking_threshold = getattr(cfg, "hard_tracking_threshold", None)
    if recovery_enter_threshold is not None and hasattr(cfg, "hard_tracking_threshold"):
        # Without fall recovery, recovery-needed states should terminate instead of lingering.
        if hard_tracking_threshold is None or float(hard_tracking_threshold) > float(recovery_enter_threshold):
            updates["hard_tracking_threshold"] = recovery_enter_threshold

    return replace(cfg, **updates)


def _enable_quality_gate_without_fall_recovery(robust_tracking_cfg):
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

    if not updates:
        return robust_tracking_cfg
    return replace(robust_tracking_cfg, **updates)


def _recommended_adaptive_sampler_cfg():
    if AdaptiveSamplerCfg is None:
        return None
    return AdaptiveSamplerCfg(
        enabled=False,
        warmup_samples=96,
        anchor_drop_fail_rate=0.97,
        anchor_reenable_fail_rate=0.75,
        anchor_cooldown_resets=500,
        motion_min_samples=256,
        motion_drop_fail_rate=0.98,
        motion_drop_anchor_fraction=0.8,
        motion_cooldown_resets=1000,
        probe_probability=0.01,
        mastered_fail_rate=0.05,
        mastered_probability_scale=0.25,
        min_live_motion_fraction=0.6,
        min_live_anchor_fraction=0.5,
    )


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
        robust_tracking = _enable_quality_gate_without_fall_recovery(_BASE_G1_ENV_CFG.robust_tracking)


@configclass
class G1MultiMotionTrainingEnv(G1MultiMotionEnv):
    sampling_strategy = SamplingStrategy.FailureWeighted
    segment_source = SegmentSource.Anchor
    if AdaptiveSamplerCfg is not None:
        adaptive_sampler = _recommended_adaptive_sampler_cfg()
    random_start = True
    events = G1TrainingEventCfg()
