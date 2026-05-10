from dataclasses import replace

from isaaclab.utils import configclass

from .compat import load_env_cfg_symbols
from .motion import DEFAULT_EXPERIMENT_MOTION_FILES, resolve_motion_files
from .observation_history import build_gmtp_observation_spec

TRAINING_NUM_ENVS = 4096
END_EFFECTOR_TERMINATION_RULE_ID = "end_effector_position_failure"
END_EFFECTOR_TERMINATE_START_THRESHOLD = 0.25
END_EFFECTOR_TERMINATE_END_THRESHOLD = 0.15

_REF2ACT = load_env_cfg_symbols()
G1MotionTrackingEnvCfg = _REF2ACT.G1MotionTrackingEnvCfg
G1TrainingEventCfg = _REF2ACT.G1TrainingEventCfg
SamplingStrategy = _REF2ACT.SamplingStrategy
SegmentSource = _REF2ACT.SegmentSource
_BASE_G1_ENV_CFG = G1MotionTrackingEnvCfg()


def set_end_effector_termination_threshold(termination_cfg, threshold: float):
    if termination_cfg is None or not hasattr(termination_cfg, "failure_rules"):
        return termination_cfg

    failure_rules = tuple(getattr(termination_cfg, "failure_rules"))
    updated_failure_rules = []
    matched_rule = False
    for rule_cfg in failure_rules:
        rule_id = str(getattr(rule_cfg, "id", getattr(rule_cfg, "type", "")))
        if rule_id != END_EFFECTOR_TERMINATION_RULE_ID:
            updated_failure_rules.append(rule_cfg)
            continue
        if not hasattr(rule_cfg, "threshold"):
            raise ValueError(f"Termination rule {END_EFFECTOR_TERMINATION_RULE_ID!r} does not expose a threshold.")
        updated_failure_rules.append(replace(rule_cfg, threshold=float(threshold)))
        matched_rule = True

    if not matched_rule:
        raise ValueError(f"Termination rule {END_EFFECTOR_TERMINATION_RULE_ID!r} was not found.")
    return replace(termination_cfg, failure_rules=tuple(updated_failure_rules))


@configclass
class G1MultiMotionEnv(G1MotionTrackingEnvCfg):
    expert_motion_file = resolve_motion_files(DEFAULT_EXPERIMENT_MOTION_FILES)
    episode_length_s = 20
    observation = build_gmtp_observation_spec(add_noise=True)
    action = replace(_BASE_G1_ENV_CFG.action, mode="offset")
    random_start = False
    events = None
    if hasattr(_BASE_G1_ENV_CFG, "curriculum"):
        curriculum = None
    if hasattr(_BASE_G1_ENV_CFG, "termination_curriculum"):
        termination_curriculum = None
    root_link_name = "pelvis"
    anchor_body_name = "pelvis"
    if hasattr(_BASE_G1_ENV_CFG, "termination"):
        termination = set_end_effector_termination_threshold(
            _BASE_G1_ENV_CFG.termination,
            END_EFFECTOR_TERMINATE_END_THRESHOLD,
        )


@configclass
class G1MultiMotionTrainingEnv(G1MultiMotionEnv):
    if hasattr(_BASE_G1_ENV_CFG, "scene"):
        scene = replace(_BASE_G1_ENV_CFG.scene, num_envs=TRAINING_NUM_ENVS)
    sampling_strategy = SamplingStrategy.FailureWeighted
    segment_source = SegmentSource.Anchor
    init_failure_bins = True
    # Ref2Act's anchor sampler uses failure EMA counts plus this uniform floor
    # directly. The training runner overrides warmup_s for the sampler curriculum.
    weight_fail = 0.6
    weight_novel = 0.2
    cap_beta = 2.0
    adaptive_uniform_ratio = 0.1
    adaptive_alpha = 0.005
    adaptive_kernel_size = 1
    adaptive_lambda = 0.8
    motion_sampling_warmup_s = 0.0
    motion_sampling_ramp_s = 0.0
    motion_sampling_schedule = "cosine"
    events = G1TrainingEventCfg()
