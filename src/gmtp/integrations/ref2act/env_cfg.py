from dataclasses import replace

from isaaclab.utils import configclass

from .compat import load_env_cfg_symbols
from .motion import DEFAULT_EXPERIMENT_MOTION_FILES, resolve_motion_files
from .observation_history import build_gmtp_observation_spec

_REF2ACT = load_env_cfg_symbols()
G1MotionTrackingEnvCfg = _REF2ACT.G1MotionTrackingEnvCfg
G1TrainingEventCfg = _REF2ACT.G1TrainingEventCfg
SamplingStrategy = _REF2ACT.SamplingStrategy
_BASE_G1_ENV_CFG = G1MotionTrackingEnvCfg()


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


@configclass
class G1MultiMotionTrainingEnv(G1MultiMotionEnv):
    sampling_strategy = SamplingStrategy.FailureWeighted
    random_start = True
    events = G1TrainingEventCfg()
