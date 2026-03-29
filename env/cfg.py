from isaaclab.utils import configclass
from Ref2Act.config.env_cfg import ActionMod, G1MotionTrackingEnvCfg, G1TrainingEventCfg
from Ref2Act.sampler import SamplingStrategy
from .motions import DEFAULT_EXPERIMENT_MOTION_FILES, resolve_motion_files

@configclass
class G1MultiMotionEnv(G1MotionTrackingEnvCfg):
    expert_motion_file = resolve_motion_files(DEFAULT_EXPERIMENT_MOTION_FILES)
    episode_length_s = 20
    action_mod = ActionMod.Offset
    random_start = False
    events = None

@configclass
class G1MultiMotionTrainingEnv(G1MultiMotionEnv):
    sampling_strategy = SamplingStrategy.FailureWeighted
    random_start = True
    events = G1TrainingEventCfg()


# Backward-compatible aliases for existing imports.
G1JabEnv = G1MultiMotionEnv
G1JabTrainingEnv = G1MultiMotionTrainingEnv
