from isaaclab.utils import configclass
from Ref2Act.config.env_cfg import ActionMod, G1MotionTrackingEnvCfg, G1TrainingEventCfg
from Ref2Act.sampler import SamplingStrategy

@configclass
class G1JabEnv(G1MotionTrackingEnvCfg):
    expert_motion_file = "env/assests/pick_up.npz"
    episode_length_s = 20
    action_mod = ActionMod.Offset
    random_start = False
    events = None

@configclass
class G1JabTrainingEnv(G1JabEnv):
    sampling_strategy = SamplingStrategy.FailureWeighted
    random_start = False
    events = G1TrainingEventCfg()
