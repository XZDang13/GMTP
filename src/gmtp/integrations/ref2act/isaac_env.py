from __future__ import annotations

import gymnasium

from .env_cfg import G1MultiMotionEnv, G1MultiMotionTrainingEnv
from .motion import resolve_motion_files


ENV_NAME = "G1MotionTracking-v0"


def make_training_env():
    cfg = G1MultiMotionTrainingEnv()
    cfg.expert_motion_file = resolve_motion_files(cfg.expert_motion_file)
    env = gymnasium.make(ENV_NAME, cfg=cfg)
    return env, cfg


def make_eval_env(
    motion_files: list[str],
    *,
    show_reference_motion: bool = False,
):
    cfg = G1MultiMotionEnv()
    cfg.expert_motion_file = resolve_motion_files(motion_files)
    cfg.scene.num_envs = 1
    cfg.training = False
    cfg.add_action_noise = False
    cfg.add_obs_noise = False
    cfg.add_reset_noise = False
    cfg.random_start = False
    cfg.events = None
    cfg.reference_motion_viewer_enabled = show_reference_motion
    env = gymnasium.make(ENV_NAME, cfg=cfg)
    return env, cfg
