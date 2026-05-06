from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import gymnasium

from .env_cfg import G1MultiMotionEnv, G1MultiMotionTrainingEnv, set_robust_tracking_quality_gate_enabled
from .motion import resolve_motion_files
from .observation_history import build_gmtp_observation_spec

ENV_NAME = "G1MotionTracking-v0"
ISAAC_EVAL_CAMERA_EYE = (2.0, 2.0, 0.5)
ISAAC_EVAL_CAMERA_LOOKAT = (0.0, 0.0, 0.0)


def make_training_env(
    *,
    window_lengths: Mapping[str, int] | None = None,
    motion_files: list[str] | None = None,
    disable_quality_gate: bool = False,
):
    cfg = G1MultiMotionTrainingEnv()
    if disable_quality_gate and hasattr(cfg, "robust_tracking"):
        cfg.robust_tracking = set_robust_tracking_quality_gate_enabled(cfg.robust_tracking, False)
    motion_file_inputs = motion_files if motion_files is not None else cfg.expert_motion_file
    cfg.expert_motion_file = resolve_motion_files(motion_file_inputs)
    cfg.observation = build_gmtp_observation_spec(add_noise=True, window_lengths=window_lengths)
    env = gymnasium.make(ENV_NAME, cfg=cfg)
    return env, cfg


def make_eval_env(
    motion_files: list[str],
    *,
    show_reference_motion: bool = False,
    window_lengths: Mapping[str, int] | None = None,
    render_mode: str | None = None,
):
    cfg = G1MultiMotionEnv()
    cfg.expert_motion_file = resolve_motion_files(motion_files)
    cfg.scene.num_envs = 1
    cfg.training = False
    cfg.add_reset_noise = False
    cfg.random_start = False
    cfg.events = None
    cfg.observation = build_gmtp_observation_spec(add_noise=False, window_lengths=window_lengths)
    cfg.action = replace(cfg.action, buffer_length=1, latency_range=None, noise_scale=0.0)
    cfg.reference_motion_viewer_enabled = show_reference_motion
    cfg.viewer.origin_type = "asset_body"
    cfg.viewer.asset_name = "robot"
    cfg.viewer.body_name = cfg.root_link_name
    cfg.viewer.eye = ISAAC_EVAL_CAMERA_EYE
    cfg.viewer.lookat = ISAAC_EVAL_CAMERA_LOOKAT
    env = gymnasium.make(ENV_NAME, cfg=cfg, render_mode=render_mode)
    return env, cfg
