from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import gymnasium

from .env_cfg import G1MultiMotionEnv, G1MultiMotionTrainingEnv, SamplingStrategy
from .motion import resolve_motion_files
from .observation_history import build_gmtp_observation_spec

ENV_NAME = "G1MotionTracking-v0"
ISAAC_EVAL_CAMERA_EYE = (2.0, 2.0, 0.5)
ISAAC_EVAL_CAMERA_LOOKAT = (0.0, 0.0, 0.0)


def _coerce_sampling_strategy(strategy):
    if strategy is None:
        return None
    if isinstance(strategy, str):
        return getattr(SamplingStrategy, strategy)
    return strategy


def _set_adaptive_sampler_enabled(cfg, enabled: bool | None) -> None:
    if enabled is None:
        return
    adaptive_sampler = getattr(cfg, "adaptive_sampler", None)
    if adaptive_sampler is None or not hasattr(adaptive_sampler, "enabled"):
        return
    cfg.adaptive_sampler = replace(adaptive_sampler, enabled=bool(enabled))


def make_training_env(
    *,
    window_lengths: Mapping[str, int] | None = None,
    sampling_strategy=None,
    adaptive_sampler_enabled: bool | None = None,
):
    cfg = G1MultiMotionTrainingEnv()
    if sampling_strategy is not None:
        cfg.sampling_strategy = _coerce_sampling_strategy(sampling_strategy)
    _set_adaptive_sampler_enabled(cfg, adaptive_sampler_enabled)
    cfg.expert_motion_file = resolve_motion_files(cfg.expert_motion_file)
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
