from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import gymnasium

from .env_cfg import (
    END_EFFECTOR_TERMINATE_END_THRESHOLD,
    END_EFFECTOR_TERMINATE_START_THRESHOLD,
    G1MultiMotionEnv,
    G1MultiMotionTrainingEnv,
    set_end_effector_termination_threshold,
)
from .motion import resolve_motion_files
from .observation_history import build_gmtp_observation_spec

ENV_NAME = "G1MotionTracking-v0"
ISAAC_EVAL_CAMERA_EYE = (2.0, 2.0, 0.5)
ISAAC_EVAL_CAMERA_LOOKAT = (0.0, 0.0, 0.0)


def _policy_dt_from_cfg(cfg) -> float:
    sim_cfg = getattr(cfg, "sim", None)
    sim_dt = float(getattr(sim_cfg, "dt", 1.0 / 200.0))
    decimation = int(getattr(cfg, "decimation", 1))
    if sim_dt <= 0.0:
        raise ValueError("Ref2Act simulation dt must be positive.")
    if decimation < 1:
        raise ValueError("Ref2Act decimation must be positive.")
    return sim_dt * float(decimation)


def make_training_env(
    *,
    window_lengths: Mapping[str, int] | None = None,
    motion_files: list[str] | None = None,
    sampler_failure_warmup_steps: int | None = None,
    end_effector_termination_curriculum_enabled: bool = False,
    end_effector_termination_initial_threshold: float = END_EFFECTOR_TERMINATE_START_THRESHOLD,
    end_effector_termination_end_threshold: float = END_EFFECTOR_TERMINATE_END_THRESHOLD,
):
    cfg = G1MultiMotionTrainingEnv()
    if sampler_failure_warmup_steps is not None:
        warmup_steps = int(sampler_failure_warmup_steps)
        if warmup_steps < 0:
            raise ValueError("sampler_failure_warmup_steps must be non-negative.")
        cfg.motion_sampling_warmup_s = float(warmup_steps) * _policy_dt_from_cfg(cfg)
        cfg.motion_sampling_ramp_s = 0.0
    if end_effector_termination_curriculum_enabled and not hasattr(cfg, "termination"):
        raise ValueError("Ref2Act environment config does not expose termination.")
    if hasattr(cfg, "termination"):
        cfg.termination = set_end_effector_termination_threshold(
            cfg.termination,
            (
                end_effector_termination_initial_threshold
                if end_effector_termination_curriculum_enabled
                else end_effector_termination_end_threshold
            ),
        )
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
    end_effector_termination_threshold: float | None = None,
):
    cfg = G1MultiMotionEnv()
    cfg.expert_motion_file = resolve_motion_files(motion_files)
    cfg.scene.num_envs = 1
    cfg.training = False
    cfg.add_reset_noise = False
    cfg.random_start = False
    cfg.events = None
    if hasattr(cfg, "termination"):
        termination_threshold = (
            END_EFFECTOR_TERMINATE_END_THRESHOLD
            if end_effector_termination_threshold is None
            else float(end_effector_termination_threshold)
        )
        cfg.termination = set_end_effector_termination_threshold(
            cfg.termination,
            termination_threshold,
        )
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
