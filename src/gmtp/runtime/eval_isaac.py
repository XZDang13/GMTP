from __future__ import annotations

import re
import time
from pathlib import Path

import gymnasium as gym
import torch

from gmtp.integrations.ref2act import DEFAULT_EXPERIMENT_MOTION_FILES, infer_motion_files_from_checkpoint, motion_label
from gmtp.integrations.ref2act.observation_history import resolve_observation_window_lengths
from gmtp.models import get_actor_observation
from gmtp.runtime.checkpoints import load_checkpoint_v2
from gmtp.runtime.amp import AMP_DTYPE_NAME, autocast_context, normalize_device, resolve_amp_enabled
from gmtp.runtime.config import IsaacEvalConfig
from gmtp.runtime.debug import RolloutDebugLogger
from gmtp.runtime.io import build_run_paths, sanitize_name, write_json
from gmtp.runtime.observations import (
    build_actor_obs_log_fields,
    infer_actor_observation_dims_from_state_dict,
    infer_env_observation_dims,
    structure_env_observation,
)
from gmtp.runtime.policy import (
    load_actor_from_checkpoint,
    resolve_checkpoint_stem,
    resolve_motion_mae_checkpoint_path,
    validate_checkpoint_actor_observation_dims,
)


class IsaacEvalRunner:
    def __init__(self, config: IsaacEvalConfig):
        self.config = config
        self.checkpoint_path = Path(config.checkpoint_path).expanduser().resolve()
        self.checkpoint = load_checkpoint_v2(self.checkpoint_path)
        self.motion_files = infer_motion_files_from_checkpoint(
            self.checkpoint_path,
            self.checkpoint.actor_type,
            self.checkpoint.env,
            self.checkpoint.motion_files or DEFAULT_EXPERIMENT_MOTION_FILES,
        )
        self.observation_window_lengths = resolve_observation_window_lengths(
            robot_window_length=config.robot_window_length,
            motion_window_length=config.motion_window_length,
            checkpoint_env=self.checkpoint.env,
        )
        self.run_paths = build_run_paths(
            config.output_root,
            "eval-isaac",
            resolve_checkpoint_stem(self.checkpoint_path),
        )
        write_json(self.run_paths.config_path, {"command": "eval isaac", "config": self.config})

        from gmtp.integrations.ref2act.isaac_env import make_eval_env

        self.env, self.cfg = make_eval_env(
            self.motion_files,
            show_reference_motion=config.show_reference_motion,
            window_lengths=self.observation_window_lengths,
            render_mode="rgb_array" if config.save_video else None,
        )
        self.video_path = self._build_video_path() if config.save_video else None
        if self.video_path is not None:
            self.env = self._wrap_env_for_video(self.env, self.video_path)
        self._video_recording_stopped = self.video_path is None
        self.device = normalize_device(self.env.unwrapped.device)
        self.requested_amp = bool(config.use_amp)
        self.use_amp = resolve_amp_enabled(self.requested_amp, self.device)
        self.amp_dtype = AMP_DTYPE_NAME
        resolved_motion_mae_checkpoint = resolve_motion_mae_checkpoint_path(
            self.checkpoint,
            override=config.motion_mae_encoder_checkpoint,
        )
        self.motion_mae_encoder_checkpoint = (
            None if resolved_motion_mae_checkpoint is None else str(resolved_motion_mae_checkpoint)
        )
        self.initial_obs, _ = self.env.reset()
        self._configure_tracking_camera()
        self.initial_obs = structure_env_observation(
            self.initial_obs,
            action_dim=self.cfg.action_space,
            observation_window_lengths=self.observation_window_lengths,
        )
        self.raw_obs_dims = infer_env_observation_dims(self.initial_obs)
        self.obs_dims = self.raw_obs_dims
        checkpoint_obs_dims = infer_actor_observation_dims_from_state_dict(
            self.checkpoint.model["actor"],
            self.checkpoint.actor_type,
        )
        validate_checkpoint_actor_observation_dims(
            self.checkpoint,
            checkpoint_obs_dims=checkpoint_obs_dims,
            runtime_obs_dims=self.obs_dims,
            motion_mae_encoder_checkpoint=self.motion_mae_encoder_checkpoint,
        )
        self.actor, self.actor_type, self.actor_kwargs = load_actor_from_checkpoint(
            self.checkpoint,
            obs_dims=self.obs_dims,
            action_dim=self.cfg.action_space,
            device=self.device,
            num_blocks=config.num_blocks,
            motion_encoder_type_override=config.motion_encoder_type,
            motion_mae_encoder_checkpoint=self.motion_mae_encoder_checkpoint,
        )

    def _configure_tracking_camera(self) -> None:
        controller = getattr(self.env.unwrapped, "viewport_camera_controller", None)
        if controller is None:
            return

        try:
            controller.update_view_to_asset_body("robot", self.cfg.root_link_name)
        except (AttributeError, KeyError, ValueError):
            controller.update_view_to_asset_root("robot")
        controller.update_view_location()

    def _build_video_name_prefix(self) -> str:
        return sanitize_name(f"{resolve_checkpoint_stem(self.checkpoint_path)}_{motion_label(self.motion_files)}")

    def _build_video_path(self) -> Path:
        return self.run_paths.videos_dir / f"{self._build_video_name_prefix()}-episode-0.mp4"

    def _get_video_fps(self) -> int:
        if self.config.video_fps is not None:
            return int(self.config.video_fps)
        return max(1, round(1.0 / float(self.cfg.sim.dt * self.cfg.decimation)))

    def _wrap_env_for_video(self, env, video_path: Path):
        return gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_path.parent),
            episode_trigger=lambda episode_id: episode_id == 0,
            name_prefix=video_path.stem.removesuffix("-episode-0"),
            fps=self._get_video_fps(),
            disable_logger=True,
        )

    def _build_log_prefix(self) -> Path:
        return self.run_paths.debug_dir / resolve_checkpoint_stem(self.checkpoint_path)

    def _stop_video_recording_after_first_episode(self) -> None:
        if self._video_recording_stopped:
            return

        stop_recording = getattr(self.env, "stop_recording", None)
        is_recording = getattr(self.env, "recording", False)
        recorded_frames = getattr(self.env, "recorded_frames", None)
        if isinstance(recorded_frames, list) and recorded_frames:
            recorded_frames.pop()
        if callable(stop_recording) and is_recording:
            stop_recording()
        self._video_recording_stopped = True

    @classmethod
    def _extract_info_log_fields(
        cls,
        info: dict,
        prefix: str = "info",
    ) -> tuple[dict[str, object], dict[str, object]]:
        log_fields: dict[str, object] = {}
        metadata: dict[str, object] = {}

        for key, value in info.items():
            safe_key = re.sub(r"[^0-9A-Za-z_]+", "_", str(key)).strip("_")
            field_name = f"{prefix}_{safe_key}" if safe_key else prefix

            if isinstance(value, dict):
                nested_fields, nested_metadata = cls._extract_info_log_fields(value, field_name)
                log_fields.update(nested_fields)
                metadata.update(nested_metadata)
                continue

            normalized, error = RolloutDebugLogger._normalize_value(value)
            if error is not None or normalized is None:
                metadata[field_name] = {
                    "reason": error or "normalize_failed",
                    "value": repr(value),
                }
                continue

            if normalized.ndim > 1:
                metadata[field_name] = {
                    "reason": f"ndim_gt_1:{normalized.ndim}",
                    "shape": list(normalized.shape),
                }
                continue

            log_fields[field_name] = normalized

        return log_fields, metadata

    @classmethod
    def _build_debug_step_payload(
        cls,
        obs: dict[str, torch.Tensor],
        actor_obs: dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: torch.Tensor,
        terminate: torch.Tensor,
        timeout: torch.Tensor,
        info: dict,
    ) -> tuple[dict[str, object], dict[str, object]]:
        done = terminate | timeout
        payload: dict[str, object] = {
            "obs_motion": obs["motion"],
            "obs_robot": obs["robot"],
            "obs_privilege": obs["privilege"],
            "action": action,
            "reward": reward,
            "terminate": terminate,
            "timeout": timeout,
            "done": done,
        }
        payload.update(build_actor_obs_log_fields(actor_obs))
        info_fields, info_metadata = cls._extract_info_log_fields(info)
        payload.update(info_fields)
        return payload, info_metadata

    @torch.no_grad()
    def get_action(self, obs_batch: dict[str, torch.Tensor], determine: bool = False):
        with autocast_context(self.device, self.use_amp):
            actor_step = self.actor(obs_batch)
        action = actor_step.mean if determine else actor_step.action
        return action.to(dtype=torch.float32)

    def rollout(self, obs):
        rollout_start = time.perf_counter()
        logger = RolloutDebugLogger(self._build_log_prefix())
        first_done_step = None
        reward_history: list[float] = []
        info_metadata: dict[str, object] = {}
        error_message: str | None = None
        step_count = 0

        try:
            for step_idx in range(self.config.num_steps):
                actor_obs = get_actor_observation(obs, self.actor_type)
                action = self.get_action(actor_obs, True)
                if not torch.isfinite(action).all():
                    raise RuntimeError(
                        f"Non-finite action detected at step {step_idx + 1}: "
                        f"min={action.min().item():.6f} max={action.max().item():.6f}"
                    )

                next_obs, task_reward, terminate, timeout, info = self.env.step(action)
                next_obs = structure_env_observation(
                    next_obs,
                    action_dim=self.cfg.action_space,
                    observation_window_lengths=self.observation_window_lengths,
                )
                reward = task_reward
                if not torch.isfinite(reward).all():
                    raise RuntimeError(
                        f"Non-finite reward detected at step {step_idx + 1}: "
                        f"min={reward.min().item():.6f} max={reward.max().item():.6f}"
                    )
                done = terminate | timeout

                step_payload, step_info_metadata = self._build_debug_step_payload(
                    obs,
                    actor_obs,
                    action,
                    reward,
                    terminate,
                    timeout,
                    info,
                )
                logger.log_step(step_idx, step_payload)
                info_metadata.update(step_info_metadata)
                reward_history.append(float(reward.mean().item()))
                if first_done_step is None and bool(done.any().item()):
                    first_done_step = step_idx + 1
                    self._stop_video_recording_after_first_episode()

                if self.config.progress_interval and (
                    (step_idx + 1) % self.config.progress_interval == 0 or step_idx + 1 == self.config.num_steps
                ):
                    elapsed = time.perf_counter() - rollout_start
                    print(
                        f"Progress {step_idx + 1}/{self.config.num_steps} elapsed={elapsed:.2f}s "
                        f"reward_mean={reward.mean().item():.6f}",
                        flush=True,
                    )

                obs = next_obs
                step_count = step_idx + 1
        except Exception as exc:
            error_message = repr(exc)
            raise
        finally:
            logger.finish(
                {
                    "checkpoint": str(self.checkpoint_path),
                    "actor_type": self.actor_type.value,
                    "actor_kwargs": self.actor_kwargs,
                    "motion_files": list(self.motion_files),
                    "motion_mae_encoder_checkpoint": self.motion_mae_encoder_checkpoint,
                    "observation_window_lengths": self.observation_window_lengths,
                    "num_steps_requested": self.config.num_steps,
                    "num_steps_executed": step_count,
                    "amp_requested": self.requested_amp,
                    "amp_enabled": self.use_amp,
                    "amp_dtype": self.amp_dtype,
                    "first_done_step": first_done_step,
                    "reward_mean": (sum(reward_history) / len(reward_history)) if reward_history else None,
                    "reward_min": min(reward_history) if reward_history else None,
                    "reward_max": max(reward_history) if reward_history else None,
                    "reward_sum": sum(reward_history) if reward_history else None,
                    "video_path": str(self.video_path) if self.video_path is not None else None,
                    "video_fps": self._get_video_fps() if self.video_path is not None else None,
                    "info_metadata": info_metadata,
                    "error": error_message,
                }
            )

        return {
            "checkpoint": str(self.checkpoint_path),
            "actor_type": self.actor_type.value,
            "actor_kwargs": self.actor_kwargs,
            "motion_files": list(self.motion_files),
            "motion_mae_encoder_checkpoint": self.motion_mae_encoder_checkpoint,
            "observation_window_lengths": self.observation_window_lengths,
            "num_steps": self.config.num_steps,
            "num_steps_executed": step_count,
            "amp_requested": self.requested_amp,
            "amp_enabled": self.use_amp,
            "amp_dtype": self.amp_dtype,
            "first_done_step": first_done_step,
            "reward_mean": (sum(reward_history) / len(reward_history)) if reward_history else None,
            "reward_min": min(reward_history) if reward_history else None,
            "reward_max": max(reward_history) if reward_history else None,
            "reward_sum": sum(reward_history) if reward_history else None,
            "video_path": str(self.video_path) if self.video_path is not None else None,
            "video_fps": self._get_video_fps() if self.video_path is not None else None,
            "run_dir": str(self.run_paths.root),
        }

    def evaluate(self):
        try:
            summary = self.rollout(self.initial_obs)
        finally:
            self.env.close()
        write_json(self.run_paths.summary_path, summary)
        return summary
