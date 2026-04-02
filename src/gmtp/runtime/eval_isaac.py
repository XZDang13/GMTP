from __future__ import annotations

import re
import time
from pathlib import Path

import torch

from gmtp.integrations.ref2act import DEFAULT_EXPERIMENT_MOTION_FILES, infer_motion_files_from_checkpoint
from gmtp.models import get_actor_observation, is_recurrent_actor, unpack_actor_output
from gmtp.runtime.checkpoints import load_checkpoint_v2
from gmtp.runtime.config import IsaacEvalConfig
from gmtp.runtime.debug import RolloutDebugLogger
from gmtp.runtime.io import build_run_paths, write_json
from gmtp.runtime.observations import (
    build_actor_obs_log_fields,
    get_actor_state_log_fields,
    infer_env_observation_dims,
)
from gmtp.runtime.policy import load_actor_from_checkpoint, resolve_checkpoint_stem


class IsaacEvalRunner:
    def __init__(self, config: IsaacEvalConfig):
        self.config = config
        self.checkpoint_path = Path(config.checkpoint_path).expanduser().resolve()
        self.checkpoint = load_checkpoint_v2(self.checkpoint_path)
        self.motion_files = infer_motion_files_from_checkpoint(
            self.checkpoint_path,
            config.actor_type or self.checkpoint.actor_type,
            self.checkpoint.env,
            self.checkpoint.motion_files or DEFAULT_EXPERIMENT_MOTION_FILES,
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
        )
        self.device = self.env.unwrapped.device
        self.initial_obs, _ = self.env.reset()
        self.obs_dims = infer_env_observation_dims(self.initial_obs)
        self.actor, self.actor_type, self.actor_kwargs = load_actor_from_checkpoint(
            self.checkpoint,
            obs_dims=self.obs_dims,
            action_dim=self.cfg.action_space,
            device=self.device,
            actor_type_override=config.actor_type,
            film_res_blocks=config.film_res_blocks,
            film_attn_res_block_size=config.film_attn_res_block_size,
        )
        self.is_recurrent_actor = is_recurrent_actor(self.actor_type)
        self.actor_state = (
            self.actor.get_initial_state(self.cfg.scene.num_envs, device=self.device)
            if self.is_recurrent_actor
            else None
        )
        self.actor_episode_starts = torch.ones(self.cfg.scene.num_envs, dtype=torch.bool, device=self.device)

    def _build_log_prefix(self) -> Path:
        return self.run_paths.debug_dir / resolve_checkpoint_stem(self.checkpoint_path)

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
        actor_state: torch.Tensor | None = None,
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
        payload.update(get_actor_state_log_fields(actor_state))
        info_fields, info_metadata = cls._extract_info_log_fields(info)
        payload.update(info_fields)
        return payload, info_metadata

    @torch.no_grad()
    def get_action(self, obs_batch: dict[str, torch.Tensor], determine: bool = False):
        if self.is_recurrent_actor:
            actor_output = self.actor(
                obs_batch,
                initial_state=self.actor_state,
                episode_starts=self.actor_episode_starts,
            )
        else:
            actor_output = self.actor(obs_batch)

        actor_step, next_state = unpack_actor_output(actor_output)
        action = actor_step.mean if determine else actor_step.action
        if self.is_recurrent_actor:
            self.actor_state = next_state
        return action

    def rollout(self, obs):
        rollout_start = time.perf_counter()
        logger = RolloutDebugLogger(self._build_log_prefix())
        first_done_step = None
        reward_history: list[float] = []
        info_metadata: dict[str, object] = {}
        error_message: str | None = None
        step_count = 0

        if self.is_recurrent_actor:
            self.actor_state = self.actor.get_initial_state(self.cfg.scene.num_envs, device=self.device)
            self.actor_episode_starts = torch.ones(self.cfg.scene.num_envs, dtype=torch.bool, device=self.device)

        try:
            for step_idx in range(self.config.num_steps):
                actor_obs = get_actor_observation(obs, self.actor_type)
                actor_state_before_step = self.actor_state if self.is_recurrent_actor else None
                action = self.get_action(actor_obs, True)
                if not torch.isfinite(action).all():
                    raise RuntimeError(
                        f"Non-finite action detected at step {step_idx + 1}: "
                        f"min={action.min().item():.6f} max={action.max().item():.6f}"
                    )

                next_obs, task_reward, terminate, timeout, info = self.env.step(action)
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
                    actor_state=actor_state_before_step,
                )
                logger.log_step(step_idx, step_payload)
                info_metadata.update(step_info_metadata)
                reward_history.append(float(reward.mean().item()))
                if first_done_step is None and bool(done.any().item()):
                    first_done_step = step_idx + 1

                if self.is_recurrent_actor:
                    self.actor_episode_starts = done.to(dtype=torch.bool, device=self.device)

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
                    "num_steps_requested": self.config.num_steps,
                    "num_steps_executed": step_count,
                    "first_done_step": first_done_step,
                    "reward_mean": (sum(reward_history) / len(reward_history)) if reward_history else None,
                    "reward_min": min(reward_history) if reward_history else None,
                    "reward_max": max(reward_history) if reward_history else None,
                    "reward_sum": sum(reward_history) if reward_history else None,
                    "info_metadata": info_metadata,
                    "error": error_message,
                }
            )

        return {
            "checkpoint": str(self.checkpoint_path),
            "actor_type": self.actor_type.value,
            "actor_kwargs": self.actor_kwargs,
            "motion_files": list(self.motion_files),
            "num_steps": self.config.num_steps,
            "num_steps_executed": step_count,
            "first_done_step": first_done_step,
            "reward_mean": (sum(reward_history) / len(reward_history)) if reward_history else None,
            "reward_min": min(reward_history) if reward_history else None,
            "reward_max": max(reward_history) if reward_history else None,
            "reward_sum": sum(reward_history) if reward_history else None,
            "run_dir": str(self.run_paths.root),
        }

    def evaluate(self):
        try:
            summary = self.rollout(self.initial_obs)
        finally:
            self.env.close()
        write_json(self.run_paths.summary_path, summary)
        return summary
