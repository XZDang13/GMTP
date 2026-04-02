from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import torch

from gmtp.integrations.ref2act import (
    DEFAULT_EXPERIMENT_MOTION_FILES,
    infer_motion_files_from_checkpoint,
    motion_label,
    resolve_motion_files,
)
from gmtp.integrations.ref2act.mujoco import (
    DEFAULT_ANCHOR_BODY_NAME,
    DEFAULT_ROOT_NAME,
    get_mujoco_symbols,
    resolve_action_mode,
    resolve_name_override,
)
from gmtp.models import get_actor_observation, is_recurrent_actor, unpack_actor_output
from gmtp.runtime.checkpoints import load_checkpoint_v2
from gmtp.runtime.config import Sim2SimEvalConfig
from gmtp.runtime.debug import RolloutDebugLogger
from gmtp.runtime.io import build_run_paths, write_json
from gmtp.runtime.observations import (
    build_actor_obs_log_fields,
    extract_sim2sim_metrics,
    get_actor_state_log_fields,
    infer_actor_observation_dims_from_state_dict,
    infer_sim2sim_observation_dims,
    parse_sim2sim_obs,
)
from gmtp.runtime.policy import load_actor_from_checkpoint, resolve_checkpoint_stem

DEFAULT_VIDEO_HEIGHT = 720
DEFAULT_VIDEO_WIDTH = 1280


class OffscreenMujocoVideoRecorder:
    def __init__(
        self,
        *,
        mj_model: Any,
        mj_data: Any,
        output_path: str | Path,
        fps: int,
        width: int = DEFAULT_VIDEO_WIDTH,
        height: int = DEFAULT_VIDEO_HEIGHT,
    ):
        import mujoco

        self.output_path = Path(output_path).expanduser().resolve()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._renderer = mujoco.Renderer(mj_model, height=height, width=width)
        self._mj_data = mj_data
        self._writer = imageio.get_writer(self.output_path, fps=fps)

    def capture_frame(self) -> None:
        self._renderer.update_scene(self._mj_data)
        self._writer.append_data(self._renderer.render())

    def close(self) -> None:
        try:
            self._writer.close()
        finally:
            self._renderer.close()


def _infer_action_dim(checkpoint_env: Mapping[str, Any]) -> int:
    joint_names = checkpoint_env.get("joint_names")
    if joint_names:
        return len(joint_names)

    for key in (
        "action_offset",
        "action_scale",
        "joint_effort_limits",
        "joint_stiffness",
        "joint_damping",
        "joint_pos_limits",
    ):
        value = checkpoint_env.get(key)
        if value is None:
            continue
        tensor = torch.as_tensor(value)
        if tensor.ndim == 0:
            continue
        return int(tensor.shape[0])

    raise KeyError("Could not infer action dimension from checkpoint env payload.")


def _mean_metrics(metric_records: list[dict[str, float]]) -> dict[str, float]:
    if not metric_records:
        return {}
    keys = sorted(metric_records[0])
    return {
        key: float(sum(record[key] for record in metric_records) / len(metric_records))
        for key in keys
    }


def _weighted_mean_metrics(weighted_metrics: list[tuple[int, dict[str, float]]]) -> dict[str, float]:
    total_weight = sum(weight for weight, _ in weighted_metrics)
    if total_weight <= 0:
        return {}

    metric_keys: set[str] = set()
    for _, metrics in weighted_metrics:
        metric_keys.update(metrics)

    return {
        key: float(sum(weight * metrics.get(key, 0.0) for weight, metrics in weighted_metrics) / total_weight)
        for key in sorted(metric_keys)
    }


def _coerce_flat_obs(obs: Any) -> torch.Tensor:
    tensor = torch.as_tensor(obs, dtype=torch.float32, device="cpu")
    if tensor.ndim == 2 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 1:
        raise ValueError(f"Expected sim2sim bridge observation rank 1, got shape {tuple(tensor.shape)}.")
    return tensor


def _tensor_dict_to_batch(obs_parts: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "motion": obs_parts["motion"].unsqueeze(0),
        "robot": obs_parts["robot"].unsqueeze(0),
    }


def _extract_sim_state(env: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    mj_data = getattr(env, "mj_data", None)
    if mj_data is not None:
        for source_name, target_name in (
            ("ctrl", "sim_ctrl"),
            ("qpos", "sim_qpos"),
            ("qvel", "sim_qvel"),
        ):
            value = getattr(mj_data, source_name, None)
            if value is not None:
                payload[target_name] = np.asarray(value).copy()

    target_pos = getattr(env, "target_pos", None)
    if target_pos is not None:
        payload["sim_target_pos"] = torch.as_tensor(target_pos, dtype=torch.float32).clone()

    motion_time = getattr(env, "times", None)
    if motion_time is not None:
        payload["sim_motion_time"] = torch.as_tensor(motion_time, dtype=torch.float32).clone()

    return payload


class Sim2SimEvalRunner:
    def __init__(self, config: Sim2SimEvalConfig):
        self.config = config
        self.device = torch.device("cpu")
        self.checkpoint_path = Path(config.checkpoint_path).expanduser().resolve()
        self.checkpoint = load_checkpoint_v2(self.checkpoint_path)
        self.motion_files = self._resolve_motion_files()
        self.motion_name = motion_label(self.motion_files)
        self.run_paths = build_run_paths(
            config.output_root,
            "eval-sim2sim",
            resolve_checkpoint_stem(self.checkpoint_path),
        )
        write_json(self.run_paths.config_path, {"command": "eval sim2sim", "config": self.config})

        checkpoint_env = self.checkpoint.env
        self.action_dim = _infer_action_dim(checkpoint_env)
        self.sim2sim_obs_dims = infer_sim2sim_observation_dims(self.action_dim)
        self.action_mode, self.action_mode_source = resolve_action_mode(
            checkpoint_env,
            config.action_mode,
            torch.as_tensor(checkpoint_env["action_offset"], dtype=torch.float32),
            torch.as_tensor(checkpoint_env["action_scale"], dtype=torch.float32),
            torch.as_tensor(checkpoint_env["joint_pos_limits"], dtype=torch.float32),
        )
        self.root_name = resolve_name_override(
            config.root_name,
            checkpoint_env,
            ("root_name", "root_link_name"),
            DEFAULT_ROOT_NAME,
        )
        self.anchor_body_name = resolve_name_override(
            config.anchor_body_name,
            checkpoint_env,
            ("anchor_body_name",),
            DEFAULT_ANCHOR_BODY_NAME,
        )

        self.obs_dims = infer_actor_observation_dims_from_state_dict(
            self.checkpoint.model["actor"],
            config.actor_type or self.checkpoint.actor_type,
        )
        self.actor, self.actor_type, self.actor_kwargs = load_actor_from_checkpoint(
            self.checkpoint,
            obs_dims=self.obs_dims,
            action_dim=self.action_dim,
            device=self.device,
            actor_type_override=config.actor_type,
            film_res_blocks=config.film_res_blocks,
            film_attn_res_block_size=config.film_attn_res_block_size,
        )
        self.is_recurrent_actor = is_recurrent_actor(self.actor_type)
        self.actor_state: torch.Tensor | None = None
        self.actor_episode_starts: torch.Tensor | None = None
        self._reset_policy_state()

    def _resolve_motion_files(self) -> list[str]:
        if self.config.motion_files is not None:
            return resolve_motion_files(self.config.motion_files)
        return infer_motion_files_from_checkpoint(
            self.checkpoint_path,
            self.config.actor_type or self.checkpoint.actor_type,
            self.checkpoint.env,
            self.checkpoint.motion_files or DEFAULT_EXPERIMENT_MOTION_FILES,
        )

    def _reset_policy_state(self) -> None:
        if not self.is_recurrent_actor:
            self.actor_state = None
            self.actor_episode_starts = None
            return
        self.actor_state = self.actor.get_initial_state(1, device=self.device)
        self.actor_episode_starts = torch.ones(1, dtype=torch.bool, device=self.device)

    def _validate_obs_dims(self, obs_parts: dict[str, torch.Tensor]) -> None:
        if self.actor_type.value in {"vanila", "recurrent"}:
            actual_policy_dim = int(obs_parts["motion"].numel() + obs_parts["robot"].numel())
            expected_policy_dim = int(self.obs_dims["policy"])
            if actual_policy_dim != expected_policy_dim:
                raise ValueError(
                    f"Sim2sim policy observation dim mismatch: expected {expected_policy_dim}, got {actual_policy_dim}."
                )
            return

        expected_motion_dim = int(self.obs_dims["motion"])
        expected_robot_dim = int(self.obs_dims["robot"])
        actual_motion_dim = int(obs_parts["motion"].numel())
        actual_robot_dim = int(obs_parts["robot"].numel())
        if actual_motion_dim != expected_motion_dim or actual_robot_dim != expected_robot_dim:
            raise ValueError(
                "Sim2sim split observation dims mismatch: "
                f"expected motion={expected_motion_dim} robot={expected_robot_dim}, "
                f"got motion={actual_motion_dim} robot={actual_robot_dim}."
            )

    def _build_env(self, motion_file: str):
        symbols = get_mujoco_symbols()
        return symbols.MujocoEnv(
            simulation_dt=self.config.simulation_dt,
            decimation=self.config.decimation,
            kp=torch.as_tensor(self.checkpoint.env["joint_stiffness"], dtype=torch.float32),
            kd=torch.as_tensor(self.checkpoint.env["joint_damping"], dtype=torch.float32),
            effort_limits=torch.as_tensor(self.checkpoint.env["joint_effort_limits"], dtype=torch.float32),
            joint_pos_limits=torch.as_tensor(self.checkpoint.env["joint_pos_limits"], dtype=torch.float32),
            action_offset=torch.as_tensor(self.checkpoint.env["action_offset"], dtype=torch.float32),
            action_scale=torch.as_tensor(self.checkpoint.env["action_scale"], dtype=torch.float32),
            expert_motion_file=motion_file,
            root_link_name=self.root_name,
            anchor_body_name=self.anchor_body_name,
            render=self.config.render,
            action_mode=self.action_mode,
        )

    def _build_video_path(self, motion_index: int, motion_file: str) -> Path:
        return self.run_paths.videos_dir / (
            f"{resolve_checkpoint_stem(self.checkpoint_path)}_{motion_index:02d}_{Path(motion_file).stem}.mp4"
        )

    def _build_debug_prefix(self, motion_index: int, motion_file: str) -> Path:
        return self.run_paths.debug_dir / (
            f"{resolve_checkpoint_stem(self.checkpoint_path)}_{motion_index:02d}_{Path(motion_file).stem}"
        )

    def _get_video_fps(self) -> int:
        if self.config.video_fps is not None:
            return int(self.config.video_fps)
        return max(1, round(1.0 / (self.config.simulation_dt * self.config.decimation)))

    @torch.no_grad()
    def _get_action(
        self,
        actor_obs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        actor_state_before_step = self.actor_state
        if self.is_recurrent_actor:
            actor_output = self.actor(
                actor_obs,
                initial_state=self.actor_state,
                episode_starts=self.actor_episode_starts,
            )
        else:
            actor_output = self.actor(actor_obs)

        actor_step, next_state = unpack_actor_output(actor_output)
        action = actor_step.mean.squeeze(0).detach().to(device="cpu", dtype=torch.float32)
        if not torch.isfinite(action).all():
            raise RuntimeError(
                f"Non-finite action detected: min={float(action.min().item()):.6f} max={float(action.max().item()):.6f}"
            )
        if self.is_recurrent_actor:
            self.actor_state = next_state
            assert self.actor_episode_starts is not None
            self.actor_episode_starts.zero_()
        return action, get_actor_state_log_fields(actor_state_before_step)

    def _rollout_motion(
        self,
        *,
        env: Any,
        motion_index: int,
        motion_file: str,
    ) -> dict[str, Any]:
        logger = RolloutDebugLogger(self._build_debug_prefix(motion_index, motion_file))
        video_path = self._build_video_path(motion_index, motion_file) if self.config.save_video else None
        video_recorder = (
            OffscreenMujocoVideoRecorder(
                mj_model=env.mj_model,
                mj_data=env.mj_data,
                output_path=video_path,
                fps=self._get_video_fps(),
                width=DEFAULT_VIDEO_WIDTH,
                height=DEFAULT_VIDEO_HEIGHT,
            )
            if video_path is not None
            else None
        )
        error_message: str | None = None
        metric_records: list[dict[str, float]] = []
        step_count = 0
        debug_summary: dict[str, Any] = {}

        try:
            self._reset_policy_state()
            flat_obs = _coerce_flat_obs(env.reset())
            obs_parts = parse_sim2sim_obs(flat_obs, self.action_dim)
            self._validate_obs_dims(obs_parts)
            if video_recorder is not None:
                video_recorder.capture_frame()

            for step_idx in range(self.config.num_steps):
                actor_env_obs = _tensor_dict_to_batch(obs_parts)
                actor_obs = get_actor_observation(actor_env_obs, self.actor_type)
                action, actor_state_log_fields = self._get_action(actor_obs)

                flat_next_obs = _coerce_flat_obs(env.step(action))
                next_obs_parts = parse_sim2sim_obs(flat_next_obs, self.action_dim)
                metrics = extract_sim2sim_metrics(flat_next_obs, self.action_dim)
                metric_records.append(metrics)

                step_payload = {
                    "action": action,
                    **build_actor_obs_log_fields(actor_obs),
                    **{f"obs_{key}": value for key, value in next_obs_parts.items() if key not in {"motion", "robot"}},
                    **actor_state_log_fields,
                    **metrics,
                    **_extract_sim_state(env),
                }
                logger.log_step(step_idx, step_payload)
                if video_recorder is not None:
                    video_recorder.capture_frame()

                obs_parts = next_obs_parts
                step_count = step_idx + 1
        except Exception as exc:
            error_message = repr(exc)
            raise
        finally:
            if video_recorder is not None:
                video_recorder.close()

            debug_summary = {
                "checkpoint": str(self.checkpoint_path),
                "motion_file": motion_file,
                "motion_index": motion_index,
                "steps_executed": step_count,
                "steps_requested": self.config.num_steps,
                "metrics": _mean_metrics(metric_records),
                "actor_type": self.actor_type.value,
                "actor_kwargs": self.actor_kwargs,
                "action_mode": self.action_mode,
                "action_mode_source": self.action_mode_source,
                "root_name": self.root_name,
                "anchor_body_name": self.anchor_body_name,
                "video_path": str(video_path) if video_path is not None else None,
                "error": error_message,
            }
            logger.finish(debug_summary)

        debug_prefix = self._build_debug_prefix(motion_index, motion_file)
        return {
            "motion_index": motion_index,
            "motion_file": motion_file,
            "steps": step_count,
            "metrics": _mean_metrics(metric_records),
            "debug_json_path": str(debug_prefix.with_suffix(".json")),
            "debug_npz_path": str(debug_prefix.with_suffix(".npz")),
            "video_path": str(video_path) if video_path is not None else None,
        }

    def evaluate(self) -> dict[str, Any]:
        motion_summaries: list[dict[str, Any]] = []
        for motion_index, motion_file in enumerate(self.motion_files):
            env = self._build_env(motion_file)
            try:
                motion_summaries.append(
                    self._rollout_motion(
                        env=env,
                        motion_index=motion_index,
                        motion_file=motion_file,
                    )
                )
            finally:
                env.close()

        summary = {
            "checkpoint": str(self.checkpoint_path),
            "actor_type": self.actor_type.value,
            "actor_kwargs": self.actor_kwargs,
            "action_mode": self.action_mode,
            "action_mode_source": self.action_mode_source,
            "root_name": self.root_name,
            "anchor_body_name": self.anchor_body_name,
            "motion_files": list(self.motion_files),
            "motion_label": self.motion_name,
            "num_steps_per_motion": self.config.num_steps,
            "simulation_dt": self.config.simulation_dt,
            "decimation": self.config.decimation,
            "aggregate_steps": sum(int(item["steps"]) for item in motion_summaries),
            "aggregate_metrics": _weighted_mean_metrics(
                [(int(item["steps"]), dict(item["metrics"])) for item in motion_summaries if item["metrics"]]
            ),
            "motions": motion_summaries,
            "run_dir": str(self.run_paths.root),
        }
        write_json(self.run_paths.summary_path, summary)
        return summary
