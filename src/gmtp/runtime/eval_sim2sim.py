from __future__ import annotations

import inspect
from collections.abc import Mapping
from contextlib import nullcontext
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
from gmtp.integrations.ref2act.observation_history import (
    build_gmtp_policy_observation_spec,
    resolve_observation_window_lengths,
)
from gmtp.models import get_actor_observation
from gmtp.runtime.checkpoints import load_checkpoint_v2
from gmtp.runtime.amp import AMP_DTYPE_NAME, autocast_context, resolve_amp_enabled
from gmtp.runtime.config import Sim2SimEvalConfig
from gmtp.runtime.debug import RolloutDebugLogger
from gmtp.runtime.io import build_run_paths, write_json
from gmtp.runtime.observations import (
    build_actor_obs_log_fields,
    build_sim2sim_obs_parts_from_context,
    extract_sim2sim_metrics,
    extract_sim2sim_metrics_from_parts,
    extract_sim2sim_actor_obs_from_mapping,
    infer_actor_observation_dims_from_state_dict,
    infer_sim2sim_observation_dims,
    parse_sim2sim_obs,
    split_sim2sim_group_observations,
)
from gmtp.runtime.policy import (
    load_actor_from_checkpoint,
    resolve_checkpoint_stem,
    resolve_motion_mae_checkpoint_path,
    validate_checkpoint_actor_observation_dims,
)

DEFAULT_VIDEO_HEIGHT = 720
DEFAULT_VIDEO_WIDTH = 1280
VIDEO_MACRO_BLOCK_SIZE = 16
DEFAULT_CAMERA_DISTANCE = 4.0
DEFAULT_CAMERA_AZIMUTH = -140.0
DEFAULT_CAMERA_ELEVATION = -20.0


def _floor_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 1 or value < multiple:
        return max(1, int(value))
    return max(multiple, int(value) // multiple * multiple)


def _aligned_video_candidate(
    *,
    width: int,
    height: int,
    max_width: int,
    max_height: int,
    macro_block_size: int,
) -> tuple[int, int] | None:
    if width < macro_block_size or height < macro_block_size:
        return None

    width = min(max_width, _floor_to_multiple(width, macro_block_size))
    height = min(max_height, _floor_to_multiple(height, macro_block_size))
    if width < macro_block_size or height < macro_block_size:
        return None
    if width > max_width or height > max_height:
        return None
    return width, height


def resolve_video_frame_size(
    *,
    requested_width: int,
    requested_height: int,
    macro_block_size: int = VIDEO_MACRO_BLOCK_SIZE,
) -> tuple[int, int]:
    requested_width = max(1, int(requested_width))
    requested_height = max(1, int(requested_height))
    macro_block_size = max(1, int(macro_block_size))
    return (
        _floor_to_multiple(requested_width, macro_block_size),
        _floor_to_multiple(requested_height, macro_block_size),
    )


def resolve_mujoco_renderer_size(
    mj_model: Any,
    *,
    requested_width: int,
    requested_height: int,
    macro_block_size: int = VIDEO_MACRO_BLOCK_SIZE,
) -> tuple[int, int]:
    requested_width = max(1, int(requested_width))
    requested_height = max(1, int(requested_height))
    macro_block_size = max(1, int(macro_block_size))

    vis = getattr(mj_model, "vis", None)
    global_vis = getattr(vis, "global_", None)
    max_width = max(1, int(getattr(global_vis, "offwidth", requested_width)))
    max_height = max(1, int(getattr(global_vis, "offheight", requested_height)))

    scale = min(1.0, max_width / requested_width, max_height / requested_height)
    fit_width = max(1, min(max_width, int(np.floor(requested_width * scale))))
    fit_height = max(1, min(max_height, int(np.floor(requested_height * scale))))
    if macro_block_size <= 1 or (
        fit_width % macro_block_size == 0 and fit_height % macro_block_size == 0
    ):
        return fit_width, fit_height
    if fit_width < macro_block_size or fit_height < macro_block_size:
        return fit_width, fit_height

    aspect_ratio = requested_width / requested_height
    aligned_fit_width = _floor_to_multiple(fit_width, macro_block_size)
    aligned_fit_height = _floor_to_multiple(fit_height, macro_block_size)
    candidates = [
        _aligned_video_candidate(
            width=aligned_fit_width,
            height=aligned_fit_height,
            max_width=fit_width,
            max_height=fit_height,
            macro_block_size=macro_block_size,
        ),
        _aligned_video_candidate(
            width=aligned_fit_width,
            height=round(aligned_fit_width / aspect_ratio),
            max_width=fit_width,
            max_height=fit_height,
            macro_block_size=macro_block_size,
        ),
        _aligned_video_candidate(
            width=round(aligned_fit_height * aspect_ratio),
            height=aligned_fit_height,
            max_width=fit_width,
            max_height=fit_height,
            macro_block_size=macro_block_size,
        ),
    ]
    valid_candidates = [candidate for candidate in candidates if candidate is not None]
    if not valid_candidates:
        return fit_width, fit_height

    return min(
        valid_candidates,
        key=lambda size: (
            abs((size[0] / size[1]) - aspect_ratio),
            -(size[0] * size[1]),
        ),
    )


def _resize_rgb_frame_nearest(frame: np.ndarray, *, width: int, height: int) -> np.ndarray:
    if frame.shape[0] == height and frame.shape[1] == width:
        return np.ascontiguousarray(frame)

    y_indices = np.linspace(0, frame.shape[0] - 1, height).astype(np.intp)
    x_indices = np.linspace(0, frame.shape[1] - 1, width).astype(np.intp)
    return np.ascontiguousarray(frame[y_indices[:, None], x_indices[None, :]])


def _fit_rgb_frame_to_size(frame: np.ndarray, *, width: int, height: int) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected RGB frame with shape (H, W, 3), got {frame.shape}.")

    target_aspect = width / height
    frame_height, frame_width = frame.shape[:2]
    frame_aspect = frame_width / frame_height
    if frame_aspect > target_aspect:
        crop_width = max(1, min(frame_width, int(round(frame_height * target_aspect))))
        x0 = (frame_width - crop_width) // 2
        frame = frame[:, x0 : x0 + crop_width]
    elif frame_aspect < target_aspect:
        crop_height = max(1, min(frame_height, int(round(frame_width / target_aspect))))
        y0 = (frame_height - crop_height) // 2
        frame = frame[y0 : y0 + crop_height]

    return _resize_rgb_frame_nearest(frame, width=width, height=height)


class OffscreenMujocoVideoRecorder:
    def __init__(
        self,
        *,
        mj_model: Any,
        mj_data: Any,
        env: Any | None = None,
        output_path: str | Path,
        fps: int,
        width: int = DEFAULT_VIDEO_WIDTH,
        height: int = DEFAULT_VIDEO_HEIGHT,
    ):
        import mujoco

        self.output_path = Path(output_path).expanduser().resolve()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        width, height = resolve_mujoco_renderer_size(
            mj_model,
            requested_width=width,
            requested_height=height,
        )
        self._mujoco = mujoco
        self._mj_model = mj_model
        self._width = int(width)
        self._height = int(height)
        self._renderer = self._make_renderer()
        self._mj_data = mj_data
        self._env = env
        self._camera = getattr(mujoco, "MjvCamera", lambda: None)()
        if self._camera is not None and hasattr(mujoco, "mjv_defaultFreeCamera"):
            mujoco.mjv_defaultFreeCamera(mj_model, self._camera)
        self._writer = imageio.get_writer(self.output_path, fps=fps)

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def _make_renderer(self):
        return self._mujoco.Renderer(self._mj_model, height=self._height, width=self._width)

    def _update_camera_from_env(self) -> None:
        if self._camera is None or self._env is None:
            return

        update_tracking_camera = getattr(self._env, "_update_tracking_camera", None)
        if callable(update_tracking_camera):
            update_tracking_camera(
                self._camera,
                frame_width=self._width,
                frame_height=self._height,
                mujoco_module=self._mujoco,
            )
            return

        body_id = getattr(self._env, "root_body_id", None)
        if body_id is None:
            body_id = getattr(self._env, "anchor_body_id", None)
        if body_id is None:
            body_id = getattr(self._env, "free_root_body_id", None)

        camera_type = getattr(getattr(self._mujoco, "mjtCamera", None), "mjCAMERA_TRACKING", None)
        if camera_type is not None:
            self._camera.type = camera_type
        self._camera.fixedcamid = -1
        self._camera.distance = DEFAULT_CAMERA_DISTANCE
        self._camera.azimuth = DEFAULT_CAMERA_AZIMUTH
        self._camera.elevation = DEFAULT_CAMERA_ELEVATION

        if body_id is not None:
            self._camera.trackbodyid = int(body_id)
            xpos = getattr(self._mj_data, "xpos", None)
            if xpos is not None and hasattr(self._camera, "lookat"):
                self._camera.lookat[:] = np.asarray(xpos[int(body_id)], dtype=np.float64)
            return

        qpos = getattr(self._mj_data, "qpos", None)
        if qpos is not None and hasattr(self._camera, "lookat"):
            self._camera.lookat[:] = np.asarray(qpos[:3], dtype=np.float64)

    def capture_frame(self) -> bool:
        self._update_camera_from_env()

        renderer = self._renderer
        if self._camera is None:
            renderer.update_scene(self._mj_data)
        else:
            try:
                renderer.update_scene(self._mj_data, camera=self._camera)
            except TypeError:
                renderer.update_scene(self._mj_data)
        frame = np.asarray(renderer.render(), dtype=np.uint8).copy()
        self._writer.append_data(frame)
        return True

    def close(self) -> None:
        try:
            self._writer.close()
        finally:
            if self._renderer is not None:
                self._renderer.close()
                self._renderer = None


class LiveMujocoViewerVideoRecorder:
    def __init__(
        self,
        *,
        env: Any,
        output_path: str | Path,
        fps: int,
        width: int = DEFAULT_VIDEO_WIDTH,
        height: int = DEFAULT_VIDEO_HEIGHT,
    ):
        import glfw
        import mujoco

        self.output_path = Path(output_path).expanduser().resolve()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._env = env
        self._glfw = glfw
        self._mujoco = mujoco
        self._width, self._height = resolve_video_frame_size(
            requested_width=width,
            requested_height=height,
        )
        self._writer = imageio.get_writer(self.output_path, fps=fps)
        self.closed_by_viewer = False

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def _get_live_viewer(self):
        viewer = getattr(self._env, "mj_viewer", None)
        if viewer is None or not bool(getattr(viewer, "is_alive", True)):
            self.closed_by_viewer = True
            return None
        window = getattr(viewer, "window", None)
        if window is not None and self._glfw.window_should_close(window):
            close = getattr(viewer, "close", None)
            if callable(close):
                close()
            if hasattr(self._env, "mj_viewer"):
                self._env.mj_viewer = None
            self.closed_by_viewer = True
            return None
        return viewer

    def capture_frame(self) -> bool:
        viewer = self._get_live_viewer()
        if viewer is None:
            return False

        window_width, window_height = self._glfw.get_framebuffer_size(viewer.window)
        if window_width <= 0 or window_height <= 0:
            return False
        viewer.viewport.width = int(window_width)
        viewer.viewport.height = int(window_height)

        lock = getattr(viewer, "_gui_lock", None)
        context = lock if lock is not None else nullcontext()
        with context:
            self._mujoco.mjv_updateScene(
                viewer.model,
                viewer.data,
                viewer.vopt,
                viewer.pert,
                viewer.cam,
                self._mujoco.mjtCatBit.mjCAT_ALL.value,
                viewer.scn,
            )
            for marker in getattr(viewer, "_markers", []):
                add_marker = getattr(viewer, "_add_marker_to_scene", None)
                if callable(add_marker):
                    add_marker(marker)
            self._mujoco.mjr_render(viewer.viewport, viewer.scn, viewer.ctx)
            frame = np.zeros((window_height, window_width, 3), dtype=np.uint8)
            self._mujoco.mjr_readPixels(frame, None, viewer.viewport, viewer.ctx)
            self._glfw.swap_buffers(viewer.window)
        self._glfw.poll_events()

        frame = np.flipud(frame)
        frame = _fit_rgb_frame_to_size(frame, width=self._width, height=self._height)
        self._writer.append_data(frame)

        markers = getattr(viewer, "_markers", None)
        if isinstance(markers, list):
            markers[:] = []
        apply_perturbations = getattr(viewer, "apply_perturbations", None)
        if callable(apply_perturbations):
            apply_perturbations()
        return True

    def close(self) -> None:
        self._writer.close()


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


def _get_env_obs_dict(env: Any) -> Mapping[str, Any] | None:
    getter = getattr(env, "get_obs_dict", None)
    if not callable(getter):
        return None

    try:
        parameters = inspect.signature(getter).parameters
    except (TypeError, ValueError):
        parameters = {}

    obs = getter(advance_time=False) if "advance_time" in parameters else getter()
    if not isinstance(obs, Mapping):
        raise ValueError(f"Expected env.get_obs_dict() to return a mapping, got {type(obs).__name__}.")
    return obs


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
        self.requested_amp = bool(config.use_amp)
        self.use_amp = resolve_amp_enabled(self.requested_amp, self.device)
        self.amp_dtype = AMP_DTYPE_NAME
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
        self.observation_window_lengths = resolve_observation_window_lengths(
            robot_window_length=config.robot_window_length,
            motion_window_length=config.motion_window_length,
            checkpoint_env=checkpoint_env,
        )
        resolved_motion_mae_checkpoint = resolve_motion_mae_checkpoint_path(
            self.checkpoint,
            override=config.motion_mae_encoder_checkpoint,
        )
        self.motion_mae_encoder_checkpoint = (
            None if resolved_motion_mae_checkpoint is None else str(resolved_motion_mae_checkpoint)
        )
        self.raw_obs_dims = infer_sim2sim_observation_dims(
            self.action_dim,
            observation_window_lengths=self.observation_window_lengths,
        )
        self.obs_dims = self.raw_obs_dims
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
            action_dim=self.action_dim,
            device=self.device,
            num_blocks=config.num_blocks,
            motion_encoder_type_override=config.motion_encoder_type,
            encoder_pooling_type_override=config.encoder_pooling_type,
            motion_mae_encoder_checkpoint=self.motion_mae_encoder_checkpoint,
        )
        self._print_actor_weight_details()

    def _print_actor_weight_details(self) -> None:
        actor_weights = self.checkpoint.model["actor"]
        details = [
            f"checkpoint={self.checkpoint_path}",
            f"actor_type={self.actor_type.value}",
            f"weight_tensors={len(actor_weights)}",
        ]

        if self.actor_kwargs:
            details.append(
                "actor_kwargs="
                + ",".join(f"{key}={value}" for key, value in sorted(self.actor_kwargs.items()))
            )
        if self.observation_window_lengths:
            details.append(
                "observation_window_lengths="
                + ",".join(f"{key}={value}" for key, value in sorted(self.observation_window_lengths.items()))
            )

        details.extend(
            [
                f"root_name={self.root_name}",
                f"anchor_body_name={self.anchor_body_name}",
            ]
        )
        print("Loaded actor weights:", " ".join(details), flush=True)

    def _resolve_motion_files(self) -> list[str]:
        if self.config.motion_files is not None:
            return resolve_motion_files(self.config.motion_files)
        return infer_motion_files_from_checkpoint(
            self.checkpoint_path,
            self.checkpoint.actor_type,
            self.checkpoint.env,
            self.checkpoint.motion_files or DEFAULT_EXPERIMENT_MOTION_FILES,
        )

    def _validate_obs_dims(self, obs_parts: dict[str, torch.Tensor]) -> None:
        expected_motion_dim = int(self.raw_obs_dims["motion"])
        expected_robot_dim = int(self.raw_obs_dims["robot"])
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
        try:
            init_parameters = inspect.signature(symbols.MujocoEnv).parameters
        except (TypeError, ValueError):
            init_parameters = {}
        env_kwargs = {
            "simulation_dt": self.config.simulation_dt,
            "decimation": self.config.decimation,
            "kp": torch.as_tensor(self.checkpoint.env["joint_stiffness"], dtype=torch.float32),
            "kd": torch.as_tensor(self.checkpoint.env["joint_damping"], dtype=torch.float32),
            "effort_limits": torch.as_tensor(self.checkpoint.env["joint_effort_limits"], dtype=torch.float32),
            "joint_pos_limits": torch.as_tensor(self.checkpoint.env["joint_pos_limits"], dtype=torch.float32),
            "action_offset": torch.as_tensor(self.checkpoint.env["action_offset"], dtype=torch.float32),
            "action_scale": torch.as_tensor(self.checkpoint.env["action_scale"], dtype=torch.float32),
            "expert_motion_file": motion_file,
            "root_link_name": self.root_name,
            "anchor_body_name": self.anchor_body_name,
            "render": self.config.render,
            "action_mode": self.action_mode,
        }
        if self.config.allow_unstable_init:
            if "allow_unstable_init" not in init_parameters:
                raise ValueError(
                    "Sim2sim requested unstable-init, but the installed Ref2Act MujocoEnv "
                    "does not accept the 'allow_unstable_init' constructor argument."
                )
            env_kwargs["allow_unstable_init"] = True
        observation_builder_cls = getattr(symbols, "IsaacLabMujocoObservation", None)
        if observation_builder_cls is not None:
            if "observation_builder" in init_parameters:
                env_kwargs["observation_builder"] = observation_builder_cls(
                    spec=build_gmtp_policy_observation_spec(
                        add_noise=False,
                        window_lengths=self.observation_window_lengths or None,
                    )
                )
        return symbols.MujocoEnv(**env_kwargs)

    def _extract_obs_parts(
        self,
        env: Any,
        obs: Any,
    ) -> dict[str, torch.Tensor]:
        structured_obs = obs if isinstance(obs, Mapping) else None
        if structured_obs is None:
            try:
                structured_obs = _get_env_obs_dict(env)
            except (AttributeError, TypeError, ValueError, KeyError):
                structured_obs = None

        structured_actor_obs = None
        if structured_obs is not None:
            structured_actor_obs = extract_sim2sim_actor_obs_from_mapping(
                structured_obs,
                action_dim=self.action_dim,
                observation_window_lengths=self.observation_window_lengths,
            )

        if structured_actor_obs is not None:
            context_builder = getattr(env, "_build_observation_context", None)
            if callable(context_builder):
                context_parts = build_sim2sim_obs_parts_from_context(context_builder(advance_time=False))
            else:
                context_parts = split_sim2sim_group_observations(
                    structured_actor_obs["motion"],
                    structured_actor_obs["robot"],
                    self.action_dim,
                    observation_window_lengths=self.observation_window_lengths,
                )

            context_parts["motion"] = structured_actor_obs["motion"]
            context_parts["robot"] = structured_actor_obs["robot"]
            context_parts["motion_obs"] = structured_actor_obs["motion_obs"]
            context_parts["robot_obs"] = structured_actor_obs["robot_obs"]
            return context_parts

        return parse_sim2sim_obs(
            _coerce_flat_obs(obs),
            self.action_dim,
            observation_window_lengths=self.observation_window_lengths,
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

    def _build_video_recorder(self, *, env: Any, video_path: Path, video_fps: int):
        if self.config.render and getattr(env, "mj_viewer", None) is not None:
            return LiveMujocoViewerVideoRecorder(
                env=env,
                output_path=video_path,
                fps=video_fps,
                width=DEFAULT_VIDEO_WIDTH,
                height=DEFAULT_VIDEO_HEIGHT,
            )
        return OffscreenMujocoVideoRecorder(
            mj_model=env.mj_model,
            mj_data=env.mj_data,
            env=env,
            output_path=video_path,
            fps=video_fps,
            width=DEFAULT_VIDEO_WIDTH,
            height=DEFAULT_VIDEO_HEIGHT,
        )

    def _call_without_internal_viewer_render(self, env: Any, method, *args, suppress: bool):
        viewer = getattr(env, "mj_viewer", None) if suppress else None
        if viewer is None:
            return method(*args)

        env.mj_viewer = None
        try:
            return method(*args)
        finally:
            if getattr(env, "mj_viewer", None) is None and bool(getattr(viewer, "is_alive", True)):
                env.mj_viewer = viewer

    def _viewer_is_running(self, env: Any) -> bool:
        if not self.config.render:
            return True
        viewer = getattr(env, "mj_viewer", None)
        if viewer is None:
            return False
        return bool(getattr(viewer, "is_alive", True))

    @torch.no_grad()
    def _get_action(
        self,
        actor_obs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        with autocast_context(self.device, self.use_amp):
            actor_step = self.actor(actor_obs)
        action = actor_step.mean.squeeze(0).detach().to(device="cpu", dtype=torch.float32)
        if not torch.isfinite(action).all():
            raise RuntimeError(
                f"Non-finite action detected: min={float(action.min().item()):.6f} max={float(action.max().item()):.6f}"
            )
        return action, {}

    def _rollout_motion(
        self,
        *,
        env: Any,
        motion_index: int,
        motion_file: str,
    ) -> dict[str, Any]:
        logger = RolloutDebugLogger(self._build_debug_prefix(motion_index, motion_file))
        video_path = self._build_video_path(motion_index, motion_file) if self.config.save_video else None
        video_fps = self._get_video_fps() if video_path is not None else None
        video_recorder = None
        video_width = None
        video_height = None
        if video_path is not None:
            video_recorder = self._build_video_recorder(env=env, video_path=video_path, video_fps=video_fps)
            video_width = int(getattr(video_recorder, "width", getattr(video_recorder, "_width", DEFAULT_VIDEO_WIDTH)))
            video_height = int(
                getattr(video_recorder, "height", getattr(video_recorder, "_height", DEFAULT_VIDEO_HEIGHT))
            )
        suppress_internal_viewer_render = isinstance(video_recorder, LiveMujocoViewerVideoRecorder)
        error_message: str | None = None
        metric_records: list[dict[str, float]] = []
        step_count = 0
        debug_summary: dict[str, Any] = {}

        try:
            obs_parts = self._extract_obs_parts(
                env,
                self._call_without_internal_viewer_render(
                    env,
                    env.reset,
                    suppress=suppress_internal_viewer_render,
                ),
            )
            self._validate_obs_dims(obs_parts)
            if video_recorder is not None:
                if video_recorder.capture_frame() is False:
                    return {
                        "motion_index": motion_index,
                        "motion_file": motion_file,
                        "steps": step_count,
                        "metrics": _mean_metrics(metric_records),
                        "debug_json_path": str(self._build_debug_prefix(motion_index, motion_file).with_suffix(".json")),
                        "debug_npz_path": str(self._build_debug_prefix(motion_index, motion_file).with_suffix(".npz")),
                        "video_path": str(video_path) if video_path is not None else None,
                        "video_fps": video_fps,
                        "video_width": video_width,
                        "video_height": video_height,
                    }

            for step_idx in range(self.config.num_steps):
                if not self._viewer_is_running(env):
                    break
                actor_env_obs = _tensor_dict_to_batch(obs_parts)
                actor_obs = get_actor_observation(actor_env_obs, self.actor_type)
                action, actor_log_fields = self._get_action(actor_obs)

                next_obs_parts = self._extract_obs_parts(
                    env,
                    self._call_without_internal_viewer_render(
                        env,
                        env.step,
                        action,
                        suppress=suppress_internal_viewer_render,
                    ),
                )
                metrics = extract_sim2sim_metrics_from_parts(next_obs_parts)
                metric_records.append(metrics)

                step_payload = {
                    "action": action,
                    **build_actor_obs_log_fields(actor_obs),
                    **{f"obs_{key}": value for key, value in next_obs_parts.items() if key not in {"motion", "robot"}},
                    **actor_log_fields,
                    **metrics,
                    **_extract_sim_state(env),
                }
                logger.log_step(step_idx, step_payload)
                obs_parts = next_obs_parts
                step_count = step_idx + 1
                if video_recorder is not None:
                    if video_recorder.capture_frame() is False:
                        break
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
                "motion_mae_encoder_checkpoint": self.motion_mae_encoder_checkpoint,
                "observation_window_lengths": self.observation_window_lengths,
                "amp_requested": self.requested_amp,
                "amp_enabled": self.use_amp,
                "amp_dtype": self.amp_dtype,
                "action_mode": self.action_mode,
                "action_mode_source": self.action_mode_source,
                "root_name": self.root_name,
                "anchor_body_name": self.anchor_body_name,
                "allow_unstable_init": self.config.allow_unstable_init,
                "video_path": str(video_path) if video_path is not None else None,
                "video_fps": video_fps,
                "video_width": video_width,
                "video_height": video_height,
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
            "video_fps": video_fps,
            "video_width": video_width,
            "video_height": video_height,
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
            "motion_mae_encoder_checkpoint": self.motion_mae_encoder_checkpoint,
            "observation_window_lengths": self.observation_window_lengths,
            "amp_requested": self.requested_amp,
            "amp_enabled": self.use_amp,
            "amp_dtype": self.amp_dtype,
            "action_mode": self.action_mode,
            "action_mode_source": self.action_mode_source,
            "root_name": self.root_name,
            "anchor_body_name": self.anchor_body_name,
            "allow_unstable_init": self.config.allow_unstable_init,
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
            "video_fps": self._get_video_fps() if self.config.save_video else None,
            "run_dir": str(self.run_paths.root),
        }
        write_json(self.run_paths.summary_path, summary)
        return summary
