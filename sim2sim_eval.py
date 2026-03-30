import argparse
import re
from collections import defaultdict
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

from RLAlg.nn.steps import StochasticContinuousPolicyStep
from Ref2Act.sim2sim import MujocoEnv, quat_rotate_inverse

from debug_log import RolloutDebugLogger
from env.motions import DEFAULT_EXPERIMENT_MOTION_FILES, motion_label, resolve_motion_files
from model.actor import AdaINActor, AdaINResActor, RecurrentActor, SplitEncoderActor, VanilaActor


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MOTION_FILES = resolve_motion_files(DEFAULT_EXPERIMENT_MOTION_FILES)
DEFAULT_CAMERA_DISTANCE = 3.0
DEFAULT_CAMERA_AZIMUTH = 135.0
DEFAULT_CAMERA_ELEVATION = -20.0
DEFAULT_VIDEO_WIDTH = 1280
DEFAULT_VIDEO_HEIGHT = 720
DEFAULT_ROOT_NAME = "torso_link"
DEFAULT_ANCHOR_BODY_NAME = "torso_link"


def _quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat((quat[:1], -quat[1:]), dim=0)


def _quat_inverse(quat: torch.Tensor) -> torch.Tensor:
    return _quat_conjugate(quat) / torch.dot(quat, quat)


def _quat_mul(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = lhs
    w2, x2, y2, z2 = rhs
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )
    )


def _quat_rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat_w = quat[0]
    quat_vec = quat[1:4]
    a = vec * (2.0 * quat_w**2 - 1.0)
    b = torch.cross(quat_vec, vec, dim=-1) * quat_w * 2.0
    c = quat_vec * (torch.dot(quat_vec, vec)) * 2.0
    return a + b + c


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a trained policy in Ref2Act's MuJoCo sim2sim environment.")
    parser.add_argument("--checkpoint", default="final.pth")
    parser.add_argument(
        "--actor-type",
        default=None,
        help="Override actor architecture for checkpoints without actor metadata: vanila, recurrent, split_encoder, adain, or adain_res.",
    )
    parser.add_argument(
        "--motion-file",
        nargs="+",
        default=None,
        help="Reference motion .npz file(s) used by sim2sim. Defaults to checkpoint metadata or the walk/runing/jump experiment set.",
    )
    parser.add_argument("--num-steps", type=int, default=2000)
    parser.add_argument("--simulation-dt", type=float, default=1 / 200)
    parser.add_argument("--decimation", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--action-mode",
        default=None,
        help="Override checkpoint action mode: absolute, median, offset, or residual.",
    )
    parser.add_argument(
        "--root-name",
        default=DEFAULT_ROOT_NAME,
        help="Reference root link used to initialize the MuJoCo free body. Defaults to Ref2Act's current G1 root link.",
    )
    parser.add_argument(
        "--anchor-body-name",
        default=DEFAULT_ANCHOR_BODY_NAME,
        help="Reference/robot body used for policy observations. Defaults to Ref2Act's current G1 anchor body.",
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--random-start",
        action="store_true",
        help="Reset each replay from a uniformly sampled time within the motion clip instead of t=0.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for random-start sampling.",
    )
    parser.add_argument(
        "--camera-track-body",
        default=None,
        help="MuJoCo body name to follow. Defaults to --anchor-body-name.",
    )
    parser.add_argument("--camera-distance", type=float, default=DEFAULT_CAMERA_DISTANCE)
    parser.add_argument("--camera-azimuth", type=float, default=DEFAULT_CAMERA_AZIMUTH)
    parser.add_argument("--camera-elevation", type=float, default=DEFAULT_CAMERA_ELEVATION)
    parser.add_argument(
        "--video-dir",
        default=None,
        help="Optional output directory for replay videos. Saves one MP4 per motion file.",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=None,
        help="Replay video FPS. Defaults to the sim2sim policy rate.",
    )
    parser.add_argument("--video-width", type=int, default=DEFAULT_VIDEO_WIDTH)
    parser.add_argument("--video-height", type=int, default=DEFAULT_VIDEO_HEIGHT)
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Optional directory to save rollout debug logs as NPZ+JSON.",
    )
    return parser


class ReplayCameraRecorder:
    @staticmethod
    def _fit_video_size(
        requested_width: int,
        requested_height: int,
        max_width: int,
        max_height: int,
    ) -> tuple[int, int]:
        if requested_width <= max_width and requested_height <= max_height:
            return requested_width, requested_height

        scale = min(max_width / requested_width, max_height / requested_height)
        width = max(1, int(requested_width * scale))
        height = max(1, int(requested_height * scale))
        return width, height

    def __init__(
        self,
        env: MujocoEnv,
        track_body_name: str,
        camera_distance: float,
        camera_azimuth: float,
        camera_elevation: float,
        video_path: Path | None = None,
        video_width: int = DEFAULT_VIDEO_WIDTH,
        video_height: int = DEFAULT_VIDEO_HEIGHT,
        video_fps: int = 30,
    ):
        self.env = env
        self.body_id = self._resolve_body_id(track_body_name)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.camera.fixedcamid = -1
        self.camera.trackbodyid = self.body_id
        self.camera.distance = camera_distance
        self.camera.azimuth = camera_azimuth
        self.camera.elevation = camera_elevation
        self.renderer: mujoco.Renderer | None = None
        self.writer = None
        self.video_path = video_path
        self.video_size = (video_width, video_height)

        self._sync_camera_pose()
        self._apply_to_viewer()

        if self.video_path is not None:
            max_width = int(self.env.mj_model.vis.global_.offwidth)
            max_height = int(self.env.mj_model.vis.global_.offheight)
            self.video_size = self._fit_video_size(video_width, video_height, max_width, max_height)
            self.video_path.parent.mkdir(parents=True, exist_ok=True)
            self.renderer = mujoco.Renderer(
                self.env.mj_model,
                height=self.video_size[1],
                width=self.video_size[0],
            )
            self.writer = imageio.get_writer(
                self.video_path,
                fps=video_fps,
                codec="libx264",
                macro_block_size=None,
            )

    def _resolve_body_id(self, track_body_name: str) -> int:
        body_id = mujoco.mj_name2id(
            self.env.mj_model,
            mujoco.mjtObj.mjOBJ_BODY,
            track_body_name,
        )
        if body_id == -1:
            raise ValueError(f"MuJoCo body '{track_body_name}' does not exist in the robot model.")
        return body_id

    def _sync_camera_pose(self) -> None:
        self.camera.lookat[:] = self.env.mj_data.xpos[self.body_id]

    def _apply_to_viewer(self) -> None:
        if self.env.mj_viewer is None:
            return

        self._sync_camera_pose()
        viewer_camera = self.env.mj_viewer.cam
        viewer_camera.type = self.camera.type
        viewer_camera.fixedcamid = self.camera.fixedcamid
        viewer_camera.trackbodyid = self.camera.trackbodyid
        viewer_camera.distance = self.camera.distance
        viewer_camera.azimuth = self.camera.azimuth
        viewer_camera.elevation = self.camera.elevation
        viewer_camera.lookat[:] = self.camera.lookat

    def render_viewer(self) -> None:
        self._apply_to_viewer()
        if self.env.mj_viewer is not None and self.env.mj_viewer.is_alive:
            self.env.mj_viewer.render()

    def capture_frame(self) -> None:
        if self.renderer is None or self.writer is None:
            return

        self._sync_camera_pose()
        self.renderer.update_scene(self.env.mj_data, camera=self.camera)
        self.writer.append_data(self.renderer.render())

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


class Sim2SimEvaluator:
    @staticmethod
    def _normalize_actor_type(actor_type: str | None) -> str:
        normalized = (actor_type or "vanila").lower().replace("-", "_")
        alias_map = {
            "vanila": "vanila",
            "vanilla": "vanila",
            "recurrent": "recurrent",
            "gru": "recurrent",
            "vanila_gru": "recurrent",
            "vanilla_gru": "recurrent",
            "split": "split_encoder",
            "split_encoder": "split_encoder",
            "adain": "adain",
            "adain_res": "adain_res",
            "adainres": "adain_res",
        }
        try:
            return alias_map[normalized]
        except KeyError as exc:
            raise ValueError(f"Unsupported actor type '{actor_type}'.") from exc

    @staticmethod
    def _normalize_adain_res_blocks(num_blocks: int) -> int:
        if num_blocks < 1:
            raise ValueError(f"adain_res_blocks must be positive, got {num_blocks}.")
        return num_blocks

    @staticmethod
    def _is_concat_actor(actor_type: str) -> bool:
        return actor_type in {"vanila", "recurrent"}

    @staticmethod
    def _is_recurrent_actor(actor_type: str) -> bool:
        return actor_type == "recurrent"

    @staticmethod
    def _unpack_actor_output(actor_output):
        if isinstance(actor_output, tuple):
            if len(actor_output) != 2:
                raise ValueError(
                    f"Expected recurrent actor output to be (step, next_state), got tuple of length {len(actor_output)}."
                )
            return actor_output
        return actor_output, None

    @staticmethod
    def _normalize_action_mode(action_mode: object | None) -> str:
        normalized = str(action_mode or "absolute").split(".")[-1].lower().replace("-", "_")
        alias_map = {
            "absolute": "absolute",
            "median": "median",
            "offset": "offset",
            "residual": "residual",
        }
        try:
            return alias_map[normalized]
        except KeyError as exc:
            raise ValueError(f"Unsupported action mode '{action_mode}'.") from exc

    @staticmethod
    def _infer_adain_res_blocks(actor_weights: dict[str, torch.Tensor]) -> int:
        block_pattern = re.compile(r"^block_(\d+)\.")
        block_ids = [
            int(match.group(1))
            for key in actor_weights
            if (match := block_pattern.match(key)) is not None
        ]
        return max(block_ids, default=5)

    @staticmethod
    def _infer_recurrent_actor_kwargs(actor_weights: dict[str, torch.Tensor]) -> dict[str, int]:
        layer_pattern = re.compile(r"^gru\.gru\.weight_ih_l(\d+)$")
        layer_ids = [
            int(match.group(1))
            for key in actor_weights
            if (match := layer_pattern.match(key)) is not None
        ]
        if not layer_ids:
            raise ValueError("Could not infer recurrent actor configuration from checkpoint weights.")

        hidden_size = int(actor_weights["gru.gru.weight_hh_l0"].shape[1])
        return {
            "hidden_size": hidden_size,
            "num_layers": max(layer_ids) + 1,
        }

    @staticmethod
    def _resolve_existing_path(path_str: str) -> Path:
        path = Path(path_str).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        return path

    @staticmethod
    def _resolve_output_path(path_str: str) -> Path:
        path = Path(path_str).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return path

    @staticmethod
    def _build_actor_obs_log_fields(actor_obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {f"actor_obs_{key}": value for key, value in actor_obs.items()}

    @staticmethod
    def _get_actor_state_log_fields(actor_state: torch.Tensor | None) -> dict[str, float]:
        if actor_state is None:
            return {}
        actor_state = actor_state.detach()
        return {
            "actor_state_l2": float(actor_state.norm().item()),
            "actor_state_max_abs": float(actor_state.abs().max().item()),
        }

    @classmethod
    def _infer_motion_files(
        cls,
        checkpoint_path: Path,
        actor_type: str,
        checkpoint_weights: dict,
        motion_files: list[str] | None,
    ) -> list[str]:
        if motion_files is not None:
            return resolve_motion_files(motion_files)

        checkpoint_motion_files = checkpoint_weights.get("motion_files")
        if checkpoint_motion_files:
            try:
                return resolve_motion_files(checkpoint_motion_files)
            except FileNotFoundError:
                pass

        checkpoint_motion_names = checkpoint_weights.get("motion_names")
        if checkpoint_motion_names:
            return resolve_motion_files(checkpoint_motion_names)

        marker = f"_{actor_type}_"
        checkpoint_stem = checkpoint_path.stem
        if marker in checkpoint_stem:
            motion_name = checkpoint_stem.rsplit(marker, 1)[0]
            candidate = PROJECT_ROOT / "env" / "assests" / f"{motion_name}.npz"
            if candidate.exists():
                return [str(candidate.resolve())]

        return list(DEFAULT_MOTION_FILES)

    @staticmethod
    def _infer_observation_dims(actor_weights: dict[str, torch.Tensor], actor_type: str) -> dict[str, int]:
        if Sim2SimEvaluator._is_concat_actor(actor_type):
            policy_dim = actor_weights["normlizer.mean"].shape[0]
            return {"policy": policy_dim}

        motion_dim = actor_weights["motion_obs_normlizer.mean"].shape[0]
        robot_dim = actor_weights["robot_obs_normlizer.mean"].shape[0]
        return {
            "motion": motion_dim,
            "robot": robot_dim,
            "policy": motion_dim + robot_dim,
        }

    @staticmethod
    def _infer_legacy_action_mode(
        action_offset: torch.Tensor,
        action_scale: torch.Tensor,
        joint_pos_limits: torch.Tensor,
    ) -> str:
        lower_limits = joint_pos_limits[:, 0]
        upper_limits = joint_pos_limits[:, 1]
        median_offset = 0.5 * (upper_limits + lower_limits)
        median_scale = 0.5 * (upper_limits - lower_limits)

        if torch.allclose(action_offset, median_offset, atol=1e-5, rtol=1e-4) and torch.allclose(
            action_scale, median_scale, atol=1e-5, rtol=1e-4
        ):
            return "median"

        if torch.allclose(action_offset, torch.zeros_like(action_offset), atol=1e-6, rtol=0.0):
            return "residual"

        return "offset"

    @classmethod
    def _resolve_action_mode(
        cls,
        checkpoint_weights: dict,
        action_mode: object | None,
        action_offset: torch.Tensor,
        action_scale: torch.Tensor,
        joint_pos_limits: torch.Tensor,
    ) -> tuple[str, str]:
        if action_mode is not None:
            return cls._normalize_action_mode(action_mode), "argument"

        for key in ("action_mode", "action_mod"):
            checkpoint_action_mode = checkpoint_weights.get(key)
            if checkpoint_action_mode is not None:
                return cls._normalize_action_mode(checkpoint_action_mode), f"checkpoint:{key}"

        return cls._infer_legacy_action_mode(action_offset, action_scale, joint_pos_limits), "inferred"

    @staticmethod
    def _build_actor(
        obs_dims: dict[str, int],
        actor_type: str,
        action_dim: int,
        adain_res_blocks: int,
        actor_kwargs: dict[str, int] | None = None,
    ) -> torch.nn.Module:
        actor_kwargs = actor_kwargs or {}
        if actor_type == "vanila":
            return VanilaActor(obs_dims["policy"], action_dim)
        if actor_type == "recurrent":
            return RecurrentActor(
                obs_dims["policy"],
                action_dim,
                hidden_size=int(actor_kwargs.get("hidden_size", RecurrentActor.DEFAULT_HIDDEN_SIZE)),
                num_layers=int(actor_kwargs.get("num_layers", RecurrentActor.DEFAULT_NUM_LAYERS)),
            )
        if actor_type == "split_encoder":
            return SplitEncoderActor(obs_dims["robot"], obs_dims["motion"], action_dim)
        if actor_type == "adain":
            return AdaINActor(obs_dims["robot"], obs_dims["motion"], action_dim)
        if actor_type == "adain_res":
            return AdaINResActor(
                obs_dims["robot"],
                obs_dims["motion"],
                action_dim,
                num_blocks=adain_res_blocks,
            )
        raise ValueError(f"Unsupported actor type '{actor_type}'.")

    @staticmethod
    def _resolve_motion_body_index(env: MujocoEnv, body_name: str) -> int:
        try:
            return env.motion_lib.body_names.index(body_name)
        except ValueError as exc:
            raise ValueError(f"Motion body '{body_name}' does not exist in the loaded motion clip.") from exc

    @staticmethod
    def _resolve_mujoco_body_id(model: mujoco.MjModel, body_name: str) -> int:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"MuJoCo body '{body_name}' does not exist in the robot model.")
        return body_id

    @staticmethod
    def _resolve_free_joint(model: mujoco.MjModel) -> tuple[slice, slice, int]:
        for joint_id in range(model.njnt):
            if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                qpos_adr = int(model.jnt_qposadr[joint_id])
                dof_adr = int(model.jnt_dofadr[joint_id])
                body_id = int(model.jnt_bodyid[joint_id])
                return slice(qpos_adr, qpos_adr + 7), slice(dof_adr, dof_adr + 6), body_id
        raise ValueError("MuJoCo model does not contain a free joint for the floating base.")

    @staticmethod
    def _get_body_spatial_velocity(env: MujocoEnv, body_id: int) -> torch.Tensor:
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            env.mj_model,
            env.mj_data,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
            velocity,
            0,
        )
        return torch.from_numpy(velocity.astype(np.float32))

    def _sample_reference_motion(self, env: MujocoEnv):
        return env.motion_lib.sample_motion(motion_ids=env.motion_id, times=env.times)

    def _build_sim2sim_flat_obs(self, env: MujocoEnv, reference_motion=None) -> torch.Tensor:
        if reference_motion is None:
            reference_motion = self._sample_reference_motion(env)

        anchor_motion_index = self._resolve_motion_body_index(env, self.anchor_body_name)
        anchor_body_id = self._resolve_mujoco_body_id(env.mj_model, self.anchor_body_name)

        target_joint_pos = reference_motion["joint_pos"].squeeze(0).detach().cpu().clone()
        target_joint_vel = reference_motion["joint_vel"].squeeze(0).detach().cpu().clone()
        target_anchor_quat = reference_motion["body_quaternions"].squeeze(0)[anchor_motion_index].detach().cpu().clone()
        target_projected_gravity = quat_rotate_inverse(target_anchor_quat, env.gravity_vector)

        robot_anchor_quat = torch.from_numpy(env.mj_data.xquat[anchor_body_id].copy()).float()
        robot_projected_gravity = quat_rotate_inverse(robot_anchor_quat, env.gravity_vector)
        anchor_ang_vel = self._get_body_spatial_velocity(env, anchor_body_id)[:3]
        robot_joint_pos = env.get_joint_pos().clone()
        robot_joint_vel = env.get_joint_vel().clone()
        previous_action = env.previous_action.clone()

        return torch.cat(
            (
                target_projected_gravity,
                target_joint_pos,
                target_joint_vel,
                robot_projected_gravity,
                anchor_ang_vel,
                robot_joint_pos,
                robot_joint_vel,
                previous_action,
            )
        )

    @staticmethod
    def _parse_sim2sim_obs(flat_obs: torch.Tensor, action_dim: int) -> dict[str, torch.Tensor]:
        expected_dim = action_dim * 5 + 9
        if flat_obs.ndim != 1:
            raise ValueError(f"Expected a flat sim2sim observation, got shape {tuple(flat_obs.shape)}.")
        if flat_obs.numel() != expected_dim:
            raise ValueError(f"Expected sim2sim observation dim {expected_dim}, got {flat_obs.numel()}.")

        offset = 0

        target_projected_gravity = flat_obs[offset : offset + 3]
        offset += 3

        target_joint_pos = flat_obs[offset : offset + action_dim]
        offset += action_dim

        target_joint_vel = flat_obs[offset : offset + action_dim]
        offset += action_dim

        robot_projected_gravity = flat_obs[offset : offset + 3]
        offset += 3

        anchor_ang_vel = flat_obs[offset : offset + 3]
        offset += 3

        robot_joint_pos = flat_obs[offset : offset + action_dim]
        offset += action_dim

        robot_joint_vel = flat_obs[offset : offset + action_dim]
        offset += action_dim

        previous_action = flat_obs[offset : offset + action_dim]

        return {
            "motion_obs": torch.cat((target_projected_gravity, target_joint_pos, target_joint_vel), dim=-1),
            "robot_obs": torch.cat(
                (
                    robot_projected_gravity,
                    anchor_ang_vel,
                    robot_joint_pos,
                    robot_joint_vel,
                    previous_action,
                ),
                dim=-1,
            ),
            "target_projected_gravity": target_projected_gravity,
            "target_joint_pos": target_joint_pos,
            "target_joint_vel": target_joint_vel,
            "robot_projected_gravity": robot_projected_gravity,
            "anchor_ang_vel": anchor_ang_vel,
            "robot_joint_pos": robot_joint_pos,
            "robot_joint_vel": robot_joint_vel,
            "previous_action": previous_action,
        }

    @staticmethod
    def _extract_metrics(flat_obs: torch.Tensor, action_dim: int) -> dict[str, float]:
        obs_parts = Sim2SimEvaluator._parse_sim2sim_obs(flat_obs, action_dim)

        return {
            "joint_pos_mae": torch.mean(
                torch.abs(obs_parts["target_joint_pos"] - obs_parts["robot_joint_pos"])
            ).item(),
            "joint_vel_mae": torch.mean(
                torch.abs(obs_parts["target_joint_vel"] - obs_parts["robot_joint_vel"])
            ).item(),
            "gravity_mae": torch.mean(
                torch.abs(obs_parts["target_projected_gravity"] - obs_parts["robot_projected_gravity"])
            ).item(),
        }

    @classmethod
    def _build_debug_step_payload(
        cls,
        action_dim: int,
        flat_obs: torch.Tensor,
        actor_obs: dict[str, torch.Tensor],
        action: torch.Tensor,
        metrics: dict[str, float],
        sim_target_pos,
        sim_ctrl,
        sim_qpos,
        sim_qvel,
        actor_state: torch.Tensor | None = None,
        sim_motion_time: float | None = None,
    ) -> dict[str, object]:
        obs_parts = cls._parse_sim2sim_obs(flat_obs, action_dim)
        payload: dict[str, object] = {
            "obs_target_projected_gravity": obs_parts["target_projected_gravity"],
            "obs_target_joint_pos": obs_parts["target_joint_pos"],
            "obs_target_joint_vel": obs_parts["target_joint_vel"],
            "obs_robot_projected_gravity": obs_parts["robot_projected_gravity"],
            "obs_anchor_ang_vel": obs_parts["anchor_ang_vel"],
            "obs_robot_joint_pos": obs_parts["robot_joint_pos"],
            "obs_robot_joint_vel": obs_parts["robot_joint_vel"],
            "obs_previous_action": obs_parts["previous_action"],
            "action": action,
            "sim_target_pos": sim_target_pos,
            "sim_ctrl": sim_ctrl,
            "sim_qpos": sim_qpos,
            "sim_qvel": sim_qvel,
            **metrics,
        }
        if sim_motion_time is not None:
            payload["sim_motion_time"] = sim_motion_time
        payload.update(cls._build_actor_obs_log_fields(actor_obs))
        payload.update(cls._get_actor_state_log_fields(actor_state))
        return payload

    def __init__(
        self,
        checkpoint_path: str,
        actor_type: str | None = None,
        motion_files: list[str] | None = None,
        num_steps: int = 4000,
        simulation_dt: float = 1 / 200,
        decimation: int = 4,
        device: str = "cpu",
        action_mode: str | None = None,
        root_name: str = DEFAULT_ROOT_NAME,
        anchor_body_name: str = DEFAULT_ANCHOR_BODY_NAME,
        render: bool = False,
        random_start: bool = False,
        seed: int | None = None,
        adain_res_blocks: int | None = None,
        camera_track_body: str | None = None,
        camera_distance: float = DEFAULT_CAMERA_DISTANCE,
        camera_azimuth: float = DEFAULT_CAMERA_AZIMUTH,
        camera_elevation: float = DEFAULT_CAMERA_ELEVATION,
        video_dir: str | None = None,
        video_fps: int | None = None,
        video_width: int = DEFAULT_VIDEO_WIDTH,
        video_height: int = DEFAULT_VIDEO_HEIGHT,
        log_dir: str | None = None,
    ):
        if num_steps < 1:
            raise ValueError(f"num_steps must be positive, got {num_steps}.")
        if video_width < 1 or video_height < 1:
            raise ValueError(
                f"Video dimensions must be positive, got width={video_width}, height={video_height}."
            )
        if video_fps is not None and video_fps < 1:
            raise ValueError(f"video_fps must be positive, got {video_fps}.")

        self.checkpoint_path = self._resolve_existing_path(checkpoint_path)
        self.device = torch.device(device)
        self.num_steps = num_steps
        self.render = render
        self.random_start = random_start
        self.random_generator = torch.Generator(device="cpu")
        if seed is None:
            self.seed = int(self.random_generator.seed())
        else:
            self.seed = int(seed)
            self.random_generator.manual_seed(self.seed)

        weights = torch.load(self.checkpoint_path, map_location="cpu")
        self.actor_type = self._normalize_actor_type(actor_type or weights.get("actor_type"))
        self.is_recurrent_actor = self._is_recurrent_actor(self.actor_type)
        actor_weights = weights["actor"]
        actor_kwargs = dict(weights.get("actor_kwargs", {}))

        self.action_dim = int(weights["action_scale"].numel())
        self.obs_dims = self._infer_observation_dims(actor_weights, self.actor_type)

        actor_block_count = 5
        if self.actor_type == "adain_res":
            if adain_res_blocks is not None:
                actor_block_count = self._normalize_adain_res_blocks(adain_res_blocks)
            else:
                actor_block_count = self._normalize_adain_res_blocks(
                    actor_kwargs.get("num_blocks", self._infer_adain_res_blocks(actor_weights))
                )
            actor_kwargs = {"num_blocks": actor_block_count}
        elif self.actor_type == "recurrent":
            inferred_actor_kwargs = self._infer_recurrent_actor_kwargs(actor_weights)
            actor_kwargs = {
                "hidden_size": int(actor_kwargs.get("hidden_size", inferred_actor_kwargs["hidden_size"])),
                "num_layers": int(actor_kwargs.get("num_layers", inferred_actor_kwargs["num_layers"])),
            }
        else:
            actor_kwargs = {}

        self.actor = self._build_actor(
            self.obs_dims,
            self.actor_type,
            self.action_dim,
            actor_block_count,
            actor_kwargs=actor_kwargs,
        ).to(self.device)
        self.actor.load_state_dict(actor_weights)
        self.actor.eval()
        self.actor_kwargs = actor_kwargs
        self.actor_state = (
            self.actor.get_initial_state(1, device=self.device)
            if self.is_recurrent_actor
            else None
        )
        self.actor_episode_starts = torch.ones(1, dtype=torch.bool, device=self.device)

        self.motion_files = self._infer_motion_files(
            self.checkpoint_path,
            self.actor_type,
            weights,
            motion_files,
        )
        self.motion_name = motion_label(self.motion_files)
        self.simulation_dt = simulation_dt
        self.decimation = decimation
        self.root_name = root_name
        self.anchor_body_name = anchor_body_name
        self.camera_track_body = camera_track_body or anchor_body_name
        self.camera_distance = camera_distance
        self.camera_azimuth = camera_azimuth
        self.camera_elevation = camera_elevation
        self.video_dir = self._resolve_output_path(video_dir) if video_dir is not None else None
        self.log_dir = self._resolve_output_path(log_dir) if log_dir is not None else None
        self.video_fps = video_fps or max(1, round(1.0 / (self.simulation_dt * self.decimation)))
        self.video_width = video_width
        self.video_height = video_height
        self.kp = weights["joint_stiffness"].detach().cpu()
        self.kd = weights["joint_damping"].detach().cpu()
        self.effort_limits = weights["joint_effort_limits"].detach().cpu()
        self.joint_pos_limits = weights["joint_pos_limits"].detach().cpu()
        self.action_offset = weights["action_offset"].detach().cpu()
        self.action_scale = weights["action_scale"].detach().cpu()
        self.action_mode, self.action_mode_source = self._resolve_action_mode(
            weights,
            action_mode,
            self.action_offset,
            self.action_scale,
            self.joint_pos_limits,
        )

    def _sample_start_time(self, env: MujocoEnv) -> float:
        if not self.random_start:
            return 0.0

        duration = float(env.motion_lib.get_duration(env.motion_id).squeeze(0).item())
        if duration <= 0.0:
            return 0.0

        start_time = float(torch.rand(1, generator=self.random_generator).item() * duration)
        if duration > 1e-6:
            start_time = min(start_time, duration - 1e-6)
        return start_time

    def _reset_env(self, env: MujocoEnv, start_time: float) -> torch.Tensor:
        mujoco.mj_resetData(env.mj_model, env.mj_data)

        env.previous_action[:] = 0.0
        env.motion_id.zero_()
        env.times = torch.tensor([start_time], dtype=torch.float32)
        env.n_steps = 0

        reference_motion = self._sample_reference_motion(env)
        joint_positions = reference_motion["joint_pos"].squeeze(0).detach().cpu().numpy()[env.isaac2mujoco]
        joint_velocities = reference_motion["joint_vel"].squeeze(0).detach().cpu().numpy()[env.isaac2mujoco]

        root_motion_index = self._resolve_motion_body_index(env, self.root_name)
        root_body_id = self._resolve_mujoco_body_id(env.mj_model, self.root_name)
        free_qpos_slice, free_qvel_slice, _ = self._resolve_free_joint(env.mj_model)

        desired_root_pos = reference_motion["body_positions"].squeeze(0)[root_motion_index].detach().cpu().clone()
        desired_root_pos[2] += 0.05
        desired_root_quat = reference_motion["body_quaternions"].squeeze(0)[root_motion_index].detach().cpu().clone()
        desired_root_lin_vel = reference_motion["body_linear_velocities"].squeeze(0)[root_motion_index].detach().cpu().numpy()
        desired_root_ang_vel = reference_motion["body_angular_velocities"].squeeze(0)[root_motion_index].detach().cpu().numpy()
        desired_root_spatial = np.concatenate((desired_root_ang_vel, desired_root_lin_vel), axis=0)

        env.mj_data.qpos[:] = 0.0
        env.mj_data.qvel[:] = 0.0
        env.mj_data.qpos[free_qpos_slice.start + 3] = 1.0
        env.mj_data.qpos[free_qpos_slice.stop :] = joint_positions
        mujoco.mj_forward(env.mj_model, env.mj_data)

        root_local_pos = torch.from_numpy(env.mj_data.xpos[root_body_id].copy()).float()
        root_local_quat = torch.from_numpy(env.mj_data.xquat[root_body_id].copy()).float()
        base_quat = _quat_mul(desired_root_quat, _quat_inverse(root_local_quat))
        base_pos = desired_root_pos - _quat_rotate(base_quat, root_local_pos)

        env.mj_data.qpos[:] = 0.0
        env.mj_data.qvel[:] = 0.0
        env.mj_data.qpos[free_qpos_slice.start : free_qpos_slice.start + 3] = base_pos.numpy()
        env.mj_data.qpos[free_qpos_slice.start + 3 : free_qpos_slice.stop] = base_quat.numpy()
        env.mj_data.qpos[free_qpos_slice.stop :] = joint_positions
        env.mj_data.qvel[free_qvel_slice.stop :] = joint_velocities
        mujoco.mj_forward(env.mj_model, env.mj_data)

        joint_velocity_only = self._get_body_spatial_velocity(env, root_body_id).numpy()
        free_to_root_velocity = np.zeros((6, 6), dtype=np.float64)
        for i in range(6):
            env.mj_data.qvel[:] = 0.0
            env.mj_data.qvel[free_qvel_slice.start + i] = 1.0
            mujoco.mj_forward(env.mj_model, env.mj_data)
            free_to_root_velocity[:, i] = self._get_body_spatial_velocity(env, root_body_id).numpy()

        free_velocity = np.linalg.solve(free_to_root_velocity, desired_root_spatial - joint_velocity_only)

        env.mj_data.qvel[:] = 0.0
        env.mj_data.qvel[free_qvel_slice] = free_velocity
        env.mj_data.qvel[free_qvel_slice.stop :] = joint_velocities

        mujoco.mj_forward(env.mj_model, env.mj_data)

        if env.mj_viewer is not None and env.mj_viewer.is_alive:
            env.mj_viewer.render()
        else:
            env.mj_viewer = None

        obs = self._build_sim2sim_flat_obs(env, reference_motion=reference_motion)
        env.target_pos = reference_motion["joint_pos"].squeeze(0).clone()
        return obs

    def _build_video_path(self, motion_file: str, motion_index: int) -> Path | None:
        if self.video_dir is None:
            return None

        motion_stem = Path(motion_file).stem
        safe_motion_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", motion_stem)
        filename = f"{self.checkpoint_path.stem}_{motion_index:02d}_{safe_motion_stem}.mp4"
        return self.video_dir / filename

    def _build_log_prefix(self, motion_file: str, motion_index: int) -> Path | None:
        if self.log_dir is None:
            return None

        motion_stem = Path(motion_file).stem
        safe_motion_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", motion_stem)
        filename = f"{self.checkpoint_path.stem}_{motion_index:02d}_{safe_motion_stem}"
        return self.log_dir / filename

    def _build_replay_camera(self, env: MujocoEnv, motion_file: str, motion_index: int) -> ReplayCameraRecorder | None:
        if not self.render and self.video_dir is None:
            return None

        return ReplayCameraRecorder(
            env=env,
            track_body_name=self.camera_track_body,
            camera_distance=self.camera_distance,
            camera_azimuth=self.camera_azimuth,
            camera_elevation=self.camera_elevation,
            video_path=self._build_video_path(motion_file, motion_index),
            video_width=self.video_width,
            video_height=self.video_height,
            video_fps=self.video_fps,
        )

    def _build_env(self, motion_file: str) -> MujocoEnv:
        return MujocoEnv(
            simulation_dt=self.simulation_dt,
            decimation=self.decimation,
            kp=self.kp,
            kd=self.kd,
            effort_limits=self.effort_limits,
            joint_pos_limits=self.joint_pos_limits,
            action_offset=self.action_offset,
            action_scale=self.action_scale,
            expert_motion_file=motion_file,
            root_name=self.root_name,
            render=self.render,
            action_mode=self.action_mode,
        )

    def _get_actor_observation(self, flat_obs: torch.Tensor) -> dict[str, torch.Tensor]:
        obs_parts = self._parse_sim2sim_obs(flat_obs, self.action_dim)
        motion_obs = obs_parts["motion_obs"]
        robot_obs = obs_parts["robot_obs"]

        if "motion" in self.obs_dims and motion_obs.numel() != self.obs_dims["motion"]:
            raise ValueError(f"Expected motion observation dim {self.obs_dims['motion']}, got {motion_obs.numel()}.")
        if "robot" in self.obs_dims and robot_obs.numel() != self.obs_dims["robot"]:
            raise ValueError(f"Expected robot observation dim {self.obs_dims['robot']}, got {robot_obs.numel()}.")

        if self._is_concat_actor(self.actor_type):
            policy_obs = torch.cat((motion_obs, robot_obs), dim=-1)
            return {"obs": policy_obs.unsqueeze(0).to(self.device)}

        return {
            "motion_obs": motion_obs.unsqueeze(0).to(self.device),
            "robot_obs": robot_obs.unsqueeze(0).to(self.device),
        }

    @torch.no_grad()
    def get_action(self, obs_batch: dict[str, torch.Tensor], determine: bool = True) -> torch.Tensor:
        if self.is_recurrent_actor:
            actor_output = self.actor(
                obs_batch,
                initial_state=self.actor_state,
                episode_starts=self.actor_episode_starts,
            )
        else:
            actor_output = self.actor(obs_batch)

        actor_step, next_state = self._unpack_actor_output(actor_output)
        if determine:
            action = actor_step.mean
        else:
            action = actor_step.action

        if self.is_recurrent_actor:
            self.actor_state = next_state

        return action

    def _eval_motion_file(self, motion_file: str, motion_index: int) -> tuple[int, dict[str, float], Path | None, float]:
        env = self._build_env(motion_file)
        metrics = defaultdict(float)
        steps_run = 0
        replay_camera = None
        start_time = 0.0
        logger = RolloutDebugLogger(self._build_log_prefix(motion_file, motion_index))
        error_message: str | None = None

        try:
            start_time = self._sample_start_time(env)
            obs = self._reset_env(env, start_time)
            if self.is_recurrent_actor:
                self.actor_state = self.actor.get_initial_state(1, device=self.device)
                self.actor_episode_starts = torch.ones(1, dtype=torch.bool, device=self.device)
            replay_camera = self._build_replay_camera(env, motion_file, motion_index)

            if replay_camera is not None:
                replay_camera.render_viewer()
                replay_camera.capture_frame()

            for step_idx in range(self.num_steps):
                current_obs = obs
                actor_obs = self._get_actor_observation(current_obs)
                actor_state_before_step = self.actor_state if self.is_recurrent_actor else None
                action = self.get_action(actor_obs, determine=True).squeeze(0).detach().cpu()
                env.step(action)
                obs = self._build_sim2sim_flat_obs(env)
                if self.is_recurrent_actor:
                    self.actor_episode_starts = torch.zeros(1, dtype=torch.bool, device=self.device)

                if replay_camera is not None:
                    replay_camera.capture_frame()

                step_metrics = self._extract_metrics(obs, self.action_dim)
                for key, value in step_metrics.items():
                    metrics[key] += value
                logger.log_step(
                    step_idx,
                    self._build_debug_step_payload(
                        self.action_dim,
                        flat_obs=current_obs,
                        actor_obs=actor_obs,
                        action=action,
                        metrics=step_metrics,
                        sim_target_pos=env.target_pos,
                        sim_ctrl=env.mj_data.ctrl,
                        sim_qpos=env.mj_data.qpos,
                        sim_qvel=env.mj_data.qvel,
                        actor_state=actor_state_before_step,
                        sim_motion_time=float(env.times.item()),
                    ),
                )

                steps_run += 1

                if self.render and env.mj_viewer is None:
                    break
        except Exception as exc:
            error_message = repr(exc)
            raise
        finally:
            log_paths = logger.finish(
                {
                    "checkpoint": str(self.checkpoint_path),
                    "actor_type": self.actor_type,
                    "actor_kwargs": self.actor_kwargs,
                    "action_mode": self.action_mode,
                    "action_mode_source": self.action_mode_source,
                    "motion_file": motion_file,
                    "motion_index": motion_index,
                    "start_time": start_time,
                    "num_steps_requested": self.num_steps,
                    "num_steps_executed": steps_run,
                    "joint_pos_mae_mean": (metrics["joint_pos_mae"] / steps_run) if steps_run else None,
                    "joint_vel_mae_mean": (metrics["joint_vel_mae"] / steps_run) if steps_run else None,
                    "gravity_mae_mean": (metrics["gravity_mae"] / steps_run) if steps_run else None,
                    "error": error_message,
                }
            )
            if log_paths is not None:
                npz_path, json_path = log_paths
                print(f"debug_log_npz: {npz_path}")
                print(f"debug_log_json: {json_path}")
            if replay_camera is not None:
                replay_camera.close()
            env.close()

        if steps_run == 0:
            return 0, {}, replay_camera.video_path if replay_camera is not None else None, start_time

        return (
            steps_run,
            {key: value / steps_run for key, value in metrics.items()},
            replay_camera.video_path if replay_camera is not None else None,
            start_time,
        )

    def eval(self) -> None:
        aggregate_metrics = defaultdict(float)
        aggregate_steps = 0

        print(f"checkpoint: {self.checkpoint_path}")
        print(f"motion_label: {self.motion_name}")
        print(f"actor_type: {self.actor_type}")
        print(f"action_mode: {self.action_mode}")
        print(f"action_mode_source: {self.action_mode_source}")
        print(f"root_name: {self.root_name}")
        print(f"anchor_body_name: {self.anchor_body_name}")
        print(f"random_start: {self.random_start}")
        if self.random_start:
            print(f"random_seed: {self.seed}")

        if self.video_dir is not None:
            print(f"video_dir: {self.video_dir}")
            print(f"video_fps: {self.video_fps}")

        for motion_index, motion_file in enumerate(self.motion_files):
            steps_run, metrics, video_path, start_time = self._eval_motion_file(motion_file, motion_index)

            print(f"motion_file: {motion_file}")
            if self.random_start:
                print(f"start_time: {start_time:.6f}")
            print(f"steps: {steps_run}")
            if video_path is not None:
                print(f"video_path: {video_path}")

            for key, value in metrics.items():
                print(f"{key}: {value:.6f}")
                aggregate_metrics[key] += value * steps_run

            aggregate_steps += steps_run

        if aggregate_steps == 0:
            return

        if len(self.motion_files) > 1:
            print(f"aggregate_steps: {aggregate_steps}")
            for key, total_value in aggregate_metrics.items():
                print(f"aggregate_{key}: {total_value / aggregate_steps:.6f}")


def main():
    args = build_arg_parser().parse_args()
    evaluator = Sim2SimEvaluator(
        checkpoint_path=args.checkpoint,
        actor_type=args.actor_type,
        motion_files=args.motion_file,
        num_steps=args.num_steps,
        simulation_dt=args.simulation_dt,
        decimation=args.decimation,
        device=args.device,
        action_mode=args.action_mode,
        root_name=args.root_name,
        anchor_body_name=args.anchor_body_name,
        render=args.render,
        random_start=args.random_start,
        seed=args.seed,
        camera_track_body=args.camera_track_body,
        camera_distance=args.camera_distance,
        camera_azimuth=args.camera_azimuth,
        camera_elevation=args.camera_elevation,
        video_dir=args.video_dir,
        video_fps=args.video_fps,
        video_width=args.video_width,
        video_height=args.video_height,
        log_dir=args.log_dir,
    )
    evaluator.eval()


if __name__ == "__main__":
    main()
