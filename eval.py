import argparse
from datetime import datetime
import re
import time
from pathlib import Path

from isaaclab.app import AppLauncher

import gymnasium
import torch

from RLAlg.nn.steps import StochasticContinuousPolicyStep

from debug_log import RolloutDebugLogger
from env.motions import resolve_motion_files
from model.actor import AdaINActor, AdaINResActor, RecurrentActor, SplitEncoderActor, VanilaActor


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
    parser.add_argument("--checkpoint", default="final.pth")
    parser.add_argument(
        "--actor-type",
        default=None,
        help="Override actor architecture for checkpoints without actor metadata: vanila, recurrent, split_encoder, adain, or adain_res.",
    )
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=50,
        help="Print progress every N evaluation steps. Set to 0 to disable periodic progress logs.",
    )
    parser.add_argument(
        "--show-reference-motion",
        action="store_true",
        help="Enable reference motion markers in the viewer. Disabled by default to reduce GUI startup overhead.",
    )
    parser.add_argument(
        "--adain-res-blocks",
        type=int,
        default=None,
        help="Override the number of AdaIN-Res blocks. Defaults to checkpoint metadata.",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Optional directory to save rollout debug logs as NPZ+JSON.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser

class Evaluator:
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
    def _infer_observation_dims(obs: dict[str, torch.Tensor]) -> dict[str, int]:
        required_keys = ("motion", "robot", "privilege")
        missing_keys = [key for key in required_keys if key not in obs]
        if missing_keys:
            raise KeyError(f"Environment observation is missing required keys: {missing_keys}.")

        motion_dim = obs["motion"].shape[-1]
        robot_dim = obs["robot"].shape[-1]

        return {
            "motion": motion_dim,
            "robot": robot_dim,
            "policy": motion_dim + robot_dim,
        }

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
        payload.update(cls._build_actor_obs_log_fields(actor_obs))
        payload.update(cls._get_actor_state_log_fields(actor_state))

        info_fields, info_metadata = cls._extract_info_log_fields(info)
        payload.update(info_fields)
        return payload, info_metadata

    @staticmethod
    def _resolve_motion_files(
        checkpoint_path: str,
        checkpoint_weights: dict,
        default_motion_files: str | list[str] | None,
    ) -> list[str]:
        checkpoint_motion_files = checkpoint_weights.get("motion_files")
        if checkpoint_motion_files:
            try:
                return resolve_motion_files(checkpoint_motion_files)
            except FileNotFoundError:
                pass

        checkpoint_motion_names = checkpoint_weights.get("motion_names")
        if checkpoint_motion_names:
            return resolve_motion_files(checkpoint_motion_names)

        checkpoint_stem = Path(checkpoint_path).stem
        for actor_marker in ("_vanila_", "_recurrent_", "_split_encoder_", "_adain_", "_adain_res_"):
            if actor_marker not in checkpoint_stem:
                continue

            motion_name = checkpoint_stem.rsplit(actor_marker, 1)[0]
            candidate = f"env/assests/{motion_name}.npz"
            try:
                return resolve_motion_files(candidate)
            except FileNotFoundError:
                break

        return resolve_motion_files(default_motion_files)

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
    def _get_actor_observation(obs: dict[str, torch.Tensor], actor_type: str) -> dict[str, torch.Tensor]:
        if Evaluator._is_concat_actor(actor_type):
            return {"obs": torch.cat((obs["motion"], obs["robot"]), dim=-1)}
        return {
            "motion_obs": obs["motion"],
            "robot_obs": obs["robot"],
        }

    def _build_log_prefix(self) -> Path | None:
        if self.log_dir is None:
            return None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.log_dir / f"{self.checkpoint_path.stem}_{timestamp}"

    def __init__(
        self,
        checkpoint_path: str,
        actor_type: str | None = None,
        adain_res_blocks: int | None = None,
        num_steps: int = 1000,
        progress_interval: int = 50,
        show_reference_motion: bool = False,
        log_dir: str | None = None,
    ):
        from env.cfg import G1MultiMotionEnv

        self.cfg = G1MultiMotionEnv()
        self.checkpoint_path = Path(checkpoint_path).expanduser()
        self.log_dir = self._resolve_output_path(log_dir) if log_dir is not None else None
        if num_steps < 1:
            raise ValueError(f"num_steps must be positive, got {num_steps}.")
        if progress_interval < 0:
            raise ValueError(f"progress_interval must be non-negative, got {progress_interval}.")
        self.num_steps = num_steps
        self.progress_interval = progress_interval
        weights = torch.load(self.checkpoint_path, map_location="cpu")
        self.cfg.expert_motion_file = self._resolve_motion_files(
            str(self.checkpoint_path),
            weights,
            self.cfg.expert_motion_file,
        )

        self.env_name = "G1MotionTracking-v0"
        
        self.cfg.scene.num_envs = 1
        self.cfg.training = False
        self.cfg.add_action_noise = False
        self.cfg.add_obs_noise = False
        self.cfg.add_reset_noise = False
        self.cfg.random_start = False
        self.cfg.events = None
        self.cfg.reference_motion_viewer_enabled = show_reference_motion

        print("Create env", flush=True)
        self.env = gymnasium.make(self.env_name, cfg=self.cfg)

        self.device = self.env.unwrapped.device
        print("Reset env", flush=True)
        self.initial_obs, _ = self.env.reset()
        self.obs_dims = self._infer_observation_dims(self.initial_obs)

        action_dim = self.cfg.action_space
        print("Load checkpoint", flush=True)
        self.actor_type = self._normalize_actor_type(actor_type or weights.get("actor_type"))
        self.is_recurrent_actor = self._is_recurrent_actor(self.actor_type)
        actor_weights = weights["actor"]
        actor_kwargs = dict(weights.get("actor_kwargs", {}))

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
            action_dim,
            actor_block_count,
            actor_kwargs=actor_kwargs,
        ).to(self.device)

        self.actor.load_state_dict(actor_weights)
        self.actor.eval()
        self.actor_kwargs = actor_kwargs
        self.actor_state = (
            self.actor.get_initial_state(self.cfg.scene.num_envs, device=self.device)
            if self.is_recurrent_actor
            else None
        )
        self.actor_episode_starts = torch.ones(
            self.cfg.scene.num_envs,
            dtype=torch.bool,
            device=self.device,
        )

        print(
            f"Start actor={self.actor_type} steps={self.num_steps} motions={len(self.cfg.expert_motion_file)} "
            f"reference_motion_viewer={show_reference_motion}",
            flush=True,
        )
        #self.tracker = MetricsTracker()
        #self.tracker.add_batch_metrics("episode_return", self.cfg.scene.num_envs)
        #self.tracker.add_batch_metrics("episode_length", self.cfg.scene.num_envs)

        #WandbLogger.init_project("Mimic_Eval", "G1_multi_motion")

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

        actor_step, next_state = self._unpack_actor_output(actor_output)
        action = actor_step.action
        if determine:
            action = actor_step.mean

        if self.is_recurrent_actor:
            self.actor_state = next_state

        return action

    def rollout(self, obs, info):
        rollout_start = time.perf_counter()
        logger = RolloutDebugLogger(self._build_log_prefix())
        first_done_step = None
        reward_history: list[float] = []
        info_metadata: dict[str, object] = {}
        error_message: str | None = None

        if self.is_recurrent_actor:
            self.actor_state = self.actor.get_initial_state(self.cfg.scene.num_envs, device=self.device)
            self.actor_episode_starts = torch.ones(
                self.cfg.scene.num_envs,
                dtype=torch.bool,
                device=self.device,
            )
        step_count = 0
        try:
            for step_idx in range(self.num_steps):
                actor_obs = self._get_actor_observation(obs, self.actor_type)
                actor_state_before_step = self.actor_state if self.is_recurrent_actor else None
                action = self.get_action(actor_obs, True)
                if not torch.isfinite(action).all():
                    raise RuntimeError(
                        f"Non-finite action detected at step {step_idx + 1}: "
                        f"min={action.min().item():.6f} max={action.max().item():.6f}"
                    )

                next_obs, task_reward, terminate, timeout, info = self.env.step(action)
                if not torch.isfinite(task_reward).all():
                    raise RuntimeError(
                        f"Non-finite reward detected at step {step_idx + 1}: "
                        f"min={task_reward.min().item():.6f} max={task_reward.max().item():.6f}"
                    )
                reward = task_reward
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

                if self.progress_interval and ((step_idx + 1) % self.progress_interval == 0 or step_idx + 1 == self.num_steps):
                    elapsed = time.perf_counter() - rollout_start
                    print(
                        f"Progress {step_idx + 1}/{self.num_steps} elapsed={elapsed:.2f}s reward_mean={reward.mean().item():.6f}",
                        flush=True,
                    )

                obs = next_obs
                step_count = step_idx + 1
        except Exception as exc:
            error_message = repr(exc)
            raise
        finally:
            log_paths = logger.finish(
                {
                    "checkpoint": str(self.checkpoint_path),
                    "actor_type": self.actor_type,
                    "actor_kwargs": self.actor_kwargs,
                    "motion_files": list(self.cfg.expert_motion_file),
                    "num_steps_requested": self.num_steps,
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
            if log_paths is not None:
                npz_path, json_path = log_paths
                print(f"debug_log_npz: {npz_path}", flush=True)
                print(f"debug_log_json: {json_path}", flush=True)

        return obs, info

    def eval(self):
        obs = self.initial_obs
        info = {}
        obs, info = self.rollout(obs, info)

        self.env.close()

if __name__ == "__main__":
    args_cli = build_arg_parser().parse_args()
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    try:
        evaluator = Evaluator(
            args_cli.checkpoint,
            args_cli.actor_type,
            args_cli.adain_res_blocks,
            args_cli.num_steps,
            args_cli.progress_interval,
            args_cli.show_reference_motion,
            args_cli.log_dir,
        )
        evaluator.eval()
    finally:
        simulation_app.close()
