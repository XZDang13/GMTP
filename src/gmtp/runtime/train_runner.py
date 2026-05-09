from __future__ import annotations

import json
import math
import random
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from RLAlg.alg.ppo import PPO
from RLAlg.buffer.replay_buffer import ReplayBuffer, compute_gae
from RLAlg.logger import MetricsTracker, WandbLogger
from RLAlg.nn.steps import ValueStep
from RLAlg.scheduler import KLAdaptiveLR
from tqdm import trange

from gmtp.integrations.ref2act.motion import motion_label, motion_names
from gmtp.integrations.ref2act.observation_history import resolve_observation_window_lengths
from gmtp.models import (
    ActorType,
    Critic,
    build_actor,
    build_critic_privilege_layout,
    get_actor_kwargs,
    get_actor_observation,
    get_critic_batch,
    get_critic_observation,
    get_critic_records,
    get_policy_batch,
    get_policy_records,
    get_policy_storage_specs,
)
from gmtp.runtime.checkpoints import CheckpointV2, build_training_checkpoint, load_checkpoint_v2, save_checkpoint_v2
from gmtp.runtime.amp import AMP_DTYPE_NAME, autocast_context, build_grad_scaler, normalize_device, resolve_amp_enabled
from gmtp.runtime.config import RunConfig
from gmtp.runtime.io import build_run_paths, write_json
from gmtp.runtime.observations import infer_env_observation_dims, structure_env_observation
from gmtp.runtime.policy import (
    load_actor_checkpoint_state,
    resolve_checkpoint_actor_spec,
    resolve_motion_mae_checkpoint_path,
)

RELATIVE_ANCHOR_POS_DIM = 3
PPO_CLIP_RATIO = 0.2
ENTROPY_COEF = 0.01
CURRICULUM_METRIC_PREFIX = "curriculum/"
END_EFFECTOR_TERMINATION_RULE_ID = "end_effector_position_failure"
END_EFFECTOR_TERMINATION_STATE_KEY = "end_effector_termination_curriculum"
ANCHOR_CONSOLE_TOP_K = 20
ANCHOR_DASHBOARD_MAX_RANK_BANDS = 80
ANCHOR_DASHBOARD_TOP_K = 20
ANCHOR_DASHBOARD_LABEL_CHARS = 34
REQUIRED_TRAINER_STATE_KEYS = (
    "actor_optimizer",
    "critic_optimizer",
    "lr_scheduler",
    "grad_scaler",
    "update_count",
    "global_step",
)


@dataclass(frozen=True)
class AnchorProbabilityArrays:
    motion_index: np.ndarray
    motion_name: np.ndarray
    anchor_index: np.ndarray
    anchor_time: np.ndarray
    probability: np.ndarray


@dataclass(frozen=True)
class AnchorRankBandGrid:
    values: np.ndarray
    num_bins: int
    num_motions: int
    num_rank_bands: int


@dataclass
class EndEffectorTerminationCurriculumState:
    enabled: bool
    thresholds: tuple[float, ...]
    stage_index: int
    current_threshold: float
    warmup_fraction: float
    deadline_fraction: float
    ema_alpha: float
    end_threshold: float
    tighten_step: float
    ema_terminate_rate: float | None = None
    ema_error_mean: float | None = None
    ema_sample_count: int = 0
    ema_error_sample_count: int = 0
    last_tighten_update: int = 0
    gate_pass: bool = False
    deadline_forced: bool = False
    gate_reason: str = "initial"


class StartupTimer:
    def __init__(self, *, prefix: str = "train startup") -> None:
        self.prefix = prefix
        self.start_time = perf_counter()

    def log(self, message: str) -> None:
        elapsed = perf_counter() - self.start_time
        print(f"{self.prefix} [{elapsed:7.2f}s]: {message}", flush=True)


class OptimizerCollection(torch.optim.Optimizer):
    def __init__(self, *optimizers: torch.optim.Optimizer):
        self.optimizers = [optimizer for optimizer in optimizers if optimizer is not None]
        if not self.optimizers:
            raise ValueError("OptimizerCollection requires at least one optimizer.")

        params = []
        seen_params = set()
        for optimizer in self.optimizers:
            for group in optimizer.param_groups:
                for param in group["params"]:
                    param_id = id(param)
                    if param_id in seen_params:
                        raise ValueError("OptimizerCollection does not support duplicated parameters.")
                    seen_params.add(param_id)
                    params.append(param)

        super().__init__(params, defaults={})
        self.param_groups = [group for optimizer in self.optimizers for group in optimizer.param_groups]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for optimizer in self.optimizers:
            optimizer.step()

        return loss

    def zero_grad(self, set_to_none: bool = True):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"optimizers": [optimizer.state_dict() for optimizer in self.optimizers]}

    def load_state_dict(self, state_dict):
        optimizer_states = state_dict["optimizers"]
        if len(optimizer_states) != len(self.optimizers):
            raise ValueError(f"Expected {len(self.optimizers)} optimizer states, got {len(optimizer_states)}.")

        for optimizer, optimizer_state in zip(self.optimizers, optimizer_states, strict=True):
            optimizer.load_state_dict(optimizer_state)

        self.param_groups = [group for optimizer in self.optimizers for group in optimizer.param_groups]


class TrainRunner:
    def __init__(self, config: RunConfig):
        self.config = config
        startup_timer = StartupTimer()
        self.resume_checkpoint_path = (
            None if config.resume_checkpoint_path is None else Path(config.resume_checkpoint_path).expanduser().resolve()
        )
        self.resume_checkpoint: CheckpointV2 | None = None
        self.resume_mode = "none"
        self.resume_trainer_state_restored = False
        if self.resume_checkpoint_path is not None:
            startup_timer.log(f"loading resume checkpoint {self.resume_checkpoint_path}")
            self.resume_checkpoint = load_checkpoint_v2(self.resume_checkpoint_path)

        self.actor_type, self.actor_config_kwargs = self._resolve_actor_config(config, self.resume_checkpoint)
        self.motion_file_inputs = self._resolve_motion_file_inputs(config, self.resume_checkpoint)

        startup_timer.log("importing Isaac/Ref2Act training environment")
        from gmtp.integrations.ref2act.isaac_env import make_training_env

        startup_timer.log("configuring training environment")
        self.observation_window_lengths = resolve_observation_window_lengths(
            robot_window_length=int(self.actor_config_kwargs["robot_window_length"]),
            motion_window_length=int(self.actor_config_kwargs["motion_window_length"]),
        )
        self.end_effector_termination_curriculum = self._build_end_effector_termination_curriculum_state(
            config,
            checkpoint=self.resume_checkpoint,
        )
        self.sampler_failure_warmup_steps = self._sampler_failure_warmup_steps(config)
        make_training_kwargs = self._build_make_training_env_kwargs(
            make_training_env,
            window_lengths=self.observation_window_lengths,
            motion_files=self.motion_file_inputs,
            sampler_failure_warmup_steps=self.sampler_failure_warmup_steps,
            end_effector_termination_curriculum=self.end_effector_termination_curriculum,
        )
        if self.motion_file_inputs is None:
            startup_timer.log("creating training environment with default motion set")
        else:
            startup_timer.log(f"creating training environment with {len(self.motion_file_inputs)} motion input(s)")
        self.env, self.cfg = make_training_env(**make_training_kwargs)
        self._set_runtime_end_effector_termination_threshold(
            self.env,
            self.end_effector_termination_curriculum.current_threshold,
        )
        startup_timer.log(f"training environment ready with {len(self.cfg.expert_motion_file)} resolved motion clip(s)")
        self.device = normalize_device(self.env.unwrapped.device)
        self.requested_amp = bool(config.use_amp)
        self.use_amp = resolve_amp_enabled(self.requested_amp, self.device)
        self.amp_dtype = AMP_DTYPE_NAME
        self.segment_source = self._normalize_choice_name(getattr(self.cfg, "segment_source", None))
        self.sampling_strategy = self._normalize_choice_name(getattr(self.cfg, "sampling_strategy", None))
        self.motion_files = list(self.cfg.expert_motion_file)
        self.motion_name = motion_label(self.motion_files)
        startup_timer.log(self._format_sampler_config_summary())
        resolved_motion_mae_checkpoint = resolve_motion_mae_checkpoint_path(
            self.resume_checkpoint,
            override=config.motion_mae_encoder_checkpoint,
        )
        self.motion_mae_encoder_checkpoint = (
            None if resolved_motion_mae_checkpoint is None else str(resolved_motion_mae_checkpoint)
        )
        self.run_date = datetime.now().strftime("%Y%m%d")
        self.checkpoint_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_run_name = f"G1_{len(self.motion_files)}_{self.run_date}"
        if self.resume_checkpoint_path is not None:
            default_run_name = f"{default_run_name}_resume"
        self.run_name = config.run_name or default_run_name
        self.run_paths = build_run_paths(config.output_root, "train", self.run_name)
        self.checkpoint_interval = config.checkpoint_interval
        if config.anchor_log_interval < 1:
            raise ValueError("anchor_log_interval must be positive.")
        if config.anchor_heatmap_bins < 1:
            raise ValueError("anchor_heatmap_bins must be positive.")
        self.anchor_log_interval = int(config.anchor_log_interval)
        self.anchor_heatmap_bins = int(config.anchor_heatmap_bins)
        self._anchor_heatmap_warning_emitted = False
        self.steps = config.rollout_steps
        self.global_step = 0
        self.update_count = 0

        write_json(self.run_paths.config_path, {"command": "train", "config": self.config})

        startup_timer.log("resetting training environment")
        self.initial_obs, _ = self.env.reset()
        self.initial_obs = structure_env_observation(
            self.initial_obs,
            action_dim=self.cfg.action_space,
            observation_window_lengths=self.observation_window_lengths,
        )
        startup_timer.log("building policy and value models")
        startup_timer.log("inferring observation dimensions")
        self.raw_obs_dims = infer_env_observation_dims(self.initial_obs)
        self.obs_dims = self.raw_obs_dims
        startup_timer.log(
            "building actor model: "
            f"type={self.actor_type.value} "
            f"robot_encoder={self.actor_config_kwargs['robot_encoder_type']} "
            f"robot_window={self.actor_config_kwargs['robot_window_length']} "
            f"motion_encoder={self.actor_config_kwargs['motion_encoder_type']} "
            f"motion_window={self.actor_config_kwargs['motion_window_length']}"
        )
        if self.motion_mae_encoder_checkpoint is not None:
            startup_timer.log(f"loading Motion MAE encoder checkpoint {self.motion_mae_encoder_checkpoint}")
        self.actor = build_actor(
            self.obs_dims,
            self.actor_type,
            self.cfg.action_space,
            actor_kwargs=self._build_actor_kwargs(),
            motion_mae_encoder_checkpoint=self.motion_mae_encoder_checkpoint,
            device=self.device,
        ).to(self.device)
        self.actor_kwargs = get_actor_kwargs(self.actor, self.actor_type)
        startup_timer.log("actor model ready")
        startup_timer.log("resolving critic observation layout")
        self.critic_key_body_count = self._resolve_critic_key_body_count(
            self.cfg,
            self.env,
            critic_obs_dim=self.obs_dims["critic"],
            action_dim=self.cfg.action_space,
            observation_window_lengths=self.observation_window_lengths,
        )
        startup_timer.log(
            "building critic model: "
            f"obs_dim={self.obs_dims['critic']} "
            f"key_body_count={self.critic_key_body_count}"
        )
        self.critic = self._build_critic_model().to(self.device)
        startup_timer.log(f"critic model ready: type={self.critic.critic_type}")
        startup_timer.log("creating optimizers and rollout storage")

        self.actor_optimizer, actor_optimizer_stats = self._build_optimizer_collection(
            {"actor": self.actor},
            prefer_muon=self.device.type == "cuda",
        )
        self.critic_optimizer, critic_optimizer_stats = self._build_optimizer_collection(
            {"critic": self.critic},
            prefer_muon=self.device.type == "cuda",
        )
        self.grad_scaler = build_grad_scaler(self.use_amp)
        self.lr_scheduler = KLAdaptiveLR(self.actor_optimizer, 0.01, min_lr = 5e-6)
        self._restore_resume_checkpoint(startup_timer)
        self._sync_sampler_global_step()
        self.start_update_count = int(self.update_count)
        self.start_global_step = int(self.global_step)

        self.rollout_buffer = ReplayBuffer(self.cfg.scene.num_envs, self.steps)
        self.policy_storage_specs = get_policy_storage_specs(
            self.obs_dims,
            self.actor_type,
            actor_kwargs=self.actor_kwargs,
        )
        self.policy_batch_keys = list(self.policy_storage_specs)
        self.critic_storage_specs = self.critic.observation_storage_specs()
        self.critic_batch_keys = list(self.critic_storage_specs)
        self.batch_keys = [
            *self.policy_batch_keys,
            *self.critic_batch_keys,
            "actions",
            "log_probs",
            "rewards",
            "values",
            "returns",
            "advantages",
        ]
        for key, shape in self.policy_storage_specs.items():
            self.rollout_buffer.create_storage_space(key, shape, torch.float32)
        for key, shape in self.critic_storage_specs.items():
            self.rollout_buffer.create_storage_space(key, shape, torch.float32)
        self.rollout_buffer.create_storage_space("actions", (self.cfg.action_space,), torch.float32)
        self.rollout_buffer.create_storage_space("log_probs", (), torch.float32)
        self.rollout_buffer.create_storage_space("rewards", (), torch.float32)
        self.rollout_buffer.create_storage_space("values", (), torch.float32)
        self.rollout_buffer.create_storage_space("terminate", (), torch.float32)

        self.tracker = MetricsTracker()
        self.tracker.add_batch_metrics("episode_return", self.cfg.scene.num_envs)
        self.tracker.add_batch_metrics("episode_length", self.cfg.scene.num_envs)
        self.tracker.add_list_metrics("policy_loss")
        self.tracker.add_list_metrics("entropy_loss")
        self.tracker.add_list_metrics("kl_divergence")
        self.tracker.add_list_metrics("value_loss")
        self.tracker.add_list_metrics("policy_clip_fraction")
        self.tracker.add_list_metrics("action_log_std")
        self.tracker.add_list_metrics("action_std")
        self.tracker.add_list_metrics("advantage_mean")
        self.tracker.add_list_metrics("advantage_std")
        self.tracker.add_list_metrics("value_explained_variance")
        self.tracker.add_list_metrics("value_clip_fraction")

        self.use_wandb = bool(config.use_wandb)
        if self.use_wandb:
            startup_timer.log("initializing W&B logger")
            WandbLogger.init_project("Mimic", self.run_name)
        startup_timer.log("ready to enter PPO loop")

        print(
            "actor optimizer split:",
            f"Muon={actor_optimizer_stats['muon_tensors']} tensors / {actor_optimizer_stats['muon_numel']} params,",
            f"AdamW={actor_optimizer_stats['adamw_tensors']} tensors / {actor_optimizer_stats['adamw_numel']} params",
        )
        print(
            "critic optimizer split:",
            f"Muon={critic_optimizer_stats['muon_tensors']} tensors / {critic_optimizer_stats['muon_numel']} params,",
            f"AdamW={critic_optimizer_stats['adamw_tensors']} tensors / {critic_optimizer_stats['adamw_numel']} params",
        )

    @staticmethod
    def _sized_optional_sequence_length(value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, str):
            return 1
        try:
            return len(tuple(value))
        except TypeError:
            return None

    @staticmethod
    def _infer_critic_key_body_count(
        *,
        critic_obs_dim: int,
        action_dim: int,
        observation_window_lengths: Mapping[str, int],
        max_key_body_count: int = 128,
    ) -> int | None:
        for key_body_count in range(max_key_body_count + 1):
            layout = build_critic_privilege_layout(
                action_dim,
                key_body_count=key_body_count,
                observation_window_lengths=observation_window_lengths,
            )
            if layout.obs_dim == int(critic_obs_dim):
                return key_body_count
        return None

    @classmethod
    def _resolve_critic_key_body_count(
        cls,
        cfg: Any,
        env: Any,
        *,
        critic_obs_dim: int,
        action_dim: int,
        observation_window_lengths: Mapping[str, int],
    ) -> int:
        cfg_count = cls._sized_optional_sequence_length(getattr(cfg, "key_body_names", None))
        if cfg_count is not None:
            return cfg_count

        env_unwrapped = getattr(env, "unwrapped", env)
        env_count = cls._sized_optional_sequence_length(getattr(env_unwrapped, "key_body_indices", None))
        if env_count is not None:
            return env_count

        inferred_count = cls._infer_critic_key_body_count(
            critic_obs_dim=critic_obs_dim,
            action_dim=action_dim,
            observation_window_lengths=observation_window_lengths,
        )
        return 0 if inferred_count is None else inferred_count

    @staticmethod
    def _checkpoint_critic_meta(checkpoint: CheckpointV2) -> Mapping[str, Any] | None:
        critic_meta = checkpoint.meta.get("critic")
        return critic_meta if isinstance(critic_meta, Mapping) else None

    def _build_critic_model(self) -> Critic:
        action_dim: int | None = self.cfg.action_space
        key_body_count = self.critic_key_body_count

        if self.resume_checkpoint is not None and self.resume_checkpoint.training:
            critic_meta = self._checkpoint_critic_meta(self.resume_checkpoint)
            critic_type = "flat" if critic_meta is None else str(critic_meta.get("critic_type", "flat"))
            if critic_type == "flat":
                action_dim = None
                key_body_count = int(critic_meta.get("key_body_count", 0)) if critic_meta is not None else 0
            else:
                key_body_count = int(critic_meta.get("key_body_count", key_body_count))

        self.critic_key_body_count = key_body_count
        return Critic(
            self.obs_dims["critic"],
            action_dim=action_dim,
            key_body_count=key_body_count,
            observation_window_lengths=self.observation_window_lengths,
        )

    @staticmethod
    def _build_config_actor_kwargs(config: RunConfig) -> dict[str, int | str]:
        return {
            "num_blocks": config.num_blocks,
            "robot_window_length": config.robot_window_length,
            "robot_encoder_type": config.robot_encoder_type,
            "motion_window_length": config.motion_window_length,
            "motion_encoder_type": config.motion_encoder_type,
            "actor_fusion_type": config.actor_fusion_type,
        }

    @classmethod
    def _resolve_actor_config(
        cls,
        config: RunConfig,
        checkpoint: CheckpointV2 | None,
    ) -> tuple[ActorType, dict[str, int | str]]:
        if checkpoint is None:
            return ActorType.FILM_RES, cls._build_config_actor_kwargs(config)

        return resolve_checkpoint_actor_spec(checkpoint)

    @staticmethod
    def _resolve_motion_file_inputs(
        config: RunConfig,
        checkpoint: CheckpointV2 | None,
    ) -> list[str] | None:
        if config.motion_files is not None:
            return list(config.motion_files)
        if checkpoint is None:
            return None
        checkpoint_motion_files = checkpoint.motion_files
        return checkpoint_motion_files or None

    def _build_actor_kwargs(self) -> dict[str, int | str]:
        return dict(self.actor_config_kwargs)

    def _format_sampler_config_summary(self) -> str:
        env_unwrapped = getattr(getattr(self, "env", None), "unwrapped", None)
        sampler = getattr(env_unwrapped, "sampler", None)
        num_anchors = getattr(sampler, "num_bins", None)
        num_anchors_text = "unknown" if num_anchors is None else str(int(num_anchors))
        return (
            "sampler ready: "
            f"strategy={self.sampling_strategy} "
            f"source={self.segment_source} "
            f"motions={len(self.motion_files)} "
            f"anchors={num_anchors_text} "
            f"weight_fail={float(getattr(self.cfg, 'weight_fail', 0.0)):.3g} "
            f"weight_novel={float(getattr(self.cfg, 'weight_novel', 0.0)):.3g} "
            f"adaptive_alpha={float(getattr(self.cfg, 'adaptive_alpha', 0.0)):.3g} "
            f"adaptive_uniform={float(getattr(self.cfg, 'adaptive_uniform_ratio', 0.0)):.3g} "
            f"warmup_s={float(getattr(self.cfg, 'motion_sampling_warmup_s', 0.0)):.3g} "
            f"ramp_s={float(getattr(self.cfg, 'motion_sampling_ramp_s', 0.0)):.3g}"
        )

    @staticmethod
    def _normalize_choice_name(value: Any) -> str | None:
        if value is None:
            return None

        text = str(getattr(value, "name", value)).split(".")[-1].replace("-", "_")
        normalized: list[str] = []
        for index, char in enumerate(text):
            if char.isupper() and index > 0 and normalized and normalized[-1] != "_" and not text[index - 1].isupper():
                normalized.append("_")
            normalized.append(char.lower())
        return "".join(normalized)

    @staticmethod
    def _validate_end_effector_termination_curriculum_config(config: RunConfig) -> None:
        if config.rollout_steps < 1:
            raise ValueError("rollout_steps must be positive.")
        if config.num_updates < 1:
            raise ValueError("num_updates must be positive.")

        warmup_fraction = float(config.end_effector_termination_warmup_fraction)
        deadline_fraction = float(config.end_effector_termination_deadline_fraction)
        if not 0.0 <= warmup_fraction <= deadline_fraction <= 1.0:
            raise ValueError(
                "end-effector termination curriculum fractions must satisfy "
                "0 <= warmup_fraction <= deadline_fraction <= 1."
            )

        start_threshold = float(config.end_effector_termination_start_threshold)
        end_threshold = float(config.end_effector_termination_end_threshold)
        if start_threshold <= 0.0 or end_threshold <= 0.0:
            raise ValueError("end-effector termination thresholds must be positive.")
        if start_threshold < end_threshold:
            raise ValueError("end-effector termination start threshold must be >= end threshold.")
        if float(config.end_effector_termination_tighten_step) <= 0.0:
            raise ValueError("end-effector termination tighten step must be positive.")
        if int(config.end_effector_termination_ema_updates) < 1:
            raise ValueError("end-effector termination EMA update window must be positive.")
        if int(config.end_effector_termination_min_ema_samples) < 1:
            raise ValueError("end-effector termination min EMA samples must be positive.")
        if int(config.end_effector_termination_hold_updates) < 0:
            raise ValueError("end-effector termination hold updates must be non-negative.")
        if float(config.end_effector_termination_max_terminate_rate) < 0.0:
            raise ValueError("end-effector termination max terminate rate must be non-negative.")
        if float(config.end_effector_termination_error_margin) < 0.0:
            raise ValueError("end-effector termination error margin must be non-negative.")
        sampler_warmup_fraction = float(config.sampler_failure_warmup_fraction)
        if not 0.0 <= sampler_warmup_fraction <= 1.0:
            raise ValueError("sampler failure warmup fraction must be in [0, 1].")

    @staticmethod
    def _sampler_failure_warmup_steps(config: RunConfig) -> int:
        total_steps = int(config.rollout_steps) * int(config.num_updates)
        warmup_fraction = float(config.sampler_failure_warmup_fraction)
        if warmup_fraction <= 0.0:
            return 0
        return int(math.ceil(float(total_steps) * warmup_fraction))

    @staticmethod
    def _build_end_effector_threshold_stages(
        *,
        start_threshold: float,
        end_threshold: float,
        tighten_step: float,
    ) -> tuple[float, ...]:
        start_threshold = float(start_threshold)
        end_threshold = float(end_threshold)
        tighten_step = float(tighten_step)
        if start_threshold <= end_threshold:
            return (end_threshold,)

        thresholds = [start_threshold]
        current = start_threshold
        while current - tighten_step > end_threshold:
            current = round(current - tighten_step, 10)
            thresholds.append(current)
        if thresholds[-1] != end_threshold:
            thresholds.append(end_threshold)
        return tuple(thresholds)

    @staticmethod
    def _ema_alpha(window_updates: int) -> float:
        return 2.0 / float(max(1, int(window_updates)) + 1)

    @staticmethod
    def _update_ema(previous: float | None, value: float, alpha: float) -> float:
        if previous is None:
            return float(value)
        return float(previous) + float(alpha) * (float(value) - float(previous))

    @staticmethod
    def _coerce_optional_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(scalar):
            return None
        return scalar

    @staticmethod
    def _restore_curriculum_state_payload(checkpoint: CheckpointV2 | None) -> Mapping[str, Any] | None:
        if checkpoint is None or not checkpoint.training:
            return None
        training_state = dict(checkpoint.training)
        if TrainRunner._training_state_missing_keys(training_state):
            return None
        payload = training_state.get(END_EFFECTOR_TERMINATION_STATE_KEY)
        return payload if isinstance(payload, Mapping) else None

    @classmethod
    def _build_end_effector_termination_curriculum_state(
        cls,
        config: RunConfig,
        *,
        checkpoint: CheckpointV2 | None = None,
    ) -> EndEffectorTerminationCurriculumState:
        cls._validate_end_effector_termination_curriculum_config(config)

        start_threshold = float(config.end_effector_termination_start_threshold)
        end_threshold = float(config.end_effector_termination_end_threshold)
        thresholds = cls._build_end_effector_threshold_stages(
            start_threshold=start_threshold,
            end_threshold=end_threshold,
            tighten_step=float(config.end_effector_termination_tighten_step),
        )
        ema_alpha = cls._ema_alpha(int(config.end_effector_termination_ema_updates))

        if not config.end_effector_termination_curriculum_enabled:
            return EndEffectorTerminationCurriculumState(
                enabled=False,
                thresholds=(end_threshold,),
                stage_index=0,
                current_threshold=end_threshold,
                warmup_fraction=float(config.end_effector_termination_warmup_fraction),
                deadline_fraction=float(config.end_effector_termination_deadline_fraction),
                ema_alpha=ema_alpha,
                end_threshold=end_threshold,
                tighten_step=float(config.end_effector_termination_tighten_step),
                gate_reason="disabled",
            )

        restored = cls._restore_curriculum_state_payload(checkpoint)
        stage_index = 0
        ema_terminate_rate = None
        ema_error_mean = None
        ema_sample_count = 0
        ema_error_sample_count = 0
        last_tighten_update = 0
        gate_reason = "initial"
        if restored is not None:
            stage_index = min(max(0, int(restored.get("stage_index", 0))), len(thresholds) - 1)
            ema_terminate_rate = cls._coerce_optional_float(restored.get("ema_terminate_rate"))
            ema_error_mean = cls._coerce_optional_float(restored.get("ema_error_mean"))
            ema_sample_count = max(0, int(restored.get("ema_sample_count", 0)))
            ema_error_sample_count = max(0, int(restored.get("ema_error_sample_count", 0)))
            last_tighten_update = max(0, int(restored.get("last_tighten_update", 0)))
            gate_reason = "restored"

        return EndEffectorTerminationCurriculumState(
            enabled=True,
            thresholds=thresholds,
            stage_index=stage_index,
            current_threshold=float(thresholds[stage_index]),
            warmup_fraction=float(config.end_effector_termination_warmup_fraction),
            deadline_fraction=float(config.end_effector_termination_deadline_fraction),
            ema_alpha=ema_alpha,
            end_threshold=end_threshold,
            tighten_step=float(config.end_effector_termination_tighten_step),
            ema_terminate_rate=ema_terminate_rate,
            ema_error_mean=ema_error_mean,
            ema_sample_count=ema_sample_count,
            ema_error_sample_count=ema_error_sample_count,
            last_tighten_update=last_tighten_update,
            gate_reason=gate_reason,
        )

    @staticmethod
    def _build_make_training_env_kwargs(
        make_training_env,
        *,
        window_lengths: Mapping[str, int],
        motion_files: list[str] | None,
        sampler_failure_warmup_steps: int | None = None,
        end_effector_termination_curriculum: EndEffectorTerminationCurriculumState | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"window_lengths": window_lengths}
        if sampler_failure_warmup_steps is not None:
            kwargs["sampler_failure_warmup_steps"] = int(sampler_failure_warmup_steps)
        if end_effector_termination_curriculum is not None:
            kwargs.update(
                {
                    "end_effector_termination_curriculum_enabled": (
                        end_effector_termination_curriculum.enabled
                    ),
                    "end_effector_termination_initial_threshold": (
                        end_effector_termination_curriculum.current_threshold
                    ),
                    "end_effector_termination_end_threshold": (
                        end_effector_termination_curriculum.end_threshold
                    ),
                }
            )

        if motion_files is not None:
            kwargs["motion_files"] = motion_files
        return kwargs

    @staticmethod
    def _find_end_effector_failure_rule(termination_model: Any) -> Any | None:
        getter = getattr(termination_model, "get_failure_rule", None)
        if callable(getter):
            try:
                return getter(END_EFFECTOR_TERMINATION_RULE_ID)
            except (KeyError, ValueError, AttributeError):
                pass

        for rule in getattr(termination_model, "failure_rules", ()) or ():
            if str(getattr(rule, "id", "")) == END_EFFECTOR_TERMINATION_RULE_ID:
                return rule
        return None

    @classmethod
    def _set_runtime_end_effector_termination_threshold(cls, env: Any, threshold: float) -> bool:
        env_unwrapped = getattr(env, "unwrapped", env)
        threshold_value = float(threshold)
        updated = False

        termination_model = getattr(env_unwrapped, "termination_model", None)
        if termination_model is not None:
            rule = cls._find_end_effector_failure_rule(termination_model)
            if rule is not None and hasattr(rule, "threshold"):
                rule.threshold = threshold_value
                updated = True

        curriculum = getattr(env_unwrapped, "termination_curriculum", None)
        if curriculum is not None:
            for attr_name in ("_base_values", "_current_values"):
                values = getattr(curriculum, attr_name, None)
                if isinstance(values, dict) and END_EFFECTOR_TERMINATION_RULE_ID in values:
                    values[END_EFFECTOR_TERMINATION_RULE_ID] = threshold_value
                    updated = True
            rules = getattr(curriculum, "_rules", None)
            if isinstance(rules, dict):
                rule = rules.get(END_EFFECTOR_TERMINATION_RULE_ID)
                if rule is not None and hasattr(rule, "threshold"):
                    rule.threshold = threshold_value
                    updated = True

        return updated

    def _extract_end_effector_termination_error_mean(self) -> float | None:
        env_unwrapped = getattr(self.env, "unwrapped", self.env)
        extras = getattr(env_unwrapped, "extras", None)
        if isinstance(extras, Mapping):
            scalar = self._coerce_metric_scalar(extras.get("curriculum/end_effector/error_mean"))
            if scalar is not None:
                return scalar

        termination_model = getattr(env_unwrapped, "termination_model", None)
        if termination_model is None:
            return None
        rule = self._find_end_effector_failure_rule(termination_model)
        if rule is None or not callable(getattr(rule, "error", None)):
            return None
        context_builder = getattr(termination_model, "build_context", None)
        if not callable(context_builder):
            return None

        try:
            context = context_builder(
                getattr(env_unwrapped, "episode_length_buf"),
                getattr(env_unwrapped, "max_episode_length"),
                getattr(env_unwrapped, "robot"),
                getattr(env_unwrapped, "reference_motion"),
                getattr(env_unwrapped, "sampler"),
            )
            error = rule.error(context)
        except (AttributeError, TypeError, RuntimeError, ValueError):
            return None
        return self._coerce_metric_scalar(error)

    @staticmethod
    def _deadline_stage_index(
        state: EndEffectorTerminationCurriculumState,
        *,
        update_count: int,
        num_updates: int,
    ) -> int:
        final_stage = len(state.thresholds) - 1
        if final_stage <= 0:
            return 0
        deadline_update = int(math.ceil(float(num_updates) * state.deadline_fraction))
        if update_count < deadline_update:
            return 0
        remaining_updates = max(1, int(num_updates) - deadline_update)
        progress = min(1.0, max(0.0, (int(update_count) - deadline_update) / remaining_updates))
        return min(final_stage, int(math.ceil(progress * final_stage)))

    def _update_end_effector_curriculum_ema(
        self,
        *,
        terminate_rate: float,
        error_mean: float | None,
    ) -> None:
        state = self.end_effector_termination_curriculum
        state.ema_terminate_rate = self._update_ema(
            state.ema_terminate_rate,
            terminate_rate,
            state.ema_alpha,
        )
        state.ema_sample_count += 1
        if error_mean is not None:
            state.ema_error_mean = self._update_ema(
                state.ema_error_mean,
                error_mean,
                state.ema_alpha,
            )
            state.ema_error_sample_count += 1

    def _normal_end_effector_gate_passes(self, next_threshold: float) -> tuple[bool, str]:
        state = self.end_effector_termination_curriculum
        config = self.config
        warmup_update = int(math.ceil(int(config.num_updates) * state.warmup_fraction))
        if self.update_count < warmup_update:
            return False, "warmup"

        updates_since_tighten = self.update_count - int(state.last_tighten_update)
        if updates_since_tighten < int(config.end_effector_termination_hold_updates):
            return False, "min_hold"
        if state.ema_sample_count < int(config.end_effector_termination_min_ema_samples):
            return False, "insufficient_samples"

        terminate_ema = state.ema_terminate_rate
        if terminate_ema is None or terminate_ema > float(config.end_effector_termination_max_terminate_rate):
            return False, "terminate_rate"

        if state.ema_error_mean is None:
            if bool(config.end_effector_termination_allow_error_fallback):
                return True, "passed_terminate_only"
            return False, "missing_error"
        if state.ema_error_sample_count < int(config.end_effector_termination_min_ema_samples):
            return False, "insufficient_error_samples"

        error_limit = float(next_threshold) * float(config.end_effector_termination_error_margin)
        if state.ema_error_mean > error_limit:
            return False, "error_mean"
        return True, "passed"

    def _end_effector_curriculum_metrics_payload(
        self,
        *,
        rollout_terminate_rate: float | None = None,
        rollout_error_mean: float | None = None,
    ) -> dict[str, float]:
        state = self.end_effector_termination_curriculum
        next_stage = min(state.stage_index + 1, len(state.thresholds) - 1)
        payload = {
            "curriculum/end_effector_position_failure": float(state.current_threshold),
            "curriculum/end_effector/current_threshold": float(state.current_threshold),
            "curriculum/end_effector/stage_index": float(state.stage_index),
            "curriculum/end_effector/stage_count": float(len(state.thresholds)),
            "curriculum/end_effector/next_threshold": float(state.thresholds[next_stage]),
            "curriculum/end_effector/gate_pass": float(state.gate_pass),
            "curriculum/end_effector/deadline_forced": float(state.deadline_forced),
            "curriculum/end_effector/ema_sample_count": float(state.ema_sample_count),
            "curriculum/end_effector/ema_error_sample_count": float(state.ema_error_sample_count),
            "curriculum/end_effector/updates_since_tighten": float(
                self.update_count - int(state.last_tighten_update)
            ),
        }
        if state.ema_terminate_rate is not None:
            payload["curriculum/end_effector/ema_terminate_rate"] = float(state.ema_terminate_rate)
        if state.ema_error_mean is not None:
            payload["curriculum/end_effector/ema_error_mean"] = float(state.ema_error_mean)
        if rollout_terminate_rate is not None:
            payload["curriculum/end_effector/terminate_rate"] = float(rollout_terminate_rate)
        if rollout_error_mean is not None:
            payload["curriculum/end_effector/error_mean"] = float(rollout_error_mean)
        return payload

    def _advance_end_effector_termination_curriculum(self) -> None:
        state = self.end_effector_termination_curriculum
        if not state.enabled:
            return

        rollout_metrics = getattr(self, "_last_end_effector_curriculum_rollout_metrics", {})
        terminate_rate = self._coerce_optional_float(rollout_metrics.get("terminate_rate"))
        if terminate_rate is None:
            return
        error_mean = self._coerce_optional_float(rollout_metrics.get("error_mean"))
        self._update_end_effector_curriculum_ema(
            terminate_rate=terminate_rate,
            error_mean=error_mean,
        )

        state.gate_pass = False
        state.deadline_forced = False
        final_stage = len(state.thresholds) - 1
        if state.stage_index >= final_stage:
            state.gate_reason = "final"
            self._log_metrics(
                self._end_effector_curriculum_metrics_payload(
                    rollout_terminate_rate=terminate_rate,
                    rollout_error_mean=error_mean,
                )
            )
            return

        deadline_stage = self._deadline_stage_index(
            state,
            update_count=self.update_count,
            num_updates=int(self.config.num_updates),
        )
        target_stage = state.stage_index
        if deadline_stage > state.stage_index:
            target_stage = deadline_stage
            state.deadline_forced = True
            state.gate_reason = "deadline"
        else:
            next_threshold = state.thresholds[state.stage_index + 1]
            gate_pass, reason = self._normal_end_effector_gate_passes(next_threshold)
            state.gate_pass = gate_pass
            state.gate_reason = reason
            if gate_pass:
                target_stage = state.stage_index + 1

        if target_stage > state.stage_index:
            state.stage_index = min(target_stage, final_stage)
            state.current_threshold = float(state.thresholds[state.stage_index])
            state.last_tighten_update = int(self.update_count)
            self._set_runtime_end_effector_termination_threshold(self.env, state.current_threshold)

        self._log_metrics(
            self._end_effector_curriculum_metrics_payload(
                rollout_terminate_rate=terminate_rate,
                rollout_error_mean=error_mean,
            )
        )

    def _end_effector_curriculum_state_dict(self) -> dict[str, Any]:
        state = self.end_effector_termination_curriculum
        return {
            "enabled": bool(state.enabled),
            "thresholds": list(state.thresholds),
            "stage_index": int(state.stage_index),
            "current_threshold": float(state.current_threshold),
            "warmup_fraction": float(state.warmup_fraction),
            "deadline_fraction": float(state.deadline_fraction),
            "end_threshold": float(state.end_threshold),
            "tighten_step": float(state.tighten_step),
            "ema_terminate_rate": state.ema_terminate_rate,
            "ema_error_mean": state.ema_error_mean,
            "ema_sample_count": int(state.ema_sample_count),
            "ema_error_sample_count": int(state.ema_error_sample_count),
            "last_tighten_update": int(state.last_tighten_update),
            "gate_reason": str(state.gate_reason),
        }

    @staticmethod
    def _split_optimizer_param_groups(
        modules: dict[str, torch.nn.Module],
        *,
        prefer_muon: bool = True,
    ) -> tuple[list[dict], list[dict], dict[str, int]]:
        muon_groups = []
        adamw_groups = []
        stats = {
            "muon_tensors": 0,
            "muon_numel": 0,
            "adamw_tensors": 0,
            "adamw_numel": 0,
        }

        for module_name, module in modules.items():
            muon_params = []
            adamw_params = []

            for _, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                if prefer_muon and param.ndim == 2:
                    muon_params.append(param)
                    stats["muon_tensors"] += 1
                    stats["muon_numel"] += param.numel()
                else:
                    adamw_params.append(param)
                    stats["adamw_tensors"] += 1
                    stats["adamw_numel"] += param.numel()

            if muon_params:
                muon_groups.append({"params": muon_params, "name": module_name})
            if adamw_params:
                adamw_groups.append({"params": adamw_params, "name": f"{module_name}_adamw"})

        return muon_groups, adamw_groups, stats

    @classmethod
    def _build_optimizer_collection(
        cls,
        modules: dict[str, torch.nn.Module],
        *,
        prefer_muon: bool = True,
    ) -> tuple[OptimizerCollection, dict[str, int]]:
        muon_groups, adamw_groups, stats = cls._split_optimizer_param_groups(
            modules,
            prefer_muon=prefer_muon,
        )
        optimizers = []
        if muon_groups:
            optimizers.append(torch.optim.Muon(muon_groups, lr=1e-3, weight_decay=0.0))
        if adamw_groups:
            optimizers.append(torch.optim.AdamW(adamw_groups, lr=1e-3, weight_decay=0.0))
        return OptimizerCollection(*optimizers), stats

    @staticmethod
    def _optimizer_lr(optimizer: torch.optim.Optimizer) -> float:
        if not optimizer.param_groups:
            return 0.0
        return float(optimizer.param_groups[0]["lr"])

    @staticmethod
    def _training_state_missing_keys(training_state: Mapping[str, Any]) -> list[str]:
        return [key for key in REQUIRED_TRAINER_STATE_KEYS if key not in training_state]

    @staticmethod
    def _list_to_tuple(value: Any) -> Any:
        if isinstance(value, list):
            return tuple(TrainRunner._list_to_tuple(item) for item in value)
        return value

    @staticmethod
    def _capture_rng_state() -> dict[str, Any]:
        rng_state: dict[str, Any] = {
            "torch_cpu": torch.get_rng_state(),
            "python": random.getstate(),
        }

        numpy_state = np.random.get_state()
        rng_state["numpy"] = {
            "algorithm": str(numpy_state[0]),
            "state": numpy_state[1].astype(np.uint32, copy=False).tolist(),
            "pos": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        }

        if torch.cuda.is_available():
            rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()

        return rng_state

    @staticmethod
    def _restore_rng_state(rng_state: Mapping[str, Any] | None) -> None:
        if not isinstance(rng_state, Mapping):
            return

        torch_cpu_state = rng_state.get("torch_cpu")
        if torch_cpu_state is not None:
            torch.set_rng_state(torch.as_tensor(torch_cpu_state, dtype=torch.uint8).cpu())

        torch_cuda_state = rng_state.get("torch_cuda")
        if torch_cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(
                [torch.as_tensor(state, dtype=torch.uint8).cpu() for state in torch_cuda_state]
            )

        numpy_state = rng_state.get("numpy")
        if isinstance(numpy_state, Mapping):
            np.random.set_state(
                (
                    str(numpy_state["algorithm"]),
                    np.asarray(numpy_state["state"], dtype=np.uint32),
                    int(numpy_state["pos"]),
                    int(numpy_state["has_gauss"]),
                    float(numpy_state["cached_gaussian"]),
                )
            )

        python_state = rng_state.get("python")
        if python_state is not None:
            random.setstate(TrainRunner._list_to_tuple(python_state))

    def _build_training_state(self) -> dict[str, Any]:
        return {
            "update_count": int(self.update_count),
            "global_step": int(self.global_step),
            "sampler_failure_warmup_steps": int(self.sampler_failure_warmup_steps),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "grad_scaler": self.grad_scaler.state_dict(),
            "amp": {
                "requested": bool(self.requested_amp),
                "enabled": bool(self.use_amp),
                "dtype": self.amp_dtype,
            },
            END_EFFECTOR_TERMINATION_STATE_KEY: self._end_effector_curriculum_state_dict(),
            "rng_state": self._capture_rng_state(),
        }

    def _restore_resume_checkpoint(self, startup_timer: StartupTimer) -> None:
        if self.resume_checkpoint is None:
            return

        checkpoint = self.resume_checkpoint
        startup_timer.log("restoring actor and critic weights from resume checkpoint")
        actor_incompatible_keys = load_actor_checkpoint_state(self.actor, checkpoint.model["actor"])
        if actor_incompatible_keys.missing_keys or actor_incompatible_keys.unexpected_keys:
            print(
                "resume checkpoint actor was loaded across the encoder redesign; "
                "compatible legacy encoder keys were remapped and new policy-owned pooling/projection "
                "weights were initialized where no compatible checkpoint weights existed.",
                flush=True,
            )
        training_state = dict(checkpoint.training)

        critic_weights_restored = True
        try:
            self.critic.load_state_dict(checkpoint.model["critic"])
        except RuntimeError as exc:
            if training_state:
                raise
            critic_weights_restored = False
            print(
                "resume checkpoint critic weights are incompatible with the current critic architecture; "
                "warm-starting actor only and keeping a freshly initialized critic. "
                f"Reason: {self._format_failure_reason(exc)}",
                flush=True,
            )

        if not training_state:
            self.resume_mode = "warm_start"
            warm_start_message = (
                "resume checkpoint does not contain trainer state; warm-starting actor/critic with fresh "
                "optimizers, scheduler, scaler, and counters."
            )
            if not critic_weights_restored:
                warm_start_message = (
                    "resume checkpoint does not contain trainer state; warm-starting actor with a fresh critic, "
                    "optimizers, scheduler, scaler, and counters."
                )
            print(warm_start_message, flush=True)
            return

        missing_keys = self._training_state_missing_keys(training_state)
        if missing_keys:
            raise ValueError(
                "Resume checkpoint contains incomplete trainer state. "
                f"Missing keys: {', '.join(missing_keys)}."
            )

        self.actor_optimizer.load_state_dict(training_state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(training_state["critic_optimizer"])
        self.lr_scheduler.load_state_dict(training_state["lr_scheduler"])
        self.grad_scaler.load_state_dict(training_state["grad_scaler"])
        self.update_count = int(training_state["update_count"])
        self.global_step = int(training_state["global_step"])
        self._restore_rng_state(training_state.get("rng_state"))
        self.resume_mode = "full_state"
        self.resume_trainer_state_restored = True
        startup_timer.log(
            f"restored trainer state at update={self.update_count} global_step={self.global_step}"
        )

    def _sync_sampler_global_step(self) -> None:
        sampler = getattr(getattr(self.env, "unwrapped", None), "sampler", None)
        if sampler is not None and hasattr(sampler, "_global_step"):
            sampler._global_step = int(self.global_step)

    def _log_metrics(self, payload: dict[str, float]) -> None:
        if self.use_wandb:
            WandbLogger.log_metrics(payload, self.global_step)

    @staticmethod
    def _iter_metric_items(payload: Mapping[str, Any], prefix: str = ""):
        for key, value in payload.items():
            name = str(key)
            metric_name = f"{prefix}/{name}" if prefix else name
            if isinstance(value, Mapping):
                yield from TrainRunner._iter_metric_items(value, metric_name)
            else:
                yield metric_name, value

    @staticmethod
    def _coerce_metric_scalar(value: Any) -> float | None:
        try:
            tensor = torch.as_tensor(value, dtype=torch.float32)
        except (TypeError, ValueError):
            return None
        if tensor.numel() == 0:
            return None
        if tensor.device.type != "cpu":
            tensor = tensor.detach().to(device="cpu")
        if not torch.isfinite(tensor).all():
            return None
        return float(tensor.mean().item())

    @classmethod
    def _extract_curriculum_metric_sample_from_mapping(cls, payload: Mapping[str, Any] | None) -> dict[str, float]:
        if not isinstance(payload, Mapping):
            return {}

        sample: dict[str, float] = {}
        for metric_name, value in cls._iter_metric_items(payload):
            normalized_name = metric_name.strip("/")
            prefix_index = normalized_name.find(CURRICULUM_METRIC_PREFIX)
            if prefix_index < 0:
                continue
            scalar = cls._coerce_metric_scalar(value)
            if scalar is None:
                continue
            sample[normalized_name[prefix_index:]] = scalar
        return sample

    def _extract_curriculum_metric_sample(self, info: Mapping[str, Any] | None) -> dict[str, float]:
        sample = self._extract_curriculum_metric_sample_from_mapping(info)
        extras = getattr(self.env.unwrapped, "extras", None)
        sample.update(self._extract_curriculum_metric_sample_from_mapping(extras))
        return sample

    @staticmethod
    def _build_episode_metrics_payload(mean_return: float, mean_length: float) -> dict[str, float]:
        return {
            "episode/returns": float(mean_return),
            "episode/lengths": float(mean_length),
        }

    @staticmethod
    def _build_episode_finish_metrics_payload(terminate: Any, timeout: Any) -> dict[str, float]:
        terminated = torch.as_tensor(terminate, dtype=torch.bool)
        timed_out = torch.as_tensor(timeout, dtype=torch.bool)
        if terminated.shape != timed_out.shape:
            raise ValueError("terminate and timeout tensors must have the same shape.")

        done = terminated | timed_out
        done_count = float(done.sum().detach().to(device="cpu").item())
        if done_count <= 0.0:
            return {}

        terminate_count = float(terminated[done].sum().detach().to(device="cpu").item())
        timeout_count = float(timed_out[done].sum().detach().to(device="cpu").item())
        return {
            "episode/finished_count": done_count,
            "episode/terminate_count": terminate_count,
            "episode/timeout_count": timeout_count,
            "episode/terminate_ratio": terminate_count / done_count,
            "episode/timeout_ratio": timeout_count / done_count,
        }

    @staticmethod
    def _extract_relative_anchor_pos_sample(
        privilege_obs: Any,
        *,
        action_dim: int,
    ) -> torch.Tensor | None:
        try:
            privilege = torch.as_tensor(privilege_obs)
        except (TypeError, ValueError, RuntimeError):
            return None
        if privilege.ndim < 1:
            return None

        try:
            resolved_action_dim = int(action_dim)
        except (TypeError, ValueError):
            return None
        if resolved_action_dim < 1:
            return None

        # GMTP privilege order: target projected gravity, target joint pos, target joint vel, relative anchor pos.
        relative_anchor_pos_start = 3 + 2 * resolved_action_dim
        relative_anchor_pos_stop = relative_anchor_pos_start + RELATIVE_ANCHOR_POS_DIM
        if int(privilege.shape[-1]) < relative_anchor_pos_stop:
            return None

        relative_anchor_pos = privilege[..., relative_anchor_pos_start:relative_anchor_pos_stop].detach()
        if relative_anchor_pos.numel() == 0:
            return None
        if not bool(torch.isfinite(relative_anchor_pos).all().item()):
            return None
        return relative_anchor_pos

    @staticmethod
    def _build_location_tracking_metrics(relative_anchor_pos_samples: list[torch.Tensor]) -> dict[str, float]:
        if not relative_anchor_pos_samples:
            return {}

        flattened_samples: list[torch.Tensor] = []
        for sample in relative_anchor_pos_samples:
            sample = sample.detach()
            if sample.ndim < 1 or int(sample.shape[-1]) != RELATIVE_ANCHOR_POS_DIM or sample.numel() == 0:
                return {}
            if not bool(torch.isfinite(sample).all().item()):
                return {}
            flattened_samples.append(sample.reshape(-1, RELATIVE_ANCHOR_POS_DIM).to(dtype=torch.float32))

        if not flattened_samples:
            return {}

        relative_anchor_pos = torch.cat(flattened_samples, dim=0)
        if relative_anchor_pos.numel() == 0:
            return {}

        location_error = torch.linalg.vector_norm(relative_anchor_pos, dim=-1)
        if location_error.numel() == 0 or not bool(torch.isfinite(location_error).all().item()):
            return {}

        xy_error = torch.linalg.vector_norm(relative_anchor_pos[:, :2], dim=-1)
        z_error = torch.abs(relative_anchor_pos[:, 2])
        return {
            "tracking/location_error_m": float(location_error.mean().detach().to(device="cpu").item()),
            "tracking/location_error_xy_m": float(xy_error.mean().detach().to(device="cpu").item()),
            "tracking/location_error_z_m": float(z_error.mean().detach().to(device="cpu").item()),
            "tracking/location_error_p95_m": float(
                torch.quantile(location_error, 0.95).detach().to(device="cpu").item()
            ),
            "tracking/location_error_max_m": float(location_error.max().detach().to(device="cpu").item()),
        }

    @staticmethod
    def _sanitize_metric_component(value: str) -> str:
        sanitized = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value)).strip("_")
        return sanitized or "unknown"

    @classmethod
    def _build_guarded_sampling_probabilities(
        cls,
        fail_counts: torch.Tensor,
        sample_counts: torch.Tensor,
        *,
        temperature: float,
        uniform_mix: float,
        eligible_mask: torch.Tensor | None = None,
        exploration_bonus: float = 0.0,
        max_uniform_ratio: float | None = None,
    ) -> torch.Tensor:
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0.")

        fail_counts = torch.as_tensor(fail_counts, dtype=torch.float32).reshape(-1)
        sample_counts = torch.as_tensor(sample_counts, dtype=torch.float32).reshape(-1)
        if fail_counts.shape != sample_counts.shape:
            raise ValueError("fail_counts and sample_counts must have the same shape.")

        if eligible_mask is None:
            eligible = torch.ones_like(fail_counts, dtype=torch.bool)
        else:
            eligible = torch.as_tensor(eligible_mask, dtype=torch.bool).reshape(-1)
            if eligible.shape != fail_counts.shape:
                raise ValueError("eligible_mask must have the same shape as fail_counts.")

        uniform_probs = eligible.to(dtype=torch.float32)
        uniform_sum = torch.sum(uniform_probs)
        if float(uniform_sum.item()) <= 0.0:
            raise ValueError("eligible_mask must include at least one entry.")
        uniform_probs = uniform_probs / uniform_sum

        eligible_sample_counts = torch.where(
            eligible,
            torch.clamp(sample_counts, min=0.0),
            torch.zeros_like(sample_counts),
        )
        eligible_count = torch.sum(eligible.to(dtype=torch.float32))
        total_eligible_samples = torch.sum(eligible_sample_counts)
        exploration_scale = torch.log(total_eligible_samples + eligible_count + 1.0)
        exploration = torch.sqrt(exploration_scale / (eligible_sample_counts + 1.0))
        fail_rate = fail_counts / torch.clamp(sample_counts, min=1.0)
        score = fail_rate + float(exploration_bonus) * exploration
        learned_weights = score.pow(1.0 / temperature)
        learned_weights = torch.where(eligible, learned_weights, torch.zeros_like(learned_weights))

        learned_sum = torch.sum(learned_weights)
        if bool(torch.all(torch.isfinite(learned_weights)).item()) and float(learned_sum.item()) > 0.0:
            learned_probs = learned_weights / learned_sum
        else:
            learned_probs = uniform_probs

        probs = (1.0 - uniform_mix) * learned_probs + uniform_mix * uniform_probs
        probs = torch.where(eligible, probs, torch.zeros_like(probs))
        probs_sum = torch.sum(probs)
        if float(probs_sum.item()) > 0.0:
            probs = probs / probs_sum
        else:
            probs = uniform_probs

        probs = cls._apply_max_uniform_probability_cap(
            probs,
            eligible_mask=eligible,
            uniform_probs=uniform_probs,
            max_uniform_ratio=max_uniform_ratio,
        )
        return probs / torch.clamp(torch.sum(probs), min=torch.finfo(probs.dtype).eps)

    @staticmethod
    def _apply_max_uniform_probability_cap(
        probs: torch.Tensor,
        *,
        eligible_mask: torch.Tensor,
        uniform_probs: torch.Tensor,
        max_uniform_ratio: float | None,
    ) -> torch.Tensor:
        probs = torch.as_tensor(probs, dtype=torch.float32).reshape(-1)
        eligible = torch.as_tensor(eligible_mask, dtype=torch.bool).reshape(-1)
        uniform_probs = torch.as_tensor(uniform_probs, dtype=torch.float32).reshape(-1)
        if probs.shape != eligible.shape or probs.shape != uniform_probs.shape:
            raise ValueError("Probability cap inputs must have the same shape.")

        probs = torch.where(eligible, probs, torch.zeros_like(probs))
        probs = probs / torch.clamp(torch.sum(probs), min=torch.finfo(probs.dtype).eps)
        if max_uniform_ratio is None:
            return probs

        resolved_max_uniform_ratio = float(max_uniform_ratio)
        if resolved_max_uniform_ratio < 1.0:
            raise ValueError("max_uniform_ratio must be None or >= 1.")

        eligible_count = int(torch.sum(eligible).item())
        if eligible_count == 0:
            raise ValueError("eligible_mask must include at least one entry.")

        max_prob = resolved_max_uniform_ratio / float(eligible_count)
        if max_prob >= 1.0:
            return probs

        eps = torch.finfo(probs.dtype).eps
        if not bool(torch.all(torch.isfinite(probs)).item()):
            return uniform_probs
        if float(torch.sum(probs).item()) <= eps:
            return uniform_probs
        if bool(torch.all(probs[eligible] <= max_prob + eps).item()):
            return probs

        eligible_probs = probs[eligible]
        lower = torch.min(eligible_probs - max_prob)
        upper = torch.max(eligible_probs)
        for _ in range(64):
            threshold = 0.5 * (lower + upper)
            projected_probs = torch.clamp(eligible_probs - threshold, min=0.0, max=max_prob)
            if float(torch.sum(projected_probs).item()) > 1.0:
                lower = threshold
            else:
                upper = threshold

        eligible_projected_probs = torch.clamp(eligible_probs - upper, min=0.0, max=max_prob)
        projected_sum = torch.sum(eligible_projected_probs)
        if not bool(torch.isfinite(projected_sum).item()) or float(projected_sum.item()) <= eps:
            return uniform_probs

        capped_probs = torch.zeros_like(probs)
        capped_probs[eligible] = eligible_projected_probs
        return capped_probs / projected_sum

    @staticmethod
    def _as_cpu_tensor(
        value: Any,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=dtype)
        if tensor.device.type != "cpu":
            tensor = tensor.detach().to(device="cpu")
        return tensor

    @classmethod
    def _build_sampler_sampling_probabilities(
        cls,
        sampler: Any,
        fail_counts: torch.Tensor,
        sample_counts: torch.Tensor,
        *,
        temperature: float,
        eligible_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sampler_builder = getattr(sampler, "_build_guarded_sampling_probabilities", None)
        if callable(sampler_builder):
            probabilities = sampler_builder(
                fail_counts,
                sample_counts,
                temperature=temperature,
                eligible_mask=eligible_mask,
            )
            return cls._as_cpu_tensor(probabilities, dtype=torch.float32).reshape(-1)

        bin_builder = getattr(sampler, "_build_bin_sampling_probabilities", None)
        if callable(bin_builder):
            probabilities = bin_builder(
                fail_counts,
                sample_counts,
                eligible_mask=eligible_mask,
            )
            return cls._as_cpu_tensor(probabilities, dtype=torch.float32).reshape(-1)

        return cls._build_guarded_sampling_probabilities(
            fail_counts,
            sample_counts,
            temperature=temperature,
            uniform_mix=float(getattr(sampler, "failure_weight_uniform_mix", 0.0)),
            eligible_mask=None if eligible_mask is None else cls._as_cpu_tensor(eligible_mask, dtype=torch.bool),
            exploration_bonus=float(getattr(sampler, "failure_weight_exploration_bonus", 0.0)),
            max_uniform_ratio=getattr(sampler, "failure_weight_max_uniform_ratio", None),
        )

    @staticmethod
    def _resolve_anchor_index(anchor_times: torch.Tensor, reset_time: torch.Tensor) -> int:
        matches = torch.nonzero(torch.isclose(anchor_times, reset_time, atol=1e-6, rtol=0.0), as_tuple=False)
        if matches.numel() > 0:
            return int(matches[0, 0].item())
        return int(torch.argmin(torch.abs(anchor_times - reset_time)).item())

    @classmethod
    def _compute_anchor_reset_probabilities(
        cls,
        sampler: Any,
        *,
        temperature: float,
    ) -> list[dict[str, float | int | str]]:
        motion_lib = getattr(sampler, "motion_lib", None)
        clips = getattr(motion_lib, "clips", None)
        bin_fail_counts = getattr(sampler, "bin_fail_counts", None)
        bin_sample_counts = getattr(sampler, "bin_sample_counts", None)
        bin_reset_eligible = getattr(sampler, "bin_reset_eligible", None)
        bin_reset_times = getattr(sampler, "bin_reset_times", None)
        if (
            not clips
            or bin_fail_counts is None
            or bin_sample_counts is None
            or bin_reset_eligible is None
            or bin_reset_times is None
        ):
            return []

        motion_fail_counts = torch.stack(
            [cls._as_cpu_tensor(fail_counts, dtype=torch.float32).sum() for fail_counts in bin_fail_counts],
            dim=0,
        )
        motion_sample_counts = torch.stack(
            [cls._as_cpu_tensor(sample_counts, dtype=torch.float32).sum() for sample_counts in bin_sample_counts],
            dim=0,
        )
        motion_builder = getattr(sampler, "_build_motion_sampling_probabilities", None)
        if callable(motion_builder):
            motion_eligible_mask = getattr(sampler, "motion_reset_eligible", None)
            motion_probs = cls._as_cpu_tensor(
                motion_builder(eligible_mask=motion_eligible_mask),
                dtype=torch.float32,
            ).reshape(-1)
        else:
            motion_probs = cls._build_sampler_sampling_probabilities(
                sampler,
                motion_fail_counts,
                motion_sample_counts,
                temperature=temperature,
            )

        results: list[dict[str, float | int | str]] = []
        for motion_index, clip in enumerate(clips):
            anchor_times_raw = getattr(clip, "anchor_times", None)
            if anchor_times_raw is None:
                continue

            anchor_times = cls._as_cpu_tensor(anchor_times_raw, dtype=torch.float32).reshape(-1)
            anchor_probs = torch.zeros(anchor_times.shape, dtype=torch.float32, device=anchor_times.device)

            if anchor_times.numel() > 0:
                fail_counts = cls._as_cpu_tensor(bin_fail_counts[motion_index], dtype=torch.float32).reshape(-1)
                sample_counts = cls._as_cpu_tensor(bin_sample_counts[motion_index], dtype=torch.float32).reshape(-1)
                eligible_mask = cls._as_cpu_tensor(bin_reset_eligible[motion_index], dtype=torch.bool).reshape(-1)
                reset_times = cls._as_cpu_tensor(bin_reset_times[motion_index], dtype=torch.float32).reshape(-1)
                bin_probs = cls._build_sampler_sampling_probabilities(
                    sampler,
                    fail_counts,
                    sample_counts,
                    temperature=temperature,
                    eligible_mask=eligible_mask,
                )
                for bin_index in torch.nonzero(bin_probs > 0.0, as_tuple=False).squeeze(-1).tolist():
                    anchor_index = cls._resolve_anchor_index(anchor_times, reset_times[bin_index])
                    anchor_probs[anchor_index] += motion_probs[motion_index] * bin_probs[bin_index]

            motion_name = str(getattr(clip, "name", f"motion_{motion_index}"))
            for anchor_index, anchor_time in enumerate(anchor_times.tolist()):
                results.append(
                    {
                        "motion_index": motion_index,
                        "motion_name": motion_name,
                        "anchor_index": anchor_index,
                        "anchor_time": float(anchor_time),
                        "probability": float(anchor_probs[anchor_index].item()),
                    }
                )

        return results

    def _collect_anchor_reset_probabilities(self) -> list[dict[str, float | int | str]]:
        env_unwrapped = getattr(getattr(self, "env", None), "unwrapped", None)
        sampler = getattr(env_unwrapped, "sampler", None)
        if sampler is None:
            return []
        if self._normalize_choice_name(getattr(sampler, "segment_source", self.segment_source)) != "anchor":
            return []
        return self._compute_anchor_reset_probabilities(
            sampler,
            temperature=1.0,
        )

    @classmethod
    def _build_anchor_reset_probability_metrics(
        cls,
        anchor_probabilities: list[dict[str, float | int | str]],
    ) -> dict[str, float]:
        arrays = cls._build_anchor_probability_arrays(anchor_probabilities)
        probabilities = np.maximum(arrays.probability.astype(np.float64, copy=False), 0.0)
        motion_probabilities = cls._aggregate_probabilities_by_motion(arrays)
        active_anchors = float(np.count_nonzero(probabilities > 0.0))
        num_anchors = float(probabilities.size)
        active_motions = float(np.count_nonzero(motion_probabilities > 0.0))
        num_motions = float(motion_probabilities.size)
        effective_anchors = cls._effective_probability_count(probabilities)
        effective_motions = cls._effective_probability_count(motion_probabilities)
        anchor_counts_by_motion = cls._anchor_counts_by_motion(arrays)
        top_anchor_indices = np.argsort(-probabilities)[: min(ANCHOR_CONSOLE_TOP_K, probabilities.size)]
        top_anchor_counts = np.asarray(
            [anchor_counts_by_motion.get(int(arrays.motion_index[index]), 0) for index in top_anchor_indices],
            dtype=np.int64,
        )
        top_anchor_ids = arrays.anchor_index[top_anchor_indices] if top_anchor_indices.size else np.asarray([])
        single_anchor_motion_count = float(sum(1 for count in anchor_counts_by_motion.values() if count == 1))

        return {
            "sampler/reset_distribution/anchors/probability_sum": float(np.sum(probabilities)),
            "sampler/reset_distribution/anchors/max_probability": (
                float(np.max(probabilities)) if probabilities.size else 0.0
            ),
            "sampler/reset_distribution/anchors/entropy": cls._probability_entropy(probabilities),
            "sampler/reset_distribution/anchors/effective_count": effective_anchors,
            "sampler/reset_distribution/anchors/effective_fraction": cls._safe_fraction(
                effective_anchors,
                num_anchors,
            ),
            "sampler/reset_distribution/anchors/active_count": active_anchors,
            "sampler/reset_distribution/anchors/total_count": num_anchors,
            "sampler/reset_distribution/anchors/active_fraction": cls._safe_fraction(active_anchors, num_anchors),
            "sampler/reset_distribution/anchors/top1_mass": cls._top_probability_mass(probabilities, 1),
            "sampler/reset_distribution/anchors/top5_mass": cls._top_probability_mass(probabilities, 5),
            "sampler/reset_distribution/anchors/top20_mass": cls._top_probability_mass(probabilities, 20),
            "sampler/reset_distribution/anchors/top20_anchor0_count": (
                float(np.count_nonzero(top_anchor_ids == 0)) if top_anchor_ids.size else 0.0
            ),
            "sampler/reset_distribution/anchors/top20_single_anchor_motion_count": (
                float(np.count_nonzero(top_anchor_counts == 1)) if top_anchor_counts.size else 0.0
            ),
            "sampler/reset_distribution/motions/max_probability": (
                float(np.max(motion_probabilities)) if motion_probabilities.size else 0.0
            ),
            "sampler/reset_distribution/motions/entropy": cls._probability_entropy(motion_probabilities),
            "sampler/reset_distribution/motions/effective_count": effective_motions,
            "sampler/reset_distribution/motions/effective_fraction": cls._safe_fraction(
                effective_motions,
                num_motions,
            ),
            "sampler/reset_distribution/motions/active_count": active_motions,
            "sampler/reset_distribution/motions/total_count": num_motions,
            "sampler/reset_distribution/motions/active_fraction": cls._safe_fraction(active_motions, num_motions),
            "sampler/reset_distribution/motions/top1_mass": cls._top_probability_mass(motion_probabilities, 1),
            "sampler/reset_distribution/motions/top5_mass": cls._top_probability_mass(motion_probabilities, 5),
            "sampler/reset_distribution/motions/top10_mass": cls._top_probability_mass(motion_probabilities, 10),
            "sampler/reset_distribution/motions/single_anchor_count": single_anchor_motion_count,
            "sampler/reset_distribution/motions/single_anchor_fraction": cls._safe_fraction(
                single_anchor_motion_count,
                num_motions,
            ),
        }

    @staticmethod
    def _safe_fraction(numerator: float, denominator: float) -> float:
        if denominator <= 0.0:
            return 0.0
        return float(numerator) / float(denominator)

    def _build_sampler_config_metrics(self) -> dict[str, float]:
        env_unwrapped = getattr(getattr(self, "env", None), "unwrapped", None)
        sampler = getattr(env_unwrapped, "sampler", None)
        cfg = getattr(self, "cfg", None)
        weight_fail = float(getattr(cfg, "weight_fail", getattr(sampler, "weight_fail", 0.0)))
        weight_novel = float(getattr(cfg, "weight_novel", getattr(sampler, "weight_novel", 0.0)))
        metrics = {
            "sampler/config/uses_failure_weighted": float(
                getattr(self, "sampling_strategy", None) == "failure_weighted"
            ),
            "sampler/config/uses_anchor_source": float(getattr(self, "segment_source", None) == "anchor"),
            "sampler/config/motion_count": float(len(getattr(self, "motion_files", []))),
            "sampler/config/anchor_bin_count": float(getattr(sampler, "num_bins", 0) or 0),
            "sampler/config/weight_fail": weight_fail,
            "sampler/config/weight_novel": weight_novel,
            "sampler/config/weight_uniform": max(0.0, 1.0 - weight_fail - weight_novel),
            "sampler/config/cap_beta": float(getattr(cfg, "cap_beta", getattr(sampler, "cap_beta", 0.0))),
            "sampler/config/adaptive_uniform_ratio": float(
                getattr(cfg, "adaptive_uniform_ratio", getattr(sampler, "adaptive_uniform_ratio", 0.0))
            ),
            "sampler/config/adaptive_alpha": float(
                getattr(cfg, "adaptive_alpha", getattr(sampler, "adaptive_alpha", 0.0))
            ),
            "sampler/config/adaptive_kernel_size": float(
                getattr(cfg, "adaptive_kernel_size", getattr(sampler, "adaptive_kernel_size", 1))
            ),
            "sampler/config/adaptive_lambda": float(
                getattr(cfg, "adaptive_lambda", getattr(sampler, "adaptive_lambda", 0.0))
            ),
            "sampler/config/motion_sampling_warmup_s": float(
                getattr(cfg, "motion_sampling_warmup_s", getattr(sampler, "motion_sampling_warmup_s", 0.0))
            ),
            "sampler/config/motion_sampling_ramp_s": float(
                getattr(cfg, "motion_sampling_ramp_s", getattr(sampler, "motion_sampling_ramp_s", 0.0))
            ),
            "sampler/config/failure_warmup_steps": float(
                getattr(self, "sampler_failure_warmup_steps", 0)
            ),
        }
        return metrics

    @classmethod
    def _build_sampler_failure_stats(cls, sampler: Any) -> dict[str, float]:
        bin_fail_counts = getattr(sampler, "bin_fail_counts", None)
        bin_sample_counts = getattr(sampler, "bin_sample_counts", None)
        if bin_fail_counts is None or bin_sample_counts is None:
            return {}

        bin_reset_eligible = getattr(sampler, "bin_reset_eligible", None)
        anchor_failures_by_motion: list[torch.Tensor] = []
        anchor_samples_by_motion: list[torch.Tensor] = []
        for motion_index, (fail_counts, sample_counts) in enumerate(
            zip(bin_fail_counts, bin_sample_counts, strict=False)
        ):
            fail_tensor = cls._as_cpu_tensor(fail_counts, dtype=torch.float32).reshape(-1)
            sample_tensor = cls._as_cpu_tensor(sample_counts, dtype=torch.float32).reshape(-1)
            if fail_tensor.shape != sample_tensor.shape:
                return {}
            if bin_reset_eligible is None:
                eligible = torch.ones_like(sample_tensor, dtype=torch.bool)
            else:
                eligible = cls._as_cpu_tensor(bin_reset_eligible[motion_index], dtype=torch.bool).reshape(-1)
                if eligible.shape != sample_tensor.shape:
                    return {}
            anchor_failures_by_motion.append(fail_tensor[eligible])
            anchor_samples_by_motion.append(sample_tensor[eligible])

        if not anchor_samples_by_motion:
            return {}

        anchor_samples = torch.cat(anchor_samples_by_motion) if anchor_samples_by_motion else torch.empty(0)
        anchor_failures = torch.cat(anchor_failures_by_motion) if anchor_failures_by_motion else torch.empty(0)
        motion_samples = torch.as_tensor(
            [float(samples.sum().item()) for samples in anchor_samples_by_motion],
            dtype=torch.float32,
        )
        motion_failures = torch.as_tensor(
            [float(failures.sum().item()) for failures in anchor_failures_by_motion],
            dtype=torch.float32,
        )
        return cls._build_sampler_failure_stats_from_counts(
            anchor_samples=anchor_samples,
            anchor_failures=anchor_failures,
            motion_samples=motion_samples,
            motion_failures=motion_failures,
        )

    @classmethod
    def _build_sampler_failure_stats_from_counts(
        cls,
        *,
        anchor_samples: torch.Tensor,
        anchor_failures: torch.Tensor,
        motion_samples: torch.Tensor,
        motion_failures: torch.Tensor,
    ) -> dict[str, float]:
        anchor_samples = cls._as_cpu_tensor(anchor_samples, dtype=torch.float32).reshape(-1)
        anchor_failures = cls._as_cpu_tensor(anchor_failures, dtype=torch.float32).reshape(-1)
        motion_samples = cls._as_cpu_tensor(motion_samples, dtype=torch.float32).reshape(-1)
        motion_failures = cls._as_cpu_tensor(motion_failures, dtype=torch.float32).reshape(-1)

        if anchor_samples.shape != anchor_failures.shape or motion_samples.shape != motion_failures.shape:
            raise ValueError("sample and failure count tensors must have matching shapes.")

        total_samples = float(torch.sum(anchor_samples).item()) if anchor_samples.numel() else 0.0
        total_failures = float(torch.sum(anchor_failures).item()) if anchor_failures.numel() else 0.0
        anchor_sampled = anchor_samples > 0.0
        anchor_failed = anchor_failures > 0.0
        motion_sampled = motion_samples > 0.0
        motion_failed = motion_failures > 0.0

        anchor_failure_rates = torch.zeros_like(anchor_samples)
        if bool(torch.any(anchor_sampled).item()):
            anchor_failure_rates[anchor_sampled] = anchor_failures[anchor_sampled] / torch.clamp(
                anchor_samples[anchor_sampled],
                min=torch.finfo(anchor_samples.dtype).eps,
            )
        motion_failure_rates = torch.zeros_like(motion_samples)
        if bool(torch.any(motion_sampled).item()):
            motion_failure_rates[motion_sampled] = motion_failures[motion_sampled] / torch.clamp(
                motion_samples[motion_sampled],
                min=torch.finfo(motion_samples.dtype).eps,
            )

        sampled_anchor_count = float(torch.sum(anchor_sampled.to(dtype=torch.float32)).item())
        sampled_motion_count = float(torch.sum(motion_sampled.to(dtype=torch.float32)).item())
        total_anchor_count = float(anchor_samples.numel())
        total_motion_count = float(motion_samples.numel())
        return {
            "sampler/failure_stats/effective_sample_count_sum": total_samples,
            "sampler/failure_stats/effective_failure_count_sum": total_failures,
            "sampler/failure_stats/failure_rate": cls._safe_fraction(total_failures, total_samples),
            "sampler/failure_stats/anchors/total_count": total_anchor_count,
            "sampler/failure_stats/anchors/sampled_count": sampled_anchor_count,
            "sampler/failure_stats/anchors/sampled_fraction": cls._safe_fraction(
                sampled_anchor_count,
                total_anchor_count,
            ),
            "sampler/failure_stats/anchors/failed_count": float(torch.sum(anchor_failed.to(dtype=torch.float32)).item()),
            "sampler/failure_stats/anchors/max_sample_count": (
                float(torch.max(anchor_samples).item()) if anchor_samples.numel() else 0.0
            ),
            "sampler/failure_stats/anchors/max_failure_rate": (
                float(torch.max(anchor_failure_rates).item()) if anchor_failure_rates.numel() else 0.0
            ),
            "sampler/failure_stats/anchors/mean_failure_rate_sampled": (
                float(torch.mean(anchor_failure_rates[anchor_sampled]).item())
                if bool(torch.any(anchor_sampled).item())
                else 0.0
            ),
            "sampler/failure_stats/motions/total_count": total_motion_count,
            "sampler/failure_stats/motions/sampled_count": sampled_motion_count,
            "sampler/failure_stats/motions/sampled_fraction": cls._safe_fraction(
                sampled_motion_count,
                total_motion_count,
            ),
            "sampler/failure_stats/motions/failed_count": float(torch.sum(motion_failed.to(dtype=torch.float32)).item()),
            "sampler/failure_stats/motions/max_sample_count": (
                float(torch.max(motion_samples).item()) if motion_samples.numel() else 0.0
            ),
            "sampler/failure_stats/motions/max_failure_rate": (
                float(torch.max(motion_failure_rates).item()) if motion_failure_rates.numel() else 0.0
            ),
        }

    @staticmethod
    def _probability_entropy(probabilities: np.ndarray) -> float:
        probabilities = np.maximum(np.asarray(probabilities, dtype=np.float64).reshape(-1), 0.0)
        total = float(np.sum(probabilities))
        if total <= 0.0:
            return 0.0

        normalized = probabilities[probabilities > 0.0] / total
        return float(-np.sum(normalized * np.log(normalized)))

    @classmethod
    def _effective_probability_count(cls, probabilities: np.ndarray) -> float:
        probabilities = np.maximum(np.asarray(probabilities, dtype=np.float64).reshape(-1), 0.0)
        if float(np.sum(probabilities)) <= 0.0:
            return 0.0
        return float(np.exp(cls._probability_entropy(probabilities)))

    @staticmethod
    def _top_probability_mass(probabilities: np.ndarray, k: int) -> float:
        probabilities = np.maximum(np.asarray(probabilities, dtype=np.float64).reshape(-1), 0.0)
        if probabilities.size == 0 or k <= 0:
            return 0.0
        return float(np.sum(np.sort(probabilities)[-min(k, probabilities.size) :]))

    @staticmethod
    def _build_anchor_probability_arrays(
        anchor_probabilities: list[dict[str, float | int | str]],
    ) -> AnchorProbabilityArrays:
        motion_name_to_index: dict[str, int] = {}
        motion_indices: list[int] = []
        motion_names_list: list[str] = []
        anchor_indices: list[int] = []
        anchor_times: list[float] = []
        probabilities: list[float] = []

        for entry in anchor_probabilities:
            motion_name = str(entry["motion_name"])
            if "motion_index" in entry:
                motion_index = int(entry["motion_index"])
            else:
                motion_index = motion_name_to_index.setdefault(motion_name, len(motion_name_to_index))

            motion_indices.append(motion_index)
            motion_names_list.append(motion_name)
            anchor_indices.append(int(entry["anchor_index"]))
            anchor_times.append(float(entry["anchor_time"]))
            probabilities.append(float(entry["probability"]))

        return AnchorProbabilityArrays(
            motion_index=np.asarray(motion_indices, dtype=np.int64),
            motion_name=np.asarray(motion_names_list, dtype=np.str_),
            anchor_index=np.asarray(anchor_indices, dtype=np.int64),
            anchor_time=np.asarray(anchor_times, dtype=np.float32),
            probability=np.asarray(probabilities, dtype=np.float32),
        )

    @staticmethod
    def _aggregate_probabilities_by_motion(arrays: AnchorProbabilityArrays) -> np.ndarray:
        if arrays.motion_index.size == 0:
            return np.asarray([], dtype=np.float64)

        motion_probabilities = []
        for motion_index in np.unique(arrays.motion_index):
            mask = arrays.motion_index == motion_index
            motion_probabilities.append(float(np.sum(np.maximum(arrays.probability[mask], 0.0))))
        return np.asarray(motion_probabilities, dtype=np.float64)

    @staticmethod
    def _anchor_counts_by_motion(arrays: AnchorProbabilityArrays) -> dict[int, int]:
        return {
            int(motion_index): int(np.count_nonzero(arrays.motion_index == motion_index))
            for motion_index in np.unique(arrays.motion_index)
        }

    @staticmethod
    def _truncate_dashboard_label(value: str, *, max_chars: int = ANCHOR_DASHBOARD_LABEL_CHARS) -> str:
        if len(value) <= max_chars:
            return value
        if max_chars <= 3:
            return value[:max_chars]
        return value[: max_chars - 3] + "..."

    @classmethod
    def _build_anchor_motion_time_grid(
        cls,
        arrays: AnchorProbabilityArrays,
        *,
        num_bins: int,
    ) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
        if num_bins < 1:
            raise ValueError("num_bins must be positive.")
        if arrays.motion_index.size == 0:
            return (
                np.zeros((0, num_bins), dtype=np.float32),
                np.asarray([], dtype=np.int64),
                [],
                np.asarray([], dtype=np.float32),
            )

        rows: list[np.ndarray] = []
        motion_indices: list[int] = []
        motion_names_list: list[str] = []
        motion_probabilities: list[float] = []

        for motion_index in np.unique(arrays.motion_index):
            mask = arrays.motion_index == motion_index
            times = arrays.anchor_time[mask].astype(np.float64, copy=False)
            probabilities = np.maximum(arrays.probability[mask].astype(np.float64, copy=False), 0.0)
            row = np.zeros(num_bins, dtype=np.float64)

            if times.size > 0:
                min_time = float(np.min(times))
                max_time = float(np.max(times))
                if max_time > min_time:
                    normalized_times = (times - min_time) / (max_time - min_time)
                else:
                    normalized_times = np.zeros_like(times)
                bin_indices = np.floor(np.clip(normalized_times, 0.0, 1.0) * num_bins).astype(np.int64)
                bin_indices = np.clip(bin_indices, 0, num_bins - 1)
                np.add.at(row, bin_indices, probabilities)

            first_entry_index = int(np.nonzero(mask)[0][0])
            rows.append(row.astype(np.float32))
            motion_indices.append(int(motion_index))
            motion_names_list.append(str(arrays.motion_name[first_entry_index]))
            motion_probabilities.append(float(np.sum(probabilities)))

        order = sorted(range(len(rows)), key=lambda index: (-motion_probabilities[index], motion_indices[index]))
        return (
            np.stack([rows[index] for index in order], axis=0),
            np.asarray([motion_indices[index] for index in order], dtype=np.int64),
            [motion_names_list[index] for index in order],
            np.asarray([motion_probabilities[index] for index in order], dtype=np.float32),
        )

    @classmethod
    def _build_anchor_rank_band_heatmap_grid(
        cls,
        anchor_probabilities: list[dict[str, float | int | str]],
        *,
        num_bins: int,
        max_rank_bands: int = ANCHOR_DASHBOARD_MAX_RANK_BANDS,
    ) -> AnchorRankBandGrid:
        arrays = cls._build_anchor_probability_arrays(anchor_probabilities)
        motion_time_grid, _, _, _ = cls._build_anchor_motion_time_grid(arrays, num_bins=num_bins)
        num_motions = int(motion_time_grid.shape[0])
        if num_motions == 0:
            return AnchorRankBandGrid(
                values=np.zeros((0, num_bins), dtype=np.float32),
                num_bins=num_bins,
                num_motions=0,
                num_rank_bands=0,
            )

        rank_band_count = min(max(1, int(max_rank_bands)), num_motions)
        rank_bands = np.array_split(np.arange(num_motions), rank_band_count)
        values = np.stack(
            [np.sum(motion_time_grid[rank_band], axis=0) for rank_band in rank_bands],
            axis=0,
        ).astype(np.float32)
        return AnchorRankBandGrid(
            values=values,
            num_bins=num_bins,
            num_motions=num_motions,
            num_rank_bands=rank_band_count,
        )

    @classmethod
    def _build_top_motion_rows(
        cls,
        anchor_probabilities: list[dict[str, float | int | str]],
        *,
        limit: int = ANCHOR_DASHBOARD_TOP_K,
        max_name_chars: int = ANCHOR_DASHBOARD_LABEL_CHARS,
    ) -> list[dict[str, float | int | str]]:
        arrays = cls._build_anchor_probability_arrays(anchor_probabilities)
        _, motion_indices, motion_names_list, motion_probabilities = cls._build_anchor_motion_time_grid(
            arrays,
            num_bins=1,
        )
        rows: list[dict[str, float | int | str]] = []
        for rank, (motion_index, motion_name, probability) in enumerate(
            zip(motion_indices[:limit], motion_names_list[:limit], motion_probabilities[:limit], strict=False),
            start=1,
        ):
            motion_mask = arrays.motion_index == int(motion_index)
            motion_entry_indices = np.nonzero(motion_mask)[0]
            anchor_count = int(motion_entry_indices.size)
            motion_anchor_probabilities = np.maximum(
                arrays.probability[motion_entry_indices].astype(np.float64, copy=False),
                0.0,
            )
            active_anchor_count = int(np.count_nonzero(motion_anchor_probabilities > 0.0))
            max_anchor_entry_index = int(motion_entry_indices[0])
            max_anchor_probability = 0.0
            if motion_entry_indices.size > 0:
                max_anchor_order = np.lexsort(
                    (
                        arrays.anchor_index[motion_entry_indices],
                        -motion_anchor_probabilities,
                    )
                )
                max_anchor_entry_index = int(motion_entry_indices[int(max_anchor_order[0])])
                max_anchor_probability = float(motion_anchor_probabilities[int(max_anchor_order[0])])
            rows.append(
                {
                    "rank": rank,
                    "motion_index": int(motion_index),
                    "motion_name": cls._truncate_dashboard_label(str(motion_name), max_chars=max_name_chars),
                    "full_motion_name": str(motion_name),
                    "anchor_count": anchor_count,
                    "active_anchor_count": active_anchor_count,
                    "max_anchor_index": int(arrays.anchor_index[max_anchor_entry_index]),
                    "max_anchor_time": float(arrays.anchor_time[max_anchor_entry_index]),
                    "max_anchor_probability": max_anchor_probability,
                    "probability": float(probability),
                }
            )
        return rows

    @classmethod
    def _build_top_anchor_rows(
        cls,
        anchor_probabilities: list[dict[str, float | int | str]],
        *,
        limit: int = ANCHOR_DASHBOARD_TOP_K,
        max_name_chars: int = ANCHOR_DASHBOARD_LABEL_CHARS,
    ) -> list[dict[str, float | int | str]]:
        arrays = cls._build_anchor_probability_arrays(anchor_probabilities)
        probabilities = np.maximum(arrays.probability.astype(np.float64, copy=False), 0.0)
        if probabilities.size == 0:
            return []

        total_probability = float(np.sum(probabilities))
        uniform_probability = total_probability / float(probabilities.size) if total_probability > 0.0 else 0.0
        motion_probabilities_by_index: dict[int, float] = {}
        for motion_index in np.unique(arrays.motion_index):
            motion_mask = arrays.motion_index == int(motion_index)
            motion_probabilities_by_index[int(motion_index)] = float(np.sum(probabilities[motion_mask]))
        order = np.lexsort((arrays.anchor_index, arrays.motion_index, -probabilities))
        rows: list[dict[str, float | int | str]] = []
        for rank, entry_index in enumerate(order[:limit], start=1):
            probability = float(probabilities[entry_index])
            uniform_ratio = probability / uniform_probability if uniform_probability > 0.0 else 0.0
            motion_probability = motion_probabilities_by_index.get(int(arrays.motion_index[entry_index]), 0.0)
            anchor_share = probability / motion_probability if motion_probability > 0.0 else 0.0
            motion_name = str(arrays.motion_name[entry_index])
            rows.append(
                {
                    "rank": rank,
                    "motion_index": int(arrays.motion_index[entry_index]),
                    "motion_name": cls._truncate_dashboard_label(motion_name, max_chars=max_name_chars),
                    "full_motion_name": motion_name,
                    "anchor_index": int(arrays.anchor_index[entry_index]),
                    "anchor_time": float(arrays.anchor_time[entry_index]),
                    "probability": probability,
                    "motion_probability": float(motion_probability),
                    "anchor_share": float(anchor_share),
                    "uniform_ratio": float(uniform_ratio),
                }
            )
        return rows

    @classmethod
    def _build_anchor_cumulative_mass_curve(
        cls,
        anchor_probabilities: list[dict[str, float | int | str]],
        *,
        checkpoints: tuple[int, ...] = (1, 5, 20, 100, 1000),
    ) -> dict[str, np.ndarray]:
        arrays = cls._build_anchor_probability_arrays(anchor_probabilities)
        probabilities = np.maximum(arrays.probability.astype(np.float64, copy=False), 0.0)
        if probabilities.size == 0:
            return {
                "anchor_count": np.asarray([], dtype=np.int64),
                "mass": np.asarray([], dtype=np.float64),
                "uniform_mass": np.asarray([], dtype=np.float64),
            }

        total_probability = float(np.sum(probabilities))
        counts = sorted({min(int(checkpoint), probabilities.size) for checkpoint in checkpoints if checkpoint > 0})
        if probabilities.size not in counts:
            counts.append(int(probabilities.size))

        sorted_probabilities = np.sort(probabilities)[::-1]
        cumulative = np.cumsum(sorted_probabilities)
        anchor_counts = np.asarray(counts, dtype=np.int64)
        mass = np.asarray([float(cumulative[count - 1]) for count in counts], dtype=np.float64)
        uniform_mass = total_probability * anchor_counts.astype(np.float64) / float(probabilities.size)
        return {
            "anchor_count": anchor_counts,
            "mass": mass,
            "uniform_mass": uniform_mass,
        }

    @staticmethod
    def _build_anchor_probability_metadata(
        arrays: AnchorProbabilityArrays,
        *,
        heatmap_bins: int,
    ) -> dict[str, object]:
        motions = []
        for motion_index in np.unique(arrays.motion_index):
            mask = arrays.motion_index == motion_index
            first_entry_index = int(np.nonzero(mask)[0][0])
            motions.append(
                {
                    "motion_index": int(motion_index),
                    "motion_name": str(arrays.motion_name[first_entry_index]),
                    "anchor_count": int(np.count_nonzero(mask)),
                    "probability": float(np.sum(np.maximum(arrays.probability[mask], 0.0))),
                }
            )

        return {
            "heatmap_bins": int(heatmap_bins),
            "num_motions": len(motions),
            "num_anchors": int(arrays.probability.size),
            "motions": sorted(motions, key=lambda item: int(item["motion_index"])),
        }

    @staticmethod
    def _write_anchor_probability_heatmap(
        grid: AnchorRankBandGrid,
        top_motion_rows: list[dict[str, float | int | str]],
        top_anchor_rows: list[dict[str, float | int | str]],
        cumulative_curve: dict[str, np.ndarray],
        metrics_payload: dict[str, float],
        output_path: Path,
        *,
        update_count: int,
        global_step: int,
    ) -> Path:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from matplotlib.colors import LogNorm

        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure = plt.figure(figsize=(18.0, 12.0))
        grid_spec = figure.add_gridspec(
            2,
            2,
            width_ratios=(2.15, 1.25),
            height_ratios=(1.2, 1.0),
        )
        axis_heatmap = figure.add_subplot(grid_spec[0, 0])
        axis_curve = figure.add_subplot(grid_spec[0, 1])
        axis_motion = figure.add_subplot(grid_spec[1, 0])
        axis_anchor = figure.add_subplot(grid_spec[1, 1])
        figure.subplots_adjust(left=0.08, right=0.97, bottom=0.07, top=0.84, wspace=0.28, hspace=0.38)

        values = np.asarray(grid.values, dtype=np.float32)
        if values.size == 0:
            values = np.zeros((1, grid.num_bins), dtype=np.float32)
        masked_values = np.ma.masked_less_equal(values, 0.0)
        positive_values = values[values > 0.0]

        color_map = plt.get_cmap("viridis").copy()
        color_map.set_bad("#f0f0f0")
        image_extent = [0.0, 1.0, 100.0, 0.0]
        if positive_values.size:
            max_probability = float(np.max(positive_values))
            min_probability = float(np.min(positive_values))
            if min_probability >= max_probability:
                min_probability = max_probability * 0.1
            image = axis_heatmap.imshow(
                masked_values,
                aspect="auto",
                interpolation="nearest",
                cmap=color_map,
                norm=LogNorm(vmin=max(min_probability, 1.0e-12), vmax=max_probability),
                extent=image_extent,
            )
        else:
            image = axis_heatmap.imshow(
                values,
                aspect="auto",
                interpolation="nearest",
                cmap=color_map,
                vmin=0.0,
                vmax=1.0,
                extent=image_extent,
            )

        axis_heatmap.set_title(f"Reset mass by rank band ({grid.num_rank_bands} bands, {grid.num_motions} motions)")
        axis_heatmap.set_xlabel("normalized motion time")
        axis_heatmap.set_ylabel("motion reset-probability rank percentile")
        axis_heatmap.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0], ["0.00", "0.25", "0.50", "0.75", "1.00"])
        axis_heatmap.set_yticks([0, 25, 50, 75, 100], ["top", "25", "50", "75", "bottom"])
        figure.colorbar(image, ax=axis_heatmap, label="reset probability mass", fraction=0.046, pad=0.025)

        anchor_counts = cumulative_curve.get("anchor_count", np.asarray([], dtype=np.int64))
        cumulative_mass = cumulative_curve.get("mass", np.asarray([], dtype=np.float64))
        uniform_mass = cumulative_curve.get("uniform_mass", np.asarray([], dtype=np.float64))
        if anchor_counts.size > 0:
            axis_curve.plot(anchor_counts, cumulative_mass, marker="o", linewidth=2.0, label="sampler")
            axis_curve.plot(anchor_counts, uniform_mass, marker="o", linestyle="--", linewidth=1.5, label="uniform")
            if int(np.max(anchor_counts)) > 1:
                axis_curve.set_xscale("log")
            axis_curve.set_xticks(anchor_counts)
            axis_curve.set_xticklabels([str(int(count)) for count in anchor_counts], rotation=30, ha="right")
            y_max = max(float(np.max(cumulative_mass)), float(np.max(uniform_mass)), 1.0e-8)
            axis_curve.set_ylim(0.0, min(1.0, y_max * 1.15))
            axis_curve.grid(True, alpha=0.25)
            axis_curve.legend(loc="upper left", fontsize=8)
        else:
            axis_curve.text(0.5, 0.5, "no anchor probabilities", ha="center", va="center")
            axis_curve.set_xticks([])
            axis_curve.set_yticks([])
        axis_curve.set_title("Top-k anchor mass")
        axis_curve.set_xlabel("top k anchors")
        axis_curve.set_ylabel("cumulative mass")

        if top_motion_rows:
            motion_rows = list(reversed(top_motion_rows))
            y_positions = np.arange(len(motion_rows))
            axis_motion.barh(
                y_positions,
                [float(row["probability"]) for row in motion_rows],
                color="#4477aa",
            )
            axis_motion.set_yticks(
                y_positions,
                [f"{int(row['rank']):02d}. {row['motion_name']}" for row in motion_rows],
                fontsize=8,
            )
            axis_motion.grid(True, axis="x", alpha=0.25)
            axis_motion.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        else:
            axis_motion.text(0.5, 0.5, "no motion probabilities", ha="center", va="center")
            axis_motion.set_xticks([])
            axis_motion.set_yticks([])
        axis_motion.set_title(f"Top {min(ANCHOR_DASHBOARD_TOP_K, len(top_motion_rows))} motions by reset mass")
        axis_motion.set_xlabel("reset probability mass")

        axis_anchor.axis("off")
        axis_anchor.set_title(f"Top {min(ANCHOR_DASHBOARD_TOP_K, len(top_anchor_rows))} anchors", pad=8)
        cell_text = [
            [
                str(int(row["rank"])),
                str(row["motion_name"]),
                f"A{int(row['anchor_index'])}",
                f"{float(row['anchor_time']):.2f}",
                f"{float(row['probability']):.2e}",
                f"{float(row['uniform_ratio']):.1f}x",
            ]
            for row in top_anchor_rows
        ]
        column_labels = ["#", "motion", "anchor", "time", "p", "vs uniform"]

        if cell_text:
            table = axis_anchor.table(
                cellText=cell_text,
                colLabels=column_labels,
                cellLoc="left",
                colLoc="left",
                colWidths=[0.06, 0.42, 0.11, 0.12, 0.14, 0.15],
                bbox=[0.0, 0.0, 1.0, 0.94],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(6)
            for (row_index, _column_index), cell in table.get_celld().items():
                if row_index == 0:
                    cell.set_text_props(weight="bold")
                    cell.set_facecolor("#e8e8e8")
                else:
                    cell.set_facecolor("#ffffff" if row_index % 2 else "#f7f7f7")
        else:
            axis_anchor.text(0.5, 0.5, "no anchor probabilities", ha="center", va="center")

        active_anchors = int(metrics_payload.get("sampler/reset_distribution/anchors/active_count", 0.0))
        total_anchors = int(metrics_payload.get("sampler/reset_distribution/anchors/total_count", 0.0))
        active_motions = int(metrics_payload.get("sampler/reset_distribution/motions/active_count", 0.0))
        total_motions = int(metrics_payload.get("sampler/reset_distribution/motions/total_count", 0.0))
        effective_anchors = metrics_payload.get("sampler/reset_distribution/anchors/effective_count", 0.0)
        effective_anchor_fraction = metrics_payload.get("sampler/reset_distribution/anchors/effective_fraction", 0.0)
        effective_motions = metrics_payload.get("sampler/reset_distribution/motions/effective_count", 0.0)
        effective_motion_fraction = metrics_payload.get("sampler/reset_distribution/motions/effective_fraction", 0.0)
        max_probability = metrics_payload.get("sampler/reset_distribution/anchors/max_probability", 0.0)
        top20_mass = metrics_payload.get("sampler/reset_distribution/anchors/top20_mass", 0.0)
        figure.suptitle(
            "Sampler reset dashboard\n"
            f"update={update_count} step={global_step} | "
            f"anchors active={active_anchors}/{total_anchors} effective={effective_anchors:.0f} "
            f"({effective_anchor_fraction:.1%}) | "
            f"motions active={active_motions}/{total_motions} effective={effective_motions:.0f} "
            f"({effective_motion_fraction:.1%}) | "
            f"max_p={max_probability:.3g} top20={top20_mass:.3g}",
            fontsize=14,
        )
        figure.savefig(output_path, dpi=120)
        plt.close(figure)
        return output_path

    def _write_anchor_reset_probability_artifacts(
        self,
        anchor_probabilities: list[dict[str, float | int | str]],
        metrics_payload: dict[str, float],
    ) -> dict[str, str]:
        output_dir = self.run_paths.debug_dir / "anchor_reset_probabilities"
        output_dir.mkdir(parents=True, exist_ok=True)

        arrays = self._build_anchor_probability_arrays(anchor_probabilities)
        metadata_path = write_json(
            output_dir / "metadata.json",
            self._build_anchor_probability_metadata(arrays, heatmap_bins=self.anchor_heatmap_bins),
        )

        snapshot_stem = f"update_{self.update_count:06d}"
        npz_path = output_dir / f"{snapshot_stem}.npz"
        np.savez_compressed(
            npz_path,
            motion_index=arrays.motion_index,
            motion_name=arrays.motion_name,
            anchor_index=arrays.anchor_index,
            anchor_time=arrays.anchor_time,
            probability=arrays.probability,
        )

        summary_path = output_dir / "summary.jsonl"
        summary_row = {
            "update": int(self.update_count),
            "global_step": int(self.global_step),
            "heatmap_bins": int(self.anchor_heatmap_bins),
            "num_anchor_entries": int(arrays.probability.size),
            "metrics": metrics_payload,
        }
        with summary_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary_row, sort_keys=True) + "\n")

        artifacts = {
            "metadata": str(metadata_path),
            "summary_jsonl": str(summary_path),
            "snapshot_npz": str(npz_path),
        }

        grid = self._build_anchor_rank_band_heatmap_grid(
            anchor_probabilities,
            num_bins=self.anchor_heatmap_bins,
        )
        top_motion_rows = self._build_top_motion_rows(anchor_probabilities)
        top_anchor_rows = self._build_top_anchor_rows(anchor_probabilities)
        cumulative_curve = self._build_anchor_cumulative_mass_curve(anchor_probabilities)
        heatmap_path = output_dir / f"{snapshot_stem}_heatmap.png"
        latest_heatmap_path = output_dir / "latest_heatmap.png"
        try:
            self._write_anchor_probability_heatmap(
                grid,
                top_motion_rows,
                top_anchor_rows,
                cumulative_curve,
                metrics_payload,
                heatmap_path,
                update_count=self.update_count,
                global_step=self.global_step,
            )
            shutil.copyfile(heatmap_path, latest_heatmap_path)
            artifacts["heatmap_png"] = str(heatmap_path)
            artifacts["latest_heatmap_png"] = str(latest_heatmap_path)
        except Exception as exc:
            if not self._anchor_heatmap_warning_emitted:
                print(f"skipping anchor probability heatmap rendering: {exc}", flush=True)
                self._anchor_heatmap_warning_emitted = True

        return artifacts

    def _sync_anchor_reset_probability_summary_to_wandb(
        self,
        metrics_payload: dict[str, float],
        artifacts: dict[str, str],
    ) -> None:
        if not self.use_wandb:
            return

        try:
            import wandb
        except ImportError:
            return

        run = getattr(wandb, "run", None)
        if run is None:
            return

        for metric_name, value in metrics_payload.items():
            run.summary[metric_name] = float(value)
        run.summary["sampler/reset_distribution/latest_update"] = int(self.update_count)
        run.summary["sampler/reset_distribution/latest_global_step"] = int(self.global_step)

        latest_heatmap_path = artifacts.get("latest_heatmap_png")
        if latest_heatmap_path is None:
            return

        try:
            run.summary["sampler/reset_distribution/latest_heatmap"] = wandb.Image(
                latest_heatmap_path,
                caption=f"sampler reset distribution update={self.update_count} step={self.global_step}",
            )
        except Exception as exc:
            print(f"failed to update W&B anchor heatmap summary: {exc}", flush=True)

        try:
            run.save(latest_heatmap_path, base_path=str(self.run_paths.debug_dir), policy="live")
        except Exception as exc:
            print(f"failed to sync latest anchor heatmap to W&B files: {exc}", flush=True)

    def _log_anchor_reset_probabilities(self) -> None:
        anchor_probabilities = self._collect_anchor_reset_probabilities()
        if not anchor_probabilities:
            return

        env_unwrapped = getattr(getattr(self, "env", None), "unwrapped", None)
        sampler = getattr(env_unwrapped, "sampler", None)
        metrics_payload = self._build_sampler_config_metrics()
        metrics_payload.update(self._build_anchor_reset_probability_metrics(anchor_probabilities))
        if sampler is not None:
            metrics_payload.update(self._build_sampler_failure_stats(sampler))
        if metrics_payload:
            self._log_metrics(metrics_payload)
        artifacts = self._write_anchor_reset_probability_artifacts(anchor_probabilities, metrics_payload)
        self._sync_anchor_reset_probability_summary_to_wandb(metrics_payload, artifacts)

        anchor_total = int(metrics_payload.get("sampler/reset_distribution/anchors/total_count", 0.0))
        anchor_active = int(metrics_payload.get("sampler/reset_distribution/anchors/active_count", 0.0))
        top20_anchor0_count = int(
            metrics_payload.get("sampler/reset_distribution/anchors/top20_anchor0_count", 0.0)
        )
        top20_single_anchor_motion_count = int(
            metrics_payload.get("sampler/reset_distribution/anchors/top20_single_anchor_motion_count", 0.0)
        )
        motion_total = int(metrics_payload.get("sampler/reset_distribution/motions/total_count", 0.0))
        motion_active = int(metrics_payload.get("sampler/reset_distribution/motions/active_count", 0.0))
        single_anchor_motion_count = int(
            metrics_payload.get("sampler/reset_distribution/motions/single_anchor_count", 0.0)
        )
        print(
            f"sampler snapshot after update {self.update_count} step={self.global_step}: "
            f"strategy={self.sampling_strategy} source={self.segment_source} "
            f"weight_fail={metrics_payload['sampler/config/weight_fail']:.3g} "
            f"weight_novel={metrics_payload['sampler/config/weight_novel']:.3g} "
            f"weight_uniform={metrics_payload['sampler/config/weight_uniform']:.3g} "
            f"adaptive_alpha={metrics_payload['sampler/config/adaptive_alpha']:.3g}",
            flush=True,
        )
        print(
            "  reset anchors: "
            f"active={anchor_active}/{anchor_total} "
            f"effective={metrics_payload['sampler/reset_distribution/anchors/effective_count']:.2f} "
            f"max_anchor_p={metrics_payload['sampler/reset_distribution/anchors/max_probability']:.6f} "
            f"top20={metrics_payload['sampler/reset_distribution/anchors/top20_mass']:.6f} "
            f"top20_a0={top20_anchor0_count} "
            f"top20_single_anchor_motions={top20_single_anchor_motion_count}",
            flush=True,
        )
        print(
            "  reset motions: "
            f"active={motion_active}/{motion_total} "
            f"effective={metrics_payload['sampler/reset_distribution/motions/effective_count']:.2f} "
            f"max_motion_p={metrics_payload['sampler/reset_distribution/motions/max_probability']:.6f} "
            f"top10={metrics_payload['sampler/reset_distribution/motions/top10_mass']:.6f} "
            f"single_anchor_motions={single_anchor_motion_count}/{motion_total}",
            flush=True,
        )
        if "sampler/failure_stats/effective_sample_count_sum" in metrics_payload:
            sampled_anchor_count = int(metrics_payload.get("sampler/failure_stats/anchors/sampled_count", 0.0))
            failure_anchor_total = int(metrics_payload.get("sampler/failure_stats/anchors/total_count", 0.0))
            sampled_motion_count = int(metrics_payload.get("sampler/failure_stats/motions/sampled_count", 0.0))
            failure_motion_total = int(metrics_payload.get("sampler/failure_stats/motions/total_count", 0.0))
            print(
                "  failure stats: "
                f"effective_samples={metrics_payload['sampler/failure_stats/effective_sample_count_sum']:.1f} "
                f"effective_failures={metrics_payload['sampler/failure_stats/effective_failure_count_sum']:.1f} "
                f"fail_rate={metrics_payload['sampler/failure_stats/failure_rate']:.4f} "
                f"sampled_anchors={sampled_anchor_count}/{failure_anchor_total} "
                f"sampled_motions={sampled_motion_count}/{failure_motion_total}",
                flush=True,
            )
        print("  top reset motions:", flush=True)
        top_motion_entries = self._build_top_motion_rows(
            anchor_probabilities,
            limit=ANCHOR_CONSOLE_TOP_K,
            max_name_chars=10_000,
        )
        for entry in top_motion_entries:
            print(
                f"  {entry['full_motion_name']} "
                f"motion_p={float(entry['probability']):.6f} "
                f"active_anchors={int(entry['active_anchor_count'])}/{int(entry['anchor_count'])} "
                f"max_anchor=A{int(entry['max_anchor_index'])}@{float(entry['max_anchor_time']):.3f}s "
                f"max_anchor_p={float(entry['max_anchor_probability']):.6f}",
                flush=True,
            )
        print("  top reset anchors:", flush=True)
        top_anchor_entries = self._build_top_anchor_rows(
            anchor_probabilities,
            limit=ANCHOR_CONSOLE_TOP_K,
            max_name_chars=10_000,
        )
        for entry in top_anchor_entries:
            print(
                f"  {entry['full_motion_name']} A{int(entry['anchor_index'])} "
                f"t={float(entry['anchor_time']):.3f}s "
                f"anchor_p={float(entry['probability']):.6f} "
                f"motion_p={float(entry['motion_probability']):.6f} "
                f"anchor_share={float(entry['anchor_share']) * 100.0:.1f}%",
                flush=True,
            )
        if latest_heatmap_path := artifacts.get("latest_heatmap_png"):
            print(f"  latest heatmap: {latest_heatmap_path}", flush=True)

    @torch.no_grad()
    def _update_actor_statistics(
        self,
        actor_obs_batch: dict[str, torch.Tensor],
    ) -> None:
        self.actor(actor_obs_batch, update_normlizer=True)

    @torch.no_grad()
    def get_value(
        self,
        critic_obs_batch: torch.Tensor | Mapping[str, torch.Tensor],
        update_normlizer: bool = True,
    ) -> torch.Tensor:
        critic_step: ValueStep = self.critic(critic_obs_batch, update_normlizer=update_normlizer)
        return critic_step.value

    @torch.no_grad()
    def get_action(
        self,
        actor_obs_batch: dict[str, torch.Tensor],
        critic_obs_batch: torch.Tensor | Mapping[str, torch.Tensor],
        determine: bool = False,
    ):
        actor_step = self.actor(actor_obs_batch, update_normlizer=True)
        action = actor_step.mean if determine else actor_step.action
        log_prob = actor_step.log_prob
        value = self.get_value(critic_obs_batch, update_normlizer=True)
        return action, log_prob, value

    def rollout(self, obs):
        latest_curriculum_metrics: dict[str, float] = {}
        terminate_rate_samples: list[float] = []
        end_effector_error_samples: list[float] = []
        relative_anchor_pos_samples: list[torch.Tensor] = []
        location_tracking_available = True
        for _ in range(self.steps):
            self.global_step += 1
            actor_obs = get_actor_observation(obs, self.actor_type)
            critic_obs = get_critic_observation(self.critic, obs)
            action, log_prob, value = self.get_action(actor_obs, critic_obs)
            next_obs, task_reward, terminate, timeout, info = self.env.step(action)
            terminate_rate = self._coerce_metric_scalar(terminate.to(dtype=torch.float32))
            if terminate_rate is not None:
                terminate_rate_samples.append(terminate_rate)
            end_effector_error_mean = self._extract_end_effector_termination_error_mean()
            if end_effector_error_mean is not None:
                end_effector_error_samples.append(end_effector_error_mean)
            curriculum_metric_sample = self._extract_curriculum_metric_sample(info)
            if curriculum_metric_sample:
                latest_curriculum_metrics.update(curriculum_metric_sample)
            next_obs = structure_env_observation(
                next_obs,
                action_dim=self.cfg.action_space,
                observation_window_lengths=self.observation_window_lengths,
            )
            if location_tracking_available:
                relative_anchor_pos = self._extract_relative_anchor_pos_sample(
                    next_obs.get("privilege"),
                    action_dim=self.cfg.action_space,
                )
                if relative_anchor_pos is None:
                    location_tracking_available = False
                    relative_anchor_pos_samples.clear()
                else:
                    relative_anchor_pos_samples.append(relative_anchor_pos)
            reward = task_reward

            self.tracker.add_values("episode_return", reward)
            self.tracker.add_values("episode_length", 1)
            done = terminate | timeout

            if done.any():
                episode_metrics = self._build_episode_metrics_payload(
                    self.tracker.get_mean("episode_return", done),
                    self.tracker.get_mean("episode_length", done),
                )
                episode_metrics.update(self._build_episode_finish_metrics_payload(terminate, timeout))
                self._log_metrics(episode_metrics)
                self.tracker.reset("episode_return", done)
                self.tracker.reset("episode_length", done)

            records = {
                "actions": action,
                "log_probs": log_prob,
                "rewards": reward,
                "values": value,
                "terminate": terminate,
            }
            records.update(get_policy_records(actor_obs, self.actor_type))
            records.update(get_critic_records(critic_obs))

            self.rollout_buffer.add_records(records)
            obs = next_obs

        if latest_curriculum_metrics:
            self._log_metrics(latest_curriculum_metrics)
        if location_tracking_available:
            location_tracking_payload = self._build_location_tracking_metrics(relative_anchor_pos_samples)
            if location_tracking_payload:
                self._log_metrics(location_tracking_payload)

        terminate_rate = (
            float(sum(terminate_rate_samples) / len(terminate_rate_samples)) if terminate_rate_samples else None
        )
        error_mean = (
            float(sum(end_effector_error_samples) / len(end_effector_error_samples))
            if end_effector_error_samples
            else None
        )
        self._last_end_effector_curriculum_rollout_metrics = {
            "terminate_rate": terminate_rate,
            "error_mean": error_mean,
        }

        actor_obs = get_actor_observation(obs, self.actor_type)
        critic_obs = get_critic_observation(self.critic, obs)
        self._update_actor_statistics(actor_obs)
        last_value = self.get_value(critic_obs, update_normlizer=True)
        returns, advantages = compute_gae(
            self.rollout_buffer.data["rewards"],
            self.rollout_buffer.data["values"],
            self.rollout_buffer.data["terminate"],
            last_value,
            0.99,
            0.95,
        )
        self.rollout_buffer.add_storage("returns", returns)
        self.rollout_buffer.add_storage("advantages", advantages)
        return obs

    def update(self):
        self.tracker.reset("policy_loss")
        self.tracker.reset("entropy_loss")
        self.tracker.reset("kl_divergence")
        self.tracker.reset("value_loss")
        self.tracker.reset("policy_clip_fraction")
        self.tracker.reset("action_log_std")
        self.tracker.reset("action_std")
        self.tracker.reset("advantage_mean")
        self.tracker.reset("advantage_std")
        self.tracker.reset("value_explained_variance")
        self.tracker.reset("value_clip_fraction")

        for _ in range(5):
            batch_iter = self.rollout_buffer.sample_batchs(self.batch_keys, 4096 * 10)

            for batch in batch_iter:
                policy_obs_batch = get_policy_batch(batch, self.actor_type, self.device)
                critic_obs_batch = get_critic_batch(batch, self.critic, self.device)
                action_batch = batch["actions"].to(self.device)
                log_prob_batch = batch["log_probs"].to(self.device)
                value_batch = batch["values"].to(self.device)
                return_batch = batch["returns"].to(self.device)
                advantage_batch = batch["advantages"].to(self.device)

                with autocast_context(self.device, self.use_amp):
                    policy_loss_dict = PPO.compute_policy_loss(
                        self.actor,
                        log_prob_batch,
                        policy_obs_batch,
                        action_batch,
                        advantage_batch,
                        PPO_CLIP_RATIO,
                        0.0,
                    )
                    value_loss_dict = PPO.compute_clipped_value_loss(
                        self.critic,
                        critic_obs_batch,
                        value_batch,
                        return_batch,
                        PPO_CLIP_RATIO,
                    )

                    policy_loss = policy_loss_dict["loss"]
                    entropy = policy_loss_dict["entropy"]
                    kl_divergence = policy_loss_dict["kl_divergence"]
                    value_loss = value_loss_dict["loss"]
                    actor_loss = policy_loss - entropy * ENTROPY_COEF
                    critic_loss = value_loss
                    ac_loss = actor_loss + critic_loss

                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                if self.use_amp:
                    self.grad_scaler.scale(ac_loss).backward()
                    self.grad_scaler.unscale_(self.actor_optimizer)
                    self.grad_scaler.unscale_(self.critic_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                    self.grad_scaler.step(self.actor_optimizer)
                    self.grad_scaler.step(self.critic_optimizer)
                    self.grad_scaler.update()
                else:
                    ac_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                    self.actor_optimizer.step()
                    self.critic_optimizer.step()
                self.lr_scheduler.set_kl(float(kl_divergence.detach().to(device="cpu", dtype=torch.float32).item()))
                self.lr_scheduler.step()

                self.tracker.add_values("policy_loss", policy_loss)
                self.tracker.add_values("entropy_loss", entropy)
                self.tracker.add_values("kl_divergence", kl_divergence)
                self.tracker.add_values("value_loss", value_loss)
                self.tracker.add_values("policy_clip_fraction", policy_loss_dict["policy_clip_fraction"])
                self.tracker.add_values("action_log_std", policy_loss_dict["action_log_std"])
                self.tracker.add_values("action_std", policy_loss_dict["action_std"])
                self.tracker.add_values("advantage_mean", policy_loss_dict["advantage_mean"])
                self.tracker.add_values("advantage_std", policy_loss_dict["advantage_std"])
                self.tracker.add_values("value_explained_variance", value_loss_dict["value_explained_variance"])
                self.tracker.add_values("value_clip_fraction", value_loss_dict["value_clip_fraction"])

        self._log_metrics(
            {
                "update/avg_policy_loss": self.tracker.get_mean("policy_loss"),
                "update/avg_value_loss": self.tracker.get_mean("value_loss"),
                "update/avg_entropy": self.tracker.get_mean("entropy_loss"),
                "update/avg_kl_divergence": self.tracker.get_mean("kl_divergence"),
                "update/avg_policy_clip_fraction": self.tracker.get_mean("policy_clip_fraction"),
                "update/avg_action_log_std": self.tracker.get_mean("action_log_std"),
                "update/avg_action_std": self.tracker.get_mean("action_std"),
                "update/avg_advantage_mean": self.tracker.get_mean("advantage_mean"),
                "update/avg_advantage_std": self.tracker.get_mean("advantage_std"),
                "update/avg_value_explained_variance": self.tracker.get_mean("value_explained_variance"),
                "update/avg_value_clip_fraction": self.tracker.get_mean("value_clip_fraction"),
                "update/actor_lr": self._optimizer_lr(self.actor_optimizer),
                "update/critic_lr": self._optimizer_lr(self.critic_optimizer),
            }
        )
        self.update_count += 1
        self._advance_end_effector_termination_curriculum()
        if self.update_count % self.anchor_log_interval == 0:
            self._log_anchor_reset_probabilities()

    def save_checkpoint(self, name: str) -> Path:
        joint_params = self.env.unwrapped.get_joint_params()
        checkpoint_name = f"{self.checkpoint_date}_{self.actor_type.value}_{name}"
        action_mode = joint_params.get("action_mode")
        if action_mode is None:
            action_mode = getattr(getattr(self.cfg, "action", None), "mode", None)
        if action_mode is None:
            action_mode = getattr(self.cfg, "action_mod", None)
        if hasattr(action_mode, "name"):
            action_mode = action_mode.name
        if action_mode is not None:
            action_mode = str(action_mode).replace("-", "_").lower()

        checkpoint_artifacts = {"run_dir": str(self.run_paths.root)}
        if self.resume_checkpoint_path is not None:
            checkpoint_artifacts["resume_checkpoint"] = str(self.resume_checkpoint_path)

        checkpoint = build_training_checkpoint(
            actor=self.actor,
            critic=self.critic,
            motion_files=self.motion_files,
            joint_params=joint_params,
            action_mode=action_mode,
            root_name=getattr(self.cfg, "root_link_name", None),
            anchor_body_name=getattr(self.cfg, "anchor_body_name", None),
            segment_source=self.segment_source,
            sampling_strategy=self.sampling_strategy,
            motion_mae_encoder_checkpoint=self.motion_mae_encoder_checkpoint,
            observation_window_lengths=self.observation_window_lengths,
            artifacts=checkpoint_artifacts,
            training=self._build_training_state(),
        )
        checkpoint_path = save_checkpoint_v2(checkpoint, self.run_paths.checkpoints_dir / f"{checkpoint_name}.pth")
        self._sync_latest_checkpoint_to_wandb(checkpoint_path)
        return checkpoint_path

    def _sync_latest_checkpoint_to_wandb(self, checkpoint_path: Path) -> None:
        if not self.use_wandb:
            return

        try:
            import wandb
        except ImportError:
            return

        run = getattr(wandb, "run", None)
        if run is None:
            return

        checkpoint_path = Path(checkpoint_path)
        latest_checkpoint_path = self.run_paths.checkpoints_dir / "latest_checkpoint.pth"
        try:
            if checkpoint_path.resolve() != latest_checkpoint_path.resolve():
                shutil.copyfile(checkpoint_path, latest_checkpoint_path)
        except Exception as exc:
            print(f"failed to prepare latest checkpoint for W&B: {exc}", flush=True)
            return

        run.summary["checkpoint/latest_local_path"] = str(checkpoint_path)
        run.summary["checkpoint/latest_wandb_file"] = "checkpoints/latest_checkpoint.pth"
        run.summary["checkpoint/latest_update"] = int(self.update_count)
        run.summary["checkpoint/latest_global_step"] = int(self.global_step)

        try:
            run.save(
                str(latest_checkpoint_path),
                base_path=str(self.run_paths.root),
                policy="now",
            )
        except Exception as exc:
            print(f"failed to sync latest checkpoint to W&B files: {exc}", flush=True)

    @staticmethod
    def _format_failure_reason(exc: BaseException) -> str:
        message = str(exc).strip()
        if not message:
            return type(exc).__name__
        return f"{type(exc).__name__}: {message}"

    def train(self):
        obs = self.initial_obs
        final_checkpoint_path: Path | None = None
        failure_stage = "startup"
        try:
            for _ in trange(self.config.num_updates):
                failure_stage = f"rollout update {self.update_count + 1}"
                obs = self.rollout(obs)
                failure_stage = f"policy update {self.update_count + 1}"
                self.update()
                if self.checkpoint_interval > 0 and self.update_count % self.checkpoint_interval == 0:
                    failure_stage = f"checkpoint update {self.update_count}"
                    final_checkpoint_path = self.save_checkpoint(str(self.update_count))

            failure_stage = "final checkpoint"
            final_checkpoint_path = self.save_checkpoint(f"{self.update_count}_final")
        except Exception as exc:
            print(f"training script failed during {failure_stage}: {self._format_failure_reason(exc)}", flush=True)
            raise
        finally:
            self.env.close()
            if self.use_wandb:
                WandbLogger.finish_project()

        summary = {
            "actor_type": self.actor_type.value,
            "actor_kwargs": self.actor_kwargs,
            "motion_files": self.motion_files,
            "motion_names": motion_names(self.motion_files),
            "motion_label": self.motion_name,
            "motion_file_inputs": self.motion_file_inputs,
            "motion_mae_encoder_checkpoint": self.motion_mae_encoder_checkpoint,
            "resume_checkpoint": str(self.resume_checkpoint_path) if self.resume_checkpoint_path is not None else None,
            "resume_mode": self.resume_mode,
            "resume_trainer_state_restored": self.resume_trainer_state_restored,
            "segment_source": self.segment_source,
            "sampling_strategy": self.sampling_strategy,
            "end_effector_termination_curriculum": self._end_effector_curriculum_state_dict(),
            "observation_window_lengths": self.observation_window_lengths,
            "num_updates": self.config.num_updates,
            "start_update_count": self.start_update_count,
            "final_update_count": self.update_count,
            "start_global_step": self.start_global_step,
            "final_global_step": self.global_step,
            "rollout_steps": self.steps,
            "sampler_failure_warmup_steps": self.sampler_failure_warmup_steps,
            "checkpoint_interval": self.checkpoint_interval,
            "anchor_log_interval": self.anchor_log_interval,
            "anchor_heatmap_bins": self.anchor_heatmap_bins,
            "anchor_probability_debug_dir": str(self.run_paths.debug_dir / "anchor_reset_probabilities"),
            "amp_requested": self.requested_amp,
            "amp_enabled": self.use_amp,
            "amp_dtype": self.amp_dtype,
            "final_checkpoint": str(final_checkpoint_path) if final_checkpoint_path is not None else None,
            "run_dir": str(self.run_paths.root),
        }
        write_json(self.run_paths.summary_path, summary)
        return summary
