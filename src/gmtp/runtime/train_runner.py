from __future__ import annotations

import json
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
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
    get_actor_kwargs,
    get_actor_observation,
    get_policy_batch,
    get_policy_records,
    get_policy_storage_specs,
)
from gmtp.runtime.checkpoints import build_training_checkpoint, save_checkpoint_v2
from gmtp.runtime.amp import AMP_DTYPE_NAME, autocast_context, build_grad_scaler, normalize_device, resolve_amp_enabled
from gmtp.runtime.config import RunConfig
from gmtp.runtime.io import build_run_paths, write_json
from gmtp.runtime.observations import infer_env_observation_dims, structure_env_observation
from gmtp.runtime.policy import resolve_motion_mae_checkpoint_path

RECOVERY_METRIC_LOG_NAMES = {
    "fall_recovery/active_rate": "recovery/active_rate",
    "fall_recovery/entry_rate": "recovery/entry_rate",
    "fall_recovery/exit_rate": "recovery/exit_rate",
    "fall_recovery/timeout_rate": "recovery/timeout_rate",
    "fall_recovery/reference_time_scale_mean": "recovery/reference_time_scale_mean",
    "tracking_quality/score_mean": "recovery/tracking_score_mean",
}
RECOVERY_ENTRY_RATE_KEY = "fall_recovery/entry_rate"
RECOVERY_EXIT_RATE_KEY = "fall_recovery/exit_rate"
RECOVERY_TIMEOUT_RATE_KEY = "fall_recovery/timeout_rate"
RECOVERY_RATIO_EPS = 1.0e-8
PPO_CLIP_RATIO = 0.2
ENTROPY_COEF = 0.005
ANCHOR_HEATMAP_TOP_LABELS = 30
ANCHOR_CONSOLE_TOP_K = 10


@dataclass(frozen=True)
class AnchorProbabilityArrays:
    motion_index: np.ndarray
    motion_name: np.ndarray
    anchor_index: np.ndarray
    anchor_time: np.ndarray
    probability: np.ndarray


@dataclass(frozen=True)
class AnchorHeatmapGrid:
    values: np.ndarray
    motion_indices: np.ndarray
    motion_names: list[str]
    motion_probabilities: np.ndarray
    num_bins: int


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
        from gmtp.integrations.ref2act.isaac_env import make_training_env

        self.observation_window_lengths = resolve_observation_window_lengths(
            robot_window_length=config.robot_window_length,
            motion_window_length=config.motion_window_length,
        )
        self.env, self.cfg = make_training_env(window_lengths=self.observation_window_lengths)
        self.device = normalize_device(self.env.unwrapped.device)
        self.requested_amp = bool(config.use_amp)
        self.use_amp = resolve_amp_enabled(self.requested_amp, self.device)
        self.amp_dtype = AMP_DTYPE_NAME
        self.actor_type = ActorType.FILM_RES
        self.segment_source = self._normalize_choice_name(getattr(self.cfg, "segment_source", None))
        self.sampling_strategy = self._normalize_choice_name(getattr(self.cfg, "sampling_strategy", None))
        self.motion_files = list(self.cfg.expert_motion_file)
        self.motion_name = motion_label(self.motion_files)
        resolved_motion_mae_checkpoint = resolve_motion_mae_checkpoint_path(
            override=config.motion_mae_encoder_checkpoint,
        )
        self.motion_mae_encoder_checkpoint = (
            None if resolved_motion_mae_checkpoint is None else str(resolved_motion_mae_checkpoint)
        )
        self.run_date = datetime.now().strftime("%Y%m%d")
        self.checkpoint_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = config.run_name or f"G1_{len(self.motion_files)}_{self.run_date}"
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

        self.initial_obs, _ = self.env.reset()
        self.initial_obs = structure_env_observation(
            self.initial_obs,
            action_dim=self.cfg.action_space,
            observation_window_lengths=self.observation_window_lengths,
        )
        self.raw_obs_dims = infer_env_observation_dims(self.initial_obs)
        self.obs_dims = self.raw_obs_dims
        self.actor = build_actor(
            self.obs_dims,
            self.actor_type,
            self.cfg.action_space,
            actor_kwargs=self._build_actor_kwargs(),
            motion_mae_encoder_checkpoint=self.motion_mae_encoder_checkpoint,
            device=self.device,
        ).to(self.device)
        self.actor_kwargs = get_actor_kwargs(self.actor, self.actor_type)
        self.critic = Critic(self.obs_dims["critic"], action_dim=self.cfg.action_space).to(self.device)

        self.actor_optimizer, actor_optimizer_stats = self._build_optimizer_collection(
            {"actor": self.actor},
            prefer_muon=self.device.type == "cuda",
        )
        self.critic_optimizer, critic_optimizer_stats = self._build_optimizer_collection(
            {"critic": self.critic},
            prefer_muon=self.device.type == "cuda",
        )
        self.grad_scaler = build_grad_scaler(self.use_amp)
        self.lr_scheduler = KLAdaptiveLR(self.actor_optimizer, 0.01)

        self.rollout_buffer = ReplayBuffer(self.cfg.scene.num_envs, self.steps)
        self.policy_storage_specs = get_policy_storage_specs(
            self.obs_dims,
            self.actor_type,
            actor_kwargs=self.actor_kwargs,
        )
        self.policy_batch_keys = list(self.policy_storage_specs)
        self.batch_keys = [
            *self.policy_batch_keys,
            "critic_observations",
            "actions",
            "log_probs",
            "rewards",
            "values",
            "returns",
            "advantages",
        ]
        for key, shape in self.policy_storage_specs.items():
            self.rollout_buffer.create_storage_space(key, shape, torch.float32)
        self.rollout_buffer.create_storage_space("critic_observations", (self.obs_dims["critic"],), torch.float32)
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
            WandbLogger.init_project("Mimic", self.run_name)

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

    def _build_actor_kwargs(self) -> dict[str, int | str]:
        return {
            "num_blocks": self.config.num_blocks,
            "robot_window_length": self.config.robot_window_length,
            "robot_encoder_type": self.config.robot_encoder_type,
            "motion_window_length": self.config.motion_window_length,
            "motion_encoder_type": self.config.motion_encoder_type,
            "actor_fusion_type": self.config.actor_fusion_type,
        }

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
    def _extract_recovery_metric_sample_from_mapping(cls, payload: Mapping[str, Any] | None) -> dict[str, float]:
        if not isinstance(payload, Mapping):
            return {}

        sample: dict[str, float] = {}
        for metric_name, value in cls._iter_metric_items(payload):
            normalized_name = metric_name.strip("/")
            for source_name in RECOVERY_METRIC_LOG_NAMES:
                if normalized_name != source_name and not normalized_name.endswith(f"/{source_name}"):
                    continue
                scalar = cls._coerce_metric_scalar(value)
                if scalar is not None:
                    sample[source_name] = scalar
                break
        return sample

    def _extract_recovery_metric_sample(self, info: Mapping[str, Any] | None) -> dict[str, float]:
        sample = self._extract_recovery_metric_sample_from_mapping(info)
        missing_source_names = set(RECOVERY_METRIC_LOG_NAMES) - set(sample)
        if not missing_source_names:
            return sample

        extras = getattr(self.env.unwrapped, "extras", None)
        extras_sample = self._extract_recovery_metric_sample_from_mapping(extras)
        for source_name in missing_source_names:
            if source_name in extras_sample:
                sample[source_name] = extras_sample[source_name]
        return sample

    @staticmethod
    def _build_recovery_metrics_payload(metric_samples: list[dict[str, float]]) -> dict[str, float]:
        if not metric_samples:
            return {}

        payload: dict[str, float] = {}
        for source_name, log_name in RECOVERY_METRIC_LOG_NAMES.items():
            values = [sample[source_name] for sample in metric_samples if source_name in sample]
            if values:
                payload[log_name] = float(sum(values) / len(values))

        entry_sum = sum(sample.get(RECOVERY_ENTRY_RATE_KEY, 0.0) for sample in metric_samples)
        exit_sum = sum(sample.get(RECOVERY_EXIT_RATE_KEY, 0.0) for sample in metric_samples)
        timeout_sum = sum(sample.get(RECOVERY_TIMEOUT_RATE_KEY, 0.0) for sample in metric_samples)
        if any(
            key in sample
            for sample in metric_samples
            for key in (RECOVERY_ENTRY_RATE_KEY, RECOVERY_EXIT_RATE_KEY, RECOVERY_TIMEOUT_RATE_KEY)
        ):
            denominator = max(entry_sum, RECOVERY_RATIO_EPS)
            payload["recovery/exit_to_entry_ratio"] = float(exit_sum / denominator)
            payload["recovery/timeout_to_entry_ratio"] = float(timeout_sum / denominator)

        return payload

    @staticmethod
    def _build_episode_metrics_payload(mean_return: float, mean_length: float) -> dict[str, float]:
        return {
            "episode/returns": float(mean_return),
            "episode/lengths": float(mean_length),
        }

    @staticmethod
    def _sanitize_metric_component(value: str) -> str:
        sanitized = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value)).strip("_")
        return sanitized or "unknown"

    @staticmethod
    def _build_guarded_sampling_probabilities(
        fail_counts: torch.Tensor,
        sample_counts: torch.Tensor,
        *,
        temperature: float,
        uniform_mix: float,
        eligible_mask: torch.Tensor | None = None,
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

        fail_rate = fail_counts / torch.clamp(sample_counts, min=1.0)
        learned_weights = fail_rate.pow(1.0 / temperature)
        learned_weights = torch.where(eligible, learned_weights, torch.zeros_like(learned_weights))

        learned_sum = torch.sum(learned_weights)
        if bool(torch.all(torch.isfinite(learned_weights)).item()) and float(learned_sum.item()) > 0.0:
            learned_probs = learned_weights / learned_sum
        else:
            learned_probs = uniform_probs

        probs = (1.0 - uniform_mix) * learned_probs + uniform_mix * uniform_probs
        return probs / torch.clamp(torch.sum(probs), min=torch.finfo(probs.dtype).eps)

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

        uniform_mix = float(getattr(sampler, "failure_weight_uniform_mix", 0.0))
        motion_fail_counts = torch.stack(
            [cls._as_cpu_tensor(fail_counts, dtype=torch.float32).sum() for fail_counts in bin_fail_counts],
            dim=0,
        )
        motion_sample_counts = torch.stack(
            [cls._as_cpu_tensor(sample_counts, dtype=torch.float32).sum() for sample_counts in bin_sample_counts],
            dim=0,
        )
        motion_probs = cls._build_guarded_sampling_probabilities(
            motion_fail_counts,
            motion_sample_counts,
            temperature=temperature,
            uniform_mix=uniform_mix,
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
                bin_probs = cls._build_guarded_sampling_probabilities(
                    fail_counts,
                    sample_counts,
                    temperature=temperature,
                    uniform_mix=uniform_mix,
                    eligible_mask=eligible_mask,
                )
                for bin_index in torch.nonzero(eligible_mask, as_tuple=False).squeeze(-1).tolist():
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
        sampler = getattr(self.env.unwrapped, "sampler", None)
        if sampler is None:
            return []
        if self._normalize_choice_name(getattr(sampler, "segment_source", self.segment_source)) != "anchor":
            return []
        return self._compute_anchor_reset_probabilities(
            sampler,
            temperature=float(getattr(self.cfg, "failure_temperature", 1.0)),
        )

    @classmethod
    def _build_anchor_reset_probability_metrics(
        cls,
        anchor_probabilities: list[dict[str, float | int | str]],
    ) -> dict[str, float]:
        arrays = cls._build_anchor_probability_arrays(anchor_probabilities)
        probabilities = np.maximum(arrays.probability.astype(np.float64, copy=False), 0.0)
        motion_probabilities = cls._aggregate_probabilities_by_motion(arrays)

        return {
            "sampling/anchor_reset_probability/sum": float(np.sum(probabilities)),
            "sampling/anchor_reset_probability/max": float(np.max(probabilities)) if probabilities.size else 0.0,
            "sampling/anchor_reset_probability/entropy": cls._probability_entropy(probabilities),
            "sampling/anchor_reset_probability/effective_anchors": cls._effective_probability_count(probabilities),
            "sampling/anchor_reset_probability/active_anchors": float(np.count_nonzero(probabilities > 0.0)),
            "sampling/anchor_reset_probability/num_anchors": float(probabilities.size),
            "sampling/anchor_reset_probability/top1_mass": cls._top_probability_mass(probabilities, 1),
            "sampling/anchor_reset_probability/top5_mass": cls._top_probability_mass(probabilities, 5),
            "sampling/anchor_reset_probability/top20_mass": cls._top_probability_mass(probabilities, 20),
            "sampling/motion_reset_probability/max": (
                float(np.max(motion_probabilities)) if motion_probabilities.size else 0.0
            ),
            "sampling/motion_reset_probability/entropy": cls._probability_entropy(motion_probabilities),
            "sampling/motion_reset_probability/effective_motions": cls._effective_probability_count(
                motion_probabilities
            ),
            "sampling/motion_reset_probability/active_motions": float(np.count_nonzero(motion_probabilities > 0.0)),
            "sampling/motion_reset_probability/num_motions": float(motion_probabilities.size),
            "sampling/motion_reset_probability/top1_mass": cls._top_probability_mass(motion_probabilities, 1),
            "sampling/motion_reset_probability/top5_mass": cls._top_probability_mass(motion_probabilities, 5),
            "sampling/motion_reset_probability/top10_mass": cls._top_probability_mass(motion_probabilities, 10),
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

    @classmethod
    def _build_anchor_reset_probability_heatmap_grid(
        cls,
        anchor_probabilities: list[dict[str, float | int | str]],
        *,
        num_bins: int,
    ) -> AnchorHeatmapGrid:
        if num_bins < 1:
            raise ValueError("num_bins must be positive.")

        arrays = cls._build_anchor_probability_arrays(anchor_probabilities)
        if arrays.motion_index.size == 0:
            return AnchorHeatmapGrid(
                values=np.zeros((0, num_bins), dtype=np.float32),
                motion_indices=np.asarray([], dtype=np.int64),
                motion_names=[],
                motion_probabilities=np.asarray([], dtype=np.float32),
                num_bins=num_bins,
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
        return AnchorHeatmapGrid(
            values=np.stack([rows[index] for index in order], axis=0),
            motion_indices=np.asarray([motion_indices[index] for index in order], dtype=np.int64),
            motion_names=[motion_names_list[index] for index in order],
            motion_probabilities=np.asarray([motion_probabilities[index] for index in order], dtype=np.float32),
            num_bins=num_bins,
        )

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
        grid: AnchorHeatmapGrid,
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
        num_motions = max(1, len(grid.motion_names))
        figure_height = min(18.0, max(6.0, 3.0 + num_motions * 0.018))
        figure, axis = plt.subplots(figsize=(14.0, figure_height))

        values = np.asarray(grid.values, dtype=np.float32)
        if values.size == 0:
            values = np.zeros((1, grid.num_bins), dtype=np.float32)
        masked_values = np.ma.masked_less_equal(values, 0.0)
        positive_values = values[values > 0.0]

        color_map = plt.get_cmap("magma").copy()
        color_map.set_bad("#f0f0f0")
        if positive_values.size:
            max_probability = float(np.max(positive_values))
            min_probability = float(np.min(positive_values))
            if min_probability >= max_probability:
                min_probability = max_probability * 0.1
            image = axis.imshow(
                masked_values,
                aspect="auto",
                interpolation="nearest",
                cmap=color_map,
                norm=LogNorm(vmin=max(min_probability, 1.0e-12), vmax=max_probability),
            )
        else:
            image = axis.imshow(values, aspect="auto", interpolation="nearest", cmap=color_map, vmin=0.0, vmax=1.0)

        axis.set_xlabel("normalized motion time")
        axis.set_ylabel("motion rank by reset probability")
        last_bin_index = max(0, grid.num_bins - 1)
        axis.set_xticks(
            [0, last_bin_index * 0.25, last_bin_index * 0.5, last_bin_index * 0.75, last_bin_index],
            ["0.00", "0.25", "0.50", "0.75", "1.00"],
        )
        label_count = min(ANCHOR_HEATMAP_TOP_LABELS, len(grid.motion_names))
        axis.set_yticks(np.arange(label_count), grid.motion_names[:label_count])

        active_anchors = int(metrics_payload.get("sampling/anchor_reset_probability/active_anchors", 0.0))
        effective_anchors = metrics_payload.get("sampling/anchor_reset_probability/effective_anchors", 0.0)
        entropy = metrics_payload.get("sampling/anchor_reset_probability/entropy", 0.0)
        max_probability = metrics_payload.get("sampling/anchor_reset_probability/max", 0.0)
        axis.set_title(
            "Motion anchor reset probability heatmap\n"
            f"update={update_count} step={global_step} active={active_anchors} "
            f"max={max_probability:.3g} entropy={entropy:.3f} effective={effective_anchors:.1f}"
        )
        figure.colorbar(image, ax=axis, label="reset probability mass")
        figure.tight_layout()
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

        grid = self._build_anchor_reset_probability_heatmap_grid(
            anchor_probabilities,
            num_bins=self.anchor_heatmap_bins,
        )
        heatmap_path = output_dir / f"{snapshot_stem}_heatmap.png"
        latest_heatmap_path = output_dir / "latest_heatmap.png"
        try:
            self._write_anchor_probability_heatmap(
                grid,
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
        run.summary["sampling/anchor_reset_probability/latest_update"] = int(self.update_count)
        run.summary["sampling/anchor_reset_probability/latest_global_step"] = int(self.global_step)

        latest_heatmap_path = artifacts.get("latest_heatmap_png")
        if latest_heatmap_path is None:
            return

        try:
            run.summary["sampling/anchor_reset_probability/latest_heatmap"] = wandb.Image(
                latest_heatmap_path,
                caption=f"anchor reset probabilities update={self.update_count} step={self.global_step}",
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

        metrics_payload = self._build_anchor_reset_probability_metrics(anchor_probabilities)
        if metrics_payload:
            self._log_metrics(metrics_payload)
        artifacts = self._write_anchor_reset_probability_artifacts(anchor_probabilities, metrics_payload)
        self._sync_anchor_reset_probability_summary_to_wandb(metrics_payload, artifacts)

        print(
            f"anchor reset probabilities after update {self.update_count}: "
            f"max={metrics_payload['sampling/anchor_reset_probability/max']:.6f} "
            f"active={int(metrics_payload['sampling/anchor_reset_probability/active_anchors'])} "
            f"effective={metrics_payload['sampling/anchor_reset_probability/effective_anchors']:.2f} "
            f"top20={metrics_payload['sampling/anchor_reset_probability/top20_mass']:.6f}",
            flush=True,
        )
        top_entries = sorted(
            anchor_probabilities,
            key=lambda entry: float(entry["probability"]),
            reverse=True,
        )[:ANCHOR_CONSOLE_TOP_K]
        for entry in top_entries:
            print(
                f"  {entry['motion_name']} A{int(entry['anchor_index'])} "
                f"t={float(entry['anchor_time']):.3f}s p={float(entry['probability']):.6f}",
                flush=True,
            )
        if latest_heatmap_path := artifacts.get("latest_heatmap_png"):
            print(f"  latest heatmap: {latest_heatmap_path}", flush=True)

    @staticmethod
    def _get_critic_observation(obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return obs["privilege"]

    @torch.no_grad()
    def _update_actor_statistics(
        self,
        actor_obs_batch: dict[str, torch.Tensor],
    ) -> None:
        self.actor(actor_obs_batch, update_normlizer=True)

    @torch.no_grad()
    def get_value(self, critic_obs_batch: torch.Tensor, update_normlizer: bool = True) -> torch.Tensor:
        critic_step: ValueStep = self.critic(critic_obs_batch, update_normlizer=update_normlizer)
        return critic_step.value

    @torch.no_grad()
    def get_action(
        self,
        actor_obs_batch: dict[str, torch.Tensor],
        critic_obs_batch: torch.Tensor,
        determine: bool = False,
    ):
        actor_step = self.actor(actor_obs_batch, update_normlizer=True)
        action = actor_step.mean if determine else actor_step.action
        log_prob = actor_step.log_prob
        value = self.get_value(critic_obs_batch, update_normlizer=True)
        return action, log_prob, value

    def rollout(self, obs):
        recovery_metric_samples: list[dict[str, float]] = []
        for _ in range(self.steps):
            self.global_step += 1
            actor_obs = get_actor_observation(obs, self.actor_type)
            critic_obs = self._get_critic_observation(obs)
            action, log_prob, value = self.get_action(actor_obs, critic_obs)
            next_obs, task_reward, terminate, timeout, info = self.env.step(action)
            recovery_metric_sample = self._extract_recovery_metric_sample(info)
            if recovery_metric_sample:
                recovery_metric_samples.append(recovery_metric_sample)
            next_obs = structure_env_observation(
                next_obs,
                action_dim=self.cfg.action_space,
                observation_window_lengths=self.observation_window_lengths,
            )
            reward = task_reward

            self.tracker.add_values("episode_return", reward)
            self.tracker.add_values("episode_length", 1)
            done = terminate | timeout

            if done.any():
                self._log_metrics(
                    self._build_episode_metrics_payload(
                        self.tracker.get_mean("episode_return", done),
                        self.tracker.get_mean("episode_length", done),
                    )
                )
                self.tracker.reset("episode_return", done)
                self.tracker.reset("episode_length", done)

            records = {
                "critic_observations": critic_obs,
                "actions": action,
                "log_probs": log_prob,
                "rewards": reward,
                "values": value,
                "terminate": terminate,
            }
            records.update(get_policy_records(actor_obs, self.actor_type))

            self.rollout_buffer.add_records(records)
            obs = next_obs

        recovery_metrics_payload = self._build_recovery_metrics_payload(recovery_metric_samples)
        if recovery_metrics_payload:
            self._log_metrics(recovery_metrics_payload)

        actor_obs = get_actor_observation(obs, self.actor_type)
        critic_obs = self._get_critic_observation(obs)
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
                critic_obs_batch = batch["critic_observations"].to(self.device)
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
            artifacts={"run_dir": str(self.run_paths.root)},
        )
        return save_checkpoint_v2(checkpoint, self.run_paths.checkpoints_dir / f"{checkpoint_name}.pth")

    def train(self):
        obs = self.initial_obs
        final_checkpoint_path: Path | None = None
        try:
            for epoch in trange(self.config.num_updates):
                obs = self.rollout(obs)
                self.update()
                if (epoch + 1) % self.checkpoint_interval == 0:
                    final_checkpoint_path = self.save_checkpoint(str(epoch + 1))

            final_checkpoint_path = self.save_checkpoint("final")
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
            "motion_mae_encoder_checkpoint": self.motion_mae_encoder_checkpoint,
            "segment_source": self.segment_source,
            "sampling_strategy": self.sampling_strategy,
            "observation_window_lengths": self.observation_window_lengths,
            "num_updates": self.config.num_updates,
            "rollout_steps": self.steps,
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
