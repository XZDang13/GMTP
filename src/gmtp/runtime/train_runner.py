from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

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
        self.critic = Critic(self.obs_dims["critic"]).to(self.device)

        muon_groups, adamw_groups, optimizer_stats = self._split_optimizer_param_groups(
            {"actor": self.actor, "critic": self.critic},
            prefer_muon=self.device.type == "cuda",
        )
        optimizers = []
        if muon_groups:
            optimizers.append(torch.optim.Muon(muon_groups, lr=1e-3, weight_decay=0.0))
        if adamw_groups:
            optimizers.append(torch.optim.AdamW(adamw_groups, lr=1e-3, weight_decay=0.0))
        self.ac_optimizer = OptimizerCollection(*optimizers)
        self.grad_scaler = build_grad_scaler(self.use_amp)
        self.lr_scheduler = KLAdaptiveLR(self.ac_optimizer, 0.01)

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

        self.use_wandb = bool(config.use_wandb)
        if self.use_wandb:
            WandbLogger.init_project("Mimic", self.run_name)

        print(
            "optimizer split:",
            f"Muon={optimizer_stats['muon_tensors']} tensors / {optimizer_stats['muon_numel']} params,",
            f"AdamW={optimizer_stats['adamw_tensors']} tensors / {optimizer_stats['adamw_numel']} params",
        )

    def _build_actor_kwargs(self) -> dict[str, int | str]:
        return {
            "num_blocks": self.config.num_blocks,
            "robot_window_length": self.config.robot_window_length,
            "robot_encoder_type": self.config.robot_encoder_type,
            "motion_window_length": self.config.motion_window_length,
            "motion_encoder_type": self.config.motion_encoder_type,
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

    def _log_metrics(self, payload: dict[str, float]) -> None:
        if self.use_wandb:
            WandbLogger.log_metrics(payload, self.global_step)

    @staticmethod
    def _build_episode_metrics_payload(mean_return: float, mean_length: float) -> dict[str, float]:
        return {
            "episode/returns": float(mean_return),
            "episode/lengths": float(mean_length),
        }

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
        if not clips or bin_fail_counts is None or bin_sample_counts is None or bin_reset_eligible is None or bin_reset_times is None:
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

    def _log_anchor_reset_probabilities(self) -> None:
        anchor_probabilities = self._collect_anchor_reset_probabilities()
        if not anchor_probabilities:
            return

        print(f"anchor reset probabilities after update {self.update_count}:", flush=True)
        current_motion_name: str | None = None
        for entry in anchor_probabilities:
            motion_name = str(entry["motion_name"])
            if motion_name != current_motion_name:
                current_motion_name = motion_name
                print(f"  motion={motion_name}", flush=True)
            print(
                f"    A{int(entry['anchor_index'])} t={float(entry['anchor_time']):.3f}s "
                f"p={float(entry['probability']):.6f}",
                flush=True,
            )

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
        for _ in range(self.steps):
            self.global_step += 1
            actor_obs = get_actor_observation(obs, self.actor_type)
            critic_obs = self._get_critic_observation(obs)
            action, log_prob, value = self.get_action(actor_obs, critic_obs)
            next_obs, task_reward, terminate, timeout, _info = self.env.step(action)
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
                        0.2,
                        0.0,
                    )
                    value_loss_dict = PPO.compute_clipped_value_loss(
                        self.critic,
                        critic_obs_batch,
                        value_batch,
                        return_batch,
                        0.2,
                    )

                    policy_loss = policy_loss_dict["loss"]
                    entropy = policy_loss_dict["entropy"]
                    kl_divergence = policy_loss_dict["kl_divergence"]
                    value_loss = value_loss_dict["loss"]
                    ac_loss = policy_loss - entropy * 0.005 + value_loss

                self.ac_optimizer.zero_grad(set_to_none=True)
                if self.use_amp:
                    self.grad_scaler.scale(ac_loss).backward()
                    self.grad_scaler.unscale_(self.ac_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                    self.grad_scaler.step(self.ac_optimizer)
                    self.grad_scaler.update()
                else:
                    ac_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                    self.ac_optimizer.step()
                self.lr_scheduler.set_kl(kl_divergence)
                self.lr_scheduler.step()

                self.tracker.add_values("policy_loss", policy_loss)
                self.tracker.add_values("entropy_loss", entropy)
                self.tracker.add_values("kl_divergence", kl_divergence)
                self.tracker.add_values("value_loss", value_loss)

        self._log_metrics(
            {
                "update/avg_policy_loss": self.tracker.get_mean("policy_loss"),
                "update/avg_value_loss": self.tracker.get_mean("value_loss"),
                "update/avg_entropy": self.tracker.get_mean("entropy_loss"),
                "update/avg_kl_divergence": self.tracker.get_mean("kl_divergence"),
            }
        )
        self.update_count += 1
        if self.update_count % 100 == 0:
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
            "amp_requested": self.requested_amp,
            "amp_enabled": self.use_amp,
            "amp_dtype": self.amp_dtype,
            "final_checkpoint": str(final_checkpoint_path) if final_checkpoint_path is not None else None,
            "run_dir": str(self.run_paths.root),
        }
        write_json(self.run_paths.summary_path, summary)
        return summary
