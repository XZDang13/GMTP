from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch
from RLAlg.alg.ppo import PPO
from RLAlg.buffer.replay_buffer import ReplayBuffer, compute_gae
from RLAlg.logger import MetricsTracker, WandbLogger
from RLAlg.nn.steps import ValueStep
from RLAlg.scheduler import KLAdaptiveLR
from tqdm import trange

from gmtp.integrations.ref2act.motion import motion_label, motion_names
from gmtp.models import (
    ActorType,
    Critic,
    RecurrentActor,
    build_actor,
    get_actor_kwargs,
    get_actor_observation,
    get_policy_batch,
    get_policy_records,
    get_policy_storage_specs,
    is_recurrent_actor,
    normalize_actor_type,
    policy_state_for_storage,
    policy_state_from_storage,
    unpack_actor_output,
)
from gmtp.runtime.checkpoints import build_training_checkpoint, save_checkpoint_v2
from gmtp.runtime.config import RunConfig
from gmtp.runtime.io import build_run_paths, write_json
from gmtp.runtime.observations import infer_env_observation_dims


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

        self.env, self.cfg = make_training_env()
        self.device = self.env.unwrapped.device
        self.actor_type = normalize_actor_type(config.actor_type)
        self.is_recurrent_actor = is_recurrent_actor(self.actor_type)
        self.motion_files = list(self.cfg.expert_motion_file)
        self.motion_name = motion_label(self.motion_files)
        self.run_date = datetime.now().strftime("%Y%m%d")
        self.checkpoint_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = config.run_name or f"G1_{len(self.motion_files)}_{self.run_date}"
        self.run_paths = build_run_paths(config.output_root, "train", self.run_name)
        self.checkpoint_interval = config.checkpoint_interval
        self.steps = config.rollout_steps
        self.sequence_batch_size = max(1, (4096 * 10) // self.steps)
        self.global_step = 0

        write_json(self.run_paths.config_path, {"command": "train", "config": self.config})

        self.initial_obs, _ = self.env.reset()
        self.obs_dims = infer_env_observation_dims(self.initial_obs)
        self.actor = build_actor(
            self.obs_dims,
            self.actor_type,
            self.cfg.action_space,
            actor_kwargs=self._build_actor_kwargs(),
        ).to(self.device)
        self.actor_kwargs = get_actor_kwargs(self.actor, self.actor_type)
        self.critic = Critic(self.obs_dims["critic"]).to(self.device)

        muon_groups, adamw_groups, optimizer_stats = self._split_optimizer_param_groups(
            {"actor": self.actor, "critic": self.critic}
        )
        optimizers = []
        if muon_groups:
            optimizers.append(torch.optim.Muon(muon_groups, lr=1e-3, weight_decay=0.0))
        if adamw_groups:
            optimizers.append(torch.optim.AdamW(adamw_groups, lr=1e-3, weight_decay=0.0))
        self.ac_optimizer = OptimizerCollection(*optimizers)
        self.lr_scheduler = KLAdaptiveLR(self.ac_optimizer, 0.01)

        self.rollout_buffer = ReplayBuffer(self.cfg.scene.num_envs, self.steps)
        self.policy_storage_specs = get_policy_storage_specs(self.obs_dims, self.actor_type)
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
        self.sequence_state_keys: list[str] = []

        for key, shape in self.policy_storage_specs.items():
            self.rollout_buffer.create_storage_space(key, shape, torch.float32)
        self.rollout_buffer.create_storage_space("critic_observations", (self.obs_dims["critic"],), torch.float32)
        self.rollout_buffer.create_storage_space("actions", (self.cfg.action_space,), torch.float32)
        self.rollout_buffer.create_storage_space("log_probs", (), torch.float32)
        self.rollout_buffer.create_storage_space("rewards", (), torch.float32)
        self.rollout_buffer.create_storage_space("values", (), torch.float32)
        self.rollout_buffer.create_storage_space("terminate", (), torch.float32)

        if self.is_recurrent_actor:
            self.rollout_buffer.create_storage_space("episode_starts", (), torch.bool)
            self.rollout_buffer.create_storage_space(
                "policy_rnn_state",
                (self.actor.num_layers, self.actor.hidden_size),
                torch.float32,
            )
            self.batch_keys.append("episode_starts")
            self.sequence_state_keys.append("policy_rnn_state")
            self.policy_state = self.actor.get_initial_state(self.cfg.scene.num_envs, device=self.device)
            self.episode_starts = torch.ones(self.cfg.scene.num_envs, dtype=torch.bool, device=self.device)
        else:
            self.policy_state = None
            self.episode_starts = None

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

    def _build_actor_kwargs(self) -> dict[str, int] | None:
        if self.is_recurrent_actor:
            return {
                "hidden_size": RecurrentActor.DEFAULT_HIDDEN_SIZE,
                "num_layers": RecurrentActor.DEFAULT_NUM_LAYERS,
            }
        if self.actor_type == ActorType.FILM_RES:
            return {"num_blocks": self.config.film_res_blocks}
        if self.actor_type == ActorType.FILM_ATTN_RES:
            return {
                "num_blocks": self.config.film_res_blocks,
                "attn_block_size": self.config.film_attn_res_block_size,
            }
        return None

    @staticmethod
    def _split_optimizer_param_groups(
        modules: dict[str, torch.nn.Module],
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
                if param.ndim == 2:
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
    def _get_critic_observation(obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return obs["privilege"]

    @torch.no_grad()
    def _update_actor_statistics(
        self,
        actor_obs_batch: dict[str, torch.Tensor],
        policy_state: torch.Tensor | None = None,
        episode_starts: torch.Tensor | None = None,
    ) -> None:
        if self.is_recurrent_actor:
            self.actor(
                actor_obs_batch,
                initial_state=policy_state,
                episode_starts=episode_starts,
                update_normlizer=True,
            )
            return
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
        policy_state: torch.Tensor | None = None,
        episode_starts: torch.Tensor | None = None,
    ):
        if self.is_recurrent_actor:
            actor_output = self.actor(
                actor_obs_batch,
                initial_state=policy_state,
                episode_starts=episode_starts,
                update_normlizer=True,
            )
        else:
            actor_output = self.actor(actor_obs_batch, update_normlizer=True)

        actor_step, next_policy_state = unpack_actor_output(actor_output)
        action = actor_step.mean if determine else actor_step.action
        log_prob = actor_step.log_prob
        value = self.get_value(critic_obs_batch, update_normlizer=True)
        return action, log_prob, value, next_policy_state

    def rollout(self, obs):
        for _ in range(self.steps):
            self.global_step += 1
            actor_obs = get_actor_observation(obs, self.actor_type)
            critic_obs = self._get_critic_observation(obs)
            current_policy_state = self.policy_state if self.is_recurrent_actor else None
            current_episode_starts = self.episode_starts if self.is_recurrent_actor else None
            action, log_prob, value, next_policy_state = self.get_action(
                actor_obs,
                critic_obs,
                policy_state=current_policy_state,
                episode_starts=current_episode_starts,
            )
            next_obs, task_reward, terminate, timeout, _info = self.env.step(action)
            reward = task_reward

            self.tracker.add_values("episode_return", reward)
            self.tracker.add_values("episode_length", 1)
            done = terminate | timeout

            if done.any():
                self._log_metrics(
                    {
                        "episode/mean_returns": self.tracker.get_mean("episode_return", done),
                        "episode/mean_length": self.tracker.get_mean("episode_length", done),
                    }
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
            if self.is_recurrent_actor:
                records["episode_starts"] = current_episode_starts
                records["policy_rnn_state"] = policy_state_for_storage(current_policy_state)

            self.rollout_buffer.add_records(records)

            if self.is_recurrent_actor:
                self.policy_state = next_policy_state
                self.episode_starts = done.to(dtype=torch.bool, device=self.device)
            obs = next_obs

        actor_obs = get_actor_observation(obs, self.actor_type)
        critic_obs = self._get_critic_observation(obs)
        if self.is_recurrent_actor:
            self._update_actor_statistics(actor_obs, self.policy_state, self.episode_starts)
        else:
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
            if self.is_recurrent_actor:
                batch_iter = self.rollout_buffer.sample_sequence_batches(
                    self.batch_keys,
                    seq_len=self.steps,
                    batch_size=self.sequence_batch_size,
                    state_keys=self.sequence_state_keys,
                )
            else:
                batch_iter = self.rollout_buffer.sample_batchs(self.batch_keys, 4096 * 10)

            for batch in batch_iter:
                policy_obs_batch = get_policy_batch(batch, self.actor_type, self.device)
                critic_obs_batch = batch["critic_observations"].to(self.device)
                action_batch = batch["actions"].to(self.device)
                log_prob_batch = batch["log_probs"].to(self.device)
                value_batch = batch["values"].to(self.device)
                return_batch = batch["returns"].to(self.device)
                advantage_batch = batch["advantages"].to(self.device)

                if self.is_recurrent_actor:
                    episode_starts_batch = batch["episode_starts"].to(self.device)
                    valid_mask = batch["valid_mask"].to(self.device)
                    initial_policy_state = policy_state_from_storage(batch["policy_rnn_state_init"], self.device)
                    policy_loss_dict = PPO.compute_policy_loss_recurrent(
                        self.actor,
                        log_prob_batch,
                        policy_obs_batch,
                        action_batch,
                        advantage_batch,
                        0.2,
                        episode_starts=episode_starts_batch,
                        initial_state=initial_policy_state,
                        valid_mask=valid_mask,
                        regularization_weight=0.0,
                    )
                    flat_valid_mask = valid_mask.reshape(-1)
                    flat_critic_obs = critic_obs_batch.reshape(-1, critic_obs_batch.shape[-1])[flat_valid_mask]
                    flat_value = value_batch.reshape(-1)[flat_valid_mask]
                    flat_return = return_batch.reshape(-1)[flat_valid_mask]
                    value_loss_dict = PPO.compute_clipped_value_loss(
                        self.critic,
                        flat_critic_obs,
                        flat_value,
                        flat_return,
                        0.2,
                    )
                else:
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

    def save_checkpoint(self, name: str) -> Path:
        joint_params = self.env.unwrapped.get_joint_params()
        checkpoint_name = f"{self.checkpoint_date}_{self.actor_type.value}_{name}"
        action_mode = getattr(self.cfg.action_mod, "name", self.cfg.action_mod)
        if action_mode is not None:
            action_mode = str(action_mode).replace("-", "_").lower()

        checkpoint = build_training_checkpoint(
            actor_type=self.actor_type.value,
            actor=self.actor,
            critic=self.critic,
            motion_files=self.motion_files,
            joint_params=joint_params,
            action_mode=action_mode,
            root_name=getattr(self.cfg, "root_link_name", None),
            anchor_body_name=getattr(self.cfg, "anchor_body_name", None),
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
            "num_updates": self.config.num_updates,
            "rollout_steps": self.steps,
            "checkpoint_interval": self.checkpoint_interval,
            "final_checkpoint": str(final_checkpoint_path) if final_checkpoint_path is not None else None,
            "run_dir": str(self.run_paths.root),
        }
        write_json(self.run_paths.summary_path, summary)
        return summary
