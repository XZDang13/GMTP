import argparse

from isaaclab.app import AppLauncher

from tqdm import trange
import gymnasium
import torch

from RLAlg.scheduler import KLAdaptiveLR
from RLAlg.buffer.replay_buffer import ReplayBuffer, compute_gae
from RLAlg.nn.steps import StochasticContinuousPolicyStep, ValueStep
from RLAlg.alg.ppo import PPO
from RLAlg.logger import WandbLogger, MetricsTracker

from model.actor import AdaINActor, AdaINResActor, SplitEncoderActor, VanilaActor
from model.critic import Critic
from env.motions import motion_label, motion_names, resolve_motion_files


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
    parser.add_argument(
        "--actor-type",
        default="vanila",
        help="Actor architecture to train: vanila, split_encoder, adain, or adain_res.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


class OptimizerCollection(torch.optim.Optimizer):
    """Expose multiple optimizers as a single Optimizer for schedulers."""

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
        self.param_groups = [
            group
            for optimizer in self.optimizers
            for group in optimizer.param_groups
        ]

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
            raise ValueError(
                f"Expected {len(self.optimizers)} optimizer states, got {len(optimizer_states)}."
            )

        for optimizer, optimizer_state in zip(self.optimizers, optimizer_states):
            optimizer.load_state_dict(optimizer_state)

        self.param_groups = [
            group
            for optimizer in self.optimizers
            for group in optimizer.param_groups
        ]

class Trainer:
    @staticmethod
    def _normalize_actor_type(actor_type: str) -> str:
        normalized = actor_type.lower().replace("-", "_")
        alias_map = {
            "vanila": "vanila",
            "vanilla": "vanila",
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
    def _infer_observation_dims(obs: dict[str, torch.Tensor]) -> dict[str, int]:
        required_keys = ("motion", "robot", "privilege")
        missing_keys = [key for key in required_keys if key not in obs]
        if missing_keys:
            raise KeyError(f"Environment observation is missing required keys: {missing_keys}.")

        motion_dim = obs["motion"].shape[-1]
        robot_dim = obs["robot"].shape[-1]
        critic_dim = obs["privilege"].shape[-1]

        return {
            "motion": motion_dim,
            "robot": robot_dim,
            "critic": critic_dim,
            "policy": motion_dim + robot_dim,
        }

    @staticmethod
    def _build_actor(
        obs_dims: dict[str, int],
        actor_type: str,
        action_dim: int,
        adain_res_blocks: int,
    ) -> torch.nn.Module:
        if actor_type == "vanila":
            return VanilaActor(obs_dims["policy"], action_dim)
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
        if actor_type == "vanila":
            return {"obs": torch.cat((obs["motion"], obs["robot"]), dim=-1)}
        return {
            "motion_obs": obs["motion"],
            "robot_obs": obs["robot"],
        }

    @staticmethod
    def _get_policy_storage_specs(obs_dims: dict[str, int], actor_type: str) -> dict[str, tuple[int, ...]]:
        if actor_type == "vanila":
            return {"policy_observations": (obs_dims["policy"],)}
        return {
            "motion_observations": (obs_dims["motion"],),
            "robot_observations": (obs_dims["robot"],),
        }

    @staticmethod
    def _get_policy_records(actor_obs: dict[str, torch.Tensor], actor_type: str) -> dict[str, torch.Tensor]:
        if actor_type == "vanila":
            return {"policy_observations": actor_obs["obs"]}
        return {
            "motion_observations": actor_obs["motion_obs"],
            "robot_observations": actor_obs["robot_obs"],
        }

    @staticmethod
    def _get_policy_batch(
        batch: dict[str, torch.Tensor],
        actor_type: str,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        if actor_type == "vanila":
            return {"obs": batch["policy_observations"].to(device)}
        return {
            "motion_obs": batch["motion_observations"].to(device),
            "robot_obs": batch["robot_observations"].to(device),
        }

    @staticmethod
    def _get_critic_observation(obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return obs["privilege"]

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

    def __init__(self, actor_type: str, adain_res_blocks: int):
        from env.cfg import G1MultiMotionTrainingEnv

        self.cfg = G1MultiMotionTrainingEnv()
        self.cfg.expert_motion_file = resolve_motion_files(self.cfg.expert_motion_file)
        self.env_name = "G1MotionTracking-v0"
        #self.cfg.random_start = False

        self.env = gymnasium.make(self.env_name, cfg=self.cfg)

        print(self.cfg.scene.num_envs)

        self.device = self.env.unwrapped.device
        self.actor_type = self._normalize_actor_type(actor_type)
        self.adain_res_blocks = self._normalize_adain_res_blocks(adain_res_blocks)
        self.motion_files = list(self.cfg.expert_motion_file)
        self.motion_name = motion_label(self.motion_files)

        self.initial_obs, _ = self.env.reset()
        self.obs_dims = self._infer_observation_dims(self.initial_obs)

        critic_obs_dim = self.obs_dims["critic"]
        action_dim = self.cfg.action_space

        self.actor = self._build_actor(
            self.obs_dims,
            self.actor_type,
            action_dim,
            self.adain_res_blocks,
        ).to(self.device)
        self.critic = Critic(critic_obs_dim).to(self.device)
        muon_groups, adamw_groups, optimizer_stats = self._split_optimizer_param_groups(
            {
                "actor": self.actor,
                "critic": self.critic,
            }
        )

        optimizers = []
        if muon_groups:
            # Muon only supports 2D tensors such as linear weights.
            optimizers.append(torch.optim.Muon(muon_groups, lr=1e-3, weight_decay=0.0))
        if adamw_groups:
            optimizers.append(torch.optim.AdamW(adamw_groups, lr=1e-3, weight_decay=0.0))
        self.ac_optimizer = OptimizerCollection(*optimizers)

        print(
            "optimizer split:",
            f"Muon={optimizer_stats['muon_tensors']} tensors / {optimizer_stats['muon_numel']} params,",
            f"AdamW={optimizer_stats['adamw_tensors']} tensors / {optimizer_stats['adamw_numel']} params",
        )

        self.lr_scheduler = KLAdaptiveLR(self.ac_optimizer, 0.01)
        self.steps = 20

        self.rollout_buffer = ReplayBuffer(
            self.cfg.scene.num_envs,
            self.steps
        )

        self.policy_storage_specs = self._get_policy_storage_specs(self.obs_dims, self.actor_type)
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
        self.rollout_buffer.create_storage_space("critic_observations", (critic_obs_dim,), torch.float32)
        self.rollout_buffer.create_storage_space("actions", (action_dim,), torch.float32)
        self.rollout_buffer.create_storage_space("log_probs", (), torch.float32)
        self.rollout_buffer.create_storage_space("rewards", (), torch.float32)
        self.rollout_buffer.create_storage_space("values", (), torch.float32)
        self.rollout_buffer.create_storage_space("terminate", (), torch.float32)
        self.global_step = 0
        self.tracker = MetricsTracker()

        self.tracker.add_batch_metrics("episode_return", self.cfg.scene.num_envs)
        self.tracker.add_batch_metrics("episode_length", self.cfg.scene.num_envs)
        self.tracker.add_list_metrics("policy_loss")
        self.tracker.add_list_metrics("entropy_loss")
        self.tracker.add_list_metrics("kl_divergence")
        self.tracker.add_list_metrics("value_loss")

        WandbLogger.init_project("Mimic", f"G1_{self.motion_name}")
        
    @torch.no_grad()
    def get_action(
        self,
        actorobs_batch: dict[str, torch.Tensor],
        criticobs_batch: torch.Tensor,
        determine: bool = False,
    ):
        actor_step:StochasticContinuousPolicyStep = self.actor(actorobs_batch, update_normlizer=True)
        action = actor_step.action
        log_prob = actor_step.log_prob
        if determine:
            action = actor_step.mean
        
        critic_step:ValueStep = self.critic(criticobs_batch, update_normlizer=True)
        value = critic_step.value

        return action, log_prob, value
    
    def rollout(self, obs):
        for _ in range(self.steps):
            self.global_step += 1
            actor_obs = self._get_actor_observation(obs, self.actor_type)
            critic_obs = self._get_critic_observation(obs)
            action, log_prob, value = self.get_action(actor_obs, critic_obs)
            next_obs, task_reward, terminate, timeout, info = self.env.step(action)
            
            #reward = torch.sigmoid(task_reward)
            reward = task_reward
            #step_info = {}
            #for key, value in info.items():
            #    step_info[f"step/{key}"] = value

            #WandbLogger.log_metrics(step_info, self.global_step)

            self.tracker.add_values("episode_return", reward)
            self.tracker.add_values("episode_length", 1)

            done = terminate | timeout
            
            if done.any():
                log_ep_ret = self.tracker.get_mean("episode_return", done)
                log_ep_len = self.tracker.get_mean("episode_length", done)

                episode_info = {}
                episode_info['episode/mean_returns'] = log_ep_ret
                episode_info['episode/mean_length'] = log_ep_len

                self.tracker.reset("episode_return", done)
                self.tracker.reset("episode_length", done)

                WandbLogger.log_metrics(episode_info, self.global_step)

            records = {
                "critic_observations": critic_obs,
                "actions": action,
                "log_probs": log_prob,
                "rewards": reward,
                "values": value,
                "terminate": terminate
            }
            records.update(self._get_policy_records(actor_obs, self.actor_type))

            self.rollout_buffer.add_records(records)

            obs = next_obs

        actor_obs = self._get_actor_observation(obs, self.actor_type)
        critic_obs = self._get_critic_observation(obs)
        _, _, last_value = self.get_action(actor_obs, critic_obs)
        returns, advantages = compute_gae(
            self.rollout_buffer.data["rewards"],
            self.rollout_buffer.data["values"],
            self.rollout_buffer.data["terminate"],
            last_value,
            0.99,
            0.95
        )
        

        self.rollout_buffer.add_storage("returns", returns)
        self.rollout_buffer.add_storage("advantages", advantages)

        return obs
    
    def update(self):
        self.tracker.reset("policy_loss")
        self.tracker.reset("entropy_loss")
        self.tracker.reset("kl_divergence")
        self.tracker.reset("value_loss")

        for i in range(5):
            for batch in self.rollout_buffer.sample_batchs(self.batch_keys, 4096*10):
                policy_obs_batch = self._get_policy_batch(batch, self.actor_type, self.device)
                critic_obs_batch = batch["critic_observations"].to(self.device)
                action_batch = batch["actions"].to(self.device)
                log_prob_batch = batch["log_probs"].to(self.device)
                value_batch = batch["values"].to(self.device)
                return_batch = batch["returns"].to(self.device)
                advantage_batch = batch["advantages"].to(self.device)

                policy_loss_dict = PPO.compute_policy_loss(self.actor,
                                                           log_prob_batch,
                                                           policy_obs_batch,
                                                           action_batch,
                                                           advantage_batch,
                                                           0.2,
                                                           0.0)
                
                policy_loss = policy_loss_dict["loss"]
                entropy = policy_loss_dict["entropy"]
                kl_divergence = policy_loss_dict["kl_divergence"]

                value_loss_dict = PPO.compute_clipped_value_loss(self.critic,
                                                    critic_obs_batch,
                                                    value_batch,
                                                    return_batch,
                                                    0.2)
                
                value_loss = value_loss_dict["loss"]

                ac_loss = policy_loss - entropy * 0.005 + value_loss * 1.0

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

        avg_policy_loss = self.tracker.get_mean("policy_loss")
        avg_value_loss = self.tracker.get_mean("value_loss")
        avg_entropy = self.tracker.get_mean("entropy_loss")
        avg_kl_divergence = self.tracker.get_mean("kl_divergence")
        
        train_info = {
            "update/avg_policy_loss": avg_policy_loss,
            "update/avg_value_loss": avg_value_loss,
            "update/avg_entropy": avg_entropy,
            "update/avg_kl_divergence": avg_kl_divergence
        }

        WandbLogger.log_metrics(train_info, self.global_step)

    def save_weight(self, name:str):
        joint_params = self.env.unwrapped.get_joint_params()
        checkpoint_name = f"{self.motion_name}_{self.actor_type}_{name}"

        torch.save(
            {
                "actor_type": self.actor_type,
                "actor_kwargs": {"num_blocks": self.adain_res_blocks} if self.actor_type == "adain_res" else {},
                "motion_files": self.motion_files,
                "motion_names": motion_names(self.motion_files),
                "motion_label": self.motion_name,
                "actor": self.actor.state_dict(), 
                "critic": self.critic.state_dict(),
                "joint_names": joint_params["joint_names"],
                "joint_effort_limits": joint_params["joint_effort_limits"],
                "joint_pos_limits": joint_params["joint_pos_limits"],
                "joint_stiffness": joint_params["joint_stiffness"],
                "joint_damping": joint_params["joint_damping"],
                "action_offset": joint_params["action_offset"],
                "action_scale": joint_params["action_scale"],
            },
            f"{checkpoint_name}.pth"
        )

    def train(self):
        obs = self.initial_obs
        try:
            for epoch in trange(1000):
                obs = self.rollout(obs)
                self.update()

                #if (epoch+1) % 1000 == 0:
                #    self.save_weight(epoch+1)
            self.save_weight("final")
            print(self.env.unwrapped.sampler.bin_sample_counts)
        finally:
            self.env.close()
            WandbLogger.finish_project()


def main():
    args_cli = build_arg_parser().parse_args()
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    try:
        trainer = Trainer(args_cli.actor_type, 3)
        trainer.train()
    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()
