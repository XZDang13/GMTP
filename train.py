import argparse
from pathlib import Path

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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
    parser.add_argument(
        "--actor-type",
        default="vanila",
        help="Actor architecture to train: vanila, split_encoder, adain, or adain_res.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser

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
    def _build_actor(cfg, actor_type: str, action_dim: int) -> torch.nn.Module:
        if actor_type == "vanila":
            return VanilaActor(cfg.policy_observation_space, action_dim)
        if actor_type == "split_encoder":
            return SplitEncoderActor(cfg.robot_observation_space, cfg.motion_observation_space, action_dim)
        if actor_type == "adain":
            return AdaINActor(cfg.robot_observation_space, cfg.motion_observation_space, action_dim)
        if actor_type == "adain_res":
            return AdaINResActor(cfg.robot_observation_space, cfg.motion_observation_space, action_dim)
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
    def _get_policy_storage_specs(cfg, actor_type: str) -> dict[str, tuple[int, ...]]:
        if actor_type == "vanila":
            return {"policy_observations": (cfg.policy_observation_space,)}
        return {
            "motion_observations": (cfg.motion_observation_space,),
            "robot_observations": (cfg.robot_observation_space,),
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

    def __init__(self, actor_type: str):
        from env.cfg import G1JabTrainingEnv

        self.cfg = G1JabTrainingEnv()
        self.env_name = "G1MotionTracking-v0"
        #self.cfg.random_start = False

        self.env = gymnasium.make(self.env_name, cfg=self.cfg)

        print(self.cfg.scene.num_envs)

        self.device = self.env.unwrapped.device
        self.actor_type = self._normalize_actor_type(actor_type)
        self.motion_name = Path(self.cfg.expert_motion_file).stem if self.cfg.expert_motion_file else "motion_tracking"
        
        critic_obs_dim = self.cfg.critic_observation_space
        action_dim = self.cfg.action_space

        self.actor = self._build_actor(self.cfg, self.actor_type, action_dim).to(self.device)
        self.critic = Critic(critic_obs_dim).to(self.device)

        self.ac_optimizer = torch.optim.Adam(
            [
                {'params': self.actor.parameters(),
                 "name": "actor"},
                 {'params': self.critic.parameters(),
                 "name": "critic"},
            ],
            lr=1e-3
        )

        self.lr_scheduler = KLAdaptiveLR(self.ac_optimizer, 0.01)

        self.steps = 20

        self.rollout_buffer = ReplayBuffer(
            self.cfg.scene.num_envs,
            self.steps
        )

        self.policy_storage_specs = self._get_policy_storage_specs(self.cfg, self.actor_type)
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


        WandbLogger.init_project("Mimic", f"G1_Pick")
        
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
        obs, _ = self.env.reset()
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
        trainer = Trainer(args_cli.actor_type)
        trainer.train()
    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()
