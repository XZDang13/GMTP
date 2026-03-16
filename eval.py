import argparse

from isaaclab.app import AppLauncher

import gymnasium
import torch

from RLAlg.nn.steps import StochasticContinuousPolicyStep

from model.actor import AdaINActor, AdaINResActor, SplitEncoderActor, VanilaActor


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
    parser.add_argument("--checkpoint", default="final.pth")
    parser.add_argument(
        "--actor-type",
        default=None,
        help="Override actor architecture for checkpoints without actor metadata: vanila, split_encoder, adain, or adain_res.",
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

    def __init__(self, checkpoint_path: str, actor_type: str | None = None):
        from env.cfg import G1JabEnv

        self.cfg = G1JabEnv()
        self.checkpoint_path = checkpoint_path

        self.env_name = "G1MotionTracking-v0"
        
        self.cfg.scene.num_envs = 1
        self.cfg.training = False
        self.cfg.add_action_noise = False
        self.cfg.add_obs_noise = False
        self.cfg.add_reset_noise = False
        self.cfg.random_start = False

        self.env = gymnasium.make(self.env_name, cfg=self.cfg)

        self.device = self.env.unwrapped.device

        action_dim = self.cfg.action_space
        print("Load")
        weights = torch.load(self.checkpoint_path, map_location=self.device)
        self.actor_type = self._normalize_actor_type(actor_type or weights.get("actor_type"))
        self.actor = self._build_actor(self.cfg, self.actor_type, action_dim).to(self.device)
        actor_weights = weights["actor"]

        self.actor.load_state_dict(actor_weights)
        self.actor.eval()

        print("Start")
        #self.tracker = MetricsTracker()
        #self.tracker.add_batch_metrics("episode_return", self.cfg.scene.num_envs)
        #self.tracker.add_batch_metrics("episode_length", self.cfg.scene.num_envs)

        #WandbLogger.init_project("Mimic_Eval", f"G1_Pick")

    @torch.no_grad()
    def get_action(self, obs_batch: dict[str, torch.Tensor], determine: bool = False):
        actor_step:StochasticContinuousPolicyStep = self.actor(obs_batch)
        action = actor_step.action
        if determine:
            action = actor_step.mean
        
        return action

    
    def rollout(self, obs, info):
        for i in range(1000):
            actor_obs = self._get_actor_observation(obs, self.actor_type)
            action = self.get_action(actor_obs, True)
            #print("Action:")
            #print(action)
            #print(action[0])
            next_obs, task_reward, terminate, timeout, info = self.env.step(action)
            reward = task_reward
            #step_info = {}
            #for key, value in info.items():
            #    step_info[f"step/{key}"] = value

            #WandbLogger.log_metrics(step_info, i)

            #self.tracker.add_values("episode_return", reward)
            #self.tracker.add_values("episode_length", 1)

            done = terminate | timeout
            
            #if done.any():
                #log_ep_ret = self.tracker.get_mean("episode_return", done)
                #log_ep_len = self.tracker.get_mean("episode_length", done)

                #episode_info = {}
                #episode_info['episode/mean_returns'] = log_ep_ret
                #episode_info['episode/mean_length'] = log_ep_len

                #self.tracker.reset("episode_return", done)
                #self.tracker.reset("episode_length", done)

                #WandbLogger.log_metrics(episode_info, i)

            obs = next_obs

        return obs, info

    def eval(self):
        obs, info = self.env.reset()
        obs, info = self.rollout(obs, info)

        self.env.close()

if __name__ == "__main__":
    args_cli = build_arg_parser().parse_args()
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    try:
        evaluator = Evaluator(args_cli.checkpoint, args_cli.actor_type)
        evaluator.eval()
    finally:
        simulation_app.close()
