import argparse

from isaaclab.app import AppLauncher

import gymnasium
import torch
import numpy as np

from RLAlg.nn.steps import StochasticContinuousPolicyStep

from model.actor import Actor


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
    parser.add_argument("--checkpoint", default="final.pth")
    AppLauncher.add_app_launcher_args(parser)
    return parser

class Evaluator:
    @staticmethod
    def _get_policy_observation(obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat((obs["motion"], obs["robot"]), dim=-1)

    @staticmethod
    def _get_critic_observation(obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return obs["privilege"]

    def __init__(self, checkpoint_path: str):
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

        policy_obs_dim = self.cfg.policy_observation_space
        action_dim = self.cfg.action_space
        print("Load")
        self.actor = Actor(policy_obs_dim, action_dim).to(self.device)

        weights = torch.load(self.checkpoint_path, map_location=self.device)
        actor_weights = weights["actor"]

        self.actor.load_state_dict(actor_weights)
        self.actor.eval()

        print("Start")
        #self.tracker = MetricsTracker()
        #self.tracker.add_batch_metrics("episode_return", self.cfg.scene.num_envs)
        #self.tracker.add_batch_metrics("episode_length", self.cfg.scene.num_envs)

        #WandbLogger.init_project("Mimic_Eval", f"G1_Pick")

    @torch.no_grad()
    def get_action(self, obs_batch:torch.Tensor, determine:bool=False):
        actor_step:StochasticContinuousPolicyStep = self.actor(obs_batch)
        action = actor_step.action
        if determine:
            action = actor_step.mean
        
        return action

    
    def rollout(self, obs, info):
        for i in range(1000):
            policy_obs = self._get_policy_observation(obs)
            action = self.get_action(policy_obs, True)
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
        evaluator = Evaluator(args_cli.checkpoint)
        evaluator.eval()
    finally:
        simulation_app.close()
