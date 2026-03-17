import argparse
import re

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
    def _normalize_adain_res_blocks(num_blocks: int) -> int:
        if num_blocks < 1:
            raise ValueError(f"adain_res_blocks must be positive, got {num_blocks}.")
        return num_blocks

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

    def __init__(
        self,
        checkpoint_path: str,
        actor_type: str | None = None,
        adain_res_blocks: int | None = None,
    ):
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
        self.cfg.events = None

        self.env = gymnasium.make(self.env_name, cfg=self.cfg)

        self.device = self.env.unwrapped.device
        self.initial_obs, _ = self.env.reset()
        self.obs_dims = self._infer_observation_dims(self.initial_obs)

        action_dim = self.cfg.action_space
        print("Load")
        weights = torch.load(self.checkpoint_path, map_location=self.device)
        self.actor_type = self._normalize_actor_type(actor_type or weights.get("actor_type"))
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

        self.actor = self._build_actor(
            self.obs_dims,
            self.actor_type,
            action_dim,
            actor_block_count,
        ).to(self.device)

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
        obs = self.initial_obs
        info = {}
        obs, info = self.rollout(obs, info)

        self.env.close()

if __name__ == "__main__":
    args_cli = build_arg_parser().parse_args()
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    try:
        evaluator = Evaluator(args_cli.checkpoint, args_cli.actor_type, 10)
        evaluator.eval()
    finally:
        simulation_app.close()
