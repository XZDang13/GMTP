import argparse
import re
from collections import defaultdict
from pathlib import Path

import torch

from RLAlg.nn.steps import StochasticContinuousPolicyStep
from Ref2Act.sim2sim import MujocoEnv

from model.actor import AdaINActor, AdaINResActor, SplitEncoderActor, VanilaActor


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MOTION_FILE = PROJECT_ROOT / "env" / "assests" / "pick_up.npz"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a trained policy in Ref2Act's MuJoCo sim2sim environment.")
    parser.add_argument("--checkpoint", default="final.pth")
    parser.add_argument(
        "--actor-type",
        default=None,
        help="Override actor architecture for checkpoints without actor metadata: vanila, split_encoder, adain, or adain_res.",
    )
    parser.add_argument(
        "--motion-file",
        default=None,
        help="Reference motion .npz used by sim2sim. Defaults to inferring from checkpoint name.",
    )
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--simulation-dt", type=float, default=1 / 200)
    parser.add_argument("--decimation", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--root-name", default="pelvis")
    parser.add_argument("--render", action="store_true")
    return parser


class Sim2SimEvaluator:
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
    def _resolve_existing_path(path_str: str) -> Path:
        path = Path(path_str).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        return path

    @classmethod
    def _infer_motion_file(cls, checkpoint_path: Path, actor_type: str, motion_file: str | None) -> Path:
        if motion_file is not None:
            return cls._resolve_existing_path(motion_file)

        marker = f"_{actor_type}_"
        checkpoint_stem = checkpoint_path.stem
        if marker in checkpoint_stem:
            motion_name = checkpoint_stem.rsplit(marker, 1)[0]
            candidate = PROJECT_ROOT / "env" / "assests" / f"{motion_name}.npz"
            if candidate.exists():
                return candidate

        if not DEFAULT_MOTION_FILE.exists():
            raise FileNotFoundError(
                "Failed to infer the reference motion file from the checkpoint name, and "
                f"the fallback file is missing: {DEFAULT_MOTION_FILE}"
            )
        return DEFAULT_MOTION_FILE

    @staticmethod
    def _infer_observation_dims(actor_weights: dict[str, torch.Tensor], actor_type: str) -> dict[str, int]:
        if actor_type == "vanila":
            policy_dim = actor_weights["normlizer.mean"].shape[0]
            return {"policy": policy_dim}

        motion_dim = actor_weights["motion_obs_normlizer.mean"].shape[0]
        robot_dim = actor_weights["robot_obs_normlizer.mean"].shape[0]
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
    def _split_sim2sim_obs(flat_obs: torch.Tensor, action_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        expected_dim = action_dim * 5 + 9
        if flat_obs.ndim != 1:
            raise ValueError(f"Expected a flat sim2sim observation, got shape {tuple(flat_obs.shape)}.")
        if flat_obs.numel() != expected_dim:
            raise ValueError(f"Expected sim2sim observation dim {expected_dim}, got {flat_obs.numel()}.")

        offset = 0

        target_joint_pos = flat_obs[offset : offset + action_dim]
        offset += action_dim

        target_joint_vel = flat_obs[offset : offset + action_dim]
        offset += action_dim

        target_projected_gravity = flat_obs[offset : offset + 3]
        offset += 3

        robot_obs = flat_obs[offset:]
        motion_obs = torch.cat((target_projected_gravity, target_joint_pos, target_joint_vel), dim=-1)

        return motion_obs, robot_obs

    @staticmethod
    def _extract_metrics(flat_obs: torch.Tensor, action_dim: int) -> dict[str, float]:
        offset = 0

        target_joint_pos = flat_obs[offset : offset + action_dim]
        offset += action_dim

        target_joint_vel = flat_obs[offset : offset + action_dim]
        offset += action_dim

        target_projected_gravity = flat_obs[offset : offset + 3]
        offset += 3

        robot_projected_gravity = flat_obs[offset : offset + 3]
        offset += 3

        offset += 3

        robot_joint_pos = flat_obs[offset : offset + action_dim]
        offset += action_dim

        robot_joint_vel = flat_obs[offset : offset + action_dim]

        return {
            "joint_pos_mae": torch.mean(torch.abs(target_joint_pos - robot_joint_pos)).item(),
            "joint_vel_mae": torch.mean(torch.abs(target_joint_vel - robot_joint_vel)).item(),
            "gravity_mae": torch.mean(torch.abs(target_projected_gravity - robot_projected_gravity)).item(),
        }

    def __init__(
        self,
        checkpoint_path: str,
        actor_type: str | None = None,
        motion_file: str | None = None,
        num_steps: int = 1000,
        simulation_dt: float = 1 / 200,
        decimation: int = 4,
        device: str = "cpu",
        root_name: str = "pelvis",
        render: bool = False,
        adain_res_blocks: int | None = None,
    ):
        if num_steps < 1:
            raise ValueError(f"num_steps must be positive, got {num_steps}.")

        self.checkpoint_path = self._resolve_existing_path(checkpoint_path)
        self.device = torch.device(device)
        self.num_steps = num_steps
        self.render = render

        weights = torch.load(self.checkpoint_path, map_location="cpu")
        self.actor_type = self._normalize_actor_type(actor_type or weights.get("actor_type"))
        actor_weights = weights["actor"]
        actor_kwargs = dict(weights.get("actor_kwargs", {}))

        self.action_dim = int(weights["action_scale"].numel())
        self.obs_dims = self._infer_observation_dims(actor_weights, self.actor_type)

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
            self.action_dim,
            actor_block_count,
        ).to(self.device)
        self.actor.load_state_dict(actor_weights)
        self.actor.eval()

        self.motion_file = self._infer_motion_file(self.checkpoint_path, self.actor_type, motion_file)

        self.env = MujocoEnv(
            simulation_dt=simulation_dt,
            decimation=decimation,
            kp=weights["joint_stiffness"].detach().cpu(),
            kd=weights["joint_damping"].detach().cpu(),
            effort_limits=weights["joint_effort_limits"].detach().cpu(),
            joint_pos_limits=weights["joint_pos_limits"].detach().cpu(),
            action_offset=weights["action_offset"].detach().cpu(),
            action_scale=weights["action_scale"].detach().cpu(),
            expert_motion_file=str(self.motion_file),
            root_name=root_name,
            render=render,
        )

    def _get_actor_observation(self, flat_obs: torch.Tensor) -> dict[str, torch.Tensor]:
        motion_obs, robot_obs = self._split_sim2sim_obs(flat_obs, self.action_dim)

        if "motion" in self.obs_dims and motion_obs.numel() != self.obs_dims["motion"]:
            raise ValueError(f"Expected motion observation dim {self.obs_dims['motion']}, got {motion_obs.numel()}.")
        if "robot" in self.obs_dims and robot_obs.numel() != self.obs_dims["robot"]:
            raise ValueError(f"Expected robot observation dim {self.obs_dims['robot']}, got {robot_obs.numel()}.")

        if self.actor_type == "vanila":
            policy_obs = torch.cat((motion_obs, robot_obs), dim=-1)
            return {"obs": policy_obs.unsqueeze(0).to(self.device)}

        return {
            "motion_obs": motion_obs.unsqueeze(0).to(self.device),
            "robot_obs": robot_obs.unsqueeze(0).to(self.device),
        }

    @torch.no_grad()
    def get_action(self, obs_batch: dict[str, torch.Tensor], determine: bool = True) -> torch.Tensor:
        actor_step: StochasticContinuousPolicyStep = self.actor(obs_batch)
        if determine:
            return actor_step.mean
        return actor_step.action

    def eval(self) -> None:
        obs = self.env.reset()
        metrics = defaultdict(float)
        steps_run = 0

        try:
            for _ in range(self.num_steps):
                actor_obs = self._get_actor_observation(obs)
                action = self.get_action(actor_obs, determine=True).squeeze(0).detach().cpu()
                obs = self.env.step(action)

                for key, value in self._extract_metrics(obs, self.action_dim).items():
                    metrics[key] += value

                steps_run += 1

                if self.render and self.env.mj_viewer is None:
                    break
        finally:
            self.env.close()

        print(f"checkpoint: {self.checkpoint_path}")
        print(f"motion_file: {self.motion_file}")
        print(f"actor_type: {self.actor_type}")
        print(f"steps: {steps_run}")

        if steps_run == 0:
            return

        for key, value in metrics.items():
            print(f"{key}: {value / steps_run:.6f}")


def main():
    args = build_arg_parser().parse_args()
    evaluator = Sim2SimEvaluator(
        checkpoint_path=args.checkpoint,
        actor_type=args.actor_type,
        motion_file=args.motion_file,
        num_steps=args.num_steps,
        simulation_dt=args.simulation_dt,
        decimation=args.decimation,
        device=args.device,
        root_name=args.root_name,
        render=args.render,
    )
    evaluator.eval()


if __name__ == "__main__":
    main()
