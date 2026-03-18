import argparse
import re
from collections import defaultdict
from pathlib import Path

import torch

from RLAlg.nn.steps import StochasticContinuousPolicyStep
from Ref2Act.sim2sim import MujocoEnv

from env.motions import DEFAULT_EXPERIMENT_MOTION_FILES, motion_label, resolve_motion_files
from model.actor import AdaINActor, AdaINResActor, SplitEncoderActor, VanilaActor


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MOTION_FILES = resolve_motion_files(DEFAULT_EXPERIMENT_MOTION_FILES)


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
        nargs="+",
        default=None,
        help="Reference motion .npz file(s) used by sim2sim. Defaults to checkpoint metadata or the walk/runing/jump experiment set.",
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
    def _infer_motion_files(
        cls,
        checkpoint_path: Path,
        actor_type: str,
        checkpoint_weights: dict,
        motion_files: list[str] | None,
    ) -> list[str]:
        if motion_files is not None:
            return resolve_motion_files(motion_files)

        checkpoint_motion_files = checkpoint_weights.get("motion_files")
        if checkpoint_motion_files:
            try:
                return resolve_motion_files(checkpoint_motion_files)
            except FileNotFoundError:
                pass

        checkpoint_motion_names = checkpoint_weights.get("motion_names")
        if checkpoint_motion_names:
            return resolve_motion_files(checkpoint_motion_names)

        marker = f"_{actor_type}_"
        checkpoint_stem = checkpoint_path.stem
        if marker in checkpoint_stem:
            motion_name = checkpoint_stem.rsplit(marker, 1)[0]
            candidate = PROJECT_ROOT / "env" / "assests" / f"{motion_name}.npz"
            if candidate.exists():
                return [str(candidate.resolve())]

        return list(DEFAULT_MOTION_FILES)

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
        motion_files: list[str] | None = None,
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

        self.motion_files = self._infer_motion_files(
            self.checkpoint_path,
            self.actor_type,
            weights,
            motion_files,
        )
        self.motion_name = motion_label(self.motion_files)
        self.simulation_dt = simulation_dt
        self.decimation = decimation
        self.root_name = root_name
        self.kp = weights["joint_stiffness"].detach().cpu()
        self.kd = weights["joint_damping"].detach().cpu()
        self.effort_limits = weights["joint_effort_limits"].detach().cpu()
        self.joint_pos_limits = weights["joint_pos_limits"].detach().cpu()
        self.action_offset = weights["action_offset"].detach().cpu()
        self.action_scale = weights["action_scale"].detach().cpu()

    def _build_env(self, motion_file: str) -> MujocoEnv:
        return MujocoEnv(
            simulation_dt=self.simulation_dt,
            decimation=self.decimation,
            kp=self.kp,
            kd=self.kd,
            effort_limits=self.effort_limits,
            joint_pos_limits=self.joint_pos_limits,
            action_offset=self.action_offset,
            action_scale=self.action_scale,
            expert_motion_file=motion_file,
            root_name=self.root_name,
            render=self.render,
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

    def _eval_motion_file(self, motion_file: str) -> tuple[int, dict[str, float]]:
        env = self._build_env(motion_file)
        obs = env.reset()
        metrics = defaultdict(float)
        steps_run = 0

        try:
            for _ in range(self.num_steps):
                actor_obs = self._get_actor_observation(obs)
                action = self.get_action(actor_obs, determine=True).squeeze(0).detach().cpu()
                obs = env.step(action)

                for key, value in self._extract_metrics(obs, self.action_dim).items():
                    metrics[key] += value

                steps_run += 1

                if self.render and env.mj_viewer is None:
                    break
        finally:
            env.close()

        if steps_run == 0:
            return 0, {}

        return steps_run, {key: value / steps_run for key, value in metrics.items()}

    def eval(self) -> None:
        aggregate_metrics = defaultdict(float)
        aggregate_steps = 0

        print(f"checkpoint: {self.checkpoint_path}")
        print(f"motion_label: {self.motion_name}")
        print(f"actor_type: {self.actor_type}")

        for motion_file in self.motion_files:
            steps_run, metrics = self._eval_motion_file(motion_file)

            print(f"motion_file: {motion_file}")
            print(f"steps: {steps_run}")

            for key, value in metrics.items():
                print(f"{key}: {value:.6f}")
                aggregate_metrics[key] += value * steps_run

            aggregate_steps += steps_run

        if aggregate_steps == 0:
            return

        if len(self.motion_files) > 1:
            print(f"aggregate_steps: {aggregate_steps}")
            for key, total_value in aggregate_metrics.items():
                print(f"aggregate_{key}: {total_value / aggregate_steps:.6f}")


def main():
    args = build_arg_parser().parse_args()
    evaluator = Sim2SimEvaluator(
        checkpoint_path=args.checkpoint,
        actor_type=args.actor_type,
        motion_files=args.motion_file,
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
