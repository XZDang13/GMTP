from __future__ import annotations

import argparse

from gmtp.runtime.config import IsaacEvalConfig, RunConfig, Sim2SimEvalConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gmtp", description="GMTP training and evaluation CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a policy in Isaac Lab.")
    train_parser.add_argument("--actor-type", default="vanila")
    train_parser.add_argument("--adain-res-blocks", type=int, default=3)
    train_parser.add_argument("--rollout-steps", type=int, default=20)
    train_parser.add_argument("--num-updates", type=int, default=1000)
    train_parser.add_argument("--checkpoint-interval", type=int, default=4000)
    train_parser.add_argument("--output-root", default="runs")
    train_parser.add_argument("--run-name", default=None)
    train_parser.add_argument("--disable-wandb", action="store_true")
    train_parser.add_argument("--headless", action="store_true")

    eval_parser = subparsers.add_parser("eval", help="Evaluate a policy.")
    eval_subparsers = eval_parser.add_subparsers(dest="eval_target", required=True)

    isaac_parser = eval_subparsers.add_parser("isaac", help="Evaluate a checkpoint in Isaac Lab.")
    isaac_parser.add_argument("--checkpoint", required=True)
    isaac_parser.add_argument("--actor-type", default=None)
    isaac_parser.add_argument("--adain-res-blocks", type=int, default=None)
    isaac_parser.add_argument("--num-steps", type=int, default=1000)
    isaac_parser.add_argument("--progress-interval", type=int, default=50)
    isaac_parser.add_argument("--show-reference-motion", action="store_true")
    isaac_parser.add_argument("--output-root", default="runs")
    isaac_parser.add_argument("--headless", action="store_true")

    sim2sim_parser = eval_subparsers.add_parser("sim2sim", help="Evaluate a checkpoint in MuJoCo.")
    sim2sim_parser.add_argument("--checkpoint", required=True)
    sim2sim_parser.add_argument("--motion-files", nargs="+", default=None)
    sim2sim_parser.add_argument("--actor-type", default=None)
    sim2sim_parser.add_argument("--adain-res-blocks", type=int, default=None)
    sim2sim_parser.add_argument("--num-steps", type=int, default=2000)
    sim2sim_parser.add_argument("--simulation-dt", type=float, default=1 / 200)
    sim2sim_parser.add_argument("--decimation", type=int, default=4)
    sim2sim_parser.add_argument("--action-mode", default=None)
    sim2sim_parser.add_argument("--root-name", default=None)
    sim2sim_parser.add_argument("--anchor-body-name", default=None)
    sim2sim_parser.add_argument("--render", action="store_true")
    sim2sim_parser.add_argument("--save-video", action="store_true")
    sim2sim_parser.add_argument("--video-fps", type=int, default=None)
    sim2sim_parser.add_argument("--output-root", default="runs")

    return parser


def _run_train(args) -> int:
    from isaaclab.app import AppLauncher

    from gmtp.runtime.train_runner import TrainRunner

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        TrainRunner(
            RunConfig(
                actor_type=args.actor_type,
                adain_res_blocks=args.adain_res_blocks,
                rollout_steps=args.rollout_steps,
                num_updates=args.num_updates,
                checkpoint_interval=args.checkpoint_interval,
                output_root=args.output_root,
                run_name=args.run_name,
                use_wandb=not args.disable_wandb,
            )
        ).train()
    finally:
        simulation_app.close()
    return 0


def _run_eval_isaac(args) -> int:
    from isaaclab.app import AppLauncher

    from gmtp.runtime.eval_isaac import IsaacEvalRunner

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        IsaacEvalRunner(
            IsaacEvalConfig(
                checkpoint_path=args.checkpoint,
                actor_type=args.actor_type,
                adain_res_blocks=args.adain_res_blocks,
                num_steps=args.num_steps,
                progress_interval=args.progress_interval,
                show_reference_motion=args.show_reference_motion,
                output_root=args.output_root,
            )
        ).evaluate()
    finally:
        simulation_app.close()
    return 0


def _run_eval_sim2sim(args) -> int:
    from gmtp.runtime.eval_sim2sim import Sim2SimEvalRunner

    Sim2SimEvalRunner(
        Sim2SimEvalConfig(
            checkpoint_path=args.checkpoint,
            motion_files=args.motion_files,
            actor_type=args.actor_type,
            adain_res_blocks=args.adain_res_blocks,
            num_steps=args.num_steps,
            simulation_dt=args.simulation_dt,
            decimation=args.decimation,
            action_mode=args.action_mode,
            root_name=args.root_name,
            anchor_body_name=args.anchor_body_name,
            render=args.render,
            save_video=args.save_video,
            video_fps=args.video_fps,
            output_root=args.output_root,
        )
    ).evaluate()
    return 0





def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        return _run_train(args)
    if args.command == "eval" and args.eval_target == "isaac":
        return _run_eval_isaac(args)
    if args.command == "eval" and args.eval_target == "sim2sim":
        return _run_eval_sim2sim(args)

    raise ValueError(f"Unsupported command selection: {args}")
