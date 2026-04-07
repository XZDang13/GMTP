from __future__ import annotations

import argparse

from gmtp.runtime.config import IsaacEvalConfig, RunConfig, Sim2SimEvalConfig


def _add_num_blocks_argument(parser: argparse.ArgumentParser, *, default) -> None:
    parser.add_argument("--num-blocks", dest="num_blocks", type=int, default=default)
    parser.add_argument("--film-res-blocks", dest="num_blocks", type=int, help=argparse.SUPPRESS)


def _add_robot_window_length_argument(parser: argparse.ArgumentParser, *, default) -> None:
    parser.add_argument("--robot-window-length", type=int, default=default)


def _add_robot_encoder_type_argument(parser: argparse.ArgumentParser, *, default) -> None:
    parser.add_argument(
        "--robot-encoder-type",
        choices=("cnn", "transformer"),
        default=default,
        help="Windowed robot-history encoder. robot-window-length=1 always uses the flat MLP path.",
    )


def _add_disable_amp_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--disable-amp", action="store_true", help="Disable CUDA automatic mixed precision.")


def _add_motion_mae_encoder_checkpoint_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--motion-mae-encoder-checkpoint", default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gmtp", description="GMTP training and evaluation CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a policy in Isaac Lab.")
    _add_num_blocks_argument(train_parser, default=4)
    _add_robot_window_length_argument(train_parser, default=4)
    _add_robot_encoder_type_argument(train_parser, default="transformer")
    _add_motion_mae_encoder_checkpoint_argument(train_parser)
    train_parser.add_argument("--rollout-steps", type=int, default=20)
    train_parser.add_argument("--num-updates", type=int, default=1000)
    train_parser.add_argument("--checkpoint-interval", type=int, default=4000)
    train_parser.add_argument("--output-root", default="runs")
    train_parser.add_argument("--run-name", default=None)
    train_parser.add_argument("--disable-wandb", action="store_true")
    _add_disable_amp_argument(train_parser)
    train_parser.add_argument("--headless", action="store_true")

    pretrain_parser = subparsers.add_parser("pretrain", help="Offline pretraining utilities.")
    pretrain_subparsers = pretrain_parser.add_subparsers(dest="pretrain_target", required=True)

    motion_mae_parser = pretrain_subparsers.add_parser("motion-mae", help="Pretrain the reference motion Motion MAE.")
    motion_mae_parser.add_argument("--config", required=True)
    motion_mae_parser.add_argument("--motion-files", nargs="+", default=None)
    motion_mae_parser.add_argument("--output-root", default=None)
    motion_mae_parser.add_argument("--run-name", default=None)
    motion_mae_parser.add_argument("--device", default=None)

    motion_mae_latents_parser = pretrain_subparsers.add_parser(
        "motion-mae-latents",
        help="Export deterministic Motion MAE latents for configured legal windows.",
    )
    motion_mae_latents_parser.add_argument("--checkpoint", required=True)
    motion_mae_latents_parser.add_argument("--config", required=True)
    motion_mae_latents_parser.add_argument("--motion-files", nargs="+", default=None)
    motion_mae_latents_parser.add_argument("--output-root", default=None)
    motion_mae_latents_parser.add_argument("--run-name", default=None)
    motion_mae_latents_parser.add_argument("--device", default=None)

    eval_parser = subparsers.add_parser("eval", help="Evaluate a policy.")
    eval_subparsers = eval_parser.add_subparsers(dest="eval_target", required=True)

    isaac_parser = eval_subparsers.add_parser("isaac", help="Evaluate a checkpoint in Isaac Lab.")
    isaac_parser.add_argument("--checkpoint", required=True)
    _add_num_blocks_argument(isaac_parser, default=None)
    _add_robot_window_length_argument(isaac_parser, default=None)
    _add_motion_mae_encoder_checkpoint_argument(isaac_parser)
    isaac_parser.add_argument("--num-steps", type=int, default=1000)
    isaac_parser.add_argument("--progress-interval", type=int, default=50)
    isaac_parser.add_argument("--show-reference-motion", action="store_true")
    isaac_parser.add_argument("--output-root", default="runs")
    _add_disable_amp_argument(isaac_parser)
    isaac_parser.add_argument("--headless", action="store_true")

    sim2sim_parser = eval_subparsers.add_parser("sim2sim", help="Evaluate a checkpoint in MuJoCo.")
    sim2sim_parser.add_argument("--checkpoint", required=True)
    sim2sim_parser.add_argument("--motion-files", nargs="+", default=None)
    _add_num_blocks_argument(sim2sim_parser, default=None)
    _add_robot_window_length_argument(sim2sim_parser, default=None)
    _add_motion_mae_encoder_checkpoint_argument(sim2sim_parser)
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
    _add_disable_amp_argument(sim2sim_parser)

    return parser


def _run_train(args) -> int:
    from isaaclab.app import AppLauncher

    from gmtp.runtime.train_runner import TrainRunner

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        TrainRunner(
            RunConfig(
                num_blocks=args.num_blocks,
                robot_window_length=args.robot_window_length,
                robot_encoder_type=args.robot_encoder_type,
                motion_mae_encoder_checkpoint=args.motion_mae_encoder_checkpoint,
                use_amp=not args.disable_amp,
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
                num_blocks=args.num_blocks,
                robot_window_length=args.robot_window_length,
                motion_mae_encoder_checkpoint=args.motion_mae_encoder_checkpoint,
                use_amp=not args.disable_amp,
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
            num_blocks=args.num_blocks,
            robot_window_length=args.robot_window_length,
            motion_mae_encoder_checkpoint=args.motion_mae_encoder_checkpoint,
            use_amp=not args.disable_amp,
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


def _run_pretrain_motion_mae(args) -> int:
    from gmtp.motion_mae import apply_motion_mae_cli_overrides, load_motion_mae_pretrain_config
    from gmtp.runtime.motion_mae_pretrain import MotionMAEPretrainRunner

    config = load_motion_mae_pretrain_config(args.config)
    config = apply_motion_mae_cli_overrides(
        config,
        motion_files=args.motion_files,
        output_root=args.output_root,
        run_name=args.run_name,
        device=args.device,
    )
    MotionMAEPretrainRunner(config).train()
    return 0


def _run_pretrain_motion_mae_latents(args) -> int:
    from gmtp.motion_mae import apply_motion_mae_cli_overrides, load_motion_mae_pretrain_config
    from gmtp.runtime.motion_mae_export import MotionMAELatentExportRunner

    config = load_motion_mae_pretrain_config(args.config)
    config = apply_motion_mae_cli_overrides(
        config,
        motion_files=args.motion_files,
        output_root=args.output_root,
        run_name=args.run_name,
        device=args.device,
    )
    MotionMAELatentExportRunner(checkpoint_path=args.checkpoint, config=config).export()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        return _run_train(args)
    if args.command == "pretrain" and args.pretrain_target == "motion-mae":
        return _run_pretrain_motion_mae(args)
    if args.command == "pretrain" and args.pretrain_target == "motion-mae-latents":
        return _run_pretrain_motion_mae_latents(args)
    if args.command == "eval" and args.eval_target == "isaac":
        return _run_eval_isaac(args)
    if args.command == "eval" and args.eval_target == "sim2sim":
        return _run_eval_sim2sim(args)

    raise ValueError(f"Unsupported command selection: {args}")


if __name__ == "__main__":
    raise SystemExit(main())
