from __future__ import annotations

import argparse

from gmtp.runtime.config import IsaacEvalConfig, MotionMAEVisualizationConfig, RunConfig, Sim2SimEvalConfig


def _add_num_blocks_argument(parser: argparse.ArgumentParser, *, default) -> None:
    parser.add_argument("--num-blocks", dest="num_blocks", type=int, default=default)
    parser.add_argument("--film-res-blocks", dest="num_blocks", type=int, help=argparse.SUPPRESS)


def _add_robot_window_length_argument(parser: argparse.ArgumentParser, *, default) -> None:
    parser.add_argument("--robot-window-length", type=int, default=default)


def _add_motion_window_length_argument(parser: argparse.ArgumentParser, *, default) -> None:
    parser.add_argument("--motion-window-length", type=int, default=default)


def _add_robot_encoder_type_argument(parser: argparse.ArgumentParser, *, default) -> None:
    parser.add_argument(
        "--robot-encoder-type",
        choices=("transformer",),
        default=default,
        help="Windowed robot-history encoder. robot-window-length=1 always uses the flat MLP path.",
    )


def _add_motion_encoder_type_argument(parser: argparse.ArgumentParser, *, default) -> None:
    parser.add_argument(
        "--motion-encoder-type",
        choices=("transformer", "mae"),
        default=default,
        help="Windowed motion-history encoder. motion-window-length=1 always uses the flat MLP path.",
    )


def _add_actor_fusion_type_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--actor-fusion-type",
        choices=("film", "motion_residual", "concat_mlp"),
        default="film",
        help="Actor motion-fusion ablation. Default keeps the baseline FiLM-only actor.",
    )


def _add_encoder_pooling_type_argument(parser: argparse.ArgumentParser, *, default) -> None:
    parser.add_argument(
        "--encoder-pooling-type",
        choices=("learned", "last_token"),
        default=default,
        help="Pooling used by windowed robot and motion encoders.",
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
    _add_motion_window_length_argument(train_parser, default=1)
    _add_motion_encoder_type_argument(train_parser, default="transformer")
    _add_actor_fusion_type_argument(train_parser)
    _add_encoder_pooling_type_argument(train_parser, default="learned")
    _add_motion_mae_encoder_checkpoint_argument(train_parser)
    train_parser.add_argument(
        "--motion-files",
        nargs="+",
        default=None,
        help="Override training motions. Accepts files, directories, or CMU/OMOMO dataset aliases.",
    )
    train_parser.add_argument(
        "--resume-checkpoint",
        dest="resume_checkpoint_path",
        default=None,
        help="Load a CheckpointV2 policy checkpoint and continue training in a new run directory.",
    )
    train_parser.add_argument("--rollout-steps", type=int, default=20)
    train_parser.add_argument("--num-updates", type=int, default=1000)
    train_parser.add_argument("--checkpoint-interval", type=int, default=4000)
    train_parser.add_argument("--output-root", default="runs")
    train_parser.add_argument("--run-name", default=None)
    train_parser.add_argument("--disable-wandb", action="store_true")
    train_parser.add_argument(
        "--enable-end-effector-termination-curriculum",
        dest="end_effector_termination_curriculum_enabled",
        action="store_true",
        default=False,
        help="Enable the performance-gated end-effector termination curriculum.",
    )
    train_parser.add_argument(
        "--disable-end-effector-termination-curriculum",
        dest="end_effector_termination_curriculum_enabled",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    train_parser.add_argument("--end-effector-termination-start-threshold", type=float, default=0.25)
    train_parser.add_argument("--end-effector-termination-end-threshold", type=float, default=0.15)
    train_parser.add_argument("--end-effector-termination-tighten-step", type=float, default=0.03)
    train_parser.add_argument(
        "--end-effector-termination-warmup-fraction",
        "--end-effector-termination-start-fraction",
        dest="end_effector_termination_warmup_fraction",
        type=float,
        default=0.10,
    )
    train_parser.add_argument(
        "--end-effector-termination-deadline-fraction",
        "--end-effector-termination-end-fraction",
        dest="end_effector_termination_deadline_fraction",
        type=float,
        default=0.50,
    )
    train_parser.add_argument("--end-effector-termination-ema-updates", type=int, default=20)
    train_parser.add_argument("--end-effector-termination-min-ema-samples", type=int, default=10)
    train_parser.add_argument("--end-effector-termination-hold-updates", type=int, default=20)
    train_parser.add_argument("--end-effector-termination-max-terminate-rate", type=float, default=0.03)
    train_parser.add_argument("--end-effector-termination-error-margin", type=float, default=1.10)
    train_parser.add_argument(
        "--sampler-failure-warmup-fraction",
        type=float,
        default=0.0,
        help="Use uniform motion/anchor sampling for this fraction of training before failure weighting turns on.",
    )
    train_parser.add_argument("--anchor-log-interval", type=int, default=100)
    train_parser.add_argument("--anchor-heatmap-bins", type=int, default=128)
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

    motion_mae_visualize_parser = pretrain_subparsers.add_parser(
        "motion-mae-visualize",
        help="Render a side-by-side GT vs predicted future-motion comparison video for a Motion MAE checkpoint.",
    )
    motion_mae_visualize_parser.add_argument("--checkpoint", required=True)
    motion_mae_visualize_parser.add_argument("--config", required=True)
    motion_mae_visualize_parser.add_argument("--motion-files", nargs="+", default=None)
    motion_mae_visualize_parser.add_argument("--split", choices=("train", "val"), default="val")
    motion_mae_visualize_parser.add_argument("--motion-name", default=None)
    motion_mae_visualize_parser.add_argument("--sample-index", type=int, default=0)
    motion_mae_visualize_parser.add_argument("--whole-motion", action="store_true")
    motion_mae_visualize_parser.add_argument("--future-frame-index", type=int, default=None)
    motion_mae_visualize_parser.add_argument("--fps", type=int, default=None)
    motion_mae_visualize_parser.add_argument("--output-root", default=None)
    motion_mae_visualize_parser.add_argument("--run-name", default=None)
    motion_mae_visualize_parser.add_argument("--device", default=None)

    eval_parser = subparsers.add_parser("eval", help="Evaluate a policy.")
    eval_subparsers = eval_parser.add_subparsers(dest="eval_target", required=True)

    isaac_parser = eval_subparsers.add_parser("isaac", help="Evaluate a checkpoint in Isaac Lab.")
    isaac_parser.add_argument("--checkpoint", required=True)
    isaac_parser.add_argument(
        "--motion-files",
        nargs="+",
        default=None,
        help="Override evaluation motions. Accepts files, directories, or CMU/OMOMO dataset aliases.",
    )
    isaac_parser.add_argument(
        "--end-effector-termination-threshold",
        type=float,
        default=None,
        help=(
            "Override Isaac eval end-effector termination threshold. "
            "By default, checkpoint curriculum state is used when available."
        ),
    )
    _add_num_blocks_argument(isaac_parser, default=None)
    _add_robot_window_length_argument(isaac_parser, default=None)
    _add_motion_window_length_argument(isaac_parser, default=None)
    _add_motion_encoder_type_argument(isaac_parser, default=None)
    _add_encoder_pooling_type_argument(isaac_parser, default=None)
    _add_motion_mae_encoder_checkpoint_argument(isaac_parser)
    isaac_parser.add_argument("--num-steps", type=int, default=1000)
    isaac_parser.add_argument("--progress-interval", type=int, default=50)
    isaac_parser.add_argument("--show-reference-motion", action="store_true")
    isaac_parser.add_argument("--save-video", action="store_true")
    isaac_parser.add_argument("--video-fps", type=int, default=None)
    isaac_parser.add_argument("--output-root", default="runs")
    _add_disable_amp_argument(isaac_parser)
    isaac_parser.add_argument("--headless", action="store_true")

    sim2sim_parser = eval_subparsers.add_parser("sim2sim", help="Evaluate a checkpoint in MuJoCo.")
    sim2sim_parser.add_argument("--checkpoint", required=True)
    sim2sim_parser.add_argument("--motion-files", nargs="+", default=None)
    _add_num_blocks_argument(sim2sim_parser, default=None)
    _add_robot_window_length_argument(sim2sim_parser, default=None)
    _add_motion_window_length_argument(sim2sim_parser, default=None)
    _add_motion_encoder_type_argument(sim2sim_parser, default=None)
    _add_encoder_pooling_type_argument(sim2sim_parser, default=None)
    _add_motion_mae_encoder_checkpoint_argument(sim2sim_parser)
    sim2sim_parser.add_argument("--num-steps", type=int, default=2000)
    sim2sim_parser.add_argument("--simulation-dt", type=float, default=1 / 200)
    sim2sim_parser.add_argument("--decimation", type=int, default=4)
    sim2sim_parser.add_argument("--action-mode", default=None)
    sim2sim_parser.add_argument("--root-name", default=None)
    sim2sim_parser.add_argument("--anchor-body-name", default=None)
    sim2sim_parser.add_argument(
        "--allow-unstable-init",
        action="store_true",
        help="Sample a large random unstable reset in MuJoCo around the reference state instead of the default stabilized lift.",
    )
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
                motion_window_length=args.motion_window_length,
                motion_encoder_type=args.motion_encoder_type,
                actor_fusion_type=args.actor_fusion_type,
                encoder_pooling_type=args.encoder_pooling_type,
                motion_mae_encoder_checkpoint=args.motion_mae_encoder_checkpoint,
                motion_files=args.motion_files,
                resume_checkpoint_path=args.resume_checkpoint_path,
                use_amp=not args.disable_amp,
                end_effector_termination_curriculum_enabled=args.end_effector_termination_curriculum_enabled,
                end_effector_termination_start_threshold=args.end_effector_termination_start_threshold,
                end_effector_termination_end_threshold=args.end_effector_termination_end_threshold,
                end_effector_termination_tighten_step=args.end_effector_termination_tighten_step,
                end_effector_termination_warmup_fraction=args.end_effector_termination_warmup_fraction,
                end_effector_termination_deadline_fraction=args.end_effector_termination_deadline_fraction,
                end_effector_termination_ema_updates=args.end_effector_termination_ema_updates,
                end_effector_termination_min_ema_samples=args.end_effector_termination_min_ema_samples,
                end_effector_termination_hold_updates=args.end_effector_termination_hold_updates,
                end_effector_termination_max_terminate_rate=args.end_effector_termination_max_terminate_rate,
                end_effector_termination_error_margin=args.end_effector_termination_error_margin,
                sampler_failure_warmup_fraction=args.sampler_failure_warmup_fraction,
                rollout_steps=args.rollout_steps,
                num_updates=args.num_updates,
                checkpoint_interval=args.checkpoint_interval,
                output_root=args.output_root,
                run_name=args.run_name,
                use_wandb=not args.disable_wandb,
                anchor_log_interval=args.anchor_log_interval,
                anchor_heatmap_bins=args.anchor_heatmap_bins,
            )
        ).train()
    finally:
        simulation_app.close()
    return 0


def _run_eval_isaac(args) -> int:
    from isaaclab.app import AppLauncher

    from gmtp.runtime.eval_isaac import IsaacEvalRunner

    app_launcher = AppLauncher(args, enable_cameras=args.save_video)
    simulation_app = app_launcher.app
    try:
        IsaacEvalRunner(
            IsaacEvalConfig(
                checkpoint_path=args.checkpoint,
                motion_files=args.motion_files,
                end_effector_termination_threshold=args.end_effector_termination_threshold,
                num_blocks=args.num_blocks,
                robot_window_length=args.robot_window_length,
                motion_window_length=args.motion_window_length,
                motion_encoder_type=args.motion_encoder_type,
                encoder_pooling_type=args.encoder_pooling_type,
                motion_mae_encoder_checkpoint=args.motion_mae_encoder_checkpoint,
                use_amp=not args.disable_amp,
                num_steps=args.num_steps,
                progress_interval=args.progress_interval,
                show_reference_motion=args.show_reference_motion,
                save_video=args.save_video,
                video_fps=args.video_fps,
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
            motion_window_length=args.motion_window_length,
            motion_encoder_type=args.motion_encoder_type,
            encoder_pooling_type=args.encoder_pooling_type,
            motion_mae_encoder_checkpoint=args.motion_mae_encoder_checkpoint,
            use_amp=not args.disable_amp,
            num_steps=args.num_steps,
            simulation_dt=args.simulation_dt,
            decimation=args.decimation,
            action_mode=args.action_mode,
            root_name=args.root_name,
            anchor_body_name=args.anchor_body_name,
            allow_unstable_init=args.allow_unstable_init,
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


def _run_pretrain_motion_mae_visualize(args) -> int:
    from gmtp.runtime.motion_mae_visualize import MotionMAEVisualizerRunner

    MotionMAEVisualizerRunner(
        MotionMAEVisualizationConfig(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            motion_files=args.motion_files,
            split=args.split,
            motion_name=args.motion_name,
            sample_index=args.sample_index,
            whole_motion=args.whole_motion,
            future_frame_index=args.future_frame_index,
            fps=args.fps,
            output_root=args.output_root,
            run_name=args.run_name,
            device=args.device,
        )
    ).visualize()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        return _run_train(args)
    if args.command == "pretrain" and args.pretrain_target == "motion-mae":
        return _run_pretrain_motion_mae(args)
    if args.command == "pretrain" and args.pretrain_target == "motion-mae-visualize":
        return _run_pretrain_motion_mae_visualize(args)
    if args.command == "eval" and args.eval_target == "isaac":
        return _run_eval_isaac(args)
    if args.command == "eval" and args.eval_target == "sim2sim":
        return _run_eval_sim2sim(args)

    raise ValueError(f"Unsupported command selection: {args}")


if __name__ == "__main__":
    raise SystemExit(main())
