from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunConfig:
    num_blocks: int = 4
    robot_window_length: int = 4
    robot_encoder_type: str = "transformer"
    motion_mae_encoder_checkpoint: str | None = None
    use_amp: bool = True
    rollout_steps: int = 20
    num_updates: int = 1000
    checkpoint_interval: int = 4000
    output_root: str = "runs"
    run_name: str | None = None
    use_wandb: bool = True


@dataclass(frozen=True)
class IsaacEvalConfig:
    checkpoint_path: str
    num_blocks: int | None = None
    robot_window_length: int | None = None
    motion_mae_encoder_checkpoint: str | None = None
    use_amp: bool = True
    num_steps: int = 1000
    progress_interval: int = 50
    show_reference_motion: bool = False
    output_root: str = "runs"


@dataclass(frozen=True)
class Sim2SimEvalConfig:
    checkpoint_path: str
    motion_files: list[str] | None = None
    num_blocks: int | None = None
    robot_window_length: int | None = None
    motion_mae_encoder_checkpoint: str | None = None
    use_amp: bool = True
    num_steps: int = 2000
    simulation_dt: float = 1 / 200
    decimation: int = 4
    action_mode: str | None = None
    root_name: str | None = None
    anchor_body_name: str | None = None
    render: bool = False
    save_video: bool = False
    video_fps: int | None = None
    output_root: str = "runs"
