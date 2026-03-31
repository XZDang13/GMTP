from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunConfig:
    actor_type: str = "vanila"
    adain_res_blocks: int = 3
    rollout_steps: int = 20
    num_updates: int = 2000
    checkpoint_interval: int = 4000
    output_root: str = "runs"
    run_name: str | None = None
    use_wandb: bool = True


@dataclass(frozen=True)
class IsaacEvalConfig:
    checkpoint_path: str
    actor_type: str | None = None
    adain_res_blocks: int | None = None
    num_steps: int = 1000
    progress_interval: int = 50
    show_reference_motion: bool = False
    output_root: str = "runs"


@dataclass(frozen=True)
class Sim2SimEvalConfig:
    checkpoint_path: str
    actor_type: str | None = None
    motion_files: list[str] | None = None
    adain_res_blocks: int | None = None
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
