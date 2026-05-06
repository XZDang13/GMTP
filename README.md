# GMTP

GMTP is a policy-training and evaluation layer built on top of Ref2Act. It provides a `gmtp` CLI for Isaac Lab policy training, Isaac evaluation, MuJoCo sim-to-sim evaluation, and Motion MAE pretraining utilities.

## Setup

Use an environment with Isaac Lab, PyTorch, and Ref2Act-compatible assets available when running Isaac training or Isaac evaluation. In this workspace, the `isaaclab` conda environment is the expected runtime environment.

```bash
conda activate isaaclab
pip install -e .
```

For MuJoCo sim-to-sim evaluation, install the optional sim2sim dependencies if they are not already present:

```bash
pip install -e ".[sim2sim]"
```

The default training motion set resolves through Ref2Act to:

```text
~/Desktop/mocap_data/CMU
~/Desktop/mocap_data/OMOMO
```

Local sample motion files live under `env/assests/`. The directory name is intentionally spelled `assests` in this repository.

## Quick Start

Train a default policy:

```bash
gmtp train --headless
```

Train on one local motion file:

```bash
gmtp train \
  --headless \
  --motion-files env/assests/jump_anchor.npz \
  --num-updates 1000 \
  --checkpoint-interval 100
```

Continue training from a policy checkpoint:

```bash
gmtp train \
  --headless \
  --resume-checkpoint runs/train/<run>/checkpoints/<checkpoint>.pth \
  --num-updates 500
```

Evaluate in Isaac:

```bash
gmtp eval isaac \
  --headless \
  --checkpoint runs/train/<run>/checkpoints/<checkpoint>.pth
```

Evaluate in MuJoCo:

```bash
gmtp eval sim2sim \
  --checkpoint runs/train/<run>/checkpoints/<checkpoint>.pth \
  --motion-files env/assests/walk_anchor.npz \
  --save-video
```

## Training

The base training command is:

```bash
gmtp train [options]
```

Common training options:

| Option | Default | Use |
| --- | --- | --- |
| `--motion-files ...` | Ref2Act default motion set | Override training motions. Accepts files, directories, or dataset aliases understood by Ref2Act. |
| `--num-updates N` | `1000` | Number of PPO updates to run. On resume, this means additional updates. |
| `--rollout-steps N` | `20` | Environment steps collected per update. |
| `--checkpoint-interval N` | `4000` | Save periodic checkpoints every N absolute updates. |
| `--output-root PATH` | `runs` | Root directory for generated runs. |
| `--run-name NAME` | auto-generated | Human-readable run suffix. |
| `--disable-wandb` | off | Disable W&B logging. |
| `--disable-amp` | off | Disable CUDA automatic mixed precision. |
| `--disable-quality-gate` | off | Disable the Ref2Act terminal tracking quality gate during training. |
| `--anchor-log-interval N` | `100` | Log anchor reset sampler diagnostics every N updates. |
| `--anchor-heatmap-bins N` | `128` | Number of time bins for anchor reset heatmap artifacts. |
| `--headless` | off | Passed to Isaac Lab AppLauncher for non-GUI runs. |

Training uses failure-weighted anchor resets from the first update. Every `--anchor-log-interval` updates, GMTP logs sampler coverage, concentration, failure statistics, and writes anchor reset debug artifacts.

Training outputs are written to:

```text
runs/train/<timestamp>_<run-name>/
  config.json
  summary.json
  checkpoints/
  debug/
  videos/
```

The final policy checkpoint path is recorded in `summary.json` as `final_checkpoint`.

## Actor And Motion Encoders

GMTP currently trains the `film_res` actor. The actor can use flat or windowed robot and motion observations.

Train with a transformer robot history encoder:

```bash
gmtp train \
  --headless \
  --robot-window-length 4 \
  --robot-encoder-type transformer
```

Train with an integrated Motion MAE motion encoder:

```bash
gmtp train \
  --headless \
  --robot-window-length 4 \
  --robot-encoder-type transformer \
  --motion-window-length 4 \
  --motion-encoder-type mae \
  --motion-mae-encoder-checkpoint \
  weights/mae/20260429_154039_motion_mae_actor_motion_obs_cmu_omomo_w10_f10/checkpoints/best_motion_mae_encoder.pth
```

Actor options:

| Option | Values | Notes |
| --- | --- | --- |
| `--num-blocks N` | positive integer | FiLM residual block count. |
| `--robot-window-length N` | positive integer | `1` uses the flat MLP path; `>1` uses `--robot-encoder-type`. |
| `--robot-encoder-type` | `cnn`, `transformer` | Robot history encoder for windowed robot observations. |
| `--motion-window-length N` | positive integer | `1` uses the flat MLP path; `>1` uses `--motion-encoder-type`. |
| `--motion-encoder-type` | `transformer`, `mae` | Motion history encoder for windowed motion observations. |
| `--actor-fusion-type` | `film`, `motion_residual`, `concat_mlp` | Actor motion-fusion ablation. |
| `--motion-mae-encoder-checkpoint PATH` | path | Required when `--motion-encoder-type mae` and `--motion-window-length > 1`. |

## Resuming Training

Resume uses:

```bash
gmtp train \
  --headless \
  --resume-checkpoint path/to/model_v2.pth \
  --num-updates 500
```

Resume behavior:

- GMTP creates a new timestamped run directory and leaves the source run untouched.
- `--num-updates` means additional updates. A checkpoint saved at update `1000` with `--num-updates 500` finishes at update `1500`.
- Checkpoint actor settings are authoritative. Architecture-changing flags such as `--num-blocks`, `--robot-window-length`, and `--motion-window-length` are ignored during resume because they would invalidate checkpoint weights.
- If `--motion-files` is not provided, training uses the motion files recorded in the checkpoint.
- If `--motion-files` is provided, the resumed run uses those motions for fine-tuning.
- If `--motion-mae-encoder-checkpoint` is not provided, GMTP uses the Motion MAE encoder path recorded in the checkpoint when present.

New checkpoints contain trainer state: actor and critic weights, actor and critic optimizer state, KL scheduler state, AMP scaler state, update counters, global step counters, AMP metadata, and RNG state. These checkpoints restore exact trainer state.

Older `CheckpointV2` files that do not contain trainer state still load. They warm-start actor and critic weights, then continue with fresh optimizer, scheduler, scaler, and counters.

## Isaac Evaluation

Run a trained policy inside Isaac Lab:

```bash
gmtp eval isaac \
  --headless \
  --checkpoint path/to/model_v2.pth
```

Save an Isaac evaluation video:

```bash
gmtp eval isaac \
  --headless \
  --checkpoint path/to/model_v2.pth \
  --save-video
```

Common Isaac eval options:

| Option | Default | Use |
| --- | --- | --- |
| `--checkpoint PATH` | required | Policy checkpoint to load. |
| `--num-steps N` | `1000` | Number of simulation steps. |
| `--progress-interval N` | `50` | Console progress logging interval. |
| `--show-reference-motion` | off | Show reference motion during evaluation. |
| `--save-video` | off | Save rendered video output. |
| `--video-fps N` | auto | Override saved video FPS. |
| `--output-root PATH` | `runs` | Output root for eval runs. |
| `--disable-amp` | off | Disable CUDA automatic mixed precision. |
| `--headless` | off | Passed to Isaac Lab AppLauncher. |

Evaluation restores actor configuration from checkpoint metadata. Use override flags such as `--num-blocks`, `--motion-window-length`, or `--motion-encoder-type` only for compatibility with intentionally overridden checkpoint specs.

## MuJoCo Sim2Sim Evaluation

Run a policy in MuJoCo:

```bash
gmtp eval sim2sim \
  --checkpoint path/to/model_v2.pth
```

Evaluate against a specific motion file:

```bash
gmtp eval sim2sim \
  --checkpoint path/to/model_v2.pth \
  --motion-files env/assests/85_09_stageii.npz
```

Render and save a video:

```bash
gmtp eval sim2sim \
  --checkpoint path/to/model_v2.pth \
  --motion-files env/assests/walk_anchor.npz \
  --render \
  --save-video
```

Common sim2sim options:

| Option | Default | Use |
| --- | --- | --- |
| `--checkpoint PATH` | required | Policy checkpoint to load. |
| `--motion-files ...` | checkpoint/default inference | Override evaluation motions. |
| `--num-steps N` | `2000` | Number of MuJoCo simulation steps. |
| `--simulation-dt X` | `0.005` | MuJoCo simulation timestep. |
| `--decimation N` | `4` | Policy control decimation. |
| `--action-mode MODE` | checkpoint metadata | Override action mode. |
| `--root-name NAME` | checkpoint metadata | Override robot root body name. |
| `--anchor-body-name NAME` | checkpoint metadata | Override reference anchor body name. |
| `--allow-unstable-init` | off | Use a large random unstable reset around the reference state. |
| `--render` | off | Open/render the MuJoCo viewer. |
| `--save-video` | off | Save video output. |
| `--video-fps N` | auto | Override saved video FPS. |
| `--output-root PATH` | `runs` | Output root for eval runs. |
| `--disable-amp` | off | Disable CUDA automatic mixed precision. |

By default, MuJoCo initialization uses a stabilized `+0.05` root-height lift. `--allow-unstable-init` requests a larger random unstable reset around the reference state and is useful for stress testing.

## Motion MAE Utilities

Motion MAE commands are under:

```bash
gmtp pretrain ...
```

Pretrain a Motion MAE from a JSON config:

```bash
gmtp pretrain motion-mae \
  --config configs/motion_mae_actor_motion_obs_cmu_omomo_w10_f10.json
```

Useful overrides:

```bash
gmtp pretrain motion-mae \
  --config configs/motion_mae_actor_motion_obs_cmu_omomo_w10_f10.json \
  --motion-files env/assests/jump_anchor.npz env/assests/walk_anchor.npz \
  --output-root weights \
  --run-name debug_motion_mae \
  --device cuda:0
```

Pretraining outputs are written to:

```text
<output-root>/pretrain-motion-mae/<timestamp>_<run-name>/
  config.json
  summary.json
  checkpoints/
    best_motion_mae.pth
    best_motion_mae_encoder.pth
    final_motion_mae.pth
    final_motion_mae_encoder.pth
```

Export Motion MAE latents:

```bash
gmtp pretrain motion-mae-latents \
  --checkpoint weights/mae/20260429_154039_motion_mae_actor_motion_obs_cmu_omomo_w10_f10/checkpoints/best_motion_mae_encoder.pth \
  --config configs/motion_mae_actor_motion_obs_cmu_omomo_w10_f10.json
```

Visualize a full Motion MAE checkpoint:

```bash
gmtp pretrain motion-mae-visualize \
  --checkpoint path/to/best_motion_mae.pth \
  --config configs/motion_mae_actor_motion_obs_cmu_omomo_w10_f10.json \
  --split val \
  --sample-index 0
```

Visualization options:

| Option | Use |
| --- | --- |
| `--motion-files ...` | Override config motion files. |
| `--split train|val` | Choose source split. |
| `--motion-name NAME` | Select a specific motion by name. |
| `--sample-index N` | Select a window index. |
| `--whole-motion` | Render an entire motion instead of one sampled window. |
| `--future-frame-index N` | Render a single predicted future frame. |
| `--fps N` | Override video FPS. |
| `--output-root PATH` | Override output root. |
| `--run-name NAME` | Override run name. |
| `--device DEVICE` | Override device. |

## Checkpoints

Policy checkpoints use `CheckpointV2` and contain:

- `meta`: actor type, actor kwargs, creation time, and motion label.
- `model`: actor and critic state dicts.
- `env`: motion files, joint parameters, action metadata, and observation window lengths.
- `artifacts`: run directory and optional Motion MAE encoder path.
- `training`: optional trainer state used for exact resume.

Evaluation only needs the actor weights and metadata. Training resume uses the `training` section when present.

## Troubleshooting

- If `gmtp train` fails to import Isaac Lab, activate the Isaac Lab environment before running the command.
- If Motion MAE actor training fails with a missing encoder checkpoint, pass `--motion-mae-encoder-checkpoint` and confirm the file exists.
- If sim2sim cannot render, install the optional sim2sim dependencies and confirm MuJoCo can open in the current display environment.
- If full test runs fail on `env/assests/115_02_stageii.npz`, that asset is not present in this checkout. The core trainer and checkpoint tests can still be run independently.
