# GMTP

GMTP is a policy-training and evaluation layer built on top of Ref2Act.

## CLI

Train with a Motion MAE motion encoder and a transformer robot encoder:

```bash
gmtp train \
  --robot-window-length 4 \
  --robot-encoder-type transformer \
  --motion-window-length 4 \
  --motion-encoder-type mae \
  --motion-mae-encoder-checkpoint \
  weights/mae/20260408_150430_motion_mae_policy_obs_compact_g1_23dof/checkpoints/best_motion_mae_encoder.pth
```

```bash
gmtp train
gmtp eval isaac --checkpoint path/to/model_v2.pth
gmtp eval isaac --checkpoint path/to/model_v2.pth --save-video
gmtp eval sim2sim --checkpoint path/to/model_v2.pth
gmtp eval sim2sim --checkpoint path/to/model_v2.pth --motion-files env/assests/115_06_stageii.npz
gmtp eval sim2sim --checkpoint path/to/model_v2.pth --allow-unstable-init
gmtp eval sim2sim --checkpoint path/to/model_v2.pth --save-video
```

`--allow-unstable-init` requests a large random unstable reset around the reference state in the MuJoCo bridge instead of the default stabilized `+0.05` root-height lift.
