# GMTP

GMTP is a policy-training and evaluation layer built on top of Ref2Act.

## CLI

```bash
gmtp train
gmtp eval isaac --checkpoint path/to/model_v2.pth
gmtp eval sim2sim --checkpoint path/to/model_v2.pth
gmtp eval sim2sim --checkpoint path/to/model_v2.pth --motion-files env/assests/115_06_stageii.npz
gmtp eval sim2sim --checkpoint path/to/model_v2.pth --save-video
```
