# Actor/Critic 设计建议

这份文档记录当前 GMTP actor/critic 设计的后续改进建议，重点覆盖 critic 结构、actor 融合 ablation、actor/critic 优化器解耦，以及训练诊断指标。

## 1. 强化 Critic 结构

当前 critic 是 asymmetric privileged critic，这个方向适合当前任务。Actor 只依赖可部署的 policy observation，critic 在训练时使用 `privilege` observation。

主要问题是当前 critic 对完整 privileged vector 使用 flat MLP。对于 random-start、anchor-based、failure-weighted 的 motion tracking 任务，value function 需要区分多类状态信息：

- reference 或 target motion state
- robot proprioception
- relative anchor 和 key body geometry
- velocity 与 previous action context

建议改动：

- 保留 asymmetric actor-critic 设定。
- 将 flat critic encoder 改成 structured critic。
- 按语义拆分 privileged observation，分别编码后 concat，再接 value head。

一个保守的第一版可以包含：

- target/reference encoder
- robot proprioception encoder
- relative anchor/key body encoder
- final MLP value head

这样不会改变可部署 actor 的输入契约，但有机会提升 value learning 的稳定性和样本效率。

## 2. 做一个小规模 Actor Fusion Ablation

当前 actor 以 robot state 为主控制流，以 reference motion 作为 FiLM conditioning signal。这个设计适合 motion-conditioned control。

当前流程：

```text
robot_obs -> robot_encoder -> x_robot
motion_obs -> motion_encoder -> x_motion
x_robot + FiLM(x_motion) -> residual stack -> Gaussian action head
```

这个结构紧凑，并且让 reference motion 作为控制条件影响 robot latent。不过对于 phase 或 timing 要求强的动作，FiLM-only conditioning 可能没有充分使用 motion latent。

建议做一个轻量 ablation：

- 保留当前 FiLM residual stack 作为 baseline。
- 在 FiLM stack 后增加一个轻量 motion skip path。

可选方案：

```text
x = film_stack(x_robot, x_motion)
x = x + motion_proj(x_motion)
action = head(x)
```

or:

```text
x = film_stack(x_robot, x_motion)
x = fusion_mlp(concat(x, x_motion))
action = head(x)
```

不建议一开始就上大规模 cross-attention actor。先验证一个小的显式 motion path 是否能改善 tracking。

## 3. 解耦 Actor 和 Critic 优化

当前训练循环通过一个 optimizer collection 同时优化 actor 和 critic。KL-adaptive learning-rate scheduler 由 policy KL 驱动，但实际会影响整体 optimizer。

这会把两个不同的学习问题耦合在一起：

- actor update stability 应该由 policy KL 控制
- critic fitting 应该由 value loss 和 return target 控制

建议改动：

- 使用单独的 actor optimizer。
- 使用单独的 critic optimizer。
- KL-adaptive scheduling 只作用在 actor optimizer 上。
- Critic learning rate 第一版保持固定，后续再考虑单独 scheduler。

这样训练行为更容易解释：

- high KL 只降低 actor step size，不会拖慢 critic fitting。
- critic instability 可以单独处理。
- value loss 趋势不会和 policy update 过度纠缠。

## 4. 在大架构改动前补训练诊断

在做更大的模型改动前，先补充诊断指标，用来判断瓶颈到底是 actor expressiveness、critic quality，还是 sampling distribution。

建议新增指标：

- value explained variance
- PPO clip fraction
- action standard deviation 或 log standard deviation
- advantage mean 和 standard deviation
- 按 motion 拆分的 return 和 fall metrics
- 环境已经暴露的 recovery 或 tracking-quality metrics
- anchor reset probabilities 同步记录到 wandb，而不是只打印到控制台

这些指标可以帮助回答：

- Critic 是否学到了有用的 value function？
- Actor update 是否频繁被 PPO clipping 限制？
- Exploration 是否过早 collapse？
- Failure 是否集中在特定 motion 或 anchor？
- Failure-weighted sampling 是否过度聚焦在数据集的一小部分？

建议实现顺序：

1. 增加 value explained variance 和 advantage statistics。
2. 增加 policy clip fraction 和 action std logging。
3. 如果环境暴露 motion ID，增加 per-motion tracking/failure summaries。
4. 将 anchor reset probabilities 同步到 wandb。

## 建议优先级

1. 先补 diagnostics，因为它能降低后续所有实验的不确定性。
2. 再解耦 actor 和 critic optimizers，这个风险较低，并且能提高训练可解释性。
3. 如果 value explained variance 或 value loss 显示 critic 较弱，再引入 structured critic。
4. 在带有更完整 diagnostics 的 baseline 稳定后，再做 actor fusion ablation。
