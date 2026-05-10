import pytest
import torch
import torch.nn as nn

from gmtp.models import (
    Critic,
    FiLMResActor,
    build_actor,
    get_actor_kwargs,
    infer_actor_fusion_type,
    infer_film_res_blocks,
)
from gmtp.models.actor import ACTOR_HIDDEN_DIM
from gmtp.models.critic import CRITIC_GROUP_HIDDEN_DIM, CRITIC_HIDDEN_DIM, build_critic_privilege_layout
from gmtp.models.film import FiLMResStack
from gmtp.models.motion_encoder import (
    MOTION_ENCODER_HIDDEN_DIM,
    MOTION_ENCODER_OUTPUT_DIM,
    MotionWindowEncoder,
    MotionEncoderType,
    build_motion_window_layout,
)
from gmtp.models.robot_encoder import (
    ROBOT_ENCODER_HIDDEN_DIM,
    ROBOT_ENCODER_OUTPUT_DIM,
    RobotWindowEncoder,
    build_robot_window_layout,
)
from gmtp.motion_mae import (
    FeatureSliceSpec,
    MotionFeatureSchema,
    MotionMAEDataConfig,
    MotionMAEFeatureConfig,
    MotionMAEModelConfig,
    MotionMAEPretrainConfig,
    ReferenceMotionMAE,
    build_motion_mae_encoder_checkpoint,
    save_motion_mae_encoder_checkpoint,
)
from gmtp.models.pooling import LastTokenPool, LearnedQueryAttentionPool
from gmtp.runtime.checkpoints import CheckpointV2, build_training_checkpoint
from gmtp.runtime.policy import (
    load_actor_from_checkpoint,
    resolve_checkpoint_actor_spec,
    validate_checkpoint_actor_observation_dims,
)


def _joint_params(action_dim: int = 3) -> dict[str, torch.Tensor | list[str]]:
    return {
        "joint_names": [f"joint_{idx}" for idx in range(action_dim)],
        "joint_effort_limits": torch.ones(action_dim),
        "joint_pos_limits": torch.tensor([[-1.0, 1.0]] * action_dim),
        "joint_stiffness": torch.ones(action_dim),
        "joint_damping": torch.full((action_dim,), 0.1),
        "action_offset": torch.zeros(action_dim),
        "action_scale": torch.ones(action_dim),
    }


def _actor_obs_dims(
    action_dim: int,
    *,
    robot_window_length: int = 1,
    motion_window_length: int = 1,
) -> tuple[int, int]:
    motion_dim = build_motion_window_layout(action_dim, motion_window_length).motion_obs_dim
    robot_dim = build_robot_window_layout(action_dim, robot_window_length).robot_obs_dim
    return motion_dim, robot_dim


def test_last_token_pool_returns_final_token_and_validates_shape():
    pool = LastTokenPool(embed_dim=3)
    tokens = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)

    torch.testing.assert_close(pool(tokens), tokens[:, -1, :])

    with pytest.raises(ValueError, match="shape"):
        pool(torch.randn(2, 3))
    with pytest.raises(ValueError, match="token dim"):
        pool(torch.randn(2, 4, 2))
    with pytest.raises(ValueError, match="sequence length"):
        pool(torch.empty(2, 0, 3))


def _motion_mae_schema(
    action_dim: int = 2,
    *,
    reference_feature_names: tuple[str, ...] = ("root", "joint"),
    target_feature_names: tuple[str, ...] | None = None,
    policy_feature_names: tuple[str, ...] = ("root", "joint"),
) -> MotionFeatureSchema:
    base_dims = {
        "root": 3,
        "joint": action_dim,
        "end_effector": 6,
    }
    base_slices = []
    offset = 0
    for name in ("root", "joint", "end_effector"):
        next_offset = offset + base_dims[name]
        base_slices.append(FeatureSliceSpec(name, offset, next_offset))
        offset = next_offset
    base_slice_map = {item.name: item for item in base_slices}

    target_feature_names = target_feature_names or reference_feature_names

    def _named_slices(names: tuple[str, ...]) -> tuple[FeatureSliceSpec, ...]:
        running_offset = 0
        slices = []
        for name in names:
            base_slice = base_slice_map[name]
            next_offset = running_offset + base_slice.dim
            slices.append(FeatureSliceSpec(name, running_offset, next_offset))
            running_offset = next_offset
        return tuple(slices)

    target_slices = _named_slices(target_feature_names)
    policy_motion_dim = sum(next(item.dim for item in target_slices if item.name == name) for name in policy_feature_names)

    return MotionFeatureSchema(
        d_ref=sum(base_slice_map[name].dim for name in reference_feature_names),
        d_target=sum(base_slice_map[name].dim for name in target_feature_names),
        full_feature_dim=sum(item.dim for item in base_slices),
        base_slices=tuple(base_slices),
        reference_slices=_named_slices(reference_feature_names),
        target_slices=target_slices,
        policy_motion_slice=FeatureSliceSpec("policy_motion", 0, policy_motion_dim),
        anchor_body_name="pelvis",
        end_effector_body_names=("left_hand", "right_hand"),
        reference_feature_names=reference_feature_names,
        target_feature_names=target_feature_names,
        policy_feature_names=policy_feature_names,
        joint_names=tuple(f"j{idx}" for idx in range(action_dim)),
        body_names=("pelvis", "left_hand", "right_hand"),
        reference_mean=tuple(0.0 for _ in range(sum(base_slice_map[name].dim for name in reference_feature_names))),
        reference_std=tuple(1.0 for _ in range(sum(base_slice_map[name].dim for name in reference_feature_names))),
        target_mean=tuple(0.0 for _ in range(sum(base_slice_map[name].dim for name in target_feature_names))),
        target_std=tuple(1.0 for _ in range(sum(base_slice_map[name].dim for name in target_feature_names))),
    )


def _write_motion_mae_encoder_checkpoint(
    tmp_path,
    *,
    action_dim: int = 2,
    past_frames: int = 4,
    reference_feature_names: tuple[str, ...] = ("root", "joint"),
    target_feature_names: tuple[str, ...] | None = None,
    policy_feature_names: tuple[str, ...] = ("root", "joint"),
) -> str:
    schema = _motion_mae_schema(
        action_dim=action_dim,
        reference_feature_names=reference_feature_names,
        target_feature_names=target_feature_names,
        policy_feature_names=policy_feature_names,
    )
    model = ReferenceMotionMAE(
        input_dim=schema.d_ref,
        target_dim=schema.d_target,
        past_frames=past_frames,
        future_frames=2,
        latent_dim=6,
        d_model=16,
        encoder_layers=2,
        decoder_layers=1,
        nhead=4,
        dim_feedforward=32,
    )
    checkpoint = build_motion_mae_encoder_checkpoint(
        model=model,
        schema=schema,
        config=MotionMAEPretrainConfig(
            data=MotionMAEDataConfig(
                motion_files=("env/assests/115_02_stageii.npz",),
                past_frames=past_frames,
                future_frames=2,
                split_mode="by_window",
                val_ratio=0.5,
            ),
            feature=MotionMAEFeatureConfig(
                reference_feature_names=reference_feature_names,
                target_feature_names=target_feature_names or reference_feature_names,
                policy_feature_names=policy_feature_names,
                end_effector_body_names=("left_hand", "right_hand"),
            ),
            model=MotionMAEModelConfig(
                d_model=16,
                latent_dim=6,
                encoder_layers=2,
                decoder_layers=1,
                nhead=4,
                dim_feedforward=32,
            ),
        ),
        epoch=1,
        best_metric=0.7,
    )
    return str(save_motion_mae_encoder_checkpoint(checkpoint, tmp_path / "motion_mae_encoder.pth"))


class _RecordingPool(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tokens: torch.Tensor | None = None

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        self.tokens = tokens.detach().clone()
        return torch.zeros(tokens.shape[0], tokens.shape[-1], dtype=tokens.dtype, device=tokens.device)


def test_film_res_actor_forward_returns_step():
    motion_obs_dim, robot_obs_dim = _actor_obs_dims(action_dim=3)
    actor = FiLMResActor(
        robot_obs_dim=robot_obs_dim,
        motion_obs_dim=motion_obs_dim,
        action_dim=3,
        num_blocks=5,
    )

    step = actor(
        {
            "robot_obs": torch.randn(4, robot_obs_dim),
            "motion_obs": torch.randn(4, motion_obs_dim),
        }
    )

    assert step.action.shape == (4, 3)
    assert step.log_prob.shape == (4,)
    assert actor.num_blocks == 5
    assert len(actor.blocks) == 5


def test_film_res_actor_defaults_to_four_blocks_and_512_width():
    motion_obs_dim, robot_obs_dim = _actor_obs_dims(action_dim=3)
    actor = FiLMResActor(
        robot_obs_dim=robot_obs_dim,
        motion_obs_dim=motion_obs_dim,
        action_dim=3,
    )

    assert actor.num_blocks == 4
    assert actor.actor_fusion_type == "film"
    assert not hasattr(actor, "motion_skip")
    assert not hasattr(actor, "fusion_mlp")
    assert len(actor.blocks) == 4
    assert actor.motion_encoder.motion_encoder_type == "mlp"
    assert actor.motion_encoder.single_frame_encoder[0].linear.out_features == MOTION_ENCODER_OUTPUT_DIM
    assert actor.motion_encoder.single_frame_encoder[-1].linear.out_features == MOTION_ENCODER_OUTPUT_DIM
    assert actor.head.mu_layer.out_features == 3
    assert actor.head.mu_layer.in_features == ACTOR_HIDDEN_DIM
    assert actor.stack.blocks[0].fc1.in_features == ACTOR_HIDDEN_DIM
    assert actor.stack.blocks[0].fc2.out_features == ACTOR_HIDDEN_DIM


def test_build_actor_constructs_film_res_with_requested_depth():
    motion_obs_dim, robot_obs_dim = _actor_obs_dims(action_dim=3, robot_window_length=4, motion_window_length=4)
    actor = build_actor(
        {"robot": robot_obs_dim, "motion": motion_obs_dim, "policy": motion_obs_dim + robot_obs_dim},
        "film_res",
        action_dim=3,
        actor_kwargs={
            "num_blocks": 4,
            "robot_window_length": 4,
            "robot_encoder_type": "transformer",
            "motion_window_length": 4,
            "motion_encoder_type": "transformer",
            "actor_fusion_type": "concat_mlp",
        },
    )

    assert isinstance(actor, FiLMResActor)
    assert actor.num_blocks == 4
    assert actor.actor_fusion_type == "concat_mlp"
    assert len(actor.blocks) == 4
    assert get_actor_kwargs(actor, "film_res") == {
        "num_blocks": 4,
        "robot_window_length": 4,
        "robot_encoder_type": "transformer",
        "motion_window_length": 4,
        "motion_encoder_type": "transformer",
        "actor_fusion_type": "concat_mlp",
        "encoder_pooling_type": "learned",
    }
    assert infer_film_res_blocks(actor.state_dict()) == 4
    assert infer_actor_fusion_type(actor.state_dict()) == "concat_mlp"


def test_film_res_actor_supports_motion_residual_fusion():
    motion_obs_dim, robot_obs_dim = _actor_obs_dims(action_dim=2)
    actor = FiLMResActor(
        robot_obs_dim=robot_obs_dim,
        motion_obs_dim=motion_obs_dim,
        action_dim=2,
        actor_fusion_type="motion_residual",
    )
    step = actor(
        {
            "robot_obs": torch.randn(3, robot_obs_dim),
            "motion_obs": torch.randn(3, motion_obs_dim),
        }
    )

    assert actor.actor_fusion_type == "motion_residual"
    assert actor.motion_skip.in_features == ACTOR_HIDDEN_DIM
    assert actor.motion_skip.out_features == ACTOR_HIDDEN_DIM
    assert actor.motion_skip_scale.shape == (ACTOR_HIDDEN_DIM,)
    assert infer_actor_fusion_type(actor.state_dict()) == "motion_residual"
    assert step.action.shape == (3, 2)
    assert step.log_prob.shape == (3,)


def test_film_res_actor_supports_concat_mlp_fusion():
    motion_obs_dim, robot_obs_dim = _actor_obs_dims(action_dim=2)
    actor = FiLMResActor(
        robot_obs_dim=robot_obs_dim,
        motion_obs_dim=motion_obs_dim,
        action_dim=2,
        actor_fusion_type="concat_mlp",
    )
    step = actor(
        {
            "robot_obs": torch.randn(3, robot_obs_dim),
            "motion_obs": torch.randn(3, motion_obs_dim),
        }
    )

    assert actor.actor_fusion_type == "concat_mlp"
    assert actor.fusion_mlp[0].linear.in_features == ACTOR_HIDDEN_DIM * 2
    assert actor.fusion_mlp[0].linear.out_features == ACTOR_HIDDEN_DIM
    assert actor.fusion_mlp[-1].linear.out_features == ACTOR_HIDDEN_DIM
    assert infer_actor_fusion_type(actor.state_dict()) == "concat_mlp"
    assert step.action.shape == (3, 2)
    assert step.log_prob.shape == (3,)


def test_film_res_actor_reshapes_windowed_robot_obs_from_ref2act_layout():
    motion_obs_dim, robot_obs_dim = _actor_obs_dims(action_dim=2, robot_window_length=4)
    actor = FiLMResActor(
        robot_obs_dim=robot_obs_dim,
        motion_obs_dim=motion_obs_dim,
        action_dim=2,
        robot_window_length=4,
    )
    robot_obs = torch.arange(robot_obs_dim, dtype=torch.float32).reshape(1, robot_obs_dim)

    reshaped = actor._reshape_robot_obs(robot_obs)

    assert reshaped.shape == (1, 4, 12)
    torch.testing.assert_close(
        reshaped,
        torch.tensor(
            [
                [
                    [0.0, 1.0, 2.0, 12.0, 13.0, 14.0, 24.0, 25.0, 32.0, 33.0, 40.0, 41.0],
                    [3.0, 4.0, 5.0, 15.0, 16.0, 17.0, 26.0, 27.0, 34.0, 35.0, 42.0, 43.0],
                    [6.0, 7.0, 8.0, 18.0, 19.0, 20.0, 28.0, 29.0, 36.0, 37.0, 44.0, 45.0],
                    [9.0, 10.0, 11.0, 21.0, 22.0, 23.0, 30.0, 31.0, 38.0, 39.0, 46.0, 47.0],
                ]
            ]
        ),
    )


def test_film_res_actor_windowed_robot_encoder_rejects_cnn_mode():
    motion_obs_dim, robot_obs_dim = _actor_obs_dims(action_dim=2, robot_window_length=4)

    with pytest.raises(ValueError, match="transformer"):
        FiLMResActor(
            robot_obs_dim=robot_obs_dim,
            motion_obs_dim=motion_obs_dim,
            action_dim=2,
            robot_window_length=4,
            robot_encoder_type="cnn",
        )


def test_film_res_actor_windowed_robot_encoder_supports_transformer_mode():
    motion_obs_dim, robot_obs_dim = _actor_obs_dims(action_dim=2, robot_window_length=4)
    actor = FiLMResActor(
        robot_obs_dim=robot_obs_dim,
        motion_obs_dim=motion_obs_dim,
        action_dim=2,
        robot_window_length=4,
        robot_encoder_type="transformer",
    )

    assert actor.robot_encoder.robot_encoder_type == "transformer"
    assert actor.encoder_pooling_type == "learned"
    assert isinstance(actor.robot_encoder.window_encoder.encoder.transformer, nn.TransformerEncoder)
    assert isinstance(actor.robot_encoder.window_encoder.encoder.input_proj, nn.Linear)
    assert isinstance(actor.robot_encoder.window_encoder.pooling, LearnedQueryAttentionPool)
    assert actor.robot_encoder.window_encoder.output_proj.out_features == ROBOT_ENCODER_OUTPUT_DIM
    assert not hasattr(actor.robot_encoder.window_encoder, "temporal_conv")


def test_film_res_actor_windowed_motion_encoder_supports_transformer_mode():
    motion_obs_dim, robot_obs_dim = _actor_obs_dims(action_dim=2, motion_window_length=4)
    actor = FiLMResActor(
        robot_obs_dim=robot_obs_dim,
        motion_obs_dim=motion_obs_dim,
        action_dim=2,
        motion_window_length=4,
        motion_encoder_type="transformer",
    )

    assert actor.motion_encoder.motion_encoder_type == "transformer"
    assert actor.encoder_pooling_type == "learned"
    assert isinstance(actor.motion_encoder.window_encoder.encoder.transformer, nn.TransformerEncoder)
    assert isinstance(actor.motion_encoder.window_encoder.encoder.input_proj, nn.Linear)
    assert isinstance(actor.motion_encoder.window_encoder.pooling, LearnedQueryAttentionPool)
    assert actor.motion_encoder.window_encoder.output_proj.out_features == MOTION_ENCODER_OUTPUT_DIM
    assert actor.motion_obs_normlizer.mean.shape == (4, actor.motion_step_dim)


def test_robot_window_encoder_pools_all_time_tokens():
    encoder = RobotWindowEncoder(robot_step_dim=5, robot_window_length=4)
    pool = _RecordingPool()
    encoder.pooling = pool

    output = encoder(torch.randn(2, 4, 5))

    assert output.shape == (2, ROBOT_ENCODER_OUTPUT_DIM)
    assert pool.tokens is not None
    assert pool.tokens.shape == (2, 4, ROBOT_ENCODER_HIDDEN_DIM)


def test_motion_transformer_window_encoder_pools_all_time_tokens():
    encoder = MotionWindowEncoder(
        motion_step_dim=5,
        motion_window_length=4,
        motion_encoder_type=MotionEncoderType.TRANSFORMER,
        device="cpu",
    )
    pool = _RecordingPool()
    encoder.pooling = pool

    output = encoder(torch.randn(2, 4, 5))

    assert output.shape == (2, MOTION_ENCODER_OUTPUT_DIM)
    assert pool.tokens is not None
    assert pool.tokens.shape == (2, 4, MOTION_ENCODER_HIDDEN_DIM)


def test_windowed_actor_supports_last_token_pooling_mode():
    motion_obs_dim, robot_obs_dim = _actor_obs_dims(
        action_dim=2,
        robot_window_length=4,
        motion_window_length=4,
    )
    actor = FiLMResActor(
        robot_obs_dim=robot_obs_dim,
        motion_obs_dim=motion_obs_dim,
        action_dim=2,
        robot_window_length=4,
        motion_window_length=4,
        encoder_pooling_type="last_token",
    )

    assert actor.encoder_pooling_type == "last_token"
    assert actor.robot_encoder.encoder_pooling_type == "last_token"
    assert actor.motion_encoder.encoder_pooling_type == "last_token"
    assert isinstance(actor.robot_encoder.window_encoder.pooling, LastTokenPool)
    assert isinstance(actor.motion_encoder.window_encoder.pooling, LastTokenPool)
    assert not any(".pooling." in key for key in actor.state_dict())
    assert get_actor_kwargs(actor, "film_res")["encoder_pooling_type"] == "last_token"


def test_window_encoder_rejects_invalid_pooling_mode():
    with pytest.raises(ValueError, match="encoder pooling type"):
        RobotWindowEncoder(robot_step_dim=5, robot_window_length=4, encoder_pooling_type="mean")


def test_film_res_actor_single_frame_motion_forces_mlp_mode():
    motion_obs_dim, robot_obs_dim = _actor_obs_dims(action_dim=2)
    actor = FiLMResActor(
        robot_obs_dim=robot_obs_dim,
        motion_obs_dim=motion_obs_dim,
        action_dim=2,
        motion_window_length=1,
        motion_encoder_type="mae",
    )

    assert actor.motion_encoder.motion_encoder_type == "mlp"
    assert actor.motion_encoder.single_frame_encoder[-1].linear.out_features == MOTION_ENCODER_OUTPUT_DIM
    assert actor.motion_obs_normlizer.mean.shape == (motion_obs_dim,)


def test_film_res_actor_forward_supports_windowed_robot_obs_with_transformer_encoder():
    motion_obs_dim, robot_obs_dim = _actor_obs_dims(action_dim=2, robot_window_length=4)
    actor = FiLMResActor(
        robot_obs_dim=robot_obs_dim,
        motion_obs_dim=motion_obs_dim,
        action_dim=2,
        num_blocks=3,
        robot_window_length=4,
        robot_encoder_type="transformer",
    )

    step = actor(
        {
            "robot_obs": torch.randn(3, 4, 12),
            "motion_obs": torch.randn(3, motion_obs_dim),
        }
    )

    assert step.action.shape == (3, 2)
    assert step.log_prob.shape == (3,)


def test_film_res_actor_forward_supports_windowed_motion_obs_with_transformer_encoder():
    motion_obs_dim, robot_obs_dim = _actor_obs_dims(action_dim=2, motion_window_length=4)
    actor = FiLMResActor(
        robot_obs_dim=robot_obs_dim,
        motion_obs_dim=motion_obs_dim,
        action_dim=2,
        num_blocks=3,
        motion_window_length=4,
        motion_encoder_type="transformer",
    )

    step = actor(
        {
            "robot_obs": torch.randn(3, robot_obs_dim),
            "motion_obs": torch.randn(3, 4, actor.motion_step_dim),
        }
    )

    assert step.action.shape == (3, 2)
    assert step.log_prob.shape == (3,)


def test_film_res_actor_forward_supports_windowed_motion_obs_with_mae_encoder(tmp_path):
    checkpoint_path = _write_motion_mae_encoder_checkpoint(tmp_path, action_dim=2, past_frames=4)
    motion_obs_dim, robot_obs_dim = _actor_obs_dims(action_dim=2, motion_window_length=4)
    actor = FiLMResActor(
        robot_obs_dim=robot_obs_dim,
        motion_obs_dim=motion_obs_dim,
        action_dim=2,
        num_blocks=3,
        motion_window_length=4,
        motion_encoder_type="mae",
        motion_mae_encoder_checkpoint=checkpoint_path,
    )

    step = actor(
        {
            "robot_obs": torch.randn(3, robot_obs_dim),
            "motion_obs": torch.randn(3, 4, actor.motion_step_dim),
        }
    )

    assert step.action.shape == (3, 2)
    assert step.log_prob.shape == (3,)
    assert actor.motion_encoder.motion_encoder_type == "mae"
    assert not any("motion_encoder.window_encoder._encoder" in key for key in actor.state_dict())
    assert isinstance(actor.motion_encoder.window_encoder.pooling, LearnedQueryAttentionPool)
    assert any("motion_encoder.window_encoder.pooling." in key for key in actor.state_dict())
    assert all(
        not parameter.requires_grad for parameter in actor.motion_encoder.window_encoder.encoder.parameters()
    )


def test_motion_mae_window_encoder_supports_last_token_pooling(tmp_path):
    checkpoint_path = _write_motion_mae_encoder_checkpoint(tmp_path, action_dim=2, past_frames=4)
    motion_obs_dim, robot_obs_dim = _actor_obs_dims(action_dim=2, motion_window_length=4)
    actor = FiLMResActor(
        robot_obs_dim=robot_obs_dim,
        motion_obs_dim=motion_obs_dim,
        action_dim=2,
        motion_window_length=4,
        motion_encoder_type="mae",
        encoder_pooling_type="last_token",
        motion_mae_encoder_checkpoint=checkpoint_path,
    )

    step = actor(
        {
            "robot_obs": torch.randn(3, robot_obs_dim),
            "motion_obs": torch.randn(3, 4, actor.motion_step_dim),
        }
    )

    assert step.action.shape == (3, 2)
    assert actor.motion_encoder.motion_encoder_type == "mae"
    assert isinstance(actor.motion_encoder.window_encoder.pooling, LastTokenPool)
    assert not any("motion_encoder.window_encoder.pooling." in key for key in actor.state_dict())


def test_motion_mae_encoder_rejects_mismatched_past_frames(tmp_path):
    checkpoint_path = _write_motion_mae_encoder_checkpoint(tmp_path, action_dim=2, past_frames=5)
    motion_obs_dim, robot_obs_dim = _actor_obs_dims(action_dim=2, motion_window_length=4)

    with pytest.raises(ValueError, match="past_frames"):
        FiLMResActor(
            robot_obs_dim=robot_obs_dim,
            motion_obs_dim=motion_obs_dim,
            action_dim=2,
            motion_window_length=4,
            motion_encoder_type="mae",
            motion_mae_encoder_checkpoint=checkpoint_path,
        )


def test_motion_mae_encoder_rejects_mismatched_input_dim(tmp_path):
    checkpoint_path = _write_motion_mae_encoder_checkpoint(
        tmp_path,
        action_dim=2,
        reference_feature_names=("root", "joint", "end_effector"),
        target_feature_names=("root", "joint", "end_effector"),
        policy_feature_names=("root", "joint", "end_effector"),
    )
    motion_obs_dim, robot_obs_dim = _actor_obs_dims(action_dim=2, motion_window_length=4)

    with pytest.raises(ValueError, match="policy motion dim"):
        FiLMResActor(
            robot_obs_dim=robot_obs_dim,
            motion_obs_dim=motion_obs_dim,
            action_dim=2,
            motion_window_length=4,
            motion_encoder_type="mae",
            motion_mae_encoder_checkpoint=checkpoint_path,
        )


def test_motion_mae_encoder_rejects_non_policy_only_feature_schema(tmp_path):
    checkpoint_path = _write_motion_mae_encoder_checkpoint(
        tmp_path,
        action_dim=2,
        reference_feature_names=("root", "joint"),
        target_feature_names=("root", "joint"),
        policy_feature_names=("root",),
    )
    motion_obs_dim, robot_obs_dim = _actor_obs_dims(action_dim=2, motion_window_length=4)

    with pytest.raises(ValueError, match="reference_feature_names"):
        FiLMResActor(
            robot_obs_dim=robot_obs_dim,
            motion_obs_dim=motion_obs_dim,
            action_dim=2,
            motion_window_length=4,
            motion_encoder_type="mae",
            motion_mae_encoder_checkpoint=checkpoint_path,
        )


def test_film_res_actor_forward_supports_single_frame_robot_obs():
    motion_obs_dim, robot_obs_dim = _actor_obs_dims(action_dim=2)
    actor = FiLMResActor(
        robot_obs_dim=robot_obs_dim,
        motion_obs_dim=motion_obs_dim,
        action_dim=2,
        num_blocks=3,
        robot_window_length=1,
        robot_encoder_type="transformer",
    )

    step = actor(
        {
            "robot_obs": torch.randn(3, robot_obs_dim),
            "motion_obs": torch.randn(3, motion_obs_dim),
        }
    )

    assert step.action.shape == (3, 2)
    assert step.log_prob.shape == (3,)
    assert actor.robot_encoder.robot_encoder_type == "mlp"
    assert actor.robot_encoder.single_frame_encoder[-1].linear.out_features == ROBOT_ENCODER_OUTPUT_DIM
    assert actor.robot_obs_normlizer.mean.shape == (robot_obs_dim,)


def test_critic_uses_flat_fallback_for_unknown_privilege_layout():
    critic = Critic(obs_dim=5, action_dim=2)
    step = critic(torch.randn(4, 5))

    assert critic.critic_type == "flat"
    assert critic.privilege_layout is None
    assert len(critic.encoder) == 4
    assert critic.encoder[0].linear.out_features == CRITIC_HIDDEN_DIM
    assert critic.encoder[1].linear.in_features == CRITIC_HIDDEN_DIM
    assert critic.encoder[1].linear.out_features == CRITIC_HIDDEN_DIM
    assert critic.encoder[2].linear.out_features == CRITIC_HIDDEN_DIM
    assert critic.encoder[3].linear.out_features == CRITIC_HIDDEN_DIM
    assert critic.head.critic_layer.in_features == CRITIC_HIDDEN_DIM
    assert step.value.shape == (4,)


def test_critic_privilege_layout_uses_ref2act_term_order_and_contiguous_slices():
    layout = build_critic_privilege_layout(2)

    assert layout.obs_dim == 31
    assert tuple(term.term_id for term in layout.terms) == (
        "priv_target_projected_gravity",
        "priv_target_joint_pos",
        "priv_target_joint_vel",
        "relative_anchor_pos",
        "relative_anchor_tangent_and_normal",
        "relative_key_pos",
        "relative_key_tangent_and_normal",
        "priv_projected_gravity",
        "anchor_lin_vel",
        "priv_anchor_ang_vel_b",
        "priv_joint_pos",
        "priv_joint_vel",
        "priv_previous_action",
    )

    offset = 0
    for term in layout.terms:
        assert term.flat_slice.start == offset
        assert term.flat_slice.stop == offset + term.flat_dim
        offset = term.flat_slice.stop
    assert offset == layout.obs_dim

    assert layout.group("target").obs_dim == 7
    assert layout.group("geometry").obs_dim == 9
    assert layout.group("robot").obs_dim == 15


def test_critic_privilege_layout_includes_key_body_geometry_dims():
    layout = build_critic_privilege_layout(2, key_body_count=4)

    assert layout.obs_dim == 67
    assert layout.group("target").obs_dim == 7
    assert layout.group("geometry").obs_dim == 45
    assert layout.group("robot").obs_dim == 15

    terms_by_id = {term.term_id: term for term in layout.terms}
    assert terms_by_id["relative_key_pos"].flat_dim == 12
    assert terms_by_id["relative_key_tangent_and_normal"].flat_dim == 24


def test_structured_critic_encodes_privilege_groups_before_fusion():
    layout = build_critic_privilege_layout(2)
    critic = Critic(obs_dim=layout.obs_dim, action_dim=2)
    step = critic(torch.randn(4, layout.obs_dim))

    assert critic.critic_type == "structured"
    assert critic.privilege_layout is not None
    assert critic.privilege_layout.obs_dim == layout.obs_dim
    assert critic.target_encoder[0].linear.in_features == layout.group("target").obs_dim
    assert critic.target_encoder[-1].linear.out_features == CRITIC_GROUP_HIDDEN_DIM
    assert critic.geometry_encoder[0].linear.in_features == layout.group("geometry").obs_dim
    assert critic.geometry_encoder[-1].linear.out_features == CRITIC_GROUP_HIDDEN_DIM
    assert critic.robot_encoder[0].linear.in_features == layout.group("robot").obs_dim
    assert critic.robot_encoder[-1].linear.out_features == CRITIC_GROUP_HIDDEN_DIM
    assert critic.fusion_encoder[0].linear.in_features == len(layout.groups) * CRITIC_GROUP_HIDDEN_DIM
    assert critic.fusion_encoder[0].linear.out_features == CRITIC_HIDDEN_DIM
    assert critic.head.critic_layer.in_features == CRITIC_HIDDEN_DIM
    assert step.value.shape == (4,)


def test_structured_critic_accepts_group_mapping_and_single_observation():
    layout = build_critic_privilege_layout(2, key_body_count=4)
    critic = Critic(obs_dim=layout.obs_dim, action_dim=2, key_body_count=4)
    flat_obs = torch.randn(4, layout.obs_dim)
    structured_obs = critic.split_observation(flat_obs)

    batch_step = critic(structured_obs, update_normlizer=True)
    single_step = critic({name: value[0] for name, value in structured_obs.items()})

    assert critic.observation_storage_specs() == {
        "critic_target_observations": (7,),
        "critic_geometry_observations": (45,),
        "critic_robot_observations": (15,),
    }
    assert batch_step.value.shape == (4,)
    assert single_step.value.numel() == 1
    torch.testing.assert_close(critic.flatten_observation(structured_obs), flat_obs)


def test_structured_critic_rejects_missing_or_missized_group_observations():
    layout = build_critic_privilege_layout(2, key_body_count=4)
    critic = Critic(obs_dim=layout.obs_dim, action_dim=2, key_body_count=4)
    structured_obs = critic.split_observation(torch.randn(4, layout.obs_dim))

    with pytest.raises(KeyError, match="robot"):
        critic({name: value for name, value in structured_obs.items() if name != "robot"})

    bad_geometry_obs = dict(structured_obs)
    bad_geometry_obs["geometry"] = bad_geometry_obs["geometry"][..., :-1]
    with pytest.raises(ValueError, match="geometry"):
        critic(bad_geometry_obs)


def test_film_res_stack_accumulates_residuals_layer_by_layer():
    class RecordingBlock(nn.Module):
        def __init__(self, value: torch.Tensor):
            super().__init__()
            self.res_scale = nn.Parameter(torch.ones(4))
            self.value = value
            self.last_input: torch.Tensor | None = None

        def forward(self, x, cond):
            self.last_input = x.clone()
            return self.value

    stack = FiLMResStack(dim=4, cond_dim=4, num_layers=2)
    block_1 = RecordingBlock(torch.full((2, 4), 2.0))
    block_2 = RecordingBlock(torch.full((2, 4), 3.0))
    stack.blocks = nn.ModuleList([block_1, block_2])

    x0 = torch.zeros(2, 4)
    output = stack(x0, torch.randn(2, 4))

    torch.testing.assert_close(block_1.last_input, x0)
    torch.testing.assert_close(block_2.last_input, torch.full((2, 4), 2.0))
    torch.testing.assert_close(output, torch.full((2, 4), 5.0))


def test_film_res_stack_uses_current_state_for_shortcut_and_branch():
    class RecordingBlock(nn.Module):
        def __init__(self, value: torch.Tensor, scale: float):
            super().__init__()
            self.res_scale = nn.Parameter(torch.full((4,), scale))
            self.value = value
            self.last_input: torch.Tensor | None = None

        def forward(self, x, cond):
            self.last_input = x.clone()
            return self.value

    stack = FiLMResStack(dim=4, cond_dim=4, num_layers=1)
    delta = torch.full((2, 4), 8.0)
    block = RecordingBlock(delta, scale=0.25)
    stack.blocks = nn.ModuleList([block])

    x0 = torch.randn(2, 4)
    output = stack(x0, torch.randn(2, 4))

    torch.testing.assert_close(block.last_input, x0)
    torch.testing.assert_close(output, x0 + 0.25 * delta)


def test_checkpoint_spec_preserves_num_blocks():
    motion_obs_dim, robot_obs_dim = _actor_obs_dims(action_dim=3)
    actor = FiLMResActor(robot_obs_dim=robot_obs_dim, motion_obs_dim=motion_obs_dim, action_dim=3, num_blocks=4)
    critic = Critic(obs_dim=3)
    checkpoint = build_training_checkpoint(
        actor=actor,
        critic=critic,
        motion_files=["env/assests/jump_anchor.npz"],
        joint_params=_joint_params(),
        action_mode="offset",
        root_name="torso_link",
        anchor_body_name="torso_link",
    )

    actor_type, actor_kwargs = resolve_checkpoint_actor_spec(checkpoint)

    assert actor_type.value == "film_res"
    assert actor_kwargs == {
        "num_blocks": 4,
        "robot_window_length": 1,
        "robot_encoder_type": "mlp",
        "motion_window_length": 1,
        "motion_encoder_type": "mlp",
        "actor_fusion_type": "film",
        "encoder_pooling_type": "learned",
    }


def test_checkpoint_override_replaces_num_blocks():
    motion_obs_dim, robot_obs_dim = _actor_obs_dims(action_dim=3)
    actor = FiLMResActor(robot_obs_dim=robot_obs_dim, motion_obs_dim=motion_obs_dim, action_dim=3, num_blocks=4)
    critic = Critic(obs_dim=3)
    checkpoint = build_training_checkpoint(
        actor=actor,
        critic=critic,
        motion_files=["env/assests/jump_anchor.npz"],
        joint_params=_joint_params(),
        action_mode="offset",
        root_name="torso_link",
        anchor_body_name="torso_link",
    )

    actor_type, actor_kwargs = resolve_checkpoint_actor_spec(checkpoint, num_blocks=5)

    assert actor_type.value == "film_res"
    assert actor_kwargs == {
        "num_blocks": 5,
        "robot_window_length": 1,
        "robot_encoder_type": "mlp",
        "motion_window_length": 1,
        "motion_encoder_type": "mlp",
        "actor_fusion_type": "film",
        "encoder_pooling_type": "learned",
    }


def test_checkpoint_spec_defaults_motion_and_robot_window_lengths_when_missing():
    motion_obs_dim, robot_obs_dim = _actor_obs_dims(action_dim=3)
    actor = FiLMResActor(robot_obs_dim=robot_obs_dim, motion_obs_dim=motion_obs_dim, action_dim=3, num_blocks=2)
    critic = Critic(obs_dim=3)
    checkpoint = CheckpointV2(
        meta={"actor_type": "film_res", "actor_kwargs": {"num_blocks": 2}},
        model={
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
        },
        env={},
        artifacts={},
    )

    actor_type, actor_kwargs = resolve_checkpoint_actor_spec(checkpoint)

    assert actor_type.value == "film_res"
    assert actor_kwargs == {
        "num_blocks": 2,
        "robot_window_length": 1,
        "robot_encoder_type": "mlp",
        "motion_window_length": 1,
        "motion_encoder_type": "mlp",
        "actor_fusion_type": "film",
        "encoder_pooling_type": "learned",
    }


def test_checkpoint_spec_restores_encoder_pooling_type_from_meta():
    motion_obs_dim, robot_obs_dim = _actor_obs_dims(action_dim=3, robot_window_length=4)
    actor = FiLMResActor(
        robot_obs_dim=robot_obs_dim,
        motion_obs_dim=motion_obs_dim,
        action_dim=3,
        num_blocks=2,
        robot_window_length=4,
        encoder_pooling_type="last_token",
    )
    checkpoint = CheckpointV2(
        meta={
            "actor_type": "film_res",
            "actor_kwargs": {
                "num_blocks": 2,
                "robot_window_length": 4,
                "robot_encoder_type": "transformer",
                "encoder_pooling_type": "last_token",
            },
        },
        model={"actor": actor.state_dict(), "critic": Critic(obs_dim=3).state_dict()},
        env={},
        artifacts={},
    )

    _, actor_kwargs = resolve_checkpoint_actor_spec(checkpoint)

    assert actor_kwargs["encoder_pooling_type"] == "last_token"


def test_load_actor_from_checkpoint_rejects_forced_pooling_mismatch():
    motion_obs_dim, robot_obs_dim = _actor_obs_dims(action_dim=3, robot_window_length=4)
    actor = FiLMResActor(
        robot_obs_dim=robot_obs_dim,
        motion_obs_dim=motion_obs_dim,
        action_dim=3,
        num_blocks=2,
        robot_window_length=4,
        encoder_pooling_type="last_token",
    )
    checkpoint = CheckpointV2(
        meta={
            "actor_type": "film_res",
            "actor_kwargs": {
                "num_blocks": 2,
                "robot_window_length": 4,
                "robot_encoder_type": "transformer",
                "encoder_pooling_type": "last_token",
            },
        },
        model={"actor": actor.state_dict(), "critic": Critic(obs_dim=3).state_dict()},
        env={},
        artifacts={},
    )

    with pytest.raises(RuntimeError, match="pooling"):
        load_actor_from_checkpoint(
            checkpoint,
            obs_dims={"robot": robot_obs_dim, "motion": motion_obs_dim, "policy": motion_obs_dim + robot_obs_dim},
            action_dim=3,
            device=torch.device("cpu"),
            encoder_pooling_type_override="learned",
        )


def test_load_actor_from_checkpoint_rejects_forced_mae_pooling_mismatch(tmp_path):
    checkpoint_path = _write_motion_mae_encoder_checkpoint(tmp_path, action_dim=2, past_frames=4)
    motion_obs_dim, robot_obs_dim = _actor_obs_dims(action_dim=2, motion_window_length=4)
    actor = FiLMResActor(
        robot_obs_dim=robot_obs_dim,
        motion_obs_dim=motion_obs_dim,
        action_dim=2,
        num_blocks=2,
        motion_window_length=4,
        motion_encoder_type="mae",
        encoder_pooling_type="last_token",
        motion_mae_encoder_checkpoint=checkpoint_path,
    )
    checkpoint = CheckpointV2(
        meta={
            "actor_type": "film_res",
            "actor_kwargs": {
                "num_blocks": 2,
                "motion_window_length": 4,
                "motion_encoder_type": "mae",
                "encoder_pooling_type": "last_token",
            },
        },
        model={"actor": actor.state_dict(), "critic": Critic(obs_dim=3).state_dict()},
        env={},
        artifacts={"motion_mae_encoder_checkpoint": str(checkpoint_path)},
    )

    with pytest.raises(RuntimeError, match="pooling"):
        load_actor_from_checkpoint(
            checkpoint,
            obs_dims={"robot": robot_obs_dim, "motion": motion_obs_dim, "policy": motion_obs_dim + robot_obs_dim},
            action_dim=2,
            device=torch.device("cpu"),
            encoder_pooling_type_override="learned",
            motion_mae_encoder_checkpoint=checkpoint_path,
        )


def test_checkpoint_spec_restores_actor_fusion_type_from_meta():
    motion_obs_dim, robot_obs_dim = _actor_obs_dims(action_dim=3)
    actor = FiLMResActor(
        robot_obs_dim=robot_obs_dim,
        motion_obs_dim=motion_obs_dim,
        action_dim=3,
        num_blocks=2,
        actor_fusion_type="motion_residual",
    )
    critic = Critic(obs_dim=3)
    checkpoint = CheckpointV2(
        meta={"actor_type": "film_res", "actor_kwargs": {"num_blocks": 2, "actor_fusion_type": "motion_residual"}},
        model={
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
        },
        env={},
        artifacts={},
    )

    actor_type, actor_kwargs = resolve_checkpoint_actor_spec(checkpoint)

    assert actor_type.value == "film_res"
    assert actor_kwargs["actor_fusion_type"] == "motion_residual"


def test_checkpoint_spec_infers_actor_fusion_type_from_state_dict_when_meta_missing():
    motion_obs_dim, robot_obs_dim = _actor_obs_dims(action_dim=3)
    actor = FiLMResActor(
        robot_obs_dim=robot_obs_dim,
        motion_obs_dim=motion_obs_dim,
        action_dim=3,
        num_blocks=2,
        actor_fusion_type="concat_mlp",
    )
    critic = Critic(obs_dim=3)
    checkpoint = CheckpointV2(
        meta={"actor_type": "film_res", "actor_kwargs": {"num_blocks": 2}},
        model={
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
        },
        env={},
        artifacts={},
    )

    _, actor_kwargs = resolve_checkpoint_actor_spec(checkpoint)

    assert actor_kwargs["actor_fusion_type"] == "concat_mlp"


def test_load_actor_from_checkpoint_restores_film_res_weights():
    motion_obs_dim, robot_obs_dim = _actor_obs_dims(action_dim=3)
    actor = FiLMResActor(robot_obs_dim=robot_obs_dim, motion_obs_dim=motion_obs_dim, action_dim=3, num_blocks=2)
    critic = Critic(obs_dim=3)
    checkpoint = CheckpointV2(
        meta={
            "actor_type": "film_res",
            "actor_kwargs": {"num_blocks": 2, "robot_window_length": 1, "motion_window_length": 1},
        },
        model={
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
        },
        env={},
        artifacts={},
    )

    loaded_actor, actor_type, actor_kwargs = load_actor_from_checkpoint(
        checkpoint,
        obs_dims={"robot": robot_obs_dim, "motion": motion_obs_dim, "policy": motion_obs_dim + robot_obs_dim},
        action_dim=3,
        device=torch.device("cpu"),
    )

    assert isinstance(loaded_actor, FiLMResActor)
    assert actor_type.value == "film_res"
    assert actor_kwargs == {
        "num_blocks": 2,
        "robot_window_length": 1,
        "robot_encoder_type": "mlp",
        "motion_window_length": 1,
        "motion_encoder_type": "mlp",
        "actor_fusion_type": "film",
        "encoder_pooling_type": "learned",
    }
    torch.testing.assert_close(
        loaded_actor.state_dict()["stack.blocks.0.res_scale"],
        actor.state_dict()["stack.blocks.0.res_scale"],
    )


def test_load_actor_from_checkpoint_accepts_missing_policy_mae_pooling(tmp_path):
    checkpoint_path = _write_motion_mae_encoder_checkpoint(tmp_path, action_dim=2, past_frames=4)
    motion_obs_dim, robot_obs_dim = _actor_obs_dims(action_dim=2, motion_window_length=4)
    actor = FiLMResActor(
        robot_obs_dim=robot_obs_dim,
        motion_obs_dim=motion_obs_dim,
        action_dim=2,
        num_blocks=2,
        motion_window_length=4,
        motion_encoder_type="mae",
        motion_mae_encoder_checkpoint=checkpoint_path,
    )
    legacy_actor_state = {
        key: value
        for key, value in actor.state_dict().items()
        if not key.startswith("motion_encoder.window_encoder.pooling.")
    }
    checkpoint = CheckpointV2(
        meta={
            "actor_type": "film_res",
            "actor_kwargs": {
                "num_blocks": 2,
                "robot_window_length": 1,
                "motion_window_length": 4,
                "motion_encoder_type": "mae",
            },
        },
        model={"actor": legacy_actor_state, "critic": Critic(obs_dim=3).state_dict()},
        env={},
        artifacts={"motion_mae_encoder_checkpoint": checkpoint_path},
    )

    loaded_actor, _, _ = load_actor_from_checkpoint(
        checkpoint,
        obs_dims={"robot": robot_obs_dim, "motion": motion_obs_dim, "policy": motion_obs_dim + robot_obs_dim},
        action_dim=2,
        device=torch.device("cpu"),
        motion_mae_encoder_checkpoint=checkpoint_path,
    )

    assert isinstance(loaded_actor.motion_encoder.window_encoder.pooling, LearnedQueryAttentionPool)
    assert any("motion_encoder.window_encoder.pooling." in key for key in loaded_actor.state_dict())


def test_checkpoint_spec_rejects_legacy_film_attn_res_actor_type():
    motion_obs_dim, robot_obs_dim = _actor_obs_dims(action_dim=3)
    actor = FiLMResActor(robot_obs_dim=robot_obs_dim, motion_obs_dim=motion_obs_dim, action_dim=3, num_blocks=2)
    critic = Critic(obs_dim=3)
    checkpoint = CheckpointV2(
        meta={"actor_type": "film_attn_res", "actor_kwargs": {"num_blocks": 2, "robot_window_length": 1}},
        model={
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
        },
        env={},
        artifacts={},
    )

    with pytest.raises(ValueError, match="film_res"):
        resolve_checkpoint_actor_spec(checkpoint)


def test_checkpoint_spec_rejects_removed_robot_cnn_encoder_type():
    motion_obs_dim, robot_obs_dim = _actor_obs_dims(action_dim=3, robot_window_length=4)
    actor = FiLMResActor(
        robot_obs_dim=robot_obs_dim,
        motion_obs_dim=motion_obs_dim,
        action_dim=3,
        num_blocks=2,
        robot_window_length=4,
        robot_encoder_type="transformer",
    )
    checkpoint = CheckpointV2(
        meta={
            "actor_type": "film_res",
            "actor_kwargs": {"num_blocks": 2, "robot_window_length": 4, "robot_encoder_type": "cnn"},
        },
        model={"actor": actor.state_dict(), "critic": Critic(obs_dim=3).state_dict()},
        env={},
        artifacts={},
    )

    with pytest.raises(ValueError, match="robot encoder type 'cnn'"):
        resolve_checkpoint_actor_spec(checkpoint)


def test_validate_checkpoint_actor_observation_dims_rejects_removed_motion_mae_latent_adapter(tmp_path):
    motion_obs_dim, robot_obs_dim = _actor_obs_dims(action_dim=2)
    checkpoint_actor = FiLMResActor(robot_obs_dim=robot_obs_dim, motion_obs_dim=motion_obs_dim, action_dim=2, num_blocks=2)
    critic = Critic(obs_dim=3)
    checkpoint = CheckpointV2(
        meta={"actor_type": "film_res", "actor_kwargs": {"num_blocks": 2, "robot_window_length": 1}},
        model={
            "actor": checkpoint_actor.state_dict(),
            "critic": critic.state_dict(),
        },
        env={},
        artifacts={"motion_mae_encoder_checkpoint": str(tmp_path / "encoder.pth")},
    )

    with pytest.raises(ValueError, match="removed Motion MAE latent-append adapter"):
        validate_checkpoint_actor_observation_dims(
            checkpoint,
            checkpoint_obs_dims={"motion": 13, "robot": robot_obs_dim, "policy": 13 + robot_obs_dim},
            runtime_obs_dims={"motion": motion_obs_dim, "robot": robot_obs_dim, "policy": motion_obs_dim + robot_obs_dim},
            motion_mae_encoder_checkpoint=str(tmp_path / "encoder.pth"),
        )
