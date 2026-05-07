import torch

from gmtp.motion_mae import FeatureSliceSpec, ReferenceMotionMAE, compute_motion_mae_losses


def test_reference_motion_mae_shapes_and_deterministic_encode():
    model = ReferenceMotionMAE(
        input_dim=7,
        target_dim=9,
        past_frames=4,
        future_frames=3,
        latent_dim=5,
        d_model=16,
        encoder_layers=2,
        decoder_layers=1,
        nhead=4,
        dim_feedforward=32,
    )
    reference = torch.randn(2, 4, 7)

    outputs = model(reference)
    encoded = model.encode(reference)

    assert outputs["prediction"].shape == (2, 3, 9)
    assert outputs["encoded_visible"].shape == (2, 4, 16)
    assert "latent" not in outputs
    torch.testing.assert_close(encoded, outputs["encoded_visible"])


def test_reference_motion_mae_has_no_pooling_or_latent_modules():
    model = ReferenceMotionMAE(
        input_dim=4,
        target_dim=4,
        past_frames=3,
        future_frames=1,
        latent_dim=4,
        d_model=4,
        encoder_layers=1,
        decoder_layers=1,
        nhead=2,
        dim_feedforward=8,
    )

    state_keys = tuple(model.state_dict())

    assert not hasattr(model, "latent_pool")
    assert not hasattr(model, "latent_norm")
    assert not hasattr(model, "latent_proj")
    assert not hasattr(model, "decoder_condition_proj")
    assert not any(
        key.startswith(("latent_pool.", "latent_norm.", "latent_proj.", "decoder_condition_proj."))
        for key in state_keys
    )


def test_compute_motion_mae_losses_uses_structured_slices():
    prediction = torch.zeros(2, 3, 6)
    target = torch.ones(2, 3, 6)
    losses = compute_motion_mae_losses(
        prediction,
        target,
        target_slices=(
            FeatureSliceSpec(name="root", start=0, end=2, weight=2.0),
            FeatureSliceSpec(name="joint", start=2, end=6, weight=1.0),
        ),
        reconstruction_loss="mse",
    )

    assert torch.isclose(losses["root_loss"], torch.tensor(1.0))
    assert "joint_loss" not in losses
    assert torch.isclose(losses["joint_pos_loss"], torch.tensor(1.0))
    assert torch.isclose(losses["joint_vel_loss"], torch.tensor(1.0))
    assert torch.isclose(losses["joint_pos_weighted_loss"], torch.tensor(0.5))
    assert torch.isclose(losses["joint_vel_weighted_loss"], torch.tensor(0.5))
    assert torch.isclose(losses["joint_pos_error"], torch.tensor(1.0))
    assert torch.isclose(losses["joint_vel_error"], torch.tensor(1.0))
    assert torch.isclose(losses["reconstruction_loss"], torch.tensor(3.0))
    assert torch.isclose(losses["loss"], torch.tensor(3.0))


def test_reference_motion_mae_reconstruction_trains_token_encoder_and_decoder():
    model = ReferenceMotionMAE(
        input_dim=7,
        target_dim=9,
        past_frames=4,
        future_frames=3,
        latent_dim=5,
        d_model=16,
        encoder_layers=2,
        decoder_layers=1,
        nhead=4,
        dim_feedforward=32,
    )
    reference = torch.randn(2, 4, 7)

    outputs = model(reference)
    loss = outputs["prediction"].square().mean()
    loss.backward()

    assert model.input_proj.weight.grad is not None
    assert model.encoder.layers[0].self_attn.in_proj_weight.grad is not None
    assert model.decoder.layers[0].self_attn.in_proj_weight.grad is not None
    assert model.output_proj.weight.grad is not None
