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
    assert outputs["latent"].shape == (2, 5)
    torch.testing.assert_close(encoded, outputs["latent"])


def test_reference_motion_mae_latent_uses_last_encoded_token():
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
    model.latent_norm = torch.nn.Identity()
    model.latent_proj = torch.nn.Identity()
    encoded_visible = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [10.0, 20.0, 30.0, 40.0],
                [100.0, 200.0, 300.0, 400.0],
            ]
        ]
    )

    latent = model._pool_latent(encoded_visible)

    torch.testing.assert_close(latent, encoded_visible[:, -1])
    assert not torch.allclose(latent, encoded_visible.mean(dim=1))


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


def test_reference_motion_mae_reconstruction_trains_latent_head():
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

    assert model.latent_norm.weight.grad is not None
    assert model.latent_proj.weight.grad is not None
    assert model.decoder_condition_proj.weight.grad is not None
