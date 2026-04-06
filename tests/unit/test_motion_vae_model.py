import torch

from gmtp.motion_vae import FeatureSliceSpec, ReferenceMotionVAE, TemporalConvEncoder, compute_motion_vae_losses


def test_reference_motion_vae_shapes_and_deterministic_encode():
    model = ReferenceMotionVAE(
        input_dim=7,
        target_dim=9,
        past_frames=4,
        future_frames=3,
        latent_dim=5,
        encoder_channels=(16, 16),
        decoder_hidden_dims=(12,),
    )
    reference = torch.randn(2, 4, 7)

    outputs = model(reference, deterministic=True)
    encoded = model.encode(reference, deterministic=True)

    assert outputs["prediction"].shape == (2, 3, 9)
    assert outputs["mu"].shape == (2, 5)
    assert outputs["logvar"].shape == (2, 5)
    torch.testing.assert_close(outputs["latent"], outputs["mu"])
    torch.testing.assert_close(encoded, outputs["mu"])


def test_temporal_conv_encoder_supports_stochastic_encoding():
    encoder = TemporalConvEncoder(
        input_dim=6,
        window_length=5,
        latent_dim=4,
        channels=(8, 8),
    )
    reference = torch.randn(3, 5, 6)

    mu, logvar = encoder(reference)
    sample = encoder.encode(reference, deterministic=False)

    assert mu.shape == (3, 4)
    assert logvar.shape == (3, 4)
    assert sample.shape == (3, 4)


def test_compute_motion_vae_losses_uses_structured_slices_and_beta():
    prediction = torch.zeros(2, 3, 5)
    target = torch.ones(2, 3, 5)
    mu = torch.ones(2, 2)
    logvar = torch.zeros(2, 2)
    losses = compute_motion_vae_losses(
        prediction,
        target,
        mu=mu,
        logvar=logvar,
        target_slices=(
            FeatureSliceSpec(name="root", start=0, end=2, weight=2.0),
            FeatureSliceSpec(name="joint", start=2, end=5, weight=1.0),
        ),
        beta=0.1,
        reconstruction_loss="mse",
    )

    assert torch.isclose(losses["root_loss"], torch.tensor(1.0))
    assert torch.isclose(losses["joint_loss"], torch.tensor(1.0))
    assert torch.isclose(losses["reconstruction_loss"], torch.tensor(3.0))
    assert torch.isclose(losses["kl_loss"], torch.tensor(0.5))
    assert torch.isclose(losses["loss"], torch.tensor(3.05), atol=1.0e-5)
