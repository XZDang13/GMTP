from __future__ import annotations

import math

import torch
import torch.nn as nn


def _resolve_activation(name: str) -> nn.Module:
    normalized = str(name).strip().lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized == "elu":
        return nn.ELU()
    return nn.SiLU()


def _conv_output_length(length: int, kernel_size: int, stride: int, padding: int) -> int:
    return math.floor((length + 2 * padding - kernel_size) / stride + 1)


class TemporalConvEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        window_length: int,
        latent_dim: int,
        channels: tuple[int, ...],
        kernel_size: int = 3,
        stride: int = 1,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if input_dim < 1 or window_length < 1 or latent_dim < 1:
            raise ValueError("TemporalConvEncoder input_dim, window_length, and latent_dim must be positive.")
        if not channels:
            raise ValueError("TemporalConvEncoder requires at least one convolution channel.")

        padding = kernel_size // 2
        conv_layers = []
        current_channels = input_dim
        current_length = window_length
        for output_channels in channels:
            conv_layers.append(
                nn.Conv1d(
                    current_channels,
                    output_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            conv_layers.append(_resolve_activation(activation))
            current_channels = output_channels
            current_length = _conv_output_length(current_length, kernel_size, stride, padding)
            if current_length < 1:
                raise ValueError("TemporalConvEncoder convolution stack collapsed the temporal dimension to zero.")

        self.input_dim = int(input_dim)
        self.window_length = int(window_length)
        self.latent_dim = int(latent_dim)
        self.channels = tuple(int(value) for value in channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.activation = str(activation)
        self.conv = nn.Sequential(*conv_layers)
        hidden_dim = current_channels * current_length
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if reference.ndim != 3:
            raise ValueError(f"Expected reference tensor shape [B, T, D], got {tuple(reference.shape)}.")
        x = reference.transpose(1, 2)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        return self.mu_head(x), self.logvar_head(x)

    def encode(self, reference: torch.Tensor, *, deterministic: bool = True) -> torch.Tensor:
        mu, logvar = self(reference)
        if deterministic:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std


class MLPDecoder(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        future_frames: int,
        target_dim: int,
        hidden_dims: tuple[int, ...],
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if latent_dim < 1 or future_frames < 1 or target_dim < 1:
            raise ValueError("MLPDecoder latent_dim, future_frames, and target_dim must be positive.")
        layers = []
        current_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(_resolve_activation(activation))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, future_frames * target_dim))
        self.latent_dim = int(latent_dim)
        self.future_frames = int(future_frames)
        self.target_dim = int(target_dim)
        self.hidden_dims = tuple(int(value) for value in hidden_dims)
        self.activation = str(activation)
        self.network = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        prediction = self.network(latent)
        return prediction.reshape(latent.shape[0], self.future_frames, self.target_dim)


class ReferenceMotionVAE(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        target_dim: int,
        past_frames: int,
        future_frames: int,
        latent_dim: int,
        encoder_channels: tuple[int, ...],
        kernel_size: int = 3,
        stride: int = 1,
        activation: str = "silu",
        decoder_hidden_dims: tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.target_dim = int(target_dim)
        self.past_frames = int(past_frames)
        self.future_frames = int(future_frames)
        self.latent_dim = int(latent_dim)
        self.encoder = TemporalConvEncoder(
            input_dim=input_dim,
            window_length=past_frames,
            latent_dim=latent_dim,
            channels=encoder_channels,
            kernel_size=kernel_size,
            stride=stride,
            activation=activation,
        )
        self.decoder = MLPDecoder(
            latent_dim=latent_dim,
            future_frames=future_frames,
            target_dim=target_dim,
            hidden_dims=decoder_hidden_dims,
            activation=activation,
        )

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def encode(self, reference: torch.Tensor, *, deterministic: bool = True) -> torch.Tensor:
        return self.encoder.encode(reference, deterministic=deterministic)

    def forward(self, reference: torch.Tensor, *, deterministic: bool = False) -> dict[str, torch.Tensor]:
        mu, logvar = self.encoder(reference)
        latent = mu if deterministic else self.reparameterize(mu, logvar)
        prediction = self.decoder(latent)
        return {
            "prediction": prediction,
            "mu": mu,
            "logvar": logvar,
            "latent": latent,
        }
