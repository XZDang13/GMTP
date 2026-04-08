from __future__ import annotations

import torch
import torch.nn as nn


def _normalize_activation(name: str) -> str:
    normalized = str(name).strip().lower()
    if normalized not in {"relu", "gelu"}:
        raise ValueError(f"Unsupported activation '{name}'.")
    return normalized


class ReferenceMotionMAE(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        target_dim: int,
        past_frames: int,
        future_frames: int,
        d_model: int = 256,
        latent_dim: int = 64,
        encoder_layers: int = 4,
        decoder_layers: int = 2,
        nhead: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if input_dim < 1 or target_dim < 1 or past_frames < 1 or future_frames < 1:
            raise ValueError("input_dim, target_dim, past_frames, and future_frames must be positive.")
        if d_model < 1 or latent_dim < 1:
            raise ValueError("d_model and latent_dim must be positive.")
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}.")

        activation = _normalize_activation(activation)
        self.input_dim = int(input_dim)
        self.target_dim = int(target_dim)
        self.past_frames = int(past_frames)
        self.future_frames = int(future_frames)
        self.d_model = int(d_model)
        self.latent_dim = int(latent_dim)
        self.encoder_layers = int(encoder_layers)
        self.decoder_layers = int(decoder_layers)
        self.nhead = int(nhead)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout = float(dropout)
        self.activation = activation

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )

        self.input_proj = nn.Linear(input_dim, d_model)
        self.encoder_position_embedding = nn.Parameter(torch.zeros(1, past_frames, d_model))
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=encoder_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.latent_norm = nn.LayerNorm(d_model)
        self.latent_proj = nn.Linear(d_model, latent_dim)
        self.decoder_condition_proj = nn.Linear(latent_dim, d_model)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.decoder_position_embedding = nn.Parameter(torch.zeros(1, past_frames + future_frames, d_model))
        self.decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=decoder_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.output_proj = nn.Linear(d_model, target_dim)

        nn.init.normal_(self.encoder_position_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.decoder_position_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)

    def model_kwargs(self) -> dict[str, int | float | str]:
        return {
            "input_dim": self.input_dim,
            "target_dim": self.target_dim,
            "past_frames": self.past_frames,
            "future_frames": self.future_frames,
            "d_model": self.d_model,
            "latent_dim": self.latent_dim,
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
            "nhead": self.nhead,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "activation": self.activation,
        }

    def encode_visible(self, reference: torch.Tensor) -> torch.Tensor:
        reference = torch.as_tensor(reference, dtype=torch.float32)
        if reference.ndim != 3:
            raise ValueError(f"Expected reference tensor shape [B, T, D], got {tuple(reference.shape)}.")
        if tuple(reference.shape[-2:]) != (self.past_frames, self.input_dim):
            raise ValueError(
                f"Expected reference trailing shape {(self.past_frames, self.input_dim)}, "
                f"got {tuple(reference.shape[-2:])}."
            )
        visible_tokens = self.input_proj(reference)
        visible_tokens = visible_tokens + self.encoder_position_embedding
        return self.encoder(visible_tokens)

    def _pool_latent(self, encoded_visible: torch.Tensor) -> torch.Tensor:
        pooled = encoded_visible.mean(dim=1)
        return self.latent_proj(self.latent_norm(pooled))

    def encode(self, reference: torch.Tensor) -> torch.Tensor:
        encoded_visible = self.encode_visible(reference)
        return self._pool_latent(encoded_visible)

    def decode(self, encoded_visible: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        if encoded_visible.ndim != 3:
            raise ValueError(
                f"Expected encoded_visible tensor shape [B, T_visible, D], got {tuple(encoded_visible.shape)}."
            )
        if tuple(encoded_visible.shape[-2:]) != (self.past_frames, self.d_model):
            raise ValueError(
                f"Expected encoded_visible trailing shape {(self.past_frames, self.d_model)}, "
                f"got {tuple(encoded_visible.shape[-2:])}."
            )
        if latent.ndim != 2:
            raise ValueError(f"Expected latent tensor shape [B, D_latent], got {tuple(latent.shape)}.")
        if tuple(latent.shape[-1:]) != (self.latent_dim,):
            raise ValueError(
                f"Expected latent trailing shape {(self.latent_dim,)}, got {tuple(latent.shape[-1:])}."
            )

        batch_size = int(encoded_visible.shape[0])
        latent_condition = self.decoder_condition_proj(latent).unsqueeze(1)
        future_tokens = self.mask_token.expand(batch_size, self.future_frames, self.d_model) + latent_condition
        decoder_input = torch.cat((encoded_visible, future_tokens), dim=1)
        decoder_input = decoder_input + self.decoder_position_embedding
        decoded = self.decoder(decoder_input)
        return self.output_proj(decoded[:, -self.future_frames :])

    def forward(self, reference: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded_visible = self.encode_visible(reference)
        latent = self._pool_latent(encoded_visible)
        prediction = self.decode(encoded_visible, latent)
        return {
            "prediction": prediction,
            "encoded_visible": encoded_visible,
            "latent": latent,
        }
