from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from gmtp.integrations.ref2act.motion import motion_label, motion_names
from gmtp.motion_mae import (
    MotionMAEPretrainConfig,
    build_frozen_motion_mae_encoder,
    build_motion_mae_datasets,
    export_motion_mae_latents,
)
from gmtp.runtime.io import build_run_paths, write_json


def resolve_export_device(device: str) -> torch.device:
    normalized = str(device).strip().lower()
    if normalized in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(normalized)


class MotionMAELatentExportRunner:
    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        config: MotionMAEPretrainConfig,
    ) -> None:
        self.config = config
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.device = resolve_export_device(config.training.device)
        self.data_bundle = build_motion_mae_datasets(
            config.data,
            feature_config=config.feature,
            slice_weights=config.loss.slice_weights,
        )
        self.encoder = build_frozen_motion_mae_encoder(self.checkpoint_path, device=self.device)
        self.motion_files = [sequence.motion_file for sequence in self.data_bundle.train_dataset.sequences]
        self.motion_name = motion_label(self.motion_files)
        default_run_name = config.run_name or f"motion_mae_latents_{len(self.motion_files)}motions"
        self.run_paths = build_run_paths(config.output_root, "pretrain-motion-mae-latents", default_run_name)
        write_json(
            self.run_paths.config_path,
            {
                "command": "pretrain motion-mae-latents",
                "config": config.to_dict(),
                "checkpoint": str(self.checkpoint_path),
            },
        )

    def _save_split(self, split_name: str, payload: dict[str, np.ndarray]) -> str:
        output_path = self.run_paths.checkpoints_dir / f"{split_name}_latents.npz"
        np.savez(
            output_path,
            latents=payload["latents"],
            motion_names=payload["motion_names"],
            motion_files=payload["motion_files"],
            center_t=payload["center_t"],
        )
        return str(output_path.resolve())

    def export(self) -> dict[str, object]:
        pin_memory = bool(self.config.data.pin_memory and self.device.type != "cpu")
        train_payload = export_motion_mae_latents(
            self.data_bundle.train_dataset,
            self.encoder,
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
            pin_memory=pin_memory,
        )
        val_payload = export_motion_mae_latents(
            self.data_bundle.val_dataset,
            self.encoder,
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
            pin_memory=pin_memory,
        )
        train_path = self._save_split("train", train_payload)
        val_path = self._save_split("val", val_payload)
        summary = {
            "checkpoint": str(self.checkpoint_path),
            "motion_files": list(self.motion_files),
            "motion_names": motion_names(self.motion_files),
            "motion_label": self.motion_name,
            "train_window_count": int(train_payload["latents"].shape[0]),
            "val_window_count": int(val_payload["latents"].shape[0]),
            "latent_dim": int(self.encoder.latent_dim),
            "schema": self.encoder.schema.to_dict(),
            "artifacts": {
                "train_latents": train_path,
                "val_latents": val_path,
            },
            "run_dir": str(self.run_paths.root),
        }
        write_json(self.run_paths.summary_path, summary)
        return summary
