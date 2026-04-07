from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gmtp.integrations.ref2act.motion import motion_label, motion_names
from gmtp.motion_mae import (
    MotionMAEPretrainConfig,
    ReferenceMotionMAE,
    build_motion_mae_checkpoint,
    build_motion_mae_datasets,
    build_motion_mae_encoder_checkpoint,
    compute_motion_mae_losses,
    save_motion_mae_checkpoint,
    save_motion_mae_encoder_checkpoint,
)
from gmtp.runtime.io import build_run_paths, write_json


def resolve_training_device(device: str) -> torch.device:
    normalized = str(device).strip().lower()
    if normalized in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(normalized)


class MotionMAEPretrainRunner:
    def __init__(self, config: MotionMAEPretrainConfig) -> None:
        self.config = config
        self.device = resolve_training_device(config.training.device)
        self.data_bundle = build_motion_mae_datasets(
            config.data,
            feature_config=config.feature,
            slice_weights=config.loss.slice_weights,
        )
        self.schema = self.data_bundle.schema
        self.motion_files = [sequence.motion_file for sequence in self.data_bundle.train_dataset.sequences]
        self.motion_name = motion_label(self.motion_files)
        default_run_name = config.run_name or f"motion_mae_{len(self.motion_files)}motions"
        self.run_paths = build_run_paths(config.output_root, "pretrain-motion-mae", default_run_name)
        write_json(self.run_paths.config_path, {"command": "pretrain motion-mae", "config": config.to_dict()})

        self.model = ReferenceMotionMAE(
            input_dim=self.schema.d_ref,
            target_dim=self.schema.d_target,
            past_frames=config.data.past_frames,
            future_frames=config.data.future_frames,
            d_model=config.model.d_model,
            latent_dim=config.model.latent_dim,
            encoder_layers=config.model.encoder_layers,
            decoder_layers=config.model.decoder_layers,
            nhead=config.model.nhead,
            dim_feedforward=config.model.dim_feedforward,
            dropout=config.model.dropout,
            activation=config.model.activation,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
        )

        pin_memory = bool(config.data.pin_memory and self.device.type != "cpu")
        generator = torch.Generator().manual_seed(config.data.seed)
        self.train_loader = DataLoader(
            self.data_bundle.train_dataset,
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            pin_memory=pin_memory,
            generator=generator,
        )
        self.val_loader = DataLoader(
            self.data_bundle.val_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=pin_memory,
        )

    def _run_epoch(self, loader: DataLoader, *, training: bool) -> dict[str, float]:
        self.model.train(mode=training)
        aggregates: dict[str, float] = {}
        total_samples = 0
        for batch_index, batch in enumerate(loader, start=1):
            reference = batch["reference"].to(self.device, non_blocking=True)
            target = batch["target"].to(self.device, non_blocking=True)

            with torch.set_grad_enabled(training):
                outputs = self.model(reference)
                losses = compute_motion_mae_losses(
                    outputs["prediction"],
                    target,
                    target_slices=self.schema.target_slices,
                    reconstruction_loss=self.config.loss.reconstruction_loss,
                )
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    losses["loss"].backward()
                    if self.config.training.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip_norm)
                    self.optimizer.step()

            batch_size = int(reference.shape[0])
            total_samples += batch_size
            for key, value in losses.items():
                aggregates[key] = aggregates.get(key, 0.0) + float(value.detach().cpu()) * batch_size

            if training and (
                batch_index % self.config.training.log_interval == 0 or batch_index == len(loader)
            ):
                print(
                    f"train batch {batch_index}/{len(loader)} "
                    f"loss={float(losses['loss'].detach().cpu()):.6f} "
                    f"root={float(losses['root_loss'].detach().cpu()):.6f} "
                    f"joint={float(losses['joint_loss'].detach().cpu()):.6f} "
                    f"eef={float(losses['end_effector_loss'].detach().cpu()):.6f}"
                )

        return {key: value / total_samples for key, value in aggregates.items()}

    def _save_artifacts(self, prefix: str, *, epoch: int, best_metric: float) -> dict[str, str]:
        artifacts = {
            "run_dir": str(self.run_paths.root),
            "train_motion_names": list(self.data_bundle.train_motion_names),
            "val_motion_names": list(self.data_bundle.val_motion_names),
        }
        mae_checkpoint = build_motion_mae_checkpoint(
            model=self.model,
            optimizer=self.optimizer if self.config.export.save_optimizer_state else None,
            schema=self.schema,
            config=self.config,
            epoch=epoch,
            best_metric=best_metric,
            artifacts=artifacts,
        )
        encoder_checkpoint = build_motion_mae_encoder_checkpoint(
            model=self.model,
            schema=self.schema,
            config=self.config,
            epoch=epoch,
            best_metric=best_metric,
            artifacts=artifacts,
        )
        mae_path = save_motion_mae_checkpoint(
            mae_checkpoint,
            self.run_paths.checkpoints_dir / f"{prefix}_motion_mae.pth",
        )
        encoder_path = save_motion_mae_encoder_checkpoint(
            encoder_checkpoint,
            self.run_paths.checkpoints_dir / f"{prefix}_motion_mae_encoder.pth",
        )
        return {
            f"{prefix}_motion_mae": str(mae_path),
            f"{prefix}_motion_mae_encoder": str(encoder_path),
        }

    def train(self) -> dict[str, object]:
        best_metric = float("inf")
        best_epoch = 0
        best_paths: dict[str, str] = {}
        final_paths: dict[str, str] = {}
        final_train_metrics: dict[str, float] = {}
        final_val_metrics: dict[str, float] = {}

        for epoch in range(1, self.config.training.epochs + 1):
            train_metrics = self._run_epoch(self.train_loader, training=True)
            val_metrics = self._run_epoch(self.val_loader, training=False)
            final_train_metrics = train_metrics
            final_val_metrics = val_metrics
            val_loss = float(val_metrics["loss"])
            print(
                f"epoch {epoch}/{self.config.training.epochs} "
                f"train_loss={train_metrics['loss']:.6f} val_loss={val_loss:.6f}"
            )
            if val_loss < best_metric:
                best_metric = val_loss
                best_epoch = epoch
                best_paths = self._save_artifacts("best", epoch=epoch, best_metric=best_metric)

        final_paths = self._save_artifacts(
            "final",
            epoch=self.config.training.epochs,
            best_metric=best_metric,
        )
        summary = {
            "motion_files": list(self.motion_files),
            "motion_names": motion_names(self.motion_files),
            "motion_label": self.motion_name,
            "train_motion_names": list(self.data_bundle.train_motion_names),
            "val_motion_names": list(self.data_bundle.val_motion_names),
            "train_window_count": self.data_bundle.train_window_count,
            "val_window_count": self.data_bundle.val_window_count,
            "schema": self.schema.to_dict(),
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "train_metrics": final_train_metrics,
            "val_metrics": final_val_metrics,
            "artifacts": {
                **best_paths,
                **final_paths,
            },
            "run_dir": str(self.run_paths.root),
        }
        write_json(self.run_paths.summary_path, summary)
        return summary
