import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


class RolloutDebugLogger:
    def __init__(self, output_prefix: str | Path | None):
        self.output_prefix = Path(output_prefix) if output_prefix is not None else None
        self.enabled = self.output_prefix is not None
        self._records: dict[str, list[np.ndarray]] = {}
        self._shapes: dict[str, tuple[int, ...]] = {}
        self._excluded: dict[str, dict[str, str]] = {}
        self._step_count = 0

    @staticmethod
    def _jsonify(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): RolloutDebugLogger._jsonify(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [RolloutDebugLogger._jsonify(item) for item in value]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if value.ndim == 0:
                return value.item()
            return value.tolist()
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return value.item()
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)

    @staticmethod
    def _squeeze_single_env(array: np.ndarray) -> np.ndarray:
        if array.ndim >= 1 and array.shape[0] == 1:
            return np.squeeze(array, axis=0)
        return array

    @classmethod
    def _normalize_value(cls, value: Any) -> tuple[np.ndarray | None, str | None]:
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            array = value
        elif isinstance(value, np.generic):
            array = np.asarray(value)
        elif isinstance(value, bool):
            array = np.asarray(value, dtype=np.bool_)
        elif isinstance(value, (int, float)):
            array = np.asarray(value, dtype=np.float32 if isinstance(value, float) else np.int64)
        elif isinstance(value, (list, tuple)):
            array = np.asarray(value)
        else:
            return None, f"unsupported_type:{type(value).__name__}"

        array = cls._squeeze_single_env(np.asarray(array))
        if array.dtype.kind == "b":
            return array.astype(np.bool_), None
        if array.dtype.kind in {"i", "u"}:
            return array.astype(np.int64), None
        if array.dtype.kind == "f":
            return array.astype(np.float32), None
        return None, f"unsupported_dtype:{array.dtype}"

    def _exclude_key(self, key: str, reason: str, value: Any) -> None:
        self._excluded[key] = {
            "reason": reason,
            "sample": repr(self._jsonify(value)),
        }
        self._records.pop(key, None)
        self._shapes.pop(key, None)

    def log_step(self, step_idx: int, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return

        step_payload = {"step": step_idx}
        step_payload.update(payload)
        self._step_count += 1

        for key, value in step_payload.items():
            if key in self._excluded:
                continue

            array, error = self._normalize_value(value)
            if error is not None or array is None:
                self._exclude_key(key, error or "normalize_failed", value)
                continue

            if key not in self._records:
                self._records[key] = [array]
                self._shapes[key] = array.shape
                continue

            if array.shape != self._shapes[key]:
                self._exclude_key(
                    key,
                    f"unstable_shape:expected={self._shapes[key]} actual={array.shape}",
                    value,
                )
                continue

            self._records[key].append(array)

    def finish(self, summary: dict[str, Any]) -> tuple[Path, Path] | None:
        if not self.enabled or self.output_prefix is None:
            return None

        self.output_prefix.parent.mkdir(parents=True, exist_ok=True)
        npz_path = self.output_prefix.with_suffix(".npz")
        json_path = self.output_prefix.with_suffix(".json")

        stacked = {
            key: np.stack(values, axis=0)
            for key, values in self._records.items()
            if values
        }
        np.savez_compressed(npz_path, **stacked)

        final_summary = dict(summary)
        final_summary.update(
            {
                "num_logged_steps": self._step_count,
                "logged_keys": sorted(stacked),
                "excluded_npz_keys": self._excluded,
                "npz_path": str(npz_path),
                "json_path": str(json_path),
            }
        )

        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(self._jsonify(final_summary), handle, indent=2, sort_keys=True)

        return npz_path, json_path
