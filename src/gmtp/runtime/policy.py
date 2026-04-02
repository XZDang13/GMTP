from __future__ import annotations

import re
from pathlib import Path

import torch

from gmtp.models import (
    ActorType,
    build_actor,
    infer_film_res_blocks,
    infer_recurrent_actor_kwargs,
    normalize_actor_type,
)
from gmtp.runtime.checkpoints import CheckpointV2


def _is_legacy_adain_res_state_dict(actor_weights: dict[str, torch.Tensor]) -> bool:
    return any(re.match(r"^block_\d+\.", key) for key in actor_weights)


def _upgrade_legacy_adain_res_weights(
    actor_weights: dict[str, torch.Tensor],
    target_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    upgraded_state = dict(target_state_dict)

    for key, value in actor_weights.items():
        match = re.match(r"^block_(\d+)\.(.+)$", key)
        if match is None:
            upgraded_state[key] = value
            continue

        block_idx = int(match.group(1)) - 1
        suffix = match.group(2)
        if suffix.startswith("adain.style."):
            suffix = f"modulation.affine.{suffix.removeprefix('adain.style.')}"
        upgraded_state[f"blocks.{block_idx}.{suffix}"] = value

    return upgraded_state


def resolve_checkpoint_actor_spec(
    checkpoint: CheckpointV2,
    *,
    actor_type_override: str | None = None,
    film_res_blocks: int | None = None,
    film_attn_res_block_size: int | None = None,
) -> tuple[ActorType, dict[str, int]]:
    actor_type = normalize_actor_type(actor_type_override or checkpoint.meta.get("actor_type"))
    actor_weights = checkpoint.model["actor"]
    actor_kwargs = dict(checkpoint.meta.get("actor_kwargs", {}))

    if actor_type == ActorType.FILM_RES:
        actor_kwargs = {
            "num_blocks": int(
                film_res_blocks
                if film_res_blocks is not None
                else actor_kwargs.get("num_blocks", infer_film_res_blocks(actor_weights))
            )
        }
    elif actor_type == ActorType.FILM_ATTN_RES:
        actor_kwargs = {
            "num_blocks": int(
                film_res_blocks
                if film_res_blocks is not None
                else actor_kwargs.get("num_blocks", infer_film_res_blocks(actor_weights))
            ),
            "attn_block_size": int(
                film_attn_res_block_size
                if film_attn_res_block_size is not None
                else actor_kwargs.get("attn_block_size", 4)
            ),
        }
    elif actor_type == ActorType.RECURRENT:
        inferred_actor_kwargs = infer_recurrent_actor_kwargs(actor_weights)
        actor_kwargs = {
            "hidden_size": int(actor_kwargs.get("hidden_size", inferred_actor_kwargs["hidden_size"])),
            "num_layers": int(actor_kwargs.get("num_layers", inferred_actor_kwargs["num_layers"])),
        }
    else:
        actor_kwargs = {}

    return actor_type, actor_kwargs


def load_actor_from_checkpoint(
    checkpoint: CheckpointV2,
    *,
    obs_dims: dict[str, int],
    action_dim: int,
    device: torch.device,
    actor_type_override: str | None = None,
    film_res_blocks: int | None = None,
    film_attn_res_block_size: int | None = None,
) -> tuple[torch.nn.Module, ActorType, dict[str, int]]:
    actor_type, actor_kwargs = resolve_checkpoint_actor_spec(
        checkpoint,
        actor_type_override=actor_type_override,
        film_res_blocks=film_res_blocks,
        film_attn_res_block_size=film_attn_res_block_size,
    )
    actor = build_actor(obs_dims, actor_type, action_dim, actor_kwargs=actor_kwargs).to(device)
    actor_weights = checkpoint.model["actor"]
    if actor_type == ActorType.FILM_RES and _is_legacy_adain_res_state_dict(actor_weights):
        actor_weights = _upgrade_legacy_adain_res_weights(actor_weights, actor.state_dict())
    actor.load_state_dict(actor_weights)
    actor.eval()
    return actor, actor_type, actor_kwargs


def resolve_checkpoint_stem(path: str | Path) -> str:
    return Path(path).expanduser().resolve().stem
