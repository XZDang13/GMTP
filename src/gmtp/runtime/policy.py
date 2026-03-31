from __future__ import annotations

from pathlib import Path

import torch

from gmtp.models import (
    ActorType,
    build_actor,
    infer_adain_res_blocks,
    infer_recurrent_actor_kwargs,
    normalize_actor_type,
)
from gmtp.runtime.checkpoints import CheckpointV2


def resolve_checkpoint_actor_spec(
    checkpoint: CheckpointV2,
    *,
    actor_type_override: str | None = None,
    adain_res_blocks: int | None = None,
) -> tuple[ActorType, dict[str, int]]:
    actor_type = normalize_actor_type(actor_type_override or checkpoint.meta.get("actor_type"))
    actor_weights = checkpoint.model["actor"]
    actor_kwargs = dict(checkpoint.meta.get("actor_kwargs", {}))

    if actor_type == ActorType.ADAIN_RES:
        actor_kwargs = {
            "num_blocks": int(
                adain_res_blocks
                if adain_res_blocks is not None
                else actor_kwargs.get("num_blocks", infer_adain_res_blocks(actor_weights))
            )
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
    adain_res_blocks: int | None = None,
) -> tuple[torch.nn.Module, ActorType, dict[str, int]]:
    actor_type, actor_kwargs = resolve_checkpoint_actor_spec(
        checkpoint,
        actor_type_override=actor_type_override,
        adain_res_blocks=adain_res_blocks,
    )
    actor = build_actor(obs_dims, actor_type, action_dim, actor_kwargs=actor_kwargs).to(device)
    actor.load_state_dict(checkpoint.model["actor"])
    actor.eval()
    return actor, actor_type, actor_kwargs


def resolve_checkpoint_stem(path: str | Path) -> str:
    return Path(path).expanduser().resolve().stem
