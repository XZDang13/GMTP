from __future__ import annotations

from pathlib import Path

import torch

from gmtp.models import (
    ActorType,
    FiLMAttnResActor,
    build_actor,
    infer_film_res_blocks,
    normalize_actor_type,
)
from gmtp.runtime.checkpoints import CheckpointV2


def resolve_checkpoint_actor_spec(
    checkpoint: CheckpointV2,
    *,
    actor_type_override: str | None = None,
    num_blocks: int | None = None,
    attn_block_size: int | None = None,
) -> tuple[ActorType, dict[str, int]]:
    actor_type = normalize_actor_type(actor_type_override or checkpoint.meta.get("actor_type"))
    actor_weights = checkpoint.model["actor"]
    checkpoint_actor_kwargs = dict(checkpoint.meta.get("actor_kwargs", {}))
    actor_kwargs = {
        "num_blocks": int(
            num_blocks
            if num_blocks is not None
            else checkpoint_actor_kwargs.get("num_blocks", infer_film_res_blocks(actor_weights))
        ),
        "attn_block_size": int(
            attn_block_size
            if attn_block_size is not None
            else checkpoint_actor_kwargs.get("attn_block_size", FiLMAttnResActor.DEFAULT_ATTN_BLOCK_SIZE)
        ),
    }
    return actor_type, actor_kwargs


def load_actor_from_checkpoint(
    checkpoint: CheckpointV2,
    *,
    obs_dims: dict[str, int],
    action_dim: int,
    device: torch.device,
    actor_type_override: str | None = None,
    num_blocks: int | None = None,
    attn_block_size: int | None = None,
) -> tuple[torch.nn.Module, ActorType, dict[str, int]]:
    actor_type, actor_kwargs = resolve_checkpoint_actor_spec(
        checkpoint,
        actor_type_override=actor_type_override,
        num_blocks=num_blocks,
        attn_block_size=attn_block_size,
    )
    actor = build_actor(obs_dims, actor_type, action_dim, actor_kwargs=actor_kwargs).to(device)
    actor.load_state_dict(checkpoint.model["actor"])
    actor.eval()
    return actor, actor_type, actor_kwargs


def resolve_checkpoint_stem(path: str | Path) -> str:
    return Path(path).expanduser().resolve().stem
