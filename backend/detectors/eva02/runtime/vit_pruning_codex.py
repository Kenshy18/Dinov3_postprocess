from __future__ import annotations

import torch.nn as nn


def parse_block_indices(spec: str | None) -> list[int]:
    if not spec:
        return []
    indices: list[int] = []
    for token in str(spec).split(","):
        token = token.strip()
        if not token:
            continue
        indices.append(int(token))
    return sorted(set(indices))


def drop_blocks(model, block_indices: list[int]) -> tuple[object, list[int]]:
    if not block_indices:
        return model, []

    blocks = getattr(getattr(model, "backbone", None), "net", None)
    blocks = getattr(blocks, "blocks", None)
    if blocks is None:
        raise RuntimeError("model.backbone.net.blocks not found; cannot prune blocks")

    original_count = len(blocks)
    valid = [idx for idx in block_indices if 0 <= idx < original_count]
    if len(valid) != len(block_indices):
        raise ValueError(f"invalid block indices: requested={block_indices}, total_blocks={original_count}")
    if len(valid) >= original_count:
        raise ValueError(f"cannot drop all blocks: requested={valid}, total_blocks={original_count}")

    keep = [i for i in range(original_count) if i not in set(valid)]
    getattr(model.backbone, "net").blocks = nn.ModuleList([blocks[i] for i in keep])
    return model, valid
