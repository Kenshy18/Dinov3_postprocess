#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Safe fp16 conversion helpers for the DINOv3 backbone."""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F


def _safe_layer_norm_to_half(norm: torch.nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
    y = F.layer_norm(
        x.float(),
        norm.normalized_shape,
        norm.weight.float() if norm.weight is not None else None,
        norm.bias.float() if norm.bias is not None else None,
        norm.eps,
    )
    return y.to(torch.float16)


def _safe_layer_scale_fp32(layer_scale: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    gamma = getattr(layer_scale, "gamma", None)
    if gamma is None:
        return x.float()
    return x.float() * gamma.float()


def _safe_fp16_block_forward_list(self, x_list, rope_list=None):
    if rope_list is None:
        rope_list = [None for _ in x_list]

    outputs = []
    for x, rope in zip(x_list, rope_list):
        x_base = x.float()
        norm1 = _safe_layer_norm_to_half(self.norm1, x_base)
        attn_out = self.attn(norm1, rope=rope)
        x_attn = x_base + _safe_layer_scale_fp32(self.ls1, attn_out)

        norm2 = _safe_layer_norm_to_half(self.norm2, x_attn)
        mlp_out = self.mlp(norm2)
        x_ffn = x_attn + _safe_layer_scale_fp32(self.ls2, mlp_out)
        outputs.append(x_ffn)
    return outputs


def apply_safe_fp16_islands(backbone_net: torch.nn.Module, *, verbose: bool = True) -> torch.nn.Module:
    """Use fp16 weights while keeping DINOv3 numerically fragile islands in fp32."""
    from dinov3.layers.block import SelfAttentionBlock
    from dinov3.layers.layer_scale import LayerScale

    backbone_net = backbone_net.half()
    patched_blocks = 0
    for module in backbone_net.modules():
        if isinstance(module, (torch.nn.LayerNorm, LayerScale)):
            module.float()
        if isinstance(module, SelfAttentionBlock):
            module._forward_list = types.MethodType(_safe_fp16_block_forward_list, module)
            patched_blocks += 1
    if verbose:
        print(f"[INFO] safe full-half backbone enabled: patched SelfAttentionBlock={patched_blocks}")
    return backbone_net
