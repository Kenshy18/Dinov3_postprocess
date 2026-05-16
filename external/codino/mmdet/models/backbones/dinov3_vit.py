# Copyright (c) OpenMMLab. All rights reserved.
# DINOv3 ViT-L backbone wrapper for MMDetection / Co-DETR.

import torch
from mmcv.runner import BaseModule

from mmdet.models.builder import BACKBONES

try:
    from dinov3.hub.backbones import dinov3_vitl16  # standard install
except ImportError:
    try:
        from dinov3.dinov3.hub.backbones import dinov3_vitl16  # fallback for nested package layout
    except ImportError as e:
        raise ImportError('Please install the DINOv3 package (dinov3) to use DINOv3ViT.') from e


@BACKBONES.register_module()
class DINOv3ViT(BaseModule):
    """DINOv3 ViT-L/16 backbone wrapper.

    Returns a single feature map at 16x downsampled resolution, compatible
    with SFP neck (in_channels=1024).
    """

    def __init__(
        self,
        img_size=1536,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        layers_to_use=1,
        pretrained=True,
        weights=None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg)
        self.layers_to_use = layers_to_use

        # Build DINOv3 ViT-L backbone.
        # weights: path or URL to DINOv3 checkpoint.
        self.backbone = dinov3_vitl16(
            pretrained=pretrained,
            weights=weights,
            check_hash=False,
        )

        # Sanity check: ensure dimensions match config expectation.
        if self.backbone.embed_dim != embed_dim:
            raise ValueError(
                f'embed_dim mismatch: expected {embed_dim}, got {self.backbone.embed_dim}'
            )

        self.embed_dim = self.backbone.embed_dim
        self.out_indices = (0,)
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_heads = num_heads
        self.depth = depth

    def init_weights(self):
        # We rely on the weights argument to dinov3_vitl16; nothing additional here.
        pass

    def forward(self, x):
        # x: Tensor[B, 3, H, W]
        feats = self.backbone.get_intermediate_layers(
            x,
            n=self.layers_to_use,
            reshape=True,
            return_class_token=False,
            return_extra_tokens=False,
            norm=True,
        )
        if isinstance(feats, tuple):
            feats = list(feats)

        if self.layers_to_use == 1:
            out = feats[-1]
        else:
            # Concatenate selected layers along channel dimension.
            out = torch.cat(feats, dim=1)

        return [out]
