import logging
from typing import Dict, List

import torch
from detectron2.layers import ShapeSpec

from .backbone import Backbone

try:
    from dinov3.hub.backbones import dinov3_vitl16
except ImportError:
    try:
        from dinov3.dinov3.hub.backbones import dinov3_vitl16
    except ImportError as exc:
        raise ImportError(
            "Please install the DINOv3 package (dinov3) to use DINOv3Backbone."
        ) from exc

logger = logging.getLogger(__name__)


class DINOv3Backbone(Backbone):
    """DINOv3 ViT-L/16 backbone wrapper for Detectron2."""

    def __init__(
        self,
        img_size: int = 1280,
        patch_size: int = 16,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        layers_to_use: int = 1,
        out_feature: str = "last_feat",
        weights: str | None = None,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self._out_feature = out_feature
        self._out_features = [out_feature]
        self._out_feature_channels = {out_feature: embed_dim}
        self._out_feature_strides = {out_feature: patch_size}

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.layers_to_use = layers_to_use

        self.backbone = dinov3_vitl16(
            pretrained=pretrained,
            weights=weights,
            check_hash=False,
        )

        if getattr(self.backbone, "embed_dim", embed_dim) != embed_dim:
            raise ValueError(
                f"embed_dim mismatch: expected {embed_dim}, got {self.backbone.embed_dim}"
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
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
            out = torch.cat(feats, dim=1)

        return {self._out_feature: out}

    def output_shape(self) -> Dict[str, ShapeSpec]:
        return {
            name: ShapeSpec(channels=self._out_feature_channels[name], stride=self._out_feature_strides[name])
            for name in self._out_features
        }
