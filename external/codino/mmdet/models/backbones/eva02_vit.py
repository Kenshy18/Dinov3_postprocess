import warnings
from collections import OrderedDict

import torch
from mmcv.runner import _load_checkpoint
from mmcv.cnn.utils.weight_init import trunc_normal_init, constant_init

from mmdet.utils import get_root_logger
from mmdet.models.builder import BACKBONES

from .vit import ViT


@BACKBONES.register_module()
class EVA02ViT(ViT):
    """ViT backbone loader for EVA-02 Detectron2 checkpoints."""

    def __init__(
        self,
        img_size=1280,
        patch_size=16,
        weights=None,
        init_cfg=None,
        pretrained=None,
        **kwargs,
    ):
        self._eva02_img_size = img_size
        self._eva02_patch_size = patch_size
        if weights is not None and init_cfg is None:
            init_cfg = dict(type='Pretrained', checkpoint=weights)
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            init_cfg=init_cfg,
            pretrained=pretrained,
            **kwargs,
        )

    def init_weights(self):
        logger = get_root_logger()
        if self.init_cfg is None:
            logger.warn(
                f'No pre-trained weights for {self.__class__.__name__}, training from scratch'
            )
            for m in self.modules():
                if isinstance(m, torch.nn.Linear):
                    trunc_normal_init(m, std=0.02, bias=0.0)
                elif isinstance(m, torch.nn.LayerNorm):
                    constant_init(m, 1.0)
            return

        if isinstance(self.init_cfg, dict):
            checkpoint = self.init_cfg.get('checkpoint')
        else:
            checkpoint = getattr(self.init_cfg, 'checkpoint', None)
        assert checkpoint, (
            f'Only support specify `Pretrained` in `init_cfg` in {self.__class__.__name__}'
        )
        ckpt = _load_checkpoint(checkpoint, logger=logger, map_location='cpu')
        if 'model_ema' in ckpt:
            _state_dict = ckpt['model_ema']
        elif 'state_dict' in ckpt:
            _state_dict = ckpt['state_dict']
        elif 'model' in ckpt:
            _state_dict = ckpt['model']
        elif 'module' in ckpt:
            _state_dict = ckpt['module']
        else:
            _state_dict = ckpt

        state_dict = OrderedDict()
        for k, v in _state_dict.items():
            if k.startswith('backbone.net.'):
                if 'relative_position_index' in k:
                    continue
                state_dict[k[len('backbone.net.'):]] = v.float()
            elif k.startswith('backbone.simfp_'):
                continue
            elif k.startswith('backbone.'):
                if 'relative_position_index' in k:
                    continue
                state_dict[k[len('backbone.'):]] = v.float()
            elif 'rope' in k:
                continue
            else:
                state_dict[k] = v.float()

        if not state_dict:
            raise ValueError('No usable backbone weights found in checkpoint.')

        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        if 'patch_embed.proj.weight' in state_dict:
            patch_embed = state_dict['patch_embed.proj.weight']
            patch_embed = torch.nn.functional.interpolate(
                patch_embed.float(),
                size=(self._eva02_patch_size, self._eva02_patch_size),
                mode='bicubic',
                align_corners=False,
            )
            state_dict['patch_embed.proj.weight'] = patch_embed

        if 'pos_embed' in state_dict:
            pos_embed_checkpoint = state_dict['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_extra_tokens = 1 if self.pretrain_use_cls_token else 0
            orig_size = int((pos_embed_checkpoint.shape[1] - num_extra_tokens) ** 0.5)
            if self.pos_embed is not None:
                new_size = int((self.pos_embed.shape[1] - num_extra_tokens) ** 0.5)
            else:
                num_patches = (self._eva02_img_size // self._eva02_patch_size) ** 2
                new_size = int(num_patches ** 0.5)
            if orig_size != new_size:
                logger.info('Position interpolate from %dx%d to %dx%d', orig_size, orig_size, new_size, new_size)
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False
                )
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                state_dict['pos_embed'] = new_pos_embed

        msg = self.load_state_dict(state_dict, False)
        logger.info(msg)
