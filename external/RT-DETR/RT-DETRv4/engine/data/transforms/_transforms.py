"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn

import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

import PIL
import PIL.Image

from typing import Any, Dict, List, Optional

from .._misc import convert_to_tv_tensor, _boxes_keys
from .._misc import Image, Video, Mask, BoundingBoxes
from .._misc import SanitizeBoundingBoxes

from ...core import register
torchvision.disable_beta_transforms_warning()


RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
RandomZoomOut = register()(T.RandomZoomOut)
RandomHorizontalFlip = register()(T.RandomHorizontalFlip)
Resize = register()(T.Resize)
ClampBoundingBoxes = register()(T.ClampBoundingBoxes)
# ToImageTensor = register()(T.ToImageTensor)
# ConvertDtype = register()(T.ConvertDtype)
# PILToTensor = register()(T.PILToTensor)
SanitizeBoundingBoxes = register(name='SanitizeBoundingBoxes')(SanitizeBoundingBoxes)
RandomCrop = register()(T.RandomCrop)
Normalize = register()(T.Normalize)


def _as_sample(inputs):
    return inputs if len(inputs) > 1 else inputs[0]


def _xyxy_boxes(boxes):
    tensor = torch.as_tensor(boxes, dtype=torch.float32).clone()
    if tensor.numel() == 0:
        return tensor.reshape(0, 4)
    fmt = getattr(boxes, 'format', None)
    if fmt is not None:
        in_fmt = fmt.value.lower()
        if in_fmt != 'xyxy':
            tensor = torchvision.ops.box_convert(tensor, in_fmt=in_fmt, out_fmt='xyxy')
    return tensor.reshape(-1, 4)


def _replace_boxes(target, boxes, spatial_size):
    target = dict(target)
    target['boxes'] = convert_to_tv_tensor(boxes, key='boxes', box_format='XYXY', spatial_size=spatial_size)
    return target


def _clamp_xyxy(boxes, width, height):
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    boxes[:, 0::2].clamp_(0, width)
    boxes[:, 1::2].clamp_(0, height)
    return boxes


def _get_spatial_size(image):
    height, width = F.get_size(image)
    return int(height), int(width)


@register()
class EmptyTransform(T.Transform):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register()
class PadToSize(T.Pad):
    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sp = _get_spatial_size(flat_inputs[0])
        h, w = self.size[1] - sp[0], self.size[0] - sp[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def __init__(self, size, fill=0, padding_mode='constant') -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        super().__init__(0, fill, padding_mode)

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)
    
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        fill = self._fill[type(inpt)]
        padding = params['padding']
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]['padding'] = torch.tensor(self.padding)
        return outputs


@register()
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1, min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2, sampler_options: Optional[List[float]] = None, trials: int = 40, p: float = 1.0):
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


@register()
class RandomClippedCrop(object):
    """Random crop that clips boxes to the crop instead of requiring containment."""

    def __init__(self, p=0.7, min_scale=0.5, max_scale=1.0, aspect_ratio=None, trials=10):
        self.p = float(p)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.aspect_ratio = None if aspect_ratio is None else float(aspect_ratio)
        self.trials = int(trials)

    def __call__(self, *inputs: Any) -> Any:
        sample = _as_sample(inputs)
        if not isinstance(sample, (tuple, list)) or len(sample) < 2:
            return sample
        if torch.rand(1) >= self.p:
            return sample

        image, target = sample[0], sample[1]
        height, width = _get_spatial_size(image)
        if height <= 1 or width <= 1:
            return sample

        aspect = self.aspect_ratio if self.aspect_ratio is not None else width / max(1, height)
        if aspect <= 0:
            return sample

        max_crop_h = min(float(height), float(width) / aspect)
        if max_crop_h < 1:
            return sample
        min_scale = min(max(self.min_scale, 1e-6), 1.0)
        max_scale = min(max(self.max_scale, min_scale), 1.0)
        boxes = target.get('boxes', None) if isinstance(target, dict) else None

        crop_params = None
        cropped_boxes = None
        for _ in range(max(1, self.trials)):
            scale = float(torch.empty(1).uniform_(min_scale, max_scale).item())
            crop_h = max(1, min(height, int(round(max_crop_h * scale))))
            crop_w = max(1, min(width, int(round(crop_h * aspect))))
            if crop_w > width:
                crop_w = width
                crop_h = max(1, min(height, int(round(crop_w / aspect))))
            if crop_h > height:
                crop_h = height
                crop_w = max(1, min(width, int(round(crop_h * aspect))))
            if crop_w >= width and crop_h >= height:
                continue

            left = int(torch.randint(0, width - crop_w + 1, (1,)).item())
            top = int(torch.randint(0, height - crop_h + 1, (1,)).item())

            if boxes is None:
                crop_params = (top, left, crop_h, crop_w)
                break

            xyxy = _xyxy_boxes(boxes)
            candidate_boxes = xyxy - torch.tensor([left, top, left, top], dtype=xyxy.dtype)
            candidate_boxes = _clamp_xyxy(candidate_boxes, crop_w, crop_h)
            valid = (candidate_boxes[:, 2] > candidate_boxes[:, 0]) & (candidate_boxes[:, 3] > candidate_boxes[:, 1])
            if bool(valid.any()):
                crop_params = (top, left, crop_h, crop_w)
                cropped_boxes = candidate_boxes
                break

        if crop_params is None:
            return sample

        top, left, crop_h, crop_w = crop_params

        cropped_image = F.crop(image, top=top, left=left, height=crop_h, width=crop_w)
        cropped_target = target
        if cropped_boxes is not None:
            cropped_target = _replace_boxes(target, cropped_boxes, (crop_h, crop_w))

        return (cropped_image, cropped_target, *sample[2:])


@register()
class LetterboxResize(object):
    """Aspect-ratio preserving resize with padding to a fixed size."""

    def __init__(self, size, fill=0):
        if isinstance(size, int):
            size = [size, size]
        if len(size) != 2:
            raise ValueError('LetterboxResize size must be [height, width].')
        self.size = [int(size[0]), int(size[1])]
        self.fill = fill

    def __call__(self, *inputs: Any) -> Any:
        sample = _as_sample(inputs)
        if not isinstance(sample, (tuple, list)) or len(sample) < 2:
            return sample

        image, target = sample[0], sample[1]
        out_h, out_w = self.size
        in_h, in_w = _get_spatial_size(image)
        scale = min(out_w / max(1, in_w), out_h / max(1, in_h))
        new_w = max(1, min(out_w, int(round(in_w * scale))))
        new_h = max(1, min(out_h, int(round(in_h * scale))))
        pad_left = (out_w - new_w) // 2
        pad_top = (out_h - new_h) // 2
        pad_right = out_w - new_w - pad_left
        pad_bottom = out_h - new_h - pad_top

        resized_image = F.resize(image, size=[new_h, new_w], antialias=True)
        padded_image = F.pad(
            resized_image,
            padding=[pad_left, pad_top, pad_right, pad_bottom],
            fill=self.fill,
        )

        boxes = target.get('boxes', None) if isinstance(target, dict) else None
        if boxes is not None:
            xyxy = _xyxy_boxes(boxes)
            xyxy = xyxy * scale + torch.tensor(
                [pad_left, pad_top, pad_left, pad_top],
                dtype=xyxy.dtype,
            )
            xyxy = _clamp_xyxy(xyxy, out_w, out_h)
            target = _replace_boxes(target, xyxy, (out_h, out_w))

        if isinstance(target, dict):
            target = dict(target)
            target['letterbox_scale'] = torch.tensor([scale], dtype=torch.float32)
            target['letterbox_padding'] = torch.tensor(
                [pad_left, pad_top, pad_right, pad_bottom],
                dtype=torch.int64,
            )

        return (padded_image, target, *sample[2:])


@register()
class ConvertBoxes(T.Transform):
    _transformed_types = (
        BoundingBoxes,
    )
    def __init__(self, fmt='', normalize=False) -> None:
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)
    
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        spatial_size = getattr(inpt, _boxes_keys[1])
        if self.fmt:
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower())
            inpt = convert_to_tv_tensor(inpt, key='boxes', box_format=self.fmt.upper(), spatial_size=spatial_size)

        if self.normalize:
            inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2)[None]

        return inpt


@register()
class ConvertPILImage(T.Transform):
    _transformed_types = (
        PIL.Image.Image,
    )
    def __init__(self, dtype='float32', scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)
    
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt = F.pil_to_tensor(inpt)
        if self.dtype == 'float32':
            inpt = inpt.float()

        if self.scale:
            inpt = inpt / 255.

        inpt = Image(inpt)

        return inpt
