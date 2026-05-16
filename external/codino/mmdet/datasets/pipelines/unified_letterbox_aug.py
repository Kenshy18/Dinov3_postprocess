# Copyright (c) OpenMMLab. All rights reserved.
"""Unified training augmentations for the local one-class dataset.

These transforms mirror the Detectron2 letterbox pipeline used by
``scripts/train_dinov3_cascade_unified.py`` closely enough for Co-DINO smoke
tests while keeping the implementation inside MMDetection's pipeline system.
"""

import random

import cv2
import numpy as np

from mmdet.core import BitmapMasks
from mmdet.datasets.builder import PIPELINES


def _as_hw(size):
    if isinstance(size, int):
        return int(size), int(size)
    return int(size[0]), int(size[1])


def _bbox_from_binary_masks(masks):
    bboxes = []
    keep = []
    for idx, mask in enumerate(masks):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            continue
        bboxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        keep.append(idx)
    if bboxes:
        return np.asarray(bboxes, dtype=np.float32), np.asarray(keep, dtype=np.int64)
    return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)


def _filter_by_keep(results, keep, filter_bboxes=True):
    if filter_bboxes and 'gt_bboxes' in results:
        results['gt_bboxes'] = results['gt_bboxes'][keep]
    if 'gt_labels' in results:
        results['gt_labels'] = results['gt_labels'][keep]
    if 'gt_bboxes_ignore' in results and len(results['gt_bboxes_ignore']) > 0:
        # Ignore boxes are not mask-aligned. Leave them untouched.
        pass
    return results


@PIPELINES.register_module()
class UnifiedLetterboxResize:
    """Resize with preserved aspect ratio and pad to a fixed HxW canvas."""

    def __init__(self, target_size=(720, 1280), pad_val=128):
        self.target_h, self.target_w = _as_hw(target_size)
        self.pad_val = int(pad_val)

    def __call__(self, results):
        img = results['img']
        h, w = img.shape[:2]
        scale = min(self.target_h / h, self.target_w / w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        pad_h = self.target_h - new_h
        pad_w = self.target_w - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        border_value = (self.pad_val,) * img.shape[2] if img.ndim == 3 else self.pad_val
        padded = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=border_value)
        results['img'] = padded
        results['img_shape'] = padded.shape
        results['pad_shape'] = padded.shape
        results['scale_factor'] = np.array([scale, scale, scale, scale], dtype=np.float32)
        results['letterbox_meta'] = dict(
            scale=scale,
            pad_top=pad_top,
            pad_left=pad_left,
            new_h=new_h,
            new_w=new_w,
            target_h=self.target_h,
            target_w=self.target_w)

        for key in results.get('bbox_fields', []):
            bboxes = results[key]
            if bboxes.size == 0:
                continue
            bboxes[:, 0::2] = bboxes[:, 0::2] * scale + pad_left
            bboxes[:, 1::2] = bboxes[:, 1::2] * scale + pad_top
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, self.target_w)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, self.target_h)
            results[key] = bboxes

        for key in results.get('mask_fields', []):
            masks = results[key]
            if not isinstance(masks, BitmapMasks):
                masks = masks.to_bitmap()
            out = []
            for mask in masks.masks:
                mask_u8 = mask.astype(np.uint8)
                resized_mask = cv2.resize(mask_u8, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                padded_mask = cv2.copyMakeBorder(
                    resized_mask,
                    pad_top,
                    pad_bottom,
                    pad_left,
                    pad_right,
                    cv2.BORDER_CONSTANT,
                    value=0)
                out.append(padded_mask)
            if out:
                results[key] = BitmapMasks(np.stack(out), self.target_h, self.target_w)
            else:
                results[key] = BitmapMasks(np.zeros((0, self.target_h, self.target_w), dtype=np.uint8),
                                           self.target_h, self.target_w)
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(target_size=({self.target_h}, {self.target_w}), '
                f'pad_val={self.pad_val})')


@PIPELINES.register_module()
class UnifiedRandomRotate:
    """Random full-angle rotation before letterboxing.

    The Detectron2 pipeline used ``RandomRotation(angle=[-180, 180],
    expand=True)``. Here masks are rotated first and boxes are recomputed from
    the transformed masks, which is robust for the one-class instance data.
    """

    def __init__(self, prob=0.2, angle_range=(-180, 180), expand=True, pad_val=128):
        self.prob = float(prob)
        self.angle_range = tuple(angle_range)
        self.expand = bool(expand)
        self.pad_val = int(pad_val)

    def __call__(self, results):
        if np.random.rand() >= self.prob:
            return results

        img = results['img']
        h, w = img.shape[:2]
        angle = float(np.random.uniform(self.angle_range[0], self.angle_range[1]))
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        if self.expand:
            cos = abs(matrix[0, 0])
            sin = abs(matrix[0, 1])
            new_w = int(h * sin + w * cos)
            new_h = int(h * cos + w * sin)
            matrix[0, 2] += (new_w - w) * 0.5
            matrix[1, 2] += (new_h - h) * 0.5
        else:
            new_h, new_w = h, w

        border_value = (self.pad_val,) * img.shape[2] if img.ndim == 3 else self.pad_val
        rotated_img = cv2.warpAffine(
            img,
            matrix,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value)
        results['img'] = rotated_img
        results['img_shape'] = rotated_img.shape

        if 'gt_masks' in results:
            masks = results['gt_masks']
            if not isinstance(masks, BitmapMasks):
                masks = masks.to_bitmap()
            rotated_masks = []
            for mask in masks.masks:
                rotated = cv2.warpAffine(
                    mask.astype(np.uint8),
                    matrix,
                    (new_w, new_h),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0)
                rotated_masks.append(rotated)
            if rotated_masks:
                rotated_masks = np.stack(rotated_masks)
            else:
                rotated_masks = np.zeros((0, new_h, new_w), dtype=np.uint8)
            bboxes, keep = _bbox_from_binary_masks(rotated_masks)
            rotated_masks = rotated_masks[keep] if len(keep) else rotated_masks[:0]
            results['gt_masks'] = BitmapMasks(rotated_masks, new_h, new_w)
            results['gt_bboxes'] = bboxes
            _filter_by_keep(results, keep, filter_bboxes=False)
        elif 'gt_bboxes' in results:
            bboxes = results['gt_bboxes']
            if bboxes.size:
                corners = np.stack([
                    bboxes[:, [0, 1]], bboxes[:, [2, 1]],
                    bboxes[:, [0, 3]], bboxes[:, [2, 3]]
                ], axis=1)
                ones = np.ones((*corners.shape[:2], 1), dtype=np.float32)
                warped = np.concatenate([corners, ones], axis=-1) @ matrix.T
                x1 = np.clip(warped[:, :, 0].min(axis=1), 0, new_w)
                y1 = np.clip(warped[:, :, 1].min(axis=1), 0, new_h)
                x2 = np.clip(warped[:, :, 0].max(axis=1), 0, new_w)
                y2 = np.clip(warped[:, :, 1].max(axis=1), 0, new_h)
                new_boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
                keep = np.where((new_boxes[:, 2] > new_boxes[:, 0]) & (new_boxes[:, 3] > new_boxes[:, 1]))[0]
                results['gt_bboxes'] = new_boxes[keep]
                _filter_by_keep(results, keep)
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(prob={self.prob}, '
                f'angle_range={self.angle_range}, expand={self.expand})')


@PIPELINES.register_module()
class UnifiedPhotometricAug:
    """Photometric augmentations matching the current Detectron2 recipe."""

    def __init__(self,
                 brightness_prob=0.2,
                 dark_prob=0.25,
                 darker_prob=0.10,
                 contrast_prob=0.2,
                 low_contrast_prob=0.25,
                 gamma_prob=0.2,
                 saturation_prob=0.2,
                 lighting_prob=0.1,
                 clahe_prob=0.2,
                 motion_blur_prob=0.35):
        self.brightness_prob = brightness_prob
        self.dark_prob = dark_prob
        self.darker_prob = darker_prob
        self.contrast_prob = contrast_prob
        self.low_contrast_prob = low_contrast_prob
        self.gamma_prob = gamma_prob
        self.saturation_prob = saturation_prob
        self.lighting_prob = lighting_prob
        self.clahe_prob = clahe_prob
        self.motion_blur_prob = motion_blur_prob

    @staticmethod
    def _clip(img):
        return np.clip(img, 0, 255).astype(np.uint8)

    @staticmethod
    def _adjust_brightness(img, factor):
        return UnifiedPhotometricAug._clip(img.astype(np.float32) * factor)

    @staticmethod
    def _adjust_contrast(img, factor):
        mean = img.astype(np.float32).mean(axis=(0, 1), keepdims=True)
        return UnifiedPhotometricAug._clip((img.astype(np.float32) - mean) * factor + mean)

    @staticmethod
    def _adjust_gamma(img, gamma, gain=1.0):
        x = img.astype(np.float32) / 255.0
        y = gain * np.power(np.clip(x, 0.0, 1.0), gamma)
        return UnifiedPhotometricAug._clip(y * 255.0)

    @staticmethod
    def _adjust_saturation(img, factor):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= factor
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    @staticmethod
    def _apply_clahe(img, clip_limit, tile_grid):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        l_channel = clahe.apply(l_channel)
        return cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

    @staticmethod
    def _motion_kernel(ksize, angle_deg):
        ksize = max(3, int(ksize) | 1)
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        kernel[ksize // 2, :] = 1.0
        matrix = cv2.getRotationMatrix2D((ksize / 2 - 0.5, ksize / 2 - 0.5), angle_deg, 1.0)
        kernel = cv2.warpAffine(kernel, matrix, (ksize, ksize))
        return kernel / max(kernel.sum(), 1e-6)

    def __call__(self, results):
        img = results['img']
        if np.random.rand() < self.brightness_prob:
            img = self._adjust_brightness(img, np.random.uniform(0.8, 1.2))
        if np.random.rand() < self.dark_prob:
            img = self._adjust_brightness(img, np.random.uniform(0.4, 1.0))
        if np.random.rand() < self.darker_prob:
            img = self._adjust_brightness(img, np.random.uniform(0.3, 0.7))
        if np.random.rand() < self.contrast_prob:
            img = self._adjust_contrast(img, np.random.uniform(0.7, 1.3))
        if np.random.rand() < self.low_contrast_prob:
            img = self._adjust_contrast(img, np.random.uniform(0.4, 1.0))
        if np.random.rand() < self.gamma_prob:
            img = self._adjust_gamma(img, np.random.uniform(0.6, 0.95))
        if np.random.rand() < self.saturation_prob:
            img = self._adjust_saturation(img, np.random.uniform(0.7, 1.3))
        if np.random.rand() < self.lighting_prob:
            noise = np.random.normal(0, 0.7 * 255.0 / 10.0, size=(1, 1, img.shape[2]))
            img = self._clip(img.astype(np.float32) + noise)
        if np.random.rand() < self.clahe_prob:
            clip = float(np.random.uniform(1.8, 3.2))
            tile = random.choice(((4, 4), (8, 8), (12, 12)))
            img = self._apply_clahe(img, clip, tile)
        if np.random.rand() < self.motion_blur_prob:
            ksize = int(np.random.randint(7, 26))
            angle = float(np.random.uniform(-90, 90))
            img = cv2.filter2D(img, -1, self._motion_kernel(ksize, angle))
        results['img'] = img
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}()'
