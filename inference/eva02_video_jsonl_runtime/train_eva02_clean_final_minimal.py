#!/usr/bin/env python
"""
Minimal extract from train_eva02_clean_final.py for inference only
Only contains Config, LetterboxTransform, compute_letterbox_params, unletterbox_instances
"""
import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List

# Check if CV2 is available
CV2_AVAILABLE = True
try:
    import cv2
except ImportError:
    CV2_AVAILABLE = False

# Import from detectron2
from detectron2.data.transforms import Transform
from detectron2.structures import ROIMasks

@dataclass
class Config:
    """Minimal config for inference"""
    # Model
    model_size: str = "eva02_L_8attn_1280"
    backbone_checkpoint: str = ""
    
    # Dataset
    class_name: str = "censorship"
    
    # Inference
    train_size: int = 1280
    test_size: int = 1280
    
    # Output
    output_dir: str = "./output_inference"

class LetterboxTransform(Transform):
    """
    Transform that applies letterbox resize and padding.
    """
    
    def __init__(self, src_shape, target_size, pad_value=128):
        super().__init__()
        h, w = src_shape[:2]
        
        # Calculate scale to fit image within target_size
        self.scale = min(target_size / h, target_size / w)
        self.new_h = int(h * self.scale)
        self.new_w = int(w * self.scale)
        
        # Calculate padding
        pad_h = target_size - self.new_h
        pad_w = target_size - self.new_w
        self.pad_top = pad_h // 2
        self.pad_bottom = pad_h - self.pad_top
        self.pad_left = pad_w // 2
        self.pad_right = pad_w - self.pad_left
        
        self.target_size = target_size
        self.pad_value = pad_value
        self.h_orig = h
        self.w_orig = w
        
        self._set_attributes(locals())
    
    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """Apply letterbox transform to image."""
        # Resize image
        if CV2_AVAILABLE:
            resized = cv2.resize(img, (self.new_w, self.new_h), interpolation=cv2.INTER_LINEAR)
        else:
            from PIL import Image
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize((self.new_w, self.new_h), Image.BILINEAR)
            resized = np.array(pil_img)
        
        # Add padding
        if CV2_AVAILABLE:
            if len(img.shape) == 2:
                padded = cv2.copyMakeBorder(
                    resized, self.pad_top, self.pad_bottom, self.pad_left, self.pad_right,
                    cv2.BORDER_CONSTANT, value=self.pad_value
                )
            else:
                padded = cv2.copyMakeBorder(
                    resized, self.pad_top, self.pad_bottom, self.pad_left, self.pad_right,
                    cv2.BORDER_CONSTANT, 
                    value=(self.pad_value, self.pad_value, self.pad_value)
                )
        else:
            # Numpy fallback
            if len(resized.shape) == 2:
                pad_width = ((self.pad_top, self.pad_bottom), (self.pad_left, self.pad_right))
                padded = np.pad(resized, pad_width, mode='constant', constant_values=self.pad_value)
            else:
                pad_width = ((self.pad_top, self.pad_bottom), (self.pad_left, self.pad_right), (0, 0))
                padded = np.pad(resized, pad_width, mode='constant', constant_values=self.pad_value)
        
        return padded
    
    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """Apply transform to coordinates."""
        coords = coords.astype(np.float32)
        coords[:, 0] = coords[:, 0] * self.scale + self.pad_left  # x
        coords[:, 1] = coords[:, 1] * self.scale + self.pad_top   # y
        return coords
    
    def apply_box(self, box: np.ndarray) -> np.ndarray:
        """Apply transform to bounding box."""
        # box is in [x1, y1, x2, y2] format
        coords = box.reshape(-1, 2)
        coords = self.apply_coords(coords)
        return coords.reshape(box.shape)

def compute_letterbox_params(h, w, target):
    """
    Compute letterbox transformation parameters.
    
    Args:
        h: Original height
        w: Original width
        target: Target size (square)
    
    Returns:
        dict with scale, new_h, new_w, pad_top, pad_left
    """
    s = min(target / h, target / w)
    new_h = int(round(h * s))
    new_w = int(round(w * s))
    pad_top = (target - new_h) // 2
    pad_left = (target - new_w) // 2
    return dict(scale=s, new_h=new_h, new_w=new_w, pad_top=pad_top, pad_left=pad_left)

def unletterbox_instances(instances, lb, orig_h, orig_w, target_size=1280):
    """
    Remove letterbox padding and scale from predicted instances.
    
    Args:
        instances: Detectron2 Instances object with predictions
        lb: Letterbox parameters dict (scale, pad_top, pad_left, new_h, new_w)
        orig_h: Original image height
        orig_w: Original image width
        target_size: Target letterbox size (default: 1280)
    
    Returns:
        Modified instances with correct coordinates for original image
    """
    use_inplace_boxes = os.environ.get("EVA_UNLETTERBOX_BOX_INPLACE", "0") == "1"
    use_raw_to_orig_masks = os.environ.get("EVA_RAW_TO_ORIG_MASK_POSTPROCESS", "0") == "1"

    # Shift boxes to remove padding
    boxes = (
        instances.pred_boxes.tensor
        if use_inplace_boxes
        else instances.pred_boxes.tensor.detach().clone()
    )
    boxes[:, [0, 2]] -= lb["pad_left"]  # x coordinates
    boxes[:, [1, 3]] -= lb["pad_top"]   # y coordinates
    
    # Scale back to original size
    boxes /= lb["scale"]
    
    # Clamp to image bounds
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, orig_w - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, orig_h - 1)
    if not use_inplace_boxes:
        instances.pred_boxes.tensor = boxes
    
    # Handle masks if present
    if instances.has("pred_masks"):
        masks = instances.pred_masks
        mask_device = masks.device if isinstance(masks, torch.Tensor) else boxes.device
        
        # Check if masks is not empty
        if len(masks) > 0:
            # Check mask dimensions (could be 28x28 from RoI or full size)
            if len(masks.shape) == 4:
                _, _, mask_h, mask_w = masks.shape
            elif len(masks.shape) == 3:
                _, mask_h, mask_w = masks.shape
            else:
                mask_h, mask_w = masks.shape[1:]
            
            # Only process if mask size matches letterbox size (not RoI masks)
            if mask_h == target_size and mask_w == target_size:
                # Remove padding from masks (full-size masks)
                if len(masks.shape) == 3:
                    masks_4d = masks[:, None, :, :].float()
                else:
                    masks_4d = masks.float()
                masks_4d = masks_4d[
                    :,
                    :,
                    lb["pad_top"]:lb["pad_top"] + lb["new_h"],
                    lb["pad_left"]:lb["pad_left"] + lb["new_w"],
                ]
                
                # Check if masks are not empty after cropping
                if masks_4d.shape[2] > 0 and masks_4d.shape[3] > 0:
                    masks_resized = F.interpolate(
                        masks_4d,
                        size=(orig_h, orig_w),
                        mode="nearest",
                    )
                    instances.pred_masks = masks_resized[:, 0] > 0.5
                else:
                    # Cropping resulted in empty masks
                    instances.pred_masks = torch.zeros((len(masks), orig_h, orig_w), dtype=torch.bool, device=mask_device)
            else:
                # Optionally fuse raw ROI-mask paste directly into original-resolution
                # masks, bypassing detector_postprocess(target_size)->crop->resize.
                if use_raw_to_orig_masks:
                    if len(masks.shape) == 4:
                        roi_masks = masks[:, 0, :, :].float()
                    else:
                        roi_masks = masks.float()
                    instances.pred_masks = ROIMasks(roi_masks).to_bitmasks(
                        instances.pred_boxes,
                        int(orig_h),
                        int(orig_w),
                        0.5,
                    ).tensor
        else:
            # If originally no masks, keep empty tensor with correct size
            instances.pred_masks = torch.zeros((0, orig_h, orig_w), dtype=torch.bool, device=mask_device)
    
    return instances
