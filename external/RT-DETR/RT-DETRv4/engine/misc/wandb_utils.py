import os
from typing import Dict, List, Optional

import torch
from torchvision.utils import draw_bounding_boxes

from . import dist_utils


class WandbLogger:
    COCO_BBOX_METRICS = [
        'AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
        'AR_1', 'AR_10', 'AR_100', 'AR_small', 'AR_medium', 'AR_large'
    ]

    def __init__(self, cfg) -> None:
        self.enabled = False
        self.run = None
        self.wandb = None
        self.num_val_images = int(getattr(cfg, 'wandb_num_val_images', 4) or 4)
        self.val_score_thr = float(getattr(cfg, 'wandb_val_score_thr', 0.35) or 0.35)
        self.log_val_images = bool(getattr(cfg, 'wandb_log_val_images', True))

        if not getattr(cfg, 'use_wandb', False) or not dist_utils.is_main_process():
            return

        import wandb

        init_kwargs = {
            'project': getattr(cfg, 'wandb_project', None) or 'rtdetrv4',
            'entity': getattr(cfg, 'wandb_entity', None),
            'name': getattr(cfg, 'wandb_run_name', None),
            'mode': getattr(cfg, 'wandb_mode', None) or os.getenv('WANDB_MODE', 'online'),
            'config': getattr(cfg, 'yaml_cfg', None),
        }
        init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

        try:
            self.run = wandb.init(**init_kwargs)
        except Exception as exc:
            if init_kwargs.get('mode') == 'online':
                print(f'wandb online init failed ({exc}); falling back to offline mode.')
                init_kwargs['mode'] = 'offline'
                self.run = wandb.init(**init_kwargs)
            else:
                raise

        self.enabled = self.run is not None
        self.wandb = wandb
        if self.enabled:
            self.wandb.define_metric('train/global_step')
            self.wandb.define_metric('train/*', step_metric='train/global_step')
            self.wandb.define_metric('epoch')
            self.wandb.define_metric('val/*', step_metric='epoch')

    def log(self, payload: Dict, step: Optional[int] = None, commit: bool = True):
        if not self.enabled or not payload:
            return
        self.wandb.log(payload, step=step, commit=commit)

    def log_train_step(self, loss_value, loss_dict_reduced, optimizer, global_step: int):
        if not self.enabled:
            return
        payload = {
            'train/global_step': int(global_step),
            'train/loss_total': float(loss_value),
        }
        for j, pg in enumerate(optimizer.param_groups):
            payload[f'train/lr_pg_{j}'] = float(pg['lr'])
        for key, value in loss_dict_reduced.items():
            payload[f'train/{key}'] = float(value.item() if hasattr(value, 'item') else value)
        self.log(payload)

    def log_eval_epoch(self, test_stats: Dict, epoch: int):
        if not self.enabled:
            return

        payload = {'epoch': epoch}
        for key, value in test_stats.items():
            if isinstance(value, list):
                if key == 'coco_eval_bbox':
                    for metric_name, metric_value in zip(self.COCO_BBOX_METRICS, value):
                        payload[f'val/{metric_name}'] = float(metric_value)
                else:
                    for idx, metric_value in enumerate(value):
                        payload[f'val/{key}_{idx}'] = float(metric_value)
            else:
                payload[f'val/{key}'] = float(value)

        self.log(payload)

    def build_detection_image(self, image: torch.Tensor, target: Dict, prediction: Dict, category2name: Optional[Dict] = None, epoch: Optional[int] = None):
        if not self.enabled:
            return None

        category2name = category2name or {}
        image = image.detach().cpu()
        if image.dtype != torch.uint8:
            image = (image.clamp(0, 1) * 255).to(torch.uint8)

        rendered = image
        gt_boxes = target.get('boxes')
        if gt_boxes is not None and len(gt_boxes) > 0:
            rendered = draw_bounding_boxes(rendered, gt_boxes.detach().cpu(), colors='green', width=2)

        pred_boxes = prediction['boxes'].detach().cpu()
        pred_scores = prediction['scores'].detach().cpu()
        pred_labels = prediction['labels'].detach().cpu()
        keep = pred_scores >= self.val_score_thr
        keep_idx = torch.where(keep)[0][:30]
        if len(keep_idx) > 0:
            labels = []
            for idx in keep_idx.tolist():
                label_id = int(pred_labels[idx].item())
                label_name = category2name.get(label_id, str(label_id))
                labels.append(f'{label_name} {pred_scores[idx].item():.2f}')
            rendered = draw_bounding_boxes(
                rendered,
                pred_boxes[keep_idx],
                labels=labels,
                colors='red',
                width=2,
            )

        image_id = int(target['image_id'].flatten()[0].item()) if 'image_id' in target else -1
        gt_count = int(len(gt_boxes)) if gt_boxes is not None else 0
        pred_count = int(len(keep_idx))
        caption = f'epoch={epoch} image_id={image_id} gt={gt_count} pred@{self.val_score_thr:.2f}={pred_count}'
        return self.wandb.Image(rendered.permute(1, 2, 0).numpy(), caption=caption)

    def log_eval_images(self, images: List, epoch: int):
        if not self.enabled or not images:
            return
        self.log({'epoch': epoch, 'val/predictions': images})

    def finish(self):
        if self.enabled and self.run is not None:
            self.run.finish()
