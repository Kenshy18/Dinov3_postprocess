"""
RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models
Copyright (c) 2025 The RT-DETRv4 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import sys
import math
from typing import Iterable

import torch
import torch.amp
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils

def _compute_encoder_transformer_grad_percentage(model: torch.nn.Module) -> float:
    """Compute percentage of gradients attributed to encoder transformer only.
    This avoids collecting/printing any other stats for speed.
    """
    total_l1 = 0.0
    enc_l1 = 0.0
    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None:
            continue
        val = grad.detach().abs().sum().item()
        total_l1 += val
        # Support both DDP ('module.') and non-DDP naming
        if name.startswith('module.encoder.encoder'):
            enc_l1 += val
    if total_l1 <= 0.0 or not math.isfinite(total_l1):
        return 0.0
    return 100.0 * enc_l1 / total_l1


def train_one_epoch(self_lr_scheduler, lr_scheduler, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    print_freq = kwargs.get('print_freq', 10)
    writer :SummaryWriter = kwargs.get('writer', None)
    wandb_logger = kwargs.get('wandb_logger', None)
    memory_debugger = kwargs.get('memory_debugger', None)

    ema :ModelEMA = kwargs.get('ema', None)
    scaler :GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler :Warmup = kwargs.get('lr_warmup_scheduler', None)

    # Gradient Analysis
    encoder_grad_percentages = []
    cur_iters = epoch * len(data_loader)

    teacher_model = kwargs.get('teacher_model', None)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        cpu_sample_shape = list(samples.shape)
        num_gt_boxes = 0
        for target in targets:
            labels = target.get('labels')
            if labels is not None:
                num_gt_boxes += int(len(labels))

        if memory_debugger is not None:
            memory_debugger.start_iteration()

        try:
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            sample_shape = list(samples.shape)
            memory_extra = {
                'batch_size': int(sample_shape[0]),
                'sample_shape': sample_shape,
                'cpu_sample_shape': cpu_sample_shape,
                'num_targets': len(targets),
                'num_gt_boxes': num_gt_boxes,
            }
            if memory_debugger is not None:
                memory_debugger.record('after_data_to_device', epoch, i, global_step, memory_extra)

            teacher_encoder_output_for_distillation = None
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_encoder_output_for_distillation = teacher_model(samples).detach()
                if memory_debugger is not None:
                    teacher_shape = list(teacher_encoder_output_for_distillation.shape)
                    memory_debugger.record('after_teacher', epoch, i, global_step, {**memory_extra, 'teacher_shape': teacher_shape})

            if scaler is not None:
                with torch.autocast(device_type=str(device), cache_enabled=True):
                    outputs = model(samples, targets=targets,
                                    teacher_encoder_output=teacher_encoder_output_for_distillation)
                if memory_debugger is not None:
                    memory_debugger.record('after_forward', epoch, i, global_step, memory_extra)

                if torch.isnan(outputs['pred_boxes']).any() or torch.isinf(outputs['pred_boxes']).any():
                    print(outputs['pred_boxes'])
                    state = model.state_dict()
                    new_state = {}
                    for key, value in model.state_dict().items():
                        new_key = key.replace('module.', '')
                        state[new_key] = value
                    new_state['model'] = state
                    dist_utils.save_on_master(new_state, "./NaN.pth")

                with torch.autocast(device_type=str(device), enabled=False):
                    loss_dict = criterion(outputs, targets, **metas)

                loss = sum(loss_dict.values())
                if memory_debugger is not None:
                    memory_debugger.record('after_loss', epoch, i, global_step, {**memory_extra, 'loss_value': float(loss.detach().item())})
                scaler.scale(loss).backward()
                if memory_debugger is not None:
                    memory_debugger.record('after_backward', epoch, i, global_step, memory_extra)

                if max_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                # Collect gradient
                if dist_utils.is_main_process() and hasattr(criterion, 'distill_adaptive_params') and \
                   getattr(criterion, 'distill_adaptive_params') and \
                   criterion.distill_adaptive_params.get('enabled', False):
                    pct = _compute_encoder_transformer_grad_percentage(model)
                    encoder_grad_percentages.append(pct)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if memory_debugger is not None:
                    memory_debugger.record('after_optimizer', epoch, i, global_step, memory_extra)

            else:
                outputs = model(samples, targets=targets,
                                teacher_encoder_output=teacher_encoder_output_for_distillation) # NEW kwarg
                if memory_debugger is not None:
                    memory_debugger.record('after_forward', epoch, i, global_step, memory_extra)
                loss_dict = criterion(outputs, targets, **metas)

                loss : torch.Tensor = sum(loss_dict.values())
                if memory_debugger is not None:
                    memory_debugger.record('after_loss', epoch, i, global_step, {**memory_extra, 'loss_value': float(loss.detach().item())})
                optimizer.zero_grad()
                loss.backward()
                if memory_debugger is not None:
                    memory_debugger.record('after_backward', epoch, i, global_step, memory_extra)

                # Collect gradient
                if dist_utils.is_main_process() and hasattr(criterion, 'distill_adaptive_params') and \
                   getattr(criterion, 'distill_adaptive_params') and \
                   criterion.distill_adaptive_params.get('enabled', False):
                    pct = _compute_encoder_transformer_grad_percentage(model)
                    encoder_grad_percentages.append(pct)

                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                optimizer.step()
                optimizer.zero_grad()
                if memory_debugger is not None:
                    memory_debugger.record('after_optimizer', epoch, i, global_step, memory_extra)
        except torch.OutOfMemoryError as exc:
            if memory_debugger is not None:
                memory_debugger.record('oom', epoch, i, global_step, {
                    'cpu_sample_shape': cpu_sample_shape,
                    'num_targets': len(targets),
                    'num_gt_boxes': num_gt_boxes,
                    'error': str(exc).split('\n')[0],
                })
            raise

        # ema
        if ema is not None:
            ema.update(model)

        if self_lr_scheduler:
            optimizer = lr_scheduler.step(cur_iters + i, optimizer)
        else:
            if lr_warmup_scheduler is not None:
                lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar('Loss/total', loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)

        if wandb_logger is not None and getattr(wandb_logger, 'enabled', False) and global_step % 10 == 0:
            wandb_logger.log_train_step(loss_value.item(), loss_dict_reduced, optimizer, global_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, encoder_grad_percentages


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader, coco_evaluator: CocoEvaluator, device, **kwargs):
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    wandb_logger = kwargs.get('wandb_logger', None)
    epoch = kwargs.get('epoch', None)
    category2name = getattr(data_loader.dataset, 'category2name', {})
    eval_images = []

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessor.keys())
    iou_types = coco_evaluator.iou_types
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        if all('letterbox_padding' in t and 'letterbox_scale' in t for t in targets):
            letterbox_padding = torch.stack([t['letterbox_padding'] for t in targets], dim=0)
            letterbox_scale = torch.stack([t['letterbox_scale'] for t in targets], dim=0).reshape(-1)
            input_sizes = torch.tensor(
                [[samples.shape[-1], samples.shape[-2]] for _ in range(samples.shape[0])],
                device=orig_target_sizes.device,
                dtype=orig_target_sizes.dtype,
            )
            results = postprocessor(
                outputs,
                orig_target_sizes,
                letterbox_padding=letterbox_padding,
                letterbox_scale=letterbox_scale,
                input_sizes=input_sizes,
            )
        else:
            results = postprocessor(outputs, orig_target_sizes)

        if wandb_logger is not None and getattr(wandb_logger, 'enabled', False) and wandb_logger.log_val_images and len(eval_images) < wandb_logger.num_val_images and dist_utils.is_main_process():
            current_target_sizes = torch.tensor(
                [[samples.shape[-1], samples.shape[-2]] for _ in range(samples.shape[0])],
                device=orig_target_sizes.device,
            )
            viz_results = postprocessor(outputs, current_target_sizes)
            for sample, target, prediction in zip(samples, targets, viz_results):
                if len(eval_images) >= wandb_logger.num_val_images:
                    break
                image = wandb_logger.build_detection_image(sample, target, prediction, category2name, epoch)
                if image is not None:
                    eval_images.append(image)

        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    if wandb_logger is not None and getattr(wandb_logger, 'enabled', False) and eval_images:
        wandb_logger.log_eval_images(eval_images, epoch if epoch is not None else 0)

    return stats, coco_evaluator
