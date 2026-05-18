from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch

from . import dist_utils


DEFAULT_MEMORY_DEBUG_STAGES = (
    'after_data_to_device',
    'after_teacher',
    'after_forward',
    'after_loss',
    'after_backward',
    'after_optimizer',
    'oom',
)


class MemoryDebugTracker:
    def __init__(self, cfg, output_dir: Optional[Path] = None, wandb_logger=None) -> None:
        enabled = bool(getattr(cfg, 'memory_debug_enabled', False))
        self.enabled = enabled and torch.cuda.is_available() and dist_utils.is_main_process()
        self.interval = max(1, int(getattr(cfg, 'memory_debug_interval', 40) or 40))
        stages = getattr(cfg, 'memory_debug_stages', None) or DEFAULT_MEMORY_DEBUG_STAGES
        self.stages = set(stages)
        self.sync_cuda = bool(getattr(cfg, 'memory_debug_sync_cuda', True))
        self.reset_peak_per_iter = bool(getattr(cfg, 'memory_debug_reset_peak_per_iter', False))
        self.wandb_logger = wandb_logger if bool(getattr(cfg, 'memory_debug_to_wandb', False)) else None
        self.log_path = None

        if not self.enabled:
            return

        configured_path = getattr(cfg, 'memory_debug_log_file', None)
        if configured_path:
            self.log_path = Path(configured_path)
        elif output_dir is not None:
            self.log_path = Path(output_dir) / 'memory_debug.jsonl'
        else:
            self.log_path = Path('memory_debug.jsonl')
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f'Memory debug enabled. interval={self.interval}, stages={sorted(self.stages)}, log={self.log_path}')

    def start_iteration(self) -> None:
        if not self.enabled:
            return
        if self.reset_peak_per_iter:
            torch.cuda.reset_peak_memory_stats()

    def should_log(self, stage: str, step: int) -> bool:
        if not self.enabled or stage not in self.stages:
            return False
        return stage == 'oom' or step % self.interval == 0

    def record(self, stage: str, epoch: int, step: int, global_step: int, extra: Optional[Dict] = None):
        if not self.should_log(stage, step):
            return None

        if self.sync_cuda:
            torch.cuda.synchronize()

        free_b, total_b = torch.cuda.mem_get_info()
        allocated_b = torch.cuda.memory_allocated()
        reserved_b = torch.cuda.memory_reserved()
        max_allocated_b = torch.cuda.max_memory_allocated()
        max_reserved_b = torch.cuda.max_memory_reserved()
        used_b = total_b - free_b

        payload = {
            'epoch': int(epoch),
            'step': int(step),
            'global_step': int(global_step),
            'stage': stage,
            'device': torch.cuda.current_device(),
            'allocated_mb': round(allocated_b / 1024**2, 2),
            'reserved_mb': round(reserved_b / 1024**2, 2),
            'max_allocated_mb': round(max_allocated_b / 1024**2, 2),
            'max_reserved_mb': round(max_reserved_b / 1024**2, 2),
            'free_mb': round(free_b / 1024**2, 2),
            'total_mb': round(total_b / 1024**2, 2),
            'used_mb': round(used_b / 1024**2, 2),
            'cache_mb': round(max(reserved_b - allocated_b, 0) / 1024**2, 2),
            'non_torch_mb': round(max(used_b - reserved_b, 0) / 1024**2, 2),
            'used_pct': round((used_b / total_b) * 100.0, 2) if total_b else 0.0,
        }
        if extra:
            for key, value in extra.items():
                if isinstance(value, torch.Size):
                    payload[key] = list(value)
                elif isinstance(value, (list, tuple)):
                    payload[key] = list(value)
                elif isinstance(value, (int, float, str, bool)) or value is None:
                    payload[key] = value
                else:
                    payload[key] = str(value)

        if self.log_path is not None:
            with self.log_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(payload, ensure_ascii=True) + '\n')

        print(
            '[VRAM] '
            f"e{payload['epoch']} s{payload['step']} g{payload['global_step']} {stage} | "
            f"used={payload['used_mb']:.0f}/{payload['total_mb']:.0f}MB ({payload['used_pct']:.1f}%) | "
            f"alloc={payload['allocated_mb']:.0f}MB reserved={payload['reserved_mb']:.0f}MB | "
            f"cache={payload['cache_mb']:.0f}MB non_torch={payload['non_torch_mb']:.0f}MB | "
            f"peak_reserved={payload['max_reserved_mb']:.0f}MB"
        )

        if self.wandb_logger is not None and getattr(self.wandb_logger, 'enabled', False):
            wandb_payload = {'train/global_step': int(global_step)}
            for key, value in payload.items():
                if key in {'epoch', 'step', 'global_step', 'stage'}:
                    continue
                if isinstance(value, (int, float)):
                    wandb_payload[f'memory/{stage}/{key}'] = value
            self.wandb_logger.log(wandb_payload)

        return payload
