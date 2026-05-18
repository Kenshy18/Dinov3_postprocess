"""
CrowdHuman human-ROI training launcher for RT-DETRv4.

Edit the constants in this file instead of passing long CLI overrides.
This launcher still uses the same YAMLConfig -> Solver path as train.py,
so the training logic stays aligned with the upstream implementation.
"""

import os
import sys
from datetime import datetime

PYTORCH_ALLOC_CONF = 'backend:cudaMallocAsync'
os.environ.setdefault('PYTORCH_ALLOC_CONF', PYTORCH_ALLOC_CONF)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from engine.core import YAMLConfig
from engine.misc import dist_utils
from engine.solver import TASKS


CONFIG_PATH = 'configs/rtv4/rtv4_hgnetv2_x_crowdhuman_person.yml'

# Runtime mode
TEST_ONLY = False
USE_AMP = True
SEED = 0
DEVICE = ''
PRINT_METHOD = 'builtin'
PRINT_RANK = 0

# Checkpoint control
RESUME_PATH = None
TUNING_PATH = None
CHECKPOINT_FREQ = 1

# Training / logging
EPOCHES = 96
PRINT_FREQ = 40
TRAIN_TOTAL_BATCH_SIZE = 16
VAL_TOTAL_BATCH_SIZE = 16
TRAIN_NUM_WORKERS = 4
VAL_NUM_WORKERS = 4

# Dataset paths
TRAIN_IMAGE_DIR = './data/CrowdHuman/images/train'
VAL_IMAGE_DIR = './data/CrowdHuman/images/val'
TRAIN_ANN_FILE = './data/CrowdHuman/annotations/instances_train_fbox_person.json'
VAL_ANN_FILE = './data/CrowdHuman/annotations/instances_val_fbox_person.json'

# Output paths
RUN_NAME = 'rtv4_hgnetv2_x_crowdhuman_person'
OUTPUT_DIR = f'./outputs/pth/{RUN_NAME}'
SUMMARY_DIR = f'./outputs/tensorboard/{RUN_NAME}'

# Teacher paths
DINOv3_REPO_PATH = 'dinov3/'
DINOv3_WEIGHTS_PATH = 'pretrain/dinov3_vitb16_pretrain_lvd1689m.pth'

# W&B
USE_WANDB = True
WANDB_PROJECT = 'rtdetrv4-crowdhuman'
WANDB_ENTITY = None
WANDB_MODE = 'online'
WANDB_LOG_VAL_IMAGES = True
WANDB_NUM_VAL_IMAGES = 4
WANDB_VAL_SCORE_THR = 0.35
WANDB_RUN_NAME = f'rtv4-x-crowdhuman-bs{TRAIN_TOTAL_BATCH_SIZE}-{datetime.now():%Y%m%d-%H%M%S}'

# Memory debug
MEMORY_DEBUG_ENABLED = True
MEMORY_DEBUG_INTERVAL = 10
MEMORY_DEBUG_STAGES = [
    'after_data_to_device',
    'after_teacher',
    'after_forward',
    'after_loss',
    'after_backward',
    'after_optimizer',
    'oom',
]
MEMORY_DEBUG_LOG_FILE = None
MEMORY_DEBUG_TO_WANDB = False
MEMORY_DEBUG_SYNC_CUDA = True
MEMORY_DEBUG_RESET_PEAK_PER_ITER = False


def build_overrides():
    return {
        'resume': RESUME_PATH,
        'tuning': TUNING_PATH,
        'device': DEVICE,
        'seed': SEED,
        'use_amp': USE_AMP,
        'test_only': TEST_ONLY,
        'print_method': PRINT_METHOD,
        'print_rank': PRINT_RANK,
        'checkpoint_freq': CHECKPOINT_FREQ,
        'epoches': EPOCHES,
        'print_freq': PRINT_FREQ,
        'output_dir': OUTPUT_DIR,
        'summary_dir': SUMMARY_DIR,
        'use_wandb': USE_WANDB,
        'wandb_project': WANDB_PROJECT,
        'wandb_entity': WANDB_ENTITY,
        'wandb_mode': WANDB_MODE,
        'wandb_log_val_images': WANDB_LOG_VAL_IMAGES,
        'wandb_num_val_images': WANDB_NUM_VAL_IMAGES,
        'wandb_val_score_thr': WANDB_VAL_SCORE_THR,
        'wandb_run_name': WANDB_RUN_NAME,
        'memory_debug_enabled': MEMORY_DEBUG_ENABLED,
        'memory_debug_interval': MEMORY_DEBUG_INTERVAL,
        'memory_debug_stages': MEMORY_DEBUG_STAGES,
        'memory_debug_log_file': MEMORY_DEBUG_LOG_FILE,
        'memory_debug_to_wandb': MEMORY_DEBUG_TO_WANDB,
        'memory_debug_sync_cuda': MEMORY_DEBUG_SYNC_CUDA,
        'memory_debug_reset_peak_per_iter': MEMORY_DEBUG_RESET_PEAK_PER_ITER,
        'teacher_model': {
            'dinov3_repo_path': DINOv3_REPO_PATH,
            'dinov3_weights_path': DINOv3_WEIGHTS_PATH,
        },
        'train_dataloader': {
            'total_batch_size': TRAIN_TOTAL_BATCH_SIZE,
            'num_workers': TRAIN_NUM_WORKERS,
            'dataset': {
                'img_folder': TRAIN_IMAGE_DIR,
                'ann_file': TRAIN_ANN_FILE,
            },
        },
        'val_dataloader': {
            'total_batch_size': VAL_TOTAL_BATCH_SIZE,
            'num_workers': VAL_NUM_WORKERS,
            'dataset': {
                'img_folder': VAL_IMAGE_DIR,
                'ann_file': VAL_ANN_FILE,
            },
        },
    }


def print_launch_summary():
    print('launch summary:')
    print(f'  config_path={CONFIG_PATH}')
    print(f'  output_dir={OUTPUT_DIR}')
    print(f'  train_image_dir={TRAIN_IMAGE_DIR}')
    print(f'  train_ann_file={TRAIN_ANN_FILE}')
    print(f'  val_image_dir={VAL_IMAGE_DIR}')
    print(f'  val_ann_file={VAL_ANN_FILE}')
    print(f'  epoches={EPOCHES}')
    print(f'  print_freq={PRINT_FREQ}')
    print(f'  train_total_batch_size={TRAIN_TOTAL_BATCH_SIZE}')
    print(f'  val_total_batch_size={VAL_TOTAL_BATCH_SIZE}')
    print(f'  use_amp={USE_AMP}')
    print(f'  use_wandb={USE_WANDB}')
    print(f'  wandb_run_name={WANDB_RUN_NAME}')
    print(f'  memory_debug_enabled={MEMORY_DEBUG_ENABLED}')
    print(f'  memory_debug_interval={MEMORY_DEBUG_INTERVAL}')
    print(f'  memory_debug_stages={MEMORY_DEBUG_STAGES}')
    print(f'  resume_path={RESUME_PATH}')
    print(f'  tuning_path={TUNING_PATH}')


def main():
    dist_utils.setup_distributed(PRINT_RANK, PRINT_METHOD, seed=SEED)

    if RESUME_PATH and TUNING_PATH:
        raise ValueError('Only one of RESUME_PATH or TUNING_PATH can be set.')

    print_launch_summary()
    cfg = YAMLConfig(CONFIG_PATH, **build_overrides())

    if RESUME_PATH or TUNING_PATH:
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    print('cfg: ', cfg.__dict__)

    solver = TASKS[cfg.yaml_cfg['task']](cfg)

    if TEST_ONLY:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()


if __name__ == '__main__':
    main()
