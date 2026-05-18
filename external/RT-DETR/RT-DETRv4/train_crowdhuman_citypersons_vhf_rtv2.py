#!/usr/bin/env python3
"""Launcher for RT-DETRv2 training on CrowdHuman + CityPersons VHF classes."""

import os
import sys
from datetime import datetime


os.environ.setdefault("PYTORCH_ALLOC_CONF", "backend:cudaMallocAsync")
os.environ.setdefault("WANDB_MODE", "online")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from engine.core import YAMLConfig
from engine.misc import dist_utils
from engine.solver import TASKS


CONFIG_PATH = "configs/rtv2/rtv2_r18vd_72e_crowdhuman_citypersons_vhf.yml"

TEST_ONLY = False
USE_AMP = True
SEED = 0
DEVICE = ""
PRINT_METHOD = "builtin"
PRINT_RANK = 0

RESUME_PATH = None
TUNING_PATH = None
CHECKPOINT_FREQ = 4

EPOCHES = 80
PRINT_FREQ = 50
TRAIN_TOTAL_BATCH_SIZE = 16
VAL_TOTAL_BATCH_SIZE = 16
TRAIN_NUM_WORKERS = 4
VAL_NUM_WORKERS = 4

DATA_ROOT = "./data/CrowdHumanCityPersons_VHF"
TRAIN_IMAGE_DIR = f"{DATA_ROOT}/images"
VAL_IMAGE_DIR = f"{DATA_ROOT}/images"
TRAIN_ANN_FILE = f"{DATA_ROOT}/annotations/instances_train_visiblebody_head_face.json"
VAL_ANN_FILE = f"{DATA_ROOT}/annotations/instances_val_visiblebody_head_face.json"

RUN_NAME = f"rtv2-r18-vhf-512x896-80e-bs{TRAIN_TOTAL_BATCH_SIZE}-{datetime.now():%Y%m%d-%H%M%S}"
OUTPUT_DIR = f"./outputs/pth/{RUN_NAME}"
SUMMARY_DIR = f"./outputs/tensorboard/{RUN_NAME}"

USE_WANDB = True
WANDB_PROJECT = "rtdetrv4-crowdhuman"
WANDB_ENTITY = None
WANDB_MODE = "online"
WANDB_LOG_VAL_IMAGES = True
WANDB_NUM_VAL_IMAGES = 4
WANDB_VAL_SCORE_THR = 0.35
WANDB_RUN_NAME = RUN_NAME


def build_overrides():
    return {
        "resume": RESUME_PATH,
        "tuning": TUNING_PATH,
        "device": DEVICE,
        "seed": SEED,
        "use_amp": USE_AMP,
        "test_only": TEST_ONLY,
        "print_method": PRINT_METHOD,
        "print_rank": PRINT_RANK,
        "checkpoint_freq": CHECKPOINT_FREQ,
        "epoches": EPOCHES,
        "print_freq": PRINT_FREQ,
        "output_dir": OUTPUT_DIR,
        "summary_dir": SUMMARY_DIR,
        "use_wandb": USE_WANDB,
        "wandb_project": WANDB_PROJECT,
        "wandb_entity": WANDB_ENTITY,
        "wandb_mode": WANDB_MODE,
        "wandb_log_val_images": WANDB_LOG_VAL_IMAGES,
        "wandb_num_val_images": WANDB_NUM_VAL_IMAGES,
        "wandb_val_score_thr": WANDB_VAL_SCORE_THR,
        "wandb_run_name": WANDB_RUN_NAME,
        "train_dataloader": {
            "total_batch_size": TRAIN_TOTAL_BATCH_SIZE,
            "num_workers": TRAIN_NUM_WORKERS,
            "dataset": {
                "img_folder": TRAIN_IMAGE_DIR,
                "ann_file": TRAIN_ANN_FILE,
            },
        },
        "val_dataloader": {
            "total_batch_size": VAL_TOTAL_BATCH_SIZE,
            "num_workers": VAL_NUM_WORKERS,
            "dataset": {
                "img_folder": VAL_IMAGE_DIR,
                "ann_file": VAL_ANN_FILE,
            },
        },
    }


def print_launch_summary():
    print("launch summary:")
    print(f"  config_path={CONFIG_PATH}")
    print(f"  output_dir={OUTPUT_DIR}")
    print(f"  train_image_dir={TRAIN_IMAGE_DIR}")
    print(f"  train_ann_file={TRAIN_ANN_FILE}")
    print(f"  val_image_dir={VAL_IMAGE_DIR}")
    print(f"  val_ann_file={VAL_ANN_FILE}")
    print(f"  epoches={EPOCHES}")
    print(f"  train_total_batch_size={TRAIN_TOTAL_BATCH_SIZE}")
    print(f"  val_total_batch_size={VAL_TOTAL_BATCH_SIZE}")
    print(f"  use_amp={USE_AMP}")
    print(f"  use_wandb={USE_WANDB}")
    print(f"  wandb_project={WANDB_PROJECT}")
    print(f"  wandb_run_name={WANDB_RUN_NAME}")
    print(f"  resume_path={RESUME_PATH}")
    print(f"  tuning_path={TUNING_PATH}")


def main():
    dist_utils.setup_distributed(PRINT_RANK, PRINT_METHOD, seed=SEED)

    if RESUME_PATH and TUNING_PATH:
        raise ValueError("Only one of RESUME_PATH or TUNING_PATH can be set.")

    print_launch_summary()
    cfg = YAMLConfig(CONFIG_PATH, **build_overrides())

    if RESUME_PATH or TUNING_PATH:
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    print("cfg: ", cfg.__dict__)

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    if TEST_ONLY:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()


if __name__ == "__main__":
    main()
