# Heavy Data Policy

## Canonical layout

- `outputs/pth/<run>/`: checkpoints and framework-owned training artifacts
- `outputs/tensorboard/<run>/`: tensorboard summaries
- `outputs/meta/<run>/`: text logs, memory debug traces, and other lightweight run metadata
- `outputs/json/<run>/`: evaluation summaries and manifest-style JSON
- `outputs/video/<run>/`: exported inference or visualization videos

## Patched launcher

- `train_crowdhuman_rtv4.py`

## Local-only heavy folders

- `Crowdhuman/`
- `data/`
- `dinov3/`
- `pretrain/`
- `input/`
- `outputs/`
- `output/`
- `checkpoints/`
- `tensorboard/`
- `video/`
- `image/`
- `json/`
- `csv/`
- `meta/`
- `wandb/`
- `.venv-tools/`
- `.venv/`
- `.serena/`

## Git ignore targets

- `Crowdhuman/`
- `data/`
- `dinov3/`
- `pretrain/`
- `input/`
- `outputs/`
- `output/`
- `checkpoints/`
- `tensorboard/`
- `video/`
- `image/`
- `json/`
- `csv/`
- `meta/`
- `wandb/`
- `.venv-tools/`
- `.venv/`
- `.serena/`
- `*.pth`
- `*.pt`
- `*.ckpt`
- `*.bin`
- `*.json`
- `*.jsonl`
- `*.mp4`
- `*.log`
- `*.txt`

## Open items

- `train.py` and older configs still accept generic `output_dir` values, but the Crowdhuman launcher now targets the typed layout and the solvers write logs into `outputs/meta/<run>`.
- Dataset and pretrained-weight placement remain local-only and ignored.
