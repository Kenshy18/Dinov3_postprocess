# DINOv3 Training Boundary

This directory owns DINOv3 Cascade training implementation code.

Stable compatibility entrypoint:

```bash
python scripts/train_dinov3_cascade_unified.py
```

Runtime code may import `scripts/train_dinov3_cascade_unified.py` for legacy
registration symbols such as `ATSSRPN`; the implementation lives here.
