# RT-DETR Head/Face Adapter

This boundary builds backend pipeline commands for the repo-local RT-DETRv4
runtime in `external/RT-DETR/RT-DETRv4`.

Runtime defaults:

```text
config:     external/RT-DETR/RT-DETRv4/configs/rtv2/rtv2_r18vd_72e_crowdhuman_citypersons_vhf.yml
checkpoint: checkpoints/rtdetr/head_face_best_stg1.pth
```

`tools/setup_runtime.sh` installs the RT-DETR Python dependencies, verifies the
source/config/checkpoint, benchmarks the fastest local batch size, and writes
the chosen values to `.runtime/runtime_profile.json` and
`.runtime/gui_runtime.env`.
