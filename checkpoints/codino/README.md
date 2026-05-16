# Co-DINO checkpoints

Place DINOv3 + Co-DINO runtime artifacts here when running without the external
training workspace fallback:

```text
checkpoints/codino/
  detector/resolved_config.py
  detector/epoch_2.pth
  classifier/best.pt
  trt/
    codino_dinov3_vitl_backbone_736x1280_fp32_b2_fixed_bf16.engine
    codino_query_encoder_b2_736x1280_msda_plugin_sbc_fp16.engine
    codino_decoder_b2_736x1280_msda_plugin_fp16.engine
    codino_mask_head_core_n1_736x1280_fp16.engine
```

The integrated pipeline can also use the canonical training workspace under
`/home/kenke/workspace/CV/unified_training_codino_eva02`; local files here take
precedence when present.
