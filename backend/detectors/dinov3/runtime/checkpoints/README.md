# Checkpoints

ローカルで必要なcheckpointやpretrained weightsを配置する場所です。

必要ファイル:

- `detector/model_final.pth`
- `detector/last_checkpoint`
- `dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`
- `../two_stage_multiclass_20260426/outputs/roi_classifier_dinov3_rich_spatial_attn_no_expanded/run_20260426_213039/checkpoints/best.pt`

`.pth` / `.pt` などの重みファイルはGitに含めません。必要ならローカルに配置するか、symlinkを作成してください。
