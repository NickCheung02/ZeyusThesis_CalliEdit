#!/usr/bin/env bash
set -euo pipefail
# Local data: Stage 1 inscription images; Stage 2 plain images with Description 4 captions.
stage="${1:-1}"
args=(
 --output_dir=/tmp --mixed_precision=fp16 --learning_rate=1e-4
 --num_train_epochs=50 --gradient_accumulation_steps=1
 --lr_warmup_steps=500 --lr_scheduler=constant_with_warmup
 --adam_epsilon=1e-15 --dataloader_num_workers=10
 --checkpointing_steps=1000 --validation_steps=1000
 --resolution_h=1024 --resolution_w=1024
 --pretrained_model_name_or_path=./checkpoints/stable-diffusion-3-medium-diffusers
 --ctrl_layers=12 --max_num_texts=7 --char_padding_to_len=16 --cfg_scale=5.0
)
case "$stage" in
 1)
  python train_sd3_stage1.py "${args[@]}" --name=calliedit_stage1 \
   --train_batch_size=4 --text_feature_drop=0.1 --p_drop_caption=0
  ;;
 2)
  python train_sd3_stage2.py "${args[@]}" --name=calliedit_stage2 \
   --train_batch_size=2 --text_feature_drop=0 \
   --controlnet_model_name_or_path2="${TEXT_CHECKPOINT:-/tmp/calliedit_stage1/0_net_calliedit_stage1.pth}"
  ;;
 *) echo 'Usage: bash train.sh {1|2}' >&2; exit 2 ;;
esac
