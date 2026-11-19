#!/usr/bin/env bash
set -euo pipefail
python inference.py \
 --pretrained_model_name_or_path='./checkpoints/stable-diffusion-3-medium-diffusers' \
 --controlnet_model_name_or_path='./checkpoints/our_weights/scenegen_net.pth' \
 --controlnet_model_name_or_path2='./checkpoints/our_weights/textrender_net.pth' \
 --prompt='Chinese ink landscape painting, misty mountains, a river, rice paper texture' \
 --texts='[{"content":"山","pos":[850,120,930,220]},{"content":"静","pos":[850,240,930,340]}]' \
 --seed=42 --num_images_per_prompt=1 --use_float16
