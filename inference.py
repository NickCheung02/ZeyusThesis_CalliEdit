import os
import argparse

import torch
import transformers
from diffusers import FlowMatchEulerDiscreteScheduler

from models.adapter_models import *
from utils.sd3_utils import *
from utils.data_processor import UserInputProcessor


# inference arguments
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Generate Chinese landscapes with inscriptions.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None)
    parser.add_argument("--controlnet_model_name_or_path", type=str, default=None)
    parser.add_argument("--controlnet_model_name_or_path2", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None, required=False)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    parser.add_argument("--resolution_h", type=int, default=1024)
    parser.add_argument("--resolution_w", type=int, default=1024)

    # number of SD3 ControlNet Layers
    parser.add_argument("--ctrl_layers", type=int, default=23,help="control layers",)

    # inference
    parser.add_argument('--num_inference_steps', type=int, default=28)
    parser.add_argument("--cfg_scale", type=float, default=5.0, help="classifier-free guidance scale")
    parser.add_argument("--num_images_per_prompt", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument("--use_float16", action='store_true')

    parser.add_argument('--prompt', default='Chinese ink landscape painting, misty mountains, a river and pine trees, rice paper texture')
    parser.add_argument('--texts', default='[]', help='JSON inscription boxes in canvas pixel coordinates')
    parser.add_argument('--output_dir', default='./images/results')

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def load_pipeline(args):
    # load text encoders
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_3"
    )
    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
        args, text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three
    )
    # Load tokenizers
    tokenizer_one = transformers.CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = transformers.CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )
    tokenizer_three = transformers.T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_3",
        revision=args.revision,
    )

    # load vae
    vae = load_vae(args)
    # load sd3
    transformer = load_transfomer(args)
    # load scheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    from utils.checkpoint_utils import create_controlnet, load_text_checkpoint
    from models.wrapper_models import WrapperModel_SD3_ControlNet_with_Adapter
    if not args.controlnet_model_name_or_path or not args.controlnet_model_name_or_path2:
        raise ValueError('Scene and text checkpoint paths are required')
    controlnet_scene = create_controlnet(
        transformer, args.controlnet_model_name_or_path, num_layers=args.ctrl_layers
    )
    controlnet_text = create_controlnet(
        transformer, args.controlnet_model_name_or_path2, num_layers=12, text=True
    )
    adapter = LinearAdapterWithLayerNorm(128, 4096)
    load_text_checkpoint(
        WrapperModel_SD3_ControlNet_with_Adapter(controlnet_text, adapter),
        args.controlnet_model_name_or_path2,
    )

    # set device and dtype
    weight_dtype =  (torch.float16 if args.use_float16 and torch.cuda.is_available() else torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # move all models to device
    vae.to(device=device)
    text_encoder_one.to(device=device, dtype=weight_dtype)
    text_encoder_two.to(device=device, dtype=weight_dtype)
    text_encoder_three.to(device=device, dtype=weight_dtype)
    controlnet_scene.to(device=device, dtype=weight_dtype)
    controlnet_text.to(device=device, dtype=weight_dtype)
    adapter.to(device=device, dtype=weight_dtype)

    # load pipeline
    from pipelines.pipeline_sd3 import StableDiffusion3ControlNetPipeline
    pipeline = StableDiffusion3ControlNetPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler.from_config(
            noise_scheduler.config
            ),
        vae=vae,
        transformer=transformer,
        text_encoder=text_encoder_one,
        tokenizer=tokenizer_one,
        text_encoder_2=text_encoder_two,
        tokenizer_2=tokenizer_two,
        text_encoder_3=text_encoder_three,
        tokenizer_3=tokenizer_three,
        controlnet_scene=controlnet_scene,
        controlnet_text=controlnet_text,
        adapter=adapter,
    )

    pipeline = pipeline.to(dtype=weight_dtype, device=device)

    return pipeline


def generate(args, pipeline=None, processor=None):
    import json
    pipeline = pipeline if pipeline is not None else load_pipeline(args)
    processor = processor if processor is not None else UserInputProcessor()
    inputs = processor(json.loads(args.texts), args.prompt, args.resolution_w, args.resolution_h)
    generator = torch.Generator(device=pipeline._execution_device).manual_seed(args.seed)
    return pipeline(
        **inputs, height=args.resolution_h, width=args.resolution_w,
        num_inference_steps=args.num_inference_steps, guidance_scale=args.cfg_scale,
        num_images_per_prompt=args.num_images_per_prompt, generator=generator,
    ).images


if __name__ == '__main__':
    from pathlib import Path
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(generate(args)):
        image.save(output_dir / f'landscape_{args.seed}_{index}.png')
