"""Landscape generation UI. Models load only when generation is requested."""
import json
import os
import threading
from functools import lru_cache

import gradio as gr
from PIL import Image, ImageDraw

from inference import parse_args, load_pipeline, generate
from utils.data_processor import UserInputProcessor, validate_texts

_generation_lock = threading.Lock()


@lru_cache(maxsize=1)
def get_generator():
    args = parse_args([
        '--pretrained_model_name_or_path', './checkpoints/stable-diffusion-3-medium-diffusers',
        '--controlnet_model_name_or_path', os.environ.get('CALLIEDIT_SCENE_CHECKPOINT', './checkpoints/our_weights/scenegen_net.pth'),
        '--controlnet_model_name_or_path2', os.environ.get('CALLIEDIT_TEXT_CHECKPOINT', './checkpoints/our_weights/textrender_net.pth'),
        '--use_float16',
    ])
    return args, load_pipeline(args), UserInputProcessor()


def visualize_layout(texts_str, width, height):
    width, height = int(width), int(height)
    texts = validate_texts(json.loads(texts_str), width, height)
    canvas = Image.new('RGB', (width, height), '#f5f0e4')
    draw = ImageDraw.Draw(canvas)
    for index, text in enumerate(texts):
        draw.rectangle(text['pos'], outline='#9b3025', width=3)
        draw.text((text['pos'][0] + 4, text['pos'][1] + 4), str(index + 1), fill='#9b3025')
    return canvas


def generate_image(prompt, texts_str, width, height, seed, steps, cfg):
    # Serialize shared pipeline access, including its first initialization.
    with _generation_lock:
        validate_texts(json.loads(texts_str), int(width), int(height))
        defaults, pipeline, processor = get_generator()
        from copy import copy
        args = copy(defaults)
        args.prompt, args.texts = prompt, texts_str
        args.resolution_w, args.resolution_h = int(width), int(height)
        args.seed, args.num_inference_steps, args.cfg_scale = int(seed), int(steps), float(cfg)
        return generate(args, pipeline, processor)[0]


def build_app():
    with gr.Blocks() as iface:
        gr.Markdown('# 中国山水题字生成\n输入画面描述，以及题字内容和位置。竖排题字可按一字一框设置；空数组表示无题字。')
        prompt = gr.Textbox(label='画面描述', value='Chinese ink landscape painting, misty mountains, a river and pine trees, rice paper texture')
        texts = gr.Textbox(label='题字框 JSON（像素坐标）', value='[{"content":"山","pos":[850,120,930,220]},{"content":"静","pos":[850,240,930,340]}]', lines=5)
        with gr.Row():
            width = gr.Slider(256, 1536, value=1024, step=16, label='宽度')
            height = gr.Slider(256, 1536, value=1024, step=16, label='高度')
        with gr.Row():
            seed = gr.Number(value=42, precision=0, label='随机种子')
            steps = gr.Slider(1, 80, value=28, step=1, label='生成步数')
            cfg = gr.Slider(1, 15, value=5, label='提示词引导强度')
        preview = gr.Image(label='题字框布局（序号对应 JSON 顺序）')
        output = gr.Image(label='生成画作')
        gr.Button('预览布局').click(visualize_layout, [texts, width, height], preview)
        gr.Button('生成画作').click(generate_image, [prompt, texts, width, height, seed, steps, cfg], output)
    return iface


if __name__ == '__main__':
    build_app().queue().launch(server_name='0.0.0.0', server_port=7861)
