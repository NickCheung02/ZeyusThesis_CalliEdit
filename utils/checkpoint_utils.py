"""Strict loading of PosterMaker and CalliEdit ControlNet checkpoints.

The only tensor conversion is removal of the final preservation-mask channel.
Source files and source state dictionaries are never modified.
"""
import logging
import re
from collections.abc import Mapping
from pathlib import Path

import torch

logger = logging.getLogger(__name__)
CONDITION_WEIGHT = 'pos_embed_input.proj.weight'


def read_checkpoint(source):
    if isinstance(source, Mapping):
        state = dict(source)
    else:
        path = Path(source)
        if path.is_dir():
            candidates = [path / 'diffusion_pytorch_model.safetensors',
                          path / 'diffusion_pytorch_model.bin']
            path = next((p for p in candidates if p.is_file()), None)
            if path is None:
                raise ValueError(f'No single-file Diffusers weights found in {source}')
        if path.suffix == '.safetensors':
            from safetensors.torch import load_file
            state = load_file(str(path), device='cpu')
        else:
            state = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(state, Mapping):
        raise ValueError('Checkpoint must contain a state dictionary')
    if 'state_dict' in state:
        state = state['state_dict']
    if not isinstance(state, Mapping) or not state:
        raise ValueError('Checkpoint state dictionary is empty or invalid')
    state = dict(state)
    if all(k.startswith('module.') for k in state):
        state = {k[len('module.'):]: v for k, v in state.items()}
    return state


def scene_state(source):
    state = read_checkpoint(source)
    for key in ('controlnet_scene', 'controlnet_inpaint'):
        if key in state:
            return read_checkpoint(state[key])
    return state


def text_state(source):
    state = read_checkpoint(source)
    if 'controlnet_text' in state or 'adapter' in state:
        if set(state) != {'controlnet_text', 'adapter'}:
            raise ValueError('Text checkpoint must contain both controlnet_text and adapter only')
        state = {**{'controlnet.' + k: v for k, v in state['controlnet_text'].items()},
                 **{'adapter.' + k: v for k, v in state['adapter'].items()}}
    return state


def checkpoint_layers(state, prefix=''):
    indices = {int(m.group(1)) for key in state
               if (m := re.fullmatch(re.escape(prefix) + r'transformer_blocks\.(\d+)\..+', key))}
    if not indices or indices != set(range(max(indices) + 1)):
        raise ValueError('Checkpoint has missing or non-contiguous Transformer blocks')
    return len(indices)


def load_scene_checkpoint(model, source):
    state = scene_state(source)
    expected = model.state_dict()[CONDITION_WEIGHT]
    weight = state.get(CONDITION_WEIGHT)
    if isinstance(weight, torch.Tensor) and weight.ndim == 4:
        if (weight.shape[1] == expected.shape[1] + 1
                and weight.shape[0] == expected.shape[0]
                and weight.shape[2:] == expected.shape[2:]):
            state[CONDITION_WEIGHT] = weight[:, :-1].clone()
            logger.warning('Converted legacy scene condition weights from %s to %s; removed final mask channel',
                           tuple(weight.shape), tuple(expected.shape))
    # Every other key and tensor shape must match. Never suppress missing keys.
    model.load_state_dict(state, strict=True)
    return model


def load_text_checkpoint(wrapper, source):
    wrapper.load_state_dict(text_state(source), strict=True)
    return wrapper


def create_controlnet(transformer, source=None, num_layers=12, *, text=False):
    from models.controlnet_sd3 import SD3ControlNetModel
    if source is None:
        return SD3ControlNetModel.from_transformer(transformer, num_layers=num_layers)
    state = text_state(source) if text else scene_state(source)
    layers = checkpoint_layers(state, 'controlnet.' if text else '')
    if layers > len(transformer.transformer_blocks):
        raise ValueError(f'Checkpoint has {layers} blocks, exceeding the base Transformer depth')
    if layers != num_layers:
        logger.warning('Using checkpoint depth %s instead of configured depth %s', layers, num_layers)
    # Construct from the current base architecture, never restore the old mask config.
    model = SD3ControlNetModel.from_transformer(transformer, num_layers=layers)
    if not text:
        load_scene_checkpoint(model, state)
    return model
