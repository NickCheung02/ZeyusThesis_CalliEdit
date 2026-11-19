"""Regression checks using small synthetic legacy checkpoints; no downloads."""
import tempfile
import unittest
from pathlib import Path

import torch

from models.adapter_models import LinearAdapterWithLayerNorm
from models.controlnet_sd3 import SD3ControlNetModel
from models.transformer_sd3 import SD3Transformer2DModel
from models.wrapper_models import WrapperModel_SD3_ControlNet_with_Adapter
from utils.checkpoint_utils import (
    CONDITION_WEIGHT, create_controlnet, load_scene_checkpoint, load_text_checkpoint,
)

CONFIG = dict(sample_size=8, patch_size=2, in_channels=16, num_layers=1,
              attention_head_dim=8, num_attention_heads=2, joint_attention_dim=32,
              caption_projection_dim=16, pooled_projection_dim=16, pos_embed_max_size=8)


class CheckpointCompatibilityTests(unittest.TestCase):
    def test_legacy_channel_conversion_and_forward_equivalence(self):
        legacy = SD3ControlNetModel(**CONFIG)
        old_conv = legacy.pos_embed_input.proj
        legacy.pos_embed_input.proj = torch.nn.Conv2d(17, old_conv.out_channels, 2, stride=2)
        # Non-zero residuals ensure this checks actual computation, not just zero outputs.
        with torch.no_grad():
            for block in legacy.controlnet_blocks:
                block.weight.normal_(std=0.1)
        original = legacy.state_dict()
        model = load_scene_checkpoint(SD3ControlNetModel(**CONFIG), original)
        for key, value in model.state_dict().items():
            expected = original[key][:, :16] if key == CONDITION_WEIGHT else original[key]
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
        self.assertEqual(original[CONDITION_WEIGHT].shape[1], 17)
        x, condition = torch.randn(1, 16, 8, 8), torch.randn(1, 16, 8, 8)
        common = dict(hidden_states=x, timestep=torch.tensor([1]),
                      encoder_hidden_states=torch.randn(1, 4, 32),
                      pooled_projections=torch.randn(1, 16), return_dict=False)
        old = legacy(controlnet_cond=torch.cat([condition, torch.zeros(1, 1, 8, 8)], 1), **common)[0]
        new = model(controlnet_cond=condition, **common)[0]
        torch.testing.assert_close(new[0], old[0])

    def test_new_weights_load_unchanged(self):
        source = SD3ControlNetModel(**CONFIG).state_dict()
        model = load_scene_checkpoint(SD3ControlNetModel(**CONFIG), source)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, source[key], rtol=0, atol=0)

    def test_other_missing_or_mismatched_parameters_fail(self):
        source = SD3ControlNetModel(**CONFIG).state_dict()
        for state in [dict(source, **{'context_embedder.weight': torch.zeros(1)}),
                      {k: v for k, v in source.items() if k != 'context_embedder.weight'},
                      dict(source, unexpected=torch.zeros(1))]:
            with self.assertRaises(RuntimeError):
                load_scene_checkpoint(SD3ControlNetModel(**CONFIG), state)

    def test_text_nested_and_wrapper_formats(self):
        def wrapper():
            return WrapperModel_SD3_ControlNet_with_Adapter(
                SD3ControlNetModel(**CONFIG), LinearAdapterWithLayerNorm(128, 32))
        source = wrapper()
        nested = {'controlnet_text': source.controlnet.state_dict(),
                  'adapter': source.adapter.state_dict()}
        for state in [nested, source.state_dict(),
                      {'state_dict': {'module.' + k: v for k, v in source.state_dict().items()}}]:
            model = load_text_checkpoint(wrapper(), state)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, source.state_dict()[key], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            load_text_checkpoint(wrapper(), {'controlnet_text': source.controlnet.state_dict()})

    def test_files_directories_and_automatic_depth(self):
        from safetensors.torch import save_file
        transformer = SD3Transformer2DModel(**dict(CONFIG, num_layers=2))
        source = SD3ControlNetModel(**CONFIG).state_dict()
        source[CONDITION_WEIGHT] = torch.cat([
            source[CONDITION_WEIGHT], torch.randn(16, 1, 2, 2)], dim=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            torch.save(source, path / 'legacy.pth')
            save_file(source, str(path / 'diffusion_pytorch_model.safetensors'))
            for checkpoint in [path / 'legacy.pth', path]:
                model = create_controlnet(transformer, checkpoint, num_layers=2)
                self.assertEqual(len(model.transformer_blocks), 1)
                self.assertEqual(model.pos_embed_input.proj.in_channels, 16)
            text = {'controlnet.' + k: v for k, v in SD3ControlNetModel(**CONFIG).state_dict().items()}
            text_model = create_controlnet(transformer, text, num_layers=2, text=True)
            self.assertEqual(len(text_model.transformer_blocks), 1)


if __name__ == '__main__':
    unittest.main()
