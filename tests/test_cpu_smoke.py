"""Small real models exercise CPU decoding and trainable scene gradients."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler

from models.adapter_models import LinearAdapterWithLayerNorm
from models.controlnet_sd3 import SD3ControlNetModel
from models.transformer_sd3 import SD3Transformer2DModel
from models.wrapper_models import WrapperModel_SD3_ControlNet_with_Adapter
from pipelines.pipeline_sd3 import StableDiffusion3ControlNetPipeline
from utils.checkpoint_utils import CONDITION_WEIGHT, create_controlnet, load_text_checkpoint
from utils.data_processor import UserInputProcessor

CONFIG = dict(sample_size=8, patch_size=2, in_channels=16, num_layers=1,
              attention_head_dim=8, num_attention_heads=2, joint_attention_dim=32,
              caption_projection_dim=16, pooled_projection_dim=16, pos_embed_max_size=8)


class CPUSmokeTests(unittest.TestCase):
    def test_legacy_weights_to_png_on_cpu(self):
        transformer = SD3Transformer2DModel(**dict(CONFIG, num_layers=2)).to('cpu')
        legacy = SD3ControlNetModel(**CONFIG).state_dict()
        legacy[CONDITION_WEIGHT] = torch.cat([
            legacy[CONDITION_WEIGHT], torch.zeros(16, 1, 2, 2)], dim=1)
        text = SD3ControlNetModel(**CONFIG)
        adapter = LinearAdapterWithLayerNorm(128, 32)
        vae = AutoencoderKL(
            in_channels=3, out_channels=3,
            down_block_types=('DownEncoderBlock2D', 'DownEncoderBlock2D'),
            up_block_types=('UpDecoderBlock2D', 'UpDecoderBlock2D'),
            block_out_channels=(8, 8), norm_num_groups=4,
            latent_channels=16, shift_factor=0.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            torch.save(legacy, folder / 'scene.pth')
            torch.save({'controlnet_text': text.state_dict(), 'adapter': adapter.state_dict()},
                       folder / 'text.pth')
            scene = create_controlnet(transformer, folder / 'scene.pth', num_layers=1)
            load_text_checkpoint(WrapperModel_SD3_ControlNet_with_Adapter(text, adapter), folder / 'text.pth')
            pipeline = StableDiffusion3ControlNetPipeline(
                transformer=transformer, vae=vae, scheduler=FlowMatchEulerDiscreteScheduler(),
                controlnet_scene=scene, controlnet_text=text, adapter=adapter,
            ).to(device='cpu', dtype=torch.float32)
            pipeline.set_progress_bar_config(disable=True)
            self.assertEqual(pipeline._execution_device.type, 'cpu')
            # Use the real local character dictionary, but precomputed tiny scene embeddings.
            processor = UserInputProcessor()
            prompt = torch.randn(1, 4, 32)
            pooled = torch.randn(1, 16)
            for name, guidance, boxes in [
                ('empty', 1., []),
                ('inscription', 5., [{'content': '山', 'pos': [2, 2, 6, 14]}]),
            ]:
                with self.subTest(name=name):
                    inputs = processor(boxes, '', width=16, height=16)
                    images = pipeline(
                        prompt_embeds=prompt, pooled_prompt_embeds=pooled,
                        negative_prompt_embeds=torch.zeros_like(prompt),
                        negative_pooled_prompt_embeds=torch.zeros_like(pooled),
                        text_embeds=inputs['text_embeds'], height=16, width=16,
                        num_inference_steps=2, guidance_scale=guidance,
                        generator=torch.Generator(device='cpu').manual_seed(42),
                        output_type='pil',
                    ).images
                    self.assertEqual(len(images), 1)
                    pixels = np.asarray(images[0])
                    self.assertEqual(pixels.shape, (16, 16, 3))
                    self.assertGreater(float(pixels.std()), 0)
                    path = folder / f'{name}.png'
                    images[0].save(path)
                    with Image.open(path) as saved:
                        saved.verify()

    def test_scene_backward_on_cpu(self):
        scene = SD3ControlNetModel(**CONFIG).to('cpu')
        transformer = SD3Transformer2DModel(**dict(CONFIG, num_layers=2)).to('cpu').requires_grad_(False)
        latent = torch.randn(1, 16, 8, 8)
        timestep = torch.tensor([1])
        prompt, pooled = torch.randn(1, 4, 32), torch.randn(1, 16)
        residuals = scene(
            hidden_states=latent, controlnet_cond=torch.zeros_like(latent),
            timestep=timestep, encoder_hidden_states=prompt,
            pooled_projections=torch.zeros_like(pooled), return_dict=False,
        )[0]
        prediction = transformer(
            hidden_states=latent, timestep=timestep, encoder_hidden_states=prompt,
            pooled_projections=pooled, block_controlnet_hidden_states=residuals,
            return_dict=False,
        )[0]
        loss = (prediction - torch.randn_like(prediction)).square().mean()
        loss.backward()
        gradients = [p.grad for p in scene.parameters() if p.grad is not None]
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(gradients)
        self.assertTrue(all(g.device.type == 'cpu' and torch.isfinite(g).all() for g in gradients))
        self.assertTrue(any(torch.count_nonzero(g).item() > 0 for g in gradients))
        self.assertTrue(all(p.grad is None for p in transformer.parameters()))


if __name__ == '__main__':
    unittest.main()
