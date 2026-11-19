"""CPU regressions for mask-free generation and inscription input."""
import unittest
from copy import deepcopy

import torch

from models.controlnet_sd3 import SD3ControlNetModel
from utils.data_processor import UserInputProcessor, validate_texts


class LandscapeTests(unittest.TestCase):
    def test_controlnet_uses_sixteen_channels(self):
        model = SD3ControlNetModel(
            sample_size=8, patch_size=2, in_channels=16, num_layers=1,
            attention_head_dim=8, num_attention_heads=2, joint_attention_dim=32,
            caption_projection_dim=16, pooled_projection_dim=16,
            pos_embed_max_size=8,
        )
        self.assertEqual(model.pos_embed_input.proj.in_channels, 16)
        latents = torch.randn(1, 16, 8, 8)
        result = model(
            hidden_states=latents, controlnet_cond=torch.zeros_like(latents),
            timestep=torch.tensor([1]), encoder_hidden_states=torch.randn(1, 4, 32),
            pooled_projections=torch.randn(1, 16), return_dict=False,
        )[0]
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, (1, 16, 16))

    def test_vertical_and_empty_inscriptions(self):
        self.assertEqual(validate_texts([], 512, 1024), [])
        processor = UserInputProcessor()
        texts = [{'content': '山', 'pos': [10, 20, 50, 200]}]
        original = deepcopy(texts)
        populated = processor(texts, 'ink mountains', 512, 1024)
        empty = processor([], 'ink mountains', 512, 1024)
        self.assertEqual(populated['text_embeds'].shape, (1, 112, 128))
        self.assertEqual(empty['text_embeds'].shape, (1, 112, 128))
        self.assertEqual(torch.count_nonzero(empty['text_embeds']).item(), 0)
        self.assertEqual(texts, original)
        self.assertEqual(set(populated), {'prompt', 'text_embeds'})

    def test_invalid_boxes_are_rejected(self):
        for pos in [[0, 0, 0, 10], [-1, 0, 10, 20], [0, 0, 600, 20], [0, 0, float('nan'), 20]]:
            with self.assertRaises(ValueError):
                validate_texts([{'content': '山', 'pos': pos}], 512, 512)


    def test_mask_free_pipeline_with_and_without_guidance(self):
        from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
        from models.transformer_sd3 import SD3Transformer2DModel
        from pipelines.pipeline_sd3 import StableDiffusion3ControlNetPipeline
        config=dict(sample_size=8,patch_size=2,in_channels=16,num_layers=1,attention_head_dim=8,num_attention_heads=2,joint_attention_dim=32,caption_projection_dim=16,pooled_projection_dim=16,pos_embed_max_size=8)
        t=SD3Transformer2DModel(**config)
        c=SD3ControlNetModel(**config)
        v=AutoencoderKL(in_channels=3,out_channels=3,down_block_types=('DownEncoderBlock2D','DownEncoderBlock2D'),up_block_types=('UpDecoderBlock2D','UpDecoderBlock2D'),block_out_channels=(8,8),norm_num_groups=4,latent_channels=16,shift_factor=0.0)
        p=StableDiffusion3ControlNetPipeline(transformer=t,vae=v,scheduler=FlowMatchEulerDiscreteScheduler(),controlnet_scene=c,controlnet_text=SD3ControlNetModel(**config),adapter=torch.nn.Linear(128,32))
        p.set_progress_bar_config(disable=True)
        for guidance in [1.,5.]:
         result=p(prompt_embeds=torch.randn(1,4,32),pooled_prompt_embeds=torch.randn(1,16),negative_prompt_embeds=torch.randn(1,4,32),negative_pooled_prompt_embeds=torch.randn(1,16),text_embeds=torch.zeros(1,112,128),height=16,width=16,num_inference_steps=1,guidance_scale=guidance,output_type='latent')
         self.assertEqual(result.images.shape, (1, 16, 8, 8))


if __name__ == '__main__':
    unittest.main()
