import gc
import subprocess
from pathlib import Path

import torch
from diffusers import DDIMScheduler
from models.dift.CustomUNet2D import CustomUNet2D
from models.dift.OneStepSDPipeline import OneStepSDPipeline


class SDFeaturizer:
    def __init__(self, sd_id='Manojb/stable-diffusion-2-1-base', device: torch.device = torch.device('cpu')):
        self.device = device
        unet = CustomUNet2D.from_pretrained(sd_id, subfolder="unet")
        gc.collect()
        onestep_pipe = OneStepSDPipeline.from_pretrained(sd_id, unet=unet, safety_checker=None)
        onestep_pipe.vae.decoder = None
        onestep_pipe.scheduler = DDIMScheduler.from_pretrained(sd_id, subfolder="scheduler")
        onestep_pipe = onestep_pipe.to(self.device)
        onestep_pipe.enable_attention_slicing()
        try:
            import xformers
            onestep_pipe.enable_xformers_memory_efficient_attention()
            print("xformers is successfully enabled.")
        except:
            print("xformers is not installed, running without it.")
        self.pipe = onestep_pipe

    def encode_category_prompts(self, cat_list) -> dict:
        with torch.no_grad():
            cat2prompt_embeds = {}
            print("start encoding prompts for categories...")
            for cat in cat_list:
                print("encoding prompt for category:", cat)
                prompt = f"a photo of a {cat}"
                prompt_embeds = self._encode_prompt_embeds(prompt)  # [1, 77, dim]
                cat2prompt_embeds[cat] = prompt_embeds
            print("encoded prompts for categories:", list(cat2prompt_embeds.keys()))

            # free memory
            self.pipe.tokenizer = None
            self.pipe.text_encoder = None
            gc.collect()
            torch.cuda.empty_cache()

            return cat2prompt_embeds

    def encode_null_prompt(self):
        with torch.no_grad():
            prompt = ""
            prompt_embeds = self._encode_prompt_embeds(prompt)  # [1, 77, dim]
            return prompt_embeds

    def _encode_prompt_embeds(self, prompt: str):
        # Tokenizza anche la stringa vuota e produce embeddings validi [1, 77, dim]
        text_inputs = self.pipe.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = getattr(text_inputs, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            out = self.pipe.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            prompt_embeds = out[0]
        return prompt_embeds

    @torch.no_grad()
    def forward(self,
                img_tensor,
                prompt_embed: torch.FloatTensor,
                t=261,
                up_ft_index=1,
                ensemble_size=4):
        '''
        Args:
            prompt_embed: the text embedding tensor in the shape of [1, 77, dim]
            img_tensor: should be a single torch tensor in the shape of [1, C, H, W] or [C, H, W]
            t: the time step to use, should be an int in the range of [0, 1000]
            up_ft_index: which upsampling block of the U-Net to extract feature, you can choose [0, 1, 2, 3]
            ensemble_size: the number of repeated images used in the batch to extract features
        Return:
            unet_ft: a torch tensor in the shape of [1, c, h, w]
        '''

        if img_tensor is not None:
            img_tensor = img_tensor.cuda()  # ensem, c, h, w

        if up_ft_index is not list:
            up_ft_index = [up_ft_index]
        prompt_embed = prompt_embed.repeat(ensemble_size, 1, 1).to(self.device)
        unet_ft_all = self.pipe(
            img_tensor=img_tensor,
            t=t,
            up_ft_indices=up_ft_index,
            prompt_embeds=prompt_embed)
        fts = {}
        for idx in up_ft_index:
            ft = unet_ft_all['up_ft'][idx]  # [ensem, C, H, W]
            ft = ft.mean(0, keepdim=True)  # [1, C, H, W]
            fts[idx] = ft
        if len(up_ft_index) == 1:
            return  fts[up_ft_index[0]]
        else:
            return fts
