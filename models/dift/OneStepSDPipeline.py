from typing import Optional

import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler


class OneStepSDPipeline(StableDiffusionPipeline):
    @torch.no_grad()
    def __call__(
            self,
            img_tensor,
            t,
            up_ft_indices,
            prompt_embeds: torch.FloatTensor,
    ):
        device = self._execution_device
        latents = self.vae.encode(img_tensor).latent_dist.sample() * self.vae.config.scaling_factor
        t = torch.tensor(t, dtype=torch.long, device=device)
        noise = torch.randn_like(latents).to(device)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        unet_output = self.unet(
            latents_noisy,
            t,
            encoder_hidden_states=prompt_embeds,
            up_ft_indices=up_ft_indices,
        )
        return unet_output