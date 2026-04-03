import inspect
import warnings
from typing import Callable, List, Optional, Union, Dict, Any
import PIL
import torch
from packaging import version
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPTokenizer, CLIPTextModel

# Compatibility alias for older logic
CLIPFeatureExtractor = CLIPImageProcessor 

from diffusers.utils.import_utils import is_accelerate_available
from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.embeddings import get_timestep_embedding
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import deprecate, logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.pipelines.stable_diffusion.stable_unclip_image_normalizer import StableUnCLIPImageNormalizer
import os
import torchvision.transforms.functional as TF
from einops import rearrange
import comfy.utils
logger = logging.get_logger(__name__)

class StableUnCLIPImg2ImgPipeline(DiffusionPipeline):
    feature_extractor: CLIPFeatureExtractor
    image_encoder: CLIPVisionModelWithProjection
    image_normalizer: StableUnCLIPImageNormalizer
    image_noising_scheduler: KarrasDiffusionSchedulers
    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModel
    unet: UNet2DConditionModel
    scheduler: KarrasDiffusionSchedulers
    vae: AutoencoderKL

    def __init__(
        self,
        feature_extractor: CLIPFeatureExtractor,
        image_encoder: CLIPVisionModelWithProjection,
        image_normalizer: StableUnCLIPImageNormalizer,
        image_noising_scheduler: KarrasDiffusionSchedulers,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModel,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        vae: AutoencoderKL,
        num_views: int = 4,
    ):
        super().__init__()

        self.register_modules(
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            image_normalizer=image_normalizer,
            image_noising_scheduler=image_noising_scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            unet=unet,
            scheduler=scheduler,
            vae=vae,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.num_views: int = num_views

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")
        models = [self.image_encoder, self.text_encoder, self.unet, self.vae]
        for cpu_offloaded_model in models:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (hasattr(module, "_hf_hook") and 
                hasattr(module._hf_hook, "execution_device") and 
                module._hf_hook.execution_device is not None):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
    ):
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
        if do_classifier_free_guidance:
            normal_prompt_embeds, color_prompt_embeds = torch.chunk(prompt_embeds, 2, dim=0)
            prompt_embeds = torch.cat([normal_prompt_embeds, normal_prompt_embeds, color_prompt_embeds, color_prompt_embeds], 0)
        return prompt_embeds

    def _encode_image(
        self,
        image_pil,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        noise_level: int=0,
        generator: Optional[torch.Generator] = None
    ):
        dtype = next(self.image_encoder.parameters()).dtype
        image = self.feature_extractor(images=image_pil, return_tensors="pt").pixel_values
        image = image.to(device=device, dtype=dtype)
        image_embeds = self.image_encoder(image).image_embeds
        
        image_embeds = self.noise_image_embeddings(
            image_embeds=image_embeds,
            noise_level=noise_level,
            generator=generator,
            )
        image_embeds = image_embeds.repeat(num_images_per_prompt, 1)

        if do_classifier_free_guidance:
            normal_image_embeds, color_image_embeds = torch.chunk(image_embeds, 2, dim=0)
            negative_prompt_embeds = torch.zeros_like(normal_image_embeds)
            image_embeds = torch.cat([negative_prompt_embeds, normal_image_embeds, negative_prompt_embeds, color_image_embeds], 0)
            
        image_pt = torch.stack([TF.to_tensor(img) for img in image_pil], dim=0).to(device)
        image_pt = image_pt * 2.0 - 1.0
        image_latents = self.vae.encode(image_pt).latent_dist.mode() * self.vae.config.scaling_factor
        image_latents = image_latents.repeat(num_images_per_prompt, 1, 1, 1)

        if do_classifier_free_guidance:
            normal_image_latents, color_image_latents = torch.chunk(image_latents, 2, dim=0)
            image_latents = torch.cat([torch.zeros_like(normal_image_latents), normal_image_latents, 
                                       torch.zeros_like(color_image_latents), color_image_latents], 0)

        return image_embeds, image_latents

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, image, height, width, callback_steps, noise_level):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
        if (callback_steps is None) or (callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)):
            raise ValueError(f"`callback_steps` must be a positive integer.")
        if prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list`.")
        if noise_level < 0 or noise_level >= self.image_noising_scheduler.config.num_train_timesteps:
            raise ValueError(f"`noise_level` must be between 0 and {self.image_noising_scheduler.config.num_train_timesteps - 1}.")

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def noise_image_embeddings(self, image_embeds, noise_level, noise=None, generator=None):
        if noise is None:
            noise = randn_tensor(image_embeds.shape, generator=generator, device=image_embeds.device, dtype=image_embeds.dtype)
        noise_level = torch.tensor([noise_level] * image_embeds.shape[0], device=image_embeds.device)
        image_embeds = self.image_normalizer.scale(image_embeds)
        image_embeds = self.image_noising_scheduler.add_noise(image_embeds, timesteps=noise_level, noise=noise)
        image_embeds = self.image_normalizer.unscale(image_embeds)
        noise_level = get_timestep_embedding(timesteps=noise_level, embedding_dim=image_embeds.shape[-1], flip_sin_to_cos=True, downscale_freq_shift=0)
        noise_level = noise_level.to(image_embeds.dtype)
        image_embeds = torch.cat((image_embeds, noise_level), 1)
        return image_embeds

    @torch.no_grad()
    def __call__(
        self,
        image: Union[torch.FloatTensor, PIL.Image.Image],
        prompt: Union[str, List[str]],   
        prompt_embeds: torch.FloatTensor = None,
        dino_feature: torch.FloatTensor = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 10,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        noise_level: int = 0,
        image_embeds: Optional[torch.FloatTensor] = None,
        return_elevation_focal: Optional[bool] = False,
        gt_img_in: Optional[torch.FloatTensor] = None,
    ):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        self.check_inputs(prompt, image, height, width, callback_steps, noise_level)

        if isinstance(image, list):
            batch_size = len(image)
        elif isinstance(image, torch.Tensor):
            batch_size = image.shape[0]
        elif isinstance(image, PIL.Image.Image):
            image = [image] * self.num_views * 2
            batch_size = self.num_views * 2

        if isinstance(prompt, str):
            prompt = [prompt] * self.num_views * 2

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale != 1.0

        text_encoder_lora_scale = cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        prompt_embeds = self._encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        if isinstance(image, list):
            image_pil = image
        elif isinstance(image, torch.Tensor):
            image_pil = [TF.to_pil_image(image[i]) for i in range(image.shape[0])]
        
        noise_level_tensor = torch.tensor([noise_level], device=device)
        image_embeds, image_latents = self._encode_image(
            image_pil=image_pil,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            noise_level=noise_level,
            generator=generator,
        )

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        num_channels_latents = self.unet.config.out_channels
        if gt_img_in is not None:
            latents = gt_img_in * self.scheduler.init_noise_sigma
        else:
            latents = self.prepare_latents(
                batch_size=batch_size,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                dtype=prompt_embeds.dtype,
                device=device,
                generator=generator,
                latents=latents,
            )

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        eles, focals = [], []
        comfy_pbar = comfy.utils.ProgressBar(num_inference_steps)
        
        for i, t in enumerate(self.progress_bar(timesteps)):
            if do_classifier_free_guidance:
                normal_latents, color_latents = torch.chunk(latents, 2, dim=0)  
                latent_model_input = torch.cat([normal_latents, normal_latents, color_latents, color_latents], 0)
            else:
                latent_model_input = latents
            
            latent_model_input = torch.cat([latent_model_input, image_latents], dim=1)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            unet_out = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                dino_feature=dino_feature,
                class_labels=image_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False)
            
            noise_pred = unet_out[0]
            if return_elevation_focal:    
                uncond_pose, pose  = torch.chunk(unet_out[1], 2, 0) 
                pose = uncond_pose + guidance_scale * (pose - uncond_pose)
                eles.append(pose[:, 0].detach().cpu().numpy())
                focals.append(pose[:, 1].detach().cpu().numpy())
                
            if do_classifier_free_guidance:
                normal_noise_pred_uncond, normal_noise_pred_text, color_noise_pred_uncond, color_noise_pred_text = torch.chunk(noise_pred, 4, dim=0)
                noise_pred_uncond = torch.cat([normal_noise_pred_uncond, color_noise_pred_uncond], 0)
                noise_pred_text = torch.cat([normal_noise_pred_text, color_noise_pred_text], 0)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)
            comfy_pbar.update_absolute(i + 1)

        if not output_type == "latent":
            if num_channels_latents == 8:
                latents = torch.cat([latents[:, :4], latents[:, 4:]], dim=0)
            with torch.no_grad():
                image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents

        image = self.image_processor.postprocess(image, output_type=output_type)

        if not return_dict:
            return (image, )
        if return_elevation_focal:
            return ImagePipelineOutput(images=image), eles, focals
        return ImagePipelineOutput(images=image)