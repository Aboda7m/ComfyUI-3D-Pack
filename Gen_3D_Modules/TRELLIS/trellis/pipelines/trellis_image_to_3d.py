from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from torchvision import transforms
from PIL import Image
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp
from ..representations import Gaussian, Strivec, MeshExtractResult


class TrellisImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis image-to-3D models.
    Reinforced for Blackwell (RTX 50-series) and SageAttention dtype consistency.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        image_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self.rembg_session = None
        self._init_image_cond_model(image_cond_model)

    @staticmethod
    def from_pretrained(path: str) -> "TrellisImageTo3DPipeline":
        """
        Load a pretrained model and force strict half-precision for RTX 5080.
        """
        pipeline = super(TrellisImageTo3DPipeline, TrellisImageTo3DPipeline).from_pretrained(path)
        new_pipeline = TrellisImageTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        
        # Blackwell Strict Fix: Ensure ALL modules and buffers are FP16
        for name in new_pipeline.models:
            if isinstance(new_pipeline.models[name], nn.Module):
                new_pipeline.models[name] = new_pipeline.models[name].to(new_pipeline.device).half()

        args = pipeline._pretrained_args
        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']
        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']
        new_pipeline.slat_normalization = args['slat_normalization']
        new_pipeline._init_image_cond_model(args['image_cond_model'])

        return new_pipeline
    
    def _init_image_cond_model(self, name: str):
        dinov2_model = torch.hub.load('facebookresearch/dinov2', name, pretrained=True)
        dinov2_model.eval()
        self.models['image_cond_model'] = dinov2_model.to(self.device).half()
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model_transform = transform

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        if not has_alpha:
            import rembg
            input = input.convert('RGB')
            max_size = max(input.size)
            scale = min(1, 1024 / max_size)
            if scale < 1:
                input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
            if getattr(self, 'rembg_session', None) is None:
                self.rembg_session = rembg.new_session('u2net')
            input = rembg.remove(input, session=self.rembg_session)
        
        output_np = np.array(input)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = int(max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 1.2)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = input.crop(bbox).resize((518, 518), Image.Resampling.LANCZOS)
        output_data = np.array(output).astype(np.float32) / 255
        output_data = output_data[:, :, :3] * output_data[:, :, 3:4]
        return Image.fromarray((output_data * 255).astype(np.uint8))

    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, list[Image.Image]]) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            image = image.to(self.device).to(torch.float16)
        else:
            image = [torch.from_numpy(np.array(i.convert('RGB')).astype(np.float32) / 255).permute(2, 0, 1).to(torch.float16) for i in image]
            image = torch.stack(image).to(self.device)
        
        image = self.image_cond_model_transform(image).to(torch.float16)
        features = self.models['image_cond_model'](image, is_training=True)['x_prenorm']
        return F.layer_norm(features, features.shape[-1:]).to(torch.float16)
        
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]]) -> dict:
        cond = self.encode_image(image)
        return {'cond': cond, 'neg_cond': torch.zeros_like(cond)}

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        # Strict FP16 for noise and cond
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device).to(torch.float16)
        cond = {k: v.to(torch.float16) if isinstance(v, torch.Tensor) else v for k, v in cond.items()}
        
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(
            flow_model, noise, **cond, **sampler_params, verbose=True
        ).samples
        
        decoder = self.models['sparse_structure_decoder']
        return torch.argwhere(decoder(z_s.to(torch.float16)) > 0)[:, [0, 2, 3, 4]].int()

    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> sp.SparseTensor:
        flow_model = self.models['slat_flow_model']
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device).to(torch.float16),
            coords=coords,
        )
        cond = {k: v.to(torch.float16) if isinstance(v, torch.Tensor) else v for k, v in cond.items()}
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        
        slat = self.slat_sampler.sample(
            flow_model, noise, **cond, **sampler_params, verbose=True
        ).samples

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device).to(torch.float16)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device).to(torch.float16)
        return slat * std + mean

    def decode_slat(self, slat: sp.SparseTensor, formats: List[str] = ['mesh', 'gaussian', 'radiance_field']) -> dict:
        ret = {}
        slat_input = slat # Already FP16 from sample_slat
        if 'mesh' in formats: ret['mesh'] = self.models['slat_decoder_mesh'](slat_input)
        if 'gaussian' in formats: ret['gaussian'] = self.models['slat_decoder_gs'](slat_input)
        if 'radiance_field' in formats: ret['radiance_field'] = self.models['slat_decoder_rf'](slat_input)
        return ret

    @torch.no_grad()
    def run(self, image: Image.Image, num_samples: int = 1, seed: int = 42, 
            sparse_structure_sampler_params: dict = {}, slat_sampler_params: dict = {},
            formats: List[str] = ['mesh', 'gaussian', 'radiance_field'], preprocess_image: bool = True) -> dict:
        if preprocess_image:
            image = self.preprocess_image(image)
        cond = self.get_cond([image])
        torch.manual_seed(seed)
        coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)