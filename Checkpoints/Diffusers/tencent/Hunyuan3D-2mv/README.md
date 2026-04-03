---
library_name: hunyuan3d-2
license: other
license_name: tencent-hunyuan-community
license_link: https://huggingface.co/tencent/Hunyuan3D-2/blob/main/LICENSE.txt
language:
  - en
  - zh
tags:
  - image-to-3d
  - text-to-3d
pipeline_tag: image-to-3d
extra_gated_eu_disallowed: true
---

<p align="center">
  <img src="https://huggingface.co/tencent/Hunyuan3D-2/resolve/main/assets/images/teaser.jpg">
</p>

<div align="center">
  <a href=https://3d.hunyuan.tencent.com target="_blank"><img src=https://img.shields.io/badge/Hunyuan3D-black.svg?logo=homepage height=22px></a>
  <a href=https://huggingface.co/spaces/tencent/Hunyuan3D-2mv  target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Demo-276cb4.svg height=22px></a>
  <a href=https://huggingface.co/tencent/Hunyuan3D-2mv target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg height=22px></a>
  <a href=https://github.com/Tencent/Hunyuan3D-2 target="_blank"><img src= https://img.shields.io/badge/Github-bb8a2e.svg?logo=github height=22px></a>
<a href=https://discord.gg/GuaWYwzKbX target="_blank"><img src= https://img.shields.io/badge/Discord-white.svg?logo=discord height=22px></a>
    <a href=https://github.com/Tencent/Hunyuan3D-2/blob/main/assets/report/Tencent_Hunyuan3D_2_0.pdf target="_blank"><img src=https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv height=22px></a>
</div>


[//]: # (  <a href=# target="_blank"><img src=https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv height=22px></a>)

[//]: # (  <a href=# target="_blank"><img src= https://img.shields.io/badge/Colab-8f2628.svg?logo=googlecolab height=22px></a>)

[//]: # (  <a href="#"><img alt="PyPI - Downloads" src="https://img.shields.io/pypi/v/mulankit?logo=pypi"  height=22px></a>)

<br>
<p align="center">
“ Living out everyone’s imagination on creating and manipulating 3D assets.”
</p>

This repository contains the models of the paper [Hunyuan3D 2.0: Scaling Diffusion Models for High Resolution Textured 3D Assets Generation](https://huggingface.co/papers/2501.12202).

**Hunyuan3D-2mv** is finetuned from [Hunyuan3D-2](https://huggingface.co/tencent/Hunyuan3D-2) to support multiview controlled shape generation.

## 🤗 Get Started with Hunyuan3D 2mv

Here is a simple usage:

```python
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'tencent/Hunyuan3D-2mv',
    subfolder='hunyuan3d-dit-v2-mv',
    use_safetensors=True,
    device='cuda'
)
mesh = pipeline(
    image={
        "front": "your front view image.png",
        "left": "your left view image.png",
        "back": "your back view image.png"
    },
    num_inference_steps=30,
    octree_resolution=380,
    num_chunks=20000,
    generator=torch.manual_seed(12345),
    output_type='trimesh'
)[0]
```

For code and more details on how to use it, refer to the [Github repository](https://github.com/Tencent/Hunyuan3D-2).



## 🔗 BibTeX

If you found this repository helpful, please cite our report:

```bibtex
@misc{hunyuan3d22025tencent,
    title={Hunyuan3D 2.0: Scaling Diffusion Models for High Resolution Textured 3D Assets Generation},
    author={Tencent Hunyuan3D Team},
    year={2025},
    eprint={2501.12202},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@misc{yang2024tencent,
    title={Tencent Hunyuan3D-1.0: A Unified Framework for Text-to-3D and Image-to-3D Generation},
    author={Tencent Hunyuan3D Team},
    year={2024},
    eprint={2411.02293},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Community Resources

Thanks for the contributions of community members, here we have these great extensions of Hunyuan3D 2.0:

- [ComfyUI-Hunyuan3DWrapper](https://github.com/kijai/ComfyUI-Hunyuan3DWrapper)
- [Hunyuan3D-2-for-windows](https://github.com/sdbds/Hunyuan3D-2-for-windows)
- [📦 A bundle for running on Windows | 整合包](https://github.com/YanWenKun/Comfy3D-WinPortable/releases/tag/r8-hunyuan3d2)

## Acknowledgements

We would like to thank the contributors to
the [DINOv2](https://github.com/facebookresearch/dinov2), [Stable Diffusion](https://github.com/Stability-AI/stablediffusion), [FLUX](https://github.com/black-forest-labs/flux), [diffusers](https://github.com/huggingface/diffusers)
and [HuggingFace](https://huggingface.co) repositories, for their open research and exploration.

