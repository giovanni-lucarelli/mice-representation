import math
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from typing import Tuple, Union, Dict

from .transforms import (
    ToTensor, Resize, RandomImgAffine,
    GaussianBlur as _GaussianBlur, GaussianNoise as _GaussianNoise, Normalize, RgbToGray, RgbToMouseLike
)
from .mouse_params import FOV_DEG, DEFAULT_ROLL_DEG, DEFAULT_TRANSLATE

Tensor = torch.Tensor

class GaussianBlur(_GaussianBlur): pass
class GaussianNoise(_GaussianNoise): pass

def mouse_transform(
    img_size: int = 224,
    img_scale_fact: float = 1.,
    roll_deg: float = DEFAULT_ROLL_DEG,
    translate: Tuple[float, float] = DEFAULT_TRANSLATE,
    blur_sig: float = 1.4,
    blur_ker: int = 11,
    noise_std: float = 0.08,
    normalize: str = 'imagenet',
    apply_motion: bool = False,
    apply_csf: bool = True,
    apply_warp: bool = False,
    noise_rng: torch.Generator | None = None,
    affine_rng: torch.Generator | None = None,
    to_gray: bool = True,
    to_mouse: bool = False,
    gray_keep_channels: bool = True,
    resize: bool = True
) -> transforms.Compose:
    """
    Mouse-calibrated preprocessing:
    1) Gaussian blur + Gaussian noise (CSF-matched)
    """
    
    ops = [ToTensor()]

    if resize:
        #ops.append(Resize(img_size))
        ops.append(transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(1, 1)))

    if apply_motion:
        ops.append(RandomImgAffine(
            degrees=(-roll_deg, roll_deg),
            translate=translate,
            decline=False,
            generator=affine_rng,
            shear=None,
            scale=None,
            fill=0.5,
        ))

    # Convert to grayscale before applying CSF blur+noise so both operate on luminance
    if to_gray:
        ops.append(RgbToGray(keep_channels=3 if gray_keep_channels else 1))
        
    if apply_csf:
        # Dynamic kernel size to cover ~±3σ
        import math
        k_dyn = int(2 * math.ceil(3.0 * float(blur_sig))) + 1 if (blur_sig is not None and blur_sig > 0) else 1
        ops.extend([
            GaussianBlur(kernel_size=k_dyn, sigma=blur_sig, decline=(blur_sig is None or blur_sig <= 0)),
            GaussianNoise(std=noise_std, mono=True, decline=(noise_std is None or noise_std <= 0), generator=noise_rng),
        ])
        
    ops.append(Normalize(normalize))
    return transforms.Compose(ops)