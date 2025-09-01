import math
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from typing import Tuple, Union, Dict

from .transforms import (
    ToTensor, SquarePad, Resize, RandomImgAffine,
    GaussianBlur as _GaussianBlur, GaussianNoise as _GaussianNoise, Normalize, RgbToGray, RgbToMouseLike
)
from .mouse_params import FOV_DEG, DEFAULT_ROLL_DEG, DEFAULT_TRANSLATE

Tensor = torch.Tensor

class SphericalWarp(nn.Module):
    """
    Spherical warping to emulate Allen Neuropixels monitor correction.
    Maps output pixels uniformly across visual angle (equirectangular grid),
    then inverts to a flat monitor via tan mapping so apparent size/speed/SF are constant.
    """
    def __init__(self, fov_deg: Tuple[float, float] = FOV_DEG, align_corners: bool = True) -> None:
        super().__init__()
        self.fovx = math.radians(fov_deg[0])
        self.fovy = math.radians(fov_deg[1])
        self.ax = math.tan(self.fovx / 2.0)
        self.ay = math.tan(self.fovy / 2.0)
        self.align_corners = align_corners

    @torch.no_grad()
    def forward(self, inp: Union[Tensor, Dict]) -> Union[Tensor, Dict]:
        if isinstance(inp, dict): imgs = inp['imgs']
        elif isinstance(inp, torch.Tensor): imgs = inp
        else: raise TypeError(f'Unknown input type: {type(inp)}')

        single = imgs.dim() == 3
        if single:
            imgs = imgs.unsqueeze(0)  # NCHW

        N, C, H, W = imgs.shape
        device = imgs.device

        # Grid in normalized output coords (uniform in angle)
        ys = torch.linspace(-1.0, 1.0, H, device=device)
        xs = torch.linspace(-1.0, 1.0, W, device=device)
        Y, X = torch.meshgrid(ys, xs, indexing='ij')  # H x W

        # Target angles (uniform over FOV), then inverse tan to planar coords
        theta_x = (X * (self.fovx / 2.0))
        theta_y = (Y * (self.fovy / 2.0))

        # Inverse spherical correction used to pre-warp stimuli for a flat monitor
        Xp = torch.tan(theta_x) / self.ax
        Yp = torch.tan(theta_y) / self.ay

        grid = torch.stack([Xp, Yp], dim=-1)  # H x W x 2, range roughly [-1,1]
        grid = grid.expand(N, H, W, 2)

        warped = F.grid_sample(imgs, grid, mode='bilinear', padding_mode='border', align_corners=self.align_corners)

        if single: warped = warped.squeeze(0)

        if isinstance(inp, dict): return {**inp, 'imgs': warped}
        return warped

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
    resize: bool = True,
    square_pad: bool = False,
) -> transforms.Compose:
    """
    Mouse-calibrated preprocessing:
    1) SquarePad + optional spherical warping to 120x95 deg FOV
    2) Random retinal motion (yaw/pitch -> translate (not used), roll -> rotate(not used))
    3) Gaussian blur + Gaussian noise (CSF-matched)
    """
    if to_mouse:
        if to_gray:
            raise ValueError("to_gray and to_mouse cannot be True at the same time")
        if not gray_keep_channels:
            import warnings
            warnings.warn("gray_keep_channels is ignored when to_mouse=True; output is always 3 channels")
    
    ops = [ToTensor()]
    
    if square_pad:
        ops.append(SquarePad(scale=img_scale_fact, preserve=False))
    
    if apply_warp:
        ops.append(SphericalWarp(FOV_DEG))

    if resize:
        ops.append(Resize(img_size))

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

    if apply_csf:
        ops.extend([
            GaussianBlur(kernel_size=blur_ker, sigma=blur_sig, decline=(blur_sig is None or blur_sig <= 0)),
            GaussianNoise(std=noise_std, mono=True, decline=(noise_std is None or noise_std <= 0), generator=noise_rng),
        ])
    if to_gray:
        ops.append(RgbToGray(keep_channels=3 if gray_keep_channels else 1))
    if to_mouse:
        ops.append(RgbToMouseLike(decline=False))
        
    ops.append(Normalize(normalize))
    return transforms.Compose(ops)