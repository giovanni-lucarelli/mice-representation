from math import ceil
from typing import Dict, Tuple, Union

from PIL import Image
import torch
import torch.nn as nn
from torch import Tensor
from torchvision import transforms
from torchvision.transforms.functional import to_tensor, get_dimensions, affine, normalize
import numpy as np


from .mouse_params import DEFAULT_ROLL_DEG, DEFAULT_TRANSLATE



def default(var, val):
    """Return `val` when `var` is None, otherwise return `var`."""
    return val if var is None else var

# Custom Transformation to introduce Gaussian Noise as part of the Preprocessing Pipeline
class GaussianNoise(nn.Module):
    def __init__(self,
        mean : int = 0,
        std  : int = 1.,
        vmin : float = 0.,
        vmax : float = 1.,
        mono : bool = False,
        decline : bool = False,
        generator : torch.Generator | None = None,
    ) -> None:
        super(GaussianNoise, self).__init__()

        self.mean, self.std  = mean, std
        self.vmin, self.vmax = vmin, vmax

        self.mono = mono

        self.decline = decline

        self.rng = generator 

    @torch.no_grad()
    def forward(self, inp : Union[Tensor, dict], clip : bool = True) -> Union[Tensor, dict]:
        if self.decline: return inp

        if isinstance(inp, dict): imgs = inp['imgs']
        elif isinstance(inp, Tensor): imgs = inp
        else: raise TypeError(f'Unknown input type: {type(inp)}')
        
        noise = torch.randn(imgs.shape, generator = self.rng, device = imgs.device)
        noise = noise * self.std + self.mean

        # If mono is requested produce black and white noise
        if self.mono: noise = noise.mean(dim = 0, keepdim = True)
        
        out = imgs + noise

        out = torch.clip(out, self.vmin, self.vmax) if clip else out 

        if isinstance(inp, dict): ret = {**inp, 'imgs' : out}
        else: ret = out

        return ret

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(mean = {self.mean}, std = {self.std})'

class GaussianBlur(transforms.GaussianBlur):

    def __init__(self, sigma, *args, decline : bool = False, **kwargs):
        if sigma is None or sigma <= 0:
            sigma = 1
            decline = True

        super().__init__(*args, sigma = sigma, **kwargs)

        self.decline = decline

    @torch.no_grad()
    def forward(self, inp : Union[Tensor, Image.Image, dict]):
        if self.decline: return inp
        
        if isinstance(inp, dict):     imgs = inp['imgs']
        elif isinstance(inp, (Image.Image, Tensor)): imgs = inp
        else: raise TypeError(f'Unknown input type: {type(inp)}')

        out = super().forward(imgs)

        if isinstance(inp, dict): ret = {**inp, 'imgs' : out}
        else: ret = out

        return ret
    
class Resize(transforms.Resize):

    @torch.no_grad()
    def forward(self, inp : Tensor | Image.Image | dict):
        if isinstance(inp, dict):     imgs = inp['imgs']
        elif isinstance(inp, (Image.Image, Tensor)): imgs = inp
        else: raise TypeError(f'Unknown input type: {type(inp)}')

        out = super().forward(imgs)

        try:
            mask = {'mask' : super().forward(inp['mask'])}

        except (ValueError, IndexError, KeyError):
            mask = {}

        if isinstance(inp, dict): ret = {**inp, 'imgs' : out, **mask}
        else: ret = out

        return ret
    
class ToTensor(nn.Module):

    def __init__(self) -> None:
        super().__init__()
    
    @torch.no_grad()
    def forward(self, inp : Image.Image):
        if isinstance(inp, dict): imgs = inp['imgs']
        elif isinstance(inp, (Image.Image, np.ndarray)): imgs = inp
        else: raise TypeError(f'Unknown input type: {type(inp)}')

        out = to_tensor(imgs)

        if isinstance(inp, dict): ret = {**inp, 'imgs' : out}
        else: ret = out

        return ret
    
class Normalize(nn.Module):

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std  = (0.229, 0.224, 0.225)


    def __init__(self, which : str = 'none') -> None:
        super().__init__()

        assert which in ('imagenet', 'mean', 'none')

        self.which = which

    @torch.no_grad()
    def forward(self, inp : dict | Tensor) -> dict | Tensor:
        if isinstance(inp, dict):     imgs = inp['imgs']
        elif isinstance(inp, Tensor): imgs = inp
        else: raise TypeError(f'Unknown input type: {type(inp)}')

        # Normalize the output
        if self.which == 'imagenet':
            out = normalize(imgs, mean = self.imagenet_mean, std = self.imagenet_std)
        elif self.which == 'mean':
            out = imgs - imgs.mean()
        elif self.which == 'none':
            out = imgs

        if isinstance(inp, dict): ret = {**inp, 'imgs' : out}
        else: ret = out

        return ret
    
class UnNormalize(nn.Module):
    imagenet_mean = torch.tensor((0.485, 0.456, 0.406))
    imagenet_std  = torch.tensor((0.229, 0.224, 0.225))

    def __init__(self, which : str = 'none') -> None:
        super().__init__()

        assert which in ('imagenet', 'mean', 'none')

        self.which = which

    @torch.no_grad()
    def forward(self, inp : dict | Tensor) -> dict | Tensor:
        if isinstance(inp, dict):     imgs = inp['imgs']
        elif isinstance(inp, Tensor): imgs = inp
        else: raise TypeError(f'Unknown input type: {type(inp)}')

        torch.atleast_3d

        # Normalize the output
        if self.which == 'imagenet':
            if imgs.dim() == 3:
                mean = self.imagenet_mean.view(3, 1, 1)
                std  = self.imagenet_std.view (3, 1, 1)
            
            elif imgs.dim() == 4:
                mean = self.imagenet_mean.view(1, 3, 1, 1)
                std  = self.imagenet_std.view (1, 3, 1, 1)
            
            else:
                raise ValueError(f'Wrong shape of imgs: {imgs.shape}')

            out = (imgs * std) + mean
        elif self.which == 'mean':
            out = imgs + imgs.mean()
        elif self.which == 'none':
            out = imgs

        if isinstance(inp, dict): ret = {**inp, 'imgs' : out}
        else: ret = out

        return ret
    
class RandomImgAffine(transforms.RandomAffine):

    def __init__(
        self,
        *args,
        decline : bool = False,
        generator : torch.Generator | None = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.decline = decline
        self.rng = generator

    @torch.no_grad()
    def forward(self, inp : Tensor | dict) -> Tensor | dict:
        if self.decline: return inp

        if isinstance(inp, dict):     imgs = inp['imgs']
        elif isinstance(inp, Tensor): imgs = inp
        else: raise TypeError(f'Unknown input type: {type(inp)}')

        # * CODE TAKEN FROM RANDOMAFFINE DOC AT:
        # http://pytorch.org/vision/main/_modules/torchvision/transforms/transforms.html#RandomAffine
        fill = self.fill
        channels, height, width = get_dimensions(imgs)
        if isinstance(imgs, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            else:
                fill = [float(f) for f in fill]

        img_size = [width, height]  # flip for keeping BC on get_params call

        par = self.get_params(self.degrees, self.translate, self.scale, self.shear, img_size)

        out = affine(imgs, *par, interpolation=self.interpolation, fill=fill, center=self.center)

        if isinstance(inp, dict):
            ret = {**inp, 'imgs' : out, 'affine' : par}
        else:
            ret = {'imgs' : out, 'affine' : par}

        return ret
    
    def get_params(self, *args):
        # If a custom generator is provided, temporarily swap global RNG state
        # so torchvision's get_params uses it, then restore the original state.
        if self.rng is None:
            return super().get_params(*args)

        init_state = torch.get_rng_state()
        try:
            torch.set_rng_state(self.rng.get_state())
            params = super().get_params(*args)
            # Capture advanced state back into the custom generator
            self.rng.set_state(torch.get_rng_state())
        finally:
            # Restore original global RNG state (do NOT reseed)
            torch.set_rng_state(init_state)

        return params

class RgbToGray(nn.Module):
    def __init__(self, keep_channels: int = 3, weights: tuple = (0.2989, 0.5870, 0.1140), decline: bool = False) -> None:
        super().__init__()
        assert keep_channels in (1, 3)
        self.keep_channels = keep_channels
        self.weights = torch.tensor(weights)
        self.decline = decline

    @torch.no_grad()
    def forward(self, inp: Tensor | Image.Image | dict):
        if self.decline: return inp

        if isinstance(inp, dict):     imgs = inp['imgs']
        elif isinstance(inp, (Image.Image, Tensor)): imgs = inp
        else: raise TypeError(f'Unknown input type: {type(inp)}')

        if isinstance(imgs, Image.Image):
            g = imgs.convert('L')
            out = Image.merge('RGB', (g, g, g)) if self.keep_channels == 3 else g
        elif isinstance(imgs, Tensor):
            x = imgs
            if x.dim() == 3:
                if x.size(0) == 3:
                    w = self.weights.to(x.dtype).to(x.device).view(3, 1, 1)
                    g = (x * w).sum(0, keepdim=False).unsqueeze(0)     # 1 x H x W
                elif x.size(0) == 1:
                    g = x
                else:
                    raise ValueError(f'Unsupported channel count: {x.size(0)}')
                out = g.expand(3, -1, -1) if self.keep_channels == 3 else g
            elif x.dim() == 4:
                if x.size(1) == 3:
                    w = self.weights.to(x.dtype).to(x.device).view(1, 3, 1, 1)
                    g = (x * w).sum(1, keepdim=True)                  # N x 1 x H x W
                elif x.size(1) == 1:
                    g = x
                else:
                    raise ValueError(f'Unsupported channel count: {x.size(1)}')
                out = g.expand(-1, 3, -1, -1) if self.keep_channels == 3 else g
            else:
                raise ValueError(f'Unsupported tensor shape: {x.shape}')
        else:
            out = imgs

        if isinstance(inp, dict): return {**inp, 'imgs': out}
        return out

def mouse_transform(
    img_size: int = 224,
    blur_sig: float = 1.76,
    noise_std: float = 0.25,
    normalize: str = 'imagenet',
    apply_blur: bool = True,
    apply_noise: bool = True,
    noise_rng: torch.Generator | None = None,
    to_gray: bool = True,
    gray_keep_channels: bool = True,
    train: bool = True,
    # affine_rng: torch.Generator | None = None,
    # apply_motion: bool = False,
    # roll_deg: float = DEFAULT_ROLL_DEG,
    # translate: Tuple[float, float] = DEFAULT_TRANSLATE,
    self_supervised: bool = False,
) -> transforms.Compose:
    """
    Mouse-calibrated preprocessing:
    1) Gaussian blur + Gaussian noise (CSF-matched)
    """
    
    ops = []

    if train:
        if self_supervised:
            ops.append(transforms.RandomResizedCrop(img_size, scale = (0.2, 1.0)))
            ops.append(transforms.RandomGrayscale(p=0.2))
            ops.append(transforms.ColorJitter(0.4, 0.4, 0.4, 0.4))
        else:
            ops.append(transforms.RandomResizedCrop(img_size))
            
        ops.append(transforms.RandomHorizontalFlip())
    else:
        ops.append(transforms.Resize(256))
        ops.append(transforms.CenterCrop(img_size))
        
    # if apply_motion:
    #     ops.append(RandomImgAffine(
    #         degrees=(-roll_deg, roll_deg),
    #         translate=translate,
    #         decline=False,
    #         generator=affine_rng,
    #         shear=None,
    #         scale=None,
    #         fill=0.5,
    #     ))

    # Convert to grayscale before applying CSF blur so both operate on luminance
    if to_gray:
        ops.append(RgbToGray(keep_channels=3 if gray_keep_channels else 1))

    if apply_blur:
        # Dynamic kernel size to cover ~±3σ
        k_dyn = int(2 * ceil(3.0 * float(blur_sig))) + 1 if (blur_sig is not None and blur_sig > 0) else 1
        ops.append(GaussianBlur(kernel_size=k_dyn, sigma=blur_sig, decline=(blur_sig is None or blur_sig <= 0)))

    # Always convert to tensor before noise/normalization
    ops.append(ToTensor())

    if apply_noise:
        ops.append(GaussianNoise(std=noise_std, mono=True, decline=(noise_std is None or noise_std <= 0), generator=noise_rng))

    ops.append(Normalize(normalize))
    return transforms.Compose(ops)