# SCRIPT FROM PAOLO

import torch
import torch.nn as nn
from torch import Tensor

import numpy as np
from math import ceil, floor
from PIL import Image

from torchvision import transforms
from torchvision.transforms import Pad
from torchvision.transforms.functional import affine
from torchvision.transforms.functional import normalize
from torchvision.transforms.functional import to_tensor
from torchvision.transforms.functional import get_dimensions, to_pil_image

from typing import Union, Tuple, Callable, Dict
from random import random
from sklearn.base import BaseEstimator, TransformerMixin
from torchvision.ops import masks_to_boxes



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
    
    
import torch
import torch.nn as nn

class RgbToMouseLike(nn.Module):
    def __init__(self, decline: bool = False):
        super().__init__()
        self.decline = decline
        # matrice 3x3: input [R,G,B] → output [R’,G’,B’]
        self.transform = torch.tensor([
            [0.2, 0.1, 0.0],   # R'
            [0.2, 0.8, 0.0],   # G'
            [0.1, 0.0, 0.9],   # B'
        ])  # (3,3)

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        if self.decline:
            return x
        if x.dim() == 3:   # C,H,W
            x = x.unsqueeze(0)  # → N,C,H,W
        assert x.size(1) == 3, "Input deve essere RGB"

        T = self.transform.to(x.device).to(x.dtype)  # (3,3)
        out = torch.einsum("ij,njhw->nihw", T, x)    # → N,3,H,W
        return out
