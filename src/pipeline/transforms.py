# SCRIPT FROM PAOLO



import torch
import torch.nn as nn
from torch import Tensor

import numpy as np
from math import ceil, floor
from PIL import Image

from random import random

from torchvision import transforms
from torchvision.ops import masks_to_boxes
from torchvision.transforms import Pad
from torchvision.transforms.functional import affine
from torchvision.transforms.functional import normalize
from torchvision.transforms.functional import to_tensor
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms.functional import get_dimensions

from torch.nn.functional import unfold
from functools import partial

from .misc import default
from sklearn.base import BaseEstimator, TransformerMixin

from typing import Any, Union, Tuple, Callable, Union, Dict

class PatchWise(nn.Module):
    def __init__(
        self,
        func : Callable,
        kernel_size : int = 3,
    ) -> Any:
        super().__init__()
    
        self.func = func
        self.kern = kernel_size

        self.patchify = partial(
            unfold,
            kernel_size = kernel_size,
            stride = kernel_size,
        )
    
    @torch.no_grad()
    def forward(self, inp : Tensor) -> Tensor:
        '''
        '''
        if inp.dim() == 3: inp = inp.unsqueeze(0)

        bs, c, h, w = inp.shape

        patches = self.patchify(inp)
        out = self.func(patches, dim = 1)

        # Output shape is [B, C x H // kern x W // kern]
        return out.squeeze()

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

class ConeNoise(nn.Module):
    '''
        Simulate the noise that a cone experience in the Retina.
        Following `Photovoltage of Rods & Cones in the Macaque
        Retina`, Science (1995), we know that cone dynamic range
        is within [5, 12]mV (membrene potential). Dark-light cone
        noise is estimated to be aroud ~1 mV.
    '''

    dark_light = 1 # mV
    cone_range = (5, 12) # mV
    cone_Vrest = -46 # mV

    def __init__(
        self,
        amp_max : float = 10,
        vmin : float = 0.,
        vmax : float = 1.,
        mono : bool = False,
        adjust : bool = False,
        decline : bool = False,
    ) -> None:
        amin, amax = self.cone_range
        assert amp_max >= amin and amp_max <= amax, 'Max amplitude not in range'
        
        super().__init__()

        self.amp_max = amp_max
        self.mono = mono

        self.vmin = vmin
        self.vmax = vmax

        self.adjust = adjust
        self.decline = decline

    @torch.no_grad()
    def forward(self, inp : Tensor | dict, clip : bool = True):
        if self.decline: return inp

        if isinstance(inp, dict): imgs = inp['imgs']
        elif isinstance(inp, Tensor): imgs = inp
        else: raise TypeError(f'Unknown input type: {type(inp)}')

        # Noise amplitude is computed by mapping the dark-light noise
        # within the dynamic range of membrane potential to the pixel
        # dynamic range [0, 1]
        pmax, pmin = (imgs.max(), imgs.min()) if self.adjust else (1, 0)

        dynm_range = (self.dark_light / self.amp_max) * (pmax - pmin)
        cone_noise = (torch.rand_like(imgs) - 0.5) * dynm_range

        # If mono is requested produce black and white noise
        if self.mono: cone_noise = cone_noise.mean(dim = 0, keepdim = True)
        
        # Add noise the the image
        out = (imgs * 0.8 + 0.1) + cone_noise

        out = torch.clip(out, self.vmin, self.vmax) if clip else out 

        if isinstance(inp, dict): ret = {**inp, 'imgs' : out}
        else: ret = out

        return ret

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

class SquarePad(Pad):
    def __init__(
        self,
        fill : int = 0, 
        mode : str = 'constant',
        scale : float = 1,
        preserve : float = False,    
    ) -> None:
        super(SquarePad, self).__init__(0, fill, mode)

        self.scale = scale
        self.preserve = preserve
        
    @torch.no_grad()
    def forward(self, img : Union[torch.FloatTensor, Image.Image]):
        if isinstance(img, Image.Image):
            w, h = img.size
        elif isinstance(img, torch.Tensor):
            h, w = img.shape[-2:]
        else:
            raise TypeError(f'Unknown input type: {type(img)}')
        
        max_wh = max(w, h) * self.scale
        l, r = floor((max_wh - w) / 2), ceil((max_wh - w) / 2) 
        b, t = floor((max_wh - h) / 2), ceil((max_wh - h) / 2)
        self.padding = (l, t, r, b)
        
        if self.preserve:
            out = {
                'orig' : img,
                'imgs' : super().forward(img)
            }

            return out
        else:
            return super().forward(img)
    
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
        # Save the global random state so that we can restore it later
        # init_state = torch.get_rng_state()
        rand_state = torch.get_rng_state() if self.rng is None else self.rng.get_state()

        try:
            torch.set_rng_state(rand_state)
            params = super().get_params(*args)
        finally:
            # Advance the internal rng if present
            torch.rand(1, generator = self.rng)

            torch.seed()
            # new_state = torch.get_rng_state() if self.rng is None else self.rng.get_state()

            # assert not torch.allclose(new_state, rand_state)

            # torch.set_rng_state(init_state)

        return params
    

class BubbleMask(torch.nn.Module):
    
    obj_frac = 0.4825
    
    def __init__(
        self,
        num_bubble : int = 100,
        bub_degree : float = 2,
        mask_frac  : float = 1,
        obj_degree : float = 35,
        num_random : bool = True,
        lim_objbox : Tuple[float] = None,
        fill_color : Tuple[int, int, int] = None,
    ) -> None:
        super(BubbleMask, self).__init__()
        
        self.num_bubble = num_bubble
        self.bub_degree = bub_degree
        self.mask_frac = mask_frac
        self.obj_degree = obj_degree
        self.lim_objbox = lim_objbox
        self.num_random = num_random
        self.fill_color = default(fill_color, (127, 127, 127))
    
    @torch.no_grad()
    def forward(
        self,
        inp : Union[Tensor, Image.Image, Dict],
    ) -> Tuple[Tensor | Image.Image, Tensor] | (Tensor | Image.Image):
        # Type check
        if isinstance(inp, dict):     img = inp['imgs']
        elif isinstance(inp, (Tensor, Image.Image)): img = inp
        else: raise TypeError(f'Unknown input type: {type(inp)}')

        # Extract whether bubble augmentation should be applied
        # if not, just return the unaltered image with no mask
        if random() > self.mask_frac:
            w, h = img.size if isinstance(img, Image.Image) else img.shape[-2:][::-1]
            out, mask = img, torch.zeros(h, w)

        else:
            # Compute `num_bubble` random positions in the domain [W, H]
            pos, (w, h) = self._get_pos(img)
                    
            # Alpha mask the image
            if isinstance(img, Image.Image):
                # Compute the single bubble dimension
                w, h = img.size
                bsize = self.bub_degree * self.obj_frac / self.obj_degree
                
                # Get the bubble mask for the image
                mask = self._get_mask((w, h), pos, bsize, val = 255)
                
                out = Image.new(size = img.size, mode = img.mode, color = self.fill_color)
                out.paste(img, mask = to_pil_image(mask))
                
            elif isinstance(img, Tensor):
                # Compute the single bubble dimension
                w, h = img.shape[-2:]
                bsize = self.bub_degree * self.obj_frac / self.obj_degree 
                
                col = self._rgb2net(self.fill_color)

                # Get the bubble mask for the image
                mask = self._get_mask((w, h), pos, bsize)
                dark = torch.ones_like(img) * col
                
                out = mask.unsqueeze(0)
                out = dark * (1. - mask) + img * mask

            else:
                raise TypeError(f'Unknown image type: {type(img)}')

        # Add fake channel dimension to mask
        if isinstance(inp, dict):
            ret = {**inp, 'imgs' : out, 'mask' : mask.unsqueeze(0)}
        else:
            ret = {'imgs' : out, 'mask' : mask.unsqueeze(0)}

        return ret
    
    def _get_pos(self, img, num : int = None, box : Tuple = None, rnd : bool = None) -> torch.FloatTensor:
        num = default(num, self.num_bubble)
        box = default(box, self.lim_objbox)
        rnd = default(rnd, self.num_random)
    
        # If the random flag is on, use num as the average value
        # for the actual number of bubbles in the image
        if rnd: num = round((torch.randn(1).item() * 0.1 + 1) * num)
        
        pos = torch.rand(num, 2)
        
        (w, h) = img.size if isinstance(img, Image.Image) else img.shape[-2:]
        
        if box is None: return pos, (w, h)
        elif len(box) == 4:
            l, b, r, t = box
            
            fl, fb = l / w, b / h
            bw, bh = (r - l) / w, (t - b) / h
        elif len(box) == 2:
            # Compute the object center
            l, b, r, t = img.getbbox() if isinstance(img, Image.Image) else masks_to_boxes(img > img.min())[0]
            
            cx, cy = (l + (r - l) * 0.5) / w, (b + (t - b) * 0.5) / h
            
            bw, bh = box
            fl, fb = cx - 0.5 * bw, cy - 0.5 * bh
        
        else: raise TypeError(f'Wrong specification of box. Got {box}')
                
        pos[..., 0] = pos[..., 0] * bw + fl
        pos[..., 1] = pos[..., 1] * bh + fb

        return pos, (w, h)
    
    def _get_mask(
        self,
        shape : Tuple[int, int],
        pos : torch.FloatTensor,
        std : float,
        val : float = 1.
    ) -> torch.FloatTensor:
        w, h = shape
        x = torch.linspace(0, 1, w)
        y = torch.linspace(0, 1, h)
        X, Y = torch.meshgrid(x, y, indexing = 'xy')

        # Construct the position tensor of shape (batch, 2, W, H)
        P = torch.stack([X, Y]).unsqueeze(0)
        
        # Make sure position tensor has shape (batch, 2, 1, 1)
        pos = torch.atleast_2d(pos)
        
        # Compute the exponent, tensor with shape (batch, W, H)
        z = torch.sum((P - pos.view(*pos.shape, 1, 1))**2, dim = 1) / (2 * std**2)

        # Return the sum along the batch dimension, clipped at val
        return torch.sum(val * torch.exp(-z), dim = 0).clamp_(0, val)

    def _rgb2net(self, fill : Tuple[int, int, int]):
        fill = Tensor(fill) / 255.

        # if norm:
        #     mean = Tensor(self.img_mean)
        #     istd = Tensor(self.img_std)
        #     fill = ((fill - mean) / istd)

        return fill.view(3, 1, 1)

class WrapInDict(torch.nn.Module):
    def __init__(
        self,
        key : str = 'imgs',
    ) -> None:
        super().__init__()
        
        self.key = key

    def forward(
        self,
        inp : Union[Tensor, Image.Image, Dict]
    ) -> dict:
        if isinstance(inp, dict):     imgs = inp['imgs']
        elif isinstance(inp, (Tensor, Image.Image)): imgs = inp
        else: raise TypeError(f'Unknown input type: {type(inp)}')

        if isinstance(inp, dict): ret = inp
        else: ret = {'mask' : torch.empty(1), self.key : inp}

        return ret

def rat_transform(
        img_size : int = 224,
        img_scale_fact : float = 1.,
        noise_std : float = 0.3,
        mask_frac : float = 0.,
        blur_sig : int = 2,
        blur_ker : int = 29,
        preserve : bool = False,
        still_img: bool = False,
        clean_img: bool = False,
        bubble_kw : dict | None = None,
        noise_type: str = 'gauss',
        normalize : str = 'none',
        noise_rng : torch.Generator | None = None,
        affine_rng: torch.Generator | None = None,
    ) -> transforms.Compose:

    assert noise_type in ('gauss', 'cone')

    # * Parameter of the transformation that should be realist for the rat
    # (derived from the Vanzella et al. paper)
    real_deg = (-17.5, 17.5)
    real_trs = (0.1030584056091575, 0.31541628719290393)

    # Retina noise matched to the noise experienced by a Cone in the retina
    noise_kw = {'amp_max' : 6, 'mono' : True} if noise_type == 'cone' else\
               {'std' : noise_std, 'mono' : True}
    Noise = ConeNoise if noise_type == 'cone' else GaussianNoise

    bubble_kw = default(bubble_kw, {})

    bubble_kw = {
        'num_bubble' : 40,
        'bub_degree' : 2,
        'lim_objbox' : (.7, .7),
        'fill_color' : (0., 0., 0.),
        'mask_frac' : mask_frac,
        **bubble_kw,
    }

    rat_vision = transforms.Compose([
        # turn images to a square format via zero-padding and enlarge
        # them by img_scale_fact to allow for the full breadth of translation
        # to occur within the frame (huge vertical translations)
        ToTensor(),
        SquarePad(scale = img_scale_fact, preserve = preserve), 
        Resize(img_size),
        BubbleMask(**bubble_kw),
        RandomImgAffine(degrees = real_deg, translate = real_trs, decline = still_img, generator = affine_rng),
        GaussianBlur(kernel_size = blur_ker, sigma = blur_sig, decline = clean_img),
        Noise(**noise_kw, decline = clean_img, generator = noise_rng),
        Normalize(normalize),
    ])

    return rat_vision

# def rat_collate_fn(batch):
#     new_batch = {}

class FlatScaler(TransformerMixin, BaseEstimator):
    def __init__(self) -> None:
        self.u = np.nan
        self.s = np.nan

    def fit(self, X, y = None):
        self.u = np.nanmean(X)
        self.s = np.nanstd (X)

        if np.isnan(self.s) or self.s == 0:
            raise ValueError('Cannot estimate parameters in Flat Scaler')

        return self

    def transform(self, X):
        X -= self.u
        X /= self.s
        
        return X
    

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