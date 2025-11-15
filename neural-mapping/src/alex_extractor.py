import os
from typing import Dict, List, Optional, Sequence, Union, Iterator, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import glob
from torch.utils.data import Dataset, DataLoader

import os, re, glob
import numpy as np
from PIL import Image

from math import ceil
from .mouse_transforms import RgbToGray, GaussianBlur, GaussianNoise


def _natural_key(s: str):
    # so 2.png comes before 10.png
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', os.path.basename(s))]


class PNGFolderDataset(Dataset):
    def __init__(self, folder: str, filenames: list[str] | None = None, pattern: str = "*.png"):
        if filenames is None:
            self.paths = sorted(glob.glob(os.path.join(folder, pattern)), key=_natural_key)
        else:
            self.paths = [os.path.join(folder, f) for f in filenames]
        if not self.paths:
            raise ValueError(f"No images found in {folder!r}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        im = Image.open(self.paths[idx])
        if im.mode in ("I;16", "I"):
            arr = np.array(im, dtype=np.uint16)
            arr = (arr / 257.0).round().clip(0, 255).astype(np.uint8)
            im = Image.fromarray(arr, mode="L").convert("RGB")
        else:
            im = im.convert("RGB")
        return im  # no label needed

# -----------------------------------
# 1) AlexNet feature extractor (224x)
# -----------------------------------

# debugging extraction
# loading pth on a dict

def _extract_alexnet_state_dict(obj: dict) -> dict:
    # If already flat
    if any(k.startswith(("features.", "classifier.")) for k in obj.keys()):
        return obj
    # Common nested keys
    for k in ("state_dict", "model_state_dict", "model", "net", "weights"):
        if k in obj and isinstance(obj[k], dict):
            return obj[k]
    raise ValueError("No state_dict found in checkpoint.")

def _strip_prefixes(sd: dict) -> dict:
    for pref in ("module.", "model.", "alexnet.", "net."):
        sd = { (k[len(pref):] if k.startswith(pref) else k): v for k, v in sd.items() }
    return sd

def load_alexnet_weights(model: nn.Module, weights: Union[str, dict]):
    if weights in ("imagenet", "random"):
        # imagenet handled earlier by constructing model with weights
        # random -> do nothing
        return None

    path = weights["file"] if isinstance(weights, dict) else str(weights)
    chk = torch.load(path, map_location="cpu")
    sd = _extract_alexnet_state_dict(chk)
    sd = _strip_prefixes(sd)

    # --- prune any key with shape mismatch (e.g., classifier.6.*) ---
    model_sd = model.state_dict()
    pruned = {}
    dropped = []
    for k, v in sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            pruned[k] = v
        else:
            dropped.append(k)

    info = model.load_state_dict(pruned, strict=False)

    # (Optional) Log what we dropped once, for sanity
    if dropped:
        keep_samples = [x for x in dropped if x.startswith("classifier.6")][:2]
        print(f"[load] Dropped {len(dropped)} mismatched keys. Examples: {keep_samples}")

    # Quick safety: most conv layers should have loaded
    conv_keys = [f"features.{i}.{p}" for i in (0,3,6,8,10) for p in ("weight","bias")]
    matched = sum(int(k not in info.missing_keys) for k in conv_keys)
    if matched < int(len(conv_keys) * 0.8):
        raise RuntimeError(
            f"Too few conv params matched ({matched}/{len(conv_keys)}). "
            f"Missing examples: {info.missing_keys[:5]}"
        )
    return info



class AlexNetFeatureExtractor(nn.Module):
    CONV_IDX = {'conv1': 0, 'conv2': 3, 'conv3': 6, 'conv4': 8, 'conv5': 10}
    FC_IDX   = {'fc6': 2, 'fc7': 5, 'fc8': 6}

    def __init__(self, diet, weights: Union[str, dict] = 'imagenet', device: str = 'cpu'):
        super().__init__()

        if weights == "imagenet":
            try:
                from torchvision.models import AlexNet_Weights
                self.model = torchvision.models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
            except Exception:
                # fallback for older torchvision
                self.model = torchvision.models.alexnet(pretrained=True)
            self.weight_source = "imagenet"
            self.load_report = None

        else:
            # random or checkpoint path
            self.model = torchvision.models.alexnet(pretrained=False)
            if weights == "random":
                self.weight_source = "random"
                self.load_report = None
            else:
                # path or {'file': path}
                info = load_alexnet_weights(self.model, weights)  # your robust loader
                self.weight_source = f"checkpoint:{os.path.abspath(weights['file'] if isinstance(weights, dict) else weights)}"
                self.load_report = info

        self.model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)

        self._hook_out = {}
        self._hooks = []
        self._register_hooks()

        with torch.no_grad():
            self.param_fingerprint = sum(p.float().abs().sum().item() for p in self.model.parameters())

        # preprocess stays the same (matches AlexNet_Weights normalization)
        if diet == "diet":
            self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                RgbToGray(keep_channels=3),
                transforms.ToTensor(),
                GaussianBlur(kernel_size=11, sigma=1.76),
                GaussianNoise(std=0.25, mono=True),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        elif diet == "diet-no-noise":
            self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                RgbToGray(keep_channels=3),
                transforms.ToTensor(),
                GaussianBlur(kernel_size=11, sigma=1.76),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        if diet == "diet-no-blur":
            self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                RgbToGray(keep_channels=3),
                transforms.ToTensor(),
                GaussianNoise(std=0.25, mono=True),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        elif diet == "no-diet":
            self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        elif diet == "nayebi-diet":
            self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Resize(64, interpolation=InterpolationMode.BICUBIC, antialias=True),  # downsample (blur)
                transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),                 # upsample (sfocatura)
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])

        elif diet == "random-diet":
            self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                # RgbToGray(keep_channels=3),
                transforms.ToTensor(),
                GaussianBlur(kernel_size=3, sigma=1.),
                GaussianNoise(std=1, mono=True),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            

    def _register_hooks(self):
        for name, idx in self.CONV_IDX.items():
            self._hooks.append(self.model.features[idx].register_forward_hook(self._save(name)))
        for name, idx in self.FC_IDX.items():
            self._hooks.append(self.model.classifier[idx].register_forward_hook(self._save(name)))

    def _save(self, key: str):
        def fn(module, inp, out):
            self._hook_out[key] = out.detach()
        return fn

    @torch.no_grad()
    def forward_batch(self, pil_images: List[Image.Image], amp: bool = False) -> Dict[str, np.ndarray]:
        x = torch.stack([self.preprocess(im) for im in pil_images], dim=0).to(self.device, non_blocking=True)
        if amp and self.device.type == 'cuda':
            with torch.cuda.amp.autocast():
                _ = self.model(x)
        else:
            _ = self.model(x)

        outputs: Dict[str, np.ndarray] = {}
        for key, tensor in self._hook_out.items():
            if tensor.ndim == 4:  # conv: (N,C,H,W) -> GAP
                vec = tensor.mean(dim=(2, 3))
            else:                 # fc: (N,D)
                vec = tensor
            outputs[key] = vec.float().cpu().numpy()
        self._hook_out.clear()

        # free batch tensors
        del x
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        return outputs


def build_alexnet_design_matrices_with_dataloader(
    folder: str,
    diet: bool, # if images with data diet at inference time
    batch_size: int = 16,
    shuffle: bool = False,
    num_workers: int = 0,
    save_dir: Optional[str] = None,
    return_in_memory: bool = True,
    **kwargs_for_extractor_fn,
) -> Union[Dict[str, np.ndarray], Dict[str, str]]:
    """
    Stream images from a folder -> AlexNet -> write per-layer design matrices.
    If save_dir is provided, allocate per-layer np.memmap arrays and fill them batch-wise.
    """
    ds = PNGFolderDataset(folder)
    # collate_fn returns the batch as a list[Image.Image]
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, collate_fn=lambda batch: batch)

    extractor = AlexNetFeatureExtractor(
        weights=kwargs_for_extractor_fn.get("weights", "imagenet"),
        device=kwargs_for_extractor_fn.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        diet=diet
    )

    # warm up on first batch to size arrays
    it = iter(loader)
    try:
        first_batch = next(it)
    except StopIteration:
        raise ValueError(f"No images found in {folder!r}")

    amp = kwargs_for_extractor_fn.get("amp", True)
    warm = extractor.forward_batch(first_batch, amp=amp)
    layers_keep = kwargs_for_extractor_fn.get("layers_keep", sorted(warm.keys()))
    F = len(ds)

    # prepare output stores (memmap if save_dir, else RAM arrays)
    stores: Dict[str, np.ndarray] = {}
    paths: Dict[str, str] = {}
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for layer in layers_keep:
        D = warm[layer].shape[1]
        if save_dir:
            path = os.path.join(save_dir, f"alexnet_{layer}.mmap")
            arr = np.memmap(path, mode='w+', dtype=np.float32, shape=(F, D))
            paths[layer] = path
        else:
            arr = np.empty((F, D), dtype=np.float32)
        stores[layer] = arr

    # write the warm batch
    n0 = first_batch.shape[0] if isinstance(first_batch, torch.Tensor) else len(first_batch)
    for layer in layers_keep:
        stores[layer][0:n0] = warm[layer]

    filled = n0
    for batch in it:
        out = extractor.forward_batch(batch, amp=amp)
        n = out[next(iter(out))].shape[0]
        for layer in layers_keep:
            stores[layer][filled:filled+n] = out[layer]
        filled += n

    # Finalize and return
    if save_dir:
        for arr in stores.values():
            arr.flush()  # ensure data is on disk
        if return_in_memory:
            # map back into RAM arrays
            return {layer: np.array(arr) for layer, arr in stores.items()}
        else:
            # return file paths
            return paths

    return stores