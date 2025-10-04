from typing import Dict, List, Optional, Sequence, Union, Iterator, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision
from torchvision import transforms

import glob
from torch.utils.data import Dataset, DataLoader
import os, re, glob


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
class AlexNetFeatureExtractor(nn.Module):
    """
    Feature extractor for torchvision AlexNet with hooks.
    Conv layers are GAP-pooled to vectors.
    Exposes: 'conv1','conv2','conv3','conv4','conv5','fc6','fc7','fc8'
    """
    CONV_IDX = {'conv1': 0, 'conv2': 3, 'conv3': 6, 'conv4': 8, 'conv5': 10}
    FC_IDX   = {'fc6': 2, 'fc7': 5, 'fc8': 6}

    def __init__(self, weights: Union[str, dict] = 'imagenet', device: str = 'cpu'):
        super().__init__()
        # load base model
        if weights == 'imagenet':
            try:
                from torchvision.models import AlexNet_Weights
                self.model = torchvision.models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
            except Exception:
                self.model = torchvision.models.alexnet(weights=None)
        else:
            # Use torchvision AlexNet with random weights (same initialization as mouse-vision)
            self.model = torchvision.models.alexnet(weights=None)
            if isinstance(weights, dict) and 'file' in weights:
                state = torch.load(weights['file'], map_location=device)
                if 'state_dict' in state:
                    state = state['state_dict']
                state = {k.replace('module.', ''): v for k, v in state.items()}
                self.model.load_state_dict(state, strict=False)
            # else: random weights (same as mouse-vision initialization)

        self.model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)
        if self.device.type == 'cuda':
            try:
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass

        self._hook_out: Dict[str, torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
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
            # Update deprecated autocast to torch.amp.autocast("cuda", ...)
            with torch.amp.autocast("cuda"):
                _ = self.model(x)
        else:
            _ = self.model(x)

        outputs: Dict[str, np.ndarray] = {}
        for key, tensor in self._hook_out.items():
            if tensor.ndim == 4:
                # NO POOLING: use all activations by flattening spatially
                # (N, C, H, W) -> (N, C*H*W)
                vec = tensor.flatten(1)
            else:
                # fc layers already (N, D)
                vec = tensor
            outputs[key] = vec.float().cpu().numpy()

        self._hook_out.clear()
        del x
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        return outputs



def build_alexnet_design_matrices_with_dataloader(
    folder: str,
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
    _dev_str = kwargs_for_extractor_fn.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    _is_cuda = torch.device(_dev_str).type == 'cuda'
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=_is_cuda,
        collate_fn=lambda batch: batch,
    )

    extractor = AlexNetFeatureExtractor(
        weights=kwargs_for_extractor_fn.get("weights", "imagenet"),
        device=_dev_str
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
