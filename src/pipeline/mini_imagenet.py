import os
from typing import Callable, Optional, Dict, Tuple, List
from PIL import Image
from torch.utils.data import Dataset

def _read_inet_labels(label_path: str) -> Dict[str, int]:
    """
    Accepts either:
      - 'rel/path/to/img.jpg <int_label>'
      - 'wnid <int_label>' for mapping synset to id (img folder name is wnid)
    Returns mapping from basename or synset to int label.
    """
    mapping = {}
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0]
                try:
                    lbl = int(parts[1])
                except ValueError:
                    continue
                mapping[key] = lbl
    return mapping

class MiniImageNet(Dataset):
    """
    Mini-ImageNet dataset with optional inet_label.txt mapping.
    Expects:
      root/
        train/
          <wnid>/*.jpg
        val/ or test/
          <wnid>/*.jpg
    label_path can map either filenames or wnids to numeric labels.
    """
    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        label_path: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        assert split in ('train', 'val', 'test')
        self.root = os.path.join(root, 'train' if split == 'train' else ('val' if os.path.isdir(os.path.join(root, 'val')) else 'test'))
        self.transform = transform
        self.label_map = _read_inet_labels(label_path) if (label_path and os.path.isfile(label_path)) else None

        self.files: List[Tuple[str, int]] = []
        for r, _, files in os.walk(self.root):
            wnid = os.path.basename(r)
            for fn in files:
                if fn.startswith('.'): continue
                p = os.path.join(r, fn)
                # label resolution priority: filename mapping > wnid mapping > alphabetic index
                if self.label_map:
                    lbl = self.label_map.get(os.path.relpath(p, root), None)
                    if lbl is None:
                        lbl = self.label_map.get(fn, None)
                    if lbl is None:
                        lbl = self.label_map.get(wnid, None)
                else:
                    lbl = None
                self.files.append((p, -1 if lbl is None else lbl))

        # If labels were not provided or incomplete, fallback to class index per wnid order to make 100 classes
        if any(lbl < 0 for _, lbl in self.files):
            wnids = sorted({os.path.basename(os.path.dirname(p)) for p, _ in self.files})
            wnid2idx = {w: i for i, w in enumerate(wnids)}
            new_files = []
            for p, lbl in self.files:
                if lbl < 0:
                    lbl = wnid2idx[os.path.basename(os.path.dirname(p))]
                new_files.append((p, lbl))
            self.files = new_files

        if verbose:
            ncls = len(set(lbl for _, lbl in self.files))
            print(f'Mini-ImageNet split={split}: {len(self.files)} images, {ncls} classes.')

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        p, lbl = self.files[idx]
        img = Image.open(p).convert('RGB')
        if self.transform is not None:
            out = self.transform(img)
            if isinstance(out, dict):
                out['lbls'] = lbl
                out['name'] = p
                return out
            return {'imgs': out, 'lbls': lbl, 'name': p}
        return {'imgs': img, 'lbls': lbl, 'name': p}