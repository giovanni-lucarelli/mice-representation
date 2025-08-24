"""
Packaged Allen datasets loader/downloader (Nayebi et al.)

- Interface to download/load/inspect packaged PKL datasets
  provided by Nayebi et al. (`mouse-vision` release), including:
    - neuropixels: mouse_neuropixels_visual_data_with_reliabilities.pkl
    - calcium   : mouse_calcium_visual_data_with_reliabilities.pkl

Examples:
  python src/load_data.py --dataset neuropixels --download --verbose
  python src/load_data.py --dataset neuropixels --inspect --verbose
  python src/load_data.py --dataset calcium --download --verbose
"""

import argparse
import os
from typing import Optional, Dict

import logging
import urllib.request
from contextlib import closing
import pickle
import pandas as pd
from pathlib import Path
from config import NEUROPIXELS_PKL_URL, CALCIUM_PKL_URL

try:
    from tqdm import tqdm 
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable

class AllenDataLoader:
    """Manage downloads, loading, and inspection for Nayebi packaged datasets."""

    BASE_DIRNAME = "AllenData"
    URLS: Dict[str, str] = {
        "neuropixels": NEUROPIXELS_PKL_URL,
        "calcium": CALCIUM_PKL_URL,
    }
    DEFAULT_FILENAMES: Dict[str, str] = {
        "neuropixels": "neuropixels.pkl",
        "calcium": "calcium.pkl",
    }

    def __init__(self, 
                 dataset: str = "neuropixels", 
                 base_dir: Optional[str] = None,
                 force: bool = False,
                 verify: bool = True,
                 verbose: bool = True):
        assert dataset in self.URLS.keys(), f"Unknown dataset '{dataset}'. Choose from {list(self.URLS.keys())}."
        self.dataset = dataset
        self.url = self.URLS[dataset]
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        self.base_dir = base_dir or os.path.join(repo_root, self.BASE_DIRNAME)
        self.default_path = os.path.join(self.base_dir, self.DEFAULT_FILENAMES[dataset])
        self.force = force
        self.verify = verify
        self.verbose = verbose
        logging.basicConfig(level=(logging.INFO if verbose else logging.WARNING), format="%(levelname)s:%(message)s")
        logging.info(f"Initialized AllenDataLoader for {dataset} dataset")

    @staticmethod
    def _stream_download(url: str, dest_path: str, desc: Optional[str] = None, chunk_size: int = 1024 * 1024) -> str:
        os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
        with closing(urllib.request.urlopen(url)) as r:  # nosec B310
            total = int(r.headers.get("Content-Length", 0))
            pbar = tqdm(total=total if total > 0 else None, unit="B", unit_scale=True, desc=desc or os.path.basename(dest_path))
            with open(dest_path, "wb") as f:
                while True:
                    chunk = r.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))
            pbar.close()
        return dest_path

    @staticmethod
    def _remote_content_length(url: str) -> Optional[int]:
        """Return Content-Length for URL, or None if unavailable."""
        try:
            req = urllib.request.Request(url, method="HEAD")
            with closing(urllib.request.urlopen(req)) as r:  # nosec B310
                cl = r.headers.get("Content-Length")
                return int(cl) if cl is not None else None
        except Exception:
            return None

    def download(self, out_path: Optional[str] = None) -> str:
        dst = out_path or self.default_path
        # Skip if exists (optionally verify size via HEAD)
        if os.path.isfile(dst) and not self.force:
            if self.verify:
                local_size = os.path.getsize(dst)
                remote_size = self._remote_content_length(self.url)
                if remote_size is not None and local_size == remote_size:
                    logging.info(f"Found existing {self.dataset} PKL with matching size; skipping download: {dst}")
                    return dst
                else:
                    logging.info("Existing file present but size mismatch or unknown remote size; re-downloading.")
            else:
                logging.info(f"Found existing {self.dataset} PKL; skipping download (no verify): {dst}")
                return dst

        logging.info(f"Downloading {self.dataset} packaged dataset to {dst}")
        return self._stream_download(self.url, dst, desc=f"{self.dataset.capitalize()} PKL")

    def load(self, pkl_path: Optional[str] = None):
        """Load either a packaged .pkl (legacy) or a converted .zarr store.

        If the provided path (or default) ends with .zarr or is a directory with a .zarr suffix,
        it is opened via xarray.open_zarr. Otherwise we fall back to the legacy pickle loaders.
        """
        path_str = pkl_path or self.default_path
        path = Path(path_str)
        if not (path.is_file() or path.is_dir()):
            raise FileNotFoundError(f"Dataset not found at {path}. For PKL, run with --download first.")

        is_zarr = (path.suffix == ".zarr") or path.name.endswith(".zarr")
        if is_zarr:
            try:
                import xarray as xr  # local import to avoid hard dependency when unused
            except ModuleNotFoundError as e:
                raise RuntimeError(
                    "Missing dependency to load Zarr dataset ('xarray'). Install it, e.g.: pip install xarray zarr"
                ) from e
            # Try consolidated metadata first for speed; fallback silently
            try:
                return xr.open_zarr(str(path), consolidated=True)
            except Exception:
                return xr.open_zarr(str(path), consolidated=False)

        # Attempt legacy PKL with pandas first
        try:
            return pd.read_pickle(open(str(path), "rb"))
        except ModuleNotFoundError as e:
            # Common when xarray is missing; the PKL contains xarray objects
            raise RuntimeError(
                "Missing dependency to unpickle dataset (likely 'xarray'). "
                "Install it, e.g.: pip install xarray"
            ) from e
        except AttributeError as e:
            # Try a compatibility shim for renamed xarray classes, then retry stdlib pickle
            if "PandasIndexAdapter" in str(e):
                try:
                    import xarray.core.indexing as _xindex
                    # Best-effort alias to handle older pickles
                    if not hasattr(_xindex, "PandasIndexAdapter"):
                        # Try to alias to a plausible modern counterpart if available
                        alias_target = None
                        for cand in ("PandasIndex", "ExplicitIndexer"):
                            if hasattr(_xindex, cand):
                                alias_target = getattr(_xindex, cand)
                                break
                        if alias_target is not None:
                            setattr(_xindex, "PandasIndexAdapter", alias_target)  # type: ignore[attr-defined]
                    return pickle.load(open(str(path), "rb"))
                except Exception:
                    pass
            # If not resolvable, surface guidance
            raise RuntimeError(
                "Failed to unpickle due to xarray API changes ('PandasIndexAdapter'). "
                "Try installing xarray==0.15.1 and pandas==0.25.3 in a separate env, "
                "then convert the dataset to a modern format (e.g., .nc or .zarr)."
            ) from e
        except Exception:
            # Fallback to stdlib pickle
            try:
                return pickle.load(open(str(path), "rb"))
            except ModuleNotFoundError as e:
                raise RuntimeError(
                    "Missing dependency to unpickle dataset (likely 'xarray'). "
                    "Install it, e.g.: pip install xarray"
                ) from e

    def inspect(self, pkl_path: Optional[str] = None) -> None:
        data = self.load(pkl_path)
        # Dict-like legacy structure
        if hasattr(data, 'keys'):
            keys = list(data.keys())
            logging.info(f"Top-level keys: {keys}")
            if 'stimuli' in keys:
                stim = data['stimuli']
                try:
                    shape = getattr(stim, 'shape', None) or (stim.sizes if hasattr(stim, 'sizes') else None)
                    logging.info(f"stimuli: shape={shape}")
                except Exception:
                    pass
            if 'neural_data' in keys:
                nd = data['neural_data']
                try:
                    areas = list(sorted(set([str(v) for v in nd.visual_area.values])))
                    logging.info(f"neural_data: visual_areas={areas}")
                except Exception:
                    logging.info("neural_data present (xarray) -- unable to list areas with current version")
            print("Inspection complete. Enable --verbose for detailed logs.")
            return

        # xarray Dataset (e.g., opened from .zarr)
        try:
            import xarray as xr  # type: ignore
        except Exception:
            xr = None  # type: ignore
        if xr is not None and hasattr(data, 'data_vars') and hasattr(data, 'dims'):
            try:
                dims_summary = {k: int(v) for k, v in getattr(data, 'sizes', {}).items()}  # type: ignore[attr-defined]
            except Exception:
                dims_summary = dict(getattr(data, 'dims', {}))  # fallback representation
            logging.info(f"xarray.Dataset dims: {dims_summary}")
            try:
                vars_list = list(getattr(data, 'data_vars', {}).keys())
            except Exception:
                vars_list = []
            logging.info(f"xarray.Dataset variables: {vars_list}")
            # Best-effort: specific fields if present
            try:
                if 'stimuli' in data:
                    s = data['stimuli']
                    logging.info(f"stimuli: shape={getattr(s, 'shape', None)}")
                if 'neural_data' in data:
                    nd = data['neural_data']
                    va = None
                    try:
                        va = list(sorted(set([str(v) for v in nd.visual_area.values])))
                    except Exception:
                        pass
                    logging.info(f"neural_data: visual_areas={va}")
            except Exception:
                pass
            print("Inspection complete. Enable --verbose for detailed logs.")
            return

        # Fallback summary
        logging.info(f"Loaded object of type {type(data)}; unable to provide detailed inspection with current dependencies.")
        print("Inspection complete. Enable --verbose for details.")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Packaged Allen datasets (Nayebi) downloader and loader.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="neuropixels",
        choices=["neuropixels", "calcium"],
        help="Which packaged dataset to operate on",
    )
    parser.add_argument(
        "--verbose",
        default=True,
        action="store_true",
        help="Enable info-level logging for progress and debugging",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the selected packaged dataset (.pkl) and exit",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if file exists",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip remote size check; if file exists it will be kept",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path for the .pkl (defaults to AllenData/…)",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Load the packaged PKL and print a short summary",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to an existing PKL (if different from default)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=(logging.INFO if args.verbose else logging.WARNING), format="%(levelname)s:%(message)s")
    logging.info("Starting PKL dataset manager")
    mgr = AllenDataLoader(dataset=args.dataset)
    if getattr(args, "inspect", False):
        mgr.inspect(pkl_path=args.path)
        return
    if getattr(args, "download", False):
        path = mgr.download(out_path=args.out)
        print(f"Saved {args.dataset} .pkl to {path}")
        return
    # Default behavior: recommend PKL workflow
    raise SystemExit(
        "No action specified. Use --download to fetch the PKL, "
        "then load it via AllenDataLoader(...).load(), or use --inspect."
    )

if __name__ == "__main__":
    main()

