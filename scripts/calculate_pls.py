#!/usr/bin/env python3
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import warnings
from sklearn.exceptions import ConvergenceWarning

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    load_and_resolve_configs,
    ensure_dirs,
    save_resolved_config,
    seed_everything,
    select_device,
)
from src.model.utils import load_yaml

from representation.mapping import compute_area_scores
from representation.alex_extractor import build_alexnet_design_matrices_with_dataloader
from representation.plotter import plot_comparison
from representation.utils import load_index

# Suppress warnings globally
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calculate PLS scores for model representations")
    p.add_argument("--project-config", type=str, default="configs/project.yaml")
    p.add_argument("--config", type=str, required=True, help="Experiment YAML file")
    p.add_argument(
        "--set",
        type=str,
        nargs="*",
        default=[],
        help="Override keys, e.g. pls.n_components=100",
    )
    p.add_argument("--image-folder", type=str, default="../Preproc2/images/", help="Path to image folder")
    p.add_argument("--index-csv", type=str, default="../Preproc2/data/combined_index.csv", help="Path to index CSV")
    p.add_argument("--results-dir", type=str, default="", help="Directory to save results (default: project_artifacts/pls)")
    p.add_argument("--batch-size", type=int, default=256, help="Batch size for data loading")
    p.add_argument("--num-workers", type=int, default=15, help="Number of workers for data loading")
    p.add_argument("--n-boot", type=int, default=5, help="Number of bootstrap iterations")
    p.add_argument("--n-splits", type=int, default=2, help="Number of splits for cross-validation")
    p.add_argument("--pls-components", type=int, default=100, help="Number of PLS components")
    p.add_argument("--layers-keep", type=str, nargs="*", 
                   default=["pool1", "pool2", "relu3", "relu4", "pool5", "fc6_relu", "fc7_relu"],
                   help="Layers to keep for analysis")
    p.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    p.add_argument("--amp", action="store_true", help="Use automatic mixed precision")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    
    resolved, flat = load_and_resolve_configs(
        project_yaml=args.project_config,
        experiment_yaml=args.config,
        overrides=args.__dict__.get("set", []),
    )

    # Create run directories (even if PLS will save under project artifacts)
    dirs = ensure_dirs(resolved)
    flat["run_dir"] = dirs["run_dir"].as_posix()

    # Set up device and seed
    seed_everything(args.seed)
    device_str = select_device(resolved.experiment.device) if hasattr(resolved, 'experiment') else args.device
    device = device_str

    # Load raw YAML to read PLS-specific fields and model paths
    raw_cfg = load_yaml(args.config)
    pls_cfg = dict(raw_cfg.get("pls", {}) or {})
    models_cfg = dict(raw_cfg.get("models", {}) or {})

    # Resolve data and loader params (YAML overrides CLI defaults if present)
    data_cfg = dict(raw_cfg.get("data", {}) or {})
    image_folder = str(data_cfg.get("data_path", args.image_folder))
    batch_size = int(data_cfg.get("batch_size", args.batch_size))
    num_workers = int(data_cfg.get("num_workers", args.num_workers))

    # PLS params (YAML preferred, fallback to CLI args)
    n_boot = int(pls_cfg.get("n_boot", args.n_boot))
    n_splits = int(pls_cfg.get("n_splits", args.n_splits))
    pls_components = int(pls_cfg.get("n_components", args.pls_components))
    layers_keep = list(pls_cfg.get("layers_keep", args.layers_keep))
    index_csv = str(pls_cfg.get("index_csv", args.index_csv))

    # Compute project-level artifacts base (project.paths.root + project.paths.artifacts_dir)
    project_root = Path(os.path.expanduser(resolved.project.paths.root)) if hasattr(resolved.project.paths, 'root') else Path.cwd()
    project_artifacts_base = Path(os.path.expanduser(resolved.project.paths.artifacts_dir))
    if not project_artifacts_base.is_absolute():
        project_artifacts_base = (project_root / project_artifacts_base).resolve()

    # Results directory: prefer CLI if provided; else YAML; else default under project artifacts
    results_dir_arg = args.results_dir.strip()
    yaml_results = str(pls_cfg.get("results_dir", "")).strip()
    if results_dir_arg:
        results_dir = Path(os.path.expanduser(results_dir_arg)).resolve()
    elif yaml_results:
        ypath = Path(os.path.expanduser(yaml_results))
        results_dir = ypath.resolve() if ypath.is_absolute() else (project_artifacts_base / ypath).resolve()
    else:
        results_dir = (project_artifacts_base / "pls").resolve()

    # Ensure results dir exists and save the fully resolved config there
    os.makedirs(results_dir.as_posix(), exist_ok=True)
    resolved_config_path = results_dir / "resolved_config.yaml"
    save_resolved_config(flat, resolved_config_path)
    print(f"Resolved config saved to: {resolved_config_path}")

    # Model paths configuration - from YAML or fallback
    paths = models_cfg if models_cfg else {
        "random": "random",
        "supervised_no-diet": "checkpoints/supervised_no-diet/sgd-correct-no-aug/checkpoint_epoch_60.pth",
        "supervised_diet": "checkpoints/supervised_diet/sgd-correct-no-aug/checkpoint_epoch_60.pth",
    }

    models = {}
    scores = {key: {} for key in paths.keys()}

    print(f"Building design matrices for {len(paths)} models...")
    for model_name, weights_path in paths.items():
        print(f"Processing {model_name}...")
        models[model_name] = build_alexnet_design_matrices_with_dataloader(
            folder=image_folder,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            weights=weights_path,
            device=device,
            amp=args.amp,
            layers_keep=layers_keep,
        )
    
    print("Loading index data...")
    index_df = load_index(index_csv)

    # Create results directory (already ensured above, but keep idempotent)
    os.makedirs(results_dir.as_posix(), exist_ok=True)

    print("Computing PLS scores...")
    for model_name in models.keys():
        print(f"Computing scores for {model_name}...")
        layer_scores, med = compute_area_scores(
            models[model_name],
            index_df,
            n_boot=n_boot,
            n_splits=n_splits,
            n_components=pls_components,
            save=True,
            model_name=model_name,
            save_dir=results_dir.as_posix(),
        )
        scores[model_name] = {
            "layer_scores": layer_scores,
            "median_scores": med,
        }

    # Save additional PLS-specific results metadata
    pls_metadata = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": {k: str(v) for k, v in paths.items()},
        "layers_keep": layers_keep,
        "pls_n_components": pls_components,
        "n_boot": n_boot,
        "n_splits": n_splits,
        "results_dir": results_dir.as_posix(),
    }

    # Save PLS metadata as JSON in the results directory
    metadata_path = os.path.join(results_dir.as_posix(), "pls_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(pls_metadata, f, indent=4)
    
    print(f"Results saved to: {results_dir}")
    print(f"PLS metadata saved to: {metadata_path}")
    print(f"Full configuration saved to: {resolved_config_path}")


if __name__ == "__main__":
    main()

