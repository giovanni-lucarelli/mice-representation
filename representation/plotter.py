import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm


def resolve_checkpoint_path(checkpoint_dir: str) -> str:
    """
    Helper function to resolve checkpoint directory paths from the representation directory.
    
    Parameters
    ----------
    checkpoint_dir : str
        Path to checkpoint directory (can be relative or absolute)
        
    Returns
    -------
    str
        Resolved absolute path to checkpoint directory
    """
    checkpoint_path = Path(checkpoint_dir)
    
    # If it's already absolute and exists, return it
    if checkpoint_path.is_absolute() and checkpoint_path.exists():
        return str(checkpoint_path)
    
    # If it's relative, try different base paths
    if not checkpoint_path.is_absolute():
        # Try relative to current directory
        if checkpoint_path.exists():
            return str(checkpoint_path.resolve())
        
        # Try relative to parent directory (if we're in representation/)
        parent_path = Path("..") / checkpoint_path
        if parent_path.exists():
            return str(parent_path.resolve())
        
        # Try relative to project root
        project_root_path = Path("../..") / checkpoint_path
        if project_root_path.exists():
            return str(project_root_path.resolve())
    
    # If nothing works, return the original path and let the caller handle the error
    return str(checkpoint_path)

def plot_comparison(median_scores_random, median_scores_inet, metric_name):
    # Add a 'model' column to each dataframe to distinguish them
    median_scores_random['model'] = 'Random'
    median_scores_inet['model'] = 'ImageNet'

    # Concatenate the two dataframes
    combined_scores = pd.concat([median_scores_random, median_scores_inet], ignore_index=True)

    # Define the order of layers for a more intuitive plot
    layer_order = sorted(combined_scores['layer'].unique(), key=lambda x: int(x.replace('conv', '')))

    # Create a FacetGrid to generate a plot for each area, with different colors for each model
    g = sns.FacetGrid(combined_scores, col="area", hue="model", col_wrap=3, height=4, aspect=1.2, sharey=False, palette={'Random': 'blue', 'ImageNet': 'orange'})

    # Define a function to plot the line and the ribbon
    def plot_with_ribbon(data, **kwargs):
        # Sort the data by the specified layer order to ensure correct line plotting
        data = data.set_index('layer').reindex(layer_order).reset_index()
        ax = plt.gca()
        # Plot the median score as a line
        sns.lineplot(data=data, x='layer', y='score', ax=ax, **kwargs)
        # Add the SEM as a shaded ribbon
        ax.fill_between(data['layer'], data['score'] - data['sem'], data['score'] + data['sem'], alpha=0.2)

    # Map the plotting function to the FacetGrid
    g.map_dataframe(plot_with_ribbon)

    # Add a legend
    g.add_legend(title='Model')

    # Set titles and labels
    g.fig.suptitle(f'Median {metric_name} Score by Layer for Each Brain Area (Random vs. ImageNet)', y=1.03, fontsize=16)
    g.set_titles("Area: {col_name}")
    g.set_axis_labels("Layer", f"Median {metric_name} Score")

    # Improve readability of x-axis labels
    g.set_xticklabels(rotation=45)

    # Adjust layout and display the plot
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def plot_comparison_multi(scores_list, names=None, metric_name='CKA'):
    """
    Plot a CKA comparison across multiple models.

    Parameters
    ----------
    scores_list : list[pd.DataFrame]
        List of DataFrames, each containing columns: 'layer', 'area', 'score', 'sem'.
    names : list[str] | None
        List of names corresponding to entries in scores_list. If None, defaults to 'Model 1', ...
    metric_name : str
        Name of the metric to display on titles/labels.
    """
    if not isinstance(scores_list, (list, tuple)) or len(scores_list) == 0:
        raise ValueError("scores_list must be a non-empty list of DataFrames")

    n_models = len(scores_list)
    if names is None:
        names = [f"Model {i+1}" for i in range(n_models)]
    if len(names) != n_models:
        raise ValueError("Length of names must match length of scores_list")

    # Create labeled copies and concatenate
    labeled_scores = [df.copy().assign(model=name) for df, name in zip(scores_list, names)]
    combined_scores = pd.concat(labeled_scores, ignore_index=True)

    # Define the order of layers for a more intuitive plot
    try:
        layer_order = sorted(
            combined_scores['layer'].unique(),
            key=lambda x: int(''.join(ch for ch in str(x) if ch.isdigit()))
        )
    except Exception:
        layer_order = list(sorted(combined_scores['layer'].unique()))

    # Build a palette for the provided names
    colors = sns.color_palette("tab10", n_colors=n_models)
    palette_map = {name: color for name, color in zip(names, colors)}

    # Create a FacetGrid to generate a plot for each area, with different colors for each model
    g = sns.FacetGrid(
        combined_scores,
        col="area",
        hue="model",
        col_wrap=3,
        height=4,
        aspect=1.2,
        sharey=False,
        palette=palette_map
    )

    # Define a function to plot the line and the ribbon
    def plot_with_ribbon(data, **kwargs):
        # Sort the data by the specified layer order to ensure correct line plotting
        data = data.set_index('layer').loc[layer_order].reset_index()
        ax = plt.gca()
        # Plot the median score as a line
        sns.lineplot(data=data, x='layer', y='score', ax=ax, **kwargs)
        # Add the SEM as a shaded ribbon
        ax.fill_between(data['layer'], data['score'] - data['sem'], data['score'] + data['sem'], alpha=0.2)

    # Map the plotting function to the FacetGrid
    g.map_dataframe(plot_with_ribbon)

    # Add a legend
    g.add_legend(title='Model')

    # Set titles and labels
    g.fig.suptitle(
        f'Median {metric_name} Score by Layer for Each Brain Area',
        y=1.03,
        fontsize=16
    )
    g.set_titles("Area: {col_name}")
    g.set_axis_labels("Layer", f"Median {metric_name} Score")

    # Improve readability of x-axis labels
    g.set_xticklabels(rotation=45)

    # Adjust layout and display the plot
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def find_checkpoint_files(checkpoint_dir: str) -> List[Tuple[str, int]]:
    """
    Find all checkpoint files in a directory and extract epoch numbers.
    
    Parameters
    ----------
    checkpoint_dir : str
        Path to the checkpoint directory (should be resolved by caller)
        
    Returns
    -------
    List[Tuple[str, int]]
        List of (checkpoint_path, epoch_number) tuples, sorted by epoch
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        raise ValueError(f"Checkpoint directory {checkpoint_dir} does not exist")
    
    checkpoint_files = []
    
    # Find all .pth files and skip best_model.pth
    for pth_file in checkpoint_dir.glob("*.pth"):
        if pth_file.name == "best_model.pth":
            # Explicitly skip best_model from sweeps
            continue
        if pth_file.name.startswith("checkpoint_epoch_"):
            # Extract epoch number from filename
            try:
                epoch_str = pth_file.name.replace("checkpoint_epoch_", "").replace(".pth", "")
                epoch = int(epoch_str)
                checkpoint_files.append((str(pth_file), epoch))
            except ValueError:
                print(f"Warning: Could not parse epoch from {pth_file}")
                continue
    
    # Sort by epoch number
    checkpoint_files.sort(key=lambda x: x[1])
    return checkpoint_files


def analyze_checkpoints(
    checkpoint_dir: str,
    image_folder: str,
    index_csv_path: str,
    metric: str = "CKA",
    layers_keep: Optional[List[str]] = None,
    batch_size: int = 16,
    num_workers: int = 12,
    device: str = "cuda",
    save: bool = True,
    save_dir: Optional[str] = None,
    chunk_size: int = 30000,
    n_boot: int = 5,
    n_splits: int = 5,
):
    """
    Minimal checkpoint sweep with tqdm prints.
    Returns (all_scores, median_scores, best_info).
    """
    from alex_extractor import build_alexnet_design_matrices_with_dataloader
    from mapping import compute_area_scores
    from utils import load_index

    if layers_keep is None:
        layers_keep = ["conv1", "conv2", "conv3", "conv4", "conv5"]

    ckpt_dir = resolve_checkpoint_path(checkpoint_dir)
    ckpts = find_checkpoint_files(ckpt_dir)
    if not ckpts:
        raise ValueError(f"No checkpoints in {ckpt_dir}")

    index_df = load_index(index_csv_path)

    per_ckpt = []  # list of (layer_scores, median_scores, epoch, name)
    base_tag = Path(ckpt_dir).name
    for path, epoch in tqdm(ckpts, desc="Checkpoints", unit="ckpt"):
        try:
            if save and save_dir is None:
                save_dir = f"alex-simple-{base_tag}-e{epoch}"
            if not save:
                save_dir = None
            
            mats = build_alexnet_design_matrices_with_dataloader(
                folder=image_folder,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                weights={"file": path},
                device=device,
                amp=True,
                layers_keep=layers_keep,
                save_dir=save_dir,
                return_in_memory=False,
            )
            layer_scores, med = compute_area_scores(mats, index_df, metric, chunk_size=chunk_size, n_boot=n_boot, n_splits=n_splits)
            layer_scores = layer_scores.assign(checkpoint=Path(path).name, epoch=epoch)
            med = med.assign(checkpoint=Path(path).name, epoch=epoch)
            per_ckpt.append((layer_scores, med, epoch, path))
        except Exception as e:
            tqdm.write(f"skip {Path(path).name}: {e}")

    if not per_ckpt:
        raise RuntimeError("No successful checkpoint analyses")

    all_scores = pd.concat([x[0] for x in per_ckpt], ignore_index=True)
    med_all = pd.concat([x[1] for x in per_ckpt], ignore_index=True)

    overall = med_all.groupby(["checkpoint", "epoch"])['score'].mean().reset_index()
    best = overall.sort_values('score', ascending=False).iloc[0]
    best_info = {
        "checkpoint_name": best["checkpoint"],
        "epoch": int(best["epoch"]),
        "overall_score": float(best["score"]),
        "checkpoint_path": next(p for _, _, ep, p in per_ckpt if ep == best["epoch"]),
    }

    return all_scores, med_all, best_info


# --- Plots over checkpoint sweeps ---
def plot_checkpoint_comparison(
    all_median_scores: pd.DataFrame,
    metric_name: str = "CKA",
    max_checkpoints: int = 20,
    figsize: Tuple[int, int] = (15, 10)
):
    """
    Plot comparison of multiple checkpoints across brain areas and layers.
    Expects a DataFrame with columns: ['checkpoint','epoch','area','layer','score','sem'].
    """
    # Select top performing checkpoints
    checkpoint_performance = all_median_scores.groupby(['checkpoint', 'epoch'])['score'].mean().reset_index()
    top_checkpoints = checkpoint_performance.nlargest(max_checkpoints, 'score')

    # Filter data to only include top checkpoints
    top_checkpoint_names = set(top_checkpoints['checkpoint'].tolist())
    filtered_scores = all_median_scores[all_median_scores['checkpoint'].isin(top_checkpoint_names)].copy()

    # Define layer order
    try:
        layer_order = sorted(
            filtered_scores['layer'].unique(),
            key=lambda x: int(''.join(ch for ch in str(x) if ch.isdigit()))
        )
    except Exception:
        layer_order = list(sorted(filtered_scores['layer'].unique()))

    # Create color palette
    n_checkpoints = len(top_checkpoint_names)
    colors = sns.color_palette("tab10", n_colors=max(1, n_checkpoints))
    palette_map = {name: color for name, color in zip(top_checkpoint_names, colors)}

    # Create the plot
    plt.figure(figsize=figsize)
    g = sns.FacetGrid(
        filtered_scores,
        col="area",
        hue="checkpoint",
        col_wrap=3,
        height=4,
        aspect=1.2,
        sharey=False,
        palette=palette_map
    )

    def plot_with_ribbon(data, **kwargs):
        data = data.set_index('layer').loc[layer_order].reset_index()
        ax = plt.gca()
        sns.lineplot(data=data, x='layer', y='score', ax=ax, **kwargs)
        if 'sem' in data:
            ax.fill_between(data['layer'], data['score'] - data['sem'], data['score'] + data['sem'], alpha=0.2)

    g.map_dataframe(plot_with_ribbon)
    g.add_legend(title='Checkpoint')
    g.fig.suptitle(
        f'Median {metric_name} Score by Layer for Each Brain Area\n(Top {max_checkpoints} Checkpoints)',
        y=1.03,
        fontsize=16
    )
    g.set_titles("Area: {col_name}")
    g.set_axis_labels("Layer", f"Median {metric_name} Score")
    g.set_xticklabels(rotation=45)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def plot_checkpoint_evolution(
    all_median_scores: pd.DataFrame,
    metric_name: str = "CKA",
    figsize: Tuple[int, int] = (12, 8)
):
    """
    Plot the evolution of scores across epochs for each area and layer.
    Expects DataFrame with ['epoch','area','layer','score'].
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    areas = sorted(all_median_scores['area'].unique())
    layers = sorted(all_median_scores['layer'].unique())

    for i, area in enumerate(areas):
        if i >= len(axes):
            break
        ax = axes[i]
        area_data = all_median_scores[all_median_scores['area'] == area]
        for layer in layers:
            layer_data = area_data[area_data['layer'] == layer]
            if len(layer_data) > 0:
                ax.plot(layer_data['epoch'], layer_data['score'], marker='o', label=layer, linewidth=2, markersize=4)
        ax.set_title(f'Area: {area}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(f'Median {metric_name} Score')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    for i in range(len(areas), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f'Evolution of {metric_name} Scores Across Training', fontsize=16)
    plt.tight_layout()
    plt.show()