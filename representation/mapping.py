import numpy as np

import pandas as pd
import itertools

from utils import get_areas, get_specimen_ids, get_trials
from neural_maps import (
    pls_corrected_single_source_to_B,
    pls_corrected_pooled_source_to_B,
    pls_corrected_model_to_B,
)

from utils import load_memmap
import os

def consistency_across_trials(index_df):
    '''
    Consistency across trials within the same area and same specimen ID (each with itself).
    '''
    consistency_list = []
    
    for area in get_areas(index_df):
        specimen_ids = get_specimen_ids(index_df, area)
        
        for specimen_id in specimen_ids:
            sid_trials = get_trials(index_df, specimen_id, area)

            sim_sid_mean, sim_sid_std = pls_corrected_single_source_to_B(
                sid_trials,
                sid_trials,
                n_components=25,
                n_boot=100,
                seed=0
            )

            consistency_list.append({
                'Area': area,
                'Specimen ID': specimen_id,
                'Mean': sim_sid_mean,
                'Std': sim_sid_std
            })

    return pd.DataFrame(consistency_list)


def interanimal_consistency_1v1(index_df, n_boot=100, n_splits=10, n_components=25):
    '''    
    Compute inter-animal consistency for each area.
    '''
    consistency_list = []
    
    for area in get_areas(index_df):
        specimen_ids = get_specimen_ids(index_df, area)

        for spec_id1, spec_id2 in itertools.combinations(specimen_ids, 2):
            trials_s1 = get_trials(index_df, spec_id1, area)
            trials_s2 = get_trials(index_df, spec_id2, area)

            sim_sid_mean, sim_sid_std = pls_corrected_single_source_to_B(
                trials_s1,
                trials_s2,
                n_components=n_components,
                n_boot=n_boot,
                n_splits=n_splits,
                seed=0
            )

            consistency_list.append({
                'Area': area,
                'Spec1': spec_id1,
                'Spec2': spec_id2,
                'Mean': sim_sid_mean,
                'Std': sim_sid_std
            })

    return pd.DataFrame(consistency_list)

def interanimal_consistency_pool(index_df, n_boot=100, n_splits=10, n_components=25):
    """
    Compute inter-animal consistency for each area, using pooled-source approach.
    """

    consistency_list = []

    for area in get_areas(index_df):
        spec_ids = get_specimen_ids(index_df, area)
        for B in spec_ids:
            # source list (A != B)
            sources = [s for s in spec_ids if s != B]
            source_trials_list = [get_trials(index_df, s, area) for s in sources]  # [(Ti, F, pi), ...]
            YB_trials = get_trials(index_df, B, area)                               # (TB, F, q)

            mean_B, std_B = pls_corrected_pooled_source_to_B(
                source_trials_list, 
                YB_trials,
                n_components=n_components,
                n_splits=n_splits,
                n_boot=n_boot,
                seed=0,
            )
            
            consistency_list.append({
                'Area': area,
                'Target Specimen ID': B,
                'Mean': mean_B,
                'Std': std_B
            })

    return pd.DataFrame(consistency_list)

def compute_all_layer_scores(X_layers, index_df, n_boot: int = 100, n_splits: int = 10, verbose: bool = False, n_components: int = 25, test_areas: list = None, test_layers: list = None):

    all_layer_scores_list = []

    for layer_name, layer_path in X_layers.items():
        
        # Skip layers not in test_layers if specified
        if test_layers is not None and layer_name not in test_layers:
            if verbose:
                print(f"Skipping layer {layer_name} (not in test_layers)")
            continue
        
        F = 118  # number of stimuli (118)
        
        # Allow either file paths (memmap on disk) or in-memory arrays
        if isinstance(layer_path, (str, os.PathLike)):
            file_size = os.path.getsize(layer_path)
            D = file_size // (F * np.dtype(np.float32).itemsize)
            shape = (F, D)
            X_model = load_memmap(layer_path, shape=shape)
        else:
            # Assume it's a numpy-like array already in memory
            X_model = np.asarray(layer_path, dtype=np.float32)
            if X_model.ndim != 2 or X_model.shape[0] != F:
                raise ValueError(f"Layer '{layer_name}' has invalid shape {X_model.shape}; expected (F={F}, D)")

        areas = get_areas(index_df)
        
        # Filter areas if test_areas specified
        if test_areas is not None:
            areas = [area for area in areas if area in test_areas]
            if verbose:
                print(f"Filtered areas: {areas}")
        
        for area in areas:
            spec_ids = get_specimen_ids(index_df, area)
            
            for B in spec_ids:
                
                Y_trials = get_trials(index_df, B, area)  # (T, F, q)

                score, _ = pls_corrected_model_to_B(
                    X_model, 
                    Y_trials, 
                    n_components=n_components,
                    n_splits=n_splits,
                    n_boot=n_boot,
                    seed=0
                )
                    
                # Optional verbose logging
                if verbose:
                    print(f"Layer: {layer_name}, Area: {area}, Specimen: {B}, Score: {score:.4f}")
                
                all_layer_scores_list.append({
                    'layer': layer_name,
                    'area': area,
                    'specimen_id': B,
                    'score': score
                })

    return pd.DataFrame(all_layer_scores_list)


def compute_area_scores(index_model, index_df, n_boot: int = 100, n_splits: int = 10, verbose: bool = False, n_components: int = 25, test_areas: list = None, test_layers: list = None, save = False, model_name: str = 'ImageNet'):    
    layer_scores = compute_all_layer_scores(index_model, index_df, n_boot=n_boot, n_splits=n_splits, verbose=verbose, n_components=n_components, test_areas=test_areas, test_layers=test_layers)
    median_scores = layer_scores.groupby(['area', 'layer'])['score'].median().reset_index()
    sem_scores = layer_scores.groupby(['area', 'layer'])['score'].sem().reset_index()
    median_scores = pd.merge(median_scores, sem_scores.rename(columns={'score': 'sem'}), on=['area', 'layer'])
    # save results in a file
    if save:
        layer_scores.to_pickle(f'layer_scores_{model_name}.pkl')
        median_scores.to_pickle(f'median_scores_{model_name}.pkl')
    return layer_scores, median_scores