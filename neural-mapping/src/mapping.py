import numpy as np

import pandas as pd
import itertools

from .utils import get_areas, get_specimen_ids, get_trials, load_memmap
from .neural_maps import (
    sim_corrected_source_pair,
    sim_corrected_pooled_source_to_B,
    pls_corrected_single_source_to_B,
    pls_corrected_pooled_source_to_B,
    sim_corrected_model_to_B,
    pls_corrected_model_to_B
)

import os

def consistency_across_trials(index_df, sim_metric='RSA'):
    '''
    Consistency across trials within the same area and same specimen ID (each with itself).
    '''

    consistency_list = []
    
    for area in get_areas(index_df):
        specimen_ids = get_specimen_ids(index_df, area)
        
        for specimen_id in specimen_ids:
            sid_trials = get_trials(index_df, specimen_id, area)

            if sim_metric.upper() in ['RSA', 'CKA']:

                sim_sid_mean, sim_sid_std = sim_corrected_source_pair(
                    sid_trials,
                    sid_trials,
                    metric=sim_metric,
                    n_boot=100, 
                    seed=0
                )

            elif sim_metric.upper() == 'PLS':

                sim_sid_mean, sim_sid_std = pls_corrected_single_source_to_B(
                    sid_trials,
                    sid_trials,
                    n_components=25,
                    n_boot=100,
                    seed=0
                )

            else:
                raise ValueError(f'Unknown similarity metric: {sim_metric}. Choose from RSA, CKA, PLS.')

            consistency_list.append({
                'Area': area,
                'Specimen ID': specimen_id,
                'Mean': sim_sid_mean,
                'Std': sim_sid_std
            })

    return pd.DataFrame(consistency_list)


def interanimal_consistency_1v1(index_df, sim_metric='RSA'):
    '''    
    Compute inter-animal consistency for each area.
    '''
    consistency_list = []
    
    for area in get_areas(index_df):
        specimen_ids = get_specimen_ids(index_df, area)

        for spec_id1, spec_id2 in itertools.combinations(specimen_ids, 2):
            trials_s1 = get_trials(index_df, spec_id1, area)
            trials_s2 = get_trials(index_df, spec_id2, area)
            if sim_metric.upper() in ['RSA', 'CKA']:

                sim_sid_mean, sim_sid_std = sim_corrected_source_pair(
                    trials_s1,
                    trials_s2,
                    metric=sim_metric,
                    n_boot=100,
                    seed=0
                )

            elif sim_metric.upper() == 'PLS':

                sim_sid_mean, sim_sid_std = pls_corrected_single_source_to_B(
                    trials_s1,
                    trials_s2,
                    n_components=25,
                    n_boot=100,
                    seed=0
                )

            else:
                raise ValueError(f'Unknown similarity metric: {sim_metric}. Choose from RSA, CKA, PLS.')

            consistency_list.append({
                'Area': area,
                'Spec1': spec_id1,
                'Spec2': spec_id2,
                'Mean': sim_sid_mean,
                'Std': sim_sid_std
            })

    return pd.DataFrame(consistency_list)

def interanimal_consistency_pool(index_df, sim_metric):
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

            if sim_metric.upper() in ['RSA', 'CKA']:
                mean_B, std_B= sim_corrected_pooled_source_to_B(
                    source_trials_list, 
                    YB_trials, 
                    metric=sim_metric,
                    n_boot=100, 
                    seed=0
                )

            elif sim_metric.upper() == 'PLS':
                med_B, std_B = pls_corrected_pooled_source_to_B(
                    source_trials_list, 
                    YB_trials, 
                    n_components=25,
                    n_splits=10, 
                    n_boot=100, 
                    seed=0
                )

            else:
                raise ValueError(f'Unknown similarity metric: {sim_metric}. Choose from RSA, CKA, PLS.')
            
            consistency_list.append({
                'Area': area,
                'Target Specimen ID': B,
                'Mean': mean_B if sim_metric.upper() in ['RSA', 'CKA'] else med_B,
                'Std': std_B
            })

    return pd.DataFrame(consistency_list)


def compute_all_layer_scores(X_layers, index_df, sim_metric):

    all_layer_scores_list = []

    for layer_name, layer_path in X_layers.items():
        
        F = 118  # number of stimuli (118)
        # Infer shape from file size
        file_size = os.path.getsize(layer_path)
        D = file_size // (F * np.dtype(np.float32).itemsize)
        shape = (F, D)
        
        # load model design matrices
        X_model = load_memmap(layer_path, shape=shape)

        areas = get_areas(index_df)
        for area in areas:
            spec_ids = get_specimen_ids(index_df, area)
            
            for B in spec_ids:
                
                Y_trials = get_trials(index_df, B, area)  # (T, F, q)

                if sim_metric in ['RSA', 'CKA']:
                
                    score, _ = sim_corrected_model_to_B(
                        X_model, 
                        Y_trials, 
                        metric=sim_metric,
                        n_boot=100, 
                        seed=0
                        )
                    
                elif sim_metric == 'PLS':
                    
                    score, _ = pls_corrected_model_to_B(
                        X_model, 
                        Y_trials, 
                        n_boot=100, 
                        seed=0
                        )
                    
                else:
                    raise ValueError(f'Unknown similarity metric: {sim_metric}. Choose from RSA, CKA, PLS.')
                
                # print(f"Layer: {layer_name}, Area: {area}, Specimen: {B}, Score: {score:.4f}")
                
                all_layer_scores_list.append({
                    'layer': layer_name,
                    'area': area,
                    'specimen_id': B,
                    'score': score
                })

    return pd.DataFrame(all_layer_scores_list)


def compute_area_scores(index_model, index_df, sim_metric):    
    layer_scores = compute_all_layer_scores(index_model, index_df, sim_metric)
    median_scores = layer_scores.groupby(['area', 'layer'])['score'].median().reset_index()
    sem_scores = layer_scores.groupby(['area', 'layer'])['score'].sem().reset_index()
    median_scores = pd.merge(median_scores, sem_scores.rename(columns={'score': 'sem'}), on=['area', 'layer'])
    return layer_scores, median_scores