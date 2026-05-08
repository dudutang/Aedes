"""
Connectome Data Loader
======================
Load and preprocess EPG–Δ7 connectivity from connectome_data.pkl.
Supports synapse-count thresholding and row-sum renormalization.
"""

import pickle
import numpy as np
import pandas as pd


def load_connectome(filepath, syn_threshold=0):
    """
    Load connectome data and optionally apply a synapse-count threshold.

    Parameters
    ----------
    filepath : str
        Path to connectome_data.pkl.
    syn_threshold : int
        Minimum synapse count for a connection to be retained.
        Connections with fewer synapses are zeroed out before
        renormalization.  Set to 0 to keep all connections.

    Returns
    -------
    data : dict
        All original keys plus thresholded 'matrix_norm_fly' and
        'matrix_norm_aedes' if syn_threshold > 0.
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    if syn_threshold > 0:
        for species in ['fly', 'aedes']:
            raw_key = f'matrix_{species}'
            norm_key = f'matrix_norm_{species}'
            if raw_key in data:
                data[norm_key] = _threshold_and_renorm(
                    data[raw_key], syn_threshold, species
                )
    return data


def _threshold_and_renorm(matrix_raw, threshold, label=''):
    """Zero connections below threshold, then mean-normalise.

    Each entry is divided by the mean of ALL entries (including
    zeros) so that the average matrix value = 1.0.  This accounts
    for both synapse strength and sparsity, putting both species
    on the same scale.
    """
    mat = matrix_raw.copy()
    before = (mat.values > 0).sum()
    mat.values[mat.values < threshold] = 0
    after = (mat.values > 0).sum()
    all_mean = mat.values.mean()
    if all_mean == 0:
        all_mean = 1.0
    print(f'  {label}: syn >= {threshold} → '
          f'{before} → {after} connections '
          f'({before - after} removed, {(before-after)/max(before,1)*100:.1f}%), '
          f'mean_all={all_mean:.2f}')
    return pd.DataFrame(
        mat.values / all_mean,
        index=mat.index, columns=mat.columns
    )


# -- Default neuron orderings -----------------------------------------------

DELTA7_ORDER_FLY = [
    'L8R1R9', 'L7R3', 'L7R2', 'L6R4', 'L6R3',
    'L5R4', 'L4R6', 'L4R5', 'L3R6', 'L3R7', 'L2R7', 'L1L9R8',
]

DELTA7_ORDER_AEDES = [
    'L1L8L9R1', 'L8R1R9', 'L7R2', 'L7R3', 'L6R3R4', 'L5R4',
    'L4L5R4R5', 'L4R5', 'L3L4R6', 'L3R7', 'L2R7', 'L1L9R8', 'L1R1R8R9',
]

EB_RING_ORDER = [
    'R1', 'L8', 'R2', 'L7', 'R3', 'L6', 'R4', 'L5',
    'R5', 'L4', 'R6', 'L3', 'R7', 'L2', 'R8', 'L1',
]


def get_species_data(data, species='fly'):
    """
    Extract sorted IDs, labels, and orderings for one species.

    Parameters
    ----------
    data : dict
        From load_connectome().
    species : str
        'fly' or 'aedes'.

    Returns
    -------
    info : dict with keys
        matrix_norm, sorted_ids_eb, sorted_ids_pb,
        id_to_type, id_to_pb, delta7_order
    """
    if species == 'fly':
        return {
            'matrix_norm': data['matrix_norm_fly'],
            'sorted_ids_eb': data['sorted_ids_fly_eb'],
            'sorted_ids_pb': data['sorted_body_ids_fly'],
            'id_to_type': data['id_to_type_fly'],
            'id_to_pb': data['id_to_pb_fly'],
            'delta7_order': DELTA7_ORDER_FLY,
        }
    else:
        return {
            'matrix_norm': data['matrix_norm_aedes'],
            'sorted_ids_eb': data['sorted_ids_aedes_eb'],
            'sorted_ids_pb': data['sorted_ids_aedes'],
            'id_to_type': data['id_to_type_aedes'],
            'id_to_pb': data['id_to_pb_aedes'],
            'delta7_order': DELTA7_ORDER_AEDES,
        }
