"""
Weight Matrix Construction
==========================
Build simulation weight matrices from connectome data using
4-block (E→E, E→I, I→E, I→I) structure with sign assignment.
"""

import numpy as np

# Dale's-law sign convention for each block
BLOCK_SIGNS = {'EE': +1, 'IE': -1, 'EI': +1, 'II': -1}


def build_weight_matrix(matrix_norm, sorted_ids_eb, id_to_type, id_to_pb,
                        gain, delta7_order):
    """
    Build the full signed weight matrix W from connectome data.

    Neurons are ordered EPG-first (EB ring order), then Δ7 (custom order).
    W[i, j] = connection from neuron j → neuron i.

    Parameters
    ----------
    matrix_norm : pd.DataFrame
        Normalised connectivity matrix (neuron_id × neuron_id).
    sorted_ids_eb : list
        Neuron IDs sorted by EB wedge position.
    id_to_type : dict
        neuron_id → 'EPG' or 'Delta7'.
    id_to_pb : dict
        neuron_id → PB glomerulus label.
    gain : float or tuple (gain_epg, gain_d7)
        Synaptic weight scaling factor.  If a single float, the same
        gain is applied to all blocks.  If a 2-tuple, the first
        element scales EPG-receiving rows (E→E, I→E) and the second
        scales Δ7-receiving rows (E→I, I→I).
    delta7_order : list
        Custom ordering for Δ7 PB glomeruli.

    Returns
    -------
    W : np.ndarray (N_total, N_total)
    sim_uids : list
        Neuron IDs in simulation order.
    N_E : int
        Number of EPG neurons.
    type_ranges : dict
        {'EPG': (0, N_E), 'Delta7': (N_E, N_total)}.
    """
    # Unpack gain
    if isinstance(gain, (list, tuple)):
        gain_epg, gain_d7 = gain
    else:
        gain_epg = gain_d7 = gain

    e_ids = [uid for uid in sorted_ids_eb if id_to_type[uid] == 'EPG']
    i_ids = [uid for uid in sorted_ids_eb if id_to_type[uid] == 'Delta7']

    # Sort Δ7 by custom PB order
    pb_rank = {pb: r for r, pb in enumerate(delta7_order)}
    i_ids = sorted(i_ids, key=lambda u: (pb_rank.get(id_to_pb.get(u, '?'), 999), u))

    sim_uids = e_ids + i_ids
    N_E = len(e_ids)
    N_total = len(sim_uids)
    type_ranges = {'EPG': (0, N_E), 'Delta7': (N_E, N_total)}

    # Extract sub-matrices and assemble with signs
    EtoE = matrix_norm.loc[e_ids, e_ids].values.T
    ItoE = matrix_norm.loc[i_ids, e_ids].values.T
    EtoI = matrix_norm.loc[e_ids, i_ids].values.T
    ItoI = matrix_norm.loc[i_ids, i_ids].values.T

    W = np.block([
        [gain_epg * BLOCK_SIGNS['EE'] * EtoE, gain_epg * BLOCK_SIGNS['IE'] * ItoE],
        [gain_d7  * BLOCK_SIGNS['EI'] * EtoI, gain_d7  * BLOCK_SIGNS['II'] * ItoI],
    ])

    return W, sim_uids, N_E, type_ranges


def find_epg_indices(sim_uids, id_to_type, id_to_pb, N_E, pb_labels):
    """
    Find EPG neuron indices matching given PB glomerulus labels.

    Parameters
    ----------
    sim_uids : list
        Neuron IDs in simulation order.
    id_to_type, id_to_pb : dict
        Mapping dicts.
    N_E : int
        Number of EPG neurons.
    pb_labels : list of str
        e.g. ['L4'] or ['R1'].

    Returns
    -------
    list of int
        Indices into the rate vector.
    """
    if pb_labels == ['none'] or pb_labels is None:
        return []
    return [
        i for i, uid in enumerate(sim_uids)
        if i < N_E
        and id_to_type.get(uid) == 'EPG'
        and id_to_pb.get(uid) in pb_labels
    ]


def extract_4blocks(matrix_norm, sorted_ids, id_to_type, id_to_pb,
                    i_order=None):
    """
    Split connectivity into four E–I blocks for visualisation.

    Returns
    -------
    blocks : dict  {'E→E', 'E→I', 'I→E', 'I→I'} → pd.DataFrame
    row_labels, col_labels : dict → list of str
    """
    e_ids = [u for u in sorted_ids if id_to_type.get(u) == 'EPG']
    i_ids = [u for u in sorted_ids if id_to_type.get(u) == 'Delta7']
    if i_order:
        rank = {pb: r for r, pb in enumerate(i_order)}
        i_ids = sorted(i_ids, key=lambda u: (rank.get(id_to_pb.get(u, '?'), 999), u))

    el = [id_to_pb.get(u, '?') for u in e_ids]
    il = [id_to_pb.get(u, '?') for u in i_ids]

    blocks = {
        'E→E': matrix_norm.loc[e_ids, e_ids],
        'I→E': matrix_norm.loc[i_ids, e_ids],
        'E→I': matrix_norm.loc[e_ids, i_ids],
        'I→I': matrix_norm.loc[i_ids, i_ids],
    }
    rl = {'E→E': el, 'I→E': il, 'E→I': el, 'I→I': il}
    cl = {'E→E': el, 'I→E': el, 'E→I': il, 'I→I': il}
    return blocks, rl, cl
