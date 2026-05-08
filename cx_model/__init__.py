"""
cx_model — Connectome-Constrained Ring Attractor Modelling
==========================================================

Comparative dynamics of *Drosophila* vs *Aedes aegypti* heading circuits.

Quick start::

    from cx_model import load_connectome, build_weight_matrix, simulate_ring_attractor

Public API
----------
Data loading:
    load_connectome, get_species_data

Weight matrix:
    build_weight_matrix, find_epg_indices, extract_4blocks

Simulation:
    simulate_ring_attractor, make_stimulus_func

Activation functions:
    phi_threshold_linear, phi_sigmoid

Analysis:
    decode_bump_angle, decode_bump_position, compute_bump_metrics,
    run_stability_sweep

Visualization:
    plot_block_heatmaps, plot_kymograph_pair, plot_epg_profiles,
    plot_stability_grid, plot_full_sweep_grid, plot_activation_comparison

Constants:
    EB_RING_ORDER, DELTA7_ORDER_FLY, DELTA7_ORDER_AEDES
"""

# Data loading
from .data_loader import (
    load_connectome,
    get_species_data,
    EB_RING_ORDER,
    DELTA7_ORDER_FLY,
    DELTA7_ORDER_AEDES,
)

# Weight matrix
from .weights import (
    build_weight_matrix,
    find_epg_indices,
    extract_4blocks,
)

# Simulation
from .simulation import (
    simulate_ring_attractor,
    make_stimulus_func,
)

# Activation functions
from .activation import (
    phi_threshold_linear,
    phi_sigmoid,
)

# Analysis
from .analysis import (
    decode_bump_angle,
    decode_bump_position,
    compute_bump_metrics,
    run_stability_sweep,
)

# Visualization
from .visualization import (
    plot_block_heatmaps,
    plot_kymograph_pair,
    plot_epg_profiles,
    plot_stability_grid,
    plot_full_sweep_grid,
    plot_activation_comparison,
)
