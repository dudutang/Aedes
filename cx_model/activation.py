"""
Activation Functions
====================
Nonlinear transfer functions that map synaptic drive to firing rate.

φ(drive) → firing rate ∈ [0, r_max]
"""

import numpy as np


def phi_threshold_linear(x, theta=0.0, r_max=100.0):
    """
    Threshold-linear activation with hard saturation.

    φ(x) = clip(x − θ, 0, r_max)

    Parameters
    ----------
    x : np.ndarray
        Total drive to each neuron.
    theta : float
        Firing threshold.
    r_max : float
        Maximum firing rate (Hz).

    Returns
    -------
    np.ndarray
        Firing rates in [0, r_max].
    """
    return np.clip(x - theta, 0.0, r_max)


def phi_sigmoid(x, theta=0.0, r_max=100.0, beta=0.1):
    """
    Sigmoid activation (smooth saturation).

    φ(x) = r_max / (1 + exp(−β(x − θ)))

    Parameters
    ----------
    x : np.ndarray
        Total drive to each neuron.
    theta : float
        Inflection point (rate = r_max/2 when drive = θ).
    r_max : float
        Maximum firing rate (Hz).
    beta : float
        Steepness. Smaller → more graded; larger → sharper.

    Returns
    -------
    np.ndarray
        Firing rates in [0, r_max].
    """
    return r_max / (1.0 + np.exp(-beta * (x - theta)))
