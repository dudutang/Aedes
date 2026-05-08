"""
Ring Attractor Simulation Engine
=================================
Rate-based leaky integrator for heading circuit dynamics.

Model equation (per neuron i of type k):
    τ_k · dr_i/dt = −r_i + φ(Σ_j W_ij · r_j + I_ext_i)

Based on Kakaria & de Bivort 2017 and Pisokas et al. 2020.
"""

import numpy as np
from .activation import phi_threshold_linear


def simulate_ring_attractor(W, N_total, type_ranges, tau_dict,
                            I_ext_func, T=10.0, dt=0.01,
                            phi_func=None, phi_kwargs=None,
                            r0=None, noise_sigma=0.0):
    """
    Simulate ring attractor dynamics via Euler integration.

    Parameters
    ----------
    W : np.ndarray (N_total, N_total)
        Weight matrix.  W[i, j] = connection from j → i.
    N_total : int
        Number of neurons.
    type_ranges : dict
        Cell type → (start_idx, end_idx).
    tau_dict : dict
        Cell type → membrane time constant (seconds).
    I_ext_func : callable(t, N_total) → np.ndarray
        External stimulus at time t.
    T, dt : float
        Duration and timestep (seconds).
    phi_func : callable, optional
        Activation function (default: phi_threshold_linear).
    phi_kwargs : dict, optional
        Keyword args for phi_func (default: {'theta': 0, 'r_max': 100}).
    r0 : np.ndarray, optional
        Initial rates (default: zeros).
    noise_sigma : float
        Gaussian noise σ added to drive each step.

    Returns
    -------
    times : np.ndarray (n_steps,)
    rates : np.ndarray (n_steps, N_total)
    """
    if phi_func is None:
        phi_func = phi_threshold_linear
    if phi_kwargs is None:
        phi_kwargs = {'theta': 0.0, 'r_max': 100.0}

    n_steps = int(T / dt)
    times = np.arange(n_steps) * dt
    rates = np.zeros((n_steps, N_total))

    # Per-neuron time constant vector
    tau_vec = np.ones(N_total) * 0.02
    for cell_type, (s, e) in type_ranges.items():
        if cell_type in tau_dict:
            tau_vec[s:e] = tau_dict[cell_type]

    if r0 is not None:
        rates[0] = r0.copy()

    # Euler integration
    for step in range(1, n_steps):
        t = times[step]
        r = rates[step - 1]
        drive = W @ r + I_ext_func(t, N_total)
        if noise_sigma > 0:
            drive += np.random.normal(0, noise_sigma, N_total)
        drdt = (-r + phi_func(drive, **phi_kwargs)) / tau_vec
        rates[step] = np.maximum(0.0, r + dt * drdt)

    return times, rates


def make_stimulus_func(stim_specs, N_total):
    """
    Build a stimulus function from a list of stimulus specifications.

    Parameters
    ----------
    stim_specs : list of dict
        Each dict has keys: 'indices', 't_on', 't_off', 'amplitude'.
    N_total : int
        Total neuron count.

    Returns
    -------
    callable(t, N_total) → np.ndarray
    """
    def I_ext(t, N):
        stim = np.zeros(N)
        for spec in stim_specs:
            if spec['t_on'] <= t <= spec['t_off']:
                for idx in spec['indices']:
                    stim[idx] = spec['amplitude']
        return stim
    return I_ext
