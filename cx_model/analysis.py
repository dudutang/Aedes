"""
Analysis Tools
==============
Bump decoding, metrics, and stability-sweep routines.
"""

import numpy as np
from .simulation import simulate_ring_attractor, make_stimulus_func
from .weights import find_epg_indices


def decode_bump_angle(rates_epg, n_epg):
    """
    Decode bump position as circular mean angle.

    Parameters
    ----------
    rates_epg : np.ndarray (n_epg,)
        EPG firing rates (single time-point).
    n_epg : int
        Number of EPG neurons.

    Returns
    -------
    angle : float
        Decoded angle in radians (−π, π].
    """
    theta = np.arange(n_epg) * (2 * np.pi / n_epg)
    cx = np.sum(rates_epg * np.cos(theta))
    cy = np.sum(rates_epg * np.sin(theta))
    return np.arctan2(cy, cx)


def decode_bump_position(rates, type_range):
    """
    Decode bump position over time via population vector.

    Returns
    -------
    positions : np.ndarray (n_steps,)   – angle in radians
    amplitudes : np.ndarray (n_steps,)  – population vector length
    """
    s, e = type_range
    N = e - s
    theta = np.arange(N) * (2 * np.pi / N)
    r = rates[:, s:e]
    cx = np.sum(r * np.cos(theta), axis=1)
    cy = np.sum(r * np.sin(theta), axis=1)
    return np.arctan2(cy, cx), np.sqrt(cx**2 + cy**2)


def compute_bump_metrics(rates, type_range, t_start=None, t_end=None,
                         times=None):
    """
    Compute ring attractor quality metrics.

    Returns
    -------
    dict with keys: peak_mean_ratio, bump_width_fwhm,
    is_ring_attractor, mean_amplitude, peak_rate, mean_rate.
    """
    s, e = type_range
    N = e - s
    if t_start is not None and times is not None:
        i0 = np.searchsorted(times, t_start)
        i1 = np.searchsorted(times, t_end) if t_end else len(times)
    else:
        i0, i1 = 0, rates.shape[0]

    r = rates[i0:i1, s:e]
    r_mean = np.mean(r, axis=0)
    peak = np.max(r_mean)
    mean = np.mean(r_mean)
    ratio = peak / mean if mean > 0 else 0.0
    fwhm = int(np.sum(r_mean >= peak / 2))

    positions, amplitudes = decode_bump_position(rates[i0:i1],
                                                 (0, N))
    # Fix: decode from sub-slice → type_range is (0, N)
    # Actually need to re-slice to match
    _rates_sub = np.zeros((i1 - i0, N))
    _rates_sub[:, :N] = rates[i0:i1, s:e]
    _, amplitudes = decode_bump_position(_rates_sub, (0, N))
    mean_amp = np.mean(amplitudes)

    return {
        'peak_mean_ratio': ratio,
        'bump_width_fwhm': fwhm,
        'is_ring_attractor': (ratio > 3.0) and (mean_amp > 0.1 * peak * N),
        'mean_amplitude': mean_amp,
        'peak_rate': peak,
        'mean_rate': mean,
    }


def run_stability_sweep(W, N_total, N_E, type_ranges, sim_uids,
                        id_to_type, id_to_pb, config, eb_ring):
    """
    Run a 17-condition sweep (1 baseline + 16 2nd-stim glomeruli).

    Parameters
    ----------
    config : dict
        Must contain: gain, r_max, theta, T, dt, tau, noise,
        stim1_pbs, stim1_t, stim1_amp, stim2_t, stim2_amp.
    eb_ring : list of str
        EB ring glomerulus order (16 entries).

    Returns
    -------
    results : dict
        glomerulus_label → {'rates_epg_final', 'rates_d7_final', ...}
    kymographs : dict
        glomerulus_label → {'times', 'rates_epg', ...}   (full time series)
    """
    phi_kw = {'theta': config['theta'], 'r_max': config['r_max']}
    tau_d = {'EPG': config['tau'], 'Delta7': config['tau']}

    s1_idx = find_epg_indices(sim_uids, id_to_type, id_to_pb, N_E,
                              config['stim1_pbs'])
    sweep_gloms = ['none'] + eb_ring
    results, kymographs = {}, {}

    for s2_pb in sweep_gloms:
        s2_idx = find_epg_indices(sim_uids, id_to_type, id_to_pb, N_E,
                                  [s2_pb])
        stim_specs = [
            {'indices': s1_idx, 't_on': config['stim1_t'][0],
             't_off': config['stim1_t'][1], 'amplitude': config['stim1_amp']},
        ]
        if s2_pb != 'none':
            stim_specs.append(
                {'indices': s2_idx, 't_on': config['stim2_t'][0],
                 't_off': config['stim2_t'][1], 'amplitude': config['stim2_amp']}
            )

        times, rates = simulate_ring_attractor(
            W=W, N_total=N_total, type_ranges=type_ranges,
            tau_dict=tau_d,
            I_ext_func=make_stimulus_func(stim_specs, N_total),
            T=config['T'], dt=config['dt'], phi_kwargs=phi_kw,
            noise_sigma=config['noise'],
        )
        results[s2_pb] = {
            'rates_epg_final': rates[-1, :N_E].copy(),
            'rates_d7_final': rates[-1, N_E:].copy(),
        }
        kymographs[s2_pb] = {
            'times': times,
            'rates_epg': rates[:, :N_E],
            'rates_d7': rates[:, N_E:],
        }

    return results, kymographs
