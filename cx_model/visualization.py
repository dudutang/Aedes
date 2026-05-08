"""
Visualization
=============
Publication-quality plotting for connectome data and simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .weights import BLOCK_SIGNS

# Reusable colour maps
YG_CMAP = LinearSegmentedColormap.from_list('yg', ['#f1c40f', '#27ae60'], N=16)


# ── Connectome block heatmaps ───────────────────────────────────────────

def plot_block_heatmaps(blocks, row_labels, col_labels, suptitle,
                        block_labels=None, vmax=None, figsize=(14, 12)):
    """
    Plot 2×2 grid of E–I connectivity blocks.

    Parameters
    ----------
    blocks : dict  {'E→E', 'E→I', 'I→E', 'I→I'} → DataFrame/ndarray
    row_labels, col_labels : dict → list of str
    suptitle : str
    block_labels : list of 4 str, optional
    vmax : float, optional

    Returns
    -------
    fig, axes
    """
    if block_labels is None:
        block_labels = ['E→E', 'E→I', 'I→E', 'I→I']

    block_names = ['E→E', 'E→I', 'I→E', 'I→I']
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    type_map = {
        'E→E': ('EPG (pre)', 'EPG (post)'),
        'E→I': ('EPG (pre)', 'Δ7 (post)'),
        'I→E': ('Δ7 (pre)', 'EPG (post)'),
        'I→I': ('Δ7 (pre)', 'Δ7 (post)'),
    }
    sign_map = {'E→E': +1, 'I→E': -1, 'E→I': +1, 'I→I': -1}

    signed = {}
    for b in block_names:
        vals = blocks[b].values if hasattr(blocks[b], 'values') else blocks[b]
        signed[b] = vals * sign_map[b]

    if vmax is None:
        vmax = max(np.abs(signed[b]).max() for b in block_names) or 0.01

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    for bname, (r, c), label in zip(block_names, positions, block_labels):
        ax = axes[r, c]
        im = ax.imshow(signed[bname], aspect='auto', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, interpolation='none')
        ax.set_title(label, fontsize=13, fontweight='bold')
        rl, cl = row_labels[bname], col_labels[bname]
        ax.set_yticks(range(len(rl)))
        ax.set_yticklabels(rl, fontsize=6)
        ax.set_xticks(range(len(cl)))
        ax.set_xticklabels(cl, fontsize=6, rotation=90)
        ax.set_ylabel(type_map[bname][0])
        ax.set_xlabel(type_map[bname][1])
        ax.grid(False)
        ax.tick_params(length=0)
        plt.colorbar(im, ax=ax, label='syn. strength')
    fig.suptitle(suptitle, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig, axes


# ── Kymographs ──────────────────────────────────────────────────────────

def plot_kymograph_pair(times, rates_fly, rates_aedes, N_E_fly, N_E_aedes,
                        stim_highlights=None, title_suffix='',
                        epg_labels_fly=None, epg_labels_aedes=None,
                        stim1_idx_fly=None, stim1_idx_aedes=None,
                        stim2_idx_fly=None, stim2_idx_aedes=None,
                        figsize=(18, 10)):
    """
    Side-by-side EPG + Δ7 kymographs for both species.

    Parameters
    ----------
    times : np.ndarray
    rates_fly, rates_aedes : np.ndarray (n_steps, N_total)
    epg_labels_fly, epg_labels_aedes : list of str, optional
        PB glomerulus label per EPG neuron.
    stim1_idx_fly, stim1_idx_aedes : list of int, optional
        EPG indices receiving 1st stimulus.
    stim2_idx_fly, stim2_idx_aedes : list of int, optional
        EPG indices receiving 2nd stimulus.
    stim_highlights : list of dict, optional
        Each dict: {'t': (t_on, t_off), 'color': str, 'label': str}

    Returns
    -------
    fig, axes  (2 rows × 2 cols)
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)

    data = [
        (rates_fly, N_E_fly, 'Drosophila', epg_labels_fly,
         stim1_idx_fly, stim2_idx_fly),
        (rates_aedes, N_E_aedes, 'Aedes', epg_labels_aedes,
         stim1_idx_aedes, stim2_idx_aedes),
    ]
    for row, (rates, N_E, name, elabels, s1_idx, s2_idx) in enumerate(data):
        N_total = rates.shape[1]
        for col, (s, e, label) in enumerate([
            (0, N_E, 'EPG'), (N_E, N_total, 'Δ7')
        ]):
            ax = axes[row, col]
            im = ax.imshow(rates[:, s:e].T, aspect='auto', origin='lower',
                           extent=[times[0], times[-1], 0, e - s],
                           cmap='hot', interpolation='none')
            plt.colorbar(im, ax=ax, label='Hz')

            # Y-axis: PB labels for EPG columns
            if col == 0 and elabels:
                ax.set_yticks([i + 0.5 for i in range(len(elabels))])
                ylabels_styled = []
                for i, lb in enumerate(elabels):
                    prefix = ''
                    if s1_idx and i in s1_idx:
                        prefix += '▶'
                    if s2_idx and i in s2_idx:
                        prefix += '◆'
                    ylabels_styled.append(f'{prefix}{lb}')
                ax.set_yticklabels(ylabels_styled, fontsize=5)
                # Highlight stim neurons with coloured bands
                if s1_idx:
                    for idx in s1_idx:
                        ax.axhspan(idx, idx + 1, facecolor='none',
                                   edgecolor='lime', lw=1.8, ls='-', alpha=0.9)
                if s2_idx:
                    for idx in s2_idx:
                        ax.axhspan(idx, idx + 1, facecolor='none',
                                   edgecolor='cyan', lw=1.8, ls='-', alpha=0.9)
                ax.set_ylabel(f'{name} EPG\n(▶=stim1  ◆=stim2)',
                              fontsize=9)
            else:
                ax.set_ylabel(f'{name} {label}')

            # Temporal stim highlights
            if stim_highlights:
                for sh in stim_highlights:
                    ax.axvspan(*sh['t'], facecolor=sh['color'], alpha=0.12)
                    for t_edge in sh['t']:
                        ax.axvline(t_edge, color=sh['color'], lw=0.8,
                                   ls='--', alpha=0.5)
            if row == 0:
                ax.set_title(f'{label} Activity {title_suffix}',
                             fontweight='bold')

    axes[-1, 0].set_xlabel('Time (s)')
    axes[-1, 1].set_xlabel('Time (s)')
    plt.tight_layout()
    return fig, axes


# ── EPG activity profiles ───────────────────────────────────────────────

def plot_epg_profiles(sweep_results, eb_ring, config, species_name,
                      ax=None):
    """
    Plot EPG activity at T=end for a sweep: baseline (black)
    + each 2nd-stim (yellow→green gradient).

    Parameters
    ----------
    sweep_results : dict
        glomerulus → {'rates_epg_final': array}
    eb_ring : list of str
    config : dict
    species_name : str
    ax : matplotlib Axes, optional

    Returns
    -------
    ax
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))

    n = len(eb_ring)
    colors = [YG_CMAP(i / max(n - 1, 1)) for i in range(n)]

    for gi, glom in enumerate(eb_ring):
        if glom in sweep_results:
            ax.plot(sweep_results[glom]['rates_epg_final'], '-',
                    color=colors[gi], alpha=0.6, lw=1.2, label=glom)

    if 'none' in sweep_results:
        ax.plot(sweep_results['none']['rates_epg_final'], 'k-',
                lw=2.5, alpha=0.9, label='baseline', zorder=10)

    ax.set_xlabel('EPG neuron index (EB order)')
    ax.set_ylabel('Rate at T=end (Hz)')
    ax.set_title(f'{species_name} — EPG at T={config["T"]}s',
                 fontweight='bold')
    ax.grid(True, alpha=0.2)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[-1:] + handles[:-1], labels[-1:] + labels[:-1],
                  fontsize=7, ncol=5, loc='upper left', framealpha=0.8)
    return ax


# ── Stability sweep grid ────────────────────────────────────────────────

def plot_stability_grid(kymographs_fly, kymographs_aedes, config,
                        eb_ring, N_E_fly, N_E_aedes,
                        epg_labels_fly=None, epg_labels_aedes=None,
                        stim1_idx_fly=None, stim1_idx_aedes=None,
                        stim2_idx_map_fly=None, stim2_idx_map_aedes=None,
                        figsize_per_row=2.5):
    """
    17-row sweep kymograph: baseline + each 2nd-stim perturbation.

    Parameters
    ----------
    epg_labels_fly, epg_labels_aedes : list of str, optional
        PB label per EPG neuron.
    stim1_idx_fly, stim1_idx_aedes : list of int, optional
        Indices of EPG neurons receiving 1st stimulus.
    stim2_idx_map_fly, stim2_idx_map_aedes : dict, optional
        glomerulus → list of int, indices of EPG neurons for each 2nd stim.

    Returns
    -------
    fig, axes
    """
    sweep_gloms = ['none'] + eb_ring
    n_rows = len(sweep_gloms)
    fig, axes = plt.subplots(n_rows, 2, figsize=(18, figsize_per_row * n_rows),
                             sharex=True)

    stim_hl = [
        {'t': config['stim1_t'], 'color': 'lime'},
        {'t': config['stim2_t'], 'color': 'cyan'},
    ]

    species_info = [
        (kymographs_fly, N_E_fly, 'Fly', epg_labels_fly,
         stim1_idx_fly, stim2_idx_map_fly),
        (kymographs_aedes, N_E_aedes, 'Aedes', epg_labels_aedes,
         stim1_idx_aedes, stim2_idx_map_aedes),
    ]

    for row, glom in enumerate(sweep_gloms):
        label = 'baseline' if glom == 'none' else f'2nd={glom}'
        for col, (kymo_dict, N_E, name, elabels, s1_idx, s2_map) in enumerate(
            species_info
        ):
            ax = axes[row, col]
            k = kymo_dict[glom]
            im = ax.imshow(k['rates_epg'].T, aspect='auto', origin='lower',
                           extent=[k['times'][0], k['times'][-1], 0, N_E],
                           cmap='hot', interpolation='none')

            # Temporal highlights
            for sh in stim_hl:
                ax.axvspan(*sh['t'], facecolor=sh['color'], alpha=0.1)

            # Arrow markers for stim1 (lime ▶) and stim2 (cyan ◆)
            t_off = (k['times'][-1] - k['times'][0]) * 0.01  # small offset
            if s1_idx:
                t1 = config['stim1_t'][0] - t_off
                for idx in s1_idx:
                    ax.plot(t1, idx + 0.5, '>',
                            color='lime', ms=7, zorder=10)

            s2_idx = (s2_map or {}).get(glom, [])
            if s2_idx:
                t2 = config['stim2_t'][0] - t_off
                for idx in s2_idx:
                    ax.plot(t2, idx + 0.5, 'D',
                            color='cyan', ms=5, zorder=10)

            # Y-axis labels
            if elabels:
                ax.set_yticks([i + 0.5 for i in range(len(elabels))])
                ylbl = []
                for i, lb in enumerate(elabels):
                    pre = ''
                    if s1_idx and i in s1_idx:
                        pre += '▶'
                    if i in s2_idx:
                        pre += '◆'
                    ylbl.append(f'{pre}{lb}')
                ax.set_yticklabels(ylbl, fontsize=4)
            if col == 0:
                ax.set_ylabel(label, fontsize=9, fontweight='bold')

            if row == 0:
                ax.set_title(f'{name} EPG (▶=stim1 ◆=stim2)',
                             fontweight='bold', fontsize=11)
            if row == n_rows - 1:
                ax.set_xlabel('Time (s)')
            ax.grid(False)
            ax.tick_params(length=0)

    g = config.get('gain', '?')
    if isinstance(g, (list, tuple)):
        gain_str = f'fly(E={g[0]},I={g[1]})'
    else:
        ge_f = config.get('gain_epg_fly', g)
        gd_f = config.get('gain_d7_fly', g)
        ge_a = config.get('gain_epg_aedes', g)
        gd_a = config.get('gain_d7_aedes', g)
        gain_str = f'fly(E={ge_f},I={gd_f}), aedes(E={ge_a},I={gd_a})'
    plt.suptitle(
        f'Stability Sweep — {gain_str}, '
        f'r_max={config["r_max"]}, θ={config["theta"]}',
        fontsize=14, fontweight='bold', y=1.001
    )
    plt.tight_layout()
    return fig, axes


# ── Full stim matrix grid ───────────────────────────────────────────────

def plot_full_sweep_grid(full_profiles, eb_ring, config, N_E_fly, N_E_aedes,
                         figsize_per_row=3):
    """
    16-row × 2-col grid: each row = different 1st stim.

    Parameters
    ----------
    full_profiles : dict
        species → stim1_pb → stim2_pb → {'epg': array}
    """
    n_init = len(eb_ring)
    colors = [YG_CMAP(i / max(n_init - 1, 1)) for i in range(n_init)]

    fig, axes = plt.subplots(n_init, 2,
                             figsize=(18, figsize_per_row * n_init),
                             sharex=True)

    for row, s1_pb in enumerate(eb_ring):
        for col, (species, N_E) in enumerate([
            ('fly', N_E_fly), ('aedes', N_E_aedes),
        ]):
            ax = axes[row, col]
            name = 'Drosophila' if species == 'fly' else 'Aedes'
            profiles = full_profiles[species].get(s1_pb, {})

            for gi, s2_pb in enumerate(eb_ring):
                if s2_pb in profiles:
                    ax.plot(profiles[s2_pb]['epg'], '-',
                            color=colors[gi], alpha=0.6, lw=1.2,
                            label=s2_pb)
            if 'none' in profiles:
                ax.plot(profiles['none']['epg'], 'k-',
                        lw=2.5, alpha=0.9, label='baseline', zorder=10)

            ax.grid(True, alpha=0.2)
            ax.tick_params(length=0)
            if col == 0:
                ax.set_ylabel(f'1st={s1_pb}', fontsize=10, fontweight='bold')
            if row == 0:
                ax.set_title(f'{name} EPG', fontsize=13, fontweight='bold')
            if row == n_init - 1:
                ax.set_xlabel('EPG neuron index (EB order)')
            if row == 0:
                h, l = ax.get_legend_handles_labels()
                if h:
                    ax.legend(h[-1:] + h[:-1], l[-1:] + l[:-1],
                              fontsize=5, ncol=5, loc='upper right',
                              framealpha=0.8)

    plt.suptitle(
        f'Full Sweep — T={config["T"]}s (gain_fly={config.get("gain_fly", "?")}, '
        f'r_max={config["r_max"]}, θ={config["theta"]})',
        fontsize=14, fontweight='bold', y=1.001
    )
    plt.tight_layout()
    return fig, axes


# ── Activation function comparison ──────────────────────────────────────

def plot_activation_comparison(r_max=100.0, theta=0.0):
    """Plot threshold-linear vs sigmoid comparison (3 panels)."""
    from .activation import phi_threshold_linear, phi_sigmoid

    x = np.linspace(-50, 200, 1000)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: threshold-linear variants
    ax = axes[0]
    ax.set_title('Threshold-Linear', fontweight='bold')
    for th in [0, 2, 5, 10]:
        ax.plot(x, np.clip(x - th, 0, r_max), '--', alpha=0.5, lw=1.5,
                label=f'θ={th}')
    ax.plot(x, np.clip(x - theta, 0, r_max), 'r-', lw=3,
            label=f'current (θ={theta})')
    ax.set_xlabel('Drive')
    ax.set_ylabel('Rate (Hz)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(-5, r_max * 1.2)

    # Panel 2: sigmoid variants
    ax = axes[1]
    ax.set_title('Sigmoid', fontweight='bold')
    for beta in [0.02, 0.05, 0.1, 0.2]:
        ax.plot(x, r_max / (1 + np.exp(-beta * (x - 50))), lw=2,
                alpha=0.8, label=f'β={beta}')
    ax.set_xlabel('Drive')
    ax.set_ylabel('Rate (Hz)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(-5, r_max * 1.2)

    # Panel 3: comparison
    ax = axes[2]
    ax.set_title('Comparison', fontweight='bold')
    y_tl = np.clip(x - theta, 0, r_max)
    y_sig = r_max / (1 + np.exp(-0.05 * (x - 30)))
    ax.plot(x, y_tl, 'r-', lw=2.5, label='Threshold-linear', alpha=0.8)
    ax.plot(x, y_sig, '#2ecc71', lw=2.5, label='Sigmoid (β=0.05)', alpha=0.8)
    ax.fill_between(x, y_tl, y_sig, alpha=0.1, color='blue',
                    where=(x > 0) & (x < 150))
    ax.set_xlabel('Drive')
    ax.set_ylabel('Rate (Hz)')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.2)
    ax.set_ylim(-5, r_max * 1.2)

    plt.suptitle('Activation Function φ(x): Drive → Firing Rate',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig, axes
