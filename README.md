# Connectome-Constrained Ring Attractor: *Drosophila* vs *Aedes*

Comparative simulation of heading-direction circuit dynamics using connectome-derived synaptic weight matrices for *Drosophila melanogaster* and *Aedes aegypti*.

## Overview

The insect central complex maintains an internal heading representation via a **ring attractor** — a bump of neural activity in EPG (compass) neurons stabilised by recurrent excitation and global inhibition from Δ7 neurons. This project tests whether the *Aedes aegypti* connectome can support ring attractor dynamics comparable to *Drosophila*.

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/aedes_CX_modeling.git
cd aedes_CX_modeling

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the notebook
jupyter notebook ring_attractor_analysis.ipynb
```

Then **run all cells** from top to bottom. The notebook is self-contained — all parameters are defined in a single `CONFIG` cell at the top.

## Project Structure

```
aedes_CX_modeling/
├── README.md
├── requirements.txt
├── data/
│   └── connectome_data.pkl          # EPG–Δ7 connectivity (Fly + Aedes)
├── cx_model/                        # Python package
│   ├── __init__.py                  # Public API & re-exports
│   ├── activation.py                # φ(x): threshold-linear & sigmoid
│   ├── data_loader.py               # Load & preprocess connectome data
│   ├── weights.py                   # W matrix construction (Dale's law signs)
│   ├── simulation.py                # Rate-model Euler integration engine
│   ├── analysis.py                  # Bump decoding & stability sweeps
│   └── visualization.py             # Kymographs, heatmaps, profiles
├── ring_attractor_analysis.ipynb    # Main analysis notebook
└── ring_attractor_analysis_colab.ipynb  # Google Colab version
```

## Notebook Sections

| # | Section | What it does |
|---|---------|-------------|
| 1 | **Configuration** | All tuneable parameters (gains, timing, thresholds) |
| 2 | **Load & Inspect** | Load connectome, visualise Raw → Global Fraction → Filtered connectivity |
| 3 | **Stability Sweep** | Baseline kymographs (16 glomeruli) + 17-condition perturbation sweep |
| 4 | **Bump Shift** | EPG firing profiles at T=end for each perturbation |
| 5 | **Full Stimulus Matrix** | Exhaustive 16×17×2 = 544 simulation sweep |

## Model

Rate-based leaky integrator with Dale's-law-signed weights (Euler integration):

```
τ_k · dr_i/dt = −r_i + φ( Σ_j W_ij · r_j + I_ext_i )
```

**Dale's law**: EPG neurons are excitatory (W > 0 for all outgoing connections), Δ7 neurons are inhibitory (W < 0). The raw synapse counts from the connectome are converted to signed weights via block signs:

| Block | Sign | Meaning |
|-------|------|---------|
| E→E | +1 | EPG excites EPG |
| E→I | +1 | EPG excites Δ7 |
| I→E | −1 | Δ7 inhibits EPG |
| I→I | −1 | Δ7 inhibits Δ7 |

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `gain_epg_*` | EPG synaptic weight scaling | 20–30 |
| `gain_d7_*` | Δ7 synaptic weight scaling | 10–20 |
| `r_max` | Maximum firing rate (Hz) | 200 |
| `theta` | Activation threshold | 0 |
| `tau` | Membrane time constant (s) | 0.05 |
| `syn_thresh` | Min raw synapse count to keep | 4 |

### Normalisation Pipeline

1. **Raw**: Integer synapse counts from connectome reconstruction
2. **Global Fraction**: `raw[i,j] / total_output_j` — fraction of neuron j's total output
3. **Filtered**: Global fraction with connections below `syn_thresh` raw synapses zeroed

The **filtered global fraction** matrix is used for all simulations.

## Data

`data/connectome_data.pkl` contains preprocessed EPG–Δ7 connectivity for both species:

- **Drosophila**: from FlyWire whole-brain connectome
- **Aedes aegypti**: from CAVE/BANC brain connectome

The pickle file includes raw synapse count matrices, normalised matrices, neuron IDs, cell-type labels, and PB glomerulus assignments.

## References

- Kakaria & de Bivort (2017). *Ring attractor dynamics in the Drosophila central body.* Frontiers in Computational Neuroscience.
- Turner-Evans et al. (2017). *Angular velocity integration in a fly heading circuit.* eLife.
- Pisokas, Heinze & Webb (2020). *The head direction circuit of two insect species.* eLife.
- Hulse et al. (2021). *A connectome of the Drosophila central complex.* eLife.

## License

Research use only. Please cite this repository if you use the code or data.
