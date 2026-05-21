# M-CaStLe: Multivariate CaStLe

## Introduction

M-CaStLe is a multivariate generalization of CaStLe for causal discovery in high-dimensional space-time systems, enabling robust identification of both spatial and inter-variable causal dynamics. Causal discovery in gridded space-time data is fundamentally challenging: the number of spatial locations often far exceeds the number of available time points, and multiple interacting variables complicate both inference and interpretation. CaStLe (Nichol et al. 2025) addressed this "large-*p*, small-*n*" problem for univariate fields by exploiting locality (stationarity of a small Moore neighborhood) and a two-stage meta-algorithm (gathering local replicates + causal estimation). However, many scientific systems—from climate models to ecological networks—are inherently multivariate, with cross-variable couplings that the original CaStLe cannot capture.

M-CaStLe generalizes CaStLe to V-variable fields by representing each grid-cell's 3x3 Moore neighborhood over V variables as a single 9V-dimensional vector, then applying time-series causal discovery algorithms under stencil-specific link assumptions to learn a 9V x 9V local causal stencil. Once learned, that multivariate stencil is stitched across every cell of a toroidal N x N grid to reconstruct a global causal graph over N^2 * V nodes. From this structure we extract:

- the multivariate causal **stencil graph** (9V x 9V)
- a compact **reaction graph** (V x V) of aggregated variable-to-variable effects
- a **spatial summary** (9 x 9) of directional influence patterns

## Repository structure

```
├── environment.yml                         # conda environment specification
├── src/                                    # source modules and experiment scripts
│   ├── mcastle_utils.py                    # core M-CaStLe algorithm and utilities
│   ├── spatiotemporal_SCM_data_generator.py
│   ├── causal_graph_metrics.py             # F1, MCC, FDR, confusion matrix
│   ├── naive_mcastle_utils.py              # Cartesian-CaStLe baseline
│   ├── trad_CD_algs.py                     # traditional causal discovery wrappers
│   ├── ADRExperiment.py                    # ADR PDE experiment class
│   ├── matlabPDE.py                        # MATLAB PDE solver interface
│   ├── helper_functions.py                 # result filename utilities
│   ├── chain_stencil_experiment.py         # chain-stencil scaling experiment
│   ├── test_MVCaStLe_PC.py                 # M-CaStLe VAR benchmark runners
│   ├── test_MVCaStLe_PCMCI.py
│   ├── test_MVCaStLe_DYNOTEARS.py
│   ├── test_CartesianUCaStLe_PC.py         # Cartesian-CaStLe baseline runners
│   ├── test_CartesianUCaStLe_PCMCI.py
│   ├── test_CartesianUCaStLe_DYNOTEARS.py
│   ├── test_PCMCI.py                       # traditional CD baseline runners
│   ├── test_PC.py
│   ├── test_DYNOTEARS.py
│   ├── test_generate_dateset.py            # data generator unit tests
│   └── adr/                               # ADR PDE workflow
│       ├── call_MVADR.py                   # run a single ADR experiment
│       ├── compute_stencil.py              # apply M-CaStLe to ADR output
│       ├── run_batch_experiments.py        # primary ADR parameter sweep
│       ├── study_full.py                   # full angle-estimation sweep
│       ├── concentration_study.py          # concentration sensitivity sweep
│       ├── post_process_results.py         # compute metrics across results
│       ├── print_completed_experiments.py  # inspect completed experiment files
│       ├── animate_adr_results.py          # animate PDE solution
│       ├── plot_stencil_results.py         # plot saved stencil results
│       ├── plot_reduced_space_ts.py        # plot reduced-space time series
│       ├── figure_angle_error2.py          # generate angle-error figure
│       ├── unitTestADRExperiment.py        # unit tests for ADRExperiment
│       ├── ADR_Driver_func.m               # MATLAB PDE driver
│       ├── Driver_Data_Generation.m        # MATLAB data generation script
│       └── Transient_ADR_2D.m             # MATLAB 2D transient ADR solver
├── tutorials/                              # interactive tutorials
│   ├── MCaStLe_tutorial.ipynb             # end-to-end M-CaStLe walkthrough
│   ├── mcastle_vs_naive_comparison.ipynb  # M-CaStLe vs Cartesian-CaStLe
│   ├── naive_mcastle_demo.ipynb           # Cartesian-CaStLe standalone demo
│   └── mcastle_pc_timestep_scaling.ipynb  # runtime scaling with T
├── paper/                                  # figure-reproduction notebooks
│   ├── figure3_var_benchmark.ipynb        # Figures 3 and 7
│   └── figure4_figure11_adr.ipynb         # Figures 4, 10, and 11
└── data/
    └── figures/                            # pre-extracted CSVs for figure notebooks
        ├── var_benchmark_data.csv
        ├── angle_error2_data.csv
        └── adr_reaction_f1_data.csv
```

## Installation

We recommend creating an isolated conda environment:

```bash
conda env create -f environment.yml
conda activate mcastle
```

If you do not have conda, install dependencies via pip:

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy scipy matplotlib pandas xarray networkx tigramite causalnex
```

> **ADR experiments only:** the `src/adr/` workflow additionally requires MATLAB and the [MATLAB Engine API for Python](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html).

## Quick start

```python
import spatiotemporal_SCM_data_generator as dg
import mcastle_utils as ms
from tigramite.independence_tests.parcorr import ParCorr

# 1. Specify a ground-truth causal structure
spatial_coefs = dg.get_empty_coefficient_matrix(n_variables=2)
spatial_coefs[0, 1, 1][0] = 0.40   # A(center) -> A(center)
spatial_coefs[1, 0, 1][1] = 0.30   # B(north)  -> B(center)
spatial_coefs[1, 1, 1][0] = 0.35   # A(center) -> B(center)

# 2. Simulate data on a 4x4 toroidal grid
data = dg.generate_dataset(T=500, grid_size=4, spatial_coefs=spatial_coefs, random_seed=0)

# 3. Build the ground-truth stencil graph
true_stencil, true_vals = ms.get_stencil_graph_from_coefficients(spatial_coefs)

# 4. Run M-CaStLe-PC
results = ms.mv_CaStLe_PC(
    data,
    cond_ind_test=ParCorr(significance='analytic'),
    pc_alpha=0.05,
    graph_p_threshold=0.05,
    rows_inverted=True,
    fdr_method='bh',
)

# 5. Evaluate
from causal_graph_metrics import get_graph_metrics
print(get_graph_metrics(true_stencil, results['graph']))
```

See `tutorials/MCaStLe_tutorial.ipynb` for a complete walkthrough including visualization and multi-experiment evaluation.

## Reproducing paper figures

The `paper/` notebooks reproduce all paper figures from pre-extracted summary CSVs in `data/figures/` — no large simulation outputs are required.

| Notebook | Figures reproduced |
|---|---|
| `paper/figure3_var_benchmark.ipynb` | Fig. 3 (VAR benchmark F1/Precision/Recall), Fig. 7 (link extent) |
| `paper/figure4_figure11_adr.ipynb` | Fig. 4 (angle estimation), Fig. 10 (ADR reaction graph), Fig. 11 (F1 histogram) |

Output PDFs and PNGs are written to `paper/figures/`.

## Tutorials

| Notebook | Description |
|---|---|
| `tutorials/MCaStLe_tutorial.ipynb` | End-to-end walkthrough: data generation, stencil learning, visualization, evaluation |
| `tutorials/mcastle_vs_naive_comparison.ipynb` | Side-by-side comparison of M-CaStLe and Cartesian-CaStLe on matched synthetic datasets |
| `tutorials/naive_mcastle_demo.ipynb` | Standalone Cartesian-CaStLe demos including center-only and spatial (off-center) examples |
| `tutorials/mcastle_pc_timestep_scaling.ipynb` | Runtime and accuracy scaling with number of time steps T |

## Running experiments

All experiment scripts are run from `src/` (add `src/` to your `PYTHONPATH` or `cd` into it first). Each script accepts command-line arguments; run with `--help` for options.

**VAR benchmark** (synthetic SCM data):
```bash
cd src
python test_MVCaStLe_PC.py --data_path <path_to_data.npz> --print
python test_PCMCI.py       --data_path <path_to_data.npz> --print
```

**ADR workflow** (requires MATLAB):
```bash
cd src/adr
python run_batch_experiments.py          # run ADR parameter sweep
python compute_stencil.py <exp.pkl>      # apply M-CaStLe to one result
python post_process_results.py <results_dir> <experiments_dir>
python figure_angle_error2.py --input <csv> --output-dir figures/
```

## Module overview

| Module | Purpose |
|---|---|
| `mcastle_utils.py` | Core algorithm: `mv_CaStLe_PC`, `mv_CaStLe_DYNOTEARS`, stencil ↔ graph conversions, plotting, angle estimation |
| `spatiotemporal_SCM_data_generator.py` | Synthetic VAR data generation on toroidal grids with stability guarantees |
| `causal_graph_metrics.py` | Evaluation: confusion matrix, F1, MCC, FDR, graph statistics |
| `naive_mcastle_utils.py` | Cartesian-CaStLe baseline: per-variable univariate CaStLe + non-spatial inter-variable discovery |
| `trad_CD_algs.py` | Thin wrappers around Tigramite PC/PCMCI and CausalNex DYNOTEARS for baseline comparisons |
| `ADRExperiment.py` | ADR PDE experiment management: parameter sweeps, caching, save/load |
| `matlabPDE.py` | Python interface to the MATLAB ADR PDE solver |
| `helper_functions.py` | Filename utilities for embedding hyperparameters in result file paths |
