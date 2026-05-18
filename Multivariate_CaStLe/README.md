# M-CaStLe: Multivariate CaStLe

TL;DR
M-CaStLe is a multivariate extension of CaStLe for causal discovery in high-dimensional space–time systems, enabling robust identification of both spatial and inter-variable dynamics.

## Introduction  
Causal discovery in gridded space–time data is fundamentally challenging: the number of spatial locations often far exceeds the number of available time points, and multiple interacting variables can complicate both inference and interpretation. CaStLe (Nichol et al. 2025) addressed this “large-$p$, small-$n$” problem for univariate fields by exploiting locality (stationarity of a small Moore neighborhood) and a two-stage meta-algorithm (gathering local replicates + causal estimation). However, many scientific systems—from climate models to ecological networks—are inherently multivariate, with cross-variable couplings that the original CaStLe cannot capture.

M-CaStLe extends CaStLe to N-variable fields by representing each grid-cell’s 3×3 Moore neighborhood over N variables as a single 9N-dimensional vector, then applying time series causal discovery algorithms under stencil-specific link assumptions to learn a 9N×9N local causal stencil. Once learned, that multivariate stencil can be “stitched” across every cell of a toroidal grid to reconstruct a global causal graph of size (grid²·N)². From this learned structure we extract  
 • the multivariate causal **stencil graph** (9N×9N),  
 • a compact **reaction graph** (N×N) of aggregated variable-to-variable effects, and  
 • a **spatial summary** (9×9) of directional influence patterns.  

This repository implements the full M-CaStLe workflow:  
 • **Data generation** (spatiotemporal_SCM_data_generator.py) provides stable synthetic multivariate VAR/ADR-style benchmarks on toroidal grids.  
 • **Algorithmic core** (mcastle_utils.py) converts coefficient arrays ↔ stencil graphs, wraps Tigramite’s PC/PCMCI in `mv_CaStLe_PC()`, and handles mapping between local and global graphs plus plotting utilities.  
 • **Evaluation metrics** (causal_graph_metrics.py) computes confusion matrices, F1 scores, Matthews correlation, false discovery rate, and basic graph-structural statistics.  
 • **Tutorial** (tutorial/MCaStLe_tutorial.pdf) walks through defining a ground-truth stencil, simulating data, learning stencils, visualizing results, and quantitatively evaluating precision/recall, MCC, and FDR.

A ready-to-use `environment.yml` specifies all required packages (NumPy, SciPy, Matplotlib, xarray, Tigramite). Clone the repo, create the conda environment, and follow the tutorial to reproduce end-to-end multivariate spatio-temporal causal discovery with M-CaStLe.

This repository contains the full M-CaStLe source, experiment scripts, tutorials, and figure-reproduction notebooks.

## Repository structure

```
Multivariate_CaStLe/
├── environment.yml                    # conda environment specification
├── src/                               # source modules and experiment scripts
│   ├── mcastle_utils.py               # core M-CaStLe utilities
│   ├── spatiotemporal_SCM_data_generator.py
│   ├── causal_graph_metrics.py
│   ├── naive_mcastle_utils.py         # Cartesian-CaStLe baseline
│   ├── trad_CD_algs.py                # traditional causal discovery wrappers
│   ├── ADRExperiment.py               # ADR PDE experiment class
│   ├── matlabPDE.py                   # MATLAB PDE solver interface
│   ├── helper_functions.py
│   ├── test_MVCaStLe_PC.py            # VAR benchmark runners
│   ├── test_MVCaStLe_PCMCI.py
│   ├── test_MVCaStLe_DYNOTEARS.py
│   ├── test_CartesianUCaStLe_PC.py    # Cartesian-CaStLe baseline runners
│   ├── test_CartesianUCaStLe_PCMCI.py
│   ├── test_CartesianUCaStLe_DYNOTEARS.py
│   ├── chain_stencil_experiment.py
│   └── test_generate_dateset.py
├── tutorials/                         # interactive tutorials
│   ├── MCaStLe_tutorial.ipynb
│   ├── mcastle_vs_naive_comparison.ipynb
│   ├── naive_mcastle_demo.ipynb
│   └── mcastle_pc_timestep_scaling.ipynb
├── paper/                             # figure-reproduction notebooks
│   ├── figure3_var_benchmark.ipynb    # Figures 3 and 7
│   └── figure4_figure11_adr.ipynb     # Figures 4, 10, and 11
└── data/
    └── figures/                       # pre-extracted CSVs for figure notebooks
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

If you do not have conda, you can install dependencies via pip:

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy scipy matplotlib pandas xarray networkx tigramite causalnex
```

## Reproducing paper figures

The `paper/` directory contains notebooks that reproduce the paper figures using pre-extracted data from `data/figures/`. No large simulation outputs are required.

| Notebook | Figures |
|---|---|
| `paper/figure3_var_benchmark.ipynb` | Fig. 3 (VAR benchmark F1/Precision/Recall), Fig. 7 (link extent) |
| `paper/figure4_figure11_adr.ipynb` | Fig. 4 (ADR angle estimation), Fig. 10 (ADR reaction graph), Fig. 11 (F1 histogram) |

Output PDFs and PNGs are written to `paper/figures/`.

## Quick start

1.  Generate or specify a local stencil of spatial coefficients (3×3×N).  
2.  Simulate data on a toroidal grid:
    ```python
    import spatiotemporal_SCM_data_generator as dg
    spatial_coeffs, data = dg.generate_dataset(
        T=500,
        grid_size=4,
        num_variables=3,
        num_links=5,
        coefficient_min_value_threshold=0.2
    )
    ```
3.  Build the ground-truth stencil graph and value matrix:
    ```python
    import mcastle_utils as mcastle
    stencil_graph, stencil_vals = mcastle.get_stencil_graph_from_coefficients(spatial_coeffs)
    ```
4.  Run multivariate CaStLe-PC:
    ```python
    from tigramite.independence_tests.parcorr import ParCorr
    ci_test = ParCorr(significance='analytic')
    results = mcastle.mv_CaStLe_PC(
        data,
        cond_ind_test=ci_test,
        pc_alpha=0.05,
        graph_p_threshold=0.05,
        rows_inverted=True,
        allow_center_directed_links=True,
        dependencies_wrap=True,
        fdr_method='bh',
        verbose=1
    )
    learned_stencil = results['graph']
    learned_vals    = results['val_matrix']
    ```
5.  Visualize and evaluate—see the tutorial in `tutorial/MCaStLe_tutorial.ipynb` for full details.

## File summaries

### src/mcastle_utils.py  
- convert_string_assumptions_to_indices, pretty_print_link_assumptions  
- print_significant_links  
- get_stencil_graph_from_coefficients → builds a local 9N×9N stencil graph and value matrix  
- plot_stencil_graph, plot_reaction_graph, plot_grid_time_series, etc.  
- mv_CaStLe_PC: main multivariate CaStLe-PC/PCMCI wrapper  
- graph → stencil → full‐grid mapping and back  
- vector/angle utilities for summarizing spatial signatures  

### src/spatiotemporal_SCM_data_generator.py  
- reshape_multivariate_coefs, create_global_dynamics_matrix, dynamics_matrix_reshaper  
- generate_chain_matrix, get_random_stable_coefficient_matrix  
- generate_dataset: simulate multivariate grid‐cell data over time with noise  
- stability checks via eigenvalues  

### src/causal_graph_metrics.py  
- get_confusion_matrix, F1_score, false_discovery_rate  
- get_graph_metrics: basic network stats (node/edge counts, in/out degree)  
- matthews_correlation_coefficient  

### src/naive_mcastle_utils.py
Cartesian-CaStLe baseline: runs independent univariate CaStLe on each variable's spatial field, then combines intra-variable stencils with inter-variable causal links discovered by non-spatial PC/PCMCI/DYNOTEARS on spatially aggregated time series.

### src/trad_CD_algs.py
Wrappers for traditional (non-spatial) causal discovery algorithms used as baselines in the VAR benchmark.

### src/ADRExperiment.py / src/matlabPDE.py
Infrastructure for the advection-diffusion-reaction (ADR) PDE experiments. Requires MATLAB and the MATLAB Python engine.

### tutorials/
- **MCaStLe_tutorial.ipynb**: end-to-end demonstration of M-CaStLe.
- **mcastle_vs_naive_comparison.ipynb**: side-by-side comparison of M-CaStLe and Cartesian-CaStLe.
- **naive_mcastle_demo.ipynb**: standalone Cartesian-CaStLe demo.
- **mcastle_pc_timestep_scaling.ipynb**: runtime scaling with number of time steps.
