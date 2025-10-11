# M-CaStLe: Multivariate CaStLe Toolbox

M-CaStLe (Multivariate CaStLe) is a Python toolkit for  
1. generating synthetic multivariate spatio-temporal data on a grid,  
2. representing local interactions via a “Moore neighborhood” stencil,  
3. discovering causal structure in that reduced space via time series causal discovery,  
4. mapping learned stencils back to a full global graph, and  
5. computing evaluation metrics (F1, MCC, FDR, graph-level stats).

This repository contains three core modules under `src/`, plus a self-contained tutorial in PDF form under `tutorial/`.

## Repository structure

• environment.yml: conda environment specification for installing all dependencies.

• src/  
  – mcastle_utils.py: core utilities for building, mapping, visualizing stencils; causal discovery wrapper (`mv_CaStLe_PC`), and helper functions (angle computations, graph mapping, etc.).  
  – spatiotemporal_SCM_data_generator.py: functions to generate synthetic multivariate spatial-temporal data (SCM), ensure stability, create global dynamics matrices, etc.  
  – causal_graph_metrics.py: confusion matrix, F1 score, false discovery rate, Matthews correlation coefficient, and simple graph-structural metrics.  

• tutorial/  
  – MCaStLe_tutorial.ipynb: end-to-end tutorial demonstrating workflow: define ground-truth stencil, simulate data, learn stencil, visualize, and evaluate.

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
pip install numpy scipy matplotlib xarray tigramite
```

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

### tutorial/MCaStLe_tutorial.pdf  
Step-by-step demonstration of M-CaStLe: defining stencils, simulating data, discovering causal graphs, and computing metrics.