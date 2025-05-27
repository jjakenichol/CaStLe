# CaStLe (**Ca**usal **S**pace-Time S**t**encil **Le**arning)

[![DOI](https://zenodo.org/badge/831173446.svg)](https://doi.org/10.5281/zenodo.15530556)

This repository contains the code and data necessary to reproduce the results presented in our paper titled *Space-Time Causal Discovery in Earth Systems Science: A Local Stencil Learning Approach* by J. Jake Nichol, Michael Weylandt, Diana Bull, G. Matthew Fricke, Melanie E. Moses, and Laura P. Swiler.

### Abstract

Causal discovery tools enable scientists to infer meaningful relationships from observational data, spurring advances in fields as diverse as biology, economics, and climate science. Despite these successes, the application of causal discovery to space-time systems remains immensely challenging due to the high-dimensional nature of the data. For example, in climate sciences, modern observational temperature records over the past few decades regularly measure thousands of locations around the globe. To address these challenges, we introduce **Ca**usal **S**pace-Time S**t**encil **Le**arning (**CaStLe**), a novel algorithm for discovering causal structures in complex space-time systems. CaStLe leverages regularities in local dependence to learn governing global dynamics. This local perspective eliminates spurious confounding and drastically reduces sample complexity, making space-time causal discovery practical and effective. These advances enable causal discovery of geophysical phenomena that were previously unapproachable, including non-periodic, transient phenomena such as volcanic eruption plumes. When applied to ever-larger spatial grids, CaStLe's performance actually improves because it transforms large grids into *informative spatial replicates*. We successfully apply CaStLe to discover the atmospheric dynamics governing the climate response to the 1991 Mount Pinatubo volcanic eruption. We additionally provide extensive validation experiments to demonstrate the effectiveness of CaStLe over existing causal-discovery frameworks on a range of climate-inspired benchmarks.

### Overview

The repository is organized into several directories:

- `benchmarking`: Contains scripts for testing different configurations of the CaStLe model on various datasets.
- `data`: Includes sample datasets used for testing and validation.
- `figure_generation`: Jupyter notebooks for generating figures and analyzing results.
- `src`: Source code for the core functionalities of the CaStLe algorithm, including causal discovery and graph metrics.
 
 
## Data
 
The data generated and used for our HSW-V, VAR, and PDE experiments are available on Zenodo via [this link](https://zenodo.org/records/12701546?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjZiODBjOGQ3LTc5NGMtNGZlYS1iMmZlLTM4MWY2ODk4ZjQ0MyIsImRhdGEiOnt9LCJyYW5kb20iOiI5YTZmNTY1ZjE5MzYyYWFmOGNmNzcxYTBhYWYzMjdmZCJ9.aki35C-lcVLEEbc4QCaxgvjkDIUZbzgWLkPwgnYtMOHYWtGdWKWChgtdQtxS14TqgYCuGRUwC7o8L0YZCggE-w) with GNU Lesser General Public License v3.0 or later. The data used for the E3SMv2-SPA experiments can be found in [1,2].

[1] Hunter York Brown, Benjamin Wagman, Diana Bull, Kara Peterson, Benjamin Hillman, Xiaohong Liu, Ziming Ke, and Lin Lin. 2024. Validating a microphysical prognostic stratospheric aerosol implementation in E3SMv2 using observations after the Mount Pinatubo eruption. Geoscientific Model Development 17, 13 (2024), 5087-5121. https://doi.org/10.5194/gmd-17-5087-2024.

[2] Tom Ehrmann, Benjamin Wagman, Diana Bull, Hunter York Brown, Benjamin Hillman, Kara Peterson, Laura Swiler, Jerry Watkins, and Joseph Hart. 2024. Identifying Northern Hemisphere Temperature Responses to the Mt. Pinatubo Eruption through Limited Variability Ensembles. To be submitted to Climate Dynamics 17, 13 (2024), 5087-5121. https://doi.org/10.5194/cd-17-5087-2024.
 
## Installation Instructions

To set up your environment with the necessary libraries, install the following packages using your preferred package manager (`conda` or `pip`):

- `python=3.9`,`3.10`
- `cartopy`
- `dask`
- `dcor`
- `matplotlib>=3.7.0`
- `numpy<1.24,>=1.18`
- `pandas`
- `scipy>=1.10.0`
- `seaborn>=0.12.2`
- `statsmodels`
- `xarray`
- `numba=0.56.4`
- `networkx>=3.0`
- `colorcet` # for plotting

```sh
conda install cartopy dask dcor "matplotlib>=3.7.0" numpy pandas "scipy>=1.10.0" seaborn statsmodels xarray numba=0.56.4 "networkx>=3.0"
```

The `clif` package is necessary for working with E3SM and HSW-V data:

```sh
pip install git+https://github.com/sandialabs/clif.git
```

The `causalnex` package is necessary for running DYNOTEARS tests:

```sh
pip install git+https://github.com/mckinsey/causalnex.git
```

The `matlab_engine` package must be installed to generate Burgers PDE experiments:
```sh
cd "matlabroot\extern\engines\python"
python setup.py install
```

Finally, `Tigramite` is necessary for other causal discovery tests, and should be installed last, which can be found here: https://github.com/jakobrunge/tigramite.

## Using this Code

### Source Code

The `src` directory contains the core functionalities of the CaStLe algorithm, including causal discovery and graph metrics. Key files include:

- `stencil_functions.py`: Contains the minimum code to run CaStLe using various causal discovery algorithms for the Parent Identification Phase (PIP). It includes implementations for CaStLe-PC, CaStLe-PCMCI, and other supporting functions.
- `graph_metrics.py`: Includes functions for computing graph metrics and evaluation measures related to causal graphs.
- `stable_SCM_generator.py`: Provides functions to generate stable Structural Causal Models (SCMs) for spatiotemporal datasets.
 
### Benchmarking Scripts

The `benchmarking` directory contains several scripts for testing different configurations of the CaStLe model. Each script follows a similar structure and can be run from the command line with various options.

Example usage:
```bash
python benchmarking/test_CaStLe_PC.py --data_path path/to/data.npy --print --verbose --time_alg
```

#### Generating 2D SCM test data with `generate_SCM_data.py`

This script generates spatiotemporal data based on a 2D structural causal model using vector autoregression.

The script utilizes the `stable_SCM_generator` module to generate random stable coefficient matrices based on the specified parameters. It then initializes the data array and runs a simulation loop to generate the spatiotemporal data by applying the coefficient matrix to the previous time step's data, incorporating noise.

The script takes command-line arguments to specify various parameters such as the number of time samples (T), the dimension of the square grid (GRID_SIZE), the density of the desired coefficient matrix (DEPENDENCE_DENSITY), the minimum value of the coefficient matrix (MIN_VALUE), the standard deviation of the added noise in simulation (ERROR_SIGMA), the number of experimental repetitions (NUM_REPETITION), the save path prefix for the output file (SAVE_PATH_PREFIX), and the verbosity level (VERBOSE).

The generated data is saved to a file in the NumPy binary format (.npy) with a unique filename based on the specified parameters. If no save path prefix is provided, the default save path is used.

Example usage:
```bash
python generate_SCM_data.py --t <number_of_time_samples> --grid_size <dimension_of_square_grid> --dependence_density <density_of_coefficient_matrix> --min_value <minimum_value_of_coefficient_matrix> --error_sigma <standard_deviation_of_noise> [--num_repetition <number_of_repetitions>] [--save_path_prefix <save_path_prefix>] [--verbose <verbosity_level>]
```

### Figure Generation

The `figure_generation` directory contains Jupyter notebooks for generating figures and analyzing results. These notebooks can be opened and run in a Jupyter environment.
