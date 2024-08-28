"""
Script for testing PCMCI on the given data.

This script performs the following steps:
1. Parses command-line arguments to get the data path and optional flags.
2. Loads spatial coefficients and data from the specified file.
3. Generates a dynamics matrix and true full graph from the spatial coefficients.
4. Reshapes the data for input to PCMCI.
5. Fits the PCMCI model to the data.
6. Computes F1 score and other graph metrics.
7. Saves the results to a file or prints them based on the provided flags.

Command-line arguments:
--data_path (str): Path to the input data file (required).
--plot (bool): Flag to plot the results (optional).
--print (bool): Flag to print the results instead of saving (optional).
--verbose (bool): Flag to enable verbose output (optional).

Example usage:
python benchmarking/test_PCMCI.py --data_path path/to/data.npy --print --verbose
"""

import argparse
import os
import warnings
import numpy as np
import sys
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import stable_SCM_generator as scm_gen
from graph_metrics import F1_score, get_graph_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--plot", action=argparse.BooleanOptionalAction)
parser.add_argument("--print", action=argparse.BooleanOptionalAction)
parser.add_argument("--verbose", action=argparse.BooleanOptionalAction)
args = parser.parse_args()
ERROR_MEAN = 0  # Mean of the added noise in simulation
N_VAR = 1  # Number of variables

DATA_PATH = args.data_path
PRINT = args.print
PLOT = args.plot
VERBOSE = args.verbose
if not VERBOSE:
    VERBOSE = 0
else:
    VERBOSE = 1

with open(
    DATA_PATH,
    "rb",
) as f:
    spatial_coefficients, data = np.load(f, allow_pickle=True)

SAVE_PATH_DIR = os.path.dirname(DATA_PATH)
DATA_FILENAME = os.path.basename(DATA_PATH)
GRID_SIZE = int(DATA_FILENAME.split("x")[0])

# Get true graph for later analysis
dynamics_matrix = scm_gen.create_coefficient_matrix(spatial_coefficients, GRID_SIZE)
true_graph = scm_gen.get_graph_from_coefficient_matrix(dynamics_matrix)

# Reshape data for input to PCMCI
read_data = np.ndarray([])
if len(data.shape) > 3:
    read_data = data[:, :, :, 0]  # only working with first variable for now

if len(read_data.shape) > 2:
    data = data.reshape(
        data.shape[0] * data.shape[1], data.shape[2]
    )  # reshape to N^2xtime
    data = data.transpose()  # Rows must be temporal
if data.shape[0] < data.shape[1]:
    warnings.warn(
        "More columns than rows! Either there are more variables than observations, or you need to transpose the data."
    )

# Begin PCMCI steps
parcorr = ParCorr(significance="analytic")
dataframe = pp.DataFrame(data)
pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=1)
min_tau = 1
max_tau = 1

pcmci.verbosity = 0
pc_alpha = 0.01
alpha_level = 0.01
results = pcmci.run_pcmci(
    tau_min=min_tau,
    tau_max=max_tau,
    pc_alpha=pc_alpha,
)

q_matrix = pcmci.get_corrected_pvalues(
    p_matrix=results["p_matrix"], tau_min=min_tau, tau_max=max_tau, fdr_method="fdr_bh"
)
reconstructed_graph = pcmci.get_graph_from_pmatrix(
    p_matrix=q_matrix,
    alpha_level=alpha_level,
    tau_min=min_tau,
    tau_max=max_tau,
)
results["graph"] = reconstructed_graph

output_object = np.array(
    [
        spatial_coefficients,
        reconstructed_graph,
        get_graph_metrics(true_graph),
        get_graph_metrics(reconstructed_graph),
        (true_graph == reconstructed_graph),
        F1_score(true_graph, reconstructed_graph),
    ],
    dtype=object,
)


if not PRINT:
    # Save to file
    SAVE_PATH = SAVE_PATH_DIR + "/results/r_" + DATA_FILENAME
    if VERBOSE:
        print("Saving data to " + SAVE_PATH)
    with open(SAVE_PATH, "wb") as f:
        np.save(f, output_object)
else:
    print(output_object)
