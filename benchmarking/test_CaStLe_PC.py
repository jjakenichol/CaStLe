"""
Script for testing CaStLe-PC on 2D SCM(VAR) data.

This script performs the following steps:
1. Parses command-line arguments to get the data path and optional flags.
2. Loads spatial coefficients and data from the specified file.
3. Generates a dynamics matrix and true full graph from the spatial coefficients.
4. Fits the CaStLe-PC model to the data.
5. Expands the reconstructed graph to the original space.
6. Computes F1 score and other graph metrics.
7. Optionally times the algorithm execution.
8. Saves the results to a file or prints them based on the provided flags.

Command-line arguments:
--data_path (str): Path to the input data file (required).
--plot (bool): Flag to plot the results (optional).
--print (bool): Flag to print the results instead of saving (optional).
--time_alg (bool): Flag to time the algorithm execution (optional).
--verbose (bool): Flag to enable verbose output (optional).

Example usage:
python benchmarking/test_CaStLe_PC.py --data_path path/to/data.npy --print --verbose --time_alg
"""

import argparse
import numpy as np
import os
import time
import stencil_functions as sf
import sys

sys.path.append(os.path.abspath(os.path.expanduser("~") + "../src/"))
import stable_SCM_generator as scm_gen
from graph_metrics import F1_score, get_graph_metrics
from tigramite.independence_tests.parcorr import ParCorr


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--plot", action=argparse.BooleanOptionalAction)
parser.add_argument("--print", action=argparse.BooleanOptionalAction)
parser.add_argument("--time_alg", action=argparse.BooleanOptionalAction)
parser.add_argument("--verbose", action=argparse.BooleanOptionalAction)
args = parser.parse_args()

DATA_PATH = args.data_path
PRINT = args.print
PLOT = args.plot
TIME_ALG = args.time_alg
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
data = data[:, :, :, 0]

SAVE_PATH_DIR = os.path.dirname(DATA_PATH)
DATA_FILENAME = os.path.basename(DATA_PATH)
GRID_SIZE = int(DATA_FILENAME.split("x")[0])

dynamics_matrix = scm_gen.create_coefficient_matrix(spatial_coefficients, GRID_SIZE)
true_full_graph = scm_gen.get_graph_from_coefficient_matrix(dynamics_matrix)

# Fit CaStLe
pc_alpha = 0.01  # None# [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
alpha_level = 0.01
parcorr = ParCorr(significance="analytic")
if TIME_ALG:
    start_time = time.time()
reconstructed_graph, val_matrix = sf.CaStLe_PC(
    data=data,
    cond_ind_test=parcorr,
    pc_alpha=pc_alpha,
    dependence_threshold=alpha_level,
    rows_inverted=True,
    dependencies_wrap=True,
)

# Expand to original space
center_parents = sf.get_parents(
    reconstructed_graph, val_matrix=val_matrix, include_lagzero_parents=True
)[4]
reconstructed_full_graph = sf.get_expanded_graph(center_parents, GRID_SIZE)

if TIME_ALG:
    end_time = time.time()

F1, P, R, TP, FP, FN, TN = F1_score(true_full_graph, reconstructed_full_graph)
if VERBOSE:
    print(
        "F1={}, P={}, R={}, TP={}, FP={}, FN={}, TN={}".format(F1, P, R, TP, FP, FN, TN)
    )

output_object = np.array(
    [
        spatial_coefficients,
        reconstructed_full_graph,
        get_graph_metrics(true_full_graph),
        get_graph_metrics(reconstructed_full_graph),
        (true_full_graph == reconstructed_full_graph),
        F1_score(true_full_graph, reconstructed_full_graph),
    ],
    dtype=object,
)

if not PRINT:
    # Save to file
    SAVE_PATH = SAVE_PATH_DIR + "/PC_CaStLe_results/r_" + DATA_FILENAME
    if VERBOSE:
        print("Saving data to " + SAVE_PATH)
    with open(SAVE_PATH, "wb") as f:
        np.save(f, output_object)
else:
    print(output_object)
    print(
        "F1={}, P={}, R={}, TP={}, FP={}, FN={}, TN={}".format(F1, P, R, TP, FP, FN, TN)
    )
if TIME_ALG:
    print(
        "Time elapsed for algorithm completion: {:.2f} seconscm_gen".format(
            end_time - start_time
        )
    )
