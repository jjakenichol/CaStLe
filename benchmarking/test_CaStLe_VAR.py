"""
Script for testing VAR-graphs with CaStLe with the given data.

This script performs the following steps:
1. Parses command-line arguments to get the data path and optional flags.
2. Loads spatial coefficients and data from the specified file.
3. Generates a dynamics matrix and true full graph from the spatial coefficients.
4. Fits a VAR model to the concatenated data.
5. Constructs a stencil graph from the VAR model coefficients.
6. Expands the reconstructed graph to the original space.
7. Computes F1 score and other graph metrics.
8. Saves the results to a file or prints them based on the provided flags.

Command-line arguments:
--data_path (str): Path to the input data file (required).
--plot (bool): Flag to plot the results (optional).
--print (bool): Flag to print the results instead of saving (optional).
--verbose (bool): Flag to enable verbose output (optional).

Example usage:
python benchmarking/test_CaStLe_VAR.py --data_path path/to/data.npy --print --verbose
"""

import argparse
import numpy as np
import os
import sys
from statsmodels.tsa.api import VAR
from tigramite.toymodels import structural_causal_processes

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import stable_SCM_generator as scm_gen
import stencil_functions as sf
from graph_metrics import F1_score, get_graph_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--plot", action=argparse.BooleanOptionalAction)
parser.add_argument("--print", action=argparse.BooleanOptionalAction)
parser.add_argument("--verbose", action=argparse.BooleanOptionalAction)
args = parser.parse_args()

DATA_PATH = args.data_path
PRINT = args.print
PLOT = args.plot
VERBOSE = args.verbose
if not VERBOSE:
    VERBOSE = 0
else:
    VERBOSE = 1

if VERBOSE:
    print(DATA_PATH)

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


def lin(x):
    """Linear function for SCM definitions"""
    return x


########### castle #############
concatenated_data = sf.concatenate_timeseries_wrapping(data, rows_inverted=True)

# Fit VAR
model = VAR(concatenated_data)
results = model.fit(1)
coefficients = results.params[1:]

# Make stencil graph:
dependence_threshold = 0.1
SCM = {}
for i in range(coefficients.shape[0]):
    if i != 4:
        SCM[i] = [((j, -1), 0, lin) for j in range(coefficients.shape[1])]
    else:
        SCM[i] = [
            (
                (j, -1),
                (
                    coefficients.transpose()[i][j]
                    if abs(coefficients.transpose()[i][j]) > dependence_threshold
                    else 0
                ),
                lin,
            )
            for j in range(coefficients.shape[1])
        ]
reconstructed_stencil_graph = structural_causal_processes.links_to_graph(SCM)
all_parents = sf.get_parents(reconstructed_stencil_graph)
reconstructed_full_graph = sf.get_expanded_graph_from_parents(
    all_parents[4], GRID_SIZE, wrapping=False
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
    SAVE_PATH = SAVE_PATH_DIR + "/VAR_CaStLe_results/r_" + DATA_FILENAME
    if VERBOSE:
        print("Saving data to " + SAVE_PATH)
    with open(SAVE_PATH, "wb") as f:
        np.save(f, output_object)
else:
    print(output_object)
