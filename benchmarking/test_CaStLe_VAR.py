"""
Script for testing VAR-graphs with CaStLe with the given data.
"""

import argparse
import stencil_functions as sf
import numpy as np
import os
import sys
from statsmodels.tsa.api import VAR
from tigramite.toymodels import structural_causal_processes

sys.path.append(os.path.abspath(os.path.expanduser("~") + "../src/"))
import stable_SCM_generator as scm_gen
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
concatenated_data = sf.concatenate_timeseries_wrapping(data, GRID_SIZE, GRID_SIZE, rows_inverted=True)

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
                coefficients.transpose()[i][j] if abs(coefficients.transpose()[i][j]) > dependence_threshold else 0,
                lin,
            )
            for j in range(coefficients.shape[1])
        ]
reconstructed_stencil_graph = structural_causal_processes.links_to_graph(SCM)
all_parents = sf.get_parents(reconstructed_stencil_graph)
reconstructed_full_graph = sf.get_expanded_graph(all_parents[4], GRID_SIZE, wrapping=False)

# if VERBOSE:
# F1, P, R, TP, FP, FN, TN = F1_score(true_full_graph, reconstructed_full_graph)
# print("F1={}, P={}, R={}, TP={}, FP={}, FN={}, TN={}".format(F1, P, R, TP, FP, FN, TN))

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
