"""
Script for testing VAR-graphs on the given data.
"""

import argparse
import DSAVAR as ds
import stencil_functions as sf
import numpy as np
import os
import time
import sys
from tigramite.independence_tests.parcorr import ParCorr

sys.path.append(
    os.path.abspath(
        os.path.expanduser("~") + "/git/cldera/attribution/causalDiscovery/src/"
    )
)
from graph_metrics import F1_score, get_graph_metrics

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

dynamics_matrix = ds.create_coefficient_matrix(spatial_coefficients, GRID_SIZE)
true_full_graph = ds.get_graph_from_coefficient_matrix(dynamics_matrix)

parcorr = ParCorr(significance="analytic")
pc_alpha = 0.01
alpha_level = 0.01
if TIME_ALG:
    start_time = time.time()
reconstructed_graph, val_matrix = sf.PC(
    data, parcorr, pc_alpha=pc_alpha, dependence_threshold=alpha_level
)
if TIME_ALG:
    end_time = time.time()

F1, P, R, TP, FP, FN, TN = F1_score(true_full_graph, reconstructed_graph)
if VERBOSE:
    print(
        "F1={}, P={}, R={}, TP={}, FP={}, FN={}, TN={}".format(F1, P, R, TP, FP, FN, TN)
    )

output_object = np.array(
    [
        spatial_coefficients,
        reconstructed_graph,
        get_graph_metrics(true_full_graph),
        get_graph_metrics(reconstructed_graph),
        (true_full_graph == reconstructed_graph),
        F1_score(true_full_graph, reconstructed_graph),
    ],
    dtype=object,
)

if not PRINT:
    # Save to file
    SAVE_PATH = SAVE_PATH_DIR + "/PC_results/r_" + DATA_FILENAME
    if VERBOSE:
        print("Saving data to " + SAVE_PATH)
    with open(SAVE_PATH, "wb") as f:
        np.save(f, output_object)
else:
    print(output_object)
if TIME_ALG:
    print(
        "Time elapsed for algorithm completion: {:.2f} seconds".format(
            end_time - start_time
        )
    )
