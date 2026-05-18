"""
Script for testing multivariate CaStLe-DYNOTEARS on 2D SCM(VAR) data.

This script evaluates the performance of the multivariate CaStLe-DYNOTEARS algorithm on 2-dimensional
Structural Causal Model (SCM) data with Vector Autoregression (VAR). It loads spatial coefficients and data,
computes the true graph based on the dataset's coefficients, applies the CaStLe-DYNOTEARS algorithm to reconstruct
the graph, and evaluates the reconstruction's accuracy against the true graph.
"""

import argparse
import numpy as np
import os
import sys
import time

import mcastle_utils as ms
from causal_graph_metrics import (
    F1_score,
    get_graph_metrics,
    matthews_correlation_coefficient as mcc,
)

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--print", action=argparse.BooleanOptionalAction)
parser.add_argument("--time_alg", action=argparse.BooleanOptionalAction)
parser.add_argument("--verbose", action=argparse.BooleanOptionalAction)
args = parser.parse_args()

DATA_PATH = args.data_path
PRINT = args.print
TIME_ALG = args.time_alg
VERBOSE = args.verbose
if not VERBOSE:
    VERBOSE = 0
else:
    VERBOSE = 1

try:
    with open(DATA_PATH, "rb") as f:
        npzfile = np.load(f, allow_pickle=True)
        spatial_coefficients = npzfile["local_coefs"]
        data = npzfile["dataset"]
except FileNotFoundError:
    print(f"Error: The file {DATA_PATH} was not found.")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred while loading the data: {e}")
    sys.exit(1)

SAVE_PATH_DIR = os.path.dirname(DATA_PATH)
# Remove /data from SAVE_PATH_DIR
SAVE_PATH_DIR = SAVE_PATH_DIR[:-5] + "/results_castle_dynotears/"
DATA_FILENAME = os.path.basename(DATA_PATH)
# Remove suffix
DATA_FILENAME = DATA_FILENAME[:-4]
# Prepend "r_" to the filename
RESULT_FILENAME = "r_" + DATA_FILENAME
# Build the complete save path, adding the desired extension (here ".npy")
SAVE_PATH = os.path.join(SAVE_PATH_DIR, RESULT_FILENAME + ".npy")

GRID_SIZE = int(DATA_FILENAME.split("x")[0])

# Compute true graph from the dataset's coefficients
true_stencil, true_stencil_val_matrix = ms.get_stencil_graph_from_coefficients(
    spatial_coefficients
)
true_full_graph = ms.map_stencil_graph_to_full_graph(true_stencil, grid_size=GRID_SIZE)

# Compute stencil with CaStLe
dependence_threshold = 0.0
lambda_a = 0.1
max_iter = 100
start_time = time.time()
reconstructed_stencil_graph, _ = ms.mv_CaStLe_DYNOTEARS(
    data=data,
    rows_inverted=True,
    dependencies_wrap=True,
    allow_center_directed_links=True,
    dependence_threshold=dependence_threshold,
    lambda_a=lambda_a,
    max_iter=max_iter,
)
end_time = time.time()
algorithm_time = end_time - start_time

# Collect results
reconstructed_full_graph = ms.map_stencil_graph_to_full_graph(
    reconstructed_stencil_graph, grid_size=GRID_SIZE
)

# Compute performance metrics
F1, P, R, TP, FP, FN, TN = F1_score(
    true_graph=true_full_graph, discovered_graph=reconstructed_full_graph
)
MCC = mcc(TP=TP, FP=FP, FN=FN, TN=TN)

output_object = np.array(
    [
        spatial_coefficients,
        reconstructed_full_graph,
        get_graph_metrics(true_full_graph),
        get_graph_metrics(reconstructed_full_graph),
        (true_full_graph == reconstructed_full_graph),
        F1,
        MCC,
        P,
        R,
        TP,
        FP,
        FN,
        TN,
        algorithm_time,
        dependence_threshold,
        lambda_a,
        DATA_FILENAME,
    ],
    dtype=object,
)

# Print or save results
if not PRINT:
    # Save to file
    if VERBOSE:
        print("Saving data to " + SAVE_PATH)
    with open(SAVE_PATH, "wb") as f:
        np.save(f, output_object)
else:
    print(output_object)
    print(
        "F1={}, P={}, R={}, TP={}, FP={}, FN={}, TN={}".format(F1, P, R, TP, FP, FN, TN)
    )
    print(f"Save path would be {SAVE_PATH}")
if TIME_ALG:
    print(
        "Time elapsed for algorithm completion: {:.2f} seconds".format(algorithm_time)
    )
