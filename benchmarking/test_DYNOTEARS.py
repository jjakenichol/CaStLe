"""
Script for testing DYNO_TEARS on 2D SCM(VAR) data.
"""

import argparse
import DSAVAR as ds
import numpy as np
import pandas as pd
import os
import sys

# print(sys.path)

# from tigramite.toymodels import structural_causal_processes
# from typing import Union

sys.path.append(
    os.path.abspath(
        os.path.expanduser("~") + "/git/cldera/attribution/causalDiscovery/src/"
    )
)
from graph_metrics import F1_score, get_graph_metrics

# sys.path.append(
#     os.path.abspath("/ascldap/users/jefnich/.conda/envs/cldera_causal/lib/python3.9/site-packages/causalnex")
# )
# print(sys.path)
from causalnex.structure.dynotears import from_pandas_dynamic


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

with open(
    DATA_PATH,
    "rb",
) as f:
    spatial_coefficients, data = np.load(f, allow_pickle=True)
data = data[:, :, :, 0]

SAVE_PATH_DIR = os.path.dirname(DATA_PATH)
DATA_FILENAME = os.path.basename(DATA_PATH)
GRID_SIZE = int(DATA_FILENAME.split("x")[0])
LINK_THRESHOLD = 0.1

dynamics_matrix = ds.create_coefficient_matrix(spatial_coefficients, GRID_SIZE)
true_full_graph = ds.get_graph_from_coefficient_matrix(dynamics_matrix)


def lin(x):
    """Linear function for SCM definitions"""
    return x


# Fit DYNOTEARs
## format data into dataframe
data = np.reshape(data, (GRID_SIZE**2, data.shape[2]))
data = data[:, :].transpose()
col_names = ["" + str(i) for i in np.arange(data.shape[1])]
df = pd.DataFrame(data=data, columns=col_names)
## fit model
structure_model = from_pandas_dynamic(df, p=1)  # p=1 indicates maximum lag=1

# Convert to graph
# TODO: import this from ds instead
reconstructed_graph, val_matrix = ds.get_graph_from_structure_model(structure_model)

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
    SAVE_PATH = SAVE_PATH_DIR + "/DYNOT_results/r_" + DATA_FILENAME
    if VERBOSE:
        print("Saving data to " + SAVE_PATH)
    with open(SAVE_PATH, "wb") as f:
        np.save(f, output_object)
else:
    print(output_object)
    print(
        "F1={}, P={}, R={}, TP={}, FP={}, FN={}, TN={}".format(F1, P, R, TP, FP, FN, TN)
    )
