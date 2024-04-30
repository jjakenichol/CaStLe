"""
Script for testing DYNO_TEARS on 2D SCM(VAR) data.
"""

import argparse
import DSAVAR as ds
import numpy as np
import pandas as pd
import os
import stencil_functions as sf
import sys
sys.path.append(
    os.path.abspath(os.path.expanduser("~") + "/git/cldera/attribution/causalDiscovery/src/")
)
from graph_metrics import F1_score, get_graph_metrics
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

# Fit DYNOTEARs
## format data into dataframe
data_castled = sf.concatenate_timeseries_nonwrapping(data, rows_inverted=True)
col_names = ['' + str(i) for i in np.arange(data_castled.shape[1])]
df_castled = pd.DataFrame(data=data_castled, columns=col_names)
## fit model
taboo_children = ["0", "1", "2", "3", "5", "6", "7", "8"]
taboo_edges = [] # (lag, from, to)
for i in col_names:
    for j in col_names:
        # Ban all links from lag 0, ie only allow lag1 -> lag0
        taboo_edges.append((0, i, j))
structure_model_castled = from_pandas_dynamic(df_castled, p=1, tabu_edges=taboo_edges, tabu_child_nodes=taboo_children)

# Convert to graph
reconstructed_graph, val_matrix = ds.get_graph_from_structure_model(structure_model_castled)

# Expand to original space
center_parents = sf.get_parents(reconstructed_graph, val_matrix=val_matrix, include_lagzero_parents=True)[4]
reconstructed_full_graph = sf.get_expanded_graph(center_parents, GRID_SIZE)

F1, P, R, TP, FP, FN, TN = F1_score(true_full_graph, reconstructed_full_graph)
if VERBOSE:
    print("F1={}, P={}, R={}, TP={}, FP={}, FN={}, TN={}".format(F1, P, R, TP, FP, FN, TN))

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
    SAVE_PATH = SAVE_PATH_DIR + "/DYNOT_CaStLe_results/r_" + DATA_FILENAME
    if VERBOSE:
        print("Saving data to " + SAVE_PATH)
    with open(SAVE_PATH, "wb") as f:
        np.save(f, output_object)
else:
    print(output_object)
    print("F1={}, P={}, R={}, TP={}, FP={}, FN={}, TN={}".format(F1, P, R, TP, FP, FN, TN))
