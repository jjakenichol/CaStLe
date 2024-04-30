"""
"""

import argparse
import DSAVAR as ds
import stencil_functions as sf
import numpy as np
import os
import sys
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

sys.path.append(
    os.path.abspath(
        os.path.expanduser("~") + "/git/cldera/attribution/causalDiscovery/src/"
    )
)
from graph_metrics import F1_score, matthews_correlation_coefficient, get_graph_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument(
    "--heuristics",
    type=int,
    required=True,
    choices=[0, 1, 2, 3],
    help="Options: 0=no heuristics; 1=prune heuristic; 2=add heuristic; 3=both heuristics",
)
parser.add_argument("--naivePIP", action=argparse.BooleanOptionalAction)
parser.add_argument("--plot", action=argparse.BooleanOptionalAction)
parser.add_argument("--print", action=argparse.BooleanOptionalAction)
parser.add_argument("--verbose", action=argparse.BooleanOptionalAction)
args = parser.parse_args()

DATA_PATH = args.data_path
HEURISTICS_MODE = args.heuristics
NAIVE_PIP = args.naivePIP
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

dynamics_matrix = ds.create_coefficient_matrix(spatial_coefficients, GRID_SIZE)
true_full_graph = ds.get_graph_from_coefficient_matrix(dynamics_matrix)

# template_dynamics_matrix = ds.create_nonwrapping_coefficient_matrix(spatial_coefficients)
# true_template_graph = ds.get_graph_from_coefficient_matrix(template_dynamics_matrix)

# Learn Template
ROWS = GRID_SIZE
COLS = GRID_SIZE


if VERBOSE:
    print("Concatenating data for template...")
concatenated_data = [[] for i in range(9)]
index = 0
for row in range(data.shape[0]):
    for col in range(data.shape[1]):
        from_left = data[row, col - 1, :]
        from_right = data[row, (col + 1) % ROWS, :]
        from_top = data[row - 1, col, :]
        from_bottom = data[(row + 1) % ROWS, col, :]
        from_top_left = data[row - 1, col - 1, :]
        from_top_right = data[row - 1, (col + 1) % COLS, :]
        from_bot_left = data[(row + 1) % ROWS, col - 1, :]
        from_bot_right = data[(row + 1) % ROWS, (col + 1) % COLS, :]
        from_self = data[row, col, :]

        concatenated_data[0].extend(from_top_left)
        concatenated_data[1].extend(from_top)
        concatenated_data[2].extend(from_top_right)
        concatenated_data[3].extend(from_left)
        concatenated_data[4].extend(from_self)
        concatenated_data[5].extend(from_right)
        concatenated_data[6].extend(from_bot_left)
        concatenated_data[7].extend(from_bottom)
        concatenated_data[8].extend(from_bot_right)

        index += 1

concatenated_data = np.array(concatenated_data).transpose()

var_names = ["NW", "N", "NE", "W", "self", "E", "SW", "S", "SE"]
pcmci_df = pp.DataFrame(concatenated_data, var_names=var_names)

# Begin PCMCI steps
if VERBOSE:
    print("Beginning PCMCI steps...")
parcorr = ParCorr(significance="analytic")
pcmci = PCMCI(dataframe=pcmci_df, cond_ind_test=parcorr, verbosity=1)
min_tau = 0
max_tau = 1

pcmci.verbosity = 0
pc_alpha = 0.01  # None# [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
alpha_level = 0.01
results = pcmci.run_pcmci(
    tau_min=min_tau,
    tau_max=max_tau,
    pc_alpha=pc_alpha,
)

q_matrix = pcmci.get_corrected_pvalues(
    p_matrix=results["p_matrix"], tau_min=min_tau, tau_max=max_tau, fdr_method="fdr_bh"
)
reconstructed_template_graph = pcmci.get_graph_from_pmatrix(
    p_matrix=q_matrix,
    alpha_level=alpha_level,
    tau_min=min_tau,
    tau_max=max_tau,
)
results["graph"] = reconstructed_template_graph

# Learn stencil
if VERBOSE:
    print("Learning stencil...")
center_cell = 4
all_parents = sf.get_parents(
    reconstructed_template_graph,
    val_matrix=results["val_matrix"],
    include_lagzero_parents=False,
    output_val_matrix=True,
)
center_parents = all_parents[center_cell]

# Run heuristics (imputation?)
if HEURISTICS_MODE == 0:
    if VERBOSE:
        print("Running with no heuristics...")
# TODO: HEURISTIC MODES ANTIQUATED, REMOVE
# elif HEURISTICS_MODE == 1:
#     if VERBOSE:
#         print("Running with prune heuristics...")
#     sf.prune_nonredundant_links(all_parents)
# elif HEURISTICS_MODE == 2:
#     if VERBOSE:
#         print("Running with add heuristics...")
#     sf.add_omitted_links(all_parents)
# elif HEURISTICS_MODE == 3:
#     if VERBOSE:
#         print("Running with both heuristics...")
#     sf.prune_nonredundant_links(all_parents)
#     sf.add_omitted_links(all_parents)
else:
    print("Unacceptable heuristics mode passed.")
    sys.exit(1)

# Get NDM from identified stencil
ndm = np.zeros((9))
for parent in center_parents:
    ndm[parent[0]] = parent[2]
ndm = ndm.reshape((3, 3))

# Construct full graph from NDM of stencil
reconstructed_dynamics_matrix = ds.create_coefficient_matrix(ndm, GRID_SIZE)
reconstructed_full_graph, reconst_val_matrix = ds.get_graph_from_coefficient_matrix(
    reconstructed_dynamics_matrix, return_val_matrix=True
)


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
    if NAIVE_PIP:
        SAVE_PATH = "{}/SL_H{}_naivePIP_results/r_{}".format(
            SAVE_PATH_DIR, HEURISTICS_MODE, DATA_FILENAME
        )
    else:
        SAVE_PATH = "{}/SL_H{}_results/r_{}".format(
            SAVE_PATH_DIR, HEURISTICS_MODE, DATA_FILENAME
        )
    if VERBOSE:
        print("Saving data to " + SAVE_PATH)
    with open(SAVE_PATH, "wb") as f:
        np.save(f, output_object)
else:
    print(output_object)
