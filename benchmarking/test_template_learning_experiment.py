"""Neighborhood Aggregation Causal Learning
"""

import argparse
import numpy as np
import os
import sys
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

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

with open(
    DATA_PATH,
    "rb",
) as f:
    spatial_coefficients, data = np.load(f, allow_pickle=True)

SAVE_PATH_DIR = os.path.dirname(DATA_PATH)
DATA_FILENAME = os.path.basename(DATA_PATH)
GRID_SIZE = int(DATA_FILENAME.split("x")[0])

# dynamics_matrix = scm_gen.create_coefficient_matrix(GRID_SIZE, spatial_coefficients)
# true_graph = scm_gen.get_graph_from_coefficient_matrix(dynamics_matrix)

template_dynamics_matrix = scm_gen.create_nonwrapping_coefficient_matrix(spatial_coefficients)
true_template_graph = scm_gen.get_graph_from_coefficient_matrix(template_dynamics_matrix)

ROWS = GRID_SIZE
COLS = GRID_SIZE

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

pcmci_df = pp.DataFrame(concatenated_data)

# Begin PCMCI steps
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
# results = pcmci.run_pcmciplus(tau_min=min_tau, tau_max=max_tau, pc_alpha=alpha_level,)


# print("p-values")
# print(results["p_matrix"].round(3))
# print("MCI partial correlations")
# print(results["val_matrix"].round(2))

q_matrix = pcmci.get_corrected_pvalues(p_matrix=results["p_matrix"], tau_min=min_tau, tau_max=max_tau, fdr_method="fdr_bh")
# pcmci.print_significant_links(
#     p_matrix=q_matrix, val_matrix=results["val_matrix"], alpha_level=alpha_level
# )
reconstructed_graph = pcmci.get_graph_from_pmatrix(
    p_matrix=q_matrix,
    alpha_level=alpha_level,
    tau_min=min_tau,
    tau_max=max_tau,
)
results["graph"] = reconstructed_graph


# F1, P, R, TP, FP, FN, TN = F1_score(true_template_graph, reconstructed_graph)
# print(
#     "F1 = {}, P = {}, R = {}, TP = {}, FP = {}, FN = {}, TN = {}: ".format(F1, P, R, TP, FP, FN, TN)
# )

# print("MCC = {}".format(matthews_correlation_coefficient(TP, FP, FN, TN)))


output_object = np.array(
    [
        spatial_coefficients,
        reconstructed_graph,
        get_graph_metrics(true_template_graph),
        get_graph_metrics(reconstructed_graph),
        (true_template_graph == reconstructed_graph),
        F1_score(true_template_graph, reconstructed_graph),
    ],
    dtype=object,
)

if not PRINT:
    # Save to file
    SAVE_PATH = SAVE_PATH_DIR + "/TL_results/r_" + DATA_FILENAME
    if VERBOSE:
        print("Saving data to " + SAVE_PATH)
    with open(SAVE_PATH, "wb") as f:
        np.save(f, output_object)
else:
    print(output_object)
