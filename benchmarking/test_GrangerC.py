import argparse
import os
import warnings
import spatial_SCM_data_generator as scmg
import numpy as np
import sys
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

sys.path.append(os.path.abspath(os.path.expanduser("~") + "/git/cldera/attribution/causalDiscovery/src/"))
from graph_metrics import F1_score, matthews_correlation_coefficient, get_graph_metrics

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
dynamics_matrix = scmg.create_coefficient_matrix(spatial_coefficients, GRID_SIZE)
true_graph = scmg.get_graph_from_coefficient_matrix(dynamics_matrix)

# Reshape data for input to PCMCI
read_data = np.ndarray([])
if len(data.shape) > 3:
    read_data = data[:, :, :, 0]  # only working with first variable for now

if len(read_data.shape) > 2:
    data = data.reshape(data.shape[0] * data.shape[1], data.shape[2])  # reshape to N^2xtime
    data = data.transpose()  # Rows must be temporal
if data.shape[0] < data.shape[1]:
    warnings.warn("More columns than rows! Either there are more variables than observations, or you need to transpose the data.")

# Begin PCMCI steps
parcorr = ParCorr(significance="analytic")
dataframe = pp.DataFrame(data)
pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=1)
min_tau = 1
max_tau = 1

pcmci.verbosity = 0
pc_alpha = 0.01  # None# [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
alpha_level = 0.01
results = pcmci.run_fullci(
    tau_min=min_tau,
    tau_max=max_tau,
    alpha_level=alpha_level,
)

q_matrix = pcmci.get_corrected_pvalues(p_matrix=results["p_matrix"], tau_min=min_tau, tau_max=max_tau, fdr_method="fdr_bh")
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
    SAVE_PATH = SAVE_PATH_DIR + "/GrangerC_results/r_" + DATA_FILENAME
    if VERBOSE:
        print("Saving data to " + SAVE_PATH)
    with open(SAVE_PATH, "wb") as f:
        np.save(f, output_object)
else:
    print(output_object)
