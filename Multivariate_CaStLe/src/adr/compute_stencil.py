"""
Compute Stencil from ADR Experiment Results

This script loads an ADRExperiment object from a specified file, computes the stencil
using the mv_CaStLe_PC method, and saves the stencil results. Optionally, it can also
plot the stencil graph.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse
import matplotlib.pyplot as plt
import pickle
import time
from ADRExperiment import ADRExperiment
from tigramite.independence_tests.parcorr import ParCorr
import mcastle_utils as ms


def compute_stencil(filepath, plot_stencil=False):
    """
    Load an ADRExperiment object, compute the stencil, and save the results.

    Args:
        filepath (str): Path to the ADR experiment results file (pkl).
        plot_stencil (bool, optional): Whether to plot the stencil graph. Defaults to False.
    """
    # Load the ADR experiment object
    experiment = ADRExperiment.load_results(filepath)
    if experiment is None:
        print(f"Failed to load ADRExperiment object from {filepath}")
        return

    # Reshape solution to from (Xs, Ys, time, species) -> (variable_n, X, Y, T)
    solution = experiment.solution
    data = solution.reshape((solution.shape[3], solution.shape[1], solution.shape[2], solution.shape[0]))

    # Compute stencil
    start_time = time.time()
    parcorr = ParCorr(significance="analytic")
    results = ms.mv_CaStLe_PC(
        data=data,
        cond_ind_test=parcorr,
        pc_alpha=0.01,
        graph_p_threshold=0.01,
    )
    print(f"MV CaStLe took {time.time() - start_time} seconds to complete.")

    # Save stencil results dictionary
    stencil_filename = os.path.splitext(filepath)[0] + "_stencil_results.pkl"
    with open(stencil_filename, "wb") as f:
        pickle.dump(results, f)
    print(f"Stencil results saved to {stencil_filename}")

    # Plot stencil if requested
    if plot_stencil:
        print("Plotting stencil...")
        ms.plot_stencil_graph(stencil_graph=results["graph"], stencil_val_matrix=results["val_matrix"])
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute stencil from ADR experiment results.")
    parser.add_argument("filepath", type=str, help="Path to the ADR experiment results file (pkl).")
    parser.add_argument("--plot", action="store_true", help="Plot the stencil graph.")
    args = parser.parse_args()

    compute_stencil(args.filepath, plot_stencil=args.plot)
