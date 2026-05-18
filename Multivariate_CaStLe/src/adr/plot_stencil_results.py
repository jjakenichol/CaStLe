"""
Plot Stencil Results

Load a saved M-CaStLe stencil results pickle and display the stencil graph
using :func:`mcastle_utils.plot_stencil_graph`.

Usage
-----
    python plot_stencil_results.py <path_to_stencil_results.pkl>
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pickle
import numpy as np
import matplotlib.pyplot as plt
import mcastle_utils as ms


def plot_stencil_results(filepath):
    """
    Load a stencil results pickle and display the stencil graph.

    Args:
        filepath (str): Path to a stencil results file (.pkl) produced by
            :func:`compute_stencil.compute_stencil` or
            :meth:`ADRExperiment.run_adr_experiment`.
    """
    # Load the stencil results dictionary
    with open(filepath, "rb") as f:
        results = pickle.load(f)

    # Extract graph and val_matrix from the dictionary
    graph = results["graph"]
    val_matrix = results["val_matrix"]

    # Plot the stencil graph
    ms.plot_stencil_graph(stencil_graph=graph, stencil_val_matrix=val_matrix)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_stencil_results.py <path_to_stencil_results.pkl>")
        sys.exit(1)

    filepath = sys.argv[1]
    plot_stencil_results(filepath)
