"""
plot_reduced_space_ts.py

This script loads an ADRExperiment object from a specified file, reduces the data using a multivariate stencil,
and plots the time series data for two species across a 3x3 grid of subplots. Each subplot corresponds to a specific
grid cell, and both species' time series are plotted in each subplot.

Usage:
    python plot_reduced_space.py <path_to_adr_experiment.pkl>

Arguments:
    <path_to_adr_experiment.pkl> : Path to the ADRExperiment pickle file.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib.pyplot as plt
import numpy as np
from ADRExperiment import ADRExperiment
import mcastle_utils as ms


def plot_time_series(filepath: str) -> None:
    """
    Plots the time series data for two species across a 3x3 grid of subplots.

    Parameters:
        filepath (str): Path to the ADRExperiment pickle file.

    Returns:
        None
    """
    experiment = ADRExperiment.load_results(filepath)
    if experiment is None:
        print(f"Failed to load ADRExperiment object from {filepath}")
        return

    reduced_data = ms.get_MV_reduced_space(
        data=experiment.solution,
        dependencies_wrap=False,
        rows_inverted=False,
    )

    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True, sharey=True)

    for i in range(9):
        row, col = divmod(i, 3)
        axes[row, col].plot(reduced_data[:, i], label="Species 1")
        axes[row, col].plot(reduced_data[:, i + 9], label="Species 2")
        axes[row, col].set_title(f"Grid {i}")
        axes[row, col].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python animate_adr_results.py <path_to_adr_experiment.pkl>")
        sys.exit(1)

    filepath = sys.argv[1]
    plot_time_series(filepath)
