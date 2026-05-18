"""
Animate ADR Experiment Results

Load a saved ADRExperiment pickle and display (or save) an animation of the
species concentration fields evolving over the simulation time window.

Usage
-----
    python animate_adr_results.py <path_to_experiment.pkl> [<output.gif>]
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib.pyplot as plt
from ADRExperiment import ADRExperiment


def animate_adr_results(filepath, save_path=None):
    """
    Load an ADRExperiment pickle and animate the PDE solution.

    Args:
        filepath (str): Path to the ADRExperiment results file (.pkl).
        save_path (str, optional): Destination path for a saved GIF.
            If None the animation is only displayed interactively.
    """
    # Load the ADR experiment object
    experiment = ADRExperiment.load_results(filepath)
    if experiment is None:
        print(f"Failed to load ADRExperiment object from {filepath}")
        return

    # Animate the results using the class method
    ani = experiment.animate_solution(save_path=save_path)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python animate_adr_results.py <path_to_adr_experiment.pkl> [<path_to_save_animation.gif>]")
        sys.exit(1)

    filepath = sys.argv[1]
    save_path = sys.argv[2] if len(sys.argv) == 3 else None

    animate_adr_results(filepath, save_path)
