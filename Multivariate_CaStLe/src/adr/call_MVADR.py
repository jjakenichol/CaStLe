"""
Multivariate ADR Experiment Runner

This script initializes and runs a multivariate Advection-Diffusion-Reaction (MVADR) experiment
using the ADRExperiment class. It allows for saving and loading experiment results,
animating the solution, and computing and plotting stencils.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
from os import path
from tigramite.independence_tests.parcorr import ParCorr

from ADRExperiment import ADRExperiment
import mcastle_utils as ms
import causal_graph_metrics as graph_metrics


def main():
    plot_matlab = False
    plot_initial_condition = False
    animate_python = True
    save_animation = True
    save_results = True
    load_results = False
    compute_stencil = True
    verbose = True

    # Initialize the ADRExperiment class with parameters
    experiment = ADRExperiment(
        mesh_shape="circle",
        init_center=[-0.0, -0.0],
        plume_size=50,
        diff_coeffs=[0.05, 0.05],
        advection_coeffs=[1.0, 1.0],
        velocity_field_type="constant",
        velocity_parameters=[5.0, 2.0],
        react_rate=5,
        reaction_scaling=1.0,
        t=np.linspace(0.0, 0.4, 51).tolist(),
        H=0.02,
        radius=4.0,
        capture_apothem=1.0,
        capture_N=50,
        parallel_interpolation=False,
        plot=plot_matlab,
        verbose=verbose,
    )

    # Generate filename based on parameters
    filename = experiment.generate_filename()

    # Run the ADR experiment
    experiment_dir = "results/ADR_models"
    solution = experiment.run_adr_experiment(cached_files_dir=experiment_dir)
    if experiment_dir is not None:
        experiment.pretty_print_parameters()

    # Print the shape of the solution to verify
    print("Shape of the solution:", solution.shape)

    # Save the results
    if save_results:
        experiment.save_results("results/ADR_models/" + filename)

    # Load the results
    if load_results:
        loaded_solution = experiment.load_results("results/ADR_models/" + filename)
        solution = loaded_solution

    # Plot the initial conditions
    if plot_initial_condition:
        experiment.pde_solver.plot_fields(solution, timestep=0, title="Initial Conditions")
        plt.show(block=False)  # Show the initial conditions plot without blocking

    animation_save_filepath = None
    if save_animation:
        # Create a descriptive filename based on the experiment parameters
        animation_save_dir = "results/animations/"
        animation_save_filepath = animation_save_dir + filename[:-4] + ".gif"
        print(f"Animation save path: {animation_save_filepath}")

    # Animate the results
    if animate_python:
        ani = experiment.animate_solution(save_path=animation_save_filepath)
        # plt.show()
        plt.show(block=False)  # Show the plot without blocking

    # Reshape solution to from (Xs, Ys, time, species) -> (variable_n, X, Y, T)
    data = solution.transpose((3, 0, 1, 2))
    # sys.exit()

    if compute_stencil:
        ground_truth_reaction_graph = np.array([[["", "-->"], ["", "-->"]], [["", ""], ["", "-->"]]], dtype=object)
        ground_truth_reaction_graph_no_auto = np.array([[["", ""], ["", "-->"]], [["", ""], ["", ""]]], dtype=object)

        parcorr = ParCorr(significance="analytic")
        start_time = time.time()
        results = ms.mv_CaStLe_PC(
            data=data,
            cond_ind_test=parcorr,
            pc_alpha=0.01,
            graph_p_threshold=0.01,
        )
        print(f"MV CaStLe took {time.time() - start_time} seconds to complete.")
        # Save stencil results dictionary
        save_filename = "results/stencil_results/" + filename[:-4] + "stencil_results.pkl"
        with open(save_filename, "wb") as f:
            pickle.dump(results, f)
        print(f"Results saved to {save_filename}")

        print("Plotting stencil...")
        graph = results["graph"]
        val_matrix = results["val_matrix"]

        ms.plot_stencil_graph(stencil_graph=graph, stencil_val_matrix=val_matrix)

        # Compute reaction results
        reaction_graph, reaction_val_matrix = ms.construct_reaction_graph(graph, val_matrix)
        reaction_results = (reaction_graph, reaction_val_matrix)

        # Compute summary results
        summary_graph, summary_val_matrix = ms.summarize_stencil(graph, val_matrix)
        summary_results = (summary_graph, summary_val_matrix)

        # Apply metrics
        reaction_mcc = graph_metrics.matthews_correlation_coefficient(ground_truth_reaction_graph, reaction_graph)
        reaction_mcc_no_auto = graph_metrics.matthews_correlation_coefficient(ground_truth_reaction_graph_no_auto, reaction_graph)
        stencil_angle = ms.get_angle_from_stencil(summary_graph, summary_val_matrix)

        # Compute angle difference
        alpha, beta = experiment.velocity_parameters
        ground_truth_angle = ms.compute_angle(alpha, beta)
        angle_difference = ms.angle_difference(ground_truth_angle, stencil_angle)

        print(f"Reaction MCC: {reaction_mcc}")
        print(f"Reaction MCC (no auto): {reaction_mcc_no_auto}")
        print(f"Stencil Angle: {stencil_angle}")
        print(f"Angle Difference: {angle_difference}")

        fig1, ax1 = plt.subplots(figsize=(5, 4))
        if graph.shape[0] // 9 == 2:
            ms.plot_reaction_graph_of_two_nodes(reaction_graph, reaction_val_matrix, fig=fig1, ax=ax1, node_aspect=4)
        else:
            ms.plot_reaction_graph(reaction_graph, reaction_val_matrix, fig=fig1, ax=ax1, node_aspect=4)

        # Simplify the stencil graph and value matrix
        ms.plot_stencil_graph(summary_graph, summary_val_matrix, directional_var_names=True)

        plt.show()


if __name__ == "__main__":
    main()
