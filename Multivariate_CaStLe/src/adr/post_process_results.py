"""
Post-Process ADR Stencil Results

Loads all stencil result pickles from a results directory, matches each to
its corresponding ADRExperiment file, extracts experiment parameters, and
computes a full suite of metrics:

* Reaction-graph MCC (with and without autocausal links)
* Precision, Recall, FDR from the confusion matrix
* M-stencil angle estimation error (standard and nonnegative variants)

The resulting ``pandas.DataFrame`` is saved as ``analysis_results.pkl`` in
the results directory for downstream plotting with ``figure_angle_error2.py``.

Usage
-----
    python post_process_results.py <results_dir> <experiments_dir>
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse
import glob
import numpy as np
import pickle
import pandas as pd
import re
from ADRExperiment import ADRExperiment
from tigramite.independence_tests.parcorr import ParCorr
from tqdm.auto import tqdm
import mcastle_utils as ms
import causal_graph_metrics as graph_metrics


def serialize_param(param):
    """
    Extract the first element of a list-valued ADR parameter for use as a scalar DataFrame column.

    Args:
        param (list): A list-valued parameter (e.g. ``[0.05, 0.05]`` for ``diff_coeffs``).

    Returns:
        The first element of ``param``.
    """
    return param[0]


def load_results(results_dir, experiments_dir):
    """
    Load stencil results and matching ADRExperiment files into a DataFrame.

    Scans ``results_dir`` for files matching ``**stencil*results*.pkl``,
    parses hyperparameters from each filename, loads the corresponding
    ADRExperiment pickle from ``experiments_dir``, and assembles all
    parameters and stencil outputs into a single row per experiment.

    Args:
        results_dir (str): Directory containing stencil result pickles.
        experiments_dir (str): Directory containing the source ADRExperiment pickles.

    Returns:
        tuple:
            - **df** (*pandas.DataFrame*): One row per successfully loaded experiment,
              with columns for all ADR parameters and the stencil results dict.
            - **corrupted_summary** (*pandas.DataFrame*): Files that could not be
              loaded, with columns ``filepath`` and ``reason``.
            - **illegal_files** (*list[str]*): Filenames skipped due to invalid
              hyperparameter values (e.g. ``graph_p_threshold >= 1.0``).
    """
    data = []
    corrupted_files = []  # Keep track of corrupted files
    illegal_files = []

    # Find all files matching the pattern
    stencil_files = glob.glob(os.path.join(results_dir, "**stencil*results*.pkl"))

    for stencil_path in tqdm(stencil_files):
        filename = os.path.basename(stencil_path)

        # Check if file is empty (0 bytes)
        if os.path.getsize(stencil_path) == 0:
            corrupted_files.append((stencil_path, "Zero-byte file"))
            print(f"Corrupted file (0 bytes): {stencil_path}")
            continue

        # Extract pc_alpha and graph_p_threshold from filename using regex
        pattern = r"_stencil_results_([0-9.]+)_([0-9.]+)_([0-9.]+)_([a-zA-Z0-9_-]+)\.pkl$"
        match = re.search(pattern, filename)

        if match:
            pc_alpha = float(match.group(1))
            graph_p_threshold = float(match.group(2))
            strength_threshold = float(match.group(3))
            cd_alg = match.group(4)
            if graph_p_threshold >= 1.0:
                # print(f"file {filename} has graph_p_threshold={graph_p_threshold}")
                illegal_files.append(filename)
                continue

            # Construct experiment path based on the stencil filename
            base_filename = filename.replace(f"_stencil_results_{pc_alpha}_{graph_p_threshold}_{strength_threshold}_{cd_alg}.pkl", ".pkl")
            experiment_path = os.path.join(experiments_dir, base_filename)

            # Check if experiment file is empty
            if os.path.exists(experiment_path) and os.path.getsize(experiment_path) == 0:
                corrupted_files.append((experiment_path, "Zero-byte file"))
                print(f"Corrupted file (0 bytes): {experiment_path}")
                continue

            try:
                # Try to read the stencil results
                try:
                    with open(stencil_path, "rb") as f:
                        stencil_results = pickle.load(f)
                except (pickle.UnpicklingError, EOFError) as e:
                    corrupted_files.append((stencil_path, f"Pickle error: {str(e)}"))
                    print(f"Corrupted file (unpickling error): {stencil_path}")
                    continue

                # Try to read the experiment file
                experiment = ADRExperiment.load_results(experiment_path)
                if experiment is None:
                    print(f"Experiment is None at path {experiment_path}")
                    continue

                params = {
                    "filename": filename,
                    "pc_alpha": pc_alpha,
                    "graph_p_threshold": graph_p_threshold,
                    "strength_threshold": strength_threshold,
                    "cd_alg": cd_alg,
                    "stencil_results": stencil_results,
                    # "experiment": experiment,
                    "mesh_shape": experiment.mesh_shape,
                    "init_center": experiment.init_center,
                    "plume_size": experiment.plume_size,
                    "diff_coeffs": experiment.diff_coeffs,
                    "advection_coeffs": experiment.advection_coeffs,
                    "velocity_angle": experiment.velocity_angle,
                    "velocity_magnitude": experiment.velocity_magnitude,
                    "velocity_field_type": experiment.velocity_field_type,
                    "velocity_parameters": experiment.velocity_parameters,
                    "react_rate": experiment.react_rate,
                    "reaction_scaling": experiment.reaction_scaling,
                    "init_concentration": experiment.init_concentration,
                    "t_length": len(experiment.t),
                    "t_min": min(experiment.t),
                    "t_max": max(experiment.t),
                    "H": experiment.H,
                    "radius": experiment.radius,
                    "capture_apothem": experiment.capture_apothem,
                    "capture_N": experiment.capture_N,
                }
                params["diff_coefs_serial"] = serialize_param(params["diff_coeffs"])
                data.append(params)
            except Exception as e:
                corrupted_files.append((stencil_path, f"Error: {str(e)}"))
                print(f"Error processing {stencil_path}: {e}")

    df = pd.DataFrame(data)

    # Create a summary of corrupted files
    corrupted_summary = pd.DataFrame(corrupted_files, columns=["filepath", "reason"])

    print(f"Loaded {len(data)} valid files")
    print(f"Found {len(corrupted_files)} corrupted files")
    print(f"Found {len(illegal_files)} illegal files")

    return df, corrupted_summary, illegal_files


def filter_dataframe(df, filter_dict):
    """
    Filter a DataFrame based on a dictionary of allowed values for columns.

    Args:
        df: pandas DataFrame to filter
        filter_dict: Dictionary mapping column names to lists of allowed values

    Returns:
        Filtered DataFrame
    """
    # Start with a copy of the DataFrame
    filtered_df = df.copy()

    # Special case for 't' -> 't_length'
    if "t" in filter_dict and "t_length" in filtered_df.columns:
        t_values = filter_dict["t"]
        if isinstance(t_values, list) and isinstance(t_values[0], list):
            # Use the length of the t array as the filter value
            lengths = [len(t) for t in t_values]
            filtered_df = filtered_df[filtered_df["t_length"].isin(lengths)]

    # Process each key in the filter dictionary
    for key, allowed_values in filter_dict.items():
        # Skip the 't' key as we've already handled it
        if key == "t":
            continue

        # Skip keys that don't exist in the DataFrame
        if key not in filtered_df.columns:
            # Skip parallel_interpolation silently
            if key != "parallel_interpolation":
                print(f"Warning: Column '{key}' not found in DataFrame")
            continue

        # For scalar columns, use standard filtering
        if not isinstance(allowed_values[0], list):
            filtered_df = filtered_df[filtered_df[key].isin(allowed_values)]
            continue

        # For list columns (like diff_coeffs), use string comparison
        str_allowed = [str(val) for val in allowed_values]
        mask = filtered_df[key].astype(str).apply(lambda x: any(x == val for val in str_allowed))
        filtered_df = filtered_df[mask]

    return filtered_df


def get_reaction_results(row):
    """
    Extract the reaction graph from a stencil result row.

    Intended for use with ``df.apply(..., axis=1)``.

    Args:
        row (pandas.Series): A results DataFrame row containing ``stencil_results``.

    Returns:
        tuple: ``(reaction_graph, reaction_val_matrix)`` from
        :func:`mcastle_utils.construct_reaction_graph`.
    """
    stencil_graph = row["stencil_results"]["graph"]
    stencil_val_matrix = row["stencil_results"]["val_matrix"]
    reaction_graph, reaction_val_matrix = ms.construct_reaction_graph(stencil_graph, stencil_val_matrix)
    reaction_results = (reaction_graph, reaction_val_matrix)
    return reaction_results


def get_summary_results(row):
    """
    Extract the summarized stencil from a stencil result row.

    Intended for use with ``df.apply(..., axis=1)``.

    Args:
        row (pandas.Series): A results DataFrame row containing ``stencil_results``.

    Returns:
        tuple: ``(summary_graph, summary_val_matrix)`` from
        :func:`mcastle_utils.summarize_stencil`.
    """
    stencil_graph = row["stencil_results"]["graph"]
    stencil_val_matrix = row["stencil_results"]["val_matrix"]
    summary_graph, summary_val_matrix = ms.summarize_stencil(stencil_graph, stencil_val_matrix)
    summary_results = (summary_graph, summary_val_matrix)
    return summary_results


def apply_separate_stencil_angle_difference(row, verbose=False):
    """
    Compute the angle estimation error by averaging estimates across all species-pair stencils.

    For each (parent, child) variable pair, the per-species spatial graph is
    extracted, an angle is estimated, and the results are averaged before
    computing the difference from the ground-truth advection angle.

    Args:
        row (pandas.Series): A results row with ``stencil_results`` and ``velocity_angle``.
        verbose (bool): Print intermediate angle estimates if True.

    Returns:
        float: Absolute angle difference (degrees) between the ground truth
        and the averaged per-species estimate.
    """
    ground_truth_angle = row["velocity_angle"]
    mstencil_graph = row["stencil_results"]["graph"]
    mstencil_val_matrix = row["stencil_results"]["val_matrix"]
    graphs, val_matrices = ms.get_species_spatial_graphs(mstencil_graph, mstencil_val_matrix)
    angles = []
    for parent_var in range(2):
        for child_var in range(2):
            graph = graphs[parent_var, child_var]
            val_matrix = val_matrices[parent_var, child_var]
            try:
                angle_est = ms.get_angle_from_stencil(graph, val_matrix)
            except:
                if verbose:
                    print(f"Parent {parent_var} | Child {child_var} has no links.")
                continue
            angles.append(angle_est)
    angle_avg = ms.angle_average(angles)
    if verbose:
        print(f"Average angle over each graph: {angle_avg}")
    separate_stencil_angle_difference = ms.angle_difference(ground_truth_angle, angle_avg)
    if verbose:
        print(f"Angle difference: {separate_stencil_angle_difference}")
    return separate_stencil_angle_difference


def apply_reaction_mcc(row, ground_truth_reaction_graph):
    """
    Compute the MCC between the discovered and ground-truth reaction graphs.

    Args:
        row (pandas.Series): A results row containing ``reaction_results``.
        ground_truth_reaction_graph (numpy.ndarray): Ground-truth reaction graph
            (species × species × 2 string array).

    Returns:
        float: Matthews Correlation Coefficient.
    """
    reaction_graph = row["reaction_results"][0]
    return graph_metrics.matthews_correlation_coefficient(true_graph=ground_truth_reaction_graph, discovered_graph=reaction_graph)


def apply_reaction_mcc_no_auto(row, ground_truth_reaction_graph_no_auto):
    """
    Compute the MCC against a ground-truth reaction graph that excludes autocausal links.

    Args:
        row (pandas.Series): A results row containing ``reaction_results``.
        ground_truth_reaction_graph_no_auto (numpy.ndarray): Ground-truth graph
            with self-links removed.

    Returns:
        float: Matthews Correlation Coefficient.
    """
    reaction_graph = row["reaction_results"][0]
    return graph_metrics.matthews_correlation_coefficient(true_graph=ground_truth_reaction_graph_no_auto, discovered_graph=reaction_graph)


def apply_summary_stencil_angle(row):
    """
    Estimate the advection angle from the summarized (species-collapsed) stencil.

    Args:
        row (pandas.Series): A results row containing ``summary_results``.

    Returns:
        float: Estimated angle in degrees.
    """
    summary_graph = row["summary_results"][0]
    summary_val_matrix = row["summary_results"][1]
    stencil_angle = ms.get_angle_from_stencil(summary_graph, summary_val_matrix)
    return stencil_angle


def apply_mstencil_angle(row):
    """
    Estimate the advection angle directly from the full multivariate stencil.

    Args:
        row (pandas.Series): A results row containing ``stencil_results``.

    Returns:
        float: Estimated angle in degrees.
    """
    mstencil_graph = row["stencil_results"]["graph"]
    mstencil_val_matrix = row["stencil_results"]["val_matrix"]
    mstencil_angle = ms.get_angle_from_stencil(mstencil_graph, mstencil_val_matrix)
    return mstencil_angle


def apply_mstencil_angle_nonnegative(row):
    """
    Estimate the advection angle using the nonnegative-coefficient angle variant.

    Args:
        row (pandas.Series): A results row containing ``stencil_results``.

    Returns:
        float: Estimated angle in degrees (nonnegative variant).
    """
    mstencil_graph = row["stencil_results"]["graph"]
    mstencil_val_matrix = row["stencil_results"]["val_matrix"]
    mstencil_angle = ms.get_angle_from_stencil_nonnegative(mstencil_graph, mstencil_val_matrix)
    return mstencil_angle


def apply_angle_difference(row, angle_col_name):
    """
    Compute the difference between the ground-truth and an estimated advection angle.

    Args:
        row (pandas.Series): A results row containing ``velocity_angle`` and the
            estimated angle column named by ``angle_col_name``.
        angle_col_name (str): Column name of the estimated angle in ``row``.

    Returns:
        float: Absolute angle difference in degrees.
    """
    ground_truth_angle = row["velocity_angle"]
    estimated_angle = row[angle_col_name]
    angle_difference = ms.angle_difference(ground_truth_angle, estimated_angle)
    return angle_difference


def apply_confusion_matrix(row, ground_truth_reaction_graph):
    """
    Compute the confusion matrix for reaction-graph recovery.

    Args:
        row (pandas.Series): A results row containing ``reaction_results``.
        ground_truth_reaction_graph (numpy.ndarray): Ground-truth reaction graph.

    Returns:
        tuple: ``(TP, FP, FN, TN)`` counts.
    """
    reaction_graph = row["reaction_results"][0]
    return graph_metrics.get_confusion_matrix(ground_truth_reaction_graph, reaction_graph)


def compute_metrics(row):
    """
    Compute precision, recall, and FDR from confusion-matrix columns.

    Expects the row to have integer columns ``TP``, ``FP``, ``FN``.

    Args:
        row (pandas.Series): A results row with ``TP``, ``FP``, ``FN`` columns.

    Returns:
        pandas.Series: Series with keys ``precision``, ``recall``, ``FDR``.
    """
    TP = row["TP"]
    FP = row["FP"]
    FN = row["FN"]
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    FDR = FP / (TP + FP) if (TP + FP) > 0 else 0
    return pd.Series({"precision": precision, "recall": recall, "FDR": FDR})


def process_univariate_estimation(row):
    """
    Run univariate M-CaStLe-PC on each species independently and return the mean angle error.

    For each variable in the dataset, the algorithm is run with permissive thresholds
    (pc_alpha=0.9, graph_p_threshold=0.9) to recover an angle estimate; the mean
    absolute difference from the ground-truth advection angle is returned.

    Args:
        row (pandas.Series): A results row containing an ``experiment``
            (ADRExperiment instance with a loaded solution).

    Returns:
        float: Mean absolute angle estimation error (degrees) across all species.
    """
    experiment = row["experiment"]
    solution = experiment.solution
    data = solution.transpose((3, 0, 1, 2))

    parcorr = ParCorr(significance="analytic")

    angle_differences = []

    for var in range(data.shape[0]):
        results = ms.mv_CaStLe_PC(data=data[var : (var + 1), :, :, :], cond_ind_test=parcorr, pc_alpha=0.9, graph_p_threshold=0.9, verbose=0)
        graph = results["graph"]
        val_matrix = results["val_matrix"]

        # Apply metrics
        stencil_angle = ms.get_angle_from_stencil(graph, val_matrix)

        # Compute angle difference
        alpha, beta = experiment.velocity_parameters
        ground_truth_angle = ms.compute_angle(alpha, beta)
        angle_difference = ms.angle_difference(ground_truth_angle, stencil_angle)
        angle_differences.append(angle_difference)

    mean_difference = np.mean(angle_differences)

    return mean_difference


def main(results_dir, experiments_dir):
    """
    Load all stencil results, compute metrics, and save the analysis DataFrame.

    Args:
        results_dir (str): Directory containing stencil result pickles and
            where ``analysis_results.pkl`` will be written.
        experiments_dir (str): Directory containing the source ADRExperiment pickles.
    """
    print("Loading data...")
    if len(experiments_dir) == 0:
        print("Empty experiments directory passed!")
        sys.exit(1)
    if len(results_dir) == 0:
        print("Empty results directory passed!")
        sys.exit(1)

    results_df, corrupted_summary, illegal_files = load_results(results_dir, experiments_dir)
    if len(corrupted_summary) != 0:
        print(f"Corrupted files found:")
        print(corrupted_summary)

    if len(illegal_files) != 0:
        print(f"Illegal files found:")
        print(illegal_files)

    if len(results_df) < 1:
        print("No results loaded!")
        sys.exit(1)
    print("Data loaded.")

    ground_truth_reaction_graph = np.array([[["", "-->"], ["", "-->"]], [["", ""], ["", "-->"]]], dtype=object)
    ground_truth_reaction_graph_no_auto = np.array([[["", ""], ["", "-->"]], [["", ""], ["", ""]]], dtype=object)

    print("Computing results..")

    print("Computing reaction and summary results..")
    results_df["reaction_results"] = results_df.apply(get_reaction_results, axis=1)
    results_df["summary_results"] = results_df.apply(get_summary_results, axis=1)

    print("Computing reaction analysis..")
    results_df["Reaction Graph MCC"] = results_df.apply(lambda row: apply_reaction_mcc(row, ground_truth_reaction_graph), axis=1)
    results_df["Reaction Graph No Auto MCC"] = results_df.apply(lambda row: apply_reaction_mcc(row, ground_truth_reaction_graph_no_auto), axis=1)
    results_df[["TP", "FP", "FN", "TN"]] = results_df.apply(lambda row: apply_confusion_matrix(row, ground_truth_reaction_graph), axis=1, result_type="expand")
    results_df[["Precision", "Recall", "FDR"]] = results_df.apply(compute_metrics, axis=1)

    # print("Computing summary angle analysis..")
    # results_df["Summary Stencil Angle"] = results_df.apply(apply_summary_stencil_angle, axis=1)
    # results_df["Summary Angle Difference"] = results_df.apply(lambda row: apply_angle_difference(row, "Summary Stencil Angle"), axis=1)

    print("Computing M-Stencil angle analysis..")
    results_df["M-Stencil Angle"] = results_df.apply(apply_mstencil_angle, axis=1)
    results_df["M-Stencil Angle Difference"] = results_df.apply(lambda row: apply_angle_difference(row, "M-Stencil Angle"), axis=1)

    print("Computing Nonnegative M-Stencil angle analysis..")
    results_df["M-Stencil Angle Nonnegative"] = results_df.apply(apply_mstencil_angle_nonnegative, axis=1)
    results_df["M-Stencil Angle Difference Nonnegative"] = results_df.apply(lambda row: apply_angle_difference(row, "M-Stencil Angle Nonnegative"), axis=1)

    # print("Computing Separate Stencils angle analysis..")
    # results_df["Separate Stencils Angle Difference"] = results_df.apply(apply_separate_stencil_angle_difference, axis=1)

    # print("Computing mean univariate stencil angle difference...")
    # # Enable progress bar for pandas operations
    # tqdm.pandas(desc="Processing experiments")
    # results_df["Univariate Angle Difference Mean"] = results_df.progress_apply(process_univariate_estimation, axis=1)
    # # results_df["Univariate Angle Difference Mean"] = results_df.apply(process_univariate_estimation, axis=1)

    save_location = f"{results_dir}/analysis_results.pkl"
    print(f"Saving pickle at {save_location}")
    results_df.to_pickle(save_location)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process results and experiments directories.")
    parser.add_argument("results_dir", type=str, help="Path to the results directory")
    parser.add_argument("experiments_dir", type=str, help="Path to the experiments directory")

    args = parser.parse_args()

    main(args.results_dir, args.experiments_dir)
