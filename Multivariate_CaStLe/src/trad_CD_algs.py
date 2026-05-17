"""
Causal Discovery Algorithms for Spatiotemporal Grid Data

This module implements causal discovery algorithms for analyzing time series data
from variables measured across 2D spatial grids. It provides wrapper functions that
adapt algorithms from the Tigramite and CausalNex libraries to work with 4D numpy
arrays of shape (variable_n, X, Y, T).

Available Algorithms:
    - PC: Constraint-based causal discovery using conditional independence tests
    - PC_stable: PC-Stable variant of the PC algorithm with order-independent results
    - PCMCI: Peter and Clark Momentary Conditional Independence algorithm combining
             PC algorithm with MCI step for robust causal discovery
    - DYNOTEARS: Score-based causal discovery using continuous optimization

All algorithms return results in Tigramite's graph format, enabling consistent
downstream analysis and visualization.

Typical Usage:
    >>> from tigramite.independence_tests import ParCorr
    >>> cond_ind_test = ParCorr()
    >>> results = PCMCI_algo(data, cond_ind_test, min_tau=0, max_tau=1, pc_alpha=0.05)
    >>> graph = results['graph']
    >>> val_matrix = results['val_matrix']

Dependencies:
    - numpy: Array operations and data handling
    - tigramite: PC, PC_stable, and PCMCI algorithms, conditional independence tests
    - causalnex: DYNOTEARS algorithm
    - pandas: DataFrame conversion for CausalNex compatibility
"""

import numpy as np
import pandas as pd
import warnings
from causalnex.structure import StructureModel
from causalnex.structure.dynotears import from_pandas_dynamic
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.independence_tests_base import CondIndTest
from typing import Dict, Any, Union


def PC(
    data: np.ndarray,
    cond_ind_test: CondIndTest,
    min_tau: int,
    max_tau: int,
    pc_alpha: float,
    pval_threshold: float = 0.01,
    fdr_method: str = None,
) -> Dict[str, Any]:
    """The PC algorithm for time series data.

    Args:
        data (np.ndarray): A 4D numpy array of shape (variable_n, X, Y, T) representing the data for multiple variables across a 2D spatial grid over time.
        cond_ind_test (CondIndTest): An instance of a conditional independence test to be used by the causal discovery algorithm.
        min_tau (int): Minimum time lag to test.
        max_tau (int): Maximum time lag to test.
        pc_alpha (float): The significance level for conditional independence tests in the PC algorithm.
        pval_threshold (float, optional): Significance level at which the p_matrix from PC is thresholded to get the graph. Defaults to 0.01.
        fdr_method : str, optional (default: None) Correction method, currently implemented is Benjamini-Hochberg False Discovery Rate method with "bh".

    Returns:
        Dict[str, Any]: Dictionary containing the reconstructed graph and the value matrix with coefficients.
    """
    # Reshape data for input to tigramite
    data = np.reshape(data, (data.shape[3], data.shape[0] * data.shape[1] * data.shape[2]))

    if data.shape[0] < data.shape[1]:
        warnings.warn("More columns than rows! Either there are more variables than observations, or you need to transpose the data.")

    pcmci_df = pp.DataFrame(data)

    pcmci = PCMCI(dataframe=pcmci_df, cond_ind_test=cond_ind_test, verbosity=0)

    results = pcmci.run_pcalg(
        tau_min=min_tau,
        tau_max=max_tau,
        pc_alpha=pc_alpha,
    )
    if fdr_method:
        if fdr_method == "bh":
            fdr_method = "fdr_bh"  # Rename to conform to tigramite's expected convention.
        q_matrix = pcmci.get_corrected_pvalues(
            p_matrix=results["p_matrix"],
            tau_min=min_tau,
            tau_max=max_tau,
            fdr_method=fdr_method,
        )
        reconstructed_graph = pcmci.get_graph_from_pmatrix(
            p_matrix=q_matrix,
            alpha_level=pval_threshold,
            tau_min=min_tau,
            tau_max=max_tau,
        )
        results["graph"] = reconstructed_graph
        results["q_matrix"] = q_matrix

    return results


def PC_stable(
    data: np.ndarray,
    cond_ind_test: CondIndTest,
    min_tau: int,
    max_tau: int,
    pc_alpha: float,
    pval_threshold: float = 0.01,
    fdr_method: str = None,
) -> Dict[str, Any]:
    """The PC-stable algorithm for time series data.

    Args:
        data (np.ndarray): A 4D numpy array of shape (variable_n, X, Y, T) representing the data for multiple variables across a 2D spatial grid over time.
        cond_ind_test (CondIndTest): An instance of a conditional independence test to be used in the PC-stable algorithm.
        min_tau (int): Minimum time lag to test.
        max_tau (int): Maximum time lag to test.
        pc_alpha (float): The significance level used in the PC-stable algorithm.
        pval_threshold (float, optional): Significance level at which the p_matrix is thresholded to obtain the reconstructed graph. Defaults to 0.01.
        fdr_method (str, optional): Correction method for p-values. Currently implemented is the Benjamini-Hochberg FDR method, indicated by "bh".

    Returns:
        Dict[str, Any]: Dictionary containing the results from the PC-stable algorithm, including the reconstructed graph and the value matrix with coefficients.
    """
    # Reshape data for input to tigramite: time x (all variables flattened)
    data = np.reshape(data, (data.shape[3], data.shape[0] * data.shape[1] * data.shape[2]))

    if data.shape[0] < data.shape[1]:
        warnings.warn("More columns than rows!")

    pcmci_df = pp.DataFrame(data)
    pcmci = PCMCI(dataframe=pcmci_df, cond_ind_test=cond_ind_test, verbosity=0)

    all_parents = pcmci.run_pc_stable(
        tau_min=min_tau,
        tau_max=max_tau,
        pc_alpha=pc_alpha,
    )

    # Build results dict from class attributes
    results = {
        "parents": all_parents,
        "p_matrix": pcmci.p_matrix,
        "val_matrix": pcmci.val_matrix,
    }

    if fdr_method:
        if fdr_method == "bh":
            fdr_method = "fdr_bh"
        q_matrix = pcmci.get_corrected_pvalues(
            p_matrix=pcmci.p_matrix,  # Use class attribute
            tau_min=min_tau,
            tau_max=max_tau,
            fdr_method=fdr_method,  # Use the converted variable
        )
        reconstructed_graph = pcmci.get_graph_from_pmatrix(
            p_matrix=q_matrix,
            alpha_level=pval_threshold,
            tau_min=min_tau,
            tau_max=max_tau,
        )
        results["graph"] = reconstructed_graph
        results["q_matrix"] = q_matrix

    return results


def PCMCI_alg(
    data: np.ndarray,
    cond_ind_test: CondIndTest,
    min_tau: int,
    max_tau: int,
    pc_alpha: float = 0.05,
    pval_threshold: float = 0.05,
    fdr_method: str = None,
) -> Dict[str, Any]:
    """The PCMCI algorithm for time series causal discovery.

    PCMCI combines the PC algorithm with the Momentary Conditional Independence (MCI)
    test to provide robust causal discovery for time series data. The PC step identifies
    potential parents, while the MCI step performs additional conditional independence
    tests to orient edges and establish causal relationships.

    Args:
        data (np.ndarray): A 4D numpy array of shape (variable_n, X, Y, T) representing
            the data for multiple variables across a 2D spatial grid over time.
        cond_ind_test (CondIndTest): An instance of a conditional independence test to
            be used by the causal discovery algorithm (e.g., ParCorr, GPDC, CMIknn).
        min_tau (int): Minimum time lag to test.
        max_tau (int): Maximum time lag to test.
        pc_alpha (float, optional): The significance level for conditional independence
            tests in the PC algorithm step. Defaults to 0.05.
        pval_threshold (float, optional): The significance level for the MCI step.
            Defaults to 0.05.
        fdr_method (str, optional): Correction method for multiple testing. Currently
            implemented is the Benjamini-Hochberg False Discovery Rate method with "bh".
            If None, no correction is applied. Defaults to None.

    Returns:
        Dict[str, Any]: Dictionary containing the results from the PCMCI algorithm:
            - 'graph': The reconstructed causal graph (array of shape [N, N, tau_max+1])
            - 'val_matrix': Matrix with estimated coefficients/test statistics
            - 'p_matrix': Matrix with p-values from conditional independence tests
            - 'q_matrix': FDR-corrected p-values (only if fdr_method is specified)
            - 'conf_matrix': Confidence intervals (if supported by the test)
    """
    # Reshape data for input to tigramite: time x (all variables flattened)
    data = np.reshape(data, (data.shape[3], data.shape[0] * data.shape[1] * data.shape[2]))

    if data.shape[0] < data.shape[1]:
        warnings.warn("More columns than rows! Either there are more variables than observations, or you need to transpose the data.")

    pcmci_df = pp.DataFrame(data)

    pcmci = PCMCI(dataframe=pcmci_df, cond_ind_test=cond_ind_test, verbosity=0)

    results = pcmci.run_pcmci(
        tau_min=min_tau,
        tau_max=max_tau,
        pc_alpha=pc_alpha,
        alpha_level=pval_threshold,
    )

    if fdr_method:
        if fdr_method == "bh":
            fdr_method = "fdr_bh"  # Rename to conform to tigramite's expected convention
        q_matrix = pcmci.get_corrected_pvalues(
            p_matrix=results["p_matrix"],
            tau_min=min_tau,
            tau_max=max_tau,
            fdr_method=fdr_method,
        )
        reconstructed_graph = pcmci.get_graph_from_pmatrix(
            p_matrix=q_matrix,
            alpha_level=pval_threshold,
            tau_min=min_tau,
            tau_max=max_tau,
        )
        results["graph"] = reconstructed_graph
        results["q_matrix"] = q_matrix

    return results


def get_graph_from_structure_model(structure_model: StructureModel, include_val_matrix: bool = True) -> Union[tuple, list]:
    """
    Convert a causalnex.structure.StructureModel to a string-graph and val_matrix in the style of the Tigramite library.

    StructureModel inherits networkx's DiGraph, which this conversion relies upon.

    Parameters
    ----------
    structure_model : causalnex.structure.StructureModel
        Graph from CausalNex.
    include_val_matrix : bool, optional
        Whether to return a value matrix. If False, only the string-graph is returned. Default is True.

    Returns
    -------
    Union[tuple, list]
        A tuple of two lists (string-graph and float-graph) if include_val_matrix is True, or just a list (string-graph) if include_val_matrix is False.
    """
    parents_to_add = []
    min_lag = 0
    max_lag = 0
    num_vars = 0
    for item in structure_model.adjacency():
        parent_variable = int(item[0].split("_lag")[0])
        parent_lag = int(item[0].split("_lag")[-1])
        children = [
            (
                child_val := int(key.split("_lag")[0]),
                int(key.split("_lag")[1]),
                value["weight"],
            )
            for key, value in item[1].items()
        ]
        if len(children) > 0:
            parents_to_add.append((parent_variable, parent_lag, children))

        if parent_lag < min_lag:
            min_lag = parent_lag
        if parent_lag > max_lag:
            max_lag = parent_lag
        if parent_variable >= num_vars:
            num_vars = parent_variable + 1

    #  (array of shape [N, N, tau_max+1])
    graph = np.full((num_vars, num_vars, max_lag + 1), fill_value="", dtype="<U3")
    if include_val_matrix:
        val_matrix = np.zeros((num_vars, num_vars, max_lag + 1), dtype=float)
    for parent in parents_to_add:
        parent_var = parent[0]
        parent_lag = parent[1]
        for child in parent[2]:
            child_var = child[0]
            child_lag = child[1]
            child_weight = child[2]
            # lag = child_lag - parent_lag
            lag = parent_lag - child_lag
            if lag < 0:
                raise ValueError(
                    f"Invalid temporal relationship: parent at lag {parent_lag} cannot cause "
                    f"child at lag {child_lag} (would require negative lag {lag}, implying future causes past). "
                    f"Parent node: {item[0]}, Child nodes: {[k for k in item[1].keys()]}"
                )
            graph[parent_var, child_var, lag] = "-->"
            if parent_lag == 0:
                graph[child_var, parent_var, 0] = "<--"  # <-- used because of what Tigramite does.
            if include_val_matrix:
                val_matrix[parent_var, child_var, lag] = child_weight
                if parent_lag == 0:
                    val_matrix[child_var, parent_var, 0] = child_weight
    if include_val_matrix:
        return graph, val_matrix
    else:
        return graph


def DYNOTEARS(
    data: np.ndarray,
    max_tau: int,
    lambda_w: float = 0.01,
    lambda_a: float = 0.01,
    max_iter: int = 100,
) -> Dict[str, Any]:
    """The DYNOTEARS score-optimization-based causal discovery algorithm for time series data.

    Args:
        data (np.ndarray): A 4D numpy array of shape (variable_n, X, Y, T) representing the data for multiple variables across a 2D spatial grid over time.
        max_tau (int): Maximum time lag to test.
        lambda_w (float, optional):
            L1 regularization penalty for contemporaneous (lag-0) edges. Higher values
            encourage sparser same-time-step relationships. Defaults to 0.01.
        lambda_a (float, optional):
            L1 regularization penalty for lagged (temporal) edges. **Primary tuning parameter.**
            Controls sparsity of discovered causal structure. Higher values (0.05-0.1) yield
            very sparse graphs with only strongest links; lower values (0.001-0.01) discover
            more relationships including weaker inter-variable links. Defaults to 0.01.
        max_iter (int, optional):
            Maximum iterations for DYNOTEARS dual ascent optimization. More iterations
            improve convergence, especially with low regularization or many variables.
            Increase to 200-500 if convergence warnings appear. Defaults to 100.

    Returns:
        Dict[str, Any]: Dictionary containing the reconstructed graph and the value matrix with coefficients.
    """
    # Reshape data for input to CausalNex: time x (all variables flattened)
    data = np.reshape(data, (data.shape[3], data.shape[0] * data.shape[1] * data.shape[2]))

    if data.shape[0] < data.shape[1]:
        warnings.warn("More columns than rows! Either there are more variables than observations, or you need to transpose the data.")

    # Create column names and DataFrame
    col_names = ["" + str(i) for i in np.arange(data.shape[1])]
    df = pd.DataFrame(data=data, columns=col_names)

    # Run DYNOTEARS
    structure_model = from_pandas_dynamic(
        df,
        p=max_tau,
        lambda_w=lambda_w,
        lambda_a=lambda_a,
        max_iter=max_iter,
    )

    # Convert to Tigramite-style graph and val_matrix
    graph, val_matrix = get_graph_from_structure_model(structure_model, include_val_matrix=True)

    results = {
        "graph": graph,
        "val_matrix": val_matrix,
    }

    return results


def plot_multigrid_graph(
    graph, val_matrix, var_names=None, fig=None, axs=None, grid_shape=(10, 10), figsize_per_panel=(4, 4), node_size=0.1, arrow_linewidth=1.0, arrowhead_size=20, link_threshold=None
):
    """
    Plot causal graphs for multi-variable gridded data in panel layout.

    Parameters:
    -----------
    graph : ndarray, shape (V*N*M, V*N*M, tau_max+1)
        Tigramite graph array
    val_matrix : ndarray, same shape as graph
        Tigramite value matrix
    var_names : list of str, optional
        Variable names. If None, uses alphanumeric labels (A, B, C, ...)
    grid_shape : tuple (N, M)
        Spatial grid dimensions (X, Y)
    figsize_per_panel : tuple
        Size of each subplot panel
    node_size : float
        Size of nodes (default 0.1)
    arrow_linewidth : float
        Width of arrow lines (default 1.0)
    arrowhead_size : float
        Size of arrowheads (default 20)
    link_threshold : float, optional
        Only show links with |val_matrix| > threshold
    """
    import matplotlib
    from matplotlib.artist import Artist
    from matplotlib.colors import ListedColormap
    from matplotlib import pyplot as plt
    from tigramite import plotting as tp
    import string

    N, M = grid_shape
    V = graph.shape[0] // (N * M)  # Number of variables

    # Default variable names
    if var_names is None:
        if V <= 26:
            var_names = list(string.ascii_uppercase[:V])
        else:
            var_names = [f"V{i}" for i in range(V)]

    # Create figure with V×V subplots
    if not fig or axs is None:
        fig, axs = plt.subplots(V, V, figsize=(figsize_per_panel[0] * V, figsize_per_panel[1] * V), constrained_layout=True)
        if V == 1:
            axs = np.array([[axs]])
        elif V > 1 and axs.ndim == 1:
            axs = axs.reshape(V, V)

    # Node positions for spatial grid
    x_pos = list(np.array([[i for i in range(M)] for j in range(N)]).flatten())
    y_pos = [i for i in range(N) for j in range(M)]
    y_pos.reverse()
    node_positions = {
        "x": x_pos,
        "y": y_pos,
    }

    # Colormaps
    cmap_N = 256
    white_vals = np.ones((cmap_N, 4))
    white_cmap = ListedColormap(white_vals)

    # Extract and plot each variable pair
    for target_var in range(V):
        for source_var in range(V):
            ax = axs[target_var, source_var]

            # Extract sub-graph for this variable pair
            target_indices = np.arange(target_var, V * N * M, V)
            source_indices = np.arange(source_var, V * N * M, V)

            sub_graph = graph[np.ix_(target_indices, source_indices)]
            sub_val_matrix = val_matrix[np.ix_(target_indices, source_indices)]

            # Apply link threshold if requested
            if link_threshold is not None:
                mask = np.abs(sub_val_matrix) < link_threshold
                sub_graph = sub_graph.copy()
                sub_graph[mask] = ""

            # Plot this variable pair
            tp.plot_graph(
                fig_ax=(fig, ax),
                graph=sub_graph,
                val_matrix=sub_val_matrix,
                link_label_fontsize=0.0,
                node_size=node_size,
                arrow_linewidth=arrow_linewidth,
                arrowhead_size=arrowhead_size,
                curved_radius=0.0,
                cmap_edges=white_cmap,
                cmap_nodes="binary",
                show_colorbar=False,
                var_names=[""] * (N * M),
                node_pos=node_positions,
            )

            # Remove "1" text labels
            for child in ax.get_children():
                if isinstance(child, matplotlib.text.Text):
                    if child.get_text() == "1":
                        Artist.set_visible(child, False)

            ax.patch.set_alpha(0)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])

            # Add labels on edges
            if target_var == 0:
                ax.set_title(f"{var_names[source_var]}", fontsize=10, fontweight="bold")
            if source_var == 0:
                ax.set_ylabel(f"{var_names[target_var]}", fontsize=10, fontweight="bold", rotation=0, ha="right", va="center", labelpad=10)

    # Add overall labels
    fig.suptitle("Source Variable →", x=0.5, y=0.99, fontsize=12, fontweight="bold")
    fig.text(0.01, 0.5, "Target Variable ↓", ha="left", va="center", rotation=90, fontsize=12, fontweight="bold")

    return fig, axs


# Preset configurations for different grid sizes
def get_size_preset(grid_size):
    """Get recommended plotting parameters for different grid sizes"""
    if grid_size <= 10:
        return {"figsize_per_panel": (4, 4), "node_size": 0.3, "arrow_linewidth": 2.0, "arrowhead_size": 30}
    elif grid_size <= 30:
        return {"figsize_per_panel": (5, 5), "node_size": 0.15, "arrow_linewidth": 1.0, "arrowhead_size": 20}
    elif grid_size <= 50:
        return {"figsize_per_panel": (6, 6), "node_size": 0.08, "arrow_linewidth": 0.5, "arrowhead_size": 10}
    else:  # 50-100+
        return {"figsize_per_panel": (8, 8), "node_size": 0.04, "arrow_linewidth": 0.3, "arrowhead_size": 5}
