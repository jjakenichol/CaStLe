"""
This file contains the minimum code to run CaStLe using various causal discovery algorithms for the Parent Identification Phase (PIP).

PC-Stable, PCMCI, and several other data structure related supporting functions are imported from the TIGRAMITE library.

Functions:
----------
1. CaStLe_PCMCI(data: np.ndarray, cond_ind_test: CondIndTest, pc_alpha: float, rows_inverted=True, dependence_threshold=0.01, dependencies_wrap=False) -> tuple
    The CaStLe algorithm implemented with PCMCI for the parent-identification phase.

2. CaStLe_PC(data: np.ndarray, cond_ind_test: CondIndTest, pc_alpha: float, rows_inverted=False, dependence_threshold=0.01, dependencies_wrap=False) -> tuple
    The CaStLe algorithm implemented with PC for the parent-identification phase.

3. PC(data: np.ndarray, cond_ind_test: CondIndTest, min_tau: int, max_tau: int, pc_alpha: float, pval_threshold=0.01) -> tuple
    The PC algorithm for time series data.

4. PC_stable(data: np.ndarray, cond_ind_test: CondIndTest, min_tau: int, max_tau: int, pc_alpha: float) -> tuple
    The PC-Stable algorithm for time series data.

5. concatenate_timeseries_wrapping(data: np.ndarray, rows_inverted=True, include_cell_index_column=False) -> np.ndarray
    Concatenates time series data with wrapping dependencies.

6. concatenate_timeseries_nonwrapping(data: np.ndarray, rows_inverted=True, include_cell_index_column=False) -> np.ndarray
    Concatenates time series data without wrapping dependencies.

Example Usage:
--------------
To run CaStLe with PCMCI:
>>> graph, val_matrix = CaStLe_PCMCI(data, cond_ind_test, pc_alpha=0.05)

To run CaStLe with PC:
>>> graph, val_matrix = CaStLe_PC(data, cond_ind_test, pc_alpha=0.05)

To run the PC algorithm:
>>> graph, val_matrix = PC(data, cond_ind_test, min_tau=1, max_tau=1, pc_alpha=0.05)

To run the PC-Stable algorithm:
>>> results = PC_stable(data, cond_ind_test, min_tau=1, max_tau=1, pc_alpha=0.05)
"""

import pc_stable_single
import stable_SCM_generator as scmg

import pandas as pd
import numpy as np
import warnings
from causalnex.structure.dynotears import from_pandas_dynamic
from tigramite import data_processing as pp
from tigramite.independence_tests.independence_tests_base import CondIndTest
from tigramite.pcmci import PCMCI
from tigramite.toymodels import structural_causal_processes


def concatenate_timeseries_wrapping(data: np.ndarray, rows_inverted: bool, include_cell_index_column=False) -> np.ndarray:
    """Concatenates time series data in a wrapping manner.

    Args:
        data (np.ndarray): The input data of shape (N, N, T).
        rows_inverted (bool): Whether data rows are inverted. Inverted means the row above is (row-1).
        include_cell_index_column (bool, optional): Whether to include a cell index column in the concatenated data. Defaults to False.

    Returns:
        np.ndarray: The concatenated data of shape (M, 9) or (M, 10) if include_cell_index_column is True.
    """
    rows = data.shape[0]
    cols = data.shape[0]
    if include_cell_index_column:
        concatenated_data = [[] for i in range(10)]
        index = 0
    else:
        concatenated_data = [[] for i in range(9)]
    # Collect time series according to their position relative to each grid cell.
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            if rows_inverted:
                from_left = data[row, col - 1, :]
                from_right = data[row, (col + 1) % rows, :]
                from_top = data[row - 1, col, :]
                from_bottom = data[(row + 1) % rows, col, :]
                from_top_left = data[row - 1, col - 1, :]
                from_top_right = data[row - 1, (col + 1) % cols, :]
                from_bot_left = data[(row + 1) % rows, col - 1, :]
                from_bot_right = data[(row + 1) % rows, (col + 1) % cols, :]
                from_self = data[row, col, :]
            else:
                from_left = data[row, col - 1, :]
                from_right = data[row, (col + 1) % rows, :]
                from_top = data[row + 1, col, :]
                from_bottom = data[(row - 1) % rows, col, :]
                from_top_left = data[row + 1, col - 1, :]
                from_top_right = data[row + 1, (col + 1) % cols, :]
                from_bot_left = data[(row - 1) % rows, col - 1, :]
                from_bot_right = data[(row - 1) % rows, (col + 1) % cols, :]
                from_self = data[row, col, :]

            # Concatenate each time series according to its position in the reduced space.
            concatenated_data[0].extend(from_top_left)
            concatenated_data[1].extend(from_top)
            concatenated_data[2].extend(from_top_right)
            concatenated_data[3].extend(from_left)
            concatenated_data[4].extend(from_self)
            concatenated_data[5].extend(from_right)
            concatenated_data[6].extend(from_bot_left)
            concatenated_data[7].extend(from_bottom)
            concatenated_data[8].extend(from_bot_right)

            if include_cell_index_column:
                concatenated_data[9].extend([index] * len(from_self))
                index += 1

    concatenated_data = np.array(concatenated_data).transpose()

    return concatenated_data


def concatenate_timeseries_nonwrapping(data: np.ndarray, rows_inverted: bool, include_cell_index_column=False) -> np.ndarray:
    """Concatenates time series data in a non-wrapping manner.

    Args:
        data (np.ndarray): The input data of shape (N, N, T).
        rows_inverted (bool): Whether data rows are inverted. Inverted means the row above is (row-1).
        include_cell_index_column (bool, optional): Whether to include a cell index column in the concatenated data. Defaults to False.

    Returns:
        np.ndarray: The concatenated data of shape (M, 9) or (M, 10) if include_cell_index_column is True.
    """
    GRID_SIZE_ROW = data.shape[0]
    GRID_SIZE_COL = data.shape[1]

    # create one cell buffer such that all cells evaluated have neighbors on all sides
    ROW_RANGE = range(1, GRID_SIZE_ROW - 1)
    # create one cell buffer such that all cells evaluated have neighbors on all sides
    COL_RANGE = range(1, GRID_SIZE_COL - 1)

    if include_cell_index_column:
        concatenated_data = [[] for i in range(10)]
        index = 0
    else:
        concatenated_data = [[] for i in range(9)]
    # Collect time series according to their position relative to each grid cell.
    for row in ROW_RANGE:
        for col in COL_RANGE:
            if rows_inverted:
                from_left = data[row, col - 1, :]
                from_right = data[row, col + 1, :]
                from_top = data[row - 1, col, :]
                from_bottom = data[row + 1, col, :]
                from_top_left = data[row - 1, col - 1, :]
                from_top_right = data[row - 1, col + 1, :]
                from_bot_left = data[row + 1, col - 1, :]
                from_bot_right = data[row + 1, col + 1, :]
                from_self = data[row, col, :]
            else:
                from_left = data[row, col - 1, :]
                from_right = data[row, col + 1, :]
                from_top = data[row + 1, col, :]
                from_bottom = data[row - 1, col, :]
                from_top_left = data[row + 1, col - 1, :]
                from_top_right = data[row + 1, col + 1, :]
                from_bot_left = data[row - 1, col - 1, :]
                from_bot_right = data[row - 1, col + 1, :]
                from_self = data[row, col, :]

            # Concatenate each time series according to its position in the reduced space.
            concatenated_data[0].extend(from_top_left)
            concatenated_data[1].extend(from_top)
            concatenated_data[2].extend(from_top_right)
            concatenated_data[3].extend(from_left)
            concatenated_data[4].extend(from_self)
            concatenated_data[5].extend(from_right)
            concatenated_data[6].extend(from_bot_left)
            concatenated_data[7].extend(from_bottom)
            concatenated_data[8].extend(from_bot_right)

            if include_cell_index_column:
                concatenated_data[9].extend([index] * data.shape[2])
                index += 1

    concatenated_data = np.array(concatenated_data).transpose()

    return concatenated_data


def get_stencil_graph(neighborhood_dependence_matrix, func=(lambda x: x), return_val_matrix=False):
    """
    Constructs a stencil graph from a neighborhood dependence matrix (NDM).

    Args:
        neighborhood_dependence_matrix (np.ndarray): A 2D numpy array representing the neighborhood dependence matrix.
        func (callable, optional): A function to apply to the dependencies. Defaults to the identity function (lambda x: x).
        return_val_matrix (bool, optional): If True, returns a value matrix along with the stencil graph. Defaults to False.

    Returns:
        np.ndarray: The stencil graph.
        np.ndarray (optional): The value matrix if `return_val_matrix` is True.
    """
    stencil_center = 4
    neighborhood_dependence_matrix = np.asarray(neighborhood_dependence_matrix)
    ndm = neighborhood_dependence_matrix.flatten()
    # Make structural causal model of NDM
    SCM = {i: [((j, -1), 0.0, func) for j in range(ndm.shape[0])] for i in range(ndm.shape[0])}

    # for row in range(neighborhood_dependence_matrix.shape[0]):
    SCM[stencil_center] = [((col, 0), ndm[col], func) for col in range(ndm.shape[0])]
    SCM[stencil_center] = [((col, -1), ndm[col], func) for col in range(ndm.shape[0])]

    stencil = structural_causal_processes.links_to_graph(SCM)

    if return_val_matrix:
        val_matrix = np.zeros(
            (
                neighborhood_dependence_matrix.shape[0] ** 2,
                neighborhood_dependence_matrix.shape[1] ** 2,
                2,
            )
        )
        for row in range(val_matrix.shape[0]):
            for col in range(val_matrix.shape[1]):
                coefficient = SCM[row][col][1]
                val_matrix[col, row, 1] = coefficient
        return stencil, val_matrix
    else:
        return stencil


def get_ndm(center_node_parents):
    """
    Constructs a neighborhood dependence matrix (NDM) from the parents of the center node.

    Args:
        center_node_parents (list of tuples): List of tuples representing the parents of the center node. Each tuple
                                              contains (parent_index, lag, value).

    Returns:
        np.ndarray: A 3x3 numpy array representing the neighborhood dependence matrix.
    """
    # Determine if center_node_parents contains any parents
    contains_parents = True
    if len(center_node_parents) == 0:
        contains_parents = False

    coefficients = False
    if contains_parents:
        if max([len(val) for val in center_node_parents]) == 3:
            coefficients = True
    else:
        coefficients = True

    # Get NDM from identified stencil
    if coefficients:
        ndm = np.zeros((9))
        for parent in center_node_parents:
            ndm[parent[0]] = parent[2]
        ndm = ndm.reshape((3, 3))
    else:
        ndm = np.zeros((9))
        for parent in center_node_parents:
            ndm[parent[0]] = 1.0
        ndm = ndm.reshape((3, 3))
    return ndm


def get_parents(graph, val_matrix=None, include_lagzero_parents=True, output_val_matrix=False):
    """
    Extracts parent nodes from a given graph and optionally returns their values from a value matrix.

    Args:
        graph (np.ndarray): A 3D numpy array representing the graph structure, where
                            graph[i, j, tau] indicates the relationship between node i at time t and node j at time t+tau.
        val_matrix (np.ndarray, optional): A 3D numpy array of the same shape as `graph` containing values (e.g., coefficients)
                                           associated with the edges in the graph. Defaults to None.
        include_lagzero_parents (bool, optional): If True, includes parents with zero lag (i.e., parents at the same time step).
                                                  Defaults to True.
        output_val_matrix (bool, optional): If True, includes the values from `val_matrix` in the output dictionary.
                                            Defaults to False.

    Returns:
        dict: A dictionary where keys are node indices and values are lists of parent nodes. If `output_val_matrix` is True,
              each parent node is represented as a tuple (parent_index, lag, value), otherwise as (parent_index, lag).
    """
    parents_dict = dict()
    if val_matrix is not None:
        assert graph.shape == val_matrix.shape, "graph and val_matrix shapes do not agree"
    for j in range(graph.shape[0]):
        # Get the good links
        if include_lagzero_parents:
            good_links = np.argwhere((graph[:, j, :] == "-->") | (graph[:, j, :] == "o-o"))
            # Build a dictionary from these links to their values
            if val_matrix is not None:
                links = {(i, -tau): val_matrix[i, j, abs(tau)] for i, tau in good_links}
            else:
                links = {(i, -tau): 1 for i, tau in good_links}
        else:
            good_links = np.argwhere((graph[:, j, 1:] == "-->") | (graph[:, j, 1:] == "o-o"))
            # Build a dictionary from these links to their values
            if val_matrix is not None:
                links = {(i, -tau - 1): val_matrix[i, j, abs(tau) + 1] for i, tau in good_links}
            else:
                links = {(i, -tau - 1): 1 for i, tau in good_links}
        # Sort by value
        if output_val_matrix:
            parents_dict[j] = [(*link, links[link]) for link in sorted(links, key=(lambda x: np.abs(links.get(x))), reverse=True)]
        else:
            parents_dict[j] = sorted(links, key=(lambda x: np.abs(links.get(x))), reverse=True)
    return parents_dict


def get_expanded_graph_from_parents(center_node_parents, full_grid_dimension, wrapping=False):
    """
    Returns the expanded graph from the stencil, i.e., the stencil applied to all nodes in a NxN graph.

    Args:
        center_node_parents (list of tuples): List of the stencil's center node's parents - encodes the stencil.
        full_grid_dimension (int): The dimension N in the NxN grid/graph.
        wrapping (bool, optional): Whether the output expanded graph should have wrapping edges. Defaults to False.

    Returns:
        np.ndarray: The expanded graph.
        np.ndarray (optional): The value matrix if coefficients are included in the center_node_parents.
    """
    # Determine if center_node_parents contains any parents
    contains_parents = True
    if len(center_node_parents) == 0:
        contains_parents = False

    coefficients = False
    if contains_parents:
        if max([len(val) for val in center_node_parents]) == 3:
            coefficients = True
    else:
        coefficients = True

    ndm = get_ndm(center_node_parents)

    # Construct full graph from NDM of stencil
    if wrapping:
        dynamics_matrix = scmg.create_coefficient_matrix(ndm, full_grid_dimension)
    else:
        dynamics_matrix = scmg.create_nonwrapping_coefficient_matrix(ndm, full_grid_dimension)
    if coefficients:
        full_graph, val_matrix = scmg.get_graph_from_coefficient_matrix(dynamics_matrix, return_val_matrix=True)
        return full_graph, val_matrix
    else:
        full_graph = scmg.get_graph_from_coefficient_matrix(dynamics_matrix, return_val_matrix=False)
        return full_graph


def get_expanded_graph_from_stencil_graph(
    graph,
    val_matrix,
    full_grid_dimension,
    include_lagzero_parents=False,
    wrapping=False,
):
    """
    Expands a stencil graph to a full NxN graph.

    Args:
        graph (np.ndarray): A 3D numpy array representing the stencil graph.
        val_matrix (np.ndarray): A 3D numpy array of the same shape as `graph` containing values (e.g., coefficients)
                                 associated with the edges in the graph.
        full_grid_dimension (int): The dimension N in the NxN grid/graph.
        include_lagzero_parents (bool, optional): If True, includes parents with zero lag (i.e., parents at the same time step).
                                                  Defaults to False.
        wrapping (bool, optional): Whether the output expanded graph should have wrapping edges. Defaults to False.

    Returns:
        np.ndarray: The expanded graph.
        np.ndarray: The value matrix for the expanded graph.
    """
    node_parents = get_parents(
        graph,
        val_matrix=val_matrix,
        include_lagzero_parents=include_lagzero_parents,
        output_val_matrix=True,
    )
    center_node_parents = node_parents[4]
    full_graph, full_val_matrix = get_expanded_graph_from_parents(center_node_parents, full_grid_dimension=full_grid_dimension, wrapping=wrapping)
    return full_graph, full_val_matrix


def CaStLe_PCstable(
    data: np.ndarray,
    cond_ind_test: CondIndTest,
    pc_alpha: float,
    dependence_threshold=0.0,
    rows_inverted=False,
    dependencies_wrap=True,
) -> tuple:
    """The CaStLe algorithm implemented with PC-stable for the parent-identification phase.

    Args:
        data (np.ndarray):
            The data of shape (N, N, T) to be given to CaStLe-PC.
        cond_ind_test (CondIndTest):
            The conditional independence test to be used in the parent-identification phase.
        pc_alpha (float):
            Significance level in PCMCI.
        rows_inverted (bool, optional):
            Whether data rows are inverted. Inverted means the row above is (row-1). Defaults to False.
        dependence_threshold (float, optional):
            Significance level at which the graph is thresholded. Defaults to 0.0.
        dependencies_wrap (bool, optional):
            Whether the dependencies sought in the data are wrapping - i.e., the dependence structure is toroidal in the space. Defaults to False.

    Returns:
        tuple: tuple of the reconstructed string-graph and the value matrix containing coefficients: (graph, val_matrix).
    """

    # Concatenate data according to locality to form the reduced rependence space
    if dependencies_wrap:
        # Concatenate data if the space is toroidal - i.e. neighborhoods wrap.
        concatenated_data = concatenate_timeseries_wrapping(data, rows_inverted=rows_inverted, include_cell_index_column=False)
    else:
        # Concatenate data if the space is not toroidal - i.e. neighborhoods wrap.
        concatenated_data = concatenate_timeseries_nonwrapping(data, rows_inverted=rows_inverted, include_cell_index_column=False)

    # Create TIGRAMITE DataFrame from the concatenated data
    pcmci_df = pp.DataFrame(concatenated_data[:, :9])
    # Initialize dictionary to store all parents
    all_parents = {
        i: {
            "parents": [],
            "val_min": {(j, -1): 0 for j in range(9)},
            "pval_max": {(k, -1): 1 for k in range(9)},
            "iterations": {},
        }
        for i in range(9)
    }
    # Initialize PC_stable_single algorithm
    PC_alg = pc_stable_single.PC_stable_single(pcmci_df, cond_ind_test)
    # Run PC-stable-single algorithm for parent identification on the center grid cell - index 4
    all_parents[4] = PC_alg._run_pc_stable_single(4, pc_alpha=pc_alpha)

    # Convert parent list of center node (4) to a structural causal model (SCM). The SCM is only used to export the results to a TIRGRAMITE graph
    SCM = {}
    for key in all_parents.keys():
        # Initialize empty list for each key in SCM
        SCM[key] = []
        # Extract parents from all_parents[key]["parents"]
        parents_list = [parent[0] for parent in all_parents[key]["parents"]]
        # Iterate over the parents
        for parent in parents_list:
            # Retrieve coefficient value from all_parents[key]["val_min"]
            coefficient = all_parents[key]["val_min"][(parent, -1)]
            # If dependence threshold is set and the coefficient is below the threshold, set coefficient to zero to indicate no link
            if dependence_threshold is not None:
                if abs(coefficient) < dependence_threshold:
                    coefficient = 0
            # Append tuple containing parent index, coefficient, and a stand-in function to comply with TIGRAMITE's expected format
            SCM[key].append(((parent, -1), coefficient, (lambda x: x)))
    # Convert links in SCM to graph
    graph = structural_causal_processes.links_to_graph(SCM)

    # Create a val_matrix containing dependence coefficients for each link in the SCM
    val_matrix = np.zeros(graph.shape)
    for row in range(val_matrix.shape[0]):
        if len(SCM[row]) != 0:
            for dependence in SCM[row]:
                coefficient = dependence[1]
                val_matrix[dependence[0][0], row, 1] = coefficient

    return graph, val_matrix


def CaStLe_PCMCI(
    data: np.ndarray,
    cond_ind_test: CondIndTest,
    pc_alpha: float,
    rows_inverted=True,
    dependence_threshold=0.01,
    dependencies_wrap=False,
) -> tuple:
    """The CaStLe algorithm implemented with PCMCI for the parent-identification phase.

    Args:
        data (np.ndarray): The data of shape (N, N, T) to be given to CaStLe-PCMCI.
        cond_ind_test (CondIndTest): The conditional independence test to be used in the parent-identification phase.
        pc_alpha (float): Significance level in PCMCI.
        rows_inverted (bool, optional): Whether data rows are inverted. Inverted means the row above is (row-1). Defaults to True.
        dependence_threshold (float, optional): Significance level at which the p_matrix from PCMCI is thresholded to get the graph. Defaults to 0.01.
        dependencies_wrap (bool, optional): Whether the dependencies sought in the data are wrapping - i.e., the dependence structure is toroidal in the space. Defaults to False.

    Returns:
        tuple: tuple of the reconstructed string-graph and the value matrix containing coefficients: (graph, val_matrix).
    """
    min_tau = 1
    max_tau = 1

    if dependencies_wrap:
        concatenated_data = concatenate_timeseries_wrapping(data, rows_inverted=rows_inverted, include_cell_index_column=False)
    else:
        concatenated_data = concatenate_timeseries_nonwrapping(data, rows_inverted=rows_inverted, include_cell_index_column=False)
    pcmci_df = pp.DataFrame(concatenated_data[:, :9])

    # Only estimate parents of variable 4
    link_assumptions = {}
    for j in range(9):
        if j in [4]:
            # Directed lagged links
            link_assumptions[j] = {(var, -lag): "-?>" for var in range(9) for lag in range(1, max_tau + 1)}
        else:
            link_assumptions[j] = {}

    pcmci = PCMCI(dataframe=pcmci_df, cond_ind_test=cond_ind_test, verbosity=0)

    results = pcmci.run_pcmci(
        tau_min=min_tau,
        tau_max=max_tau,
        pc_alpha=pc_alpha,
        alpha_level=dependence_threshold,
        link_assumptions=link_assumptions,
    )
    q_matrix = pcmci.get_corrected_pvalues(
        p_matrix=results["p_matrix"],
        tau_min=min_tau,
        tau_max=max_tau,
        fdr_method="fdr_bh",
        link_assumptions=link_assumptions,
    )
    reconstructed_graph = pcmci.get_graph_from_pmatrix(
        p_matrix=q_matrix,
        alpha_level=dependence_threshold,
        tau_min=min_tau,
        tau_max=max_tau,
        link_assumptions=link_assumptions,
    )

    return reconstructed_graph, results["val_matrix"]


def CaStLe_FullCI(
    data: np.ndarray,
    cond_ind_test: CondIndTest,
    alpha_level: float,
    rows_inverted=False,
    dependence_threshold=0.01,
    dependencies_wrap=False,
) -> tuple:
    """The CaStLe algorithm implemented with FullCI for the parent-identification phase.

    Args:
        data (np.ndarray): The data of shape (N, N, T) to be given to CaStLe-PCMCI.
        cond_ind_test (CondIndTest): The conditional independence test to be used in the parent-identification phase.
        alpha_level (float): Significance level at which the p_matrix is thresholded to get graph.
        rows_inverted (bool, optional): Whether data rows are inverted. Inverted means the row above is (row-1). Defaults to False.
        dependence_threshold (float, optional): Significance level at which the p_matrix from PCMCI is thresholded to get the graph. Defaults to 0.01.
        dependencies_wrap (bool, optional): Whether the dependencies sought in the data are wrapping - i.e., the dependence structure is toroidal in the space. Defaults to False.

    Returns:
        tuple: tuple of the reconstructed string-graph and the value matrix containing coefficients: (graph, val_matrix).
    """
    min_tau = 1
    max_tau = 1

    if dependencies_wrap:
        concatenated_data = concatenate_timeseries_wrapping(data, rows_inverted=rows_inverted, include_cell_index_column=False)
    else:
        concatenated_data = concatenate_timeseries_nonwrapping(data, rows_inverted=rows_inverted, include_cell_index_column=False)
    pcmci_df = pp.DataFrame(concatenated_data[:, :9])

    # Only estimate parents of variable 4
    link_assumptions = {}
    for j in range(9):
        if j in [4]:
            # Directed lagged links
            link_assumptions[j] = {(var, -lag): "-?>" for var in range(9) for lag in range(1, max_tau + 1)}
        else:
            link_assumptions[j] = {}

    pcmci = PCMCI(dataframe=pcmci_df, cond_ind_test=cond_ind_test, verbosity=0)

    results = pcmci.run_fullci(
        tau_min=min_tau,
        tau_max=max_tau,
        alpha_level=alpha_level,
        link_assumptions=link_assumptions,
    )
    q_matrix = pcmci.get_corrected_pvalues(
        p_matrix=results["p_matrix"],
        tau_min=min_tau,
        tau_max=max_tau,
        fdr_method="fdr_bh",
        link_assumptions=link_assumptions,
    )
    reconstructed_graph = pcmci.get_graph_from_pmatrix(
        p_matrix=q_matrix,
        alpha_level=dependence_threshold,
        tau_min=min_tau,
        tau_max=max_tau,
        link_assumptions=link_assumptions,
    )

    return reconstructed_graph, results["val_matrix"]


def CaStLe_PC(
    data: np.ndarray,
    cond_ind_test: CondIndTest,
    pc_alpha: float,
    rows_inverted=False,
    dependence_threshold=0.01,
    dependencies_wrap=False,
) -> tuple:
    """The CaStLe algorithm implemented with PC for the parent-identification phase.

    Args:
        data (np.ndarray): The data of shape (N, N, T) to be given to CaStLe-PCMCI.
        cond_ind_test (CondIndTest): The conditional independence test to be used in the parent-identification phase.
        pc_alpha (float): Significance level in PCMCI.
        rows_inverted (bool, optional): Whether data rows are inverted. Inverted means the row above is (row-1). Defaults to False.
        dependence_threshold (float, optional): Significance level at which the p_matrix from PCMCI is thresholded to get the graph. Defaults to 0.01.
        dependencies_wrap (bool, optional): Whether the dependencies sought in the data are wrapping - i.e., the dependence structure is toroidal in the space. Defaults to False.

    Returns:
        tuple: tuple of the reconstructed string-graph and the value matrix containing coefficients: (graph, val_matrix).
    """
    min_tau = 1
    max_tau = 1

    if dependencies_wrap:
        concatenated_data = concatenate_timeseries_wrapping(data, rows_inverted=rows_inverted, include_cell_index_column=False)
    else:
        concatenated_data = concatenate_timeseries_nonwrapping(data, rows_inverted=rows_inverted, include_cell_index_column=False)
    pcmci_df = pp.DataFrame(concatenated_data[:, :9])

    # Only estimate parents of variable 4
    link_assumptions = {}
    for j in range(9):
        if j in [4]:
            # Directed lagged links
            link_assumptions[j] = {(var, -lag): "-?>" for var in range(9) for lag in range(1, max_tau + 1)}
        else:
            link_assumptions[j] = {}

    pcmci = PCMCI(dataframe=pcmci_df, cond_ind_test=cond_ind_test, verbosity=0)

    results = pcmci.run_pcalg(
        tau_min=min_tau,
        tau_max=max_tau,
        pc_alpha=pc_alpha,
        link_assumptions=link_assumptions,
    )
    q_matrix = pcmci.get_corrected_pvalues(
        p_matrix=results["p_matrix"],
        tau_min=min_tau,
        tau_max=max_tau,
        fdr_method="fdr_bh",
        link_assumptions=link_assumptions,
    )
    reconstructed_graph = pcmci.get_graph_from_pmatrix(
        p_matrix=q_matrix,
        alpha_level=dependence_threshold,
        tau_min=min_tau,
        tau_max=max_tau,
        link_assumptions=link_assumptions,
    )

    return reconstructed_graph, results["val_matrix"]


def CaStLe_DYNOTEARS(data: np.ndarray, rows_inverted=False, dependence_threshold=0.01, dependencies_wrap=False) -> tuple:
    """The CaStLe algorithm implemented with DYNOTEARS for the parent-identification phase.

    Args:
        data (np.ndarray): The data of shape (N, N, T) to be given to CaStLe.
        rows_inverted (bool, optional): Whether data rows are inverted. Inverted means the row above is (row-1). Defaults to False.
        dependence_threshold (float, optional): fixed threshold for absolute edge weights. Defaults to 0.01.
        dependencies_wrap (bool, optional): Whether the dependencies sought in the data are wrapping - i.e., the dependence structure is toroidal in the space. Defaults to False.

    Returns:
        tuple: tuple of the reconstructed string-graph and the value matrix containing coefficients: (graph, val_matrix).
    """
    max_tau = 1
    if dependencies_wrap:
        concatenated_data = concatenate_timeseries_wrapping(data, rows_inverted=rows_inverted, include_cell_index_column=False)
    else:
        concatenated_data = concatenate_timeseries_nonwrapping(data, rows_inverted=rows_inverted, include_cell_index_column=False)

    # format data into dataframe
    col_names = ["" + str(i) for i in np.arange(concatenated_data.shape[1])]
    df_castled = pd.DataFrame(data=concatenated_data, columns=col_names)

    ## Prepare link assumptions
    taboo_children = ["0", "1", "2", "3", "5", "6", "7", "8"]
    taboo_edges = []  # (lag, from, to)
    for i in col_names:
        for j in col_names:
            # Ban all links from lag 0, ie only allow lag1 -> lag0
            taboo_edges.append((0, i, j))

    ## fit model
    structure_model_castled = from_pandas_dynamic(df_castled, p=max_tau, w_threshold=dependence_threshold, tabu_edges=taboo_edges, tabu_child_nodes=taboo_children)

    # Convert to graph
    reconstructed_graph, val_matrix = scmg.get_graph_from_structure_model(structure_model_castled)

    return reconstructed_graph, val_matrix


def PC(
    data: np.ndarray,
    cond_ind_test: CondIndTest,
    min_tau: int,
    max_tau: int,
    pc_alpha: float,
    pval_threshold=0.01,
) -> tuple:
    """The PC algorithm for time series data.

    Args:
        data (np.ndarray): The data of shape (N, N, T) to be given to CaStLe-PC.
        cond_ind_test (CondIndTest): The conditional independence test to be used in the parent-identification phase.
        pc_alpha (float): Significance level in PC.
        rows_inverted (bool, optional): Whether data rows are inverted. Inverted means the row above is (row-1). Defaults to True.
        dependence_threshold (float, optional): Significance level at which the p_matrix from PC is thresholded to get the graph. Defaults to 0.01.
        dependencies_wrap (bool, optional): Whether the dependencies sought in the data are wrapping - i.e., the dependence structure is toroidal in the space. Defaults to False.

    Returns:
        tuple: tuple of the reconstructed string-graph and the value matrix containing coefficients: (graph, val_matrix).
    """
    # Reshape data for input to PCMCI
    if len(data.shape) > 3:
        data = data[:, :, :, 0]  # only working with first variable for now

    if len(data.shape) > 2:
        data = data.reshape(data.shape[0] * data.shape[1], data.shape[2])  # reshape to N^2xtime
        data = data.transpose()  # Rows must be temporal
    if data.shape[0] < data.shape[1]:
        warnings.warn("More columns than rows! Either there are more variables than observations, or you need to transpose the data.")

    pcmci_df = pp.DataFrame(data)

    pcmci = PCMCI(dataframe=pcmci_df, cond_ind_test=cond_ind_test, verbosity=0)

    results = pcmci.run_pcalg(
        tau_min=min_tau,
        tau_max=max_tau,
        pc_alpha=pc_alpha,
    )
    q_matrix = pcmci.get_corrected_pvalues(
        p_matrix=results["p_matrix"],
        tau_min=min_tau,
        tau_max=max_tau,
        fdr_method="fdr_bh",
    )
    reconstructed_graph = pcmci.get_graph_from_pmatrix(
        p_matrix=q_matrix,
        alpha_level=pval_threshold,
        tau_min=min_tau,
        tau_max=max_tau,
    )

    return reconstructed_graph, results["val_matrix"]


def PC_stable(
    data: np.ndarray,
    cond_ind_test: CondIndTest,
    min_tau: int,
    max_tau: int,
    pc_alpha: float,
) -> tuple:
    """The PC algorithm for time series data.

    Args:
        data (np.ndarray): The data of shape (N, N, T) to be given to CaStLe-PC.
        cond_ind_test (CondIndTest): The conditional independence test to be used in the parent-identification phase.
        pc_alpha (float): Significance level in PC.
        rows_inverted (bool, optional): Whether data rows are inverted. Inverted means the row above is (row-1). Defaults to True.
        dependence_threshold (float, optional): Significance level at which the p_matrix from PC is thresholded to get the graph. Defaults to 0.01.
        dependencies_wrap (bool, optional): Whether the dependencies sought in the data are wrapping - i.e., the dependence structure is toroidal in the space. Defaults to False.

    Returns:
        tuple: tuple of the reconstructed string-graph and the value matrix containing coefficients: (graph, val_matrix).
    """
    # Reshape data for input to PCMCI
    if len(data.shape) > 3:
        data = data[:, :, :, 0]  # only working with first variable for now

    if len(data.shape) > 2:
        data = data.reshape(data.shape[0] * data.shape[1], data.shape[2])  # reshape to N^2xtime
        data = data.transpose()  # Rows must be temporal
    if data.shape[0] < data.shape[1]:
        warnings.warn("More columns than rows! Either there are more variables than observations, or you need to transpose the data.")

    pcmci_df = pp.DataFrame(data)

    pcmci = PCMCI(dataframe=pcmci_df, cond_ind_test=cond_ind_test, verbosity=0)

    results = pcmci.run_pc_stable(
        tau_min=min_tau,
        tau_max=max_tau,
        pc_alpha=pc_alpha,
    )
    return results
