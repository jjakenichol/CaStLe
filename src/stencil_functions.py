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

import numpy as np
from tigramite import data_processing as pp
from tigramite.independence_tests.independence_tests_base import CondIndTest
from tigramite.toymodels import structural_causal_processes
from tigramite.pcmci import PCMCI
import pc_stable_single
import warnings


def concatenate_timeseries_wrapping(
    data: np.ndarray, rows_inverted: bool, include_cell_index_column=False
) -> np.ndarray:
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


def concatenate_timeseries_nonwrapping(
    data: np.ndarray, rows_inverted: bool, include_cell_index_column=False
) -> np.ndarray:
    """Concatenates time series data in a non-wrapping manner.

    Args:
        data (np.ndarray): The input data of shape (N, N, T).
        rows_inverted (bool): Whether data rows are inverted. Inverted means the row above is (row-1).
        include_cell_index_column (bool, optional): Whether to include a cell index column in the concatenated data. Defaults to False.

    Returns:
        np.ndarray: The concatenated data of shape (M, 9) or (M, 10) if include_cell_index_column is True.
    """
    GRID_SIZE = data.shape[0]

    # create one cell buffer such that all cells evaluated have neighbors on all sides
    ROW_RANGE = range(1, GRID_SIZE - 1)
    # create one cell buffer such that all cells evaluated have neighbors on all sides
    COL_RANGE = range(1, GRID_SIZE - 1)

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


def CaStLe(
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
        concatenated_data = concatenate_timeseries_wrapping(
            data, rows_inverted=rows_inverted, include_cell_index_column=False
        )
    else:
        # Concatenate data if the space is not toroidal - i.e. neighborhoods wrap.
        concatenated_data = concatenate_timeseries_nonwrapping(
            data, rows_inverted=rows_inverted, include_cell_index_column=False
        )

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
        concatenated_data = concatenate_timeseries_wrapping(
            data, rows_inverted=rows_inverted, include_cell_index_column=False
        )
    else:
        concatenated_data = concatenate_timeseries_nonwrapping(
            data, rows_inverted=rows_inverted, include_cell_index_column=False
        )
    pcmci_df = pp.DataFrame(concatenated_data[:, :9])

    # Only estimate parents of variable 4
    link_assumptions = {}
    for j in range(9):
        if j in [4]:
            # Directed lagged links
            link_assumptions[j] = {
                (var, -lag): "-?>" for var in range(9) for lag in range(1, max_tau + 1)
            }
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
        concatenated_data = concatenate_timeseries_wrapping(
            data, rows_inverted=rows_inverted, include_cell_index_column=False
        )
    else:
        concatenated_data = concatenate_timeseries_nonwrapping(
            data, rows_inverted=rows_inverted, include_cell_index_column=False
        )
    pcmci_df = pp.DataFrame(concatenated_data[:, :9])

    # Only estimate parents of variable 4
    link_assumptions = {}
    for j in range(9):
        if j in [4]:
            # Directed lagged links
            link_assumptions[j] = {
                (var, -lag): "-?>" for var in range(9) for lag in range(1, max_tau + 1)
            }
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
        data = data.reshape(
            data.shape[0] * data.shape[1], data.shape[2]
        )  # reshape to N^2xtime
        data = data.transpose()  # Rows must be temporal
    if data.shape[0] < data.shape[1]:
        warnings.warn(
            "More columns than rows! Either there are more variables than observations, or you need to transpose the data."
        )

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
        data = data.reshape(
            data.shape[0] * data.shape[1], data.shape[2]
        )  # reshape to N^2xtime
        data = data.transpose()  # Rows must be temporal
    if data.shape[0] < data.shape[1]:
        warnings.warn(
            "More columns than rows! Either there are more variables than observations, or you need to transpose the data."
        )

    pcmci_df = pp.DataFrame(data)

    pcmci = PCMCI(dataframe=pcmci_df, cond_ind_test=cond_ind_test, verbosity=0)

    results = pcmci.run_pc_stable(
        tau_min=min_tau,
        tau_max=max_tau,
        pc_alpha=pc_alpha,
    )
    return results
