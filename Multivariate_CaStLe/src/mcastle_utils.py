"""
Multivariate CaStLe-Stencil Spatial Graph Mapping, Visualization, and Causal Discovery Utilities

Author: J. Jake Nichol
Email: jefnich@sandia.gov

Abstract:
This script offers a suite of functions for manipulating, mapping, visualizing spatial graphs, and performing causal discovery in multivariate space-time datasets. Designed to work with multivariate spatial models, particularly those on grid-based layouts, it transforms local spatial relationships into comprehensive graph structures for visualization and analysis. The utilities extend to causal discovery, allowing for the inference of causal relationships between variables across a spatial grid over time.

Introduction to Space-Time Causal Graphs:
Space-time causal graphs are used to model complex systems where interactions occur both in space and time. Nodes in these graphs represent entities or variables, while edges represent dependencies or interactions between them. These graphs are crucial for understanding the dynamics of systems such as ecological networks, climate models, and social networks.

Moore Neighborhood:
The Moore neighborhood is a concept used in spatial modeling to define the local neighborhood around a grid cell. It includes the eight surrounding cells in a 3x3 grid. This neighborhood is used to capture local interactions and dependencies between variables.

Stencil and Problem Space:
In space-time modeling, a stencil graph represents local interactions within a Moore neighborhood (3x3 grid) for each variable (species) in the dataset. The stencil captures dependencies between variables in a localized region, which can then be mapped to a full graph representing the entire spatial grid. This approach allows for detailed analysis of local interactions and their propagation across the grid. The functions provided in this script facilitate the conversion between stencil graphs and full graphs, visualization of spatial relationships, and causal discovery to understand the underlying dynamics of the system.

Stencil Graph Data Structure:
The stencil graph is a 3D numpy array with shape (9*N, 9*N, 2), where N is the number of species (variables). Each entry in the stencil graph is either "-->" indicating a dependency or "" indicating no dependency. The stencil graph captures dependencies within a 3x3 Moore neighborhood for each variable. The value matrix, which has the same shape as the stencil graph, contains values associated with the edges in the graph. These values represent the strength or significance of the dependencies.

Algorithmic Details:
The `mv_CaStLe_PC` function applies a causal discovery algorithm on a 4-dimensional dataset representing multiple variables across a 2D spatial grid over time. It projects the data to a local coordinate space by concatenating timeseries from each cell's neighborhood, either with or without wrapping at the grid edges. This projection allows the causal discovery algorithm to operate on a simplified representation of the data, while still capturing the temporal and spatial dependencies. The primary goal of CaStLe is to identify dependencies from neighboring parents to the center node/cell, which serves as a generalized representation of every grid cell in the original space. By focusing on the center node, CaStLe effectively captures the local interactions within the neighborhood. In summary, the `mv_CaStLe_PC` function provides a flexible and powerful framework for causal discovery in space-time data, leveraging the projection to a local coordinate space and the incorporation of link assumptions to produce accurate and interpretable causal graphs.

Dependencies:
- Tigramite (for causal discovery functions) providing conditional independence tests and causal discovery algorithms.
- NumPy, matplotlib, xarray, math, sys, and typing are also required for data manipulation, visualization, and type hinting support.
"""

import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import xarray as xr
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.independence_tests_base import CondIndTest
from typing import Dict, Optional, List, Tuple, Union


def convert_string_assumptions_to_indices(analysis_variables, variable_link_assumptions):
    """
    Converts link assumptions defined in terms of variable names to indices.

    Parameters
    ----------
    analysis_variables : list
        List of variable names.
        Importantly defines the index ordering of the assumptions. The assumptions and var_names passed to CaStLe will
        be the same.
    variable_link_assumptions : dict
        Dictionary specifying assumptions about links between variables using their names.

    Returns
    -------
    dict
        Dictionary specifying assumptions about links between variables using their indices.

    Example
    -------
    >>> analysis_variables = [
    ...     "TMSO201",
    ...     "TMH2SO401",
    ...     "BURDENSO401",
    ...     "AODSO401",
    ...     "FSDS01",
    ... ]
    >>> variable_link_assumptions = {
    ...     "TMSO201": {("TMSO201", -1): "-?>"},
    ...     "TMH2SO401": {("TMSO201", -1): "-?>", ("TMH2SO401", -1): "-?>"},
    ...     "BURDENSO401": {("TMH2SO401", -1): "-?>", ("BURDENSO401", -1): "-?>"},
    ...     "AODSO401": {("BURDENSO401", -1): "-?>", ("AODSO401", -1): "-?>"},
    ...     "FSDS01": {("AODSO401", -1): "-?>", ("FSDS01", -1): "-?>"},
    ... }
    >>> convert_link_assumptions_to_indices(analysis_variables, variable_link_assumptions)
    {
        0: {(0, -1): "-?>"},
        1: {(0, -1): "-?>", (1, -1): "-?>"},
        2: {(1, -1): "-?>", (2, -1): "-?>"},
        3: {(2, -1): "-?>", (3, -1): "-?>"},
        4: {(3, -1): "-?>", (4, -1): "-?>"},
    }
    """
    # Create a mapping from variable names to their indices
    variable_name_to_index = {name: idx for idx, name in enumerate(analysis_variables)}

    # Initialize the dictionary to hold the converted link assumptions
    link_assumptions_indices = {}

    # Iterate over the variable_link_assumptions to convert names to indices
    for child_name, parent_links in variable_link_assumptions.items():
        child_idx = variable_name_to_index[child_name]
        link_assumptions_indices[child_idx] = {}
        for (parent_name, lag), link_type in parent_links.items():
            parent_idx = variable_name_to_index[parent_name]
            link_assumptions_indices[child_idx][(parent_idx, lag)] = link_type

    return link_assumptions_indices


def pretty_print_link_assumptions(link_assumptions_indices):
    """
    Pretty prints the link assumptions dictionary with variable indices.

    Parameters
    ----------
    analysis_variables : list
        List of variable names.
    link_assumptions_indices : dict
        Dictionary specifying assumptions about links between variables using their indices.
    """
    print("variable_link_assumptions = {")
    for child_idx, parent_links in link_assumptions_indices.items():
        print(f"    {child_idx}: {{", end="")
        parent_links_str = ", ".join([f'({parent_idx}, {lag}): "{link_type}"' for (parent_idx, lag), link_type in parent_links.items()])
        print(f"{parent_links_str}}},")
    print("}")


def print_significant_links(
    val_matrix,
    p_matrix=None,
    q_matrix=None,
    graph=None,
    alpha_level=0.05,
    print_empty=True,
    var_names=None,
    include_noncenters=False,
) -> None:
    """Prints significant links.

    Parameters
    ----------
    alpha_level : float, optional (default: 0.05)
        Significance level.
    p_matrix : array-like, optional
        Must be of shape (N, N, tau_max + 1).
    q_matrix : array-like, optional
        Must be of shape (N, N, tau_max + 1).
    val_matrix : array-like
        Must be of shape (N, N, tau_max + 1).
    graph : array-like, optional
        Must be of shape (N, N, tau_max + 1).
    print_empty : bool, optional (default: True)
        If False, do not print variables with no significant links.
    var_names : list of str, optional (default: None)
        List of base variable names. If provided, the first 9 variables will be named
        "X.0", "X.1", ..., "X.8" for var_name X. The next 9 variables will be similarly
        "var_name.n" for n in 9.
    """
    N = val_matrix.shape[0]
    assert N % 9 == 0, "The number of variables (N) must be divisible by 9."
    assert (p_matrix is not None) ^ (q_matrix is not None), "Either p_matrix or q_matrix must be provided, but not both."

    if var_names is None:
        var_names = list(range(N))
    else:
        var_names = [f"{var_names[i // 9]}.{i % 9}" for i in range(N)]

    if graph is not None:
        sig_links = (graph != "") * (graph != "<--")
    else:
        if p_matrix is not None:
            sig_links = p_matrix <= alpha_level
        else:
            sig_links = q_matrix <= alpha_level

    matrix_type = "p-value" if p_matrix is not None else "q-value"
    matrix = p_matrix if p_matrix is not None else q_matrix

    print("## Significant links at alpha = %s using %s matrix:" % (alpha_level, matrix_type))
    for j in range(N):
        if not include_noncenters and (j - 4) % 9 != 0:
            continue
        links = {(p[0], -p[1]): np.abs(val_matrix[p[0], j, abs(p[1])]) for p in zip(*np.where(sig_links[:, j, :]))}
        # Sort by variable index
        sorted_links = sorted(links)
        n_links = len(links)

        if n_links == 0 and not print_empty:
            continue

        string = "\n    Variable %s has %d link(s):" % (var_names[j], n_links)
        for p in sorted_links:
            string += "\n        (%s % d): %s = %.5f" % (
                var_names[p[0]],
                p[1],
                matrix_type,
                matrix[p[0], j, abs(p[1])],
            )
            string += " | val = % .3f" % (val_matrix[p[0], j, abs(p[1])])
            if graph is not None:
                if p[1] == 0 and graph[j, p[0], 0] == "o-o":
                    string += " | unoriented link"
                if graph[p[0], j, abs(p[1])] == "x-x":
                    string += " | unclear orientation due to conflict"
        print(string)


def get_last_n_indices(array1: list, array2: list) -> list:
    """Given two arrays, use the length, N, of the first to find the last N indices of the second.

    Args:
        array1 (list): First array for length N
        array2 (list): Second array for finding last N indices

    Returns:
        list: List of last indices in array2
    """
    N = len(array2)
    start_index = len(array1) - N
    return list(range(start_index, len(array1))) if start_index >= 0 else list(range(len(array1)))


def get_mixed_var_graph(given_graph, given_val_matrix, var_idx):
    # Initialize data structures
    return_graph = np.full((9, 9, 2), fill_value="")
    val_matrix = np.full((9, 9, 2), fill_value=0.0)
    # Gather center node
    return_graph[4, 4, :] = given_graph[4, 4, :]
    # Gather neighbors
    for i in range(9):
        if i != 4:
            # Compute mapping to alternate variable position
            if i < 4:
                j = i + var_idx * 9
            else:
                j = i + var_idx * 9 - 1
            return_graph[i, i, :] = given_graph[j, j, :]
            val_matrix[i, i, :] = given_val_matrix[j, j, :]
    # return_graph[9, 9, :] = given_graph[-1, -1, :]
    # val_matrix[9, 9, :] = given_val_matrix[-1, -1, :]
    return return_graph, val_matrix


def char_range(c1=None, c2=None, num_characters=None):
    """Generates the characters from `c1` to `c2`, inclusive."""
    if num_characters:
        for c in range(ord("a"), ord("a") + num_characters):
            yield chr(c)
    elif c1 != None and c2 != None:
        for c in range(ord(c1), ord(c2) + 1):
            yield chr(c)
    else:
        print(c1, c2, num_characters)
        sys.exit(1)


def get_stencil_graph_from_coefficients(local_coefficients: np.ndarray, verbose=False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a stencil graph and a stencil value matrix from local coefficients.

    This function processes a 3D array of local coefficients representing the relationships between variables in a spatial model. It constructs two matrices: a stencil graph that visually represents these relationships and a stencil value matrix that quantifies the strength of these relationships.

    Parameters
    ----------
    local_coefficients : np.ndarray
        A 3D array where each element represents the coefficient between a pair of variables at a specific spatial relation. The dimensions should correspond to [variable, row, column], with 'row' and 'column' indicating the spatial relation.
    verbose : bool, optional
        If set to True, the function prints detailed information about the graph construction process. Defaults to False.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing two elements:
        1. The stencil graph (np.ndarray): A 3D array where each element is a string indicating the presence ("-->" for non-zero coefficients) and direction of the relationship between variables at specific spatial relations.
        2. The stencil value matrix (np.ndarray): A 3D array parallel to the stencil graph, but containing the numerical values of the coefficients instead of visual indicators. Non-zero elements indicate the strength of the relationship.

    Notes
    -----
    - The function iterates over the variables and their spatial relations, computes the positions in the stencil graph and value matrix, retrieves the necessary values from the local coefficients, and assigns them to the corresponding positions.
    - The first dimension of both returned arrays represents the source variable and spatial relation, the second dimension represents the target variable and spatial relation, and the third dimension is used for separating the graph representation and value matrix.
    """
    num_variables = local_coefficients.shape[0]
    stencil_graph = np.full(shape=(9 * num_variables, 9 * num_variables, 2), fill_value="", dtype="<U3")
    stencil_val_matrix = np.full(
        shape=(9 * num_variables, 9 * num_variables, 2),
        fill_value=0.0,
    )

    for child in range(local_coefficients.shape[0]):
        for row in range(local_coefficients.shape[1]):
            for col in range(local_coefficients.shape[2]):
                for parent in range(num_variables):
                    coefficient = local_coefficients[child, row, col][parent]
                    if coefficient != 0:
                        par_pos = row * 3 + col + 9 * parent
                        child_pos = child * 9 + 4
                        stencil_graph[par_pos, child_pos, 1] = "-->"
                        stencil_val_matrix[par_pos, child_pos, 1] = coefficient
                        if verbose:
                            print(
                                "child={}, row={}, col={}, parent={}, coef={}, par_pos={}, child_pos={}".format(
                                    child,
                                    row,
                                    col,
                                    parent,
                                    local_coefficients[child, row, col][parent],
                                    par_pos,
                                    child_pos,
                                )
                            )
    return stencil_graph, stencil_val_matrix


def create_angle_dict(num_species: int) -> Dict[Tuple[int, int, int], int]:
    """
    Create an angle dictionary for multiple species, mapping positions to movement angles.

    This function generates a dictionary where:
    - Keys are tuples (position, center, 1)
    - Values are angles in degrees representing direction of movement from a position toward a center
    - Each species has 9 positions (0-8) in a 3x3 grid
    - Centers are at positions 4, 13, 22, etc. (4+9*i for i in range(num_species))
    - All species have connections to all centers

    Parameters:
        num_species (int): Number of species to generate the dictionary for

    Returns:
        Dict[Tuple[int, int, int], int]: Dictionary mapping position tuples to angle values
    """
    angle_dict = {}

    # Angle mapping in a 3x3 grid (from position to center)
    angle_mapping = {
        0: 315,  # Top-left to center
        1: 270,  # Top-center to center
        2: 225,  # Top-right to center
        3: 0,  # Middle-left to center
        4: -1,  # Center (no movement)
        5: 180,  # Middle-right to center
        6: 45,  # Bottom-left to center
        7: 90,  # Bottom-center to center
        8: 135,  # Bottom-right to center
    }

    # Generate centers for all species
    centers = [4 + 9 * i for i in range(num_species)]

    # For each species
    for species in range(num_species):
        species_offset = species * 9  # Each species adds 9 positions

        # For each center (all species have connections to all centers)
        for center in centers:
            for pos in range(9):
                pos_idx = species_offset + pos
                angle_dict[(pos_idx, center, 1)] = angle_mapping[pos]

    return angle_dict


def get_position_description(key: Tuple[int, int, int]) -> str:
    """
    Get a description of the position based on the key.

    The function maps the position to a description of the movement
    toward a center position in a 3x3 grid.

    Parameters:
        key (Tuple[int, int, int]): The key representing (position, center, 1)

    Returns:
        str: A description of the position relative to center
    """
    position_descriptions = {
        0: "Top-left to center",
        1: "Top-center to center",
        2: "Top-right to center",
        3: "Middle-left to center",
        4: "Center (no movement)",
        5: "Middle-right to center",
        6: "Bottom-left to center",
        7: "Bottom-center to center",
        8: "Bottom-right to center",
    }
    base_index = key[0] % 9
    return position_descriptions.get(base_index, "Unknown position")


def pretty_print_angle_dict(angle_dict: Dict[Tuple[int, int, int], int]) -> None:
    """
    Pretty-print the angle dictionary in a structured format.

    The function prints each entry in the angle dictionary with its key, value,
    and a description of the position.

    Parameters:
        angle_dict (Dict[Tuple[int, int, int], int]): The angle dictionary to pretty-print

    Returns:
        None
    """
    for key, value in angle_dict.items():
        print(f" {key}: {value}, # {get_position_description(key)}")


def get_vectors(stencil_graph: np.ndarray, stencil_val_matrix: np.ndarray) -> list:
    """
    Extracts vectors representing dependencies from a multi-species stencil graph with their corresponding values.

    Identifies dependencies from the stencil graph, retrieves the corresponding values,
    and maps them to angle-vectors while excluding any center-to-center dependencies across all species.

    Args:
        stencil_graph (np.ndarray): A 3D numpy array representing the graph structure.
                                    The shape is (9*N, 9*N, 2) where N is the number of species.
                                    Each entry is either "-->" indicating a dependency or "" indicating no dependency.
        stencil_val_matrix (np.ndarray): A 3D numpy array of the same shape as stencil_graph,
                                         containing values associated with the edges in the graph.

    Returns:
        list: A list of tuples where each tuple contains a value from stencil_val_matrix and its corresponding angle.
              The angle represents the direction of the dependency in the Moore neighborhood.
    """
    # Assertions to ensure correct input shapes
    assert stencil_graph.ndim == 3, "stencil_graph must be a 3D numpy array"
    assert stencil_graph.shape[0] % 9 == 0, "The first dimension of stencil_graph must be a multiple of 9"
    assert stencil_graph.shape == stencil_val_matrix.shape, "stencil_graph and stencil_val_matrix must have the same shape"

    # Determine the number of species
    num_species = stencil_graph.shape[0] // 9

    dependence_dict = {}
    for i in range(stencil_graph.shape[0]):
        for j in range(stencil_graph.shape[1]):
            for k in range(stencil_graph.shape[2]):
                if stencil_graph[i, j, k] != "":
                    dependence_dict[i, j, k] = stencil_val_matrix[i, j, k]

    angle_dict = create_angle_dict(num_species=num_species)

    # Generate list of centers (one for each species)
    centers = [4 + 9 * i for i in range(num_species)]

    # Create vectors list, excluding center-to-center dependencies
    vectors = [
        (dependence_dict[dependence], angle_dict[dependence])
        for dependence in dependence_dict.keys()
        if not (dependence[0] in centers and dependence[1] in centers and dependence[2] == 1)
    ]

    return vectors


def combine_angles(vectors: list) -> float:
    """
    Combines multiple vectors into a single resultant angle using their values and angles.

    This function takes a list of vectors, each represented by a value and an angle,
    and computes the resultant angle by summing the x and y components of the vectors.
    The resultant angle is then converted back to degrees.

    Args:
        vectors (list): A list of tuples where each tuple contains a value and an angle in degrees.
            The value represents the strength of the dependency, and the angle represents its direction.
            Format: [(value1, angle1), (value2, angle2), ...]

    Returns:
        float: The resultant angle in degrees (0-360), representing the combined
            direction of all input vectors.

    Note:
        When vectors perfectly cancel out (resulting in a near-zero magnitude),
        the function returns the angle of the vector with the largest absolute magnitude.
        This ensures that even in cases of mutual cancellation, the direction with
        the strongest individual influence is preserved.

    Example:
        >>> combine_angles([(0.5, 45), (0.5, 45), (1.0, 90)])
        67.5
        >>> combine_angles([(1.0, 0), (-1.0, 180)])  # Perfect cancellation
        0.0  # Returns the angle of the first vector which has magnitude 1.0
    """
    if not vectors:
        # raise ValueError("The list of vectors cannot be empty.")
        print("The list of vectors cannot be empty. Returning np.nan")
        return np.nan

    angles_radians = np.radians([vector[1] for vector in vectors])
    coefficients = np.array([vector[0] for vector in vectors])

    xs = coefficients * np.cos(angles_radians)
    ys = coefficients * np.sin(angles_radians)

    x_sum = np.sum(xs)
    y_sum = np.sum(ys)

    # Handle the edge case where vectors cancel out (near-zero resultant)
    magnitude = np.sqrt(x_sum**2 + y_sum**2)
    if magnitude < 1e-10:  # Small threshold to account for floating-point precision
        # Return angle of the vector with largest absolute magnitude
        max_vector_idx = np.argmax(np.abs(coefficients))
        return vectors[max_vector_idx][1]

    angle_radians = np.arctan2(y_sum, x_sum)
    angle_degrees = np.degrees(angle_radians)

    if angle_degrees < 0:
        angle_degrees += 360
    return angle_degrees


def combine_angles_nonnegative(vectors: list) -> float:
    """
    Combines multiple vectors into a single resultant angle using non-negative magnitudes.

    This function performs vector addition in 2D space by decomposing each vector into
    x and y components, summing these components, and computing the resultant angle.
    All input magnitudes are converted to non-negative values using absolute value,
    preserving the original angle directions.

    Parameters
    ----------
    vectors : list of tuple
        A list of (magnitude, angle) tuples where:
        - magnitude (float): The strength/weight of the vector. Negative values are
          converted to positive using abs(). Zero values are allowed.
        - angle (float): The direction in degrees (0-360). Angles outside this range
          are valid and will be interpreted correctly.

    Returns
    -------
    float
        The resultant angle in degrees (0-360) representing the combined direction
        of all input vectors. Returns np.nan if the input list is empty.

    Notes
    -----
    - When vectors cancel out (resultant magnitude < 1e-10), returns the angle of
      the vector with the largest absolute magnitude
    - The function uses absolute values of magnitudes, so (-2, 45°) is treated
      the same as (2, 45°)
    - Angles are normalized to 0-360° range for the output

    Examples
    --------
    >>> # Two equal vectors at 45° combine to point at 45°
    >>> combine_angles_nonnegative([(1.0, 45), (1.0, 45)])
    45.0

    >>> # Vectors at right angles
    >>> combine_angles_nonnegative([(3.0, 0), (4.0, 90)])
    53.13010235415598  # arctan(4/3) in degrees

    >>> # Negative magnitude is treated as positive
    >>> combine_angles_nonnegative([(-2.0, 30), (2.0, 30)])
    30.0

    >>> # Perfect cancellation returns angle of strongest vector
    >>> combine_angles_nonnegative([(2.0, 0), (2.0, 180)])
    0.0  # or 180.0, depending on which appears first

    >>> # Empty list returns nan
    >>> combine_angles_nonnegative([])
    nan

    >>> # Mixed positive and negative magnitudes
    >>> combine_angles_nonnegative([(-1.5, 0), (1.0, 90), (-0.5, 180)])
    45.0  # Resultant of (1.5, 0°), (1.0, 90°), (0.5, 180°)
    """
    if not vectors:
        print("The list of vectors cannot be empty. Returning np.nan")
        return np.nan
    angles_radians = np.radians([vector[1] for vector in vectors])
    coefficients = np.abs(np.array([vector[0] for vector in vectors]))
    xs = coefficients * np.cos(angles_radians)
    ys = coefficients * np.sin(angles_radians)

    x_sum = np.sum(xs)
    y_sum = np.sum(ys)

    # Handle the edge case where vectors cancel out (near-zero resultant)
    magnitude = np.sqrt(x_sum**2 + y_sum**2)
    if magnitude < 1e-10:  # Small threshold to account for floating-point precision
        # Return angle of the vector with largest absolute magnitude
        max_vector_idx = np.argmax(np.abs(coefficients))
        return vectors[max_vector_idx][1]

    angle_radians = np.arctan2(y_sum, x_sum)
    angle_degrees = np.degrees(angle_radians)

    if angle_degrees < 0:
        angle_degrees += 360
    return angle_degrees


def angle_average(angles, degrees=True):
    """
    Calculate the circular mean of a set of angles.

    Parameters:
    angles (array-like): A list or array of angles
    degrees (bool): If True, angles are in degrees; if False, in radians

    Returns:
    float: The circular mean angle in the same units as input
    """
    if degrees:
        # Convert to radians for calculation
        angles = np.radians(angles)

    # Convert angles to vectors on the unit circle
    x = np.mean(np.cos(angles))
    y = np.mean(np.sin(angles))

    # Convert back to angle
    mean_angle = np.arctan2(y, x)

    # Ensure the result is in [0, 2π)
    if mean_angle < 0:
        mean_angle += 2 * np.pi

    if degrees:
        # Convert back to degrees
        mean_angle = np.degrees(mean_angle)

    return mean_angle


def get_angle_from_stencil(stencil_graph: np.ndarray, val_matrix: np.ndarray):
    """
    Computes the resultant angle from the stencil graph and value matrix.

    This function extracts vectors representing dependencies from the stencil graph and their corresponding values from the value matrix.
    It then combines these vectors to compute the resultant angle, representing the overall direction of dependencies in the stencil graph.

    Args:
        stencil_graph (np.ndarray): A 3D numpy array representing the graph structure.
                                    The shape is (9*N, 9*N, 2) where N is the number of species.
                                    Each entry is either "-->" indicating a dependency or "" indicating no dependency.
        val_matrix (np.ndarray): A 3D numpy array of the same shape as stencil_graph, containing values associated with the edges in the graph.

    Returns:
        float: The resultant angle in degrees, representing the combined direction of all dependencies in the stencil graph.
    """
    vectors = get_vectors(stencil_graph, val_matrix)
    return combine_angles(vectors)


def get_angle_from_stencil_nonnegative(stencil_graph: np.ndarray, val_matrix: np.ndarray):
    """
    Computes the resultant angle from the stencil graph and value matrix using the nonnegative combine angles function.

    This function extracts vectors representing dependencies from the stencil graph and their corresponding values from the value matrix.
    It then combines these vectors to compute the resultant angle, representing the overall direction of dependencies in the stencil graph.

    Args:
        stencil_graph (np.ndarray): A 3D numpy array representing the graph structure.
                                    The shape is (9*N, 9*N, 2) where N is the number of species.
                                    Each entry is either "-->" indicating a dependency or "" indicating no dependency.
        val_matrix (np.ndarray): A 3D numpy array of the same shape as stencil_graph, containing values associated with the edges in the graph.

    Returns:
        float: The resultant angle in degrees, representing the combined direction of all dependencies in the stencil graph.
    """
    vectors = get_vectors(stencil_graph, val_matrix)
    return combine_angles_nonnegative(vectors)


def compute_angle(alpha: float, beta: float) -> float:
    """
    Computes the angle in degrees for a vector given its x-component (alpha) and y-component (beta).

    This function computes the angle in degrees for a vector defined by its x-component (alpha) and y-component (beta).
    The angle is adjusted to be within the range [0, 360) degrees.

    Args:
        alpha (float): The x-component of the vector.
        beta (float): The y-component of the vector.

    Returns:
        float: The angle in degrees, representing the direction of the vector (alpha, beta).
    """
    degree = math.degrees(math.atan2(beta, alpha))
    if degree < 0:
        degree += 360
    return degree


def angle_difference(angle1: float, angle2: float) -> float:
    """
    Computes the difference between two angles.

    This function calculates the smallest difference between two angles, ensuring the result is always between 0 and 180 degrees.

    Args:
        angle1 (float): The first angle in degrees.
        angle2 (float): The second angle in degrees.

    Returns:
        float: The difference between the two angles in degrees.
               The result is always between 0 and 180 degrees.
    """
    # Difference formula from https://gamedev.stackexchange.com/questions/4467/comparing-angles-and-working-out-the-difference
    return 180 - abs(abs(angle1 - angle2) - 180)


def create_random_stencil_graph(num_links: int, num_variables: int, random_seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a random stencil graph and value matrix with a specified number of links.

    This function initializes a stencil graph and value matrix, then randomly creates a specified number of links that terminate at the center nodes of each species. The center nodes are defined as indices 4 + 9 * i for each species i.

    Args:
        num_links (int): The number of links to create in the stencil graph.
        num_variables (int): The number of species/variables.
        random_seed (int, optional): A seed for the random number generator to ensure reproducibility. Defaults to None.

    Returns:
        tuple: A tuple containing the stencil graph and value matrix.
               - stencil_graph (np.ndarray): A 3D numpy array of shape (9*num_variables, 9*num_variables, 2) representing the graph structure.
               - stencil_val_matrix (np.ndarray): A 3D numpy array of shape (9*num_variables, 9*num_variables, 2) containing values associated with the edges in the graph.
    """
    # Calculate the size of the arrays based on the number of variables
    size = 9 * num_variables

    # Initialize stencil_graph as an array of empty strings
    stencil_graph = np.full((size, size, 2), "", dtype=object)

    # Initialize stencil_val_matrix as an array of zeros
    stencil_val_matrix = np.zeros((size, size, 2))

    # Set a few random entries to "-->" and corresponding values in stencil_val_matrix
    if random_seed:
        np.random.seed(random_seed)  # For reproducibility

    # Define centers for each species
    centers = [4 + 9 * i for i in range(num_variables)]

    # Create the specified number of valid links
    created_links = 0
    while created_links < num_links:
        i = np.random.randint(0, size)
        j = np.random.choice(centers)
        if stencil_graph[i, j, 1] == "":
            stencil_graph[i, j, 1] = "-->"
            stencil_val_matrix[i, j, 1] = np.random.rand()
            created_links += 1
    return stencil_graph, stencil_val_matrix


def create_custom_stencil_graph(parent_indices: List[List[Union[int, Tuple[int, float]]]], verbose: bool = False) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Creates a custom stencil graph and value matrix based on the provided parent node indices and coefficients.

    This function initializes a stencil graph and value matrix, then sets the specified parent nodes and coefficients for each variable's child node.

    Args:
        parent_indices (List[List[Union[int, Tuple[int, float]]]]): A nested list of parent node indices and optionally coefficients for each variable's child node.
            The first list contains parents of the first variable's child (always 4),
            the second list contains parents of the second variable's child (always 13), and so on.
        verbose (bool, optional): If True, print detailed information during execution. Defaults to False.

    Returns:
        Union[Tuple[np.ndarray, np.ndarray], np.ndarray]: A tuple containing the stencil graph and optionally the value matrix if coefficients are provided.
            - stencil_graph (np.ndarray): A 3D numpy array of shape (9*num_variables, 9*num_variables, 2) representing the graph structure.
            - stencil_val_matrix (np.ndarray, optional): A 3D numpy array of shape (9*num_variables, 9*num_variables, 2) containing values associated with the edges in the graph.

    Raises:
        ValueError: If the parent_indices list is not well-formed or if not all parents have coefficients when any parent has a coefficient.

    Examples:
        # Example usage with coefficients
        parent_indices = [
            [(0, 0.1), (1, 0.2), (2, 0.3)],  # Parents of the first variable's child (index 4) with coefficients
            [(3, 0.4), (4, 0.5), (5, 0.6)],  # Parents of the second variable's child (index 13) with coefficients
            [(6, 0.7), (7, 0.8), (8, 0.9)]   # Parents of the third variable's child (index 22) with coefficients
        ]
        random_seed = 42
        stencil_graph, stencil_val_matrix = create_custom_stencil_graph(parent_indices, random_seed)
        print("Stencil Graph:\n", stencil_graph)
        print("Stencil Value Matrix:\n", stencil_val_matrix)

        # Example usage without coefficients
        parent_indices_no_coeff = [
            [0, 1, 2],  # Parents of the first variable's child (index 4) without coefficients
            [3, 4, 5],  # Parents of the second variable's child (index 13) without coefficients
            [6, 7, 8]   # Parents of the third variable's child (index 22) without coefficients
        ]
        stencil_graph_no_coeff = create_custom_stencil_graph(parent_indices_no_coeff, random_seed)
        print("Stencil Graph without coefficients:\n", stencil_graph_no_coeff)
    """
    # Infer the number of variables from the length of parent_indices
    num_variables = len(parent_indices)

    # Validate input
    if not all(isinstance(sublist, list) for sublist in parent_indices):
        raise ValueError("parent_indices must be a list of lists.")
    if any(not all(isinstance(item, (int, tuple)) for item in sublist) for sublist in parent_indices):
        raise ValueError("Each sublist in parent_indices must contain integers or tuples of (int, float).")
    if any(isinstance(item, tuple) for sublist in parent_indices for item in sublist):
        if not all(isinstance(item, tuple) for sublist in parent_indices for item in sublist):
            raise ValueError("All parents must have coefficients if any parent has a coefficient.")

    # Calculate the size of the arrays based on the number of variables
    size = 9 * num_variables

    # Initialize stencil_graph as an array of empty strings
    stencil_graph = np.full((size, size, 2), "", dtype=object)

    # Check if coefficients are provided
    coefficients_provided = any(isinstance(item, tuple) for sublist in parent_indices for item in sublist)

    # Initialize stencil_val_matrix as an array of zeros if coefficients are provided
    if coefficients_provided:
        stencil_val_matrix = np.zeros((size, size, 2))
    else:
        stencil_val_matrix = None

    # Define centers for each species
    centers = [4 + 9 * i for i in range(num_variables)]

    # Create the specified parent-child relationships
    for var_index, parents in enumerate(parent_indices):
        child_index = centers[var_index]
        for parent in parents:
            if isinstance(parent, tuple):
                parent_index, coefficient = parent
            else:
                parent_index = parent
                coefficient = None

            parent_index += 9 * var_index
            if stencil_graph[parent_index, child_index, 1] == "":
                stencil_graph[parent_index, child_index, 1] = "-->"
                if coefficients_provided:
                    if coefficient is not None:
                        stencil_val_matrix[parent_index, child_index, 1] = coefficient
                    else:
                        raise ValueError("All parents must have coefficients if any parent has a coefficient.")

    if verbose:
        print("Stencil Graph:\n", stencil_graph)
        if coefficients_provided:
            print("Stencil Value Matrix:\n", stencil_val_matrix)

    if coefficients_provided:
        return stencil_graph, stencil_val_matrix
    else:
        return stencil_graph


def generate_centers(num_species):
    """
    Generator function to yield the center indices for each species.

    Args:
        num_species (int): The number of species.

    Yields:
        int: The center index for each species.
    """
    for i in range(num_species):
        yield 4 + 9 * i


def fisher_z_transform(r: float) -> float:
    """
    Applies Fisher's z-transformation to a correlation coefficient.

    Fisher's z-transformation converts a correlation coefficient into a z-score, which can be used for statistical analysis and combining correlation coefficients.

    Args:
        r (float): The correlation coefficient to transform. Must be between -1 and 1.

    Returns:
        float: The z-score corresponding to the input correlation coefficient.
    """
    return 0.5 * np.log((1 + r) / (1 - r))


def inverse_fisher_z_transform(z: float) -> float:
    """
    Applies the inverse of Fisher's z-transformation to a z-score.

    This function converts a z-score back into a correlation coefficient.

    Args:
        z (float): The z-score to transform.

    Returns:
        float: The correlation coefficient corresponding to the input z-score. The result will be between -1 and 1.
    """
    return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)


def construct_reaction_graph(stencil_graph: np.ndarray, stencil_val_matrix: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Construct a reaction graph from the given stencil_graph and stencil_val_matrix.

    This function constructs a reaction graph by analyzing the provided `stencil_graph` and optionally combining values from the `stencil_val_matrix`.
        The `stencil_graph` is a 3D numpy array representing the graph structure, where each set of 9 slices corresponds to a species.
        The function identifies links between species and populates the reaction graph accordingly.

    If `stencil_val_matrix` is provided, it contains values associated with the edges in the graph.
        The function combines these values using Fisher's z-transformation to stabilize the variance of correlation coefficients.
        For each pair of species, the function extracts the correlation coefficients from the `stencil_val_matrix`, applies Fisher's z-transformation to
        these coefficients, averages the transformed values, and then converts the mean of the transformed values back to a correlation coefficient using
        the inverse Fisher z-transformation. This combined value is stored in the `reaction_val_matrix`.

    Args:
        stencil_graph (np.ndarray): A 3D numpy array representing the graph structure.
        stencil_val_matrix (np.ndarray, optional): A 3D numpy array containing values associated with the edges in the graph.

    Returns:
        reaction_graph (np.ndarray): A 3D numpy array of shape (N, N, 2) indicating how species relate to one another.
        reaction_val_matrix (np.ndarray, optional): A 3D numpy array of shape (N, N, 2) containing combined values for the relationships.
    """
    # Assertions to ensure correct input shapes
    assert stencil_graph.ndim == 3, "stencil_graph must be a 3D numpy array"
    assert stencil_graph.shape[0] % 9 == 0, "The first dimension of stencil_graph must be a multiple of 9"
    if stencil_val_matrix is not None:
        assert stencil_graph.shape == stencil_val_matrix.shape, "stencil_graph and stencil_val_matrix must have the same shape"

    # Determine the number of species
    num_species = stencil_graph.shape[0] // 9

    # Initialize the reaction graph
    reaction_graph = np.full((num_species, num_species, 2), "", dtype=object)
    if stencil_val_matrix is not None:
        reaction_val_matrix = np.zeros((num_species, num_species, 2))

    # Iterate over each combination of species
    for source_species in range(num_species):
        for target_species in range(num_species):
            source_start = source_species * 9
            target_center = target_species * 9 + 4  # The center of each neighborhood is the 4th element in the set of 9

            # Check for links between the source and target species
            if "-->" in stencil_graph[source_start : source_start + 9, target_center, 1]:
                reaction_graph[source_species, target_species, 1] = "-->"

                if stencil_val_matrix is not None:
                    # Combine values from the val_matrix using Fisher's z-transformation
                    values = stencil_val_matrix[source_start : source_start + 9, target_center, 1]
                    non_zero_values = values[values != 0]
                    if len(non_zero_values) > 0:
                        z_values = fisher_z_transform(non_zero_values)
                        mean_z = np.mean(z_values)
                        combined_value = inverse_fisher_z_transform(mean_z)
                        reaction_val_matrix[source_species, target_species, 1] = combined_value

    if stencil_val_matrix is not None:
        return reaction_graph, reaction_val_matrix
    else:
        return reaction_graph


def summarize_stencil(stencil_graph, stencil_val_matrix=None):
    """
    Summarize the stencil_graph and stencil_val_matrix into a simplified graph and value matrix.

    Args:
        stencil_graph (np.ndarray): A 3D numpy array representing the graph structure.
        stencil_val_matrix (np.ndarray, optional): A 3D numpy array containing values associated with the edges in the graph.

    Returns:
        summary_graph (np.ndarray): A 3D numpy array of shape (9, 9, 2) indicating simplified directional dependencies.
        summary_val_matrix (np.ndarray, optional): A 3D numpy array of shape (9, 9, 2) containing combined values for the simplified dependencies.
    """
    # Assertions to ensure correct input shapes
    assert stencil_graph.ndim == 3, "stencil_graph must be a 3D numpy array"
    assert stencil_graph.shape[0] % 9 == 0, "The first dimension of stencil_graph must be a multiple of 9"
    if stencil_val_matrix is not None:
        assert stencil_graph.shape == stencil_val_matrix.shape, "stencil_graph and stencil_val_matrix must have the same shape"

    # Determine the number of species
    num_species = stencil_graph.shape[0] // 9

    # Initialize the simplified graph
    summary_graph = np.full((9, 9, 2), "", dtype=object)
    if stencil_val_matrix is not None:
        summary_val_matrix = np.zeros((9, 9, 2))

    # Iterate over each position in the Moore neighborhood
    for source_pos in range(9):
        for target_pos in range(9):
            values = np.empty(0)
            for source_species in range(num_species):
                for target_species in range(num_species):
                    source_index = source_species * 9 + source_pos
                    target_index = target_species * 9 + target_pos

                    # Check for links between the source and target positions
                    if "-->" in stencil_graph[source_index, target_index, 1]:
                        summary_graph[source_pos, target_pos, 1] = "-->"

                        if stencil_val_matrix is not None:
                            # Combine values from the val_matrix using Fisher's z-transformation
                            values = np.append(values, stencil_val_matrix[source_index, target_index, 1])
            non_zero_values = values[values != 0]
            if len(non_zero_values) > 0:
                z_values = fisher_z_transform(non_zero_values)
                mean_z = np.mean(z_values)
                combined_value = inverse_fisher_z_transform(mean_z)
                summary_val_matrix[source_pos, target_pos, 1] = combined_value

    if stencil_val_matrix is not None:
        return summary_graph, summary_val_matrix
    else:
        return summary_graph


def analyze_stencil_graphs(stencil_graph: np.ndarray, stencil_val_matrix: np.ndarray) -> Tuple:
    """
    Constructs and analyzes the reaction graph and summarized stencil graph from the given stencil graph and value matrix.

    This function processes the input stencil graph and value matrix to produce two outputs:
    1. The reaction graph and its associated value matrix, which describe the interactions between species.
    2. The summarized stencil graph and its associated value matrix, which describe the spatiotemporal evolution of the species.

    Args:
        stencil_graph (np.ndarray): A 3D numpy array representing the graph structure.
                                    The shape is (9*N, 9*N, 2) where N is the number of species.
                                    Each entry is either "-->" indicating a dependency or "" indicating no dependency.
        stencil_val_matrix (np.ndarray): A 3D numpy array of the same shape as stencil_graph, containing values associated with the edges in the graph.

    Returns:
        tuple: A tuple containing the reaction graph, reaction value matrix, summarized stencil graph, and summarized stencil value matrix.
               - reaction_graph (np.ndarray): A 3D numpy array of shape (N, N, 2) indicating how species relate to one another.
               - reaction_val_matrix (np.ndarray): A 3D numpy array of shape (N, N, 2) containing combined values for the relationships.
               - summary_stencil (np.ndarray): A 3D numpy array of shape (9, 9, 2) indicating summarized directional dependencies.
               - summary_stencil_val_matrix (np.ndarray): A 3D numpy array of shape (9, 9, 2) containing combined values for the summarized dependencies.
    """
    # TODO: Write a version that outputs a separate spatial-stencil for each species.
    reaction_graph, reaction_val_matrix = construct_reaction_graph(stencil_graph, stencil_val_matrix)
    summary_stencil, summary_stencil_val_matrix = summarize_stencil(stencil_graph, stencil_val_matrix)

    return reaction_graph, reaction_val_matrix, summary_stencil, summary_stencil_val_matrix


def get_species_spatial_graphs(stencil_graph, stencil_val_matrix=None):
    """
    Constructs spatial graphs for different species based on the provided stencil graph and optional stencil value matrix.

    Args:
        stencil_graph (np.ndarray): A 3D numpy array representing the graph structure. The first dimension should be a multiple of 9,
                                    corresponding to the number of species times 9 (for the 9 positions in the Moore neighborhood).
                                    The second and third dimensions represent the connections between these positions.
        stencil_val_matrix (np.ndarray, optional): A 3D numpy array containing values associated with the edges in the graph.
                                                  It should have the same shape as stencil_graph.

    Returns:
        tuple: If stencil_val_matrix is provided, returns a tuple (graphs, val_matrices):
            - graphs (np.ndarray): A 2D numpy array of shape (num_species, num_species), where each element is a 3D numpy array of shape (9, 9, 2).
                                   The 3D arrays represent the spatial graphs for each pair of species.
            - val_matrices (np.ndarray): A 2D numpy array of shape (num_species, num_species), where each element is a 3D numpy array of shape (9, 9, 2).
                                         The 3D arrays contain the values associated with the edges in the spatial graphs.
        np.ndarray: If stencil_val_matrix is not provided, returns only the graphs (np.ndarray) as described above.

    Raises:
        AssertionError: If stencil_graph is not a 3D numpy array.
        AssertionError: If the first dimension of stencil_graph is not a multiple of 9.
        AssertionError: If stencil_val_matrix is provided and its shape does not match the shape of stencil_graph.
    """
    # Assertions to ensure correct input shapes
    assert stencil_graph.ndim == 3, "stencil_graph must be a 3D numpy array"
    assert stencil_graph.shape[0] % 9 == 0, "The first dimension of stencil_graph must be a multiple of 9"
    if stencil_val_matrix is not None:
        assert stencil_graph.shape == stencil_val_matrix.shape, "stencil_graph and stencil_val_matrix must have the same shape"

    # Determine the number of species
    num_species = stencil_graph.shape[0] // 9

    graphs = np.empty(shape=(num_species, num_species), dtype=object)
    val_matrices = np.empty(shape=(num_species, num_species), dtype=object)
    for i in range(num_species):
        for j in range(num_species):
            graphs[i, j] = np.full((9, 9, 2), "", dtype=object)
            if stencil_val_matrix is not None:
                val_matrices[i, j] = np.zeros((9, 9, 2))

    # Iterate over each position in the Moore neighborhood
    for source_pos in range(9):
        for target_pos in range(9):
            for source_species in range(num_species):
                for target_species in range(num_species):
                    source_index = source_species * 9 + source_pos
                    target_index = target_species * 9 + target_pos

                    # Check for links between the source and target positions
                    if "-->" in stencil_graph[source_index, target_index, 1]:
                        graphs[source_species, target_species][source_pos, target_pos, 1] = "-->"
                        if stencil_val_matrix is not None:
                            val_matrices[source_species, target_species][source_pos, target_pos, 1] = stencil_val_matrix[source_index, target_index, 1]

    if stencil_val_matrix is not None:
        return graphs, val_matrices
    else:
        return graphs


def pretty_print_string_graph(graph: np.ndarray) -> None:
    """
    Prints a string representation of a graph in a human-readable format.

    This function takes a graph represented as a NumPy array of strings and prints it to the console in a format that is easy to read and interpret. Each row in the graph represents a parent node, and each column represents a child node. Empty connections are represented by spaces, and existing connections are preserved as they are. The function appends the parent node index to the end of each row for clarity.

    Parameters
    ----------
    graph : np.ndarray
        A 2D NumPy array where each element is a string. The array represents a graph with rows corresponding to parent nodes and columns to child nodes. Non-empty strings indicate the presence of a connection or relationship, while empty strings indicate the absence of a connection.

    Returns
    -------
    None
        This function does not return any value. It prints the formatted graph directly to the console.

    Notes
    -----
    - The function makes a copy of the input graph to avoid modifying the original array.
    - Empty connections in the graph are represented by three spaces ("   ") for visual clarity.
    - The function iterates over each row (parent node) in the graph, printing the connections to child nodes along with the parent node index at the end of the row.
    """
    g = graph.copy()
    g[g == ""] = "   "
    print("Rows= parent index, columns = child index.")
    for i, row in enumerate(g):
        print("".join([str(elem) for elem in row]) + "p" + str(i))


def plot_stencil_graph(
    stencil_graph,
    stencil_val_matrix=None,
    head_width=2,
    head_length=1,
    tail_width=0.5,
    show_colorbar=False,
    var_names=None,
    directional_var_names=False,
    label_var_names=True,
    fig=None,
    ax=None,
):
    """
    Plots a stencil graph based on the provided stencil and value matrix.

    Parameters:
    -----------
    stencil_graph : numpy.ndarray
        An adjacency matrix representing the stencil graph. The shape of this array should be (9*N, 9*N, 2),
        where N is the number of variables. Each variable has 9 spatial positions, resulting in 9*N nodes,
        and the third dimension corresponds to two possible time lags (tau and tau-1).
    stencil_val_matrix : numpy.ndarray, optional
        The value matrix corresponding to the stencil graph. If provided, it will be used to determine the color of the links.
    head_width : float, optional
        The width of the arrow heads in the plot. Default is 2.
    head_length : float, optional
        The length of the arrow heads in the plot. Default is 1.
    tail_width : float, optional
        The width of the arrow tails in the plot. Default is 0.5.
    show_colorbar : bool, optional
        Whether to show the colorbar in the plot. Default is False.
    var_names : list of str, optional
        A list of names to give each node. If not provided, default names will be generated based on the number of nodes.
    directional_var_names : bool, optional
        If True, uses directional names (e.g., "N", "S", "E", "W") for the nodes. Default is False.
    label_var_names : bool, optional
        Whether to label nodes with var_names. Default is True.
    fig : matplotlib.figure.Figure, optional
        The figure object to use for the plot. If None, a new figure will be created. Default is None.
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        The axis object to use for the plot. If None, a new axis will be created. Default is None.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axis object containing the plot.

    Example:
    --------
    >>> fig, ax = plot_stencil_graph(stencil_graph, stencil_val_matrix)
    >>> plt.show()
    """
    from matplotlib.artist import Artist
    from tigramite import plotting as tp
    import matplotlib.pyplot as plt

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    if var_names:
        var_names_passed = True
    else:
        var_names_passed = False

    if label_var_names and directional_var_names:
        var_names = ["NW", "N", "NE", "W", "C", "E", "SW", "S", "SE"]
        var_names_passed = True

    n_variables = stencil_graph.shape[0] // 9
    x_pos = np.array([[i for i in range(3)] for _ in range(3)]).flatten()
    y_pos = [i for i in range(3) for _ in range(3)]
    y_pos.reverse()
    y_pos = np.array(y_pos)
    x_pos = x_pos * n_variables
    y_pos = y_pos * n_variables
    offset = 0.0
    x_pos_tmp = np.array(x_pos)
    y_pos_tmp = np.array(y_pos)
    if not var_names_passed:
        var_names = []

        for char in char_range(num_characters=n_variables):
            if char == "a":
                var_names = [char] * 9
            else:
                offset += 0.2 * n_variables
                var_names.extend([char] * 9)
                x_pos_tmp = np.append(x_pos_tmp, x_pos + offset)
                y_pos_tmp = np.append(y_pos_tmp, y_pos)
    elif not directional_var_names:
        tmp_var_names = []
        for idx, var_name in enumerate(var_names):
            if idx == 0:
                tmp_var_names = [var_name] * 9
            else:
                offset += 0.2 * n_variables
                tmp_var_names.extend([var_name] * 9)
                x_pos_tmp = np.append(x_pos_tmp, x_pos + offset)
                y_pos_tmp = np.append(y_pos_tmp, y_pos)
        var_names = tmp_var_names

    x_pos = x_pos_tmp
    y_pos = y_pos_tmp
    node_positions = {
        "x": x_pos,
        "y": y_pos,
    }

    tp.plot_graph(
        fig_ax=(fig, ax),
        graph=stencil_graph,
        val_matrix=stencil_val_matrix,
        link_label_fontsize=0.0,
        # head_width=head_width,
        # head_length=head_length,
        # tail_width=tail_width,
        var_names=var_names if label_var_names else [""] * n_variables * 9,
        node_pos=node_positions,
        show_colorbar=show_colorbar,
    )

    # Remove link labels which always have "1"
    for child in ax.get_children():
        if isinstance(child, matplotlib.text.Text):
            if child.get_text() == "1":
                Artist.set_visible(child, False)

    return fig, ax


def plot_reaction_graph_of_two_nodes(
    reaction_graph: np.ndarray,
    reaction_val_matrix: np.ndarray = None,
    var_names: Optional[List[str]] = None,
    node_aspect: float = None,
    fig=None,
    ax=None,
) -> Tuple:
    """
    Plots a reaction graph with two nodes, one positioned on the left and one on the right.

    Parameters:
    -----------
    reaction_graph : np.ndarray
        The adjacency matrix representing the reaction graph.
    reaction_val_matrix : np.ndarray, optional
        The matrix containing the values associated with the edges of the reaction graph.
    var_names : list
        The list of strings for node labels.
    node_aspect : float, optional
        Ratio between the heigth and width of the varible nodes. Default None.
    fig : matplotlib.figure.Figure, optional
        The figure object to plot on. If None, a new figure will be created.
    ax : matplotlib.axes.Axes, optional
        The axes object to plot on. If None, new axes will be created.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    """
    from matplotlib.artist import Artist
    from tigramite import plotting as tp
    import matplotlib.pyplot as plt

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    left_node_pos = (-1, 0)
    right_node_pos = (1, 0)
    x_pos = np.array([left_node_pos[0], right_node_pos[0]])
    y_pos = np.array([left_node_pos[1], right_node_pos[1]])

    node_positions = {
        "x": x_pos,
        "y": y_pos,
    }

    tp.plot_graph(
        fig_ax=(fig, ax),
        graph=reaction_graph,
        val_matrix=reaction_val_matrix,
        var_names=var_names if var_names else ["a", "b"],
        node_pos=node_positions,
        node_aspect=node_aspect,
        show_colorbar=False,
    )

    # Remove link labels which always have "1"
    for child in ax.get_children():
        if isinstance(child, matplotlib.text.Text):
            if child.get_text() == "1":
                Artist.set_visible(child, False)

    return fig, ax


def plot_reaction_graph(
    reaction_graph: np.ndarray,
    reaction_val_matrix: np.ndarray = None,
    var_names: Optional[List[str]] = None,
    node_aspect: float = None,
    link_colorbar_label="MCI",
    node_colorbar_label="auto-MCI",
    show_colorbar=False,
    fig=None,
    ax=None,
) -> Tuple:
    """
    Plots a reaction graph with N nodes, arranged automatically via Tigramite's plot_graph().

    Parameters:
    -----------
    reaction_graph : np.ndarray
        The adjacency matrix representing the reaction graph.
    reaction_val_matrix : np.ndarray, optional
        The matrix containing the values associated with the edges of the reaction graph.
    var_names : list
        The list of strings for node labels.
    node_aspect : float, optional
        Ratio between the heigth and width of the varible nodes. Default None.
    link_colorbar_label : str, optional (default: 'MCI')
        Test statistic label.
    node_colorbar_label : str, optional (default: 'auto-MCI')
        Test statistic label for auto-dependencies.
    show_colorbar : bool
        Whether to show colorbars for links and nodes.
    fig : matplotlib.figure.Figure, optional
        The figure object to plot on. If None, a new figure will be created.
    ax : matplotlib.axes.Axes, optional
        The axes object to plot on. If None, new axes will be created.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    """
    from matplotlib.artist import Artist
    from tigramite import plotting as tp
    import matplotlib.pyplot as plt

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    tp.plot_graph(
        fig_ax=(fig, ax),
        graph=reaction_graph,
        val_matrix=reaction_val_matrix,
        var_names=var_names if var_names else [chr(i) for i in range(97, 97 + reaction_graph.shape[0])],
        node_aspect=node_aspect,
        link_colorbar_label=link_colorbar_label,
        node_colorbar_label=node_colorbar_label,
        show_colorbar=show_colorbar,
    )

    # Remove link labels which always have "1"
    for child in ax.get_children():
        if isinstance(child, matplotlib.text.Text):
            if child.get_text() == "1":
                Artist.set_visible(child, False)

    return fig, ax


def plot_grid_time_series(data: np.ndarray):
    """
    Visualizes the time series data for each grid cell in a spatial grid for multiple variables.

    This function generates a figure for each variable in the dataset. Within each figure, it creates a grid of subplots
    that corresponds to the spatial structure of the grid. Each subplot displays the time series of a specific grid cell.

    Parameters:
    -----------
    data : np.ndarray
        The generated data with shape (num_variables, grid_size, grid_size, T), where:
        - num_variables: The number of variables (or layers) in the spatial grid.
        - grid_size: The size of the spatial grid (assumed to be square).
        - T: The number of time steps for which data has been generated.

    Returns:
    --------
    fig_list : list of matplotlib.figure.Figure
        A list of figure objects, one for each variable.
    axes_list : list of numpy.ndarray of matplotlib.axes._subplots.AxesSubplot
        A list of arrays of axes objects, one array for each variable. Each array contains the axes for the subplots corresponding to the grid cells.
    """
    num_variables, grid_size, _, T = data.shape
    fig_list = []
    axs = []

    for var in range(num_variables):
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        fig_list.append(fig)
        axs.append(axes)
        fig.suptitle(f"Variable {var + 1}", fontsize=16)

        for row in range(grid_size):
            for col in range(grid_size):
                ax = axes[row, col]
                ax.plot(data[var, row, col, :])
                ax.set_title(f"Cell ({row}, {col})")
                ax.set_xlim(0, T - 1)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig_list, axs


def plot_grid_cell_time_series(data: np.ndarray, position: tuple):
    """
    Visualizes the time series data for a chosen grid cell and each of its variables.

    This function generates a figure with subplots for each variable in the dataset. Each subplot displays the time series
    of the chosen grid cell for a specific variable.

    Parameters:
    -----------
    data : np.ndarray
        The generated data with shape (num_variables, grid_size, grid_size, T), where:
        - num_variables: The number of variables (or layers) in the spatial grid.
        - grid_size: The size of the spatial grid (assumed to be square).
        - T: The number of time steps for which data has been generated.
    position : tuple
        A tuple (row, col) representing the position of the chosen grid cell.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the subplots.
    axes : numpy.ndarray of matplotlib.axes._subplots.AxesSubplot
        The array of axes objects for the subplots.

    The function infers the number of variables and the number of time steps directly from the shape of the data array.
    Each subplot corresponds to one variable, and each subplot displays the time series of the chosen grid cell.
    The time series for the chosen grid cell is plotted in its corresponding subplot, with the x-axis representing time steps and the y-axis representing the variable's value.
    The function uses `matplotlib` for plotting, and it automatically adjusts the layout to ensure that subplots are neatly arranged.
    """
    num_variables, _, _, T = data.shape
    row, col = position

    fig, axes = plt.subplots(1, num_variables, figsize=(15, 5))
    fig.suptitle(f"Time Series for Grid Cell ({row}, {col})", fontsize=16)

    for var in range(num_variables):
        ax = axes[var]
        ax.plot(data[var, row, col, :])
        ax.set_title(f"Variable {chr(97+var)}")
        ax.set_xlim(0, T - 1)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, axes


def convert_element_index_to_matrix_coordinate(index: int, num_cols: int) -> Tuple:
    """
    Converts an index to a row and column position in a matrix.

    Parameters:
    index (int): The index to convert.
    num_cols (int): The number of columns in the matrix.

    Returns:
    tuple: A tuple containing the row and column positions.

    Example:
    >>> index = 10
    >>> num_cols = 4
    >>> row, col = index_to_position(index, num_cols)
    >>> print("Row:", row)
    >>> print("Column:", col)
    Row: 2
    Column: 2
    """
    index = index % num_cols**2
    # Calculate the row and column positions
    row = index // num_cols
    col = index % num_cols

    return row, col


def convert_matrix_coordinate_to_element_index(grid_row: int, grid_col: int, num_cols: int) -> int:
    """
    Converts a row and column position in a matrix to an index.

    Parameters:
    grid_row (int): The row position.
    grid_col (int): The column position.
    num_cols (int): The number of columns in the matrix.

    Returns:
    int: The corresponding index.

    Example:
    >>> row = 2
    >>> col = 2
    >>> num_cols = 4
    >>> index = convert_matrix_coordinate_to_element_index(grid_row, grid_col, num_cols)
    >>> print("Index:", index)
    Index: 10
    """
    index = grid_row * num_cols + grid_col
    return index


def convert_matrix_coordinate_to_variable_index(grid_row: int, grid_col: int, num_cols: int, variable: int) -> int:
    """
    Converts a row and column position in a matrix to a variable index.

    Parameters:
    grid_row (int): The row position.
    grid_col (int): The column position.
    num_cols (int): The number of columns in the matrix.
    variable (int): The variable number.

    Returns:
    int: The corresponding variable index.

    Example:
    >>> grid_row = 2
    >>> grid_col = 2
    >>> num_cols = 4
    >>> variable = 1
    >>> index = convert_matrix_coordinate_to_variable_index(grid_row, grid_col, num_cols, variable)
    >>> print("Variable Index:", index)
    Variable Index: 20
    """
    index = (grid_row * num_cols + grid_col) + (num_cols**2 * variable)
    return index


def convert_element_index_to_variable_index(element_index: int, num_cols: int, variable: int = 0) -> int:
    """
    Converts an element index to a variable index in a matrix.

    Parameters:
    index (int): The element index to convert.
    num_cols (int): The number of columns in the matrix.
    variable (int, optional): The variable number. Defaults to 0.

    Returns:
    int: The corresponding variable index.

    Example:
    >>> index = 10
    >>> num_cols = 4
    >>> variable = 1
    >>> var_index = convert_element_index_to_variable_index(index, num_cols, variable)
    >>> print("Variable Index:", var_index)
    Variable Index: 20
    """
    row = element_index // num_cols
    col = element_index % num_cols
    var_index = (row * num_cols + col) + (num_cols**2 * variable)
    return var_index


def get_moore_neighborhood_indices(center_index: int, num_cols: int, num_rows: int, variable: int) -> list:
    """
    Returns the indices of the Moore neighborhood for a given grid cell index in a toroidal grid.

    Parameters:
    center_index (int): The center grid cell index.
    num_cols (int, optional): The number of columns in the grid. Defaults to 4.
    num_rows (int, optional): The number of rows in the grid. Defaults to 4.

    Returns:
    list: A list of indices representing the Moore neighborhood around center_index.

    Example:
    >>> index = 0
    >>> neighborhood = get_moore_neighborhood_indices(index)
    >>> print("Moore Neighborhood:", neighborhood)
    Moore Neighborhood: [15, 12, 3, 0, 1, 4, 7, 14, 13]
    """
    neighborhood = []
    row = center_index // num_cols
    col = center_index % num_cols

    # Iterate over the neighboring cells
    for i in range(-1, 2):
        for j in range(-1, 2):
            # Calculate the row and column positions of the neighboring cell
            neighbor_row = (row + i) % num_rows
            neighbor_col = (col + j) % num_cols
            # Calculate the index of the neighboring cell
            neighbor_index = neighbor_row * num_cols + neighbor_col
            # Append the neighboring cell index to the neighborhood list
            neighborhood.append(neighbor_index)

    neighborhood = [convert_element_index_to_variable_index(index, num_cols, variable) for index in neighborhood]
    return neighborhood


def get_variable_array_from_dataset(dataset: xr.Dataset, variables_to_extract: list) -> np.ndarray:
    """
    A helper function that extracts specified variables from an xarray.Dataset and returns them as a 4D NumPy array.

    This function checks that all specified variables have the same dimensions, extracts them as NumPy arrays,
    and stacks them along a new axis to form a 4D array with the shape (variable_n, Y, X, T), where:
      - variable_n is the number of variables,
      - Y is the latitude (rows),
      - X is the longitude (columns),
      - T is the time dimension.

    Parameters
    -----------
    dataset : xr.Dataset
        The input xarray.Dataset containing the variables to be extracted.
    variables_to_extract : list
        A list of variable names (strings) to be extracted from the dataset.

    Returns
    --------
    np.ndarray
        A 4D NumPy array with the shape (variable_n, Y, X, T) containing the extracted variables.
    """
    # Check that all variables share the same dims
    reference_dims = dataset[variables_to_extract[0]].dims
    for var_name in variables_to_extract:
        if dataset[var_name].dims != reference_dims:
            raise ValueError(f"Variable {var_name} does not have the same dimensions as the reference variable.")

    # Derive indices corresponding to time, lat, and lon.
    dims_lower = [d.lower() for d in reference_dims]
    try:
        i_time = dims_lower.index("time")
        i_lat = dims_lower.index("lat")
        i_lon = dims_lower.index("lon")
    except ValueError:
        raise ValueError("Dataset variables must have dimensions with names including 'time', 'lat', and 'lon'")

    # We now want to reorder the array to (lat, lon, time).
    permutation = (i_lat, i_lon, i_time)

    extracted_arrays = []
    for var_name in variables_to_extract:
        arr = dataset[var_name].values  # Original shape (T, Y, X) if dims are (time, lat, lon)
        if arr.ndim != 3:
            raise ValueError(f"Variable {var_name} is expected to be 3D, but got shape {arr.shape}")
        arr_tp = np.transpose(arr, permutation)  # Now shape is (lat, lon, time)
        extracted_arrays.append(arr_tp)

    # Stack along a new axis for variables.
    stacked_array = np.stack(extracted_arrays, axis=0)
    # Now stacked_array is in shape (variable_n, Y, X, T)
    return stacked_array


def map_stencil_graph_to_full_graph(stencil_graph: np.ndarray, grid_size: int, verbose=0) -> np.ndarray:
    """
    Map a stencil graph to a full graph based on the given parameters.

    The stencil graph represents dependencies between variables in a 3x3 neighborhood.
    The full graph is constructed by iterating over grid cells and their variables,
    computing the positions in the full graph, retrieving the necessary values from
    the stencil graph, and assigning them to the corresponding positions in the full graph.

    Parameters:
    stencil_graph (np.ndarray): A 3D numpy array representing the graph structure.
        The shape is (9*N, 9*N, 2) where N is the number of species (variables).
        Each entry is either "-->" indicating a dependency or "" indicating no dependency.
        The stencil graph captures dependencies within a 3x3 neighborhood for each variable.
    grid_size (int): The size of the grid (number of cells along one dimension).
        The grid is a 2D grid of size grid_size x grid_size.
    verbose (int, optional): Verbosity level. Defaults to 0.
        If set to a non-zero value, additional debug information will be printed.

    Returns:
    np.ndarray: The mapped full graph.
        The full graph is a 3D numpy array with shape
        (grid_size * grid_size * n_variables, grid_size * grid_size * n_variables, 2).
        Each entry is either "-->" indicating a dependency or "" indicating no dependency.
        Each entry indicates a dependency or no dependency between variables in the full graph.
    """
    n_variables = stencil_graph.shape[0] // 9
    full_graph = np.full(
        shape=(
            grid_size * grid_size * n_variables,
            grid_size * grid_size * n_variables,
            2,
        ),
        fill_value="",
        dtype="<U3",
    )

    grid_idx = 0
    for grid_row in range(grid_size):
        for grid_col in range(grid_size):
            for child_var in range(n_variables):
                # Compute full-graph column position, which is the full-graph child node's position.
                full_graph_col = convert_element_index_to_variable_index(grid_idx, grid_size, child_var)
                if verbose:
                    print("full_graph_col={}".format(full_graph_col))
                for parent_var in range(n_variables):
                    # Get the child position's parents as rows in the full-graph
                    neighborhood = get_moore_neighborhood_indices(full_graph_col, grid_size, grid_size, parent_var)
                    (
                        top_left_row,
                        top_row,
                        top_right_row,
                        left_row,
                        center_row,
                        right_row,
                        bot_left_row,
                        bot_row,
                        bot_right_row,
                    ) = neighborhood

                    # Access the necessary value from the stencil and assign it to the correct parent_row, child_col in the full-graph
                    full_graph[top_left_row, full_graph_col, 1] = stencil_graph[0 + 9 * parent_var, 4 + 9 * child_var, 1]
                    full_graph[top_row, full_graph_col, 1] = stencil_graph[1 + 9 * parent_var, 4 + 9 * child_var, 1]
                    full_graph[top_right_row, full_graph_col, 1] = stencil_graph[2 + 9 * parent_var, 4 + 9 * child_var, 1]

                    full_graph[left_row, full_graph_col, 1] = stencil_graph[3 + 9 * parent_var, 4 + 9 * child_var, 1]
                    full_graph[center_row, full_graph_col, 1] = stencil_graph[4 + 9 * parent_var, 4 + 9 * child_var, 1]
                    full_graph[right_row, full_graph_col, 1] = stencil_graph[5 + 9 * parent_var, 4 + 9 * child_var, 1]

                    full_graph[bot_left_row, full_graph_col, 1] = stencil_graph[6 + 9 * parent_var, 4 + 9 * child_var, 1]
                    full_graph[bot_row, full_graph_col, 1] = stencil_graph[7 + 9 * parent_var, 4 + 9 * child_var, 1]
                    full_graph[bot_right_row, full_graph_col, 1] = stencil_graph[8 + 9 * parent_var, 4 + 9 * child_var, 1]

                    if verbose:
                        print("child={}, parent={}".format(child_var, parent_var))
                        print("TL={}".format(stencil_graph[0 + 9 * parent_var, 4 + 9 * child_var, 1]))
                        print("T={}".format(stencil_graph[1 + 9 * parent_var, 4 + 9 * child_var, 1]))
                        print("TR={}".format(stencil_graph[2 + 9 * parent_var, 4 + 9 * child_var, 1]))
                        print("L={}".format(stencil_graph[3 + 9 * parent_var, 4 + 9 * child_var, 1]))
                        print("C={}".format(stencil_graph[4 + 9 * parent_var, 4 + 9 * child_var, 1]))
                        print("R={}".format(stencil_graph[5 + 9 * parent_var, 4 + 9 * child_var, 1]))
                        print("BL={}".format(stencil_graph[6 + 9 * parent_var, 4 + 9 * child_var, 1]))
                        print("B={}".format(stencil_graph[7 + 9 * parent_var, 4 + 9 * child_var, 1]))
                        print("BR={}".format(stencil_graph[8 + 9 * parent_var, 4 + 9 * child_var, 1]))

                        print("TL={}".format(top_left_row))
                        print("T={}".format(top_row))
                        print("TR={}".format(top_right_row))
                        print("L={}".format(left_row))
                        print("C={}".format(center_row))
                        print("R={}".format(right_row))
                        print("BL={}".format(bot_left_row))
                        print("B={}".format(bot_row))
                        print("BR={}".format(bot_right_row))

            grid_idx += 1

    return full_graph


def extract_stencil_from_full_graph(full_graph: np.ndarray, grid_size: int, n_variables: int, node_position: tuple = None, verbose=0) -> np.ndarray:
    """
    TODO: Test this function.
    Extract the stencil graph from the full graph based on the given parameters.

    The stencil graph represents dependencies between variables in a 3x3 neighborhood.
    The full graph is a 3D numpy array representing the graph structure of the entire grid.

    Parameters:
    full_graph (np.ndarray): The full graph as a 3D numpy array with shape
        (grid_size * grid_size * n_variables, grid_size * grid_size * n_variables, 2).
        The first two dimensions represent the flattened indices of the grid cells and variables,
        and the third dimension represents the time lag.
        Each entry indicates a dependency or no dependency between variables in the full graph.
    grid_size (int): The size of the grid (number of cells along one dimension).
        The grid is a 2D grid of size grid_size x grid_size.
    n_variables (int): The number of variables (species) in each grid cell.
    node_position (tuple, optional): The position of the interior node in the grid (row, col). Defaults to None.
        If None, a suitable interior node position will be inferred.
    verbose (int, optional): Verbosity level. Defaults to 0.
        If set to a non-zero value, additional debug information will be printed.

    Returns:
    np.ndarray: The extracted stencil graph.
        The stencil graph is a 3D numpy array with shape (9*n_variables, 9*n_variables, 2).
        Each entry is either "-->" indicating a dependency or "" indicating no dependency.
    """
    if node_position is None:
        # Infer a suitable interior node position (close to the center)
        node_position = (grid_size // 2, grid_size // 2)

    stencil_graph = np.full(
        shape=(9 * n_variables, 9 * n_variables, 2),
        fill_value="",
        dtype="<U3",
    )

    row, col = node_position
    grid_idx = convert_matrix_coordinate_to_element_index(row, col, grid_size)

    for child_var in range(n_variables):
        full_graph_col = convert_element_index_to_variable_index(grid_idx, grid_size, child_var)
        if verbose:
            print("full_graph_col={}".format(full_graph_col))
        for parent_var in range(n_variables):
            neighborhood = get_moore_neighborhood_indices(grid_idx, grid_size, grid_size, parent_var)
            (
                top_left_row,
                top_row,
                top_right_row,
                left_row,
                center_row,
                right_row,
                bot_left_row,
                bot_row,
                bot_right_row,
            ) = neighborhood

            stencil_graph[0 + 9 * parent_var, 4 + 9 * child_var, 1] = full_graph[top_left_row, full_graph_col, 1]
            stencil_graph[1 + 9 * parent_var, 4 + 9 * child_var, 1] = full_graph[top_row, full_graph_col, 1]
            stencil_graph[2 + 9 * parent_var, 4 + 9 * child_var, 1] = full_graph[top_right_row, full_graph_col, 1]

            stencil_graph[3 + 9 * parent_var, 4 + 9 * child_var, 1] = full_graph[left_row, full_graph_col, 1]
            stencil_graph[4 + 9 * parent_var, 4 + 9 * child_var, 1] = full_graph[center_row, full_graph_col, 1]
            stencil_graph[5 + 9 * parent_var, 4 + 9 * child_var, 1] = full_graph[right_row, full_graph_col, 1]

            stencil_graph[6 + 9 * parent_var, 4 + 9 * child_var, 1] = full_graph[bot_left_row, full_graph_col, 1]
            stencil_graph[7 + 9 * parent_var, 4 + 9 * child_var, 1] = full_graph[bot_row, full_graph_col, 1]
            stencil_graph[8 + 9 * parent_var, 4 + 9 * child_var, 1] = full_graph[bot_right_row, full_graph_col, 1]

            if verbose:
                print("child={}, parent={}".format(child_var, parent_var))
                print("TL={}".format(full_graph[top_left_row, full_graph_col, 1]))
                print("T={}".format(full_graph[top_row, full_graph_col, 1]))
                print("TR={}".format(full_graph[top_right_row, full_graph_col, 1]))
                print("L={}".format(full_graph[left_row, full_graph_col, 1]))
                print("C={}".format(full_graph[center_row, full_graph_col, 1]))
                print("R={}".format(full_graph[right_row, full_graph_col, 1]))
                print("BL={}".format(full_graph[bot_left_row, full_graph_col, 1]))
                print("B={}".format(full_graph[bot_row, full_graph_col, 1]))
                print("BR={}".format(full_graph[bot_right_row, full_graph_col, 1]))

                print("TL={}".format(top_left_row))
                print("T={}".format(top_row))
                print("TR={}".format(top_right_row))
                print("L={}".format(left_row))
                print("C={}".format(center_row))
                print("R={}".format(right_row))
                print("BL={}".format(bot_left_row))
                print("B={}".format(bot_row))
                print("BR={}".format(bot_right_row))

    return stencil_graph


# TODO: For testing extract_stencil_from_full_graph
# tmp_ex = results_df.iloc[0]
# tmp_graph = tmp_ex["reconstructed_full_graph"]
# tmp_n_vars = tmp_ex["num_variables"]
# tmp_grid_size = tmp_ex["grid_size_numeric"]
# tmp_stencil_graph = extract_stencil_from_full_graph(full_graph=tmp_graph, grid_size=tmp_grid_size, n_variables=tmp_n_vars)

# ms.plot_stencil_graph(tmp_stencil_graph)

# tmp_true_graph = recover_true_graph(equality_matrix=tmp_ex["graph_equality"], reconstructed_graph=tmp_graph)
# tmp_true_stencil = extract_stencil_from_full_graph(full_graph=tmp_true_graph, grid_size=tmp_grid_size, n_variables=tmp_n_vars)
# ms.plot_stencil_graph(tmp_true_stencil)

# tmp_stencil = results_df.iloc[0]["spatial_coefficients"]
# tmp_stencil_graph_true, _ = ms.get_stencil_graph_from_coefficients(tmp_stencil)
# ms.plot_stencil_graph(tmp_stencil_graph_true)

# np.all((tmp_true_stencil == tmp_stencil_graph_true))


def concatenate_timeseries_wrapping(data, rows_inverted, include_cell_index_column=False):
    """
    Concatenates timeseries data from a 2D grid with wrapping at edges, optionally including a cell index column.

    This function processes a 3D array representing timeseries data on a 2D grid, concatenating data from each cell's
    Moore neighborhood (including itself) with wrapping at the grid edges. This means that cells on the edge of the grid
    consider neighbors from the opposite edge, creating a toroidal effect.

    Parameters
    ----------
    data : np.ndarray
        A 3D numpy array of shape (rows, cols, time) representing timeseries data on a 2D grid.
    rows_inverted : bool
        If True, the rows are considered inverted, affecting the direction of 'top' and 'bottom' neighbors.
    include_cell_index_column : bool, optional
        If True, an additional column is included in the output, indicating the index of each cell. Defaults to False.

    Returns
    -------
    np.ndarray
        A 2D numpy array where each row represents concatenated timeseries data from the Moore neighborhood of each cell.
        If include_cell_index_column is True, the last column contains the cell index.

    Notes
    -----
    - The function assumes a square grid (rows = cols).
    - The inclusion of a cell index column is useful for tracking the original position of the data in the grid.
    """
    rows = data.shape[0]
    cols = data.shape[0]
    if include_cell_index_column:
        concatenated_data = [[] for i in range(10)]
        index = 0
    else:
        concatenated_data = [[] for i in range(9)]
    index = 0
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
                from_top = data[(row + 1) % rows, col, :]
                from_bottom = data[row - 1, col, :]
                from_top_left = data[(row + 1) % rows, col - 1, :]
                from_top_right = data[(row + 1) % rows, (col + 1) % cols, :]
                from_bot_left = data[row - 1, col - 1, :]
                from_bot_right = data[row - 1, (col + 1) % cols, :]
                from_self = data[row, col, :]

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


def concatenate_timeseries_nonwrapping(data, rows_inverted, include_cell_index_column=False):
    """
    Concatenates timeseries data from a 2D grid without wrapping at edges, optionally including a cell index column.

    This function processes a 3D array representing timeseries data on a 2D grid, concatenating data from each cell's
    Moore neighborhood (including itself) without wrapping at the grid edges. This means that cells on the edge of the grid
    do not consider neighbors from the opposite edge, contrasting the toroidal effect present in the wrapping version.

    Parameters
    ----------
    data : np.ndarray
        A 3D numpy array of shape (rows, cols, time) representing timeseries data on a 2D grid.
    rows_inverted : bool
        If True, the rows are considered inverted, affecting the direction of 'top' and 'bottom' neighbors. If inverted, then position(row-1) is above position(row).
    include_cell_index_column : bool, optional
        If True, an additional column is included in the output, indicating the index of each cell. Defaults to False.

    Returns
    -------
    np.ndarray
        A 2D numpy array where each row represents concatenated timeseries data from the Moore neighborhood of each cell,
        excluding edge wrapping. If include_cell_index_column is True, the last column contains the cell index.

    Notes
    -----
    - The function assumes a square grid (rows = cols) and excludes the outermost cells to avoid edge cases without wrapping.
    - The inclusion of a cell index column is useful for tracking the original position of the data in the grid.
    """
    ROW_LEN = data.shape[0]
    COL_LEN = data.shape[1]

    # create one cell buffer such that all cells evaluated have neighbors on all sides
    ROW_RANGE = range(1, ROW_LEN - 1)
    # create one cell buffer such that all cells evaluated have neighbors on all sides
    COL_RANGE = range(1, COL_LEN - 1)

    if include_cell_index_column:
        concatenated_data = [[] for i in range(10)]
        index = 0
    else:
        concatenated_data = [[] for i in range(9)]
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


def threshold_graph(
    graph: np.ndarray,
    val_matrix: np.ndarray,
    strength_threshold: float,
    q_matrix: np.ndarray = None,
    p_matrix: np.ndarray = None,
) -> Tuple:
    """
    Thresholds the val_matrix, graph, q_matrix, and p_matrix based on the given strength_threshold.

    Parameters
    ----------
    graph : np.ndarray
        The resulting causal graph.
    val_matrix : np.ndarray
        The matrix of test statistic values regarding adjacencies.
    strength_threshold : float
        The threshold for the dependence strength. Values below this threshold will be set to 0.0 in val_matrix,
        "" in graph, and 1.0 in q_matrix and p_matrix if provided.
    q_matrix : np.ndarray, optional
        The matrix of corrected p-values. Defaults to None.
    p_matrix : np.ndarray, optional
        The matrix of p-values. Defaults to None.

    Returns
    -------
    tuple
        The thresholded val_matrix, graph, and optionally q_matrix and p_matrix if they are provided.
    """
    # Ensure the input arrays have the same shape if they are provided
    if q_matrix is not None:
        assert graph.shape == val_matrix.shape == q_matrix.shape, "graph, val_matrix, and q_matrix must have the same shape"
    if p_matrix is not None:
        assert graph.shape == val_matrix.shape == p_matrix.shape, "graph, val_matrix, and p_matrix must have the same shape"

    # Create copies of the input arrays to avoid modifying the originals
    thresholded_val_matrix = np.copy(val_matrix)
    thresholded_graph = np.copy(graph)
    thresholded_q_matrix = np.copy(q_matrix) if q_matrix is not None else None
    thresholded_p_matrix = np.copy(p_matrix) if p_matrix is not None else None

    # Apply the threshold based on the absolute value
    below_threshold = np.abs(thresholded_val_matrix) < strength_threshold
    thresholded_val_matrix[below_threshold] = 0.0
    thresholded_graph[below_threshold] = ""

    if thresholded_q_matrix is not None:
        thresholded_q_matrix[below_threshold] = 1.0
    if thresholded_p_matrix is not None:
        thresholded_p_matrix[below_threshold] = 1.0

    # Prepare the return values
    return_values = (thresholded_val_matrix, thresholded_graph)
    if thresholded_q_matrix is not None:
        return_values += (thresholded_q_matrix,)
    if thresholded_p_matrix is not None:
        return_values += (thresholded_p_matrix,)

    return return_values


def get_MV_reduced_space(data, dependencies_wrap, rows_inverted):
    """
    Generates the reduced space concatenated time series for each variable in a 4D dataset.

    This function preprocesses a 4-dimensional dataset representing multiple variables across a 2D spatial grid over time.
    It concatenates the timeseries data from each cell's neighborhood, either with or without wrapping at the grid edges,
    and combines the reduced spaces for each variable into a single reduced space.

    Parameters
    ----------
    data : np.ndarray
        A 4D numpy array of shape (variable_n, X, Y, T) representing the data for multiple variables across a 2D spatial grid over time.
    dependencies_wrap : bool
        Specifies whether to wrap dependencies at the edges of the spatial grid during data preprocessing.
    rows_inverted : bool
        Specifies whether the rows in the spatial grid are inverted, affecting the direction of 'top' and 'bottom' neighbors during data preprocessing.

    Returns
    -------
    np.ndarray
        A 2D numpy array representing the reduced space concatenated time series for each variable. The shape of the array is (T, variable_n * neighborhood_size),
        where T is the number of time points and neighborhood_size is the number of cells in the neighborhood (including the central cell).

    Notes
    -----
    - The preprocessing step concatenates timeseries data from each cell's Moore neighborhood, which can be configured to include wrapping at the grid edges.
    - The reduced coordinate space is a 2D representation of the original 4D dataset, where each node corresponds to a concatenated time series from a specific spatial location and variable.
    - This transformation allows causal discovery algorithms to operate on a spatial representation of the data, while still capturing the temporal and spatial dependencies.
    """
    # Find reduced space concatenated time series for each variable
    reduced_data = np.empty(data.shape[0], dtype=np.ndarray)
    for variable in range(data.shape[0]):
        var_data = data[variable, :, :, :]
        if dependencies_wrap:
            concatenated_data = concatenate_timeseries_wrapping(var_data, rows_inverted=rows_inverted, include_cell_index_column=False)
        else:
            concatenated_data = concatenate_timeseries_nonwrapping(var_data, rows_inverted=rows_inverted, include_cell_index_column=False)
        reduced_data[variable] = concatenated_data
    # Combine reduced spaces
    reduced_space = np.concatenate(reduced_data, axis=1)
    return reduced_space


def verify_graph_against_assumptions(graph: np.ndarray, link_assumptions: dict) -> bool:
    """
    Verifies that the given graph satisfies the provided link assumptions.

    This function checks that the graph does not contain any links that are not allowed by the link_assumptions.
    The link_assumptions dictionary specifies the allowed links, and any link not specified in this dictionary should be absent in the graph.

    Parameters
    ----------
    graph : np.ndarray
        The causal graph to be verified, of shape [N, N, tau_max+1], with links represented as strings.
    link_assumptions : dict
        A dictionary specifying assumptions about links. The dictionary should be of the form
        {j: {(i, -tau): link_type, ...}, ...}, where j and i are indices of the variables, and link_type specifies the type of link.
        This initializes the graph with entries graph[i,j,tau] = link_type. For example, graph[i,j,0] = ‘–>’ implies that a
        directed link from i to j at lag 0 must exist. Valid link types are ‘o-o’, ‘–>’, ‘<–’. In addition, the middle mark
        can be ‘?’ instead of ‘-’. Then ‘-?>’ implies that this link may not exist, but if it exists, its orientation is ‘–>’.
        Link assumptions need to be consistent, i.e., graph[i,j,0] = ‘–>’ requires graph[j,i,0] = ‘<–’ and acyclicity must hold.
        If a link does not appear in the dictionary, it is assumed absent. That is, if link_assumptions is not None, then all
        links have to be specified or the links are assumed absent.

    Returns
    -------
    bool
        True if the graph satisfies all the link assumptions, False otherwise.

    Notes
    -----
    - The function iterates over all possible child nodes, parent nodes, and lags.
    - If a link assumption is specified for a given child, parent, and lag, the function checks if the graph satisfies the assumption:
        - `o-o`: An undirected link should exist between the nodes.
        - `-->`: A directed link should exist from the parent to the child.
        - `<--`: A directed link should exist from the child to the parent.
        - `o?o`: A link of any type may exist between the nodes.
        - `-?>`: A possible directed link should exist from the parent to the child.
        - `<?-`: A possible directed link should exist from the child to the parent.
    - If no link assumption is specified, the function checks that no link exists between the nodes.
    """

    node_count = graph.shape[0]
    tau_max = graph.shape[2] - 1

    for parent_i in range(node_count):
        for child_j in range(node_count):
            for lag in range(tau_max + 1):
                if child_j in link_assumptions and (parent_i, -lag) in link_assumptions[child_j]:
                    link_type = link_assumptions[child_j][(parent_i, -lag)]
                    if link_type == "o-o":
                        # An undirected link should exist between parent_i and child_j at the given lag
                        if graph[parent_i, child_j, lag] != "o-o":
                            print(f"Failed assumption: {parent_i} o-o {child_j} at lag {lag}")
                            return False
                    elif link_type == "-->":
                        # A directed link should exist from parent_i to child_j at the given lag
                        if graph[parent_i, child_j, lag] != "-->" or graph[child_j, parent_i, lag] != "<--":
                            print(f"Failed assumption: {parent_i} --> {child_j} at lag {lag}")
                            return False
                    elif link_type == "<--":
                        # A directed link should exist from child_j to parent_i at the given lag
                        if graph[child_j, parent_i, lag] != "-->" or graph[parent_i, child_j, lag] != "<--":
                            print(f"Failed assumption: {child_j} <-- {parent_i} at lag {lag}")
                            return False
                    elif link_type == "o?o":
                        # An undirected or no link should exist between parent_i and child_j at the given lag
                        if graph[parent_i, child_j, lag] not in ["", "o-o", "<--", "-->"]:
                            print(f"Failed assumption: {parent_i} o?o {child_j} at lag {lag}")
                            return False
                    elif link_type == "-?>":
                        # A possible directed link should exist from parent_i to child_j at the given lag
                        if graph[parent_i, child_j, lag] not in ["", "-->"]:
                            print(f"Failed assumption: {parent_i} -?> {child_j} at lag {lag}")
                            return False
                    elif link_type == "<?-":
                        # A possible directed link should exist from child_j to parent_i at the given lag
                        if graph[child_j, parent_i, lag] not in ["", "-->"]:
                            print(f"Failed assumption: {child_j} <?- {parent_i} at lag {lag}")
                            return False
                else:
                    # If no link assumption is specified, no link should exist
                    if graph[parent_i, child_j, lag] != "":
                        print(f"Failed assumption: No link should exist between {parent_i} and {child_j} at lag {lag}")
                        return False

    return True


def build_link_assumptions(
    node_count: int,
    min_tau: int,
    max_tau: int,
    intervariable_link_assumptions: Optional[Dict[int, Dict[Tuple[int, int], str]]] = None,
    allow_center_directed_links: bool = False,
) -> Dict[int, Dict[Tuple[int, int], str]]:
    """
    Constructs the link assumptions dictionary for the CaStLe-PC/MCI algorithm.

    This function sets up link assumptions for standard CaStLe, without any known variable assumptions.
    It only estimates parents of specific variables. The possible children are every 9th node after the 5th (index 4).
    The link assumptions dictionary is of the form {j: {(i, -tau): link_type, ...}, ...} specifying assumptions about links.
    This initializes the graph with entries graph[i, j, tau] = link_type. For example, graph[i, j, 0] = '->' implies that a
    directed link from i to j at lag 0 must exist. Valid link types are 'o-o', '->', '<-'. In addition, the middle mark
    can be '?' instead of '-'. Then '-?>' implies that this link may not exist, but if it exists, its orientation is '->'.
    Link assumptions need to be consistent, i.e., graph[i, j, 0] = '->' requires graph[j, i, 0] = '<-' and acyclicity must hold.
    If a link does not appear in the dictionary, it is assumed absent. That is, if link_assumptions is not None, then all
    links have to be specified or the links are assumed absent.

    Parameters
    ----------
    node_count : int
        The total number of nodes in the reduced space.
    min_tau : int
        The minimum lag to consider.
    max_tau : int
        The maximum lag to consider.
    intervariable_link_assumptions : dict, optional
        A dictionary specifying assumptions about links between different variables in the reduced coordinate space.
        The dictionary should be of the form {j: {(i, -tau): link_type, ...}, ...}, where j and i are indices of the
        variables, and link_type specifies the type of link. If not provided, a variable-naive multivariate stencil is constructed.
    allow_center_directed_links : bool, optional
        Specifies whether to allow directed links between the specific variables (4, 13, 22, 31, etc.), essentially intra-grid-cell dependence.
        If set to True, directed links are allowed; otherwise, only undirected links are allowed. Defaults to False.

    Returns
    -------
    dict
        The link assumptions dictionary.
    """
    possible_children = list(range(4, node_count, 9))
    link_assumptions = {}

    for child_j in range(node_count):
        if child_j in possible_children:  # only nodes in possible_children can be children
            link_assumptions[child_j] = {}
            for parent_i in range(node_count):
                for lag in range(min_tau, max_tau + 1):
                    if intervariable_link_assumptions is not None:
                        # Build passed link_assumptions into CaStLe's reduced coordinate space
                        child_intervar_link_idx = int((child_j - 4) / 9)
                        parent_intervar_link_idx = math.floor(parent_i / 9)
                        try:
                            link_assumptions[child_j][(parent_i, -lag)] = intervariable_link_assumptions[child_intervar_link_idx][(parent_intervar_link_idx, -lag)]
                        except KeyError:
                            # No link was specified, so nothing should be specified here.
                            pass
                    elif parent_i in possible_children and not allow_center_directed_links:
                        # Links between specific variables must be undirected
                        link_assumptions[child_j][(parent_i, -lag)] = "o?o"
                    else:
                        # Possible directed links from parent to child
                        link_assumptions[child_j][(parent_i, -lag)] = "-?>"
        else:
            # No other nodes can be children
            link_assumptions[child_j] = {}

    return link_assumptions


def mv_CaStLe_PC(
    data: np.ndarray,
    cond_ind_test: CondIndTest,
    pc_alpha: float,
    graph_p_threshold: float,
    min_tau: int = 1,
    strength_threshold: float = None,
    rows_inverted=False,
    cd_function="run_pcalg",
    fdr_method: str = None,
    intervariable_link_assumptions: dict = None,
    allow_center_directed_links=False,
    dependencies_wrap=False,
    return_reduced_space=False,
    verbose=1,
) -> Tuple:
    """
    Executes the multivariate CaStLe-PC/MCI algorithm on space-time data to infer causal relationships.

    This function applies a causal discovery algorithm on a 4-dimensional dataset representing multiple variables
    across a 2D spatial grid over time. It preprocesses the data by concatenating timeseries from each cell's
    neighborhood, either with or without wrapping at the grid edges, and then applies the specified causal discovery
    algorithm, PC or PCMCI, to infer the causal graph.

    Parameters
    ----------
    Data Parameters:
    data : np.ndarray
        A 4D numpy array of shape (variable_n, X, Y, T) representing the data for multiple variables across a 2D spatial
        grid over time.
    rows_inverted : bool, optional
        Specifies whether the rows in the spatial grid are inverted, affecting the direction of 'top' and 'bottom'
        neighbors during data preprocessing. Defaults to False.
    dependencies_wrap : bool, optional
        Specifies whether to wrap dependencies at the edges of the spatial grid during data preprocessing. Defaults to False.

    Causal Discovery Parameters:
    cond_ind_test : CondIndTest
        An instance of a conditional independence test to be used by the causal discovery algorithm.
    pc_alpha : float
        The significance level for conditional independence tests in the PC algorithm. Determines whether a conditional
        independence test result is statistically significant.
    graph_p_threshold : float
        The threshold for considering dependencies as significant when reconstructing the causal graph. Links with p-values
        less than or equal to this threshold are included in the final causal graph.
    min_tau : int, optional
        The minimum lag to consider. Must be 0 or 1. Defaults to 1.
    cd_function : str, optional
        The name of the causal discovery function to be called on the PCMCI instance. Supported functions are
        "run_pcalg" for the PC (Peter-Clark) algorithm and "run_pcmci" for the PCMCI (PC-Momentary-Conditional-Independence)
        algorithm. Defaults to "run_pcalg".
    fdr_method : str, optional
        Correction method, currently implemented is Benjamini-Hochberg False Discovery Rate method with "bh". Defaults to None.
    intervariable_link_assumptions : dict, optional
        A dictionary specifying assumptions about links between different variables in the reduced coordinate space.
        The dictionary should be of the form {j: {(i, -tau): link_type, ...}, ...}, where j and i are indices of the
        variables, and link_type specifies the type of link. If not provided, a variable-naive multivariate stencil is constructed.
    allow_center_directed_links : bool, optional
        Specifies whether to allow directed links between the specific variables (4, 13, 22, 31, etc.), essentially intra-grid-cell dependence.
        If set to True, directed links are allowed; otherwise, only undirected links are allowed. Defaults to False.

    Post-Processing Parameters:
    strength_threshold : float, optional
        The threshold for filtering edges based on their strength. If provided, edges with strength below this threshold
        will be removed from the graph. Defaults to None.

    Output Parameters:
    return_reduced_space : bool, optional
        Specifies whether to return the reduced space in addition to the results. Defaults to False.
    verbose : int, optional
        Level of verbosity. Defaults to 1.

    Returns
    -------
    tuple
        The tuple contains different elements based on the choice of `cd_function`:

        If `cd_function` is "run_pcalg", the tuple contains:
            - graph : array of shape [N, N, tau_max+1]
                Resulting causal graph.
            - val_matrix : array of shape [N, N, tau_max+1]
                Estimated matrix of test statistic values regarding adjacencies.
            - p_matrix : array of shape [N, N, tau_max+1]
                Estimated matrix of p-values regarding adjacencies.
            - sepsets : dictionary
                Separating sets. See Tigramite documentation for details.
            - ambiguous_triples : list
                List of ambiguous triples, only relevant for 'majority' and 'conservative' rules.

        If `cd_function` is "run_pcmci", the tuple contains:
            - graph : array of shape [N, N, tau_max+1]
                Resulting causal graph.
            - val_matrix : array of shape [N, N, tau_max+1]
                Estimated matrix of test statistic values.
            - p_matrix : array of shape [N, N, tau_max+1]
                Estimated matrix of p-values, optionally adjusted if fdr_method is not 'none'.
            - conf_matrix : array of shape [N, N, tau_max+1,2]
                Estimated matrix of confidence intervals of test statistic values. Only computed if set in cond_ind_test, where also the percentiles are set.

        If `return_reduced_space` is True, the tuple also contains:
            - reduced_space : array
                The reduced space concatenated time series for each variable.

    Raises
    ------
    AssertionError
        If the input data does not have 4 dimensions.
    ValueError
        If there are inconsistent link assumptions that would lead to an inconsistent graph.

    Notes
    -----
    - The preprocessing step concatenates timeseries data from each cell's Moore neighborhood, which can be configured
      to include wrapping at the grid edges.
    - The causal discovery process is configurable via the `cd_function` parameter, allowing for different algorithms
      to be applied. The PC algorithm here is adapted for time series and is a standard approach for causal discovery, while the PCMCI algorithm is tailored
      for autocorrelated time series data.
    - The reduced coordinate space is a 2D representation of the original 4D dataset, where each node corresponds to a
      concatenated time series from a specific spatial location and variable. This transformation allows the causal discovery
      algorithm to operate on a simplified representation of the data, while still capturing the temporal and spatial dependencies.
    - The `intervariable_link_assumptions` parameter allows users to specify assumptions about links between different variables
      in the reduced coordinate space, providing more control over the causal discovery process. If no such dictionary is provided,
      the function defaults to its original behavior of creating a naive stencil with undirected or directed links based on the `allow_center_directed_links` flag.
    """
    assert len(data.shape) == 4, "data needs to have 4 dimensions (variable_n, X, Y, T)"
    assert graph_p_threshold < 1.0, "graph_p_threshold must be less than 1.0, otherwise illegal links are output."

    max_tau = 1

    reduced_space = get_MV_reduced_space(data=data, dependencies_wrap=dependencies_wrap, rows_inverted=rows_inverted)

    if verbose:
        print(f"Data concatenated into Moore neighborhood space (shape{reduced_space.shape})...")

    # Only estimate parents of specific variables. This sets up link_assumptions for standard CaStLe, without any known varaible assumptions.
    # Get possible children, which are every 9th node after the 5th (index 4)
    # link_assumptions (dict) – Dictionary of form {j:{(i, -tau): link_type, ...}, ...} specifying assumptions about links.
    # This initializes the graph with entries graph[i,j,tau] = link_type. For example, graph[i,j,0] = ‘–>’ implies that a
    # directed link from i to j at lag 0 must exist. Valid link types are ‘o-o’, ‘–>’, ‘<–’. In addition, the middle mark
    # can be ‘?’ instead of ‘-’. Then ‘-?>’ implies that this link may not exist, but if it exists, its orientation is ‘–>’.
    # Link assumptions need to be consistent, i.e., graph[i,j,0] = ‘–>’ requires graph[j,i,0] = ‘<–’ and acyclicity must hold.
    # If a link does not appear in the dictionary, it is assumed absent. That is, if link_assumptions is not None, then all
    # links have to be specified or the links are assumed absent.
    node_count = reduced_space.shape[1]
    link_assumptions = build_link_assumptions(
        node_count=node_count,
        min_tau=min_tau,
        max_tau=max_tau,
        intervariable_link_assumptions=intervariable_link_assumptions,
        allow_center_directed_links=allow_center_directed_links,
    )

    pcmci_df = pp.DataFrame(reduced_space)
    pcmci = PCMCI(dataframe=pcmci_df, cond_ind_test=cond_ind_test, verbosity=0)

    results = getattr(pcmci, cd_function)(
        tau_min=min_tau,
        tau_max=max_tau,
        pc_alpha=pc_alpha,
        link_assumptions=link_assumptions,
    )
    if fdr_method:
        if fdr_method == "bh":
            fdr_method = "fdr_bh"  # Rename to conform to tigramite's expected convention.
        q_matrix = pcmci.get_corrected_pvalues(
            p_matrix=results["p_matrix"],
            tau_min=min_tau,
            tau_max=max_tau,
            fdr_method=fdr_method,
            link_assumptions=link_assumptions,
        )
        reconstructed_graph = pcmci.get_graph_from_pmatrix(
            p_matrix=q_matrix,
            alpha_level=graph_p_threshold,  # graph_bool = p_matrix <= alpha_level
            tau_min=min_tau,
            tau_max=max_tau,
            link_assumptions=link_assumptions,
        )
        results["graph"] = reconstructed_graph
        results["q_matrix"] = q_matrix

    if strength_threshold:
        if fdr_method:
            results["val_matrix"], results["graph"], results["q_matrix"] = threshold_graph(
                graph=results["graph"],
                val_matrix=results["val_matrix"],
                q_matrix=results["q_matrix"],
                strength_threshold=strength_threshold,
            )
        else:
            results["val_matrix"], results["graph"], results["p_matrix"] = threshold_graph(
                graph=results["graph"],
                val_matrix=results["val_matrix"],
                p_matrix=results["p_matrix"],
                strength_threshold=strength_threshold,
            )

    if verbose:
        print("Stencil learned...")

    if return_reduced_space:
        return results, reduced_space
    else:
        return results
