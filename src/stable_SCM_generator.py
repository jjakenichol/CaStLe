import numpy as np
from numpy import linalg as LA
import random
from causalnex.structure import StructureModel
from scipy import stats
from scipy.sparse import random as rndm
from tigramite.toymodels import structural_causal_processes
from typing import Union, Callable


def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.

    Adapted from https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/

    Parameters
    ----------
    pos : numpy.ndarray
        an array constructed by packing the meshed arrays of variables
        x_1, x_2, x_3, ..., x_k into its _last_ dimension
    mu : float
        the distribution mean
    Sigma : float
        the distribution standard deviation

    Returns
    -------
    numpy.ndarray
        a multivariate gaussian distributed array
    """
    # Mean vector and covariance matrix
    mode_mu = np.array([0.0, 1.0])

    n = mode_mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2 * np.pi) ** n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum("...k,kl,...l->...", pos - mu, Sigma_inv, pos - mu)

    return np.exp(-fac / 2) / N


def get_random_positive_semidefinite_matrix(size, Sigma):
    """Generates a random positive-semidefinite matrix with given standard deviation.

    Adapted from https://stackoverflow.com/a/619406/9969212

    Parameters
    ----------
    size : int
        matrix size
    Sigma : float
        standard deviation of Gaussian distributed covariance matrix
    plot : bool, optional
        whether to plot the resulting matrix, by default False

    Returns
    -------
    numpy.ndarray
        a random positive-semidefinite matrix with given standard deviation

    Raises
    ------
    ValueError
        If the random matrix includes infinite values then fail.
    """
    # Our 2-dimensional distribution will be over variables X and Y
    X = np.linspace(-3, 3, size)  # , N)
    Y = np.linspace(-3, 4, size)  # , N)
    X, Y = np.meshgrid(X, Y)

    # Mean vector and covariance matrix
    mode_mu = np.array([0.0, 1.0])

    n = mode_mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2 * np.pi) ** n * Sigma_det)

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    # The distribution on the variables X, Y packed into pos.
    Z = multivariate_gaussian(pos, mode_mu, Sigma)

    if not np.isfinite(Z).all() or (Z > 0.5).any():
        raise ValueError("There are infinites in Z")

    return Z


def pad_n_cols_right_of_2d_matrix(arr, n, fill_value):
    """Adds n columns of zeros to right of 2D numpy array matrix.
    https://stackoverflow.com/a/72339584/9969212

    Parameters
    ----------
    arr : np.ndarray
        A two dimensional numpy array that is padded
    n : int
        The number of columns that are added to the right of the matrix
    fill_value : float
        Value inserted to empty cells

    Returns
    -------
    np.ndarray
        Padded array matrix
    """

    # Create a new array with the same number of rows and additional n columns filled with fill_value
    padded_array = np.full((arr.shape[0], arr.shape[1] + n), fill_value)
    # Copy the original array into the new padded array, leaving the additional columns as zeros
    padded_array[:, : arr.shape[1]] = arr
    return padded_array


def pad_n_rows_below_2d_matrix(arr, n, fill_value):
    """Adds n rows of zeros below 2D numpy array matrix.
    https://stackoverflow.com/a/72339584/9969212

    Parameters
    ----------
    arr : np.ndarray
        A two dimensional numpy array that is padded
    n : int
        The number of columns that are added to the right of the matrix
    fill_value : float
        Value inserted to empty cells

    Returns
    -------
    np.ndarray
        Padded array matrix
    """

    # Create a new array with the same number of columns and additional n rows filled with fill_value
    padded_array = np.full((arr.shape[0] + n, arr.shape[1]), fill_value)
    # Copy the original array into the new padded array, leaving the additional rows as zeros
    padded_array[: arr.shape[0], :] = arr
    return padded_array


def create_coefficient_matrix(local_coefficients, grid_size, fill_value=0.0):
    """A mapping function that generates a (grid_size**2, grid_size**2) size matrix from a given (3, 3) matrix.

    The given (3, 3) matrix is a local dependence coefficient matrix that determines the dependence of
    neighboring grid cells, within a (grid_size, grid_size) toroidal space, relative to the center grid cell
    (the one at position [1, 1]). This function determines the matrix that defines the dependence between every
    grid cell relative to every other grid cell.


    Parameters
    ----------
    grid_size : int
        The first dimension of the (grid_size, grid_size) toroidal space
    local_coefficients : numpy.ndarray
        The (3, 3) local dependence coefficient matrix
    fill_value : any, optional
        The value inserted for all grid cell pairs without dependence, by default 0.0

    Returns
    -------
    numpy.ndarray
        The (grid_size**2, grid_size**2) dependence matrix between all grid cells
    """
    # Check if local_coefficients is of the correct type and shape
    assert (
        type(local_coefficients) == np.ndarray or type(local_coefficients) == np.matrix
    ), "local_coefficients must be of type numpy.ndarray or numpy.matrix"
    assert local_coefficients.shape == (9,) or local_coefficients.shape == (
        3,
        3,
    ), "local_coefficients must have shape (9,) or (3, 3)."

    # Create a grid of ones with dimensions (grid_size, grid_size)
    grid = np.ones((grid_size, grid_size))

    # Create a coefficient matrix with dimensions (grid_size, grid_size, grid_size, grid_size) filled with fill_value
    coefficient_matrix = np.full((grid_size, grid_size, grid_size, grid_size), fill_value)
    # Reshape the local_coefficients to (3, 3) and make a copy
    coefficients = np.array(local_coefficients)
    coefficients = coefficients.reshape(3, 3).copy()

    # Calculate the padding width for the coefficients matrix
    pad_width = grid.shape[0] - coefficients.shape[0]
    # Pad the coefficients matrix with fill_value (default 0.0)
    padded_coeff = pad_n_cols_right_of_2d_matrix(coefficients, pad_width, fill_value)
    padded_coeff = pad_n_rows_below_2d_matrix(padded_coeff, pad_width, fill_value)
    # Define the center position of the padded coefficients matrix
    padded_coeff_center = (1, 1)

    # Iterate over each combination of row and column indices in the coefficient_matrix
    for C_rowrow in range(grid_size):
        for C_rowcol in range(grid_size):
            for C_colrow in range(grid_size):
                for C_colcol in range(grid_size):
                    # Calculate the offset between the current row and column indices
                    offset = (C_rowrow - C_colrow, C_rowcol - C_colcol)
                    # Assign the corresponding value from the padded coefficients matrix to the coefficient_matrix
                    coefficient_matrix[C_rowrow, C_rowcol, C_colrow, C_colcol] = padded_coeff[
                        (padded_coeff_center[0] - offset[0]) % padded_coeff.shape[0],
                        (padded_coeff_center[1] - offset[1]) % padded_coeff.shape[1],
                    ]

    return coefficient_matrix.reshape(grid_size**2, grid_size**2)


def create_nonwrapping_coefficient_matrix(local_coefficients, grid_size=3, fill_value=0.0):
    """A mapping function that generates a (grid_size**2, grid_size**2) size matrix from a given (3, 3) matrix.

    The given (3, 3) matrix is a local dependence coefficient matrix that determines the dependence of
    neighboring grid cells, within a (grid_size, grid_size) NON-toroidal space, relative to the center grid cell
    (the one at position [1, 1]). This function determines the matrix that defines the dependence between every
    grid cell relative to every other grid cell.


    Parameters
    ----------
    grid_size : int
        The first dimension of the (grid_size, grid_size) toroidal space
    local_coefficients : numpy.ndarray
        The (3, 3) local dependence coefficient matrix
    fill_value : any, optional
        The value inserted for all grid cell pairs without dependence, by default 0.0

    Returns
    -------
    numpy.ndarray
        The (grid_size**2, grid_size**2) dependence matrix between all grid cells
    """
    # Create a grid of ones with dimensions (grid_size, grid_size)
    grid = np.ones((grid_size, grid_size))
    # Create a coefficient matrix with dimensions (grid_size, grid_size, grid_size, grid_size) filled with fill_value
    coefficient_matrix = np.full((grid_size, grid_size, grid_size, grid_size), fill_value)
    # Reshape the local_coefficients to (3, 3) and make a copy
    coefficients = np.array(local_coefficients)
    coefficients = coefficients.reshape(3, 3).copy()

    # Calculate the padding width for the coefficients matrix
    pad_width = grid.shape[0] - coefficients.shape[0]
    # Pad the coefficients matrix with fill_value (default 0.0)
    padded_coeff = pad_n_cols_right_of_2d_matrix(coefficients, pad_width, fill_value)
    padded_coeff = pad_n_rows_below_2d_matrix(padded_coeff, pad_width, fill_value)
    # Define the center position of the padded coefficients matrix
    padded_coeff_center = (1, 1)

    # Iterate over each combination of row and column indices in the coefficient_matrix
    for C_rowrow in range(grid_size):
        for C_rowcol in range(grid_size):
            for C_colrow in range(grid_size):
                for C_colcol in range(grid_size):
                    # Calculate the offset between the current row and column indices
                    offset = (C_rowrow - C_colrow, C_rowcol - C_colcol)

                    # Check if the offset exceeds the grid boundaries, and continue if it does
                    if ((padded_coeff_center[0] - offset[0]) >= grid_size) or (
                        (padded_coeff_center[0] - offset[0]) < 0
                    ):
                        # No wrapping, reached a boundary and must ignore
                        continue
                    if ((padded_coeff_center[1] - offset[1]) >= grid_size) or (
                        (padded_coeff_center[1] - offset[1]) < 0
                    ):
                        # No wrapping, reached a boundary and must ignore
                        continue

                    # Assign the corresponding value from the padded coefficients matrix to the coefficient_matrix
                    coefficient_matrix[C_rowrow, C_rowcol, C_colrow, C_colcol] = padded_coeff[
                        padded_coeff_center[0] - offset[0],
                        padded_coeff_center[1] - offset[1],
                    ]

    return coefficient_matrix.reshape(grid_size**2, grid_size**2)


def is_stable(matrix, verbose=0):
    """Checks for numerical stability, i.e. if all eigenvalues of the given matrix are < 1.0

    Parameters
    ----------
    matrix : numpy.ndarray
        The matrix to be checked for stability
    verbose : int, optional
        Whether or not to print eigenvalues, by default 0

    Returns
    -------
    Boolean
        Whether or not the given matrix is stable
    """
    eigenvalues, _ = LA.eig(matrix)
    if verbose > 0:
        print("eigenvalues:")
        print(eigenvalues)

    return np.all(np.abs(eigenvalues) < 1.0)


def get_random_stable_coefficient_matrix(grid_size, density, min_value_threshold=None, verbose=0):
    """Generates a random local dependence coefficient matrix that will produce a stable dynamics matrix of size (grid_size**2, grid_size**2).

    The dynamics matrix is a mapping of the local spatial coefficient matrix onto gridded toroidal space of size
    (grid_size, grid_size), where the dynamics matrix describes the effect of each grid cell on each other grid cell.

    Parameters
    ----------
    grid_size : int
        First dimension of a square spatial grid, i.e. a spatial grid of size (grid_size, grid_size)
    density : float
        Density of the desired random local dependence matrix
    verbose : int, optional
        Pass 1 to print the generated random local dependence matrix, pass 2 to also print the computed eigenvalues, by default 0

    Returns
    -------
    numpy.ndarray
        A random local dependence coefficient matrix with shape (3, 3)
    """
    rng = np.random.default_rng()  # 12345
    # Choose a distribution for selecting matrices
    # TODO: determine theory for 1/np.sqrt(3) std
    rvs = stats.norm(loc=0.0, scale=1 / np.sqrt(3)).rvs

    # Loop until a stable matrix is found
    while 1:
        # Get a random sparse matrix from the distribution above and convert it to a large dynamics matrix
        local_coefficients = rndm(3, 3, density=density, random_state=rng, data_rvs=rvs)
        local_coefficients = local_coefficients.todense()
        dynamics_matrix = create_coefficient_matrix(local_coefficients, grid_size, fill_value=0.0)

        try:
            if is_stable(dynamics_matrix, verbose - 1):
                min_value_threshold
                eigenvalues, _ = LA.eig(dynamics_matrix)
                operator_norm = np.max(np.abs(eigenvalues))
                min_value_actual = np.min(np.abs(local_coefficients[np.nonzero(local_coefficients)]))

                if min_value_actual >= min_value_threshold:
                    if verbose:
                        print("No scaling is necessary.")
                        print("N = \n{}".format(local_coefficients))
                        print("N is stable: {}".format(is_stable(dynamics_matrix)))
                        print(
                            "min_value_threshold = {}, min_value_actual = {}, operator_norm = {}".format(
                                min_value_threshold, min_value_actual, operator_norm
                            )
                        )
                        print(
                            "Lowest scaled operator norm to achieve min_val_threshold = abs(min_value_threshold/min_value_actual*operator_norm) = {} <- must be less than 1.0".format(
                                np.abs(min_value_threshold / min_value_actual * operator_norm)
                            )
                        )
                    assert (
                        np.abs(min_value_threshold / min_value_actual * operator_norm) < 1
                    ), "min_value_threshold = {}, min_value_actual = {}, operator_norm = {}".format(
                        min_value_threshold, min_value_actual, operator_norm
                    )
                    break

                if np.abs(min_value_threshold / min_value_actual * operator_norm) < 1:
                    if verbose:
                        print("Scaling.")
                        print("N = \n{}".format(local_coefficients))
                        print("N is stable: {}".format(is_stable(dynamics_matrix)))
                        print(
                            "min_value_threshold = {}, min_value_actual = {}, operator_norm = {}".format(
                                min_value_threshold, min_value_actual, operator_norm
                            )
                        )
                        print(
                            "Lowest scaled operator norm to achieve min_val_threshold = abs(min_value_threshold/min_value_actual*operator_norm) = {} <- must be less than 1.0".format(
                                np.abs(min_value_threshold / min_value_actual * operator_norm)
                            )
                        )
                    assert (
                        np.abs(min_value_threshold / min_value_actual * operator_norm) < 1
                    ), "min_value_threshold = {}, min_value_actual = {},operator_norm= {}".format(
                        min_value_threshold, min_value_actual, operator_norm
                    )

                    min_scaling = min_value_threshold / min_value_actual
                    max_scaling = 1 / operator_norm
                    scaling = random.uniform(min_scaling, max_scaling)

                    if verbose:
                        print(
                            "min scaling = {}, max scaling = {}, scaling = {}".format(min_scaling, max_scaling, scaling)
                        )

                    scaled_local_coefficients = scaling * local_coefficients
                    dynamics_matrix_ = create_coefficient_matrix(scaled_local_coefficients, grid_size, fill_value=0.0)

                    if verbose:
                        print("scaled_N = \n{}".format(scaled_local_coefficients))
                        N_stability = is_stable(dynamics_matrix_)
                        print("scaled_N is stable: {}".format(N_stability))
                    assert is_stable(dynamics_matrix_), "scaled_N is not stable!"

                    if verbose:
                        min_val_met = (
                            np.min(np.abs(scaled_local_coefficients[np.nonzero(scaled_local_coefficients)]))
                            >= min_value_threshold
                        )
                        print(
                            "All values > {}: {}".format(
                                min_value_threshold,
                                min_val_met,
                            )
                        )
                        eigenvalues_, _ = LA.eig(dynamics_matrix_)
                        scaled_operator_norm = np.max(np.abs(eigenvalues_))
                        print("Operator norm of scaled_N = {}".format(scaled_operator_norm))

                    scaling_equality = np.all(np.equal(scaled_local_coefficients, (scaling * local_coefficients)))
                    assert scaling_equality, "local_coefficnets was not scaled correctly!"
                    local_coefficients = scaled_local_coefficients
                    break
                continue

        except LA.LinAlgError:
            print("Found LinAlgError, trying eigenvalue decomposition again.")
            continue

    if verbose >= 1:
        print("local coefficients:")
        print(local_coefficients)
    return local_coefficients


def get_graph_from_coefficient_matrix(dynamics_matrix, func=lambda x: x, return_val_matrix=False):
    """Get a TIGRAMITE compatible graph object from a given dynamics matrix.

    First converts the dynamics matrix into a structural causal mode (SCM) according to TIGRAMITE's link dict definition,
    then converts the SCM into a graph.

    Links must be of the form:
        links (dict) - Dictionary of form {0:[((0, -1), coeff, func), ...], 1:[...], ...}. Also format {0:[(0, -1), ...], 1:[...], ...} is allowed.

    More here https://jakobrunge.github.io/tigramite/index.html#module-tigramite.toymodels.structural_causal_processes

    Parameters
    ----------
    dynamics_matrix : numpy.ndarray
        The dynamics matrix defining the dependence between all grid cells in a 2D causal system
    func : function, optional
        A function defining the functional dependence between grid cells. I.e. a linear dependence would be lambda x: x, and
        a quadratic dependence would be lambda x: x**2, by default lambda x: x

    Returns
    -------
    TIGRAMITE graph
        Matrix format of graph with 1 for true links and 0 else

    """

    SCM = {}

    for row in range(dynamics_matrix.shape[0]):
        SCM[row] = [((col, -1), dynamics_matrix[row][col], func) for col in range(dynamics_matrix.shape[1])]

    true_graph = structural_causal_processes.links_to_graph(SCM, tau_max=1)

    if return_val_matrix:
        val_matrix = np.zeros((dynamics_matrix.shape[0], dynamics_matrix.shape[1], 2))
        for row in range(val_matrix.shape[0]):
            for col in range(val_matrix.shape[1]):
                coefficient = SCM[row][col][1]
                val_matrix[col, row, 1] = coefficient
        return true_graph, val_matrix
    else:
        return true_graph


def generate_dataset(
    T,
    grid_size,
    spatial_coefficients=None,
    dependence_density=None,
    min_value=None,
    mode=False,
    error_sigma=0.1,
    error_mean=0,
    verbose=0,
):
    ROWS = grid_size
    COLS = grid_size
    mu, sigma = (error_mean, error_sigma)  # mean and standard deviation

    if spatial_coefficients is None:
        assert (
            dependence_density is not None
        ) & min_value is not None, (
            "Since no spatial_coefficients were passed, dependence_density and min_value must be passed."
        )
        spatial_coefficients = get_random_stable_coefficient_matrix(
            grid_size, dependence_density, min_value_threshold=min_value, verbose=0
        )

    # Initialize data
    data = np.zeros((ROWS, COLS, T))
    if mode:
        print("mode not yet implemented.")
        raise ValueError
        # data[:, :, :, :] = np.random.normal(init_mu, init_sigma, size=(ROWS, COLS, T, n_var))
        # data[(y_pos-int(size/2)):(y_pos+int(size/2)), (x_pos-int(size/2)):(x_pos+int(size/2)), 0, 0] = Z
        # data[(y_pos-int(size/2)):(y_pos+int(size/2)) + 1, (x_pos-int(size/2)):(x_pos+int(size/2)) + 1, 0, 0] = Z

    # Run simulation
    for t in range(1, T):
        for row in range(ROWS):
            for col in range(COLS):
                from_left = spatial_coefficients[1, 0] * data[row, col - 1, t - 1]
                from_right = spatial_coefficients[1, 2] * data[row, (col + 1) % ROWS, t - 1]
                from_top = spatial_coefficients[0, 1] * data[row - 1, col, t - 1]
                from_bottom = spatial_coefficients[2, 1] * data[(row + 1) % ROWS, col, t - 1]

                from_top_left = spatial_coefficients[0, 0] * data[row - 1, col - 1, t - 1]
                from_top_right = spatial_coefficients[0, 2] * data[row - 1, (col + 1) % COLS, t - 1]
                from_bot_left = spatial_coefficients[2, 0] * data[(row + 1) % ROWS, col - 1, t - 1]
                from_bot_right = spatial_coefficients[2, 2] * data[(row + 1) % ROWS, (col + 1) % COLS, t - 1]

                from_self = spatial_coefficients[1, 1] * data[row, col, t - 1]

                data[row, col, t] = (
                    from_self
                    + from_left
                    + from_right
                    + from_top
                    + from_bottom
                    + from_top_left
                    + from_top_right
                    + from_bot_left
                    + from_bot_right
                    + np.random.normal(mu, sigma)
                )  # increase sigma for higher magnitudes

    return data


def get_graph_from_structure_model(structure_model: StructureModel, include_val_matrix=True) -> Union[tuple, list]:
    """Convert a causalnex.structure.StructureModel to a string-graph and val_matrix in the style of TIGRAMITE.

    StructureModel inherits networkx's DiGraph, which this conversion relies upon.

    Args:
        structure_model (StructureModel): causalnex.structure.StructureModel, graph from CausalNex
        include_val_matrix (bool, optional): Whether to return a value matrix, returns only string-graph if False. Defaults to True.

    Returns:
        Union[tuple, list]: A tuple of two lists (string-graph and float-graph) or just a list (string-graph)
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
            lag = child_lag - parent_lag
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
