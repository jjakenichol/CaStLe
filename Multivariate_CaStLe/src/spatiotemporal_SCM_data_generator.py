import numpy as np
from numpy import linalg as LA
import random
import sys
from scipy import stats


def reshape_multivariate_coefs(coefs: np.ndarray) -> np.ndarray:
    """Reshapes an array of multivariate coefficients of shape (n_variables, 3*3*n_variables) into a list of 3x3 coefficient arrays for each child variable, where each entry in the 3x3 array is a list of parent coefficients in the relative position.

    For example, for a 2-variable system, the given coefficient shape must be of the form:
    [
        [a, b, c, d, e, f, g, h, i, A, B, C, D, E, F, G, H, I],
        [r, s, t, u, v, w, x, y, z, R, S, T, U, V, W, X, Y, Z]
    ]
    and the resulting list will be of the form:
    [
        [[a, A], [b, B], [c, C]
         [d, D], [e, E], [f, F]
         [g, G], [h, H], [i, I]],

         [[r, R], [s, S], [t, T]
         [u, U], [v, V], [w, W]
         [x, X], [y, Y], [z, Z]]
    ]

    Args:
        coefs (np.ndarray): An array of multivariate coefficients of shape (n_variables, 3*3*n_variables).

    Returns:
        np.ndarray: list of 3x3 coefficient arrays for each child variable, where each entry in the 3x3 array is a list of parent coefficients in the relative position.
    """
    coef_list = []
    # Loop over all possible child variables in coefs
    for child_idx in range(coefs.shape[0]):
        new_arr = np.empty(shape=(9,), dtype=list)
        # Loop over each coefficient position in the 3x3 (currently (9,)) array
        for coef_idx in range(new_arr.shape[0]):
            # Extract the parent coefficients for the current child variable and coefficient position
            new_arr[coef_idx] = np.array([coefs[child_idx, parent_idx * 9 + coef_idx] for parent_idx in range(coefs.shape[0])])
        coef_list.append(new_arr.reshape((3, 3)))
    return np.array(coef_list)


def full_with_arrays(shape, fill_value):
    """
    Create a new array of given shape, where each entry is filled with arrays of the specified fill value.

    Parameters
    ----------
    shape : tuple of int
        Shape of the new array.
    fill_value : array-like
        Value to fill each entry of the array with.

    Returns
    -------
    ndarray
        Array of the specified shape, where each entry is filled with arrays of the specified fill value.
    """
    # Convert fill_value to a NumPy array
    fill_value = np.asarray(fill_value)
    # Create an empty array of the desired shape
    arr = np.empty(shape, dtype=object)
    # Fill each entry with the specified fill value
    for index in np.ndindex(shape):
        arr[index] = np.full(fill_value.shape, fill_value, dtype=fill_value.dtype)
    return arr


def pad_n_cols_right_of_2d_matrix(arr, n, fill_value):
    """Adds n columns of zeros to right of 2D numpy array matrix.
    https://stackoverflow.com/a/72339584/9969212

    Parameters
    ----------
    arr : np.ndarray
        A two dimensional numpy array that is padded.
    n : int
        The number of columns that are added to the right of the matrix.
    fill_value : float
        Value inserted into empty cells.

    Returns
    -------
    np.ndarray
        Padded array matrix
    """

    # Create a new array with the same number of rows and additional n columns filled with fill_value
    padded_array = full_with_arrays((arr.shape[0], arr.shape[1] + n), fill_value)
    # Copy the original array into the new padded array, leaving the additional columns as zeros
    padded_array[:, : arr.shape[1]] = arr
    return padded_array


def pad_n_rows_below_2d_matrix(arr, n, fill_value):
    """Adds n rows of zeros below 2D numpy array matrix.
    https://stackoverflow.com/a/72339584/9969212

    Parameters
    ----------
    arr : np.ndarray
        A two dimensional numpy array that is padded.
    n : int
        The number of columns that are added to the right of the matrix.
    fill_value : float
        Value inserted into empty cells.

    Returns
    -------
    np.ndarray
        Padded array matrix
    """

    # Create a new array with the same number of columns and additional n rows filled with fill_value
    padded_array = full_with_arrays((arr.shape[0] + n, arr.shape[1]), fill_value)
    # Copy the original array into the new padded array, leaving the additional rows as zeros
    padded_array[: arr.shape[0], :] = arr
    return padded_array


def move_elements_to_end(array, N):
    """
    Moves every Nth row to its respective Nth section of the matrix.

    Parameters
    ----------
    array : numpy.ndarray
        The input array to be sorted.
    N : int
        The value of N for moving rows to their respective sections.

    Returns
    -------
    numpy.ndarray
        A sorted array with rows moved to their respective sections.

    Examples:
    --------
    >>> array = np.array([["A", 2, 3, 4],
                          ["B", 6, 7, 8],
                          ["C", 10, 11, 12],
                          ["A", 14, 15, 16],
                          ["B", 18, 19, 20],
                          ["C", 22, 23, 24],
                          ["A", 26, 27, 28],
                          ["B", 30, 31, 32],
                          ["C", 34, 35, 36]])
    >>> N = 3
    >>> output = move_elements_to_end(array, N)
    >>> print(output)
    [['A' '2' '3' '4']
     ['A' '14' '15' '16']
     ['A' '26' '27' '28']
     ['B' '6' '7' '8']
     ['B' '18' '19' '20']
     ['B' '30' '31' '32']
     ['C' '10' '11' '12']
     ['C' '22' '23' '24']
     ['C' '34' '35' '36']]

    """
    num_sections = array.shape[0] // N
    sorted_array = np.empty_like(array)

    for i in range(num_sections):
        # Assign rows from the original array to their respective sections in the sorted array
        sorted_array[i::num_sections] = array[i * N : (i + 1) * N]

    return sorted_array


def dynamics_matrix_reshaper(six_dim_array: np.ndarray) -> np.ndarray:
    """
    Reshapes a 6D array into a 2D array with the correct shape, as required by the `create_global_dynamics_matrix()` function.

    The `create_global_dynamics_matrix()` function generates a (grid_size**2, grid_size**2) size matrix from a given (3, 3) matrix.
        This reshaping function is used to derive the correct shape of the 2D array that represents the dynamics matrix between all grid cells.

    Parameters
    ----------
    six_dim_array : numpy.ndarray
        The input 6D array representing the dynamics matrix between all grid cells.

    Returns
    -------
    numpy.ndarray
        The reshaped 2D array with the correct shape.

    Example
    -------
    >>> six_dim_array = np.random.random((2, 2, 3, 3, 4, 4))
    >>> reshaped_array = dynamics_matrix_reshaper(six_dim_array)
    """
    num_variables = six_dim_array.shape[0]
    num_rows = six_dim_array.shape[2]
    num_colums = six_dim_array.shape[3]
    # Derive correct shape of 2D array from given 6D array
    two_dim_shape = (
        six_dim_array.shape[0] * six_dim_array.shape[2] * six_dim_array.shape[4],
        six_dim_array.shape[1] * six_dim_array.shape[3] * six_dim_array.shape[5],
    )
    two_dim_array = np.empty(shape=two_dim_shape, dtype=six_dim_array.dtype)

    # Iterate over 6D array, accumulating coefficients in the correct order per child row, and append them to each 3D array row.
    two_dim_row_idx = 0
    for C_child_row in range(num_rows):
        for C_child_col in range(num_colums):
            for child_var in range(num_variables):
                row_values = np.empty(shape=(0,))
                for parent_var in range(num_variables):
                    for C_parent_row in range(num_rows):
                        for C_parent_col in range(num_colums):
                            row_values = np.append(row_values, six_dim_array[child_var, parent_var, C_child_row, C_child_col, C_parent_row, C_parent_col])
                two_dim_array[two_dim_row_idx, :] = row_values
                two_dim_row_idx += 1

    # Sort each Nth row to the Nth fractional part or "proportional division"
    sorted_two_dim_array = move_elements_to_end(two_dim_array, num_variables)

    return sorted_two_dim_array.astype(six_dim_array[0, 0, 0, 0, 0, 0].dtype)


def create_global_dynamics_matrix(local_coefficients, grid_size, n_variables, fill_value=None):
    """A mapping function that generates a (grid_size**2, grid_size**2) size matrix from a given list of (3, 3) matrices.

    The given list of (3, 3) matrices are local dependence coefficient matrices that determine the dependence of
    neighboring grid cells, within a (grid_size, grid_size) toroidal space, relative to the center grid cell
    (the one at position [1, 1]). This function determines the matrix that defines the dependence between every
    grid cell relative to every other grid cell.

    Parameters
    ----------
    local_coefficients : list of np.ndarray
        A list of (3, 3) local dependence coefficient matrices, one for each variable.
    grid_size : int
        The first dimension of the (grid_size, grid_size) toroidal space.
    n_variables : int
        The number of variables in the system (number of spatiotemporal coefficient matrices).
    fill_value : any, optional
        The value inserted for all grid cell pairs without dependence, by default 0.0.

    Returns
    -------
    numpy.ndarray
        The (grid_size**2, grid_size**2) dynamics matrix between all grid cells.
    """
    assert type(local_coefficients) == np.ndarray or type(local_coefficients) == np.matrix, "local_coefficients must be of type numpy.ndarray or numpy.matrix"
    fill_value = [fill_value] * n_variables
    global_dynamics_matrix = full_with_arrays((n_variables, n_variables, grid_size, grid_size, grid_size, grid_size), fill_value)

    pad_width = grid_size - 3
    padded_coef_matrices = []
    for matrix in local_coefficients:
        tmp_matrix = pad_n_cols_right_of_2d_matrix(matrix, pad_width, fill_value)
        tmp_matrix = pad_n_rows_below_2d_matrix(tmp_matrix, pad_width, fill_value)
        padded_coef_matrices.append(tmp_matrix)
    padded_coeff_center = (1, 1)

    for child_var in range(n_variables):
        for parent_var in range(n_variables):
            for C_child_row in range(grid_size):
                for C_child_col in range(grid_size):
                    for C_parent_row in range(grid_size):
                        for C_parent_col in range(grid_size):
                            offset = (C_child_row - C_parent_row, C_child_col - C_parent_col)
                            global_dynamics_matrix[child_var, parent_var, C_child_row, C_child_col, C_parent_row, C_parent_col] = padded_coef_matrices[child_var][
                                (padded_coeff_center[0] - offset[0]) % padded_coef_matrices[child_var].shape[0],
                                (padded_coeff_center[1] - offset[1]) % padded_coef_matrices[child_var].shape[1],
                            ][parent_var]

    return dynamics_matrix_reshaper(global_dynamics_matrix)


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


def flatten_array(arr):
    """
    Recursively flattens a numpy array with nested arrays into a 1D array.

    Parameters:
    arr (numpy.ndarray): The input array to be flattened.

    Returns:
    numpy.ndarray: A 1D array containing all the elements from the original array.
    """
    if isinstance(arr, np.ndarray):
        return np.concatenate([flatten_array(subarr) for subarr in arr])
    else:
        return np.array([arr])


def get_min_coef(local_coeficients):
    """
    Calculates the minimum absolute coefficient value from a numpy array with nested arrays.

    Parameters:
    local_coeficients (numpy.ndarray): The input array containing coefficients.

    Returns:
    float: The minimum absolute coefficient value.
    """
    coefs_flattened = flatten_array(local_coeficients)
    return np.min(np.abs(coefs_flattened[np.nonzero(coefs_flattened)]))


def generate_random_matrix(m, n, density, min_value):
    """
    Generate a random matrix with a specific density and minimum absolute value, with the maximum value = 1.0.

    Parameters:
        m (int): Number of rows in the matrix.
        n (int): Number of columns in the matrix.
        density (float): Density of non-zero elements in the matrix (proportion of non-zero elements).
        min_value (float): Minimum absolute value of the matrix.

    Returns:
        numpy.ndarray: Random matrix with the specified properties.

    """
    num_nonzero = int(m * n * density)  # Number of non-zero elements

    # Generate a random matrix
    matrix = np.zeros((m, n))
    indices = np.random.choice(m * n, num_nonzero, replace=False)
    signs = np.random.choice([-1, 1], size=num_nonzero)
    values = signs * np.random.uniform(min_value, size=num_nonzero)
    matrix.flat[indices] = values

    return matrix


def generate_chain_matrix(
    n_variables: int,
    coefficient_value: float,
    position: tuple = None,
    self_dependency_first: bool = False,
    self_dependency_last: bool = False,
    reverse_order: bool = False,
) -> np.ndarray:
    """
    Generate a chain matrix with a specific link value between variables, optionally including a self-dependency for the first or last variable, and optionally reversing the order of dependencies.

    Parameters:
    -----------
    n_variables : int
        The number of variables.
    coefficient_value : float
        The value of the dependency between variables.
    position : tuple, optional
        Position in the 3x3 neighborhood (row, col). If None, a random position is chosen for each variable.
    self_dependency_first : bool, optional
        If True, adds a self-dependency for the first variable. Defaults to False.
    self_dependency_last : bool, optional
        If True, adds a self-dependency for the last variable. Defaults to False.
    reverse_order : bool, optional
        If True, reverses the order of dependencies. Defaults to False.

    Returns:
    --------
    np.ndarray
        Chain matrix with the specified properties, optionally including a self-dependency for the first or last variable, and optionally reversing the order of dependencies.

    The function initializes the coefficient matrix with zeros and creates a chain of dependencies with the chosen link value.
    If the position is not specified, a random position is chosen for each variable. The first or last variable can be given a self-dependency based on the provided options.
    The order of dependencies can be reversed based on the provided option.
    """
    # Initialize the coefficient matrix with zeros
    coefs = np.zeros((n_variables, 3 * 3 * n_variables))

    # Determine the range for creating dependencies based on the reverse_order flag
    if reverse_order:
        range_start, range_end, range_step = 0, n_variables - 1, 1
    else:
        range_start, range_end, range_step = n_variables - 1, 0, -1

    # Create a chain of dependencies with the chosen link value
    for i in range(range_start, range_end, range_step):
        if position is None:
            pos = (np.random.randint(0, 3), np.random.randint(0, 3))
        else:
            pos = position
        pos_index = pos[0] * 3 + pos[1]
        if reverse_order:
            coefs[i, (i + 1) * 9 + pos_index] = coefficient_value
        else:
            coefs[i, (i - 1) * 9 + pos_index] = coefficient_value

    # Optionally create a self-dependency for the first variable
    if self_dependency_first:
        if position is None:
            self_pos = (np.random.randint(0, 3), np.random.randint(0, 3))
        else:
            self_pos = position
        self_pos_index = self_pos[0] * 3 + self_pos[1]
        coefs[0, 0 * 9 + self_pos_index] = coefficient_value

    # Optionally create a self-dependency for the last variable
    if self_dependency_last:
        if position is None:
            self_pos = (np.random.randint(0, 3), np.random.randint(0, 3))
        else:
            self_pos = position
        self_pos_index = self_pos[0] * 3 + self_pos[1]
        coefs[n_variables - 1, (n_variables - 1) * 9 + self_pos_index] = coefficient_value

    return coefs


def get_stable_coefficient_chain_matrix(
    grid_size: int,
    n_variables: int,
    coefficient_value: float,
    position: tuple = None,
    self_dependency_first: bool = False,
    self_dependency_last: bool = False,
    reverse_order: bool = False,
    check_stability: bool = True,
    verbose=0,
):
    """
    Generates a local dependence coefficient matrix that will produce a stable dynamics matrix of size (grid_size**2, grid_size**2).

    The dynamics matrix is a mapping of the local spatial coefficient matrix onto gridded toroidal space of size
    (grid_size, grid_size), where the dynamics matrix describes the effect of each grid cell on each other grid cell.

    Parameters
    ----------
    grid_size : int
        First dimension of a square spatial grid, i.e., a spatial grid of size (grid_size, grid_size).
    n_variables : int
        The number of variables, or inter-dependent grids, to generate dynamics for.
    coefficient_value : float
        The value of the dependencies between variables in the chain.
    position : tuple, optional
        Position in the 3x3 neighborhood (row, col). If None, a random position is chosen for each variable.
    self_dependency_first : bool, optional
        If True, adds a self-dependency for the first variable. Defaults to False.
    self_dependency_last : bool, optional
        If True, adds a self-dependency for the last variable. Defaults to False.
    reverse_order : bool, optional
        If True, reverses the order of dependencies. Defaults to False.
    verbose : int, optional
        Pass 1 to print the generated local dependence matrix, pass 2 to also print the computed eigenvalues, by default 0.

    Returns
    -------
    numpy.ndarray
        A local dependence coefficient matrix with shape (3 * n_variables, 3 * n_variables), which is stable under the defined conditions and parameters.
    """
    coefs = generate_chain_matrix(n_variables, coefficient_value, position, self_dependency_first, self_dependency_last, reverse_order)
    reshaped_coefs = reshape_multivariate_coefs(coefs)
    dynamics_matrix = create_global_dynamics_matrix(reshaped_coefs, grid_size, n_variables, fill_value=0.0)

    if check_stability:
        assert is_stable(dynamics_matrix, verbose=verbose), "Coefficients are not stable."
    return reshaped_coefs


def get_density(num_links: int, num_variables: int) -> float:
    """
    Calculate the density of links in a network.

    The density is defined as the ratio of the actual number of links
    to the maximum possible number of links in a fully connected network.
    In this specific calculation, the maximum number of links is assumed
    to be proportional to nine times the square of the number of variables,
    which represents the stencil network structure.

    Parameters:
    - num_links (int): The actual number of links present in the network.
    - num_variables (int): The number of variables (nodes) in the network.

    Returns:
    - float: The calculated density of links in the network.

    Example:
    >>> get_density(10, 5)
    0.044444444444444446

    This example calculates the density of a network with 10 links and 5 variables.
    """
    return num_links / (9 * num_variables**2)


def get_num_links(density: float, num_variables: int) -> int:
    """
    Calculate the number of links in a network given its density and the number of variables.

    This function performs the inverse calculation of `get_density`, determining the actual
    number of links in a network based on its density and the number of variables.
    The maximum number of links is assumed to be proportional to nine times the square of
    the number of variables, which represents the stencil network structure. The density
    is the ratio of the actual number of links to this maximum possible number of links.

    Parameters:
    - density (float): The density of links in the network, a value between 0 and 1.
    - num_variables (int): The number of variables (nodes) in the network.

    Returns:
    - int: The calculated number of links present in the network. The result is rounded
      to the nearest integer, as the number of links must be a whole number.

    Example:
    >>> get_num_links(0.044444444444444446, 5)
    10

    This example calculates the number of links in a network with a density of approximately
    0.0444 and 5 variables. The result indicates that there are 10 links in the network.
    """
    # Calculate the maximum possible number of links for the given number of variables
    max_possible_links = 9 * num_variables**2
    # Calculate the actual number of links based on the given density
    num_links = density * max_possible_links
    # Round the result to the nearest whole number, as the number of links must be an integer
    return round(num_links)


def get_empty_coefficient_matrix(n_variables: int):
    """
    Generates an empty local dependence coefficient matrix for a given number of variables.

    This function creates a matrix filled with zeros, which serves as a placeholder for local dependence coefficients in a multivariate system. The matrix is then reshaped to match the expected format for further processing.

    Parameters
    ----------
    n_variables : int
        The number of variables in the system for which the coefficient matrix is being generated.

    Returns
    -------
    numpy.ndarray
        A reshaped local dependence coefficient matrix with shape (3 * n_variables, 3 * n_variables), initialized with zeros.
    """
    matrix = np.zeros((n_variables, 3 * 3 * n_variables))
    local_coefficients = reshape_multivariate_coefs(matrix)
    return local_coefficients


def get_random_stable_coefficient_matrix(
    grid_size: int,
    n_variables: int,
    num_links: int = None,
    density: float = None,
    min_value_threshold: float = None,
    min_val_scaler: float = 1.0,
    verbose=0,
):
    """
    Generates a random local dependence coefficient matrix that will produce a stable dynamics matrix of size (grid_size**2, grid_size**2).

    The dynamics matrix is a mapping of the local spatial coefficient matrix onto gridded toroidal space of size
    (grid_size, grid_size), where the dynamics matrix describes the effect of each grid cell on each other grid cell.

    Parameters
    ----------
    grid_size : int
        First dimension of a square spatial grid, i.e., a spatial grid of size (grid_size, grid_size).
    n_variables : int
        The number of variables, or inter-dependent grids, to generate dynamics for.
    num_links : int, optional
        The number of link dependencies to exist in the stencil dynamics. If num_links is passed, density should be None.
    density : float, optional
        Density of the desired random local dependence matrix. The value must be less than or equal to 1.0. If density is passed, num_links should be None.
    min_value_threshold : float, optional
        The minimum absolute value threshold for the coefficients in the generated matrix. Coefficients with absolute values below this threshold will be considered too weak to contribute to the dynamics and will be adjusted accordingly.
    min_val_scaler : float, optional
        A scaling factor applied to `min_value_threshold` to determine the minimum value for generating random coefficients. This allows for control over the sparsity and magnitude of the coefficients in the generated matrix, by default 1.0.
    verbose : int, optional
        Pass 1 to print the generated random local dependence matrix, pass 2 to also print the computed eigenvalues, by default 0.

    Returns
    -------
    numpy.ndarray
        A random local dependence coefficient matrix with shape (3 * n_variables, 3 * n_variables), which is stable under the defined conditions and parameters.

    Notes
    -----
    - The function iteratively generates random matrices and checks for stability and adherence to the minimum value threshold. It scales the coefficients if necessary to meet the stability and threshold criteria.
    - The `min_value_threshold` and `min_val_scaler` parameters allow for fine-tuning of the generated matrix's properties, ensuring that the dynamics are not only stable but also meet specific criteria for the minimum coefficient values.
    - If both `num_links` and `density` are provided, the function will exit with an error message prompting the user to choose one.
    - The stability of the generated matrix is determined by its eigenvalues, with additional checks and adjustments made based on the `min_value_threshold`.
    """
    # Check for correct input arguments.
    if (num_links is not None) and (density is not None):
        print("Passing both a density and num_links is undefined, pass one or the other. density={}, num_links={}".format(density, num_links))
        sys.exit(1)
    elif num_links is not None:
        density = get_density(num_links=num_links, num_variables=n_variables)
        assert density <= 1.0, "Too many links were requested for the requested number of variables, density ={}.".format(density)
    elif density is not None:
        pass
    else:
        assert density is not None and num_links is not None, "density={} and num_links={}. Must pass either density or num_links.".format(density, num_links)
    assert isinstance(min_value_threshold, float), f"min_value_threshold ({min_value_threshold}) must be passed as a float."

    pass_condition = -1
    # Loop until a stable matrix is found
    while 1:
        # Get a random sparse matrix from the distribution above and convert it to a large dynamics matrix
        # local_coefficients = rndm(n_variables, 3*3*n_variables, density=density, random_state=rng, data_rvs=rvs)
        # local_coefficients = local_coefficients.todense()
        local_coefficients = generate_random_matrix(n_variables, 3 * 3 * n_variables, density=density, min_value=min_val_scaler * min_value_threshold)
        min_value_actual = np.min(np.abs(local_coefficients[np.nonzero(local_coefficients)]))
        local_coefficients = reshape_multivariate_coefs(local_coefficients)
        dynamics_matrix = create_global_dynamics_matrix(local_coefficients, grid_size, n_variables=n_variables, fill_value=0.0)

        eigenvalues, _ = LA.eig(dynamics_matrix)
        operator_norm = np.max(np.abs(eigenvalues))

        try:
            if is_stable(dynamics_matrix, verbose - 2):
                if min_value_actual >= min_value_threshold:
                    if verbose:
                        print("No scaling is necessary.")
                        print("N = \n{}".format(local_coefficients))
                        print("N is stable: {}".format(is_stable(dynamics_matrix)))
                        print("min_value_threshold = {}, min_value_actual = {}, operator_norm = {}".format(min_value_threshold, min_value_actual, operator_norm))
                        print(
                            "Lowest scaled operator norm to achieve min_val_threshold = abs(min_value_threshold/min_value_actual*operator_norm) = {} <- must be less than 1.0".format(
                                np.abs(min_value_threshold / min_value_actual * operator_norm)
                            )
                        )
                    assert np.abs(min_value_threshold / min_value_actual * operator_norm) < 1, "min_value_threshold = {}, min_value_actual = {}, operator_norm = {}".format(
                        min_value_threshold, min_value_actual, operator_norm
                    )
                    pass_condition = 1
                    break

                if np.abs(min_value_threshold / min_value_actual * operator_norm) < 1:
                    if verbose:
                        print("Scaling.")
                        print("N = \n{}".format(local_coefficients))
                        print("N is stable: {}".format(is_stable(dynamics_matrix)))
                        print("min_value_threshold = {}, min_value_actual = {}, operator_norm = {}".format(min_value_threshold, min_value_actual, operator_norm))
                        print(
                            "Lowest scaled operator norm to achieve min_val_threshold = abs(min_value_threshold/min_value_actual*operator_norm) = {} <- must be less than 1.0".format(
                                np.abs(min_value_threshold / min_value_actual * operator_norm)
                            )
                        )
                    assert np.abs(min_value_threshold / min_value_actual * operator_norm) < 1, "min_value_threshold = {}, min_value_actual = {},operator_norm= {}".format(
                        min_value_threshold, min_value_actual, operator_norm
                    )

                    min_scaling = min_value_threshold / min_value_actual
                    max_scaling = 1 / operator_norm
                    scaling = random.uniform(min_scaling, max_scaling)

                    if verbose:
                        print("min scaling = {}, max scaling = {}, scaling = {}".format(min_scaling, max_scaling, scaling))

                    scaled_local_coefficients = scaling * local_coefficients
                    dynamics_matrix_ = create_global_dynamics_matrix(scaled_local_coefficients, grid_size, n_variables=n_variables, fill_value=0.0)

                    if verbose:
                        print("scaled_N = \n{}".format(scaled_local_coefficients))
                        N_stability = is_stable(dynamics_matrix_)
                        print("scaled_N is stable: {}".format(N_stability))
                    assert is_stable(dynamics_matrix_), "scaled_N is not stable!"
                    local_coefficients = scaled_local_coefficients

                    if verbose:
                        eigenvalues_, _ = LA.eig(dynamics_matrix_)
                        scaled_operator_norm = np.max(np.abs(eigenvalues_))
                        print("Operator norm of scaled_N = {}".format(scaled_operator_norm))
                    pass_condition = 2
                    break
                continue
            else:
                min_value_actual = get_min_coef(local_coefficients)
                max_scaling = min_value_threshold / min_value_actual
                min_scaling = 1 / operator_norm
                scaling = min_scaling  # random.uniform(min_scaling, max_scaling)
                scaled_local_coefficients = local_coefficients * scaling

                min_coef = get_min_coef(scaled_local_coefficients)
                scaled_dynamics_matrx = create_global_dynamics_matrix(scaled_local_coefficients, grid_size, n_variables, fill_value=0.0)

                if is_stable(scaled_dynamics_matrx):
                    if min_coef > min_value_threshold:
                        if verbose > 1:
                            print("Scaling with operator norm worked, and min_coef={}".format(min_coef))
                        local_coefficients = scaled_local_coefficients
                        pass_condition = 3
                        break
                    else:
                        if verbose > 1:
                            print("Scaling with operator norm worked, but min_coef={}".format(min_coef))

        except LA.LinAlgError:
            print("Found LinAlgError, trying eigenvalue decomposition again.")
            continue

    if verbose >= 1:
        print("local coefficients:")
        print(local_coefficients)
        dynamics_matrix = create_global_dynamics_matrix(local_coefficients, grid_size, n_variables, fill_value=0.0)
        eigenvalues_, _ = LA.eig(dynamics_matrix)
        operator_norm = np.max(np.abs(eigenvalues_))
        min_coef = get_min_coef(local_coefficients)
        print("Max Eigenvalue={}".format(operator_norm))
        print("Min Coefficient={}".format(min_coef))
        print("Pass Condition={}".format(pass_condition))
    return local_coefficients


def generate_dataset(
    T: int,
    grid_size: int,
    spatial_coefs: np.ndarray = None,
    dependence_density: float = None,
    num_links: int = None,
    num_variables: int = None,
    coefficient_min_value_threshold: float = None,
    min_val_scaler: float = 1.0,
    error_sigma: float = 0.1,
    error_mean: float = 0.0,
    initialize_randomly: bool = False,
    detect_instability: bool = False,
    instability_threshold: float = 1000,
    return_coefs: bool = False,
    random_seed: int = None,
    verbose: int = 0,
) -> np.ndarray:
    """
    Simulates spatial data over time, given a set of parameters that define the spatial interactions and the dynamics of the system.
    Optionally generates and returns the spatial coefficients used in the simulation if they are not provided.

    Parameters
    ----------
    T : int
        The number of time steps for which to generate data.
    grid_size : int
        The size of the spatial grid (assumed square) for the simulation.
    spatial_coefs : np.ndarray, optional
        A pre-defined array of spatial coefficients. If not provided, they will be generated based on other parameters.
    dependence_density : float, optional
        The density of dependencies in the spatial coefficient matrix. Used if spatial_coefs is not provided.
    num_links : int, optional
        The number of links or dependencies to be considered in the spatial dynamics. Used if spatial_coefs is not provided.
    num_variables : int, optional
        The number of variables (or layers) in the spatial grid. Required if spatial_coefs is not provided.
    coefficient_min_value_threshold : float, optional
        The minimum value threshold for coefficients in the generated spatial coefficient matrix.
    min_val_scaler : float, optional
        A scaling factor applied to the minimum value threshold for generating coefficients. Defaults to 1.0.
    error_sigma : float, optional
        The standard deviation of the noise added to each cell at each time step. Defaults to 0.1.
    error_mean : float, optional
        The mean of the noise added to each cell at each time step. Defaults to 0.0.
    initialize_randomly : bool, optional
        Whether the dataset should be initialized randomly. If False, it is initialized at zero. Defaults to False.
    detect_instability : bool, optional
        If True, the function will check for instability in the generated data and attempt to regenerate coefficients if instability is detected. Defaults to False.
    instability_threshold : int, optional
        The threshold value beyond which the data is considered unstable. Used if detect_instability is True. Defaults to 1000.
    return_coefs : bool, optional
        If True, the function will return both the generated spatial coefficients and the data. Defaults to False.
    random_seed : int, optional
        Random seed for reproducibility. Controls coefficient generation, initialization, and noise.
        If None, results will vary between runs. Defaults to None.
    verbose : int, optional
        Controls the verbosity of the function's output. A higher value results in more detailed messages. Defaults to 0.

    Returns
    -------
    np.ndarray or tuple
        If return_coefs is False, returns an array of shape (num_variables, grid_size, grid_size, T) containing the generated data.
        If return_coefs is True, returns a tuple where the first element is the spatial coefficients array and the second element is the data array.

    Notes
    -----
    - The function generates spatial data by simulating interactions across a grid over time, incorporating both spatial dependencies (defined by spatial_coefs or generated based on dependence_density and num_links) and random noise.
    - If spatial_coefs is not provided, the function will generate a stable coefficient matrix based on the provided parameters. This requires num_variables, dependence_density or num_links, and coefficient_min_value_threshold to be specified.
    - The function can detect and attempt to correct for instability in the generated data if detect_instability is set to True. This involves regenerating the spatial coefficients and restarting the data generation process if the data exceeds the instability_threshold.
    - The function's behavior and output can be customized using the optional parameters, allowing for control over the complexity and characteristics of the generated data.
    - NEW: random_seed parameter allows for reproducible results. Set to an integer for reproducibility or None for random behavior.
    """

    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    # Check for correct input arguments.
    if detect_instability:
        assert (
            (dependence_density is not None or num_links is not None) and coefficient_min_value_threshold is not None and num_variables is not None
        ), "dependence_density={}, num_links={}, coefficient_min_value_threshold={}, min_val_scaler={}, num_variables={}".format(
            dependence_density, num_links, coefficient_min_value_threshold, min_val_scaler, num_variables
        )

    # Check for correct input arguments.
    if spatial_coefs is None:
        assert (
            (dependence_density is not None or num_links is not None) and coefficient_min_value_threshold is not None and min_val_scaler is not None and num_variables is not None
        ), "dependence_density={}, num_links={}, coefficient_min_value_threshold={}, min_val_scaler={}, num_variables={}".format(
            dependence_density, num_links, coefficient_min_value_threshold, min_val_scaler, num_variables
        )
        assert (num_links is None) ^ (dependence_density is None), "Passing both a density and num_links is undefined, pass one or the other. density={}, num_links={}".format(
            dependence_density, num_links
        )
        if num_links is not None:
            dependence_density = get_density(num_links=num_links, num_variables=num_variables)
            assert dependence_density <= 1.0, "Too many links were requested for the requested number of variables, density ={}.".format(dependence_density)
        elif dependence_density is not None:
            pass
        else:
            assert dependence_density is not None and num_links is not None, "density={} and num_links={}. Must pass either density or num_links.".format(
                dependence_density, num_links
            )

        spatial_coefs = get_random_stable_coefficient_matrix(
            grid_size,
            n_variables=num_variables,
            density=dependence_density,
            min_value_threshold=coefficient_min_value_threshold,
            min_val_scaler=min_val_scaler,
            verbose=verbose,
        )
    else:
        num_variables = spatial_coefs.shape[0]

    ROWS = grid_size
    COLS = grid_size

    exceeded_threshold = True
    while exceeded_threshold:
        exceeded_threshold = False
        # Initialize data
        if initialize_randomly:
            data = np.random.randn(num_variables, ROWS, COLS, T)
        else:
            data = np.zeros((num_variables, ROWS, COLS, T))
        # Run simulation
        for child_var in range(num_variables):
            for t in range(1, T):
                for row in range(ROWS):
                    for col in range(COLS):
                        from_top_lefts = []
                        from_tops = []
                        from_top_rights = []
                        from_lefts = []
                        from_centers = []
                        from_rights = []
                        from_bot_lefts = []
                        from_bottoms = []
                        from_bot_rights = []
                        for parent_var in range(num_variables):
                            from_top_lefts.append(spatial_coefs[child_var, 0, 0][parent_var] * data[parent_var, row - 1, col - 1, t - 1])
                            from_tops.append(spatial_coefs[child_var, 0, 1][parent_var] * data[parent_var, row - 1, col, t - 1])
                            from_top_rights.append(spatial_coefs[child_var, 0, 2][parent_var] * data[parent_var, row - 1, (col + 1) % COLS, t - 1])
                            from_lefts.append(spatial_coefs[child_var, 1, 0][parent_var] * data[parent_var, row, col - 1, t - 1])
                            from_centers.append(spatial_coefs[child_var, 1, 1][parent_var] * data[parent_var, row, col, t - 1])
                            from_rights.append(spatial_coefs[child_var, 1, 2][parent_var] * data[parent_var, row, (col + 1) % COLS, t - 1])
                            from_bot_lefts.append(spatial_coefs[child_var, 2, 0][parent_var] * data[parent_var, (row + 1) % ROWS, col - 1, t - 1])
                            from_bottoms.append(spatial_coefs[child_var, 2, 1][parent_var] * data[parent_var, (row + 1) % ROWS, col, t - 1])
                            from_bot_rights.append(spatial_coefs[child_var, 2, 2][parent_var] * data[parent_var, (row + 1) % ROWS, (col + 1) % COLS, t - 1])
                        from_lefts = np.sum(from_lefts)
                        from_rights = np.sum(from_rights)
                        from_tops = np.sum(from_tops)
                        from_bottoms = np.sum(from_bottoms)
                        from_top_lefts = np.sum(from_top_lefts)
                        from_top_rights = np.sum(from_top_rights)
                        from_bot_lefts = np.sum(from_bot_lefts)
                        from_bot_rights = np.sum(from_bot_rights)
                        from_centers = np.sum(from_centers)

                        data[child_var, row, col, t] = (
                            from_centers
                            + from_lefts
                            + from_rights
                            + from_tops
                            + from_bottoms
                            + from_top_lefts
                            + from_top_rights
                            + from_bot_lefts
                            + from_bot_rights
                            + np.random.normal(error_mean, error_sigma)
                        )  # increase sigma for higher magnitudes

                        # To understand the break/continue structure below, see https://stackoverflow.com/a/654002
                        if detect_instability:
                            if data[child_var, row, col, t] > instability_threshold:
                                # assert spatial_coefs is None, "Spatial coefficients passed are unstable."
                                if verbose:
                                    print("Instability threshold {} exceeded, recalculating coefficients.".format(instability_threshold))
                                spatial_coefs = get_random_stable_coefficient_matrix(
                                    grid_size,
                                    n_variables=num_variables,
                                    density=dependence_density,
                                    min_value_threshold=coefficient_min_value_threshold,
                                    verbose=verbose,
                                )
                                if verbose:
                                    print("Coefficients recalculated. Attempting data generation again.")
                                exceeded_threshold = True
                                break
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                continue
            break
        # The above else, continue, break logic can be explained by:
        # These lines ensure that if the instability_threshold is exceeded at any point, the function will break out of all nested loops and restart the data generation process with new spatial coefficients.
        # Flow Control: The break statements exit the current loop, while the else clauses with continue statements ensure that the function only proceeds to the next iteration of the outer loop if the inner loop completes without a break.

    if return_coefs:
        return spatial_coefs, data
    return data
