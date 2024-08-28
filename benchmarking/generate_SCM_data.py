"""
This script generates spatiotemporal data based on a 2D structural causal model using vector autoregression.

The script utilizes the stable_SCM_generator module to generate random stable coefficient matrices based on the specified parameters.
It then initializes the data array and runs a simulation loop to generate the spatiotemporal data by applying the coefficient matrix
to the previous time step's data, incorporating noise.

The script takes command-line arguments to specify various parameters such as the number of time samples (T),
the dimension of the square grid (GRID_SIZE), the density of the desired coefficient matrix (DEPENDENCE_DENSITY),
the minimum value of the coefficient matrix (MIN_VALUE), the standard deviation of the added noise in simulation (ERROR_SIGMA),
the number of experimental repetitions (NUM_REPETITION), the save path prefix for the output file (SAVE_PATH_PREFIX),
and the verbosity level (VERBOSE).

The generated data is saved to a file in the NumPy binary format (.npy) with a unique filename based on the specified parameters.
If no save path prefix is provided, the default save path is used.

Usage:
python generate_SCM_data.py --t <number_of_time_samples> --grid_size <dimension_of_square_grid> --dependence_density <density_of_coefficient_matrix>
                      --min_value <minimum_value_of_coefficient_matrix> --error_sigma <standard_deviation_of_noise> [--num_repetition <number_of_repetitions>]
                      [--save_path_prefix <save_path_prefix>] [--verbose <verbosity_level>]
"""

import argparse
import numpy as np
import os
import sys
import uuid

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import stable_SCM_generator as scm_gen

parser = argparse.ArgumentParser()
parser.add_argument("--t", type=int, required=True)
parser.add_argument("--grid_size", type=int, required=True)
parser.add_argument("--dependence_density", type=float, required=True)
parser.add_argument("--min_value", type=float, required=True)
parser.add_argument("--error_sigma", type=float, required=True)
parser.add_argument("--num_repetition", type=int, required=False, default=1)
parser.add_argument("--save_path_prefix", type=str, required=False)
parser.add_argument("--verbose", type=int, required=False)
args = parser.parse_args()

T = args.t  # Number of time samples
GRID_SIZE = args.grid_size  # Dimension of square grid
DEPENDENCE_DENSITY = (
    args.dependence_density
)  # Density of the desired coefficient matrix
MIN_VALUE = args.min_value  # Minimum value of the coefficient matrix
ERROR_SIGMA = args.error_sigma  # Standard deviation of the added noise in simulation
ERROR_MEAN = 0  # Mean of the added noise in simulation
N_VAR = 1  # Number of variables

NUM_REPETITION = args.num_repetition  # Number of experimental this repetition
SAVE_PATH_PREFIX = args.save_path_prefix
VERBOSE = args.verbose
if not VERBOSE:
    VERBOSE = 0

ROWS = GRID_SIZE
COLS = GRID_SIZE

rng = np.random.default_rng()  # 12345

init_mu, init_sigma = (ERROR_MEAN, ERROR_SIGMA)  # mean and standard deviation
mu, sigma = (ERROR_MEAN, ERROR_SIGMA)  # mean and standard deviation
data = np.zeros((ROWS, COLS, T, N_VAR))

spatial_coefficients = scm_gen.get_random_stable_coefficient_matrix(
    GRID_SIZE, DEPENDENCE_DENSITY, min_value_threshold=MIN_VALUE, verbose=0
)

# Initialize data
data = np.zeros((ROWS, COLS, T, N_VAR))

# Run simulation
for t in range(1, T):
    for row in range(ROWS):
        for col in range(COLS):
            from_left = spatial_coefficients[1, 0] * data[row, col - 1, t - 1, 0]
            from_right = (
                spatial_coefficients[1, 2] * data[row, (col + 1) % ROWS, t - 1, 0]
            )
            from_top = spatial_coefficients[0, 1] * data[row - 1, col, t - 1, 0]
            from_bottom = (
                spatial_coefficients[2, 1] * data[(row + 1) % ROWS, col, t - 1, 0]
            )

            from_top_left = (
                spatial_coefficients[0, 0] * data[row - 1, col - 1, t - 1, 0]
            )
            from_top_right = (
                spatial_coefficients[0, 2] * data[row - 1, (col + 1) % COLS, t - 1, 0]
            )
            from_bot_left = (
                spatial_coefficients[2, 0] * data[(row + 1) % ROWS, col - 1, t - 1, 0]
            )
            from_bot_right = (
                spatial_coefficients[2, 2]
                * data[(row + 1) % ROWS, (col + 1) % COLS, t - 1, 0]
            )

            from_self = spatial_coefficients[1, 1] * data[row, col, t - 1, 0]

            data[row, col, t, 0] = (
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

# Save to file
if not SAVE_PATH_PREFIX:
    SAVE_PATH = (
        "../data/"
        + str(GRID_SIZE)
        + "x"
        + str(GRID_SIZE)
        + "_"
        + str(ERROR_SIGMA)
        + "sigma_"
        + str(DEPENDENCE_DENSITY)
        + "density_"
        + str(MIN_VALUE)
        + "minval_"
        + str(uuid.uuid4().hex)
        + ".npy"
    )
else:
    SAVE_PATH = (
        SAVE_PATH_PREFIX
        + str(GRID_SIZE)
        + "x"
        + str(GRID_SIZE)
        + "_"
        + str(T)
        + "T_"
        + str(ERROR_SIGMA)
        + "sigma_"
        + str(DEPENDENCE_DENSITY)
        + "density_"
        + str(MIN_VALUE)
        + "minval_"
        + str(uuid.uuid4().hex)
        + ".npy"
    )
if VERBOSE:
    print("Saving data to " + SAVE_PATH + str("\n"))
simulation_object = np.array([spatial_coefficients, data], dtype=object)
with open(SAVE_PATH, "wb") as f:
    np.save(f, simulation_object)
