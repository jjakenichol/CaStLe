"""_summary_
"""
import argparse
import uuid
import numpy as np
import DSAVAR as ds

parser = argparse.ArgumentParser()
parser.add_argument("--t", type=int, required=True)
parser.add_argument("--grid_size", type=int, required=True)
parser.add_argument("--dependence_density", type=float, required=True)
parser.add_argument("--min_value", type=float, required=True)
parser.add_argument("--mode", action=argparse.BooleanOptionalAction)
parser.add_argument("--error_sigma", type=float, required=True)
parser.add_argument("--num_repetition", type=int, required=False, default=1)
parser.add_argument("--save_path_prefix", type=str, required=False)
parser.add_argument("--verbose", type=int, required=False)
args = parser.parse_args()

T = args.t  # Number of time samples
GRID_SIZE = args.grid_size  # Dimension of square grid
DEPENDENCE_DENSITY = args.dependence_density  # Density of the desired coefficient matrix
MIN_VALUE = args.min_value  # Minimum value of the coefficient matrix
MODE = args.mode  # Whether to initialize the field with a mode
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

spatial_coefficients = ds.get_random_stable_coefficient_matrix(
    GRID_SIZE, DEPENDENCE_DENSITY, min_value_threshold=MIN_VALUE, verbose=0
)

# Initialize data
data = np.zeros((ROWS, COLS, T, N_VAR))
if MODE:
    data[:, :, :, :] = np.random.normal(init_mu, init_sigma, size=(ROWS, COLS, T, N_VAR))
    # data[(y_pos-int(size/2)):(y_pos+int(size/2)), (x_pos-int(size/2)):(x_pos+int(size/2)), 0, 0] = Z
    # data[(y_pos-int(size/2)):(y_pos+int(size/2)) + 1, (x_pos-int(size/2)):(x_pos+int(size/2)) + 1, 0, 0] = Z

# Run simulation
for t in range(1, T):
    for row in range(ROWS):
        for col in range(COLS):
            from_left = spatial_coefficients[1, 0] * data[row, col - 1, t - 1, 0]
            from_right = spatial_coefficients[1, 2] * data[row, (col + 1) % ROWS, t - 1, 0]
            from_top = spatial_coefficients[0, 1] * data[row - 1, col, t - 1, 0]
            from_bottom = spatial_coefficients[2, 1] * data[(row + 1) % ROWS, col, t - 1, 0]

            from_top_left = spatial_coefficients[0, 0] * data[row - 1, col - 1, t - 1, 0]
            from_top_right = spatial_coefficients[0, 2] * data[row - 1, (col + 1) % COLS, t - 1, 0]
            from_bot_left = spatial_coefficients[2, 0] * data[(row + 1) % ROWS, col - 1, t - 1, 0]
            from_bot_right = (
                spatial_coefficients[2, 2] * data[(row + 1) % ROWS, (col + 1) % COLS, t - 1, 0]
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
        "../../data/DSAVAR/"
        + str(GRID_SIZE)
        + "x"
        + str(GRID_SIZE)
        + "_"
        + str(ERROR_SIGMA)
        + "sigma_"
        + str(DEPENDENCE_DENSITY)
        + "density_"
        + str(MIN_VALUE)
        + "minval_wMode-"
        + str(MODE)
        + "_"
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
        + "minval_wMode-"
        + str(MODE)
        + "_"
        + str(uuid.uuid4().hex)
        + ".npy"
    )
if VERBOSE:
    print("Saving data to " + SAVE_PATH + str("\n"))
simulation_object = np.array([spatial_coefficients, data], dtype=object)
with open(SAVE_PATH, "wb") as f:
    np.save(f, simulation_object)
