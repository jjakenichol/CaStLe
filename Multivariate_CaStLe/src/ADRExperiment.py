"""
ADRExperiment Module

This module defines the ADRExperiment class, which is used to manage and run
Advection-Diffusion-Reaction (ADR) experiments. The class provides methods to
initialize experiment parameters, run the experiment, save and load experiment
results, generate filenames based on parameters, animate the solution, and run
sweeping experiments with different parameters.
"""

import pickle
import numpy as np
import os
import re
import sys
import time
import warnings
from os import path
from typing import List

from matlabPDE import MatlabPDE


class ADRExperiment:
    """
    Class to manage and run Advection-Diffusion-Reaction (ADR) experiments.
    """

    def __init__(
        self,
        mesh_shape: str = "square",
        init_center: List[float] = None,
        plume_size: int = 50,
        diff_coeffs: List[float] = None,
        advection_coeffs: List[float] = None,
        velocity_angle: float = None,
        velocity_magnitude: float = None,
        velocity_field_type: str = None,
        velocity_parameters: List[float] = None,
        react_rate: float = 2.0,
        reaction_scaling: float = 1.0,
        init_concentration: float = 50,
        t: List[float] = None,
        H: float = 0.02,
        radius: float = 1.0,
        capture_apothem: float = 1.0,
        capture_N: int = 100,
        parallel_interpolation: bool = True,
        plot: bool = False,
        skip_initial_params=False,
        verbose: bool = False,
    ):
        if not skip_initial_params:
            # Validation for velocity parameters
            if velocity_parameters is not None and (velocity_angle is not None or velocity_magnitude is not None):
                raise ValueError("velocity_parameters cannot be passed along with velocity_angle or velocity_magnitude.")
            if (velocity_angle is not None) != (velocity_magnitude is not None):
                raise ValueError("Both velocity_angle and velocity_magnitude must be provided together.")

            if velocity_parameters is not None:
                # Validation for velocity parameters
                if velocity_field_type == "constant" and (len(velocity_parameters) != 2):
                    raise ValueError("For 'constant' velocity_field_type, velocity_parameters must be a list of two values.")
                if velocity_field_type == "sinusoidal" and (len(velocity_parameters) != 2):
                    raise ValueError("For 'sinusoidal' velocity_field_type, velocity_parameters must be a list of two values.")

            self.mesh_shape = mesh_shape
            self.init_center = init_center if init_center is not None else [-0.5, 0.5]
            self.plume_size = plume_size
            self.diff_coeffs = diff_coeffs if diff_coeffs is not None else [0.05, 0.05]
            self.advection_coeffs = advection_coeffs
            self.velocity_angle = velocity_angle
            self.velocity_magnitude = velocity_magnitude
            self.velocity_field_type = velocity_field_type
            self.velocity_parameters = velocity_parameters
            self.react_rate = react_rate
            self.reaction_scaling = reaction_scaling
            self.init_concentration = init_concentration
            self.t = t if t is not None else np.linspace(0, 0.4, 101).tolist()
            self.H = H
            self.radius = radius
            self.capture_apothem = capture_apothem
            self.capture_N = capture_N
            self.parallel_interpolation = parallel_interpolation

            # Validate and set velocity parameters
            self._set_velocity_parameters()
        self.plot = plot
        self.solution = None
        self.pde_solver = MatlabPDE()
        self.verbose = verbose

    def _set_velocity_parameters(self):
        """
        Validate and set velocity parameters.
        """
        velocity_parameters = self.velocity_parameters
        velocity_angle = self.velocity_angle
        velocity_magnitude = self.velocity_magnitude

        if velocity_parameters is not None:
            if velocity_angle is not None or velocity_magnitude is not None:
                raise ValueError("velocity_parameters cannot be passed along with velocity_angle or velocity_magnitude.")
        else:
            if (velocity_angle is not None) != (velocity_magnitude is not None):
                raise ValueError("Both velocity_angle and velocity_magnitude must be provided together.")
            if velocity_angle is not None and velocity_magnitude is not None:
                self.velocity_parameters = [float(velocity_magnitude * np.cos(np.radians(velocity_angle))), float(velocity_magnitude * np.sin(np.radians(velocity_angle)))]
            else:
                warnings.warn("Neither velocity_parameters nor both velocity_angle and velocity_magnitude are provided. Setting velocity_parameters to None.")
                self.velocity_parameters = None

    def run_adr_experiment(self, cached_files_dir: str = None) -> np.ndarray:
        """
        Run the ADR experiment using the provided parameters or load from cache if available.

        Args:
            cached_files_dir (str, optional): Directory to check for cached experiment results. Defaults to None.

        Returns:
            np.ndarray: Solution array of shape (Xs, Ys, time, species), e.g., (100, 100, 101, 2),
                        representing the species concentrations over time and space.
        """
        filename = self.generate_filename()
        cached_file_path = path.join(cached_files_dir, filename) if cached_files_dir else None

        if cached_files_dir and path.exists(cached_file_path):
            print(f"Loading cached experiment results from {cached_file_path}")
            experiment = self.load_results(cached_file_path)

            # Check for corrupted data.
            if experiment.solution.size == 0:
                print("ADRExperiment Warning: cached experiment has an empty solution. Running experiment again.")
            elif experiment.solution.size == 1 and experiment.solution.item() is None:
                print("ADRExperiment Warning: cached experiment has a None solution. Running experiment again.")
            else:
                self.solution = experiment.solution
                return self.solution

        print("Running ADR experiment...")
        matlab_path = "ADR_Driver_func.m"

        self.pde_solver.start_engine()
        if self.verbose:
            print("MATLAB engine started.")
            print(self.pretty_print_parameters())

        out = self.pde_solver.run_script(
            matlab_path,
            self.mesh_shape,
            self.init_center,
            self.plume_size,
            self.diff_coeffs,
            self.advection_coeffs,
            self.velocity_field_type,
            self.velocity_parameters,
            self.react_rate,
            self.reaction_scaling,
            self.init_concentration,
            self.t,
            self.H,
            self.radius,
            self.capture_apothem,
            self.capture_N,
            self.parallel_interpolation,
            self.plot,
            self.verbose,
            nargout=1,
        )

        self.solution = np.array(out)
        if self.verbose:
            print("Python: Data received from MATLAB.")

        if cached_files_dir:
            self.save_results(cached_file_path)

        return self.solution

    def save_results(self, path_to_save: str) -> None:
        """
        Save the ADRExperiment object to a file.

        Args:
            path_to_save (str): The path to save the ADRExperiment object.
        """
        try:
            with open(path_to_save, "wb") as f:
                pickle.dump(self, f)
            print(f"ADRExperiment object saved to {path_to_save}")
        except (pickle.PicklingError, IOError) as e:
            print(f"An error occurred while saving the ADRExperiment object: {e}.")

    @staticmethod
    def load_results(path_to_load: str, verbose=False) -> "ADRExperiment":
        """
        Load the ADRExperiment object from a file.

        Args:
            path_to_load (str): The path to load the ADRExperiment object from.

        Returns:
            ADRExperiment: The loaded ADRExperiment object.
        """
        try:
            with open(path_to_load, "rb") as f:
                experiment = pickle.load(f)
            if verbose:
                print(f"ADRExperiment object loaded from {path_to_load}.")
            return experiment
        except (pickle.UnpicklingError, IOError) as e:
            print(f"An error occurred while loading the ADRExperiment object: {e}.")
            return None

    def __getstate__(self):
        """
        Prepare the object state for pickling.
        """
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state["pde_solver"]
        return state

    def __setstate__(self, state):
        """
        Restore the object state from the unpickled state.
        """
        self.__dict__.update(state)
        # Restore the pde_solver attribute.
        self.pde_solver = MatlabPDE()

    def generate_filename(self) -> str:
        """
        Generate a consistent filename based on experimental parameters.

        Returns:
            str: The generated filename.
        """
        # Start with the base prefix
        filename_parts = ["ADR"]

        # Add parameters in a consistent order
        filename_parts.extend(
            [
                str(self.mesh_shape),
                str(self.init_center[0]),
                str(self.init_center[1]),
                str(self.plume_size),
                str(self.diff_coeffs[0]),
                str(self.diff_coeffs[1]),
            ]
        )

        # Add advection coefficients
        if self.advection_coeffs is not None:
            filename_parts.extend(
                [
                    str(self.advection_coeffs[0]),
                    str(self.advection_coeffs[1]),
                ]
            )
        else:
            filename_parts.extend(["0.0", "0.0"])

        # Add velocity field type and parameters
        filename_parts.append(str(self.velocity_field_type) if self.velocity_field_type else "none")
        if self.velocity_parameters is not None:
            filename_parts.extend(
                [
                    str(self.velocity_parameters[0]),
                    str(self.velocity_parameters[1]),
                ]
            )
        else:
            filename_parts.extend(["0.0", "0.0"])

        # Add reaction parameters
        filename_parts.extend(
            [
                str(self.react_rate),
                str(self.reaction_scaling),
            ]
        )

        # Add H, radius, capture parameters
        filename_parts.extend(
            [
                str(self.H),
                str(self.radius),
                str(self.capture_apothem),
                str(self.capture_N),
                str(self.init_concentration),
            ]
        )

        # Add time parameters at the end
        if self.t is not None:
            filename_parts.extend(
                [
                    str(self.t[0]),
                    str(self.t[-1]),
                    str(len(self.t)),
                ]
            )
        else:
            filename_parts.extend(["0.0", "0.0", "0"])

        # Join with underscores and add extension
        filename = "_".join(filename_parts) + ".pkl"
        return filename

    def is_experiment_cached(self, params, cached_files_dir):
        """
        Check if an experiment with given parameters exists in cache.

        Args:
            params (dict): Dictionary of parameters for the experiment.
            cached_files_dir (str): Directory to check for cached experiment results.

        Returns:
            tuple: (bool, str) - (True if cached, path to cached file) or (False, expected path)
        """
        if not cached_files_dir or not os.path.exists(cached_files_dir):
            return False, None

        # Create a temporary experiment to generate the filename
        temp_experiment = ADRExperiment(**params)
        expected_filename = temp_experiment.generate_filename()
        expected_path = os.path.join(cached_files_dir, expected_filename)

        # Check if the file exists
        if os.path.exists(expected_path):
            return True, expected_path

        return False, expected_path

    def debug_filename_generation(self, param_sweeps, cached_files_dir):
        """
        Debug helper to compare expected filenames with existing ones.

        Args:
            param_sweeps (dict): Dictionary of parameters to sweep through.
            cached_files_dir (str): Directory to check for cached experiment results.
        """
        from itertools import product

        if not cached_files_dir or not path.exists(cached_files_dir):
            print(f"Cache directory {cached_files_dir} does not exist.")
            return

        # Get existing files
        existing_files = [f for f in os.listdir(cached_files_dir) if f.endswith(".pkl")]
        print(f"Found {len(existing_files)} .pkl files in {cached_files_dir}")

        if existing_files:
            print("\nSample existing filenames:")
            for i in range(min(3, len(existing_files))):
                print(f"  {existing_files[i]}")

        # Generate expected filenames for some parameter combinations
        print("\nSample expected filenames:")
        keys, values = zip(*param_sweeps.items())
        param_combinations = list(product(*values))

        for i in range(min(3, len(param_combinations))):
            sample_params = dict(zip(keys, param_combinations[i]))
            temp_experiment = ADRExperiment(**sample_params)
            expected_filename = temp_experiment.generate_filename()
            print(f"  Params: {sample_params}")
            print(f"  Expected filename: {expected_filename}")

            # Check if this file exists
            if expected_filename in existing_files:
                print(f"  ✓ File exists in cache")
            else:
                print(f"  ✗ File not found in cache")
            print()

        # Check for any exact matches
        exact_matches = 0
        for params in (dict(zip(keys, combo)) for combo in param_combinations):
            temp_experiment = ADRExperiment(**params)
            expected_filename = temp_experiment.generate_filename()
            if expected_filename in existing_files:
                exact_matches += 1

        print(f"Found {exact_matches} exact filename matches out of {len(param_combinations)} parameter combinations.")

    @staticmethod
    def parse_filename(filename: str) -> dict:
        """
        Parse a filename to extract experimental parameters.

        Args:
            filename (str): The filename to parse.

        Returns:
            dict: A dictionary of variable-value pairs.
        """
        # Define the regex pattern to match the filename format
        pattern = re.compile(
            r"ADR_(?P<mesh_shape>\w+)_(?P<init_center_x>[-\d.]+)_(?P<init_center_y>[-\d.]+)_(?P<plume_size>\d+)_(?P<diff_coeff_x>[-\d.]+)_(?P<diff_coeff_y>[-\d.]+)_(?P<advection_coeff_x>[-\d.]+)_(?P<advection_coeff_y>[-\d.]+)_(?P<velocity_field_type>\w+)_(?P<velocity_param_x>[-\d.]+)_(?P<velocity_param_y>[-\d.]+)_(?P<react_rate>[-\d.]+)_(?P<reaction_scaling>[-\d.]+)_(?P<H>[-\d.]+)_(?P<radius>[-\d.]+)_(?P<capture_apothem>[-\d.]+)_(?P<capture_N>\d+)_(?P<t_start>[-\d.]+)_(?P<t_stop>[-\d.]+)_(?P<t_num>\d+)\.pkl"
        )

        # Match the pattern against the filename
        match = pattern.match(filename)
        if not match:
            raise ValueError("Filename does not match the expected format.")

        # Extract the matched groups into a dictionary
        params = match.groupdict()

        # Convert appropriate values to float or int
        params["mesh_shape"] = params["mesh_shape"]
        params["init_center"] = [float(params.pop("init_center_x")), float(params.pop("init_center_y"))]
        params["plume_size"] = int(params["plume_size"])
        params["diff_coeffs"] = [float(params.pop("diff_coeff_x")), float(params.pop("diff_coeff_y"))]
        params["advection_coeffs"] = [float(params.pop("advection_coeff_x")), float(params.pop("advection_coeff_y"))]

        # Handle velocity-related parameters
        params["velocity_field_type"] = params["velocity_field_type"] if params["velocity_field_type"] != "none" else None
        velocity_param_x = float(params.pop("velocity_param_x"))
        velocity_param_y = float(params.pop("velocity_param_y"))

        # Only set velocity_parameters if at least one parameter is non-zero
        if velocity_param_x != 0.0 or velocity_param_y != 0.0:
            params["velocity_parameters"] = [velocity_param_x, velocity_param_y]
        else:
            params["velocity_parameters"] = None

        # Convert remaining parameters
        params["react_rate"] = float(params["react_rate"])
        params["reaction_scaling"] = float(params["reaction_scaling"])
        params["init_concentration"] = float(params["init_concentration"])
        params["H"] = float(params["H"])
        params["radius"] = float(params["radius"])
        params["capture_apothem"] = float(params["capture_apothem"])
        params["capture_N"] = int(params["capture_N"])

        # Set up time parameters
        t_start = float(params.pop("t_start"))
        t_stop = float(params.pop("t_stop"))
        t_num = int(params.pop("t_num"))
        params["t"] = np.linspace(t_start, t_stop, t_num).tolist() if t_num > 0 else None

        return params

    @staticmethod
    def get_vectors_from_stencil(stencil_graph, stencil_val_matrix):
        dependence_dict = {}
        for i in range(stencil_graph.shape[0]):
            for j in range(stencil_graph.shape[1]):
                for k in range(stencil_graph.shape[2]):
                    if stencil_graph[i, j, k] != "":
                        dependence_dict[i, j, k] = stencil_val_matrix[i, j, k]

        angle_dict = {
            (0, 4, 1): 315,
            (1, 4, 1): 270,
            (2, 4, 1): 225,
            (3, 4, 1): 0,
            (4, 4, 1): -1,
            (5, 4, 1): 180,
            (6, 4, 1): 45,
            (7, 4, 1): 90,
            (8, 4, 1): 135,
        }

        vectors = [(dependence_dict[dependence], angle_dict[dependence]) for dependence in dependence_dict.keys() if dependence != (4, 4, 1)]
        return vectors

    @staticmethod
    def combine_angles(vectors):
        angles_radians = np.radians([vector[1] for vector in vectors])
        coefficients = [vector[0] for vector in vectors]
        xs = coefficients * np.cos(angles_radians)
        ys = coefficients * np.sin(angles_radians)

        x_sum = np.sum(xs)
        y_sum = np.sum(ys)

        angle_radians = np.arctan2(y_sum, x_sum)
        angle_degrees = np.degrees(angle_radians)

        if angle_degrees < 0:
            angle_degrees += 360
        return angle_degrees

    @staticmethod
    def angle_difference(angle1, angle2) -> float:
        # Difference formula from https://gamedev.stackexchange.com/questions/4467/comparing-angles-and-working-out-the-difference
        return 180 - abs(abs(angle1 - angle2) - 180)

    def animate_solution(self, save_path: str = None) -> None:
        """
        Animate the solution.

        Args:
            save_path (str, optional): Path to save the animation as a GIF. Defaults to None.
        """
        ani = self.pde_solver.animate(self.solution, save_path=save_path)
        return ani

    def pretty_print_parameters(self) -> None:
        """
        Pretty print the parameters of the ADRExperiment instance.
        """
        params = vars(self)  # or self.__dict__

        print("ADRExperiment Parameters:")
        for key, value in params.items():
            if key == "t":
                t_start = value[0]
                t_stop = value[-1]
                t_num = len(value)
                print(f"  t: start={t_start}, stop={t_stop}, num={t_num}")
            elif key == "solution":
                continue
            elif key == "pde_solver":
                continue
            elif key == "plot":
                continue
            elif key == "verbose":
                continue
            else:
                print(f"  {key}: {value}")

    def run_batch_experiments(self, param_sweeps: dict, cached_files_dir: str = None, force_rerun: bool = False) -> None:
        """
        Run a batch of ADR experiments with varying parameters, using a single MATLAB engine.

        Args:
            param_sweeps (dict): Dictionary of parameters to sweep through.
            cached_files_dir (str, optional): Directory for cached results.
            force_rerun (bool, optional): If True, rerun even if cached.
        """
        from itertools import product
        import os

        # Save original settings
        orig_verbose = self.verbose
        orig_plot = self.plot

        # Create cache directory if needed
        if cached_files_dir and not os.path.exists(cached_files_dir):
            os.makedirs(cached_files_dir)

        # Get parameter combinations
        keys, values = zip(*param_sweeps.items())
        param_combinations = [dict(zip(keys, combination)) for combination in product(*values)]
        total_experiments = len(param_combinations)

        # Find experiments to run
        completed_experiments = 0
        experiments_to_run = []

        for params in param_combinations:
            # Check cache
            if not force_rerun and cached_files_dir:
                is_cached, cached_path = self.is_experiment_cached(params, cached_files_dir)
                if is_cached:
                    completed_experiments += 1
                    continue

            # Add to run list
            expected_path = None
            if cached_files_dir:
                temp_experiment = ADRExperiment(**params)
                expected_filename = temp_experiment.generate_filename()
                expected_path = os.path.join(cached_files_dir, expected_filename)

            experiments_to_run.append((params, expected_path))

        if orig_verbose:
            print(f"Total experiments: {total_experiments}")
            print(f"Already completed: {completed_experiments}")
            print(f"Remaining to run: {len(experiments_to_run)}")

        if not experiments_to_run:
            if orig_verbose:
                print("All experiments already completed.")
            return

        # Start MATLAB engine once
        if orig_verbose:
            print("Starting MATLAB engine...")
        self.pde_solver.start_engine()

        try:
            # Track runtime statistics
            total_runtime = 0
            experiment_count = 0

            # Run each experiment
            for i, (params, cached_file_path) in enumerate(experiments_to_run):
                if orig_verbose:
                    print(f"\nRunning experiment {i+1}/{len(experiments_to_run)}:")

                start_time = time.time()

                # Set up parameters for this experiment
                exp_params = params.copy()
                exp_params["verbose"] = orig_verbose
                exp_params["plot"] = orig_plot
                exp_params["parallel_interpolation"] = False  # Avoid parallel pool issues

                # Save the current engine
                current_engine = self.pde_solver.eng if hasattr(self.pde_solver, "eng") else None

                # Reset parameters
                self.__init__(**exp_params)

                # Restore the engine
                if current_engine is not None:
                    self.pde_solver.eng = current_engine

                if orig_verbose:
                    print("Current experiment parameters:")
                    self.pretty_print_parameters()

                # Run the experiment
                matlab_path = "ADR_Driver_func.m"
                out = self.pde_solver.run_script(
                    matlab_path,
                    self.mesh_shape,
                    self.init_center,
                    self.plume_size,
                    self.diff_coeffs,
                    self.advection_coeffs,
                    self.velocity_field_type,
                    self.velocity_parameters,
                    self.react_rate,
                    self.reaction_scaling,
                    self.init_concentration,
                    self.t,
                    self.H,
                    self.radius,
                    self.capture_apothem,
                    self.capture_N,
                    self.parallel_interpolation,
                    self.plot,
                    self.verbose,
                    nargout=1,
                )

                self.solution = np.array(out)

                # Save results
                if cached_files_dir:
                    self.save_results(cached_file_path)

                # Update statistics
                runtime = time.time() - start_time
                total_runtime += runtime
                experiment_count += 1
                running_average = total_runtime / experiment_count

                # Report progress
                if orig_verbose:
                    print(f"Experiment completed in {runtime:.2f} seconds.")
                    print(f"Running average: {running_average:.2f} seconds per experiment.")
                    print(f"Progress: {experiment_count}/{len(experiments_to_run)} ({experiment_count/len(experiments_to_run)*100:.1f}%)")

                    est_remaining = running_average * (len(experiments_to_run) - experiment_count)
                    hours = int(est_remaining / 3600)
                    minutes = int((est_remaining % 3600) / 60)
                    print(f"Estimated time remaining: {hours}h {minutes}m")

                # Clear MATLAB memory
                if hasattr(self.pde_solver, "eng"):
                    self.pde_solver.eng.eval("clearvars -except pde_solver;", nargout=0)
                    if orig_verbose:
                        print("Cleared MATLAB memory for next experiment.")

        finally:
            # Always shut down the MATLAB engine
            if orig_verbose:
                print("Shutting down MATLAB engine...")
            self.pde_solver.quit_engine()

        # Final report
        if orig_verbose and experiment_count > 0:
            print(f"\nCompleted {experiment_count} new experiments.")
            print(f"Average runtime: {total_runtime/experiment_count:.2f} seconds per experiment.")
            print(f"Total completed: {completed_experiments + experiment_count}/{total_experiments}.")
