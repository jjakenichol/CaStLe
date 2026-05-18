import numpy as np
import os
import time
from ADRExperiment import ADRExperiment
from concurrent.futures import ProcessPoolExecutor
from itertools import product


def run_single_experiment_wrapper(params, cached_files_dir, verbose):
    experiment = ADRExperiment(verbose=verbose)
    experiment.run_single_experiment(params, cached_files_dir)


def main():
    param_sweeps = {
        "mesh_shape": ["circle"],
        "init_center": [[0.0, 0.0]],
        "plume_size": [10, 50],
        "diff_coeffs": [[0.01, 0.01], [0.05, 0.05]],
        "advection_coeffs": [[1.0, 1.0]],
        "velocity_magnitude": [1.0, 2.0],
        "velocity_angle": [33],
        "velocity_field_type": ["constant"],
        "react_rate": [
            1,
            # 2,
        ],  # 10
        "reaction_scaling": [1.0, 2.0],
        "init_concentration": [50, 200, 400],
        "t": [np.linspace(0.0, 0.8, 21).tolist(), np.linspace(0.0, 0.8, 51).tolist(), np.linspace(0.0, 0.8, 101).tolist()],
        "H": [
            0.02,
        ],
        "radius": [3.0],
        "capture_apothem": [1.0],
        "capture_N": [10, 40, 160],
        "parallel_interpolation": [False],
    }

    keys, values = zip(*param_sweeps.items())
    param_combinations = [dict(zip(keys, combination)) for combination in product(*values)]

    num_workers = 1  # Specify the number of workers to use # NOTE: more than one seems much slower for now.
    cached_files_dir = "results/concentration_study/ADR_model_output"
    verbose = True

    if num_workers == 1:
        experiment = ADRExperiment(verbose=verbose)

        # Run with debug first time to find issues
        # experiment.run_batch_experiments(param_sweeps, cached_files_dir=cached_files_dir, debug_filenames=True)

        # Run normally
        experiment.run_batch_experiments(param_sweeps, cached_files_dir=cached_files_dir)

        # If needed, force rerun of all experiments
        # experiment.run_batch_experiments(param_sweeps, cached_files_dir=cached_files_dir, force_rerun=True)
        return

    # Calculate the total number of experiments
    total_experiments = len(param_combinations)

    # Calculate the number of completed experiments
    completed_experiments = 0
    incomplete_experiments = []
    for params in param_combinations:
        experiment = ADRExperiment(verbose=verbose)
        for key, value in params.items():
            setattr(experiment, key, value)
        filename = experiment.generate_filename()
        cached_file_path = os.path.join(cached_files_dir, filename) if cached_files_dir else None

        if cached_files_dir and os.path.exists(cached_file_path):
            completed_experiments += 1
        else:
            incomplete_experiments.append(params)

    remaining_experiments = total_experiments - completed_experiments

    if verbose:
        print(f"Total number of experiments queued: {total_experiments}")
        print(f"Total number of completed experiments: {completed_experiments}")
        print(f"Total number of remaining experiments: {remaining_experiments}")

    start_time = time.time()

    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(run_single_experiment_wrapper, params, cached_files_dir, verbose) for params in incomplete_experiments]
            for i, future in enumerate(futures):
                try:
                    future.result()  # Wait for all futures to complete
                except Exception as e:
                    print(f"Experiment {i+1} failed with error: {e}")
                if verbose:
                    elapsed_time = time.time() - start_time
                    avg_time_per_experiment = elapsed_time / (i + 1)
                    print(f"Average completion time per experiment: {avg_time_per_experiment:.2f} seconds")
                    print(f"Experiments completed: {i + 1}/{total_experiments}")
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Terminating experiments...")
        executor.shutdown(wait=False)
    finally:
        print("Experiments terminated.")


if __name__ == "__main__":
    main()
