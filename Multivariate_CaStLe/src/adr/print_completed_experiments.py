"""
Print Completed ADR Experiments

Inspect a directory of saved ADRExperiment pickles and print the parameter
set for each completed experiment.  Accepts either a directory (prints all
completed experiments) or a single ``.pkl`` file (prints that file's
parameters).  The ``--coded`` flag formats output as Python keyword arguments
so parameters can be copy-pasted directly into a sweep configuration.

Usage
-----
    python print_completed_experiments.py <directory_or_file> [--coded]
"""

import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse
from ADRExperiment import ADRExperiment


def check_completed_experiments(directory: str, coded: bool = False) -> None:
    """
    Print the parameter set for every completed experiment in a directory.

    Args:
        directory (str): Path to the directory containing ``.pkl`` experiment files.
        coded (bool): If True, print parameters as Python keyword arguments
            (suitable for pasting into a sweep configuration).
    """
    completed_experiments = []

    for filename in os.listdir(directory):
        if filename.endswith(".pkl"):
            try:
                params = ADRExperiment.parse_filename(filename)
                completed_experiments.append(params)
            except ValueError:
                continue

    if not completed_experiments:
        print("No completed experiments found.")
        return

    print("Completed Experiments:")
    for i, params in enumerate(completed_experiments, start=1):
        print(f"\nExperiment {i}:")
        print_parameters(params=params, filename=filename, coded=coded)


def print_file_parameters(file_path: str, coded: bool = False) -> None:
    """
    Print the parameters encoded in a single ADRExperiment filename.

    Args:
        file_path (str): Path to the ``.pkl`` experiment file.
        coded (bool): If True, format parameters as Python keyword arguments.
    """
    try:
        filename = os.path.basename(file_path)
        params = ADRExperiment.parse_filename(filename)
        print("Experiment Parameters:")
        print_parameters(params=params, filename=filename, coded=coded)
    except ValueError:
        print("The provided file does not match the expected format.")


def print_parameters(params: dict, filename: str, coded: bool = False) -> None:
    """
    Print an ordered set of ADR experiment parameters.

    Args:
        params (dict): Parameter dictionary returned by
            :meth:`ADRExperiment.parse_filename`.
        filename (str): The source filename, printed as a header line.
        coded (bool): If True, format each parameter as a Python keyword
            argument (e.g. ``diff_coeffs=[0.05, 0.05],``).
    """
    order = [
        "init_center",
        "plume_size",
        "diff_coeffs",
        "advection_coeffs",
        "velocity_field_type",
        "velocity_parameters",
        "react_rate",
        "t",
        "H",
        "radius",
        "capture_apothem",
        "capture_N",
    ]

    print("  " + filename)
    for key in order:
        value = params.get(key)
        if coded:
            if key == "t":
                t_start = value[0]
                t_stop = value[-1]
                t_num = len(value)
                print(f"        t=np.linspace({t_start}, {t_stop}, {t_num}).tolist(),")
            else:
                if isinstance(value, list):
                    value_str = f"[{', '.join(map(str, value))}]"
                else:
                    value_str = f'"{value}"' if isinstance(value, str) else str(value)
                print(f"        {key}={value_str},")
        else:
            if key == "t":
                t_start = value[0]
                t_stop = value[-1]
                t_num = len(value)
                print(f"  t: start={t_start}, stop={t_stop}, num={t_num}")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check completed experiments in a directory or print parameters of a single file."
    )
    parser.add_argument(
        "path", type=str, help="Path to the directory or file to check."
    )
    parser.add_argument(
        "--coded",
        "-c",
        action="store_true",
        help="Print parameters in code format style.",
    )
    args = parser.parse_args()

    if os.path.isdir(args.path):
        check_completed_experiments(args.path, args.coded)
    elif os.path.isfile(args.path):
        print_file_parameters(args.path, args.coded)
    else:
        print("The provided path is neither a directory nor a file.")
