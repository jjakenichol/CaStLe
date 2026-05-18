import os
import argparse
from ADRExperiment import ADRExperiment


def check_completed_experiments(directory: str, coded: bool = False) -> None:
    """
    Check which parameter sets have been completed in the given directory.

    Args:
        directory (str): The directory to check.
        formatted (bool): Whether to print the parameters in a formatted style.
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
    Print the parameters of a single experiment file.

    Args:
        file_path (str): The path to the file.
        formatted (bool): Whether to print the parameters in a formatted style.
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
    Print the parameters in the specified order.

    Args:
        params (dict): The parameters to print.
        formatted (bool): Whether to print the parameters in a formatted style.
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
    parser = argparse.ArgumentParser(description="Check completed experiments in a directory or print parameters of a single file.")
    parser.add_argument("path", type=str, help="Path to the directory or file to check.")
    parser.add_argument("--coded", "-c", action="store_true", help="Print parameters in code format style.")
    args = parser.parse_args()

    if os.path.isdir(args.path):
        check_completed_experiments(args.path, args.coded)
    elif os.path.isfile(args.path):
        print_file_parameters(args.path, args.coded)
    else:
        print("The provided path is neither a directory nor a file.")
