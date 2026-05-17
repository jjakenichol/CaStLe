"""
This script runs the MV_CaStLe (Causal Space-Time Stencil Learning) experiment to generate and analyze chain graphs with specified parameters, saving the results to a dictionary.
Parameters:
- T: int, Number of time steps for the generated dataset
- grid_size: int, Size of the grid for the spatial coefficients
- num_variables: int, Number of variables in the dataset
- coefficient_value: float, Value of the coefficients used to generate the dataset
- position: Optional[Tuple[int, int]], Position in the grid (either (1, 1) for center or None)
- pc_alpha: float, Alpha value for the PC algorithm
- pval_threshold: float, P-value threshold for the graph reconstruction
- print_results: bool, Print results to the console instead of saving (default: True)
- verbose: bool, Enable verbose output (default: True)

Example usage:
python chain_stencil_experiment.py --T 1000 --grid_size 4 --num_variables 10 --coefficient_value 0.1 --position center --pc_alpha 0.01 --pval_threshold 0.01 --print --verbose

This will run the experiment with the specified parameters and print the results to the console.
To save the results instead, omit the --print_results flag. The results will be saved to a file with a descriptive name in the specified directory.
"""

import argparse
import os
import pickle
import sys
import time
import uuid
from typing import Optional, Tuple, Dict, Any
from tigramite.independence_tests.parcorr import ParCorr

import mcastle_utils as ms
import spatiotemporal_SCM_data_generator as mvdg
from causal_graph_metrics import F1_score, get_graph_metrics, get_confusion_matrix, matthews_correlation_coefficient as mcc


def run_experiment(
    T: int,
    grid_size: int,
    num_variables: int,
    coefficient_value: float,
    position: Optional[Tuple[int, int]],
    pc_alpha: float,
    pval_threshold: float,
    output_dir: str,
    print_results: bool = True,
    verbose: bool = True,
) -> None:
    """
    Run the MV_CaStLe experiment with the specified parameters.

    Parameters:
    - T: int, Number of time steps
    - grid_size: int, Grid size
    - num_variables: int, Number of variables
    - coefficient_value: int, Coefficient value
    - position: Optional[Tuple[int, int]], Position (either (1, 1) or None)
    - pc_alpha: float, PC alpha value
    - pval_threshold: float, P-value threshold
    - output_dir: str, Directory to save experiment results
    - print_results: bool, Print results instead of saving (default: True)
    - verbose: bool, Enable verbose output (default: True)
    """
    if position not in [(1, 1), None]:
        raise ValueError("Position can only be (1, 1) or None")

    # Generate stable coefficients
    spatial_coefficients = mvdg.get_stable_coefficient_chain_matrix(
        grid_size=grid_size, n_variables=num_variables, coefficient_value=coefficient_value, position=position, verbose=0
    )
    true_graph, _ = ms.get_stencil_graph_from_coefficients(spatial_coefficients)

    # Generate dataset
    data = mvdg.generate_dataset(
        T=T, grid_size=grid_size, spatial_coefs=spatial_coefficients, num_variables=num_variables, verbose=verbose, initialize_randomly=True
    )

    parcorr = ParCorr(significance="analytic")

    start_time = time.time()
    results = ms.mv_CaStLe_PC(
        data, parcorr, pc_alpha, cd_function="run_pcalg", fdr_method="bh", rows_inverted=True, graph_p_threshold=pval_threshold
    )
    algorithm_time = time.time() - start_time
    reconstructed_graph = results["graph"]

    # Compute performance metrics
    F1, P, R, TP, FP, FN, TN = F1_score(true_graph=true_graph, discovered_graph=reconstructed_graph)
    MCC = mcc(TP=TP, FP=FP, FN=FN, TN=TN)

    output_dict: Dict[str, Any] = {
        "data": data,
        "spatial_coefficients": spatial_coefficients,
        "reconstructed_graph": reconstructed_graph,
        "true_graph_metrics": get_graph_metrics(true_graph),
        "reconstructed_graph_metrics": get_graph_metrics(reconstructed_graph),
        "graphs_equal": (true_graph == reconstructed_graph),
        "F1": F1,
        "MCC": MCC,
        "Precision": P,
        "Recall": R,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "algorithm_time": algorithm_time,
        "pc_alpha": pc_alpha,
        "pval_threshold": pval_threshold,
    }

    # Construct the filename
    position_str = "center" if position == (1, 1) else "none"
    DATA_FILENAME = f"r_T_{T}_grid_{grid_size}_vars_{num_variables}_coef_{coefficient_value}_pos_{position_str}_alpha_{pc_alpha}_pval_{pval_threshold}_{uuid.uuid4()}.pkl"

    # Print or save results
    if not print_results:
        SAVE_DIR = output_dir
        SAVE_PATH = os.path.join(SAVE_DIR, DATA_FILENAME)
        if verbose:
            print("Saving data to " + SAVE_PATH)
        with open(SAVE_PATH, "wb") as f:
            pickle.dump(output_dict, f)
    else:
        print(output_dict)
        print(f"F1={F1}, P={P}, R={R}, TP={TP}, FP={FP}, FN={FN}, TN={TN}")
    if verbose:
        print(f"Time elapsed for algorithm completion: {algorithm_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MV_CaStLe experiment with specified parameters.")
    parser.add_argument("--T", type=int, required=True, help="Number of time steps")
    parser.add_argument("--grid_size", type=int, required=True, help="Grid size")
    parser.add_argument("--num_variables", type=int, required=True, help="Number of variables")
    parser.add_argument("--coefficient_value", type=float, required=True, help="Coefficient value")
    parser.add_argument("--position", type=str, choices=["center", "none"], required=True, help="Position (center or none)")
    parser.add_argument("--pc_alpha", type=float, required=True, help="PC alpha value")
    parser.add_argument("--pval_threshold", type=float, required=True, help="P-value threshold")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save experiment results")
    parser.add_argument("--print", action="store_true", help="Print results instead of saving")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    position = (1, 1) if args.position == "center" else None

    run_experiment(
        T=args.T,
        grid_size=args.grid_size,
        num_variables=args.num_variables,
        coefficient_value=args.coefficient_value,
        position=position,
        pc_alpha=args.pc_alpha,
        pval_threshold=args.pval_threshold,
        output_dir=args.output_dir,
        print_results=args.print,
        verbose=args.verbose,
    )
