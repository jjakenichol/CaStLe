# import os
# import sys
# import time
# import pickle
# import argparse
# from ADRExperiment import ADRExperiment
# from tigramite.independence_tests.parcorr import ParCorr

# import mcastle_utils as ms


# corrupted_files = []
# total_runtime = 0
# experiment_count = 0


# def process_experiment(filepath: str, results_dir: str = None, experiments_to_run: int = None) -> None:
#     """
#     Process a single ADR experiment to compute the stencil and save the results.

#     Args:
#         filepath (str): Path to the ADR experiment results file (pkl).
#         results_dir (str, optional): Directory to save the stencil results. Defaults to the same directory as the input file.
#     """
#     global total_runtime, experiment_count

#     # CaStLe hyperparameters
#     pc_alpha = 0.01
#     graph_p_threshold = 0.01

#     # Save stencil results dictionary
#     filename = os.path.basename(filepath)
#     if results_dir:
#         os.makedirs(results_dir, exist_ok=True)
#         save_filename = os.path.join(results_dir, filename[:-4] + f"_stencil_results_{pc_alpha}_{graph_p_threshold}.pkl")
#     else:
#         save_filename = os.path.join(os.path.dirname(filepath), filename[:-4] + f"_stencil_results_{pc_alpha}_{graph_p_threshold}.pkl")
#     if os.path.exists(save_filename):
#         print(f"File already processed at: {save_filename}")
#         return

#     experiment = ADRExperiment.load_results(filepath)

#     print(f"Processing MV CaStLe for")
#     experiment.pretty_print_parameters()

#     if experiment is None:
#         print(f"Failed to load ADRExperiment object from {filepath}")
#         corrupted_files.append(filepath)
#         return

#     solution = experiment.solution
#     try:
#         if solution == None:
#             print(f"Solution is {solution}, and shape is {solution.shape}, file likely corrupted: {filepath}")
#             corrupted_files.append(filepath)
#             return
#     except:
#         pass

#     # Reshape solution to from (Xs, Ys, time, species) -> (variable_n, X, Y, T)
#     data = solution.transpose((3, 0, 1, 2))

#     # Compute stencil
#     parcorr = ParCorr(significance="analytic")
#     start_time = time.time()
#     results = ms.mv_CaStLe_PC(
#         data=data,
#         cond_ind_test=parcorr,
#         pc_alpha=pc_alpha,
#         graph_p_threshold=graph_p_threshold,
#     )
#     runtime = time.time() - start_time
#     total_runtime += runtime
#     experiment_count += 1
#     running_average = total_runtime / experiment_count
#     print(f"MV CaStLe {experiment_count} took {runtime:.2f} seconds to complete.")
#     print(f"Running average time: {running_average:.2f} seconds.")

#     with open(save_filename, "wb") as f:
#         pickle.dump(results, f)
#     print(f"Results saved to {save_filename}")

#     if experiments_to_run:
#         est_remaining = running_average * (experiments_to_run - experiment_count)
#         hours = int(est_remaining / 3600)
#         minutes = int((est_remaining % 3600) / 60)
#         print(f"Estimated time remaining: {hours}h {minutes}m\n")


# def process_completed_experiments(directory: str, results_dir: str = None) -> None:
#     """
#     Process all completed ADR experiments in the given directory.

#     Args:
#         directory (str): Directory containing the completed ADR experiment results files.
#         results_dir (str, optional): Directory to save the stencil results. Defaults to the same directory as the input files.
#     """
#     experiments_to_run = len(os.listdir(directory))
#     for filename in os.listdir(directory):
#         if filename.startswith("ADR") and filename.endswith(".pkl") and "stencil_results" not in filename:
#             filepath = os.path.join(directory, filename)
#             if results_dir:
#                 save_filename = os.path.join(results_dir, filename[:-4] + "_stencil_results_0.01_0.01.pkl")
#             else:
#                 save_filename = os.path.join(directory, filename[:-4] + "_stencil_results_0.01_0.01.pkl")

#             if os.path.exists(save_filename):
#                 # print(f"Skipping {filename}, results already exist.")
#                 continue
#             process_experiment(filepath, results_dir, experiments_to_run)

#     if corrupted_files:
#         print("\nCorrupted files:")
#         for file in corrupted_files:
#             print(file)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Process completed ADR experiments to compute stencils.")
#     parser.add_argument("directory", type=str, help="Path to the directory containing completed ADR experiment results.")
#     parser.add_argument("--results_dir", type=str, help="Path to the directory to save the stencil results. Defaults to the same directory as the input files.")
#     args = parser.parse_args()

#     process_completed_experiments(args.directory, args.results_dir)
#     print(f"Processed all {experiment_count} unprocessed files.")
