from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from matplotlib import pyplot as plt
from os import path
from tigramite.independence_tests.parcorr import ParCorr
from tigramite import plotting as tp
import math
import matlab.engine
import matplotlib
import numpy as np
import pandas as pd

import stencil_functions as sf


def run_matlab_script(script_name, *args, nargout=0) -> matlab.double:
    # Start a matlab egnine
    eng = matlab.engine.start_matlab()

    d = path.dirname(script_name)
    f = path.basename(script_name.split(".")[0])
    eng.addpath(d + "/", nargout=0)

    # Convert arguments to matlab data type
    args = [
        matlab.double([arg]) if isinstance(arg, (int, float)) else arg for arg in args
    ]

    # Run the matlab script with the provided arguments
    run_string = f"{f}({', '.join(map(str, args))})"
    out = eng.eval(run_string, nargout=nargout)

    # Stop the engine
    eng.quit()

    return out


def start_matlab_engine() -> matlab.engine:
    # Start a matlab egnine
    eng = matlab.engine.start_matlab()
    return eng


def quit_matlab_engine(eng) -> None:
    # Stop the engine
    eng.quit()


def run_matlab_script(eng, path_to_matlab_script, *args, nargsout=1) -> matlab.double:
    # Add script path to engine
    script_dir = path.dirname(path_to_matlab_script)
    script_filename = path.basename(path_to_matlab_script.split(".")[0])
    if script_dir not in eng.pwd():
        eng.addpath(script_dir + "/", nargout=0)

    # Convert arguments to matlab data type
    args = [
        matlab.double([arg]) if isinstance(arg, (int, float)) else arg for arg in args
    ]

    # Run the matlab script with the provided arguments
    run_string = f"{script_filename}({', '.join(map(str, args))})"
    out = eng.eval(run_string, nargout=nargsout)

    return out


def animate(data) -> FuncAnimation:
    fig, ax = plt.subplots()
    im = ax.imshow(data[:, :, 0], cmap="viridis", interpolation="nearest")
    ax.invert_yaxis()

    def update(i):
        im.set_array(data[:, :, i])
        im.set_clim(vmin=data[:, :, i].min(), vmax=data[:, :, i].max())
        return [im]

    ani = FuncAnimation(fig, update, frames=range(data.shape[2]), blit=True)
    plt.show()
    return ani


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


def infer_burgers_angle(alpha, beta) -> float:
    degree = math.degrees(math.atan2(beta, alpha))
    if degree < 0:
        degree += 360
    return degree


def angle_difference(angle1, angle2) -> float:
    # Difference formula from https://gamedev.stackexchange.com/questions/4467/comparing-angles-and-working-out-the-difference
    return 180 - abs(abs(angle1 - angle2) - 180)


def plot_stencil(graph, v_matrix):
    fig, ax = plt.subplots()
    x_pos = list(np.array([[i for i in range(3)] for j in range(3)]).flatten())
    y_pos = [i for i in range(3) for j in range(3)]
    y_pos.reverse()
    node_positions = {
        "x": x_pos,
        "y": y_pos,
    }
    tp.plot_graph(
        val_matrix=v_matrix,  # .round(),
        graph=graph,
        link_label_fontsize=0.0,
        # show_colorbar=False,
        var_names=[""] * 9,
        # var_names=var_names,
        node_pos=node_positions,
        link_colorbar_label="Cross-Dependence (MCI)",
        node_colorbar_label="Inter-Dependence (MCI)",
        fig_ax=(fig, ax),
    )
    # Remove link labels which always have "1"
    for child in ax.get_children():
        if isinstance(child, matplotlib.text.Text):
            if child.get_text() == "1":
                Artist.set_visible(child, False)
    ax.patch.set_alpha(0)
    plt.show()


def add_noise_to_data(time_series: np.ndarray, snr: float) -> np.ndarray:
    """Add noise to a given time series with the specified signal:noise ratio (in decibels)

    Args:
        time_series (np.ndarray): the time series to add noise to
        snr (float): the desired signal to noise ratio

    Returns:
        np.ndarray: the noisy signal
    """

    # Convert from dB to linear
    snr = 10 ** (snr / 10)

    # Calculate the power of the signal
    signal_power = np.mean(time_series**2)

    # Calculate the power of the noise needed to achieve the desired SNR
    noise_power = signal_power / snr

    # Generate the noise with the calculated power
    noise = np.random.normal(0, np.sqrt(noise_power), time_series.shape)
    noisy_signal = time_series + noise

    return noisy_signal


def run_specific_experiment(
    init_diam,
    x_pos,
    y_pos,
    alpha=None,
    beta=None,
    angle=None,
    magnitude=None,
    c=0.05,
    radius=3,
    capture_apothem=1,
    capture_N=101,
    alpha_pc=0.00001,
    dependence_threshold=None,
    add_noise=False,
    noise_SNR=None,
    verbosity=0,
    matlab_verbosity=0,
    plot_stencil_graph=0,
    print_stencil=0,
    animate_numpy_data=0,
    eng=None,
):
    # Initialize script options
    plot_matlab = 0
    data_truncation_start = None
    data_truncation_end = None
    matlab_path = "2d_nonlinear_model/Burgers/Driver_func.m"

    # Initialize Burgers' parameters
    if angle is None and (alpha is None or beta is None):
        raise ValueError("angle and one of alpha or beta cannot be None.")
    elif alpha is None or beta is None:
        assert magnitude is not None, "magnitude must not be None."
        alpha = magnitude * np.cos(angle)
        beta = magnitude * np.sin(angle)
    elif angle is None:
        # angle not given, so use alpha and beta directly.
        assert (alpha is not None) and (
            beta is not None
        ), "angle not given, so alpha and beta must be."
        magnitude = np.sqrt(alpha**2 + beta**2)
        angle = np.arctan2(beta, alpha)
        pass
    else:
        raise Warning(
            "angle ({}), alpha ({}), beta ({}) conditions not satisfied. Undefined behavior follows.".format(
                angle, alpha, beta
            )
        )

    if verbosity:
        print(
            "angle: {:.2f}, alpha: {:.2f}, beta: {:.2f}, c: {:.2f}, magnitude: {}, spatial res. N: {}, capture apothem: {}, radius: {}, add noise: {}, noise SNR: {}".format(
                math.degrees(angle),
                alpha,
                beta,
                c,
                magnitude,
                capture_N,
                capture_apothem,
                radius,
                add_noise,
                noise_SNR,
            )
        )

    if eng is None:
        eng = start_matlab_engine()
        if verbosity >= 3:
            print("MATLAB engine started.")
    out = run_matlab_script(
        eng,
        matlab_path,
        init_diam,
        x_pos,
        y_pos,
        alpha,
        beta,
        c,
        radius,
        capture_apothem,
        capture_N,
        plot_matlab,
        matlab_verbosity,
        nargsout=1,
    )
    data = np.array(out)[:, :, data_truncation_start:data_truncation_end]

    if add_noise:
        assert noise_SNR is not None, "Noise SNR must be given if add_noise is True."
        data = add_noise_to_data(data, noise_SNR)

    if animate_numpy_data:
        animate(data)

    graph, v_matrix = sf.CaStLe(
        data=data,
        rows_inverted=False,
        cond_ind_test=ParCorr(significance="analytic"),
        pc_alpha=alpha_pc,
        dependence_threshold=dependence_threshold,
    )
    if verbosity >= 3:
        print("Stencil computed.")

    dependence_dict = {}
    for i in range(graph.shape[0]):
        for j in range(graph.shape[1]):
            for k in range(graph.shape[2]):
                if graph[i, j, k] != "":
                    dependence_dict[i, j, k] = v_matrix[i, j, k]
    # print(dependence_dict)

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

    # Get all vectors in the for [(magnitude, angle)...] and exclude the autodependence
    vectors = [
        (dependence_dict[dependence], angle_dict[dependence])
        for dependence in dependence_dict.keys()
        if dependence != (4, 4, 1)
    ]

    stencil_angle = combine_angles(vectors)
    burgers_angle = infer_burgers_angle(alpha, beta)
    difference = angle_difference(np.degrees(angle), stencil_angle)
    if print_stencil:
        parents = sf.get_parents(
            graph=graph, val_matrix=v_matrix, output_val_matrix=True
        )[4]
        parents_dict = {parents[idx][0]: parents[idx][2] for idx in range(len(parents))}
        stencil_mat = np.ndarray((3, 3))
        idx = 0
        for i in range(3):
            for j in range(3):
                try:
                    coef = np.round(parents_dict[idx], 3)
                except:
                    parents_dict[idx] = 0
                    coef = 0
                stencil_mat[i, j] = coef
                idx += 1
        print(stencil_mat)
    if verbosity >= 2:
        print("Burgers angle = {:.3f}".format(burgers_angle))
        print("Inferred angle from stencil = {:.3f}".format(stencil_angle))
        print("Difference: {:.3f}".format(difference))
        print("")

    if plot_stencil_graph:
        plot_stencil(graph, v_matrix)

    if eng is None:
        quit_matlab_engine(eng)
    else:
        return (
            burgers_angle,
            stencil_angle,
            difference,
            alpha,
            beta,
            c,
            magnitude,
            init_diam,
            x_pos,
            y_pos,
        )


def random_experiments(
    magnitudes,
    n_experiments,
    path_to_save,
    checkpointing=1,
    checkpoint_iteration_interval=100,
    init_diam=-200,
    x_pos=0,
    y_pos=0,
    c=0.05,
    radius=3,
    alpha_pc=0.00001,
    dependence_threshold=None,
):
    results_dict = {
        "Given Angle": [],
        "Burgers Angle": [],
        "Stencil Angle": [],
        "Stencil Angle - Non-Negative": [],
        "weighted_avg": [],
        "Difference": [],
        "Difference - Non-Negative": [],
        "Difference - weighted_avg": [],
        "alpha": [],
        "beta": [],
        "c": [],
        "magnitude": [],
        "init_diam": [],
        "x_pos": [],
        "y_pos": [],
        "radius": [],
    }
    results_df = pd.DataFrame()

    angles = np.random.uniform(low=0.0, high=2 * np.pi, size=n_experiments)
    magnitudes = magnitudes

    iteration_num = 0
    eng = start_matlab_engine()
    print(
        "MATLAB engine started. Running {} random angles per {} magnitudes.".format(
            n_experiments, magnitudes
        )
    )
    for angle in angles:
        for magnitude in magnitudes:
            print("Experiment #{}".format(iteration_num))
            (
                burgers_angle,
                stencil_angle,
                stencil_angle_nonneg,
                weighted_avg,
                difference,
                difference_nn,
                diff_weighted_avg,
                alpha,
                beta,
                c,
                magnitude,
                init_diam,
                x_pos,
                y_pos,
            ) = run_specific_experiment(
                angle=angle,
                magnitude=magnitude,
                c=c,
                init_diam=init_diam,
                x_pos=x_pos,
                y_pos=y_pos,
                radius=radius,
                alpha_pc=alpha_pc,
                dependence_threshold=dependence_threshold,
                eng=eng,
                verbosity=2,
            )
            results_dict["Given Angle"].append(angle)
            results_dict["Burgers Angle"].append(burgers_angle)
            results_dict["Stencil Angle"].append(stencil_angle)
            results_dict["Stencil Angle - Non-Negative"].append(stencil_angle_nonneg)
            results_dict["weighted_avg"].append(weighted_avg)
            results_dict["Difference"].append(difference)
            results_dict["Difference - Non-Negative"].append(difference_nn)
            results_dict["Difference - weighted_avg"].append(diff_weighted_avg)
            results_dict["alpha"].append(alpha)
            results_dict["beta"].append(beta)
            results_dict["c"].append(c)
            results_dict["magnitude"].append(magnitude)
            results_dict["init_diam"].append(init_diam)
            results_dict["x_pos"].append(x_pos)
            results_dict["y_pos"].append(y_pos)
            results_dict["radius"].append(radius)

            if (iteration_num + 1) % checkpoint_iteration_interval == 0:
                if checkpointing:
                    print("Checkpointing...")
                    if path.exists(path_to_save):
                        try:
                            with open(path_to_save, "rb") as f:
                                results_df = pd.read_pickle(f)
                        except Exception as e:
                            print(f"An error occurred while reading the file: {e}.")
                    else:
                        print(f"{path_to_save} not found. A new file will be created.")

                    results_df = pd.concat(
                        [results_df, pd.DataFrame(results_dict)], ignore_index=True
                    )
                    try:
                        with open(path_to_save, "wb") as f:
                            results_df.to_pickle(f)
                    except Exception as e:
                        print(f"An error occurred while writing to the file: {e}.")

                    print("Length of results_df = {}\n".format(len(results_df)))

                    results_dict = {
                        "Given Angle": [],
                        "Burgers Angle": [],
                        "Stencil Angle": [],
                        "Stencil Angle - Non-Negative": [],
                        "weighted_avg": [],
                        "Difference": [],
                        "Difference - Non-Negative": [],
                        "Difference - weighted_avg": [],
                        "alpha": [],
                        "beta": [],
                        "c": [],
                        "magnitude": [],
                        "init_diam": [],
                        "x_pos": [],
                        "y_pos": [],
                        "radius": [],
                    }
                    results_df = pd.DataFrame()

            iteration_num += 1

    quit_matlab_engine(eng)

    if len(results_dict["Burgers Angle"]) > 0:
        print("Checkpointing did not collect all data. Concatenating remaining...")
        results_df = pd.concat(
            [results_df, pd.DataFrame(results_dict)], ignore_index=True
        )
        try:
            with open(path_to_save, "wb") as f:
                results_df.to_pickle(f)
        except Exception as e:
            print(f"An error occurred while writing to the file: {e}.")

    try:
        with open(path_to_save, "rb") as f:
            results_df = pd.read_pickle(f)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}.")
    print(results_df)


def sweeping_experiments(
    angles,
    magnitudes,
    path_to_save,
    perturbed=0,
    init_diam=-200,
    x_pos=0,
    y_pos=0,
    diffusion_coefs=[0.05],
    radius=3,
    spatial_res_Ns=[101],
    alpha_pc=0.00001,
    dependence_threshold=None,
    add_noise=False,
    noise_SNRs=[None],
    checkpointing=1,
    checkpoint_iteration_interval=100,
    from_checkpoint=0,
):
    if path.exists(path_to_save):
        print(
            "File {} already exists, picking up from checkpointed data...".format(
                path_to_save
            )
        )
        from_checkpoint = 1
        try:
            with open(path_to_save, "rb") as f:
                checkpointed_df = pd.read_pickle(f)
            print(
                "Checkpointed data read with {} experiments completed.".format(
                    len(checkpointed_df)
                )
            )
        except Exception as e:
            print(f"An error occurred while reading the file: {e}.")

    results_dict = {
        "Given Angle": [],
        "Burgers Angle": [],
        "Stencil Angle": [],
        "Difference": [],
        "Perturbed": [],
        "alpha": [],
        "beta": [],
        "c": [],
        "magnitude": [],
        "init_diam": [],
        "x_pos": [],
        "y_pos": [],
        "radius": [],
        "Spatial Resolution N": [],
        "SNR": [],
    }
    results_df = pd.DataFrame()

    iteration_num = 0
    if not add_noise:
        noise_SNRs = [None]

    eng = start_matlab_engine()

    print(
        "MATLAB engine started. Sweeping over {} angles from {} to {}, {} magnitudes, {} d.coefs, and {} SNRs.".format(
            len(angles),
            min(angles),
            max(angles),
            magnitudes,
            diffusion_coefs,
            noise_SNRs,
        )
    )

    for snr in noise_SNRs:
        for c in diffusion_coefs:
            for angle in angles:
                if perturbed:
                    angle = angle + np.random.uniform(-np.pi / 180, np.pi / 180)
                for magnitude in magnitudes:
                    for N in spatial_res_Ns:
                        if from_checkpoint:
                            # Check if parameter combination is in checkpointed dataframe.
                            if (
                                (checkpointed_df["Given Angle"] == angle)
                                & (checkpointed_df["magnitude"] == magnitude)
                                & (checkpointed_df["c"] == c)
                                & (checkpointed_df["Spatial Resolution N"] == N)
                                & (
                                    (checkpointed_df["SNR"] == snr)
                                    | (pd.isna(checkpointed_df["SNR"]))
                                )
                            ).any():
                                print(
                                    "Given angle {}, magnitude {}, diffusion coefficient {}, SNR {}, and spatial resolution N {}, have already been computed, skipping.".format(
                                        angle, magnitude, c, snr, N
                                    )
                                )
                                # iteration_num += 1
                                continue

                        print("Experiment #{}".format(iteration_num))
                        (
                            burgers_angle,
                            stencil_angle,
                            difference,
                            alpha,
                            beta,
                            c,
                            magnitude,
                            init_diam,
                            x_pos,
                            y_pos,
                        ) = run_specific_experiment(
                            angle=angle,
                            magnitude=magnitude,
                            c=c,
                            init_diam=init_diam,
                            x_pos=x_pos,
                            y_pos=y_pos,
                            radius=radius,
                            capture_N=N,
                            alpha_pc=alpha_pc,
                            dependence_threshold=dependence_threshold,
                            add_noise=add_noise,
                            noise_SNR=snr,
                            eng=eng,
                            verbosity=2,
                        )
                        results_dict["Given Angle"].append(angle)
                        results_dict["Burgers Angle"].append(burgers_angle)
                        results_dict["Stencil Angle"].append(stencil_angle)
                        results_dict["Difference"].append(difference)
                        results_dict["Perturbed"].append(perturbed)
                        results_dict["alpha"].append(alpha)
                        results_dict["beta"].append(beta)
                        results_dict["c"].append(c)
                        results_dict["magnitude"].append(magnitude)
                        results_dict["init_diam"].append(init_diam)
                        results_dict["x_pos"].append(x_pos)
                        results_dict["y_pos"].append(y_pos)
                        results_dict["radius"].append(radius)
                        results_dict["Spatial Resolution N"].append(N)
                        results_dict["SNR"].append(snr)

                        if (iteration_num + 1) % checkpoint_iteration_interval == 0:
                            if checkpointing:
                                print("Checkpointing...")
                                if path.exists(path_to_save):
                                    try:
                                        with open(path_to_save, "rb") as f:
                                            results_df = pd.read_pickle(f)
                                    except Exception as e:
                                        print(
                                            f"An error occurred while reading the file: {e}."
                                        )
                                else:
                                    print(
                                        f"{path_to_save} not found. A new file will be created."
                                    )

                                results_df = pd.concat(
                                    [results_df, pd.DataFrame(results_dict)],
                                    ignore_index=True,
                                )
                                try:
                                    with open(path_to_save, "wb") as f:
                                        results_df.to_pickle(f)
                                except Exception as e:
                                    print(
                                        f"An error occurred while writing to the file: {e}."
                                    )

                                print(
                                    "Length of results_df = {}\n".format(
                                        len(results_df)
                                    )
                                )

                                results_dict = {
                                    "Given Angle": [],
                                    "Burgers Angle": [],
                                    "Stencil Angle": [],
                                    "Difference": [],
                                    "Perturbed": [],
                                    "alpha": [],
                                    "beta": [],
                                    "c": [],
                                    "magnitude": [],
                                    "init_diam": [],
                                    "x_pos": [],
                                    "y_pos": [],
                                    "radius": [],
                                    "Spatial Resolution N": [],
                                    "SNR": [],
                                }

                        iteration_num += 1

    quit_matlab_engine(eng)

    if len(results_dict["Burgers Angle"]) > 0:
        print("Checkpointing did not collect all data. Concatenating remaining...")
        if path.exists(path_to_save):
            try:
                with open(path_to_save, "rb") as f:
                    results_df = pd.read_pickle(f)
            except Exception as e:
                print(f"An error occurred while reading the file: {e}.")
        else:
            print(f"{path_to_save} not found. A new file will be created.")
        results_df = pd.concat(
            [results_df, pd.DataFrame(results_dict)], ignore_index=True
        )
        try:
            with open(path_to_save, "wb") as f:
                results_df.to_pickle(f)
        except Exception as e:
            print(f"An error occurred while writing to the file: {e}.")

    try:
        with open(path_to_save, "rb") as f:
            results_df = pd.read_pickle(f)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}.")
    print(results_df)


if __name__ == "__main__":
    magnitudes = range(1, 11)
    angles = np.arange(0, 2 * np.pi, 2 * np.pi / 500)
    coefs = [
        0.05,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        # 0.6,
        # 0.7,
        # 0.8,
        # 0.9,
        # 1.0,
        # 2.0,
        # 3.0,
        # 4.0,
    ]
    SNRs = np.linspace(-30, 30, 5)
    sweeping_experiments(
        angles=angles,
        magnitudes=magnitudes,
        diffusion_coefs=coefs,
        path_to_save="./sweeping_burgers_500angles_mag0-10_interpRes25_alpha1.0_depNone_reggrid_noisy-30-30_5.pkl",
        alpha_pc=1.0,
        dependence_threshold=None,
        add_noise=True,
        noise_SNRs=SNRs,
        radius=3,
        spatial_res_Ns=[25],
    )