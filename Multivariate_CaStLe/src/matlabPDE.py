"""
MatlabPDE Module

This module defines the MatlabPDE class, which is used to manage and run MATLAB
scripts for solving Partial Differential Equations (PDEs). The class provides
methods to start and stop the MATLAB engine, run MATLAB scripts, animate the
solution, and plot fields at a given timestep.
"""

from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import pyplot as plt
from os import path
import matlab.engine
import numpy as np
from typing import Union


class MatlabPDE:
    def __init__(self):
        """
        Initialize the MatlabPDE class.
        """
        self.eng: matlab.engine.MatlabEngine = None

    def start_engine(self) -> matlab.engine.MatlabEngine:
        """
        Start the MATLAB engine.

        Returns:
            matlab.engine.MatlabEngine: The started MATLAB engine.
        """
        if self.eng is None:
            self.eng = matlab.engine.start_matlab()
        return self.eng

    def quit_engine(self) -> None:
        """
        Stop the MATLAB engine.
        """
        if self.eng is not None:
            self.eng.quit()
            self.eng = None

    def run_script(
        self,
        script_path: str,
        *args: Union[int, float, matlab.double],
        nargout: int = 1,
    ) -> matlab.double:
        """
        Run a MATLAB script with the provided arguments.

        Args:
            script_path (str): Path to the MATLAB script.
            *args (Union[int, float, matlab.double]): Arguments to pass to the MATLAB script.
            nargout (int): Number of output arguments. Default is 1.

        Returns:
            matlab.double: The output from the MATLAB script.
        """
        if self.eng is None:
            raise RuntimeError(
                "MATLAB engine is not started. Call start_engine() first."
            )

        script_dir = path.dirname(script_path)
        script_filename = path.basename(script_path).split(".")[0]
        if script_dir not in self.eng.path():
            self.eng.addpath(script_dir, nargout=0)

        # Convert arguments to MATLAB data type
        matlab_args = []
        for arg in args:
            if isinstance(arg, (int, float)):
                matlab_args.append(matlab.double([arg]))
            elif isinstance(arg, str):
                matlab_args.append(f"'{arg}'")
            else:
                matlab_args.append(arg)

        # Run the MATLAB script with the provided arguments
        run_string = f"{script_filename}({', '.join(map(str, matlab_args))})"
        out = self.eng.eval(run_string, nargout=nargout)

        return out

    def animate(self, data: np.ndarray, save_path: str = None) -> FuncAnimation:
        """
        Generalized animation function to handle any number of species and display time steps.
        Optionally saves the animation as a GIF to a given file path.

        Args:
            data (np.ndarray): The data to animate. Expected shape is (Xs, Ys, time, species).
            save_path (str, optional): Path to save the animation as a GIF. Defaults to None.

        Returns:
            FuncAnimation: The animation object.
        """
        num_species = data.shape[3]
        fig, axes = plt.subplots(1, num_species, figsize=(num_species * 5, 5))

        if num_species == 1:
            axes = [axes]

        ims = []
        vmin = data.min()
        vmax = data.max()
        for i in range(num_species):
            im = axes[i].imshow(
                data[:, :, 0, i],
                cmap="viridis",
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
            )
            axes[i].set_title(f"Species {chr(97 + i)}")
            axes[i].invert_yaxis()
            ims.append(im)
            # Add a colorbar for each subplot
            fig.colorbar(
                im, ax=axes[i], orientation="vertical", fraction=0.02, pad=0.04
            )

        # Add a text annotation for the time step
        time_text = fig.text(0.5, 0.92, "", ha="center", fontsize=12)

        def update(frame):
            for i in range(num_species):
                ims[i].set_array(data[:, :, frame, i])
            time_text.set_text(f"Time step: {frame}")
            return ims + [time_text]

        ani = FuncAnimation(fig, update, frames=range(data.shape[2]), blit=False)

        plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.1, wspace=0.4)

        # Save the animation as a GIF if save_path is provided
        if save_path:
            ani.save(save_path, writer=PillowWriter(fps=10))

        return ani

    def plot_fields(self, data: np.ndarray, timestep: int, title: str) -> None:
        """
        Plot the fields at a given timestep.

        Args:
            data (np.ndarray): The data to plot. Expected shape is (Xs, Ys, time, species).
            timestep (int): The timestep to plot.
            title (str): The title of the plot.
        """
        num_species = data.shape[3]
        fig, axes = plt.subplots(1, num_species)

        if num_species == 1:
            axes = [axes]

        ims = []
        vmin = data.min()
        vmax = data.max()
        for i in range(num_species):
            im = axes[i].imshow(
                data[:, :, timestep, i],
                cmap="viridis",
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
            )
            axes[i].set_title(f"Species {i + 1}")
            axes[i].invert_yaxis()  # Invert the Y-axis
            ims.append(im)

        # Add a common colorbar
        cbar = fig.colorbar(
            ims[0], ax=axes, orientation="vertical", fraction=0.02, pad=0.04
        )

        # Add a title
        fig.suptitle(title)

        plt.show()


# Example usage
if __name__ == "__main__":
    pde_solver = MatlabPDE()
    pde_solver.start_engine()

    # Define parameters
    init_center = matlab.double([-0.5, 0.5])
    diff_coeffs = matlab.double([0.05, 0.05])
    vel_coeffs = matlab.double([4.0, 4.0])
    react_rate = 2.0
    t = matlab.double(np.linspace(0, 0.4, 401).tolist())
    Hmax = 0.02
    plot = False
    verbose = True

    # Path to the MATLAB function
    current_file_path = path.abspath(__file__)
    current_dir = path.dirname(current_file_path)
    MVADR_matlab_path = path.join(
        current_dir, "../matlab_multivar/matlab_multivar/ADR_Driver_func.m"
    )

    # Run the MATLAB function
    solution = pde_solver.run_script(
        MVADR_matlab_path,
        init_center,
        diff_coeffs,
        vel_coeffs,
        react_rate,
        t,
        Hmax,
        plot,
        verbose,
        nargout=1,
    )

    # Convert the solution to a numpy array
    solution_np = np.array(solution)

    # Animate the results
    pde_solver.animate(solution_np)

    # Stop the MATLAB engine
    pde_solver.quit_engine()
