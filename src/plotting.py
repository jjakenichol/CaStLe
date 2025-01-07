"""
plotting.py

This module provides functions for visualizing gridded time series data and stencil graphs representing spatial dependencies. 
The functions are designed to generate plots that help in analyzing the temporal evolution of data across a grid 
and the relationships between different grid cells over time.

Functions:
----------
1. plot_dataset_timeseries(dataset: np.ndarray, save_path: str = None, showlabels: bool = False, alpha: float = 1.0, linewidth: float = 2) -> sns.FacetGrid
    Plots the time series data for each grid cell in a 3D numpy array representing a gridded space-time dataset. This function 
    helps in visualizing how data values change over time for each cell in the grid.

2. plot_stencil(stencil_graph: np.ndarray, stencil_val_matrix: np.ndarray = None, label_var_names: bool = True, show_colorbar: bool = False, label_colorbars: bool = False, fig: matplotlib.figure.Figure = None, ax: matplotlib.pyplot.Axes = None)
    Plots a custom graph based on the provided stencil graph and value matrix, typically used for visualizing spatial dependencies 
    and interactions between different grid cells in a 3x3 grid over time.
"""

import colorcet as cc
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.artist import Artist
from tigramite import plotting as tp


def plot_dataset_timeseries(dataset: np.ndarray, save_path: str = None, showlabels: bool = False, alpha: float = 1.0, linewidth: float = 2) -> sns.FacetGrid:
    """
    Plots the time series data for each grid cell in the dataset.

    Parameters:
    dataset (np.ndarray): A 3D numpy array representing the gridded space-time dataset.
                          The shape of the array should be (grid_size, grid_size, T),
                          where grid_size is the size of the grid and T is the number of time steps.
    save_path (str, optional): The path to save the plot. If None, the plot is not saved. Default is None.
    showlabels (bool, optional): Whether to show axis labels on the plot. Default is False.
    alpha (float, optional): The transparency level of the lines in the plot. Default is 1.0.
    linewidth (float, optional): The width of the lines in the plot. Default is 2.

    Returns:
    sns.FacetGrid: A seaborn FacetGrid object containing the plot.

    Example:
    import numpy as np
    import plotting

    # Generate a random dataset for demonstration
    grid_size = 4
    T = 100
    dataset = np.random.rand(grid_size, grid_size, T)

    # Plot the dataset time series
    g = plotting.plot_dataset_timeseries(dataset)
    plt.show()
    """
    grid_size = dataset.shape[0]  # Assumes a square grid.

    data_flat = dataset.reshape((dataset.shape[0] ** 2, dataset.shape[2])).transpose()
    dict_list = []
    for i in range(data_flat.shape[1]):
        for j in range(data_flat.shape[0]):
            data_dict = {}
            data_dict["Cell"] = i
            data_dict["Time Step"] = j
            data_dict["Data"] = data_flat[j, i]
            dict_list.append(data_dict)
    df = pd.DataFrame(data=dict_list)

    c_palette = sns.color_palette(
        cc.glasbey,
        n_colors=grid_size**2,
    )

    g = sns.relplot(data=df, x="Time Step", y="Data", col="Cell", col_wrap=grid_size, kind="line", hue="Cell", palette=c_palette, alpha=alpha, linewidth=linewidth)
    g.set_titles("")
    if showlabels:
        g.set_xlabels("Time Step")
        g.set_ylabels("Value")
    else:
        g.set_xlabels("")
        g.set_ylabels("")
        sns.despine(g, top=False, bottom=False, left=False, right=False)
        g.set(xticklabels=[])
        g.set(yticklabels=[])
        g.figure.subplots_adjust(wspace=0.0, hspace=0.0)
        for ax in g.axes.flat:
            ax.tick_params(left=False, bottom=False)  # Remove tick lines

    g._legend.remove()

    if save_path:
        plt.savefig(save_path, transparent=True)

    return g


def plot_stencil(
    stencil_graph: np.ndarray,
    stencil_val_matrix: np.ndarray = None,
    label_var_names: bool = True,
    show_colorbar: bool = False,
    label_colorbars: bool = False,
    fig: matplotlib.figure.Figure = None,
    ax: matplotlib.pyplot.Axes = None,
):
    """
    Plots a custom graph based on the provided stencil graph and value matrix.

    Parameters:
    -----------
    stencil_graph : numpy.ndarray
        An adjacency matrix representing the stencil graph. The shape of this array should be (9, 9, 2). The first two
         dimensions are for each position of the 3x3 stencil. The third dimension corresponds to two possible time lags
         (tau and tau-1).
    stencil_val_matrix : numpy.ndarray, optional
        The value matrix corresponding to the stencil graph. If provided, it will be used to determine the color of the links.
    show_colorbar : bool, optional
        Whether to show the colorbar in the plot. Default is False.
    lable_colorbar : bool, optional
        Whether to label the colorbar in the plot. Default is False.
    label_var_names : bool, optional
        Whether to label nodes with var_names. Default is True.
    fig : matplotlib.figure.Figure, optional
        The figure object to use for the plot. If None, a new figure will be created. Default is None.
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        The axis object to use for the plot. If None, a new axis will be created. Default is None.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axis object containing the plot.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    if label_var_names:
        var_names = ["NW", "N", "NE", "W", "C", "E", "SW", "S", "SE"]
    else:
        var_names = [""] * 9

    x_pos = list(np.array([[i for i in range(3)] for j in range(3)]).flatten())
    y_pos = [i for i in range(3) for j in range(3)]
    y_pos.reverse()
    node_positions = {
        "x": x_pos,
        "y": y_pos,
    }
    tp.plot_graph(
        fig_ax=(fig, ax),
        graph=stencil_graph,
        val_matrix=stencil_val_matrix,
        link_label_fontsize=0.0,
        var_names=var_names,
        node_pos=node_positions,
        show_colorbar=show_colorbar,
        link_colorbar_label="Cross-Dependence" if label_colorbars else None,
        node_colorbar_label="Inter-Dependence" if label_colorbars else None,
    )
    # Remove link labels which always have "1"
    for child in ax.get_children():
        if isinstance(child, matplotlib.text.Text):
            if child.get_text() == "1":
                Artist.set_visible(child, False)

    return fig, ax
