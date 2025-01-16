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

3. plot_heatmap_with_stencil(data_da: xr.DataArray, mark_source: bool = False, source_coords: Tuple[float, float] = (0.0, 0.0), start_index: Optional[int] = None, stop_index: Optional[int] = None, vmin: Optional[float] = None, vmax: Optional[float] = None, heatmap_cmap: str = "viridis", source_marker_offset: int = 3, source_marker_color: str = "white", lat_bounds: Optional[Tuple[float, float]] = None, lon_bounds: Optional[Tuple[float, float]] = None, fig_ax: Optional[Tuple[plt.Figure, plt.Axes]] = None, stencil_ax: Optional[plt.Axes] = None, graph: Optional[np.ndarray] = None, v_matrix: Optional[np.ndarray] = None) -> Tuple[plt.Figure, plt.Axes]
    Plots a heatmap and stencil on a map using the provided xarray DataArray. This function helps in visualizing spatial data 
    and marking specific locations on the map.
"""

import cartopy.crs as ccrs
import clif
import colorcet as cc
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.artist import Artist
from matplotlib.colors import ListedColormap
from tigramite import plotting as tp
from typing import Optional, Tuple


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
    style: str = "color",
    monochrome_edge_color: str = "black",
):
    """
    Plots a custom graph based on the provided stencil graph and value matrix.

    Parameters:
    -----------
    stencil_graph : numpy.ndarray
        An adjacency matrix representing the stencil graph. The shape of this array should be (9, 9, 2). The first two
         dimensions are for each position of the 3x3 stencil. The third dimension corresponds to two possible time lags
         (tau and tau-1).
    stencil_val_matrix : np.ndarray, optional
        The value matrix corresponding to the stencil graph. If provided, it will be used to determine the color of the links.
    show_colorbar : bool, optional
        Whether to show the colorbar in the plot. Default is False.
    label_colorbars : bool, optional
        Whether to label the colorbar in the plot. Default is False.
    label_var_names : bool, optional
        Whether to label nodes with var_names. Default is True.
    fig : matplotlib.figure.Figure, optional
        The figure object to use for the plot. If None, a new figure will be created. Default is None.
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        The axis object to use for the plot. If None, a new axis will be created. Default is None.
    style : str, optional
        The style of the plot. Can be "color" or "monochrome". Default is "color".
    monochrome_edge_color : str, optional
        The color for the monochrome style. Can be "black" or "white". Default is "black".

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

    if style == "color":
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
    elif style == "monochrome":
        cmap_N = 256
        white_vals = np.ones((cmap_N, 4))
        black_vals = np.zeros((cmap_N, 4))
        white_cmap = ListedColormap(white_vals)
        black_cmap = ListedColormap(black_vals)

        if monochrome_edge_color == "white":
            cmap_edges = white_cmap
        else:
            cmap_edges = black_cmap

        tp.plot_graph(
            fig_ax=(fig, ax),
            val_matrix=stencil_val_matrix.round(),
            graph=stencil_graph,
            link_label_fontsize=0.0,
            arrowhead_size=80,
            cmap_edges=cmap_edges,
            cmap_nodes="binary",
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

    ax.patch.set_alpha(0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    return fig, ax


def plot_heatmap_with_stencil(
    data_da: xr.DataArray,
    mark_source: bool = False,
    source_coords: Tuple[float, float] = (0.0, 0.0),
    start_index: Optional[int] = None,
    stop_index: Optional[int] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    heatmap_cmap: str = "viridis",
    source_marker_offset: int = 3,
    source_marker_color: str = "white",
    lat_bounds: Optional[Tuple[float, float]] = None,
    lon_bounds: Optional[Tuple[float, float]] = None,
    fig_ax: Optional[Tuple[plt.Figure, plt.Axes]] = None,
    stencil_ax: Optional[plt.Axes] = None,
    graph: Optional[np.ndarray] = None,
    v_matrix: Optional[np.ndarray] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots a heatmap and stencil on a map using the provided xarray DataArray.

    Parameters:
    -----------
    data_da : xarray.DataArray
        The input data array. It should contain dimensions for time, latitude, and longitude.
    mark_source : bool, optional
        Whether to mark the source location on the map. Default is False.
    source_coords : tuple, optional
        The coordinates for the source as a tuple (latitude, longitude). Default is (0.0, 0.0).
    start_index : int, optional
        The starting index for slicing the data array along the time dimension.
    stop_index : int, optional
        The stopping index for slicing the data array along the time dimension.
    vmin : float, optional
        The minimum value for the heatmap color scale. If None, it will be computed from the data.
    vmax : float, optional
        The maximum value for the heatmap color scale. If None, it will be computed from the data.
    heatmap_cmap : str, optional
        The colormap for the heatmap. Default is "viridis".
    source_marker_offset : int, optional
        The offset for the vertices of the triangle marking the source. Default is 3.
    source_marker_color : str, optional
        The color of the triangle marking the source. Default is 'white'.
    lat_bounds : tuple, optional
        The latitude bounds for clipping the data array as a tuple (lat_min, lat_max). If None, no clipping is performed.
    lon_bounds : tuple, optional
        The longitude bounds for clipping the data array as a tuple (lon_min, lon_max). If None, no clipping is performed.
    fig_ax : tuple, optional
        A tuple containing a matplotlib figure and axis to use for the plot. If None, a new figure and axis will be created.
    stencil_ax : matplotlib.axes._subplots.AxesSubplot, optional
        An axis to use for the stencil plot. If None, a new axis will be created.
    graph : numpy.ndarray, optional
        The stencil graph for plotting. If provided, it will be used to plot the stencil.
    v_matrix : numpy.ndarray, optional
        The stencil value matrix for plotting. If provided, it will be used to determine the color of the stencil links.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axis object containing the plot.
    """

    # Ensure that either start_index and stop_index or vmin and vmax are provided
    if (start_index is None or stop_index is None) and (vmin is None or vmax is None):
        raise ValueError("Either start_index and stop_index or vmin and vmax must be provided.")

    # Optionally clip the data array along latitudes and longitudes
    if lat_bounds is not None and lon_bounds is not None:
        lat_min, lat_max = lat_bounds
        lon_min, lon_max = lon_bounds
        lat_lon_clipper = clif.preprocessing.ClipTransform(dims=["lat", "lon"], bounds=[(lat_min, lat_max + 1), (lon_min, lon_max + 1)])
        data_da = lat_lon_clipper.fit_transform(data_da)

    # Calculate spatial mean and value range if vmin and vmax are not provided
    if vmin is None or vmax is None:
        spatial_ts_mean = np.mean(data_da, axis=(1, 2))[start_index:stop_index].values
        if vmin is None:
            vmin = spatial_ts_mean.min()
        if vmax is None:
            vmax = spatial_ts_mean.max()

    # Slice and transpose data array
    if start_index is not None and stop_index is not None:
        data_arr = data_da.values[start_index:stop_index, :, :]
    else:
        data_arr = data_da.values
    data_arr = np.transpose(data_arr, (1, 2, 0))

    # Get latitude and longitude values for plotting
    lats = data_da["lat"]
    lons = data_da["lon"]
    lons, lats = np.meshgrid(lons, lats)
    plot_lon_bounds = (lons.min(), lons.max())
    plot_lat_bounds = (lats.min(), lats.max())

    # Use provided figure and axis or create new ones
    if fig_ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection=ccrs.PlateCarree())
    else:
        fig, ax = fig_ax

    # Plot heatmap
    heatmap_data = np.mean(data_arr, axis=2)
    extent = [lons.min(), lons.max(), lats.min(), lats.max()]
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    hm = ax.pcolormesh(lons, lats, heatmap_data, vmin=vmin, vmax=vmax, cmap=heatmap_cmap, snap=False, alpha=1, rasterized=False)
    ax.coastlines(linewidth=2, color="black")

    # Plot triangle over source if mark_source is True
    if mark_source:
        if plot_lat_bounds[0] <= source_coords[0] <= plot_lat_bounds[1]:
            if plot_lon_bounds[0] <= source_coords[1] <= plot_lon_bounds[1]:
                tri_vertices = [
                    (source_coords[1], source_coords[0] + source_marker_offset),
                    (source_coords[1] - source_marker_offset, source_coords[0] - source_marker_offset),
                    (source_coords[1] + source_marker_offset, source_coords[0] - source_marker_offset),
                ]
                ax.add_patch(plt.Polygon(tri_vertices, color=source_marker_color, fill=True))

    # Plot stencil if provided
    if graph is not None and v_matrix is not None:
        if stencil_ax is None:
            stencil_ax = fig.add_subplot(projection=ccrs.PlateCarree())
        plot_stencil(fig=fig, ax=stencil_ax, stencil_graph=graph, stencil_val_matrix=v_matrix, style="monochrome", monochrome_edge_color="white")

    return fig, ax
