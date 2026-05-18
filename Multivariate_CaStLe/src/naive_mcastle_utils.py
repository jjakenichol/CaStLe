"""
Cartesian U-CaStLe: Independent U-CaStLe + Non-Spatial PC/PCMCI/DYNOTEARS Combination

Authors: Anonymous

This module implements the Cartesian U-CaStLe baseline multivariate causal discovery method
that combines:
  1. Independent univariate CaStLe (U-CaStLe) runs on each variable's 2D spatial field,
     producing per-variable spatial stencils.
  2. A non-spatial PC/PCMCI/DYNOTEARS run on spatially-aggregated (mean) timeseries of each
     variable, producing inter-variable causal links without spatial structure.
  3. A combined multivariate stencil in the same (9*N, 9*N, 2) format as M-CaStLe, where:
     - Intra-variable 9×9 diagonal blocks are filled from the univariate stencils.
     - Inter-variable links connect the center nodes of each variable's neighborhood block,
       and are included only when the corresponding link exists in the non-spatial graph.

This serves as a competitive baseline between pure spatial methods (U-CaStLe applied
per-variable) and fully-coupled M-CaStLe, without requiring joint spatial causal discovery.
The "Cartesian" name reflects that per-variable U-CaStLe results are combined independently
(Cartesian combination) rather than jointly discovered.

Stencil Graph Convention (same as mcastle_utils.py):
    Shape: (9*N, 9*N, 2), where N = number of variables.
    - Third dimension index 0: lag-0 (contemporaneous) relationships.
    - Third dimension index 1: lag-1 (lagged) relationships.
    - Center node for variable k: index 4 + 9*k.
    - graph[i, j, tau] = "-->" means node i at t-tau causes node j at t.
    - graph[j, i, 0] = "<--" is the complementary entry for contemporaneous directed links.
"""

import numpy as np
import pandas as pd
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.independence_tests_base import CondIndTest
from typing import Optional, Tuple

from causalnex.structure.dynotears import from_pandas_dynamic

from mcastle_utils import (
    mv_CaStLe_PC,
    mv_CaStLe_DYNOTEARS,
    get_graph_from_structure_model,
)


def _spatially_aggregate(data: np.ndarray) -> np.ndarray:
    """
    Compute the spatial mean of each variable over the X and Y dimensions.

    Parameters
    ----------
    data : np.ndarray
        Shape (variable_n, X, Y, T).

    Returns
    -------
    np.ndarray
        Shape (T, variable_n). Each column is the spatial mean timeseries for one variable.
    """
    return data.mean(axis=(1, 2)).T  # (variable_n, T) -> (T, variable_n)


def _run_nonspatial_pcmci(
    spatial_mean: np.ndarray,
    cond_ind_test: CondIndTest,
    pc_alpha: float,
    graph_p_threshold: float,
    min_tau: int,
    cd_function: str,
    fdr_method: Optional[str],
) -> dict:
    """
    Run PC/PCMCI on spatially-aggregated (non-spatial) variable timeseries.

    Parameters
    ----------
    spatial_mean : np.ndarray
        Shape (T, variable_n).
    cond_ind_test : CondIndTest
        Tigramite conditional independence test instance.
    pc_alpha : float
        Significance level for the CD algorithm.
    graph_p_threshold : float
        P-value threshold used when reconstructing the graph after FDR correction.
    min_tau : int
        Minimum lag (0 or 1).
    cd_function : str
        Tigramite method name: "run_pcalg" or "run_pcmci".
    fdr_method : str or None
        FDR correction method. "bh" for Benjamini-Hochberg. None for no correction.

    Returns
    -------
    dict
        Tigramite results dict with keys "graph", "val_matrix", and "p_matrix"
        (or "q_matrix" if FDR was applied). Graph shape: (variable_n, variable_n, 2).
    """
    max_tau = 1
    pcmci_df = pp.DataFrame(spatial_mean)
    pcmci = PCMCI(dataframe=pcmci_df, cond_ind_test=cond_ind_test, verbosity=0)

    results = getattr(pcmci, cd_function)(
        tau_min=min_tau,
        tau_max=max_tau,
        pc_alpha=pc_alpha,
    )

    if fdr_method:
        tigramite_fdr = "fdr_bh" if fdr_method == "bh" else fdr_method
        q_matrix = pcmci.get_corrected_pvalues(
            p_matrix=results["p_matrix"],
            tau_min=min_tau,
            tau_max=max_tau,
            fdr_method=tigramite_fdr,
        )
        results["graph"] = pcmci.get_graph_from_pmatrix(
            p_matrix=q_matrix,
            alpha_level=graph_p_threshold,
            tau_min=min_tau,
            tau_max=max_tau,
        )
        results["q_matrix"] = q_matrix

    return results


def _run_nonspatial_dynotears(
    spatial_mean: np.ndarray,
    dependence_threshold: float,
    lambda_w: float,
    lambda_a: float,
    max_iter: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run DYNOTEARS on spatially-aggregated (non-spatial) variable timeseries.

    Lag-0 edges are banned (only lag-1 temporal edges are allowed). The returned
    graph and val_matrix follow the same Tigramite string-graph convention used
    throughout the CaStLe codebase.

    Parameters
    ----------
    spatial_mean : np.ndarray
        Shape (T, variable_n).
    dependence_threshold : float
        Absolute edge-weight threshold; edges below this value are discarded.
    lambda_w : float
        L1 regularization for contemporaneous (lag-0) edges.
    lambda_a : float
        L1 regularization for lagged (lag-1) edges.
    max_iter : int
        Maximum DYNOTEARS dual-ascent iterations.

    Returns
    -------
    graph : np.ndarray
        Shape (variable_n, variable_n, 2). Tigramite string graph.
    val_matrix : np.ndarray
        Shape (variable_n, variable_n, 2). Edge weights.
    """
    variable_n = spatial_mean.shape[1]
    col_names = [str(i) for i in range(variable_n)]
    df = pd.DataFrame(data=spatial_mean, columns=col_names)

    # Ban all lag-0 edges; only lag-1 temporal links are permitted
    taboo_edges = [(0, i, j) for i in col_names for j in col_names]

    sm = from_pandas_dynamic(
        df,
        p=1,
        lambda_w=lambda_w,
        lambda_a=lambda_a,
        max_iter=max_iter,
        w_threshold=dependence_threshold,
        tabu_edges=taboo_edges,
    )

    graph, val_matrix = get_graph_from_structure_model(sm)

    # Ensure shape is (variable_n, variable_n, 2) even when no edges found
    if graph.shape[0] == 0:
        graph = np.full((variable_n, variable_n, 2), fill_value="", dtype="<U3")
        val_matrix = np.zeros((variable_n, variable_n, 2))

    return graph, val_matrix


def _assemble_combined_stencil(
    variable_n: int,
    ucastle_graphs: list,
    ucastle_vals: list,
    ucastle_ps: list,
    inter_graph: np.ndarray,
    inter_val: np.ndarray,
    inter_p: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assemble the (9*N, 9*N, 2) Cartesian U-CaStLe stencil from per-variable
    U-CaStLe stencils and a non-spatial inter-variable graph.

    Intra-variable 9×9 diagonal blocks come from U-CaStLe. Inter-variable links
    are placed between center nodes (4 + 9*k) when the non-spatial graph has a
    corresponding edge.

    Parameters
    ----------
    variable_n : int
    ucastle_graphs : list of np.ndarray, each (9, 9, 2)
    ucastle_vals : list of np.ndarray, each (9, 9, 2)
    ucastle_ps : list of np.ndarray or None, each (9, 9, 2)
    inter_graph : np.ndarray, shape (variable_n, variable_n, 2)
    inter_val : np.ndarray, shape (variable_n, variable_n, 2)
    inter_p : np.ndarray or None, shape (variable_n, variable_n, 2)

    Returns
    -------
    combined_graph, combined_val, combined_p : each np.ndarray (9*N, 9*N, 2)
    """
    n_nodes = 9 * variable_n
    combined_graph = np.full((n_nodes, n_nodes, 2), fill_value="", dtype="<U3")
    combined_val = np.zeros((n_nodes, n_nodes, 2))
    combined_p = np.ones((n_nodes, n_nodes, 2))

    # Intra-variable diagonal blocks
    for k in range(variable_n):
        bs, be = 9 * k, 9 * k + 9
        combined_graph[bs:be, bs:be, :] = ucastle_graphs[k]
        combined_val[bs:be, bs:be, :] = ucastle_vals[k]
        if ucastle_ps[k] is not None:
            combined_p[bs:be, bs:be, :] = ucastle_ps[k]

    # Inter-variable center-to-center links
    for var_i in range(variable_n):
        for var_j in range(variable_n):
            if var_i == var_j:
                continue
            ci, cj = 4 + 9 * var_i, 4 + 9 * var_j
            for tau in range(2):
                link = inter_graph[var_i, var_j, tau]
                if link != "":
                    combined_graph[ci, cj, tau] = link
                    combined_val[ci, cj, tau] = inter_val[var_i, var_j, tau]
                    if inter_p is not None:
                        combined_p[ci, cj, tau] = inter_p[var_i, var_j, tau]

    return combined_graph, combined_val, combined_p


def cartesian_ucastle_pc(
    data: np.ndarray,
    cond_ind_test: CondIndTest,
    pc_alpha: float,
    graph_p_threshold: float,
    min_tau: int = 1,
    rows_inverted: bool = False,
    cd_function: str = "run_pcalg",
    fdr_method: Optional[str] = None,
    dependencies_wrap: bool = False,
    allow_center_directed_links: bool = False,
    verbose: int = 1,
) -> dict:
    """
    Cartesian U-CaStLe: combine independent U-CaStLe stencils with non-spatial PC/PCMCI.

    Algorithm
    ---------
    1. For each variable k, run U-CaStLe (mv_CaStLe_PC with variable_n=1) on data[k:k+1]
       to obtain a per-variable spatial stencil of shape (9, 9, 2).
    2. Spatially aggregate each variable to a single timeseries (spatial mean over X, Y),
       yielding a (T, variable_n) array.
    3. Run PC/PCMCI on the spatially aggregated data to obtain a non-spatial inter-variable
       causal graph of shape (variable_n, variable_n, 2).
    4. Assemble a combined stencil of shape (9*variable_n, 9*variable_n, 2):
       - Intra-variable diagonal 9×9 blocks filled from each variable's univariate stencil.
       - Inter-variable links placed between center nodes (4 + 9*k) when the non-spatial
         graph has a corresponding edge.

    Parameters
    ----------
    data : np.ndarray
        Shape (variable_n, X, Y, T).
    cond_ind_test : CondIndTest
        Tigramite conditional independence test (used for both U-CaStLe and non-spatial steps).
    pc_alpha : float
        Significance level for conditional independence tests.
    graph_p_threshold : float
        P-value threshold for graph reconstruction after FDR correction.
    min_tau : int, optional
        Minimum lag (0 or 1). Defaults to 1.
    rows_inverted : bool, optional
        Whether spatial grid rows are inverted. Defaults to False.
    cd_function : str, optional
        Tigramite causal discovery method: "run_pcalg" or "run_pcmci". Defaults to "run_pcalg".
    fdr_method : str or None, optional
        FDR correction: "bh" for Benjamini-Hochberg, None for no correction. Defaults to None.
    dependencies_wrap : bool, optional
        Toroidal boundary conditions for U-CaStLe. Defaults to False.
    allow_center_directed_links : bool, optional
        Allow directed center-to-center links within U-CaStLe. Defaults to False.
    verbose : int, optional
        Verbosity level. Defaults to 1.

    Returns
    -------
    dict
        - "graph"              : np.ndarray (9*N, 9*N, 2) — combined stencil graph
        - "val_matrix"         : np.ndarray (9*N, 9*N, 2) — test statistic matrix
        - "p_matrix"           : np.ndarray (9*N, 9*N, 2) — p-value matrix (1.0 where not estimated)
        - "ucastle_results"    : list of dict — per-variable U-CaStLe results
        - "non_spatial_results": dict — non-spatial PC/PCMCI results
    """
    assert len(data.shape) == 4, "data must have shape (variable_n, X, Y, T)"

    variable_n = data.shape[0]

    # Step 1: U-CaStLe independently on each variable
    ucastle_results = []
    for k in range(variable_n):
        if verbose:
            print(f"Running U-CaStLe on variable {k} of {variable_n}...")
        result = mv_CaStLe_PC(
            data=data[k : k + 1],
            cond_ind_test=cond_ind_test,
            pc_alpha=pc_alpha,
            graph_p_threshold=graph_p_threshold,
            min_tau=min_tau,
            rows_inverted=rows_inverted,
            cd_function=cd_function,
            fdr_method=fdr_method,
            dependencies_wrap=dependencies_wrap,
            allow_center_directed_links=allow_center_directed_links,
            verbose=0,
        )
        ucastle_results.append(result)

    # Step 2: Spatial mean → (T, variable_n)
    spatial_mean = _spatially_aggregate(data)

    if verbose:
        print(
            f"Running non-spatial {cd_function} on spatially aggregated data (shape {spatial_mean.shape})..."
        )

    # Step 3: Non-spatial PC/PCMCI
    non_spatial_results = _run_nonspatial_pcmci(
        spatial_mean=spatial_mean,
        cond_ind_test=cond_ind_test,
        pc_alpha=pc_alpha,
        graph_p_threshold=graph_p_threshold,
        min_tau=min_tau,
        cd_function=cd_function,
        fdr_method=fdr_method,
    )

    inter_graph = non_spatial_results["graph"]
    inter_val = non_spatial_results["val_matrix"]
    inter_p = non_spatial_results.get("q_matrix", non_spatial_results.get("p_matrix"))

    ucastle_graphs = [r["graph"] for r in ucastle_results]
    ucastle_vals = [r["val_matrix"] for r in ucastle_results]
    ucastle_ps = [r.get("q_matrix", r.get("p_matrix")) for r in ucastle_results]

    # Step 4: Assemble combined stencil
    combined_graph, combined_val, combined_p = _assemble_combined_stencil(
        variable_n,
        ucastle_graphs,
        ucastle_vals,
        ucastle_ps,
        inter_graph,
        inter_val,
        inter_p,
    )

    if verbose:
        print("Cartesian U-CaStLe stencil assembled.")

    return {
        "graph": combined_graph,
        "val_matrix": combined_val,
        "p_matrix": combined_p,
        "ucastle_results": ucastle_results,
        "non_spatial_results": non_spatial_results,
    }


def cartesian_ucastle_dynotears(
    data: np.ndarray,
    dependence_threshold: float = 0.0,
    rows_inverted: bool = False,
    dependencies_wrap: bool = False,
    allow_center_directed_links: bool = False,
    lambda_w: Optional[float] = None,
    lambda_a: float = 0.1,
    max_iter: int = 100,
    verbose: int = 1,
) -> dict:
    """
    Cartesian U-CaStLe with DYNOTEARS: combine independent U-CaStLe-DYNOTEARS
    stencils with non-spatial DYNOTEARS on spatially aggregated timeseries.

    Algorithm
    ---------
    1. For each variable k, run U-CaStLe-DYNOTEARS (mv_CaStLe_DYNOTEARS with
       variable_n=1) on data[k:k+1] to obtain a per-variable spatial stencil (9, 9, 2).
    2. Spatially aggregate to a (T, variable_n) array.
    3. Run DYNOTEARS on the aggregated data for a non-spatial inter-variable graph.
    4. Assemble the combined (9*N, 9*N, 2) stencil: intra-variable diagonal blocks
       from U-CaStLe, inter-variable center-to-center links from non-spatial DYNOTEARS.

    Parameters
    ----------
    data : np.ndarray
        Shape (variable_n, X, Y, T).
    dependence_threshold : float, optional
        Absolute edge-weight cutoff for both U-CaStLe and non-spatial steps. Defaults to 0.0
        (keep all edges; sparsity controlled by lambda_a).
    rows_inverted : bool, optional
        Whether spatial grid rows are inverted. Defaults to False.
    dependencies_wrap : bool, optional
        Toroidal boundary conditions for U-CaStLe. Defaults to False.
    allow_center_directed_links : bool, optional
        Allow directed center-to-center links within U-CaStLe. Defaults to False.
    lambda_w : float or None, optional
        L1 regularization for lag-0 edges. Defaults to lambda_a if None.
    lambda_a : float, optional
        L1 regularization for lag-1 edges (primary sparsity control). Defaults to 0.1.
    max_iter : int, optional
        Maximum DYNOTEARS iterations. Defaults to 100.
    verbose : int, optional
        Verbosity level. Defaults to 1.

    Returns
    -------
    dict
        - "graph"              : np.ndarray (9*N, 9*N, 2) — combined stencil graph
        - "val_matrix"         : np.ndarray (9*N, 9*N, 2) — edge weight matrix
        - "ucastle_results"    : list of tuple (graph, val_matrix) per variable
        - "non_spatial_results": dict with "graph" and "val_matrix" keys
    """
    assert len(data.shape) == 4, "data must have shape (variable_n, X, Y, T)"

    if lambda_w is None:
        lambda_w = lambda_a

    variable_n = data.shape[0]

    # Step 1: U-CaStLe-DYNOTEARS independently on each variable
    ucastle_results = []
    for k in range(variable_n):
        if verbose:
            print(f"Running U-CaStLe-DYNOTEARS on variable {k} of {variable_n}...")
        graph_k, val_k = mv_CaStLe_DYNOTEARS(
            data=data[k : k + 1],
            rows_inverted=rows_inverted,
            dependence_threshold=dependence_threshold,
            dependencies_wrap=dependencies_wrap,
            allow_center_directed_links=allow_center_directed_links,
            lambda_w=lambda_w,
            lambda_a=lambda_a,
            max_iter=max_iter,
            verbose=0,
        )
        ucastle_results.append({"graph": graph_k, "val_matrix": val_k})

    # Step 2: Spatial mean → (T, variable_n)
    spatial_mean = _spatially_aggregate(data)

    if verbose:
        print(
            f"Running non-spatial DYNOTEARS on spatially aggregated data (shape {spatial_mean.shape})..."
        )

    # Step 3: Non-spatial DYNOTEARS
    inter_graph, inter_val = _run_nonspatial_dynotears(
        spatial_mean=spatial_mean,
        dependence_threshold=dependence_threshold,
        lambda_w=lambda_w,
        lambda_a=lambda_a,
        max_iter=max_iter,
    )

    non_spatial_results = {"graph": inter_graph, "val_matrix": inter_val}

    ucastle_graphs = [r["graph"] for r in ucastle_results]
    ucastle_vals = [r["val_matrix"] for r in ucastle_results]
    ucastle_ps = [None] * variable_n  # DYNOTEARS has no p-values

    # Step 4: Assemble combined stencil (no p-matrix for DYNOTEARS)
    combined_graph, combined_val, _ = _assemble_combined_stencil(
        variable_n,
        ucastle_graphs,
        ucastle_vals,
        ucastle_ps,
        inter_graph,
        inter_val,
        inter_p=None,
    )

    if verbose:
        print("Cartesian U-CaStLe-DYNOTEARS stencil assembled.")

    return {
        "graph": combined_graph,
        "val_matrix": combined_val,
        "ucastle_results": ucastle_results,
        "non_spatial_results": non_spatial_results,
    }


# Backward-compatibility alias
naive_mv_CaStLe_PC = cartesian_ucastle_pc
