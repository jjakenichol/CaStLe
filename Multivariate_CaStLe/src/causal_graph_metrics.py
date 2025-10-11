"""
graph_metrics.py

This script provides a set of functions to compute various metrics for evaluating the quality and characteristics of graphs,
particularly in the context of causal inference and machine learning. The metrics include the Matthews correlation coefficient (MCC),
F1 score, false discovery rate (FDR), and various graph structural metrics.

Functions
---------
get_confusion_matrix(true_graph: np.ndarray, discovered_graph: np.ndarray) -> tuple
    Calculate the confusion matrix for comparing a true graph with a discovered graph.

F1_score(true_graph: np.ndarray = None, discovered_graph: np.ndarray = None, TP: int = None, FP: int = None, FN: int = None, TN: int = None) -> tuple
    Computes the F1 score of a given graph relative to a reference, "ground-truth" graph.

false_discovery_rate(FP: int, TP: int) -> float
    Computes the False Discovery Rate (FDR) given false positive (FP) and true positive (TP) counts.

get_graph_metrics(graph: np.ndarray) -> tuple
    Returns a set of graph functions for both the summary and full timeseries graphs.

matthews_correlation_coefficient(true_graph: np.ndarray = None, discovered_graph: np.ndarray = None, TP: int = None, FP: int = None, FN: int = None, TN: int = None) -> float
    Compute the Matthews correlation coefficient (MCC) (A.K.A. Phi coefficient).
"""

import numpy as np


def get_confusion_matrix(true_graph: np.ndarray, discovered_graph: np.ndarray) -> tuple:
    """
    Calculate the confusion matrix for comparing a true graph with a discovered graph.

    Parameters
    ----------
    true_graph : np.ndarray
        Array representing the true graph. Assumes the graph only contains empty strings ("") and existing forward links ("-->").
    discovered_graph : np.ndarray
        Array representing the discovered graph. Assumes the graph only contains empty strings ("") and existing forward links ("-->").

    Returns
    -------
    tuple
        A tuple containing the counts of true positives, false positives, false negatives, true negatives.

    Raises
    ------
    AssertionError
        If the shapes of true_graph and discovered_graph do not agree.

    Notes
    -----
    The confusion matrix is calculated by comparing the links in the true graph with the links in the discovered graph.
    True positives (TP) are the links that exist in both graphs.
    False positives (FP) are the links that exist in the discovered graph but not in the true graph.
    False negatives (FN) are the links that exist in the true graph but not in the discovered graph.
    True negatives (TN) are the links that do not exist in either graph.

    Missing links in the discovered graph are represented by empty strings ("").

    If a truth condition is not met during the comparison, an error message is printed and the function returns -np.inf for all counts.
    """
    assert (
        true_graph.shape == discovered_graph.shape
    ), "Graph shapes do not agree. true_graph.shape={}, reconstructed_full_graph.shape={}".format(true_graph.shape, discovered_graph.shape)

    TP = 0  # True positives
    FP = 0  # False positives
    FN = 0  # False negatives
    TN = 0  # True negatives
    for a, x in zip(true_graph, discovered_graph):
        for b, y in zip(a, x):
            for c, z in zip(b, y):
                # c is the true graph's link and z is the discovered graph's link.
                # If c is z, there is a true positive
                if c == z and z != "":
                    TP += 1
                # If c is not z and z is not a missing link in z, there is a false positive
                elif c != z and z != "":
                    FP += 1
                # If if c is not z and there is a missing link in z, there is a false negative
                elif c != z and z == "":
                    FN += 1
                elif c == z and z == "":
                    TN += 1
                else:
                    print("A truth condition was not met!")
                    return -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf

    return TP, FP, FN, TN


def F1_score(
    *, true_graph: np.ndarray = None, discovered_graph: np.ndarray = None, TP: int = None, FP: int = None, FN: int = None, TN: int = None
) -> tuple:
    """
    Computes the F1 score of a given graph relative to a reference, "ground-truth" graph.

    This function computes the true positive (TP), false positive (FP), and false negative (FN) of the test matrix given the reference matrix.
    It uses these to compute precision (P) and recall (R), defined as:
    P = TP / (TP + FP)
    R = TP / (TP + FN)

    The F1 score is then:
    F1 = (2 * P * R) / (P + R)

    Parameters
    ----------
    true_graph : array of shape [N, N, tau_max+1], optional
        The reference/"ground-truth" graph.
    discovered_graph : array of shape [N, N, tau_max+1], optional
        The graph to be measured/scored.
    TP : int, optional
        The true positive count.
    FP : int, optional
        The false positive count.
    FN : int, optional
        The false negative count.
    TN : int, optional
        The true negative count. Not used for F1 score computation but useful to capture nonetheless.

    Returns
    -------
    tuple
        [0] the F1 score computed from precision and recall of the test matrix given the reference matrix.
        [1] the precision.
        [2] the recall.
        [3] the true positive count.
        [4] the false positive count.
        [5] the false negative count.
        [6] the true negative count.

    Example
    -------
    >>> true_graph = np.array([[["-->", ""], ["", "-->"]], [["", ""], ["-->", ""]]])
    >>> discovered_graph = np.array([[["-->", ""], ["", "-->"]], [["", ""], ["", ""]]])
    >>> F1, P, R, TP, FP, FN, TN = F1_score(true_graph, discovered_graph)
    >>> print(F1, P, R, TP, FP, FN, TN)
    0.8 1.0 0.6666666666666666 2 0 1 5
    >>> F1, P, R, TP, FP, FN, TN = F1_score(TP=2, FP=0, FN=1, TN=3)
    >>> print(F1, P, R, TP, FP, FN, TN)
    0.8 1.0 0.6666666666666666 2 0 1 3
    """
    if true_graph is not None and discovered_graph is not None:
        TP, FP, FN, TN = get_confusion_matrix(true_graph, discovered_graph)
    else:
        assert (
            TP is not None and FP is not None and FN is not None and TN is not None
        ), "Either (true_graph, discovered_graph) or (TP, FP, FN, TN) must be provided."

    if TP == 0:
        if (FP == 0) and (FN == 0):
            F1, P, R = 1, 1, 1
        else:
            F1, P, R = 0, 0, 0
    else:
        P = TP / (TP + FP)
        R = TP / (TP + FN)
        F1 = (2 * P * R) / (P + R)

    return F1, P, R, TP, FP, FN, TN


def false_discovery_rate(FP: int, TP: int) -> float:
    """Computes the False Discovery Rate (FDR) given false positive (FP) and true positive (TP) counts.

    FDR is defined as:
    0 if FP + TP == 0
    FP / (FP + TP) otherwise

    Parameters
    ----------
    FP : int
        a count of false positives.
    TP : int
        a count of true positives.

    Returns
    -------
    float
        False Discovery Rate
    """
    if FP + TP == 0:
        return 0
    return FP / (FP + TP)


def get_graph_metrics(graph: np.ndarray) -> tuple:
    """Returns a set of graph functions for both the summary and full timeseries graphs

    Parameters
    ----------
    graph : array of shape [N, N, tau_max+1]
        a string array with links '-->' representing a causal graph.

    Returns
    -------
    tuple of lists ([])
        [0] List of graph metrics from the collapsed "summary" graph (non-timeseries graph)
            [0][0] n_nodes (int) : number of nodes in the summary graph
            [0][1] n_edges (int) : number of edges in the summary graph
            [0][2] max_inDegree (float) : maximum number of directed edges coming into a node in the summary graph
            [0][3] avg_inDegree (float) : average number of directed edges coming into a node in the summary graph
            [0][4] max_outDegree (float) : maximum number of directed edges coming out of a node in the summary graph
            [0][5] avg_outDegree (float) : average number of directed edges coming out of a node in the summary graph
        [1] List of graph metrics from the full timeseries graph
            [1][0] n_nodes_TS (int) : number of nodes in the timeseries graph
            [1][1] n_edges_TS (int) : number of edges in the timeseries graph
            [1][2] max_inDegree_TS (float) : maximum number of directed edges coming into a node in the timeseries graph
            [1][3] avg_inDegree_TS (float) : average number of directed edges coming into a node in the timeseries graph
            [1][4] max_outDegree_TS (float) : maximum number of directed edges coming out of a node in the timeseries graph
            [1][5] avg_outDegree_TS (float) : average number of directed edges coming out of a node in the timeseries graph
    """
    n_nodes = graph.shape[0]  # number of variables
    n_nodes_TS = graph.shape[0] * graph.shape[2]  # number of variables * number of lags

    n_edges = 0
    n_edges_TS = 0
    in_degrees = np.zeros((n_nodes))
    out_degrees = np.zeros((n_nodes))
    in_degrees_TS = np.zeros((n_nodes, graph.shape[2]))
    out_degrees_TS = np.zeros((n_nodes, graph.shape[2]))
    for i in range(graph.shape[0]):
        for j in range(graph.shape[1]):
            for lag in range(graph.shape[2]):
                if graph[i, j, lag] != "":
                    n_edges_TS += 1
                    out_degrees_TS[i, lag] += 1
                    in_degrees_TS[j, lag] += 1
                    if i != j:
                        n_edges += 1
                        out_degrees[i] += 1
                        in_degrees[j] += 1

    max_inDegree = np.max(in_degrees)
    avg_inDegree = np.mean(in_degrees)
    max_outDegree = np.max(out_degrees)
    avg_outDegree = np.mean(out_degrees)

    max_inDegree_TS = np.max(in_degrees_TS)
    avg_inDegree_TS = np.mean(in_degrees_TS)
    max_outDegree_TS = np.max(out_degrees_TS)
    avg_outDegree_TS = np.mean(out_degrees_TS)

    return [n_nodes, n_edges, max_inDegree, avg_inDegree, max_outDegree, avg_outDegree], [
        n_nodes_TS,
        n_edges_TS,
        max_inDegree_TS,
        avg_inDegree_TS,
        max_outDegree_TS,
        avg_outDegree_TS,
    ]


def matthews_correlation_coefficient(
    *, true_graph: np.ndarray = None, discovered_graph: np.ndarray = None, TP: int = None, FP: int = None, FN: int = None, TN: int = None
) -> float:
    """
    Compute the Matthews correlation coefficient (MCC) (A.K.A. Phi coefficient)

    In machine learning, it is used as a measure of the quality of binary (two-class) classifications.
    In statistics, it is called the phi coefficient (or mean square contingency coefficient and denoted by φ or rφ) is a measure of association for two binary variables.

    Traditionally defined as:
    MCC = (TPxTN - FPxFN)/(sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)))

    This function is undefined in any of the four cases in which one of the denominator terms equal zero.
    This implementation handles these cases by reasoning through what the MCC *should* be in each of the undefined cases.
    See https://stats.stackexchange.com/a/603924/240083 for a more elaborated explanation.

    This function will always return a float between [-1, 1] for all possible TP, FP, FN, and TN counts.

    Parameters
    ----------
    true_graph : array of shape [N, N, tau_max+1], optional
        The reference/"ground-truth" graph.
    discovered_graph : array of shape [N, N, tau_max+1], optional
        The graph to be measured/scored.
    TP : int, optional
        The true positive count.
    FP : int, optional
        The false positive count.
    FN : int, optional
        The false negative count.
    TN : int, optional
        The true negative count.

    Returns
    -------
    float
        The Matthews correlation coefficient between [-1, 1]

    Example
    -------
    >>> true_graph = np.array([[["-->", ""], ["", "-->"]], [["", ""], ["-->", ""]]])
    >>> discovered_graph = np.array([[["-->", ""], ["", "-->"]], [["", ""], ["", ""]]])
    >>> MCC = matthews_correlation_coefficient(true_graph=true_graph, discovered_graph=discovered_graph)
    >>> print(MCC)
    0.7453559924999299
    >>> MCC = matthews_correlation_coefficient(TP=2, FP=0, FN=1, TN=3)
    >>> print(MCC)
    0.7071067811865476
    """
    if true_graph is not None and discovered_graph is not None:
        TP, FP, FN, TN = get_confusion_matrix(true_graph, discovered_graph)
    else:
        assert (
            TP is not None and FP is not None and FN is not None and TN is not None
        ), "Either (true_graph, discovered_graph) or (TP, FP, FN, TN) must be provided."

    if TP == 0 and FP == 0:
        if FN == 0 and TN != 0:
            return 1.0
        elif FN != 0 and TN != 0:
            return 0.0
        elif FN != 0 and TN == 0:
            return -1.0
        else:
            raise ValueError("No conditions met!")
    if TP == 0 and FN == 0:
        if FP == 0 and TN != 0:
            return 1.0
        elif FP != 0 and TN != 0:
            return 0.0
        elif FP != 0 and TN == 0:
            return -1.0
        else:
            raise ValueError("No conditions met!")
    if TN == 0 and FP == 0:
        if TP != 0 and FN == 0:
            return 1.0
        elif TP != 0 and FN != 0:
            return 0.0
        elif TP == 0 and FN != 0:
            return -1.0
        else:
            raise ValueError("No conditions met!")
    if TN == 0 and FN == 0:
        if TP != 0 and FP == 0:
            return 1.0
        elif TP != 0 and FP != 0:
            return 0.0
        elif TP == 0 and FP != 0:
            return -1.0
        else:
            raise ValueError("No conditions met!")

    return (TP * TN - FP * FN) / (np.sqrt(float((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))))
