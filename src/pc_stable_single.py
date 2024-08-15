"""PC-Stable_Single parent identification algorithm

The code here is heavily adopted from the TIGRAMITE implementation:https://github.com/jakobrunge/tigramite/blob/2f289d058bfd8fb8e1841d8f4cf8fe4062975ed5/tigramite/pcmci.py

This is a minimum code implementation of PC-Stable-Single. It currently does not support many features implemented in TIGRAMITE, such as link assumptions.

Intended only for use in the CaStLe algorithm as the parent identification step.
"""

from __future__ import print_function
import itertools
from collections import defaultdict
from copy import deepcopy
import numpy as np

import stencil_functions as sf


def _create_nested_dictionary(depth=0, lowest_type=dict) -> defaultdict:
    """Create a series of nested dictionaries to a maximum depth.  The first
    depth - 1 nested dictionaries are defaultdicts, the last is a normal
    dictionary.

    Parameters
    ----------
    depth : int
        Maximum depth argument.
    lowest_type: callable (optional)
        Type contained in leaves of tree.  Ex: list, dict, tuple, int, float ...
    """
    new_depth = depth - 1
    if new_depth <= 0:
        return defaultdict(lowest_type)
    return defaultdict(lambda: _create_nested_dictionary(new_depth))


def _nested_to_normal(nested_dict) -> dict:
    """Transforms the nested default dictionary into a standard dictionaries

    Parameters
    ----------
    nested_dict : default dictionary of default dictionaries of ... etc.
    """
    if isinstance(nested_dict, defaultdict):
        nested_dict = {k: _nested_to_normal(v) for k, v in nested_dict.items()}
    return nested_dict


class PC_stable_single:
    def __init__(self, dataframe, cond_ind_test, verbosity=0):
        # Set the data for this iteration of the algorithm
        self.dataframe = dataframe
        # Set the conditional independence test to be used
        self.cond_ind_test = cond_ind_test
        if isinstance(self.cond_ind_test, type):
            raise ValueError(
                "PCMCI requires that cond_ind_test "
                "is instantiated, e.g. cond_ind_test =  "
                "ParCorr()."
            )
        self.cond_ind_test.set_dataframe(self.dataframe)
        # Set the verbosity for debugging/logging messages
        self.verbosity = verbosity
        # Set the variable names
        self.var_names = self.dataframe.var_names

        # Store the shape of the data in the T and N variables
        self.T = self.dataframe.T
        self.N = self.dataframe.N

    def _iter_conditions(self, parent, conds_dim, all_parents):
        """Yield next condition.

        Yields next condition from lexicographically ordered conditions.

        Parameters
        ----------
        parent : tuple
            Tuple of form (i, -tau).
        conds_dim : int
            Cardinality in current step.
        all_parents : list
            List of form [(0, -1), (3, -2), ...].

        Yields
        -------
        cond :  list
            List of form [(0, -1), (3, -2), ...] for the next condition.
        """
        all_parents_excl_current = [p for p in all_parents if p != parent]
        for cond in itertools.combinations(all_parents_excl_current, conds_dim):
            yield list(cond)

    def _sort_parents(self, parents_vals):
        """Sort current parents according to test statistic values.

        Sorting is from strongest to weakest absolute values.

        Parameters
        ---------
        parents_vals : dict
            Dictionary of form {(0, -1):float, ...} containing the minimum test
            statistic value of a link.

        Returns
        -------
        parents : list
            List of form [(0, -1), (3, -2), ...] containing sorted parents.
        """
        if self.verbosity > 1:
            print(
                "\n    Sorting parents in decreasing order with "
                "\n    weight(i-tau->j) = min_{iterations} |val_{ij}(tau)| "
            )
        # Get the absolute value for all the test statistics
        abs_values = {k: np.abs(parents_vals[k]) for k in list(parents_vals)}
        return sorted(abs_values, key=abs_values.get, reverse=True)

    def _set_max_condition_dim(self, max_conds_dim, tau_min, tau_max):
        """
        Set the maximum dimension of the conditions. Defaults to self.N*tau_max.

        Parameters
        ----------
        max_conds_dim : int
            Input maximum condition dimension.
        tau_max : int
            Maximum tau.

        Returns
        -------
        max_conds_dim : int
            Input maximum condition dimension or default.
        """
        # Check if an input was given
        if max_conds_dim is None:
            max_conds_dim = self.N * (tau_max - tau_min + 1)
        # Check this is a valid
        if max_conds_dim < 0:
            raise ValueError("maximum condition dimension must be >= 0")
        return max_conds_dim

    def _check_tau_limits(self, tau_min, tau_max):
        """Check the tau limits adhere to 0 <= tau_min <= tau_max.

        Parameters
        ----------
        tau_min : float
            Minimum tau value.
        tau_max : float
            Maximum tau value.
        """
        if not 0 <= tau_min <= tau_max:
            raise ValueError(
                "tau_max = %d, " % (tau_max)
                + "tau_min = %d, " % (tau_min)
                + "but 0 <= tau_min <= tau_max"
            )

    def _set_link_assumptions(
        self, link_assumptions, tau_min, tau_max, remove_contemp=True
    ):
        """Dummy function to return a fully naive set of link assumptions. This function should only pertain to the stencil learning algorithm!

        Returns
        -------
        dict
            Naive set of link assumptions.
        """
        d = {
            0: {
                (0, -1): "-?>",
                (1, -1): "-?>",
                (2, -1): "-?>",
                (3, -1): "-?>",
                (4, -1): "-?>",
                (5, -1): "-?>",
                (6, -1): "-?>",
                (7, -1): "-?>",
                (8, -1): "-?>",
            },
            1: {
                (0, -1): "-?>",
                (1, -1): "-?>",
                (2, -1): "-?>",
                (3, -1): "-?>",
                (4, -1): "-?>",
                (5, -1): "-?>",
                (6, -1): "-?>",
                (7, -1): "-?>",
                (8, -1): "-?>",
            },
            2: {
                (0, -1): "-?>",
                (1, -1): "-?>",
                (2, -1): "-?>",
                (3, -1): "-?>",
                (4, -1): "-?>",
                (5, -1): "-?>",
                (6, -1): "-?>",
                (7, -1): "-?>",
                (8, -1): "-?>",
            },
            3: {
                (0, -1): "-?>",
                (1, -1): "-?>",
                (2, -1): "-?>",
                (3, -1): "-?>",
                (4, -1): "-?>",
                (5, -1): "-?>",
                (6, -1): "-?>",
                (7, -1): "-?>",
                (8, -1): "-?>",
            },
            4: {
                (0, -1): "-?>",
                (1, -1): "-?>",
                (2, -1): "-?>",
                (3, -1): "-?>",
                (4, -1): "-?>",
                (5, -1): "-?>",
                (6, -1): "-?>",
                (7, -1): "-?>",
                (8, -1): "-?>",
            },
            5: {
                (0, -1): "-?>",
                (1, -1): "-?>",
                (2, -1): "-?>",
                (3, -1): "-?>",
                (4, -1): "-?>",
                (5, -1): "-?>",
                (6, -1): "-?>",
                (7, -1): "-?>",
                (8, -1): "-?>",
            },
            6: {
                (0, -1): "-?>",
                (1, -1): "-?>",
                (2, -1): "-?>",
                (3, -1): "-?>",
                (4, -1): "-?>",
                (5, -1): "-?>",
                (6, -1): "-?>",
                (7, -1): "-?>",
                (8, -1): "-?>",
            },
            7: {
                (0, -1): "-?>",
                (1, -1): "-?>",
                (2, -1): "-?>",
                (3, -1): "-?>",
                (4, -1): "-?>",
                (5, -1): "-?>",
                (6, -1): "-?>",
                (7, -1): "-?>",
                (8, -1): "-?>",
            },
            8: {
                (0, -1): "-?>",
                (1, -1): "-?>",
                (2, -1): "-?>",
                (3, -1): "-?>",
                (4, -1): "-?>",
                (5, -1): "-?>",
                (6, -1): "-?>",
                (7, -1): "-?>",
                (8, -1): "-?>",
            },
        }
        return d

    def _run_pc_stable_single(
        self,
        j,
        tau_min=1,
        tau_max=1,
        pc_alpha=0.2,
        max_conds_dim=None,
        max_combinations=1,
    ):
        """Lagged PC algorithm for estimating lagged parents of single variable.

        Parameters
        ----------
        j : int
            Variable index.
        link_assumptions_j : dict
            Dictionary of form {j:{(i, -tau): link_type, ...}, ...} specifying
            assumptions about links. This initializes the graph with entries
            graph[i,j,tau] = link_type. For example, graph[i,j,0] = '-->'
            implies that a directed link from i to j at lag 0 must exist.
            Valid link types are 'o-o', '-->', '<--'. In addition, the middle
            mark can be '?' instead of '-'. Then '-?>' implies that this link
            may not exist, but if it exists, its orientation is '-->'. Link
            assumptions need to be consistent, i.e., graph[i,j,0] = '-->'
            requires graph[j,i,0] = '<--' and acyclicity must hold. If a link
            does not appear in the dictionary, it is assumed absent. That is,
            if link_assumptions is not None, then all links have to be specified
            or the links are assumed absent.
        tau_min : int, optional (default: 1)
            Minimum time lag to test. Useful for variable selection in
            multi-step ahead predictions. Must be greater zero.
        tau_max : int, optional (default: 1)
            Maximum time lag. Must be larger or equal to tau_min.
        save_iterations : bool, optional (default: False)
            Whether to save iteration step results such as conditions used.
        pc_alpha : float or None, optional (default: 0.2)
            Significance level in algorithm. If a list is given, pc_alpha is
            optimized using model selection criteria provided in the
            cond_ind_test class as get_model_selection_criterion(). If None,
            a default list of values is used.
        max_conds_dim : int, optional (default: None)
            Maximum number of conditions to test. If None is passed, this number
            is unrestricted.
        max_combinations : int, optional (default: 1)
            Maximum number of combinations of conditions of current cardinality
            to test in PC1 step.

        Returns
        -------
        parents : list
            List of estimated parents.
        val_min : dict
            Dictionary of form {(0, -1):float, ...} containing the minimum test
            statistic value of a link.
        pval_max : dict
            Dictionary of form {(0, -1):float, ...} containing the maximum
            p-value of a link across different conditions.
        iterations : dict
            Dictionary containing further information on algorithm steps.
        """

        if pc_alpha < 0.0 or pc_alpha > 1.0:
            raise ValueError("Choose 0 <= pc_alpha <= 1")

        # Initialize the dictionaries for the pval_max, val_min parents_values
        # results
        pval_max = dict()
        val_min = dict()
        parents_values = dict()
        parents = [
            (0, -1),
            (1, -1),
            (2, -1),
            (3, -1),
            (4, -1),
            (5, -1),
            (6, -1),
            (7, -1),
            (8, -1),
        ]

        val_min = {(p[0], p[1]): None for p in parents}
        pval_max = {(p[0], p[1]): None for p in parents}

        # Ensure tau_min is at least 1
        tau_min = max(1, tau_min)

        # Loop over all possible condition dimensions
        # In TIGRAMITE, is set to "self.N * (tau_max - tau_min + 1)", which is always 9 here.
        max_conds_dim = 9

        # Iteration through increasing number of conditions, i.e. from
        # [0, max_conds_dim] inclusive
        converged = False
        _max_comb_index = -1
        for conds_dim in range(max_conds_dim + 1):
            # (Re)initialize the list of non-significant links
            nonsig_parents = list()
            # Check if the algorithm has converged
            if len(parents) - 1 < conds_dim:
                converged = True
                break

            # Iterate through all possible pairs (that have not converged yet)
            for parent in parents:
                # Iterate through all possible combinations
                nonsig = False
                for comb_index, Z in enumerate(
                    self._iter_conditions(parent, conds_dim, parents)
                ):
                    # Testing comb indexing (personal test)
                    if comb_index > _max_comb_index:
                        _max_comb_index = comb_index

                    # Break if we try too many combinations
                    if comb_index >= max_combinations:
                        break
                    val, pval, dependent = self.cond_ind_test.run_test(
                        X=[parent],
                        Y=[(j, 0)],
                        Z=Z,
                        tau_max=tau_max,
                        alpha_or_thres=pc_alpha,
                    )
                    # Keep track of maximum p-value and minimum estimated value
                    # for each pair (across any condition)
                    parents_values[parent] = min(
                        np.abs(val), parents_values.get(parent, float("inf"))
                    )

                    if pval_max[parent] is None or pval > pval_max[parent]:
                        pval_max[parent] = pval
                        val_min[parent] = val

                    # Delete link later and break while-loop if non-significant
                    if not dependent:  # pval > pc_alpha:
                        nonsig_parents.append((j, parent))
                        nonsig = True
                        break

            # Remove non-significant links
            for _, parent in nonsig_parents:
                del parents_values[parent]
            # Return the parents list sorted by the test metric so that the
            # updated parents list is given to the next cond_dim loop
            parents = self._sort_parents(parents_values)

        return {
            "parents": parents,
            "val_min": val_min,
            "pval_max": pval_max,
        }


if __name__ == "__main__":
    from tigramite.pcmci import PCMCI
    from tigramite.toymodels import structural_causal_processes
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.independence_tests.cmiknn import CMIknn

    import tigramite.data_processing as pp
    from tigramite.toymodels import structural_causal_processes as toys
    import tigramite.plotting as tp
    from matplotlib import pyplot as plt

    def lin_f(x):
        return x

    DATA_PATH = (
        "PATH/TO/DATA/10x10_250T_4.0sigma_0.8density_0.1minval_wMode-False.npy"
    )

    with open(
        DATA_PATH,
        "rb",
    ) as f:
        spatial_coefficients, data = np.load(f, allow_pickle=True)

    data = data[:, :, :, 0]
    concatenated_data = sf.concatenate_timeseries_nonwrapping(data, True)
    GRID_SIZE = data.shape[0]
    dataframe = pp.DataFrame(concatenated_data[:, :9])

    parcorr = ParCorr(significance="analytic")
    pcmci = PC_stable_single(dataframe=dataframe, cond_ind_test=parcorr, verbosity=1)

    selected_links = None
    link_assumptions = None
    tau_min = 1
    tau_max = 1
    save_iterations = False
    pc_alpha = 0.00001
    max_conds_dim = None
    max_combinations = 1

    # Create an internal copy of pc_alpha
    _int_pc_alpha = deepcopy(pc_alpha)
    # Check if we are selecting an optimal alpha value
    select_optimal_alpha = True
    # Set the default values for pc_alpha
    if _int_pc_alpha is None:
        _int_pc_alpha = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    elif not isinstance(_int_pc_alpha, (list, tuple, np.ndarray)):
        _int_pc_alpha = [_int_pc_alpha]
        select_optimal_alpha = False
    # Check the limits on tau_min
    pcmci._check_tau_limits(tau_min, tau_max)
    tau_min = max(1, tau_min)
    # Check that the maximum combinations variable is correct
    if max_combinations <= 0:
        raise ValueError("max_combinations must be > 0")
    # Implement defaultdict for all pval_max, val_max, and iterations
    pval_max = defaultdict(dict)
    val_min = defaultdict(dict)
    iterations = defaultdict(dict)

    # Initialize all parents
    all_parents = dict()
    # Set the maximum condition dimension
    max_conds_dim = pcmci._set_max_condition_dim(max_conds_dim, tau_min, tau_max)

    all_parents = {
        i: {
            "parents": [],
            "val_min": {(j, -1): 0 for j in range(9)},
            "pval_max": {(k, -1): 1 for k in range(9)},
            "iterations": {},
        }
        for i in range(9)
    }
    all_parents[4] = pcmci._run_pc_stable_single(
        4, pc_alpha=pc_alpha, max_combinations=100000000
    )

    # Make SCM and val_matrix for plotting
    dependence_threshold = 0.0001
    SCM = {}

    for key in all_parents.keys():
        SCM[key] = []
        parents_list = [parent[0] for parent in all_parents[key]["parents"]]
        for parent in parents_list:
            coefficient = all_parents[key]["val_min"][(parent, -1)]
            if abs(coefficient) < dependence_threshold:
                coefficient = 0
            SCM[key].append(((parent, -1), coefficient, lin_f))

    graph = structural_causal_processes.links_to_graph(SCM)

    v_matrix = np.zeros(graph.shape)
    for row in range(v_matrix.shape[0]):
        if len(SCM[row]) != 0:
            for dependence in SCM[row]:
                coefficient = dependence[1]
                v_matrix[dependence[0][0], row, 1] = coefficient

    parents = sf.get_parents(
        graph, val_matrix=v_matrix, include_lagzero_parents=True, output_val_matrix=True
    )

    print("detected parents:\n{}".format(all_parents[4]["parents"]))
    print("prior spatial_coefficients:\n{}".format(spatial_coefficients))
    print("thresholded stencil (mci >= {}):\n{}".format(dependence_threshold, parents))

    assert all_parents[4]["parents"] == [
        (4, -1),
        (3, -1),
        (1, -1),
        (0, -1),
        (6, -1),
        (8, -1),
        (7, -1),
    ], "Incorrect parents detected."
