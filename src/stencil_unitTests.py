import unittest
import stencil_functions as sf


class TestGraphEditting(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        # Create or clear parents dictionary
        self.graph_parents = {i: [] for i in range(9)}

    def test_contains_parents(self) -> None:
        self.__init__()
        child_to_add_parent = 3
        parent_to_add = (1, -1)
        self.graph_parents[child_to_add_parent].append(parent_to_add)
        self.assertTrue(sf.contains_parents(self.graph_parents[3], [parent_to_add]))

    def test_add_parent(self) -> None:
        self.__init__()
        child_to_add_parent = 0
        parent_to_add = (1, -1)
        sf.add_parent(
            child_to_add_parent,
            self.graph_parents,
            parent_to_add,
            coefficient=0.5,
            coefficient_sources=None,
        )
        self.assertTrue(
            sf.contains_parents(self.graph_parents[child_to_add_parent], [parent_to_add])
        )

    def test_remove_parent(self) -> None:
        self.__init__()
        child = 0
        parent = (1, -1)
        sf.add_parent(
            child,
            self.graph_parents,
            parent,
            coefficient=0.5,
            coefficient_sources=None,
        )
        sf.remove_parent(self.graph_parents[child], parent)
        self.assertFalse(sf.contains_parents(self.graph_parents[child], [parent]))

    def test_add_link_from_redundant_links(self) -> None:
        self.__init__()
        center_node = 4
        lag = -1

        # Test addition of southward (1->4) link:
        redundant_links = {3: (0, lag), 5: (2, lag), 6: (3, lag), 7: (4, lag), 8: (5, lag)}
        omitted_link = (1, lag)
        for child, parent in redundant_links.items():
            sf.add_parent(child, self.graph_parents, parent, coefficient=0.5)
        sf.add_link_from_redundant_links(
            self.graph_parents, omitted_link, redundant_links=redundant_links
        )
        self.assertTrue(sf.contains_parents(self.graph_parents[center_node], [omitted_link]))

    def test_remove_nonredundant_link(self) -> None:
        self.__init__()
        center_node = 4
        lag = -1
        possible_parent = (0, lag)

        # Test removal of southeastward (0->4) link:
        sf.add_parent(center_node, self.graph_parents, possible_parent, coefficient=0.5)
        sf.remove_link_from_redundant_links(
            all_nodes_parents=self.graph_parents,
            link_to_remove=possible_parent,
            redundant_links={7: (3, lag), 5: (1, lag), 8: (4, lag)},
        )
        self.assertFalse(sf.contains_parents(self.graph_parents[center_node], [possible_parent]))

    def test_add_omitted_links(self) -> None:
        self.__init__()
        center_node = 4
        lag = -1

        # Test each possible link addition, clockwise from 12 o'clock.
        ## Test addition of southward (1->4) link:
        redundant_links = {3: (0, lag), 5: (2, lag), 6: (3, lag), 7: (4, lag), 8: (5, lag)}
        for child, parent in redundant_links.items():
            sf.add_parent(child, self.graph_parents, parent, coefficient=0.5)

        ## Test addition of southeastward (0->4) link:
        redundant_links = {7: (3, lag), 5: (1, lag), 8: (4, lag)}
        for child, parent in redundant_links.items():
            sf.add_parent(child, self.graph_parents, parent, coefficient=0.5)

        ## Test addition of eastward (3->4) link:
        redundant_links = {1: (0, lag), 2: (1, lag), 5: (4, lag), 7: (6, lag), 8: (7, lag)}
        for child, parent in redundant_links.items():
            sf.add_parent(child, self.graph_parents, parent, coefficient=0.5)

        ## Test addition of northeastward (6->4) link:
        redundant_links = {1: (3, lag), 2: (4, lag), 5: (7, lag)}
        for child, parent in redundant_links.items():
            sf.add_parent(child, self.graph_parents, parent, coefficient=0.5)

        ## Test addition of northward (7->4) link:
        redundant_links = {0: (3, lag), 1: (4, lag), 2: (5, lag), 3: (6, lag), 5: (8, lag)}
        for child, parent in redundant_links.items():
            sf.add_parent(child, self.graph_parents, parent, coefficient=0.5)

        ## Test addition of northwestward (8->4) link:
        redundant_links = {0: (4, lag), 3: (7, lag), 1: (5, lag)}
        for child, parent in redundant_links.items():
            sf.add_parent(child, self.graph_parents, parent, coefficient=0.5)

        ## Test addition of westward (5->4) link:
        redundant_links = {0: (1, lag), 1: (2, lag), 3: (4, lag), 6: (7, lag), 8: (9, lag)}
        for child, parent in redundant_links.items():
            sf.add_parent(child, self.graph_parents, parent, coefficient=0.5)

        ## Test addition of southwestward (2->4) link:
        redundant_links = {3: (1, lag), 7: (5, lag), 6: (4, lag)}
        for child, parent in redundant_links.items():
            sf.add_parent(child, self.graph_parents, parent, coefficient=0.5)

        ## Test addition of selfward (4->4) link:
        redundant_links = {
            0: (0, lag),
            1: (1, lag),
            2: (2, lag),
            3: (3, lag),
            5: (5, lag),
            6: (6, lag),
            7: (7, lag),
            8: (8, lag),
        }
        for child, parent in redundant_links.items():
            sf.add_parent(child, self.graph_parents, parent, coefficient=0.5)

        # Verify 4 has no parents
        assert len(self.graph_parents[center_node]) == 0, "Center node should have no parents."

        sf.add_omitted_links(self.graph_parents)
        self.assertTrue(
            sf.contains_parents(
                self.graph_parents[center_node], [(parent, lag) for parent in range(9)]
            )
        )

    def test_prune_nonredundant_links(self) -> None:
        self.__init__()
        center_node = 4
        lag = -1

        # Add all parents to center node, and none to others.
        for parent in range(9):
            sf.add_parent(4, self.graph_parents, (parent, lag), coefficient=0.5)

        sf.prune_nonredundant_links(self.graph_parents)

        self.assertTrue(len(self.graph_parents[center_node]) == 0)

    def test_pc_stable_single(self) -> None:
        self.__init__()
        from tigramite.toymodels import structural_causal_processes
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.independence_tests.gpdc import GPDC
        from tigramite.independence_tests.cmiknn import CMIknn
        import tigramite.data_processing as pp

        from collections import defaultdict
        from copy import deepcopy
        import numpy as np

        from pc_stable_single import PC_stable_single

        def lin_f(x):
            return x

        DATA_PATH = "../../data/DSAVAR/10x10_250T_4.0sigma_0.8density_0.1minval_wMode-False.npy"

        with open(
            DATA_PATH,
            "rb",
        ) as f:
            spatial_coefficients, data = np.load(f, allow_pickle=True)

        data = data[:, :, :, 0]
        concatenated_data = sf.concatenate_timeseries_nonwrapping(data, True)
        dataframe = pp.DataFrame(concatenated_data[:, :9])

        parcorr = ParCorr(significance="analytic")
        pcmci = PC_stable_single(dataframe=dataframe, cond_ind_test=parcorr, verbosity=1)
        tau_min = 1
        tau_max = 1
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
        all_parents[4] = pcmci._run_pc_stable_single(4, pc_alpha=pc_alpha, max_combinations=100000000)

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

        parents = sf.get_parents(graph, val_matrix=v_matrix, include_lagzero_parents=True, output_val_matrix=True)

        # print("detected parents:\n{}".format(all_parents[4]["parents"]))
        # print("prior spatial_coefficients:\n{}".format(spatial_coefficients))
        # print("thresholded stencil (mci >= {}):\n{}".format(dependence_threshold, parents))

        self.assertTrue(all_parents[4]["parents"] == [
            (4, -1),
            (3, -1),
            (1, -1),
            (0, -1),
            (6, -1),
            (8, -1),
            (7, -1),
        ])#, "Incorrect parents detected."



if __name__ == "__main__":
    unittest.main()
