"""
Unit Tests for ADRExperiment

Verifies the core ADRExperiment lifecycle: initialisation, filename generation
and parsing (round-trip), save/load, and MATLAB-mocked experiment execution.
Run with ``python -m unittest unitTestADRExperiment`` from the ``src/adr/``
directory, or via ``python unitTestADRExperiment.py``.
"""

import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import unittest
import numpy as np
from unittest.mock import patch
from ADRExperiment import ADRExperiment


class TestADRExperiment(unittest.TestCase):
    """
    Test suite for the ADRExperiment class.
    """

    def setUp(self):
        """
        Set up the ADRExperiment instance with default parameters for testing.
        """
        self.experiment = ADRExperiment(
            mesh_shape="square",
            init_center=[0.0, 0.0],
            plume_size=50,
            diff_coeffs=[0.05, 0.05],
            advection_coeffs=[4.0, 4.0],
            velocity_field_type="constant",
            velocity_parameters=[1.0, 1.0],
            react_rate=2.0,
            reaction_scaling=1.0,
            t=np.linspace(0, 0.4, 101).tolist(),
            H=0.02,
            radius=1.0,
            capture_apothem=1.0,
            capture_N=100,
            plot=False,
            verbose=False,
        )

    def test_initialization(self):
        """
        Test the initialization of the ADRExperiment instance.
        """
        self.assertEqual(self.experiment.mesh_shape, "square")
        self.assertEqual(self.experiment.init_center, [0.0, 0.0])
        self.assertEqual(self.experiment.plume_size, 50)
        self.assertEqual(self.experiment.diff_coeffs, [0.05, 0.05])
        self.assertEqual(self.experiment.advection_coeffs, [4.0, 4.0])
        self.assertEqual(self.experiment.velocity_field_type, "constant")
        self.assertEqual(self.experiment.velocity_parameters, [1.0, 1.0])
        self.assertEqual(self.experiment.react_rate, 2.0)
        self.assertEqual(self.experiment.reaction_scaling, 1.0)
        self.assertEqual(self.experiment.t, np.linspace(0, 0.4, 101).tolist())
        self.assertEqual(self.experiment.H, 0.02)
        self.assertEqual(self.experiment.radius, 1.0)
        self.assertEqual(self.experiment.capture_apothem, 1.0)
        self.assertEqual(self.experiment.capture_N, 100)
        self.assertFalse(self.experiment.plot)
        self.assertFalse(self.experiment.verbose)

    @patch("ADRExperiment.MatlabPDE")
    def test_run_adr_experiment(self, MockMatlabPDE):
        """
        Test running the ADR experiment and verify the solution shape.
        """
        mock_pde_solver = MockMatlabPDE.return_value
        mock_pde_solver.run_script.return_value = np.random.rand(100, 100, 101, 2)

        # Ensure the experiment uses the mocked pde_solver
        self.experiment.pde_solver = mock_pde_solver

        print("Running ADR experiment...")
        solution = self.experiment.run_adr_experiment()

        self.assertIsInstance(solution, np.ndarray)
        self.assertEqual(solution.shape, (100, 100, 101, 2))
        mock_pde_solver.start_engine.assert_called_once()
        mock_pde_solver.run_script.assert_called_once()

    def test_save_and_load_results(self):
        """
        Test saving and loading the ADRExperiment instance.
        """
        path_to_save = "test_experiment.pkl"
        self.experiment.save_results(path_to_save)

        loaded_experiment = ADRExperiment.load_results(path_to_save)

        self.assertEqual(self.experiment.mesh_shape, loaded_experiment.mesh_shape)
        self.assertEqual(self.experiment.init_center, loaded_experiment.init_center)
        self.assertEqual(self.experiment.plume_size, loaded_experiment.plume_size)
        self.assertEqual(self.experiment.diff_coeffs, loaded_experiment.diff_coeffs)
        self.assertEqual(
            self.experiment.advection_coeffs, loaded_experiment.advection_coeffs
        )
        self.assertEqual(
            self.experiment.velocity_field_type, loaded_experiment.velocity_field_type
        )
        self.assertEqual(
            self.experiment.velocity_parameters, loaded_experiment.velocity_parameters
        )
        self.assertEqual(self.experiment.react_rate, loaded_experiment.react_rate)
        self.assertEqual(
            self.experiment.reaction_scaling, loaded_experiment.reaction_scaling
        )
        self.assertEqual(self.experiment.t, loaded_experiment.t)
        self.assertEqual(self.experiment.H, loaded_experiment.H)
        self.assertEqual(self.experiment.radius, loaded_experiment.radius)
        self.assertEqual(
            self.experiment.capture_apothem, loaded_experiment.capture_apothem
        )
        self.assertEqual(self.experiment.capture_N, loaded_experiment.capture_N)
        self.assertFalse(loaded_experiment.plot)
        self.assertFalse(loaded_experiment.verbose)

        os.remove(path_to_save)

    def test_generate_filename(self):
        """
        Test generating a filename based on the experiment parameters.
        """
        expected_filename = "ADR_square_0.0_0.0_50_0.05_0.05_4.0_4.0_constant_1.0_1.0_2.0_1.0_0.02_1.0_1.0_100_0.0_0.4_101.pkl"
        generated_filename = self.experiment.generate_filename()
        self.assertEqual(expected_filename, generated_filename)

    def test_parse_filename(self):
        """
        Test parsing a filename to extract experiment parameters.
        """
        filename = "ADR_square_0.0_0.0_50_0.05_0.05_4.0_4.0_constant_1.0_1.0_2.0_1.0_0.02_1.0_1.0_100_0.0_0.4_101.pkl"
        expected_params = {
            "mesh_shape": "square",
            "init_center": [0.0, 0.0],
            "plume_size": 50,
            "diff_coeffs": [0.05, 0.05],
            "advection_coeffs": [4.0, 4.0],
            "velocity_field_type": "constant",
            "velocity_parameters": [1.0, 1.0],
            "react_rate": 2.0,
            "reaction_scaling": 1.0,
            "H": 0.02,
            "radius": 1.0,
            "capture_apothem": 1.0,
            "capture_N": 100,
            "t": np.linspace(0, 0.4, 101).tolist(),
        }
        parsed_params = ADRExperiment.parse_filename(filename)
        self.assertEqual(expected_params, parsed_params)

    def test_generate_and_parse_filename(self):
        """
        Test the compatibility between generate_filename and parse_filename.
        Ensure that generating a filename and then parsing it returns the original parameters.
        """
        # Generate a filename from the experiment parameters
        generated_filename = self.experiment.generate_filename()

        # Parse the generated filename back into parameters
        parsed_params = ADRExperiment.parse_filename(generated_filename)

        # Verify that the parsed parameters match the original parameters
        self.assertEqual(parsed_params["mesh_shape"], self.experiment.mesh_shape)
        self.assertEqual(parsed_params["init_center"], self.experiment.init_center)
        self.assertEqual(parsed_params["plume_size"], self.experiment.plume_size)
        self.assertEqual(parsed_params["diff_coeffs"], self.experiment.diff_coeffs)
        self.assertEqual(
            parsed_params["advection_coeffs"], self.experiment.advection_coeffs
        )
        self.assertEqual(
            parsed_params["velocity_field_type"], self.experiment.velocity_field_type
        )
        self.assertEqual(
            parsed_params["velocity_parameters"], self.experiment.velocity_parameters
        )
        self.assertEqual(parsed_params["react_rate"], self.experiment.react_rate)
        self.assertEqual(
            parsed_params["reaction_scaling"], self.experiment.reaction_scaling
        )
        self.assertEqual(parsed_params["H"], self.experiment.H)
        self.assertEqual(parsed_params["radius"], self.experiment.radius)
        self.assertEqual(
            parsed_params["capture_apothem"], self.experiment.capture_apothem
        )
        self.assertEqual(parsed_params["capture_N"], self.experiment.capture_N)
        self.assertEqual(parsed_params["t"], self.experiment.t)


if __name__ == "__main__":
    unittest.main()
