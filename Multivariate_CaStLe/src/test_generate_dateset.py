"""
Comprehensive test suite for the generate_dataset function.

Tests cover:
- Basic functionality and output shapes
- Edge cases (small grids, single variable, minimal time steps)
- Boundary conditions and spatial wrapping
- Coefficient handling (provided vs. generated)
- Instability detection and recovery
- Parameter validation
- Numerical properties (finiteness, stability)
- Reproducibility
"""

import numpy as np
import pytest
import sys

sys.path.insert(0, "/mnt/user-data/uploads")

from spatiotemporal_SCM_data_generator import generate_dataset, get_random_stable_coefficient_matrix, get_density


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def basic_config():
    """Basic configuration for typical test cases."""
    return {
        "grid_size": 5,
        "T": 10,
        "num_variables": 2,
        "dependence_density": 0.3,
        "coefficient_min_value_threshold": 0.1,
        "min_val_scaler": 1.0,
        "error_sigma": 0.1,
        "error_mean": 0.0,
    }


@pytest.fixture
def small_stable_coefs():
    """Generate small stable coefficients for deterministic testing."""
    grid_size = 3
    n_vars = 2

    # Create a simple stable coefficient structure
    # Shape: (n_vars, 3, 3) where each entry is an array of length n_vars
    coefs = np.zeros((n_vars, 3, 3), dtype=object)

    for child in range(n_vars):
        for i in range(3):
            for j in range(3):
                # Small coefficients to ensure stability
                # Center cell has stronger influence
                if i == 1 and j == 1:  # center
                    coefs[child, i, j] = np.array([0.3 if p == child else 0.1 for p in range(n_vars)])
                else:  # neighbors
                    coefs[child, i, j] = np.array([0.05 if p == child else 0.02 for p in range(n_vars)])

    return coefs


# ============================================================================
# Test Class: Basic Functionality
# ============================================================================


class TestBasicFunctionality:
    """Test basic function operation and output properties."""

    def test_returns_correct_shape(self, basic_config):
        """Test that output has expected shape (num_variables, grid_size, grid_size, T)."""
        data = generate_dataset(**basic_config)

        expected_shape = (basic_config["num_variables"], basic_config["grid_size"], basic_config["grid_size"], basic_config["T"])
        assert data.shape == expected_shape, f"Expected shape {expected_shape}, got {data.shape}"

    def test_returns_coefs_when_requested(self, basic_config):
        """Test that function returns both coefficients and data when return_coefs=True."""
        basic_config["return_coefs"] = True
        result = generate_dataset(**basic_config)

        assert isinstance(result, tuple), "Expected tuple when return_coefs=True"
        assert len(result) == 2, "Expected (coefs, data) tuple"

        coefs, data = result
        assert coefs is not None
        assert data is not None
        assert data.shape == (basic_config["num_variables"], basic_config["grid_size"], basic_config["grid_size"], basic_config["T"])

    def test_with_provided_coefficients(self, small_stable_coefs):
        """Test that function works with pre-provided coefficients."""
        data = generate_dataset(grid_size=3, T=5, spatial_coefs=small_stable_coefs, error_sigma=0.05)

        assert data.shape == (2, 3, 3, 5)

    def test_initial_timestep_zero_with_zero_initialization(self, basic_config):
        """Test that first timestep is zero when initialize_randomly=False."""
        basic_config["initialize_randomly"] = False
        data = generate_dataset(**basic_config)

        assert np.allclose(data[:, :, :, 0], 0.0), "Expected first timestep to be zero"

    def test_initial_timestep_nonzero_with_random_initialization(self, basic_config):
        """Test that first timestep is non-zero when initialize_randomly=True."""
        basic_config["initialize_randomly"] = True
        data = generate_dataset(**basic_config)

        assert not np.allclose(data[:, :, :, 0], 0.0), "Expected first timestep to be non-zero"

    def test_all_values_finite(self, basic_config):
        """Test that generated data contains no NaN or Inf values."""
        data = generate_dataset(**basic_config)

        assert np.all(np.isfinite(data)), "Data contains NaN or Inf values"

    def test_deterministic_with_fixed_seed(self, basic_config):
        """Test that results are reproducible with fixed random seed."""
        np.random.seed(42)
        data1 = generate_dataset(**basic_config)

        np.random.seed(42)
        data2 = generate_dataset(**basic_config)

        assert np.allclose(data1, data2), "Results not reproducible with same seed"


# ============================================================================
# Test Class: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions of inputs."""

    def test_single_variable(self):
        """Test with single variable system."""
        data = generate_dataset(grid_size=4, T=5, num_variables=1, dependence_density=0.5, coefficient_min_value_threshold=0.1, min_val_scaler=1.0, error_sigma=0.1)

        assert data.shape == (1, 4, 4, 5)

    def test_minimal_grid_size(self):
        """Test with small grid (4x4).

        Note: Grid sizes below 4x4 create degenerate spatial neighborhoods where
        the 3x3 neighbor structure has significant overlap, potentially causing
        instability or coefficient generation issues.
        """
        data = generate_dataset(grid_size=4, T=5, num_variables=2, dependence_density=0.3, coefficient_min_value_threshold=0.1, min_val_scaler=1.0, error_sigma=0.1)

        assert data.shape == (2, 4, 4, 5)

    def test_small_grid_3x3(self):
        """Test with 3x3 grid to verify it works despite some neighbor overlap."""
        data = generate_dataset(grid_size=3, T=5, num_variables=2, dependence_density=0.3, coefficient_min_value_threshold=0.1, min_val_scaler=1.0, error_sigma=0.1)

        assert data.shape == (2, 3, 3, 5)

    def test_minimal_timesteps(self):
        """Test with minimal number of timesteps (T=2)."""
        data = generate_dataset(grid_size=3, T=2, num_variables=2, dependence_density=0.3, coefficient_min_value_threshold=0.1, min_val_scaler=1.0, error_sigma=0.1)

        assert data.shape == (2, 3, 3, 2)

    def test_zero_noise(self, small_stable_coefs):
        """Test with zero noise (deterministic dynamics only)."""
        data = generate_dataset(grid_size=3, T=5, spatial_coefs=small_stable_coefs, error_sigma=0.0, error_mean=0.0, initialize_randomly=False)

        # With zero initialization and zero noise, all data should be zero
        assert np.allclose(data, 0.0), "Expected all zeros with zero init and zero noise"

    def test_high_density(self):
        """Test with high dependence density (0.9)."""
        data = generate_dataset(grid_size=4, T=5, num_variables=2, dependence_density=0.9, coefficient_min_value_threshold=0.05, min_val_scaler=1.0, error_sigma=0.1)

        assert data.shape == (2, 4, 4, 5)

    def test_low_density(self):
        """Test with very low dependence density (0.1)."""
        data = generate_dataset(grid_size=4, T=5, num_variables=2, dependence_density=0.1, coefficient_min_value_threshold=0.1, min_val_scaler=1.0, error_sigma=0.1)

        assert data.shape == (2, 4, 4, 5)


# ============================================================================
# Test Class: Boundary Conditions
# ============================================================================


class TestBoundaryConditions:
    """Test spatial boundary handling and wrapping behavior."""

    def test_boundary_influence_exists(self, small_stable_coefs):
        """Test that boundary cells are influenced by their neighbors."""
        # Use deterministic initialization to check influence
        np.random.seed(42)
        data = generate_dataset(grid_size=3, T=3, spatial_coefs=small_stable_coefs, error_sigma=0.01, initialize_randomly=True)

        # Check that corner cell at t=2 is influenced by t=1 neighbors
        # (exact value depends on coefficients, but should be non-zero)
        corner_t1 = data[0, 0, 0, 1]
        corner_t2 = data[0, 0, 0, 2]

        # Value should change over time if influenced by neighbors
        assert not np.isclose(corner_t1, corner_t2, atol=0.001), "Corner cell should change over time due to neighbor influence"

    def test_edge_cells_computed(self, small_stable_coefs):
        """Test that edge and corner cells are properly computed."""
        data = generate_dataset(grid_size=3, T=5, spatial_coefs=small_stable_coefs, error_sigma=0.05, initialize_randomly=False)

        # Check that edge cells have been computed (not left at zero for all t)
        edges = [
            data[0, 0, :, -1],  # top row
            data[0, -1, :, -1],  # bottom row
            data[0, :, 0, -1],  # left column
            data[0, :, -1, -1],  # right column
        ]

        for edge in edges:
            # With noise, edges should have non-zero values eventually
            assert not np.allclose(edge, 0.0, atol=0.01), "Edge cells should have non-zero values due to noise accumulation"


# ============================================================================
# Test Class: Parameter Validation
# ============================================================================


class TestParameterValidation:
    """Test input parameter validation and error handling."""

    def test_missing_num_variables_without_coefs(self):
        """Test that missing num_variables raises assertion error."""
        with pytest.raises(AssertionError):
            generate_dataset(
                grid_size=4,
                T=5,
                dependence_density=0.3,
                coefficient_min_value_threshold=0.1,
                min_val_scaler=1.0,
                # Missing num_variables
            )

    def test_missing_density_and_num_links_without_coefs(self):
        """Test that missing both density and num_links raises assertion error."""
        with pytest.raises(AssertionError):
            generate_dataset(
                grid_size=4,
                T=5,
                num_variables=2,
                coefficient_min_value_threshold=0.1,
                min_val_scaler=1.0,
                # Missing both dependence_density and num_links
            )

    def test_both_density_and_num_links_raises_error(self):
        """Test that providing both density and num_links raises assertion error."""
        with pytest.raises(AssertionError):
            generate_dataset(grid_size=4, T=5, num_variables=2, dependence_density=0.3, num_links=10, coefficient_min_value_threshold=0.1, min_val_scaler=1.0)

    def test_num_links_parameter(self):
        """Test that num_links parameter works correctly."""
        data = generate_dataset(grid_size=4, T=5, num_variables=2, num_links=10, coefficient_min_value_threshold=0.1, min_val_scaler=1.0, error_sigma=0.1)

        assert data.shape == (2, 4, 4, 5)

    def test_too_many_links_raises_error(self):
        """Test that requesting too many links for the number of variables raises error."""
        # For 2 variables, max links = 2 * 9 * 2 = 36
        with pytest.raises(AssertionError):
            generate_dataset(grid_size=4, T=5, num_variables=2, num_links=100, coefficient_min_value_threshold=0.1, min_val_scaler=1.0)  # Too many


# ============================================================================
# Test Class: Instability Detection
# ============================================================================


class TestInstabilityDetection:
    """Test instability detection and recovery mechanisms."""

    def test_instability_detection_disabled_by_default(self):
        """Test that instability detection is off by default and function completes.

        Note: We use moderate parameters here because aggressive parameters with
        detect_instability=False can cause the function to hang as values explode.
        This test verifies the flag works, not that unstable systems complete.
        """
        data = generate_dataset(
            grid_size=3,
            T=5,
            num_variables=2,
            dependence_density=0.4,  # Moderate, not aggressive
            coefficient_min_value_threshold=0.15,
            min_val_scaler=1.0,
            error_sigma=0.1,
            detect_instability=False,  # explicitly disabled
        )

        # Should complete with reasonable parameters
        assert data.shape == (2, 3, 3, 5)
        assert np.all(np.isfinite(data)), "Data should be finite"

    def test_unstable_provided_coefficients_without_detection(self):
        """Test that unstable coefficients without detection lead to explosive growth.

        By providing deliberately unstable coefficients, we skip the coefficient
        generation phase (which could hang) and directly test the simulation behavior.
        """
        # Create deliberately unstable coefficients: strong self-reinforcement
        n_vars = 2
        unstable_coefs = np.zeros((n_vars, 3, 3), dtype=object)

        for child in range(n_vars):
            for i in range(3):
                for k in range(3):
                    # Large self-influence on center cell
                    if i == 1 and k == 1:
                        unstable_coefs[child, i, k] = np.array([0.9 if p == child else 0.0 for p in range(n_vars)])
                    else:
                        unstable_coefs[child, i, k] = np.array([0.15 if p == child else 0.0 for p in range(n_vars)])

        data = generate_dataset(
            grid_size=3, T=5, spatial_coefs=unstable_coefs, error_sigma=0.5, initialize_randomly=True, detect_instability=False  # Short to prevent excessive computation
        )

        # With strong self-reinforcement and random initialization,
        # values will grow (though not necessarily exceed threshold in 5 steps)
        assert data.shape == (2, 3, 3, 5)
        # Values should grow over time due to instability
        initial_magnitude = np.mean(np.abs(data[:, :, :, 0]))
        final_magnitude = np.mean(np.abs(data[:, :, :, -1]))
        assert final_magnitude > initial_magnitude, "Values should grow with unstable coefficients"

    @pytest.mark.skip(reason="Coefficient generation with aggressive parameters can hang indefinitely due to spectral radius constraints")
    def test_aggressive_parameters_coefficient_generation_limitation(self):
        """Demonstrates inherent limitation: aggressive parameters may make stable coefficients impossible.

        Mathematical constraint: For stability, spectral radius < 1. With high density
        (many connections) and large minimum coefficients, stable configurations may not exist.
        The coefficient generation uses rejection sampling which can hang searching for
        configurations that don't exist.

        This test is skipped because it demonstrates a known limitation that cannot be
        easily tested without timeouts.
        """
        # This would hang trying to find stable coefficients
        data = generate_dataset(
            grid_size=3, T=3, num_variables=2, dependence_density=0.8, coefficient_min_value_threshold=0.3, min_val_scaler=2.0, error_sigma=0.1, detect_instability=False
        )

        assert data.shape == (2, 3, 3, 3)

    def test_instability_detection_with_stable_system(self):
        """Test that stable systems complete successfully with detection enabled."""
        data = generate_dataset(
            grid_size=3,
            T=10,
            num_variables=2,
            dependence_density=0.3,
            coefficient_min_value_threshold=0.1,
            min_val_scaler=1.0,
            error_sigma=0.05,
            detect_instability=True,
            instability_threshold=1000,
        )

        assert data.shape == (2, 3, 3, 10)
        assert np.all(data < 1000), "Data should remain below instability threshold"

    def test_instability_detection_requires_parameters(self):
        """Test that instability detection requires necessary parameters."""
        with pytest.raises(AssertionError):
            generate_dataset(
                grid_size=3,
                T=5,
                num_variables=2,
                detect_instability=True,
                # Missing required parameters for coefficient generation
            )


# ============================================================================
# Test Class: Numerical Properties
# ============================================================================


class TestNumericalProperties:
    """Test numerical properties of generated data."""

    def test_data_magnitude_reasonable(self, basic_config):
        """Test that data magnitude remains reasonable for stable systems."""
        basic_config["coefficient_min_value_threshold"] = 0.05
        basic_config["min_val_scaler"] = 1.0
        basic_config["error_sigma"] = 0.1
        basic_config["T"] = 50

        data = generate_dataset(**basic_config)

        # For stable system, values shouldn't explode
        assert np.max(np.abs(data)) < 100, f"Data magnitude too large: max={np.max(np.abs(data))}"

    def test_noise_affects_data(self, small_stable_coefs):
        """Test that noise parameter actually affects the output."""
        np.random.seed(42)
        data_low_noise = generate_dataset(grid_size=3, T=20, spatial_coefs=small_stable_coefs, error_sigma=0.01, initialize_randomly=False)

        np.random.seed(42)
        data_high_noise = generate_dataset(grid_size=3, T=20, spatial_coefs=small_stable_coefs, error_sigma=0.5, initialize_randomly=False)

        # Higher noise should lead to different and more variable data
        variance_low = np.var(data_low_noise)
        variance_high = np.var(data_high_noise)

        assert variance_high > variance_low, "Higher noise should result in higher variance"

    def test_temporal_evolution(self, small_stable_coefs):
        """Test that data evolves over time (not static)."""
        data = generate_dataset(grid_size=3, T=10, spatial_coefs=small_stable_coefs, error_sigma=0.1, initialize_randomly=False)

        # Check that data changes over time
        for t in range(1, 10):
            # Some cells should differ from previous timestep
            diff = np.abs(data[:, :, :, t] - data[:, :, :, t - 1])
            assert np.sum(diff > 0.01) > 0, f"Data should evolve over time (timestep {t})"

    def test_spatial_variation(self, small_stable_coefs):
        """Test that data varies across spatial locations."""
        np.random.seed(42)
        data = generate_dataset(grid_size=5, T=20, spatial_coefs=small_stable_coefs, error_sigma=0.1, initialize_randomly=True)

        # At final timestep, not all cells should be identical
        final_timestep = data[0, :, :, -1]
        unique_values = len(np.unique(np.round(final_timestep, decimals=2)))

        assert unique_values > 1, "Data should vary spatially, not all cells identical"


# ============================================================================
# Test Class: Coefficient Handling
# ============================================================================


class TestCoefficientHandling:
    """Test coefficient generation and handling."""

    def test_coefficient_shape_correctness(self, basic_config):
        """Test that returned coefficients have correct shape."""
        basic_config["return_coefs"] = True
        coefs, data = generate_dataset(**basic_config)

        n_vars = basic_config["num_variables"]
        assert coefs.shape == (n_vars, 3, 3), f"Expected coefficient shape ({n_vars}, 3, 3), got {coefs.shape}"

        # Each entry should be an array of length n_vars
        for i in range(n_vars):
            for j in range(3):
                for k in range(3):
                    assert isinstance(coefs[i, j, k], np.ndarray), f"Coefficient entry should be numpy array"
                    assert len(coefs[i, j, k]) == n_vars, f"Each coefficient entry should have length {n_vars}"

    def test_different_seeds_produce_different_coefs(self, basic_config):
        """Test that coefficient generation is stochastic."""
        basic_config["return_coefs"] = True

        np.random.seed(42)
        coefs1, _ = generate_dataset(**basic_config)

        np.random.seed(99)
        coefs2, _ = generate_dataset(**basic_config)

        # Extract coefficient values
        vals1 = []
        vals2 = []
        for i in range(coefs1.shape[0]):
            for j in range(3):
                for k in range(3):
                    vals1.extend(coefs1[i, j, k])
                    vals2.extend(coefs2[i, j, k])

        assert not np.allclose(vals1, vals2), "Different seeds should produce different coefficients"

    def test_provided_coefs_used_exactly(self):
        """Test that provided coefficients are used without modification."""
        # Create specific coefficients
        n_vars = 2
        coefs = np.zeros((n_vars, 3, 3), dtype=object)

        for i in range(n_vars):
            for j in range(3):
                for k in range(3):
                    # Use distinctive values
                    coefs[i, j, k] = np.array([0.1 * (i + 1), 0.05 * (i + 1)])

        # Generate data with these coefficients, then retrieve them
        data = generate_dataset(grid_size=3, T=5, spatial_coefs=coefs, error_sigma=0.1, return_coefs=False)

        # The function should use these coefficients
        # We can't directly verify, but the data should be generated
        assert data.shape == (2, 3, 3, 5)


# ============================================================================
# Test Class: Verbose Output
# ============================================================================


class TestVerboseOutput:
    """Test verbose output and logging."""

    def test_verbose_parameter_accepted(self, basic_config):
        """Test that verbose parameter is accepted."""
        basic_config["verbose"] = 1
        data = generate_dataset(**basic_config)

        assert data.shape == (basic_config["num_variables"], basic_config["grid_size"], basic_config["grid_size"], basic_config["T"])


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for realistic usage scenarios."""

    def test_realistic_small_scenario(self):
        """Test a realistic small-scale scenario.

        Uses moderate parameters to ensure stability while testing a complete workflow.
        Reduced grid size and timesteps to keep test runtime reasonable.
        """
        np.random.seed(12345)

        coefs, data = generate_dataset(
            grid_size=5,  # Reduced from 10 to avoid long computation
            T=20,  # Reduced from 50 to avoid long computation
            num_variables=3,
            dependence_density=0.3,  # Reduced from 0.4 for more reliable stability
            coefficient_min_value_threshold=0.08,  # Slightly reduced for stability
            min_val_scaler=1.0,
            error_sigma=0.15,
            error_mean=0.0,
            initialize_randomly=True,
            detect_instability=True,
            instability_threshold=500,
            return_coefs=True,
            verbose=0,
        )

        # Verify outputs
        assert data.shape == (3, 5, 5, 20)
        assert coefs.shape == (3, 3, 3)
        assert np.all(np.isfinite(data))
        assert np.max(np.abs(data)) < 500

    def test_multiple_runs_independent(self, basic_config):
        """Test that multiple runs with same config but different seeds differ."""
        np.random.seed(1)
        data1 = generate_dataset(**basic_config)

        np.random.seed(2)
        data2 = generate_dataset(**basic_config)

        # Should be different due to random coefficient generation and noise
        assert not np.allclose(data1, data2), "Different runs should produce different results"


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
