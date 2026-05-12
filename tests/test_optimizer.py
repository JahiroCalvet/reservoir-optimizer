"""
Tests for the reservoir simulation engine and optimization core.
"""
import pytest
import numpy as np
from optimizer.reservoir_simulator import simulate, parameter_bounds, ReservoirState, darcy_flow_rate


class TestReservoirSimulator:

    def test_higher_porosity_increases_recovery(self):
        """More pore space should yield more oil."""
        low  = simulate(porosity=0.05, permeability=100, water_saturation=0.3, net_pay=50, pressure=3000)
        high = simulate(porosity=0.35, permeability=100, water_saturation=0.3, net_pay=50, pressure=3000)
        assert high > low

    def test_higher_permeability_increases_recovery(self):
        """Higher permeability allows faster flow."""
        low  = simulate(porosity=0.2, permeability=10,   water_saturation=0.3, net_pay=50, pressure=3000)
        high = simulate(porosity=0.2, permeability=2000, water_saturation=0.3, net_pay=50, pressure=3000)
        assert high > low

    def test_higher_water_saturation_decreases_recovery(self):
        """More water means less effective permeability to oil."""
        low_sw  = simulate(porosity=0.2, permeability=100, water_saturation=0.1, net_pay=50, pressure=3000)
        high_sw = simulate(porosity=0.2, permeability=100, water_saturation=0.8, net_pay=50, pressure=3000)
        assert low_sw > high_sw

    def test_higher_pressure_increases_recovery(self):
        """Larger pressure differential drives more flow."""
        low  = simulate(porosity=0.2, permeability=100, water_saturation=0.3, net_pay=50, pressure=800)
        high = simulate(porosity=0.2, permeability=100, water_saturation=0.3, net_pay=50, pressure=7000)
        assert high > low

    def test_recovery_is_non_negative(self):
        """Recovery should never be negative."""
        result = simulate(porosity=0.05, permeability=0.1, water_saturation=0.95, net_pay=1, pressure=500)
        assert result >= 0.0

    def test_parameter_bounds_coverage(self):
        """Simulation should run at all boundary values."""
        bounds = parameter_bounds()
        for key, (lo, hi) in bounds.items():
            params = dict(porosity=0.2, permeability=100, water_saturation=0.3, net_pay=50, pressure=3000)
            params[key] = lo
            assert simulate(**params) >= 0
            params[key] = hi
            assert simulate(**params) >= 0


class TestBayesianOptimizer:

    def test_optimizer_returns_improvement(self):
        """Bayesian optimizer should improve over random initial points."""
        from optimizer.bayesian.optimizer import run_bayesian_optimization
        result = run_bayesian_optimization(n_iterations=15, n_initial_points=5)
        history = result["convergence_history"]
        # Best at end should be >= best at start
        assert history[-1] >= history[0]

    def test_optimizer_result_structure(self):
        from optimizer.bayesian.optimizer import run_bayesian_optimization
        result = run_bayesian_optimization(n_iterations=10, n_initial_points=5)
        assert "best_params" in result
        assert "best_recovery" in result
        assert "convergence_history" in result
        assert result["best_recovery"] > 0


class TestSensitivityAnalysis:

    def test_sensitivity_returns_all_params(self):
        from optimizer.sensitivity.sobol import run_sensitivity_analysis, PARAM_LABELS
        result = run_sensitivity_analysis(n_samples=64)
        assert len(result["first_order"]) == len(PARAM_LABELS)
        assert len(result["total_order"]) == len(PARAM_LABELS)

    def test_sensitivity_indices_between_0_and_1(self):
        from optimizer.sensitivity.sobol import run_sensitivity_analysis
        result = run_sensitivity_analysis(n_samples=64)
        for s1 in result["first_order"]:
            assert -0.1 <= s1 <= 1.1   # allow small numerical noise
        for st in result["total_order"]:
            assert -0.1 <= st <= 1.1
