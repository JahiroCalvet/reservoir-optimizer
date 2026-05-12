"""
Bayesian Optimization Engine
==============================
Uses scikit-optimize (skopt) with a Gaussian Process surrogate model
to find reservoir parameters that maximise estimated oil recovery.

Why Bayesian Optimization over grid search?
    - Each simulation call is expensive (in real systems, hours of compute)
    - BO uses past evaluations to intelligently choose the next point
    - Balances exploration (trying unknown regions) vs exploitation (refining known optima)
    - Typically converges in 20-50 iterations vs thousands for grid search
"""

import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from typing import List, Tuple

from optimizer.reservoir_simulator import simulate, parameter_bounds


def run_bayesian_optimization(
    n_iterations: int = 30,
    n_initial_points: int = 10,
    random_seed: int = 42,
) -> dict:
    """
    Runs Bayesian optimization over the reservoir parameter space.

    Returns:
        dict with best parameters, best recovery, and convergence history.
    """
    bounds = parameter_bounds()

    # Define search space for skopt
    space = [
        Real(*bounds["porosity"],         name="porosity"),
        Real(*bounds["permeability"],     name="permeability"),
        Real(*bounds["water_saturation"], name="water_saturation"),
        Real(*bounds["net_pay"],          name="net_pay"),
        Real(*bounds["pressure"],         name="pressure"),
    ]

    @use_named_args(space)
    def objective(porosity, permeability, water_saturation, net_pay, pressure):
        # skopt minimizes — negate to maximize recovery
        recovery = simulate(
            porosity=porosity,
            permeability=permeability,
            water_saturation=water_saturation,
            net_pay=net_pay,
            pressure=pressure,
            noise=True,
        )
        return -recovery

    result = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=n_iterations,
        n_initial_points=n_initial_points,
        acq_func="EI",          # Expected Improvement acquisition function
        random_state=random_seed,
        verbose=False,
    )

    # Build convergence history (best value seen so far at each iteration)
    best_so_far = []
    current_best = float("inf")
    for val in result.func_vals:
        current_best = min(current_best, val)
        best_so_far.append(round(-current_best, 2))

    best_params = {
        "porosity":         round(result.x[0], 4),
        "permeability":     round(result.x[1], 2),
        "water_saturation": round(result.x[2], 4),
        "net_pay":          round(result.x[3], 2),
        "pressure":         round(result.x[4], 1),
    }

    best_recovery = round(-result.fun, 2)
    initial_recovery = best_so_far[0]
    improvement_pct = round(
        (best_recovery - initial_recovery) / max(initial_recovery, 1) * 100, 1
    )

    return {
        "best_params":          best_params,
        "best_recovery":        best_recovery,
        "convergence_history":  best_so_far,
        "improvement_pct":      improvement_pct,
        "n_iterations":         n_iterations,
    }
