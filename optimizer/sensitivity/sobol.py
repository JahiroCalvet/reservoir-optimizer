"""
Sensitivity Analysis — Sobol Method
=====================================
Uses SALib to compute Sobol sensitivity indices, answering:

    "Which reservoir parameters most influence production recovery?"

First-order index  (S1): Direct contribution of each parameter alone.
Total-order index  (ST): Contribution including interactions with others.

This is critical in reservoir engineering to prioritise what to measure
in the field (expensive) vs what can be assumed with less precision.
"""

import numpy as np
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze
from optimizer.reservoir_simulator import simulate, parameter_bounds


PARAM_NAMES = [
    "porosity",
    "permeability",
    "water_saturation",
    "net_pay",
    "pressure",
]

PARAM_LABELS = [
    "Porosity",
    "Permeability (mD)",
    "Water Saturation",
    "Net Pay (m)",
    "Pressure (psi)",
]


def run_sensitivity_analysis(n_samples: int = 512) -> dict:
    """
    Runs a Sobol global sensitivity analysis over all reservoir parameters.

    Args:
        n_samples: Base sample size (actual evaluations = n_samples × (2N+2))

    Returns:
        dict with S1 and ST indices for each parameter.
    """
    bounds = parameter_bounds()

    problem = {
        "num_vars": len(PARAM_NAMES),
        "names":    PARAM_NAMES,
        "bounds":   [list(bounds[p]) for p in PARAM_NAMES],
    }

    # Generate Sobol quasi-random samples
    param_values = sobol_sample.sample(problem, n_samples, calc_second_order=False)

    # Evaluate the simulation model at each sample point
    Y = np.array([
        simulate(
            porosity=row[0],
            permeability=row[1],
            water_saturation=row[2],
            net_pay=row[3],
            pressure=row[4],
        )
        for row in param_values
    ])

    # Compute Sobol indices
    Si = sobol_analyze.analyze(problem, Y, calc_second_order=False, print_to_console=False)

    s1 = [round(float(v), 4) for v in Si["S1"]]
    st = [round(float(v), 4) for v in Si["ST"]]

    most_influential = PARAM_LABELS[int(np.argmax(Si["ST"]))]

    return {
        "parameters":     PARAM_LABELS,
        "first_order":    s1,
        "total_order":    st,
        "most_influential": most_influential,
    }
