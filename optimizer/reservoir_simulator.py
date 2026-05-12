"""
Reservoir Simulation Engine
============================
Implements a physics-based production estimation model using
Darcy's Law and reservoir engineering fundamentals.

This is the scientific core that the Bayesian optimizer calls
as its objective function.

References:
    - Darcy, H. (1856). Les Fontaines Publiques de la Ville de Dijon.
    - Craft, B.C. & Hawkins, M. (1991). Applied Petroleum Reservoir Engineering.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class ReservoirState:
    porosity:         float   # fraction [0, 0.5]
    permeability:     float   # milliDarcies [0.1, 5000]
    water_saturation: float   # fraction [0, 1]
    net_pay:          float   # metres [1, 500]
    pressure:         float   # psi [500, 8000]


def darcy_flow_rate(state: ReservoirState) -> float:
    """
    Estimate oil production rate using a simplified Darcy flow model.

    Q = (k * A * ΔP) / (μ * L)

    Where:
        k  = effective permeability to oil (mD → m²)
        A  = cross-sectional area (net_pay × well_radius)
        ΔP = driving pressure (reservoir pressure - wellbore pressure)
        μ  = oil viscosity (assumed 2 cP for medium crude)
        L  = drainage length
    """
    # Convert milliDarcy to m²
    k_md_to_m2 = 9.869233e-16
    k_eff = state.permeability * k_md_to_m2 * (1 - state.water_saturation)

    # Effective drainage area (simplified cylindrical model)
    well_radius    = 0.1       # metres
    drainage_radius = 500.0   # metres
    area = 2 * np.pi * well_radius * state.net_pay

    # Pressure differential (assume wellbore pressure = 20% of reservoir)
    delta_p_pa = (state.pressure * 0.8) * 6894.76   # psi → Pascal

    # Darcy length (log-mean drainage radius)
    length = np.log(drainage_radius / well_radius)

    # Oil viscosity (Pa·s)
    mu = 2e-3

    # Darcy flow rate (m³/s)
    q_m3s = (k_eff * area * delta_p_pa) / (mu * length)

    # Convert m³/s → barrels/day
    q_bpd = q_m3s * 543439.65

    # Apply porosity factor (more pore space → more recoverable oil)
    recovery_factor = state.porosity * 0.35   # 35% recovery efficiency

    return max(0.0, q_bpd * recovery_factor)


def simulate(
    porosity: float,
    permeability: float,
    water_saturation: float,
    net_pay: float,
    pressure: float,
    noise: bool = False,
) -> float:
    """
    Public simulation entry point.
    Optionally adds measurement noise to simulate real sensor uncertainty.
    """
    state = ReservoirState(
        porosity=porosity,
        permeability=permeability,
        water_saturation=water_saturation,
        net_pay=net_pay,
        pressure=pressure,
    )
    result = darcy_flow_rate(state)
    if noise:
        result *= (1 + np.random.normal(0, 0.02))   # ±2% measurement noise
    return round(result, 2)


def parameter_bounds() -> dict:
    """Returns physical bounds for all reservoir parameters."""
    return {
        "porosity":         (0.05, 0.40),
        "permeability":     (1.0,  3000.0),
        "water_saturation": (0.05, 0.80),
        "net_pay":          (5.0,  400.0),
        "pressure":         (800,  7000),
    }
