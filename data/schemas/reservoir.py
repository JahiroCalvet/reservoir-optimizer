from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class ReservoirParams(BaseModel):
    """Physical parameters of a reservoir."""
    reservoir_name:   str   = Field(..., example="Marlim Sul")
    porosity:         float = Field(..., ge=0.0, le=0.5,    description="Rock porosity (fraction)")
    permeability:     float = Field(..., ge=0.1, le=5000.0, description="Permeability (mD)")
    water_saturation: float = Field(..., ge=0.0, le=1.0,    description="Water saturation (fraction)")
    net_pay:          float = Field(..., ge=1.0, le=500.0,  description="Net pay thickness (metres)")
    pressure:         float = Field(..., ge=500, le=8000,   description="Reservoir pressure (psi)")


class ReservoirResult(ReservoirParams):
    id: int
    estimated_recovery: float
    run_type: str
    created_at: datetime

    class Config:
        from_attributes = True


class OptimizationRequest(BaseModel):
    reservoir_name: str = Field(..., example="Marlim Sul")
    n_iterations:   int = Field(default=30, ge=5, le=100)


class OptimizationResult(BaseModel):
    reservoir_name:    str
    best_params:       ReservoirParams
    best_recovery:     float
    n_iterations:      int
    convergence_history: List[float]
    improvement_pct:   float


class SensitivityRequest(BaseModel):
    reservoir_name: str
    n_samples:      int = Field(default=512, ge=64, le=2048)


class SensitivityResult(BaseModel):
    reservoir_name: str
    parameters:     List[str]
    first_order:    List[float]   # S1 — direct effect
    total_order:    List[float]   # ST — total effect incl. interactions
    most_influential: str
