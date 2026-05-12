from sqlalchemy import Column, Integer, Float, String, DateTime, JSON
from sqlalchemy.sql import func
from core.database import Base


class ReservoirRun(Base):
    """Stores each optimization run and its results."""
    __tablename__ = "reservoir_runs"

    id              = Column(Integer, primary_key=True, index=True)
    reservoir_name  = Column(String, nullable=False)
    porosity        = Column(Float, nullable=False)   # fraction 0-1
    permeability    = Column(Float, nullable=False)   # mD
    water_saturation= Column(Float, nullable=False)   # fraction 0-1
    net_pay         = Column(Float, nullable=False)   # metres
    pressure        = Column(Float, nullable=False)   # psi
    estimated_recovery = Column(Float, nullable=False) # barrels/day
    run_type        = Column(String, default="manual") # manual | bayesian
    created_at      = Column(DateTime(timezone=True), server_default=func.now())


class OptimizationSession(Base):
    """Stores the result of a full Bayesian optimization run."""
    __tablename__ = "optimization_sessions"

    id              = Column(Integer, primary_key=True, index=True)
    reservoir_name  = Column(String, nullable=False)
    best_porosity   = Column(Float)
    best_permeability = Column(Float)
    best_water_saturation = Column(Float)
    best_net_pay    = Column(Float)
    best_pressure   = Column(Float)
    best_recovery   = Column(Float)
    n_iterations    = Column(Integer)
    convergence_history = Column(JSON)  # list of best values per iteration
    created_at      = Column(DateTime(timezone=True), server_default=func.now())
