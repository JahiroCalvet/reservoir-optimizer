from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from data.schemas.reservoir import OptimizationRequest, OptimizationResult, ReservoirParams
from data.repositories.models import OptimizationSession
from optimizer.bayesian.optimizer import run_bayesian_optimization

router = APIRouter()


@router.post("/", response_model=OptimizationResult)
async def optimize(req: OptimizationRequest, db: AsyncSession = Depends(get_db)):
    """
    Run Bayesian optimization to find the reservoir parameters
    that maximise estimated oil recovery.
    """
    result = run_bayesian_optimization(n_iterations=req.n_iterations)

    # Persist session
    session = OptimizationSession(
        reservoir_name=req.reservoir_name,
        best_recovery=result["best_recovery"],
        n_iterations=result["n_iterations"],
        convergence_history=result["convergence_history"],
        **result["best_params"],
    )
    db.add(session)
    await db.commit()

    best_p = result["best_params"]
    return OptimizationResult(
        reservoir_name=req.reservoir_name,
        best_params=ReservoirParams(reservoir_name=req.reservoir_name, **best_p),
        best_recovery=result["best_recovery"],
        n_iterations=result["n_iterations"],
        convergence_history=result["convergence_history"],
        improvement_pct=result["improvement_pct"],
    )
