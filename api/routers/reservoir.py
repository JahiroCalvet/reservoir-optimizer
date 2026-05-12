from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from core.database import get_db
from data.schemas.reservoir import ReservoirParams, ReservoirResult
from data.repositories.models import ReservoirRun
from optimizer.reservoir_simulator import simulate

router = APIRouter()


@router.post("/", response_model=ReservoirResult)
async def create_run(params: ReservoirParams, db: AsyncSession = Depends(get_db)):
    """Simulate recovery for a given set of reservoir parameters."""
    recovery = simulate(
        porosity=params.porosity,
        permeability=params.permeability,
        water_saturation=params.water_saturation,
        net_pay=params.net_pay,
        pressure=params.pressure,
    )
    run = ReservoirRun(**params.model_dump(), estimated_recovery=recovery, run_type="manual")
    db.add(run)
    await db.commit()
    await db.refresh(run)
    return run


@router.get("/", response_model=list[ReservoirResult])
async def list_runs(db: AsyncSession = Depends(get_db)):
    """List all simulation runs."""
    result = await db.execute(select(ReservoirRun).order_by(ReservoirRun.created_at.desc()))
    return result.scalars().all()
