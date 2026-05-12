from fastapi import APIRouter
from data.schemas.reservoir import SensitivityRequest, SensitivityResult
from optimizer.sensitivity.sobol import run_sensitivity_analysis

router = APIRouter()


@router.post("/", response_model=SensitivityResult)
async def sensitivity(req: SensitivityRequest):
    """
    Run Sobol global sensitivity analysis.
    Returns first-order and total-order indices for each parameter,
    indicating which reservoir properties most influence oil recovery.
    """
    result = run_sensitivity_analysis(n_samples=req.n_samples)
    return SensitivityResult(reservoir_name=req.reservoir_name, **result)
