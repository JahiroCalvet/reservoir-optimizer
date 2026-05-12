# api/routers/health.py
from fastapi import APIRouter
router = APIRouter()

@router.get("/health")
async def health():
    return {"status": "UP", "service": "reservoir-optimizer", "version": "1.0.0"}
