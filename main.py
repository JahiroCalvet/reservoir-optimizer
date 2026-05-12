from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.routers import optimization, reservoir, sensitivity, health
from core.config import settings
from core.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    await init_db()
    yield


app = FastAPI(
    title="Reservoir Optimizer API",
    description="""
    A scientific backend for Bayesian optimization of reservoir parameters
    in oil & gas exploration. Combines surrogate modeling, sensitivity analysis,
    and real-time performance analytics.
    
    Built to demonstrate skills for Halliburton Back-end Developer role.
    """,
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router,        prefix="/api/v1",           tags=["Health"])
app.include_router(reservoir.router,     prefix="/api/v1/reservoirs", tags=["Reservoirs"])
app.include_router(optimization.router,  prefix="/api/v1/optimize",   tags=["Optimization"])
app.include_router(sensitivity.router,   prefix="/api/v1/sensitivity", tags=["Sensitivity"])
