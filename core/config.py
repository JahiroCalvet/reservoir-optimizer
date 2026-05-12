from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Reservoir Optimizer API"
    database_url: str = "sqlite+aiosqlite:///./reservoir.db"
    n_initial_points: int = 10
    n_iterations: int = 30
    random_seed: int = 42

    class Config:
        env_file = ".env"


settings = Settings()
