import os

# Default to SQLite for local development if Postgres is unavailable
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "sqlite:///./mispriced.db"
)
