
import sys
import os
from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.db.config import DATABASE_URL

engine = create_engine(DATABASE_URL)
try:
    with engine.connect() as conn:
        result = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'valuation_results' AND column_name = 'residual_error'"))
        row = result.fetchone()
        if row:
            print("Column residual_error EXISTS")
        else:
            print("Column residual_error MISSING")
except Exception as e:
    print(f"Error: {e}")
