
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.valuation.size_correction import compute_residual_mispricing
    print("Import successful")
except Exception as e:
    print(f"Import failed: {e}")
