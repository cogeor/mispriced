"""
Verify recalculated valuations.
Checks counts and average mispricing per quarter to ensure historical fix worked.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db.session import SessionLocal
from src.db.models import ValuationResult, IndexMembership
from sqlalchemy import func
import pandas as pd

s = SessionLocal()

print("Verifying valuations...")
quarters = s.query(
    ValuationResult.snapshot_timestamp,
    func.count(ValuationResult.ticker),
    func.avg(ValuationResult.relative_error)
).group_by(ValuationResult.snapshot_timestamp).order_by(ValuationResult.snapshot_timestamp.desc()).all()

print(f"{'Quarter':<12} | {'Count':<6} | {'Avg Mispricing':<15}")
print("-" * 40)

for q in quarters:
    ts, count, avg_err = q
    print(f"{ts.date()}   | {count:<6} | {avg_err:.2%}")

# Check specific stock WKC
print("\nChecking WKC (World Kinect) valuations:")
wkc = s.query(ValuationResult).filter(ValuationResult.ticker == 'WKC').order_by(ValuationResult.snapshot_timestamp).all()
for v in wkc:
    print(f"  {v.snapshot_timestamp.date()}: Act=${v.actual_mcap/1e9:.2f}B, Pred=${v.predicted_mcap_mean/1e9:.2f}B, Err={v.relative_error:.1%}")

s.close()
