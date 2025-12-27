import sys, os
sys.path.insert(0, '.')
from src.db.session import SessionLocal
from src.db.models.snapshot import FinancialSnapshot
from sqlalchemy import func

s = SessionLocal()
result = s.query(
    FinancialSnapshot.snapshot_timestamp,
    func.count(FinancialSnapshot.ticker).label('count')
).group_by(FinancialSnapshot.snapshot_timestamp).order_by(FinancialSnapshot.snapshot_timestamp).all()
s.close()

print('SNAPSHOT DISTRIBUTION BY DATE')
print('=' * 70)
max_c = max(r[1] for r in result)
for ts, c in result:
    bar = '#' * int(40 * c / max_c)
    m = ts.month
    q = 'Q1' if m <= 3 else ('Q2' if m <= 6 else ('Q3' if m <= 9 else 'Q4'))
    date_str = ts.strftime('%Y-%m-%d')
    print(f'{date_str} ({q}): {c:>5}  {bar}')
print('=' * 70)
print(f'Total: {sum(r[1] for r in result)} snapshots across {len(result)} dates')
