import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.db.session import SessionLocal
from src.db.models import ValuationResult
from sqlalchemy import func

s = SessionLocal()

result = s.query(
    ValuationResult.snapshot_timestamp,
    func.count(ValuationResult.ticker).label('count')
).group_by(ValuationResult.snapshot_timestamp).order_by(ValuationResult.snapshot_timestamp).all()

lines = ['=== VALUATION RESULTS ===']
total = 0
for ts, c in result:
    lines.append(f'  {ts.date()}: {c:>5} valuations')
    total += c

lines.append(f'\nTotal: {total} valuations across {len(result)} quarters')
s.close()

output = '\n'.join(lines)
print(output)
with open('data/valuation_counts.txt', 'w') as f:
    f.write(output)
