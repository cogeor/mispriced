"""Quick visualization of snapshot distribution by date - uses built-in HTML."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db.session import SessionLocal
from src.db.models.snapshot import FinancialSnapshot
from sqlalchemy import func
import json

s = SessionLocal()

result = s.query(
    FinancialSnapshot.snapshot_timestamp,
    func.count(FinancialSnapshot.ticker).label('count')
).group_by(FinancialSnapshot.snapshot_timestamp).order_by(FinancialSnapshot.snapshot_timestamp).all()

s.close()

data = [(r[0].strftime('%Y-%m-%d'), r[1]) for r in result]

print("Snapshot distribution:")
max_count = max(c for _, c in data)
for date, count in data:
    bar_len = int(50 * count / max_count)
    bar = '█' * bar_len
    print(f"  {date}: {count:>5} {bar}")

# Create simple HTML with embedded chart.js
html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Snapshot Distribution</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ background: #1a1a2e; color: #eee; font-family: sans-serif; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #fff; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Financial Snapshots per Date</h1>
        <canvas id="chart"></canvas>
    </div>
    <script>
        const data = {json.dumps(data)};
        const ctx = document.getElementById('chart').getContext('2d');
        new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: data.map(d => d[0]),
                datasets: [{{
                    label: 'Snapshots',
                    data: data.map(d => d[1]),
                    backgroundColor: data.map(d => d[1] > 2000 ? '#2ecc71' : '#3498db'),
                    borderColor: data.map(d => d[1] > 2000 ? '#27ae60' : '#2980b9'),
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{ display: false }},
                    title: {{ display: false }}
                }},
                scales: {{
                    y: {{ beginAtZero: true, grid: {{ color: '#333' }}, ticks: {{ color: '#aaa' }} }},
                    x: {{ grid: {{ color: '#333' }}, ticks: {{ color: '#aaa' }} }}
                }}
            }}
        }});
    </script>
</body>
</html>'''

os.makedirs('plots', exist_ok=True)
with open('plots/snapshot_distribution.html', 'w') as f:
    f.write(html)
print('\nSaved to plots/snapshot_distribution.html')
