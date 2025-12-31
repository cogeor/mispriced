# Reporting

> Generates human-readable reports and visualizations from evaluation results.

---

## Responsibilities

- Generate human-readable outputs (HTML/Markdown/PDF)
- Create summary tables and charts
- Produce snapshot-based timelines
- Support scheduled and on-demand report generation

---

## Inputs

- DB query results:
  - Snapshots, valuations, portfolios
  - Evaluation metrics
- Report configuration:
  - Output format (HTML, Markdown, PDF)
  - Template selection
  - Date ranges and filters

---

## Outputs

- Static report files (HTML, MD, PDF)
- Inline visualizations (charts, tables)
- Report metadata (generation timestamp, config used)

---

## Dependencies

### External Packages
- `jinja2` - Template rendering
- `matplotlib` / `plotly` - Chart generation
- `weasyprint` (optional) - PDF generation
- `pandas` - Data manipulation for tables

### Internal Modules
- `src/db/` - Repository access
- `src/evaluation/` - Evaluation results

---

## Folder Structure

```
src/reporting/
  __init__.py
  service.py              # Main reporting orchestrator
  generators/
    __init__.py
    html.py               # HTML report generator
    markdown.py           # Markdown report generator
    pdf.py                # PDF report generator
  charts/
    __init__.py
    mispricing.py         # Mispricing visualizations
    convergence.py        # Convergence charts
    portfolio.py          # Portfolio composition charts
  templates/
    base.html
    universe_report.html
    ticker_report.html
    evaluation_report.html
  config.py
```

---

## Report Types

### Universe Report

Overview of entire ticker universe at a point in time.

**Contents**:
- Summary statistics (# tickers, avg mispricing, avg quality)
- Sector breakdown
- Top undervalued/overvalued tickers table
- Mispricing distribution chart

### Ticker Report

Deep-dive on a single ticker's history.

**Contents**:
- Company info and sector
- Statement history timeline
- Valuation history chart (intrinsic vs price)
- Mispricing trend
- Evaluation metrics (persistence, convergence)

### Evaluation Report

Results from an evaluation run.

**Contents**:
- Run configuration
- Aggregate metrics summary
- Cohort breakdown tables
- Signal effectiveness visualizations
- Portfolio-level results (if applicable)

---

## Visualization Examples

### Mispricing Timeline

```python
def plot_mispricing_timeline(ticker_history: List[Snapshot]) -> Figure:
    """
    Line chart showing:
    - Intrinsic value per share (blue)
    - Market price (gray)
    - Mispricing % (secondary axis, red/green bars)
    """
    fig, ax1 = plt.subplots()
    
    dates = [s.snapshot_timestamp for s in ticker_history]
    prices = [s.price_t0 for s in ticker_history]
    ivs = [s.intrinsic_value for s in ticker_history]
    
    ax1.plot(dates, prices, 'gray', label='Market Price')
    ax1.plot(dates, ivs, 'blue', label='Intrinsic Value')
    
    ax2 = ax1.twinx()
    mispricings = [s.mispricing_pct for s in ticker_history]
    colors = ['green' if m < 0 else 'red' for m in mispricings]
    ax2.bar(dates, mispricings, color=colors, alpha=0.3)
    
    return fig
```

### Sector Heatmap

```python
def plot_sector_heatmap(universe_data: pd.DataFrame) -> Figure:
    """
    Heatmap showing avg mispricing by sector x quality bucket.
    """
    pivot = universe_data.pivot_table(
        values='mispricing_pct',
        index='sector',
        columns='quality_bucket',
        aggfunc='mean'
    )
    
    return sns.heatmap(pivot, cmap='RdYlGn_r', center=0)
```

---

## Design Decisions

### ✅ DECIDED: Output Format Priority

**Decision**: HTML first with static file generation.

**Rationale**:
- Self-contained HTML files can be opened anywhere without server
- Interactive charts enhance data exploration
- Professional financial styling with Tailwind CSS
- Easy to share and archive

### ✅ DECIDED: Chart Library

**Decision**: Plotly.js (via CDN)

**Rationale**:
- Industry standard for financial dashboards (Bloomberg-style)
- Interactive hover, zoom, pan out of the box
- CDN-based, no build step required
- Excellent log-scale support for market cap visualization

### ✅ DECIDED: Dashboard Stack

**Decision**: Static HTML + Plotly.js + Tailwind CSS (CDN)

**Alternatives Considered**:
- Plotly Dash: Requires server, overkill for static reports
- Streamlit: Less customizable, generic appearance
- React + D3: Separate tech stack, complex build process

**Benefits**:
- Zero runtime dependencies (just open the HTML file)
- Professional dark theme matching fintech standards
- Portable across any environment

### ✅ IMPLEMENTED: Dashboard Features

**File**: `scripts/generate_dashboard.py` → `plots/dashboard.html`

**Components**:
1. Summary statistics cards (total market cap, undervalued/overvalued counts)
2. Index mispricing bar chart (13 global indices)
3. Valuation scatter plot (actual vs predicted market cap)
   - Toggle: color by mispricing degree OR sector
   - Log-scale axes for market cap range
   - Diagonal reference line (fair value)
4. Sector breakdown bar chart (avg mispricing by sector)
5. Top 10 undervalued/overvalued tables

**Sector Color Palette** (Bloomberg-inspired):
- Technology: #2563eb (blue)
- Healthcare: #059669 (emerald)
- Financial Services: #7c3aed (violet)
- Consumer Cyclical: #db2777 (pink)
- Energy: #dc2626 (red)
- Utilities: #65a30d (lime)
- Real Estate: #a855f7 (purple)
- Basic Materials: #d97706 (amber)

### Signal Quality Analysis Section

Displays backtest metrics from `data/signal_backtest_detailed.csv`:

**Charts**:
1. **IC by Sector** - Bar chart of Information Coefficient by sector (30-day horizon)
2. **IC by Index** - Bar chart of IC by index (30-day horizon)
3. **IC Decay by Horizon** - Dual-axis chart showing IC decay and spread across 10/30/60/90-day horizons

**Metrics**:
- **IC (Information Coefficient)**: Correlation between valuation signal and forward returns
- **p-value**: Statistical significance (* = p<0.05, ~ = p<0.10)
- **Spread**: Q5-Q1 quantile return spread
- **Hit Rate**: Percentage of correct direction predictions

**Data Requirements**:
- Uses detailed backtest data (`signal_backtest_detailed.csv`)
- Filters for groups with >100 total observations
- Aggregates across quarters using observation-weighted averages

### 📌 TODO: Template System

Design reusable template components:
- Header/footer
- Table styling
- Chart embedding
- Conditional sections

### 💡 ALTERNATIVE: Jupyter Notebook Output

Instead of static reports, generate Jupyter notebooks that users can explore interactively.

---

## Constraints

- ⚡ Reports must be self-contained (no external dependencies)
- ⚡ Charts must include legends and axis labels
- ⚡ Reports must include generation metadata
- ⚡ Must support command-line and programmatic generation

---

## Integration Tests

### Test Scope

Integration tests verify report generation with real data and file output.

### Test Cases

```python
# tests/integration/test_reporting.py

class TestReportingIntegration:
    """Integration tests for report generation."""
    
    def test_universe_report_html(self, test_db_with_data, tmp_path):
        """
        Test HTML universe report generation.
        
        Verifies:
        - HTML file is created
        - Contains expected sections
        - Includes generation timestamp
        """
        output_path = tmp_path / "universe_report.html"
        generate_universe_report(output_path, format="html")
        
        assert output_path.exists()
        content = output_path.read_text()
        assert "Summary" in content
        assert "generated_at" in content
    
    def test_ticker_report(self, test_db_with_data, tmp_path):
        """
        Test ticker-specific report generation.
        
        Verifies:
        - Report includes company info
        - Valuation history chart embedded
        - Mispricing trend visible
        """
        output_path = tmp_path / "aapl_report.html"
        generate_ticker_report("AAPL", output_path)
        
        assert output_path.exists()
    
    def test_chart_generation(self, sample_ticker_history):
        """
        Test chart rendering.
        
        Verifies:
        - Figure is created
        - Has correct axes
        - Legend is present
        """
        fig = plot_mispricing_timeline(sample_ticker_history)
        
        assert fig is not None
        assert len(fig.axes) >= 1
```

### Running Tests

```bash
# Run reporting integration tests
pytest tests/integration/test_reporting.py -v

# Run with visual output (for chart inspection)
pytest tests/integration/test_reporting.py -v --show-plots

# Run with coverage
pytest tests/integration/test_reporting.py --cov=src/reporting
```
