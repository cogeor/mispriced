# Mispriced

**Fundamental valuation software** — predicts fair market capitalization from financial statements and identifies mispriced securities. Now live at [mispriced.ch](https://mispriced.ch).

![Mispriced Dashboard](assets/mispriced.png)

## What It Does

**Mispriced** answers the question: *"Given these fundamentals, what should this company be worth?"*

- Predicts fair market cap from balance sheet data (revenue, EBITDA, debt, cash flow, etc.)
- Identifies mispriced securities by comparing predicted vs. actual market caps
- Tracks mispricing across 12 global indices (S&P 500, NASDAQ 100, DAX, FTSE 100, etc.)
- Provides uncertainty estimates for every prediction
- Backtests signal quality across multiple time horizons

This is **not** a price forecaster. It's a financial audit tool that values companies cross-sectionally based on fundamentals.

## Design Philosophy

**What this system is:**
- Fair value estimator from financial fundamentals
- Mispricing detector via cross-sectional comparison
- Index-level analyzer with uncertainty quantification

**What this system is NOT:**
- Time-series price predictor (no OHLCV modeling)
- Daily return forecaster (rebalances quarterly)
- Technical analysis tool (fundamentals only)

**Key principles:**
1. **Financial audit mindset** — Predict what companies *should* be worth
2. **Uncertainty quantification** — Every prediction has confidence intervals
3. **No data leakage** — Out-of-fold predictions ensure fair evaluation
4. **Currency normalization** — All values converted to USD
5. **Reproducibility** — Model configs and data versions tracked

---

## How It Works

### Model Architecture

| Component | Choice |
|-----------|--------|
| **Algorithm** | XGBoost (Gradient Boosted Trees) |
| **Target** | log(market_cap) for numerical stability |
| **Validation** | 10×5 Repeated K-Fold Cross-Validation |
| **Predictions per stock** | 50 (10 repeats × 5 folds) |

### Cross-Sectional Training

Each quarter is trained independently — the model only sees companies from that quarter. This is critical because:

1. **No future leakage** — The model cannot learn from future quarters
2. **Market regime adaptation** — Valuation multiples change over time (e.g., tech multiples were higher in 2021)
3. **Fair comparison** — All companies are valued against their contemporaries, not against historical norms

### Repeated Cross-Validation

The model uses **repeated K-fold cross-validation** to generate fair predictions with uncertainty:

```
For each of N repeats (N=10):
    Split data into K folds (K=5)
    For each fold:
        Train XGBoost on K-1 folds
        Predict on held-out fold

Final prediction for each ticker:
    μ = mean of all out-of-fold predictions
    σ = std of all out-of-fold predictions
```

This produces a **distribution of predictions** for each ticker, capturing model uncertainty without data leakage.

### Mispricing Calculation

Relative error measures how much a security deviates from fair value:

```
Relative Error = (Predicted - Actual) / Actual
```

- **Positive** → Underpriced (model thinks it's worth more)
- **Negative** → Overpriced (model thinks it's worth less)

### Size Premium Correction

Raw mispricing exhibits a systematic size effect: smaller companies tend to show positive mispricing while larger companies show negative. The **size-neutral** mode corrects for this by fitting a curve to the mispricing vs. market cap relationship:

```
Size-Neutral Mispricing = Raw Mispricing - Size Premium(market_cap)
```

### Features

The model uses ~30 fundamental features:

| Category | Features |
|----------|----------|
| **Income Statement** | Revenue, Gross Profit, EBITDA, Operating Income, Net Income |
| **Balance Sheet** | Total Debt, Total Cash, Total Assets, Book Value, Working Capital |
| **Cash Flow** | Free Cash Flow, Operating Cash Flow, CapEx |
| **Ratios** | Profit Margins, ROA, ROE, Debt-to-Equity, Quick Ratio, Current Ratio |
| **Other** | Shares Outstanding, Float Shares, Insider/Institutional Holdings |

No price-derived features are used — only accounting fundamentals.

---

## Index Aggregation

Given valuations for index constituents, compute aggregate index mispricing:

```
Index Mispricing = (Σ wᵢ × predictedᵢ - Σ wᵢ × actualᵢ) / Σ wᵢ × actualᵢ
```

Where weights (wᵢ) are market-cap based to match real index construction.

### Uncertainty Propagation

Assuming independent prediction errors:

```
σ_index = √(Σ wᵢ² × σᵢ²)
```

This provides a lower bound on index uncertainty.

---

## Supported Indices

| Region | Indices |
|--------|---------|
| **US** | S&P 500, NASDAQ 100, S&P 400 (Mid Cap), S&P 600 (Small Cap), Russell 1000 |
| **Europe** | DAX (Germany), FTSE 100 (UK), Euro Stoxx 50, CAC 40 (France), SMI (Switzerland) |
| **Asia** | Nifty 50 (India), SSE 50 (China) |

All indices sourced from Wikipedia and mapped to yfinance ticker format.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run valuation pipeline (all quarters)
python scripts/run_quarterly_valuation.py

# Generate dashboard data
python scripts/generate_dashboard.py

# Build and serve web dashboard
cd web && npm install && npm run dev
```

---

## Example Output

### Valuation Result

```json
{
  "ticker": "AAPL",
  "predicted_mcap_mean": 2850000000000,
  "predicted_mcap_std": 185000000000,
  "actual_mcap": 2950000000000,
  "relative_error": -0.034,
  "model_version": "gbr_baseline_v3"
}
```

**Interpretation**: Apple is **overpriced by 3.4%** relative to predicted fair value.

### Index Analysis

```json
{
  "index_id": "SP500",
  "actual_mcap": 66787073988608,
  "predicted_mcap": 63342789948551,
  "index_mispricing": -0.052,
  "n_tickers": 501
}
```

**Interpretation**: S&P 500 is **overpriced by 5.2%** on aggregate.

---

## Web Dashboard

The dashboard displays:
- **Valuation Map**: All stocks sized by market cap, colored by mispricing
- **Sector/Index Charts**: Aggregated mispricing by sector and index
- **Time Series**: Historical mispricing trends
- **Signal Backtest**: IC heatmaps and decay charts across horizons
- **All Stocks Table**: Searchable, sortable stock list with export

Built with Vite + TypeScript + Plotly.

---

## Architecture

```
mispriced/
├── src/
│   ├── ingestion/      # Fetch financial data from yfinance
│   ├── db/             # SQLite database models (SQLAlchemy)
│   ├── valuation/      # XGBoost prediction + feature engineering
│   └── evaluation/     # Repeated CV implementation
├── scripts/
│   ├── run_quarterly_valuation.py   # Main valuation pipeline
│   └── generate_dashboard.py        # Export data for web
├── web/                # Vite + TypeScript dashboard
│   ├── src/            # Frontend source
│   └── public/         # Static JSON data (quarters/*.json)
└── mispriced.db        # SQLite database
```

### Data Flow

```
yfinance API → ingestion → SQLite DB → valuation (XGBoost CV) → dashboard JSON → web
```

### Database Schema

| Table | Purpose |
|-------|---------|
| `tickers` | Company metadata (sector, industry, currency) |
| `financial_snapshots` | Point-in-time balance sheet data (all USD) |
| `valuation_results` | Model predictions with uncertainty |
| `index_memberships` | Index constituent mappings |

---

## Data Sources

- **Financial Data**: [yfinance](https://github.com/ranaroussi/yfinance)
- **FX Rates**: [exchangerate.host](https://exchangerate.host/)
- **Index Constituents**: Wikipedia (automated scraping)

## Technologies

- **Model**: [XGBoost](https://xgboost.readthedocs.io/)
- **Database**: SQLite + [SQLAlchemy](https://www.sqlalchemy.org/)
- **Validation**: [scikit-learn](https://scikit-learn.org/) cross-validation
- **Visualization**: [Plotly.js](https://plotly.com/javascript/)
- **Frontend**: [Vite](https://vitejs.dev/) + TypeScript

---

## License

This project is licensed under the GNU General Public License v3.0 — see [LICENSE](LICENSE) for details.
