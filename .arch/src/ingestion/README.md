# Data Ingestion

> Fetches financial statements from yfinance, attempts USD conversion at ingestion, and creates snapshot rows. Also manages index constituency.

---

## Responsibilities

- Resolve tickers to company metadata (name, exchange, currency, sector)
- Fetch financial statements (income, balance sheet, cashflow) and key statistics
- Fetch **minimal price context** for statement date window (t-1, t0, t+1)
- **Attempt USD conversion** at ingestion; fallback to local currency if FX rate unavailable
- **Always store original currency** field, even for USD-native tickers
- Normalize into one `FinancialSnapshot` row per `(ticker, snapshot_timestamp)`
- Store raw payloads for audit and reprocessing
- Support **ticker list file ingestion** for on-demand batch processing
- **Manage index constituents**: Retrieve and store which tickers belong to which indices over time

---

## Currency Handling at Ingestion

### ✅ RESOLVED: Attempt USD Conversion with Fallback

**Strategy**:
1. Attempt to convert all monetary values to USD using day's FX rate
2. If FX rate unavailable (API failure, missing date), **keep original currency**
3. **Always store** both `currency` and `fx_rate_to_usd` fields
4. Do NOT assume USD - even USD-native stocks explicitly store `currency="USD"`

```python
def convert_with_fallback(
    value: float,
    from_currency: str,
    as_of_date: date,
    fx_provider: FXRateProvider,
) -> tuple[float, str, float]:
    """
    Attempt USD conversion with fallback to local currency.
    
    Returns:
        (converted_value, stored_currency, fx_rate_to_usd)
    """
    if from_currency == "USD":
        return value, "USD", 1.0
    
    try:
        rate = fx_provider.get_rate(from_currency, "USD", as_of_date)
        return value * rate, "USD", rate
    except FXRateUnavailable:
        # Fallback: keep original currency
        return value, from_currency, None
```

### Currency Fields Stored

| Field | Description | Example |
|-------|-------------|---------|
| `original_currency` | Native currency of the company | `"GBP"` |
| `stored_currency` | Currency values are stored in | `"USD"` or `"GBP"` |
| `fx_rate_to_usd` | Conversion rate used (null if not converted) | `1.27` or `null` |

---

## Ticker List File Ingestion

### On-Demand Batch Ingestion

Support ingesting a series of tickers from a text file:

```python
# Example: tickers.txt
# One ticker per line, supports comments
AAPL
MSFT
# European stocks
ASML.AS
VOW3.DE
7203.T  # Toyota

class TickerListReader:
    """Read ticker lists from text files."""
    
    @staticmethod
    def read(filepath: Path) -> List[str]:
        """
        Read tickers from file.
        - One ticker per line
        - Lines starting with # are comments
        - Empty lines ignored
        - Whitespace trimmed
        """
        tickers = []
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    # Remove inline comments
                    ticker = line.split("#")[0].strip()
                    if ticker:
                        tickers.append(ticker.upper())
        return tickers
```

### Ingestion Orchestration

```python
class IngestionService:
    """Main ingestion orchestrator."""
    
    def __init__(
        self,
        provider: FinancialDataProvider,
        fx_provider: FXRateProvider,
        db: SnapshotRepository,
        rate_limiter: RateLimiter,
    ):
        self.provider = provider
        self.fx_provider = fx_provider
        self.db = db
        self.rate_limiter = rate_limiter
    
    async def ingest_from_file(
        self,
        ticker_file: Path,
        historical: bool = True,
        force_refresh: bool = False,
    ) -> IngestionReport:
        """
        Ingest all tickers from a file.
        
        Args:
            ticker_file: Path to ticker list file
            historical: Fetch all historical statements (vs latest only)
            force_refresh: Re-fetch even if data exists
        """
        tickers = TickerListReader.read(ticker_file)
        return await self.ingest_tickers(tickers, historical, force_refresh)
    
    async def ingest_tickers(
        self,
        tickers: List[str],
        historical: bool = True,
        force_refresh: bool = False,
    ) -> IngestionReport:
        """Ingest a list of tickers."""
        report = IngestionReport()
        
        for ticker in tickers:
            await self.rate_limiter.wait()
            try:
                snapshots = await self._ingest_single(ticker, historical, force_refresh)
                report.add_success(ticker, len(snapshots))
            except Exception as e:
                report.add_failure(ticker, str(e))
        
        return report
```

---

## yfinance API Details

### Required API Calls Per Ticker

| Call | Data Retrieved | Rate Impact |
|------|----------------|-------------|
| `yf.Ticker(symbol)` | Initialize ticker object | 0 (local) |
| `.info` | Company metadata + current stats | 1 request |
| `.quarterly_income_stmt` | Quarterly income statements | 1 request |
| `.quarterly_balance_sheet` | Quarterly balance sheets | 1 request |
| `.quarterly_cash_flow` | Quarterly cash flows | 1 request |
| `.history(start, end)` | Price data for specific dates | 1 request |
| `.recommendations_summary` | Analyst recommendations | 1 request |
| `.insider_purchases` | Insider activity | 1 request |

**Total: ~7 API calls per ticker** (with metadata)

### yfinance Response Schemas

```python
# ticker.info returns dict with these key fields:
INFO_FIELDS = {
    # Identification
    "symbol": str,              # "AAPL"
    "shortName": str,           # "Apple Inc."
    "longName": str,            # "Apple Inc."
    "currency": str,            # "USD" - CRITICAL for conversion
    "exchange": str,            # "NMS"
    "quoteType": str,           # "EQUITY"
    "sector": str,              # "Technology"
    "industry": str,            # "Consumer Electronics"
    
    # Current financials (already in native currency)
    "marketCap": int,           # 2890000000000
    "totalRevenue": int,        # 383285000000
    "grossProfit": int,         # 170782000000
    "ebitda": int,              # 123795000000
    "totalDebt": int,           # 111088000000
    "totalCash": int,           # 61555000000
    "freeCashflow": int,        # 99584000000
    
    # Ratios (dimensionless - no conversion needed)
    "profitMargins": float,     # 0.25
    "returnOnAssets": float,    # 0.28
    "returnOnEquity": float,    # 1.60
    "debtToEquity": float,      # 176.32
    "currentRatio": float,      # 0.94
    "quickRatio": float,        # 0.85
    
    # Shares (count - no conversion needed)
    "sharesOutstanding": int,   # 15441900000
    "floatShares": int,         # 15425600000
    "sharesShort": int,         # 120080000
}

# quarterly_income_stmt returns DataFrame:
# - Columns: statement dates (e.g., "2024-09-30")
# - Rows: line items (e.g., "Total Revenue", "Gross Profit")
# - Values: in NATIVE CURRENCY (must convert)

INCOME_STMT_FIELDS = [
    "Total Revenue",
    "Gross Profit",
    "Operating Income",
    "Net Income",
    "EBITDA",
    "Basic EPS",
    "Diluted EPS",
]

# quarterly_balance_sheet returns DataFrame (same structure)
BALANCE_SHEET_FIELDS = [
    "Total Assets",
    "Total Debt",
    "Total Liabilities Net Minority Interest",
    "Stockholders Equity",
    "Working Capital",
    "Cash And Cash Equivalents",
]

# quarterly_cash_flow returns DataFrame (same structure)
CASHFLOW_FIELDS = [
    "Operating Cash Flow",
    "Free Cash Flow",
    "Capital Expenditure",
    "Issuance Of Debt",
    "Repayment Of Debt",
]
```

---

## Primary Data Source: yfinance

Based on `snp_data.py`, using the `yfinance` library:

```python
import yfinance as yf

ticker_data = yf.Ticker("AAPL")

# Data sources
info = ticker_data.info                      # Company metadata + current stats
qis = ticker_data.quarterly_income_stmt      # Quarterly income statements
qbs = ticker_data.quarterly_balance_sheet    # Quarterly balance sheets
qcf = ticker_data.quarterly_cash_flow        # Quarterly cash flows
hist = ticker_data.history(period="5y")      # Price history (for t-1/t0/t+1)
rec = ticker_data.recommendations_summary    # Analyst recommendations
insider = ticker_data.insider_purchases      # Insider activity
```

---

## Inputs

- `tickers: List[str]` - List of ticker symbols to ingest
- `ticker_file: Path` - Path to text file with ticker list (alternative)
- `snapshot_dates` policy:
  - Either explicit statement dates requested
  - Or "fetch all available statement periods for ticker"
- Provider config (rate limits, retries)
- Price context config: `{mode: "t0" | "t-1,t0,t+1"}`

---

## Outputs

- **DB Rows**:
  - `assets` (metadata)
  - `financial_snapshots` (canonical row with all fields)
  - `raw_payloads` (audit)
- **Ingestion Report**:
  - Counts, missing fields, date alignment notes

---

## Index Constituents

### Responsibility

Ingestion manages index constituency because it is a **data acquisition** concern:
- Constituents change over time (companies added/removed from indices)
- Membership snapshots must be stored historically for reproducible index calculations

### Methods

```python
class IngestionService:
    async def ingest_index_memberships(
        self,
        index_id: str,
        as_of_time: Optional[datetime] = None,
    ) -> IndexMembershipReport:
        """
        Fetch and store current members of an index.
        
        Args:
            index_id: Index identifier (e.g., "SP500", "NASDAQ100")
            as_of_time: Optional historical date (default: now)
        
        Returns:
            Report with ticker count and any errors
        """
        tickers = await self.provider.get_index_constituents(index_id)
        # Store to index_memberships table
        await self.db.upsert_index_memberships(index_id, as_of_time, tickers)
        return IndexMembershipReport(index_id=index_id, ticker_count=len(tickers))
    
    def resolve_universe_from_index(
        self,
        index_id: str,
        as_of_time: Optional[datetime] = None,
    ) -> List[str]:
        """
        Get tickers for an index from stored memberships.
        Used to drive downstream ingestion.
        """
        return self.db.get_index_members(index_id, as_of_time)
```

### Constraint

> [!IMPORTANT]
> This ingestion module still does **NOT** ingest time-series prices.
> It only ingests minimal price anchors per financial snapshot (t-1, t0, t+1).

---

## Dependencies

### External Packages
- `yfinance` - Yahoo Finance API wrapper
- `pandas` - Data manipulation
- `httpx` - For backup HTTP requests

### Internal Modules
- `src/providers/` - Abstract provider interface (for future swappability)
- `src/db/` - Database layer for persistence
- `src/normalize/` - Normalization and currency conversion

---

## Folder Structure

```
src/ingestion/
  __init__.py
  service.py              # Main ingestion orchestrator
  snapshot_builder.py     # Builds FinancialSnapshot from raw data
  ticker_resolver.py      # Resolves tickers to metadata
  yfinance_fetcher.py     # yfinance-specific fetching logic
  index_constituent.py    # Index membership ingestion
  config.py               # Ingestion configuration
  report.py               # Ingestion report generation
```

---

## Data Fetching (from snp_data.py)

```python
from typing import Optional
import pandas as pd
import yfinance as yf

def pass_except(func):
    """Decorator to catch exceptions and return empty Series."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            return pd.Series()
    return wrapper

@pass_except
def get_info(ticker_data) -> pd.Series:
    """Get company info and current stats."""
    info = ticker_data.info
    info.pop("companyOfficers", None)  # Remove nested data
    return pd.Series(info)

@pass_except
def get_quarterly_income(ticker_data) -> pd.Series:
    """Get averaged quarterly income statement (last 4 quarters)."""
    return ticker_data.quarterly_income_stmt.T.head(4).mean(skipna=True)

@pass_except
def get_quarterly_cashflow(ticker_data) -> pd.Series:
    """Get averaged quarterly cash flow (last 4 quarters)."""
    return ticker_data.quarterly_cash_flow.T.head(4).mean(skipna=True)

@pass_except
def get_balance_sheet(ticker_data) -> pd.Series:
    """Get latest balance sheet snapshot."""
    return ticker_data.quarterly_balance_sheet.T.head(1).mean(skipna=True)

@pass_except  
def get_recommendations(ticker_data) -> pd.Series:
    """Get analyst recommendation summary."""
    return ticker_data.recommendations_summary.sum().add_prefix("Analyst ")

@pass_except
def get_insider_purchases(ticker_data) -> pd.Series:
    """Get insider purchase activity."""
    return ticker_data.insider_purchases.set_index(
        "Insider Purchases Last 6m"
    ).drop("Trans", axis=1).squeeze().add_prefix("Insider ")

def process_ticker(ticker: str) -> Optional[pd.DataFrame]:
    """
    Process a single ticker and return combined snapshot data.
    """
    ticker_data = yf.Ticker(ticker)
    
    info = get_info(ticker_data)
    qis = get_quarterly_income(ticker_data)
    qcf = get_quarterly_cashflow(ticker_data)
    bs = get_balance_sheet(ticker_data)
    rec = get_recommendations(ticker_data)
    insider = get_insider_purchases(ticker_data)
    
    try:
        combined = pd.concat([info, qis, qcf, bs, rec, insider]).drop_duplicates()
        return combined.to_frame().T
    except:
        return None
```

---

## Provider Abstraction (for swappability)

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import date

class FinancialDataProvider(ABC):
    """
    Abstract provider interface.
    Allows swapping yfinance for other providers.
    """
    
    @abstractmethod
    async def get_company_info(self, ticker: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def get_financial_statements(
        self, ticker: str, frequency: str = "quarterly"
    ) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def get_price_window(
        self, ticker: str, target_date: date
    ) -> Dict[str, Optional[float]]:
        pass

class YFinanceProvider(FinancialDataProvider):
    """yfinance implementation of the provider interface."""
    
    def __init__(self, rate_limit_delay: float = 0.25):
        self.rate_limit_delay = rate_limit_delay
    
    async def get_company_info(self, ticker: str) -> Dict[str, Any]:
        ticker_data = yf.Ticker(ticker)
        return get_info(ticker_data).to_dict()
    
    # ... other methods
```

---

## Statement Date Handling

### ✅ RESOLVED: Both Filing and Release Dates

Store **both** dates when available:
- `filing_date` - SEC filing date
- `release_date` - Earnings announcement date

```python
def extract_statement_dates(ticker_data) -> dict:
    """
    Extract filing and release dates from yfinance data.
    """
    # yfinance provides statement dates in the index
    income_dates = ticker_data.quarterly_income_stmt.columns
    
    dates = []
    for col in income_dates:
        dates.append({
            "period_end_date": col.date() if hasattr(col, 'date') else col,
            "filing_date": None,  # Not directly available in yfinance
            "release_date": None,  # Would need earnings calendar
        })
    
    return dates
```

> 📌 **TODO**: Integrate earnings calendar API for accurate release dates

---

## Rate Limiting

```python
import time
import asyncio

class RateLimiter:
    """
    Rate limiter for yfinance requests.
    Based on snp_data.py: 0.25s delay between requests.
    """
    
    def __init__(self, delay: float = 0.25):
        self.delay = delay
        self.last_request = 0.0
    
    async def wait(self):
        elapsed = time.time() - self.last_request
        if elapsed < self.delay:
            await asyncio.sleep(self.delay - elapsed)
        self.last_request = time.time()
```

---

## Design Decisions

### ✅ RESOLVED: Provider Abstraction

**Decision**: Use yfinance as primary provider but wrap behind abstract interface for swappability.

```python
# Easy to swap providers
provider: FinancialDataProvider = YFinanceProvider()
# Later: provider = AlphaVantageProvider()
```

### ⚠️ NEEDS REVIEW: Historical Snapshots

How to handle historical data vs. current snapshot?

**Options**:
1. **Current only** - Just fetch latest statement
2. **Full history** - Fetch all available historical statements
3. **Configurable** - Parameter for how much history to fetch

### 📌 TODO: Earnings Calendar Integration

Integrate with earnings calendar API to get accurate `release_date` values.

### 📌 TODO: Error Recovery

Handle partial failures gracefully:
- Log failed tickers
- Retry with exponential backoff
- Continue with successful tickers

---

## Constraints

- ⚡ **yfinance** is primary provider (abstracted for swappability)
- ⚡ Rate limit: 0.25s delay between requests minimum
- ⚡ Must store raw payloads for audit trail
- ⚡ Must not create daily OHLCV tables
- ⚡ Must handle missing/NULL fields gracefully

---

## Integration Tests

### Test Scope

Integration tests verify the full ingestion pipeline with real (or mocked) external services.

### Test Cases

```python
# tests/integration/test_ingestion.py

class TestIngestionIntegration:
    """Integration tests for data ingestion pipeline."""
    
    async def test_single_ticker_ingestion(self, test_db, mock_yfinance):
        """
        Test ingesting a single ticker end-to-end.
        
        Verifies:
        - Ticker metadata is fetched and stored
        - Financial statements are parsed correctly
        - Currency conversion is attempted (or fallback used)
        - Snapshot row is persisted to database
        - Raw payload is stored for audit
        """
        service = IngestionService(...)
        report = await service.ingest_tickers(["AAPL"])
        
        assert report.success_count == 1
        assert report.failure_count == 0
        
        # Verify DB state
        snapshots = await test_db.get_snapshots("AAPL")
        assert len(snapshots) >= 1
        assert snapshots[0].total_revenue is not None
    
    async def test_ticker_list_file_ingestion(self, test_db, tmp_path):
        """
        Test ingesting tickers from a file.
        
        Verifies:
        - File is parsed correctly
        - Comments and empty lines are ignored
        - All valid tickers are processed
        """
        ticker_file = tmp_path / "tickers.txt"
        ticker_file.write_text("AAPL\n# Comment\nMSFT\n")
        
        service = IngestionService(...)
        report = await service.ingest_from_file(ticker_file)
        
        assert len(report.attempted) == 2
    
    async def test_currency_fallback(self, test_db, mock_yfinance, mock_fx_failing):
        """
        Test currency fallback when FX rate unavailable.
        
        Verifies:
        - When FX rate fails, original currency is kept
        - fx_rate_to_usd is null
        - stored_currency matches original_currency
        """
        service = IngestionService(fx_provider=mock_fx_failing)
        report = await service.ingest_tickers(["ASML.AS"])  # EUR stock
        
        snapshot = await test_db.get_latest_snapshot("ASML.AS")
        assert snapshot.stored_currency == "EUR"
        assert snapshot.fx_rate_to_usd is None
    
    async def test_rate_limiting(self, test_db, mock_yfinance):
        """
        Test rate limiting enforces minimum delay.
        
        Verifies:
        - Multiple ticker ingestion respects rate limit
        - Total time >= (n_tickers - 1) * rate_limit_delay
        """
        import time
        
        start = time.time()
        report = await service.ingest_tickers(["AAPL", "MSFT", "GOOGL"])
        elapsed = time.time() - start
        
        # At least 0.5s for 3 tickers with 0.25s delay
        assert elapsed >= 0.5
    
    async def test_error_recovery(self, test_db, mock_yfinance_partial_fail):
        """
        Test partial failure handling.
        
        Verifies:
        - Failed tickers are logged
        - Successful tickers are still processed
        - Report includes both success and failure counts
        """
        service = IngestionService(...)
        report = await service.ingest_tickers(["AAPL", "INVALID", "MSFT"])
        
        assert report.success_count == 2
        assert report.failure_count == 1
        assert "INVALID" in report.failures

### Test Fixtures

```python
# tests/conftest.py

@pytest.fixture
def mock_yfinance(mocker):
    """Mock yfinance responses with realistic test data."""
    mock_ticker = mocker.MagicMock()
    mock_ticker.info = {
        "symbol": "AAPL",
        "currency": "USD",
        "marketCap": 2850000000000,
        "totalRevenue": 383285000000,
        # ... other fields
    }
    mock_ticker.quarterly_income_stmt = create_test_income_stmt()
    # ... other mocks
    
    mocker.patch("yfinance.Ticker", return_value=mock_ticker)
    return mock_ticker

@pytest.fixture
def mock_fx_failing(mocker):
    """Mock FX provider that always fails."""
    mock = mocker.MagicMock(spec=FXRateProvider)
    mock.get_rate.side_effect = FXRateUnavailable("API down")
    return mock
```

### Running Tests

```bash
# Run all ingestion integration tests
pytest tests/integration/test_ingestion.py -v

# Run with real yfinance (slow, requires network)
pytest tests/integration/test_ingestion.py -v --real-api

# Run with coverage
pytest tests/integration/test_ingestion.py --cov=src/ingestion
```
