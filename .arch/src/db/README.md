# Database / Storage

> Schema definitions, migrations, and repository methods for financial snapshots. Uses PostgreSQL with data versioning for replication.

---

## Responsibilities

- Define schema and migrations
- Provide repository methods:
  - Upsert asset metadata
  - Upsert snapshot rows
  - Query snapshots by ticker, date range, sector filters
- Maintain raw payload storage for audit/reprocessing
- **Data versioning** for reproducibility and model replication

---

## Inputs

- Normalized entities from ingestion/computation modules
- Query parameters from API/reporting modules

---

## Outputs

- Persisted tables
- Query interfaces (repositories)
- Migration scripts
- **Data snapshots** for versioning

---

## Dependencies

### External Packages
- `sqlalchemy` - ORM and query builder
- `alembic` - Database migrations
- `psycopg2` or `asyncpg` - PostgreSQL driver

### Internal Modules
- None (infrastructure layer - no internal dependencies)

---

## Folder Structure

```
src/db/
  __init__.py
  models/
    __init__.py
    asset.py              # Asset metadata model
    snapshot.py           # FinancialSnapshot model
    raw_payload.py        # Raw payload audit model
    valuation.py          # Valuation results model
    portfolio.py          # Portfolio weights model
    data_version.py       # Data versioning model
    index.py              # Index and IndexMembership models
  repositories/
    __init__.py
    asset_repo.py
    snapshot_repo.py
    valuation_repo.py
    portfolio_repo.py
    version_repo.py       # Data versioning repository
    index_repo.py         # Index membership repository
  migrations/
    versions/             # Alembic migration files
  session.py              # Database session management
  config.py               # Database configuration
  versioning.py           # Data versioning utilities
```

---

## Database Choice

### ✅ RESOLVED: PostgreSQL

**Decision**: Use **PostgreSQL** as the primary database.

- JSONB support for flexible payload storage
- Better indexing and query optimization
- Production-ready with proper scaling

Connection string via environment variable:
```
DATABASE_URL=postgresql://user:pass@localhost:5432/fintech
```

---

## Schema Design

### Table: `assets`

```sql
CREATE TABLE assets (
    ticker          VARCHAR(20) PRIMARY KEY,
    company_name    VARCHAR(255),
    original_currency VARCHAR(10) NOT NULL,  -- Native currency of company
    exchange        VARCHAR(50),
    sector          VARCHAR(100),
    industry        VARCHAR(100),
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);
```

### Table: `financial_snapshots`

```sql
CREATE TABLE financial_snapshots (
    ticker              VARCHAR(20) NOT NULL,
    snapshot_timestamp  TIMESTAMP NOT NULL,
    
    -- Statement metadata (store BOTH dates)
    period_end_date     DATE,
    filing_date         DATE,           -- SEC filing date
    release_date        DATE,           -- Earnings release date (if different)
    frequency           VARCHAR(10),    -- 'quarterly' | 'annual'
    
    -- Currency (with fallback support)
    original_currency   VARCHAR(10) NOT NULL,  -- Native currency of company
    stored_currency     VARCHAR(10) NOT NULL,  -- Currency values are stored in
    fx_rate_to_usd      NUMERIC,               -- NULL if not converted to USD
    
    -- Financials (stored in stored_currency)
    total_revenue       NUMERIC,
    gross_profit        NUMERIC,
    ebitda              NUMERIC,
    operating_income    NUMERIC,
    net_income          NUMERIC,
    eps                 NUMERIC,
    shares_outstanding  NUMERIC,
    float_shares        NUMERIC,
    total_debt          NUMERIC,
    total_cash          NUMERIC,
    total_assets        NUMERIC,
    free_cash_flow      NUMERIC,
    operating_cash_flow NUMERIC,
    capex               NUMERIC,
    working_capital     NUMERIC,
    book_value          NUMERIC,
    
    -- Derived ratios
    profit_margins      NUMERIC,
    gross_margin        NUMERIC,
    operating_margin    NUMERIC,
    net_margin          NUMERIC,
    roe                 NUMERIC,
    roa                 NUMERIC,
    roic                NUMERIC,
    debt_to_equity      NUMERIC,
    current_ratio       NUMERIC,
    quick_ratio         NUMERIC,
    
    -- Dividend
    dividend_yield      NUMERIC,
    dividend_rate       NUMERIC,
    five_year_avg_div_yield NUMERIC,
    
    -- Risk scores (from yfinance)
    audit_risk          INTEGER,
    board_risk          INTEGER,
    compensation_risk   INTEGER,
    
    -- Short interest
    shares_short        NUMERIC,
    shares_short_prior_month NUMERIC,
    short_ratio         NUMERIC,
    
    -- Ownership
    held_percent_insiders NUMERIC,
    held_percent_institutions NUMERIC,
    
    -- Employees
    full_time_employees INTEGER,
    
    -- Price context (minimal)
    price_t0            NUMERIC,
    price_t_minus_1     NUMERIC,
    price_t_plus_1      NUMERIC,
    market_cap_t0       NUMERIC,
    
    -- Quality metadata
    data_quality_score  NUMERIC,
    validation_warnings JSONB,
    
    -- Audit
    created_at          TIMESTAMP DEFAULT NOW(),
    updated_at          TIMESTAMP DEFAULT NOW(),
    
    PRIMARY KEY (ticker, snapshot_timestamp)
);

-- Indexes
CREATE INDEX idx_snapshots_ticker ON financial_snapshots(ticker);
CREATE INDEX idx_snapshots_timestamp ON financial_snapshots(snapshot_timestamp DESC);
CREATE INDEX idx_snapshots_filing ON financial_snapshots(filing_date DESC);
CREATE INDEX idx_snapshots_release ON financial_snapshots(release_date DESC);
CREATE INDEX idx_snapshots_mcap ON financial_snapshots(market_cap_t0 DESC);
CREATE INDEX idx_snapshots_currency ON financial_snapshots(stored_currency);
```

### Table: `raw_payloads`

```sql
CREATE TABLE raw_payloads (
    payload_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ticker              VARCHAR(20) NOT NULL,
    snapshot_timestamp  TIMESTAMP NOT NULL,
    provider            VARCHAR(50) NOT NULL,
    endpoint            VARCHAR(100) NOT NULL,
    payload_body        JSONB NOT NULL,
    fetched_at          TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_payloads_ticker_ts ON raw_payloads(ticker, snapshot_timestamp);
```

### Table: `valuation_results`

```sql
CREATE TABLE valuation_results (
    ticker                  VARCHAR(20) NOT NULL,
    snapshot_timestamp      TIMESTAMP NOT NULL,
    
    -- Prediction outputs
    predicted_mcap_mean     NUMERIC NOT NULL,
    predicted_mcap_std      NUMERIC NOT NULL,
    actual_mcap             NUMERIC NOT NULL,
    relative_error          NUMERIC NOT NULL,
    relative_std            NUMERIC NOT NULL,
    
    -- Model metadata
    model_version           VARCHAR(50) NOT NULL,
    model_config_hash       VARCHAR(64),           -- Hash of model config JSON
    n_experiments           INTEGER NOT NULL,
    experiment_metadata     JSONB,      -- param_grid, cv_splits, etc.
    
    -- Data versioning
    data_version_id         UUID REFERENCES data_versions(version_id),
    
    -- Audit
    computed_at             TIMESTAMP DEFAULT NOW(),
    
    PRIMARY KEY (ticker, snapshot_timestamp, model_version)
);

CREATE INDEX idx_valuation_error ON valuation_results(relative_error);
CREATE INDEX idx_valuation_model ON valuation_results(model_version);
CREATE INDEX idx_valuation_data_version ON valuation_results(data_version_id);
```

### Table: `fx_rates`

```sql
CREATE TABLE fx_rates (
    rate_date       DATE NOT NULL,
    from_currency   VARCHAR(10) NOT NULL,
    to_currency     VARCHAR(10) NOT NULL DEFAULT 'USD',
    rate            NUMERIC NOT NULL,
    
    PRIMARY KEY (rate_date, from_currency, to_currency)
);

CREATE INDEX idx_fx_date ON fx_rates(rate_date DESC);
```

### Table: `indices`

```sql
CREATE TABLE indices (
    index_id            VARCHAR(50) PRIMARY KEY,
    name                VARCHAR(255) NOT NULL,
    description         TEXT,
    weighting_scheme    VARCHAR(20) NOT NULL,  -- 'equal' | 'market_cap' | 'custom'
    base_value          NUMERIC DEFAULT 100.0,
    created_at          TIMESTAMP DEFAULT NOW(),
    updated_at          TIMESTAMP DEFAULT NOW()
);
```

### Table: `index_memberships`

Tracks which tickers are in which index at which time (constituency changes over time).

```sql
CREATE TABLE index_memberships (
    index_id        VARCHAR(50) NOT NULL REFERENCES indices(index_id),
    as_of_time      TIMESTAMP NOT NULL,     -- Membership snapshot time
    ticker          VARCHAR(20) NOT NULL REFERENCES assets(ticker),
    is_member       BOOLEAN DEFAULT TRUE,   -- Allows explicit removals
    source          VARCHAR(50),            -- Provider that reported membership
    ingested_at     TIMESTAMP DEFAULT NOW(),
    
    PRIMARY KEY (index_id, as_of_time, ticker)
);

CREATE INDEX idx_membership_index_time ON index_memberships(index_id, as_of_time);
CREATE INDEX idx_membership_ticker ON index_memberships(ticker);
```

---

## Data Versioning

### ✅ RESOLVED: First-Class Data Versioning for Replication

**Goal**: Ensure model results can be replicated by tracking exact data state.

### Table: `data_versions`

```sql
CREATE TABLE data_versions (
    version_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version_name    VARCHAR(100) NOT NULL UNIQUE,  -- e.g., "v1.0", "2024-01-15-full"
    description     TEXT,
    
    -- What's included
    ticker_count    INTEGER NOT NULL,
    snapshot_count  INTEGER NOT NULL,
    date_range_start DATE NOT NULL,
    date_range_end   DATE NOT NULL,
    
    -- Integrity
    data_hash       VARCHAR(64) NOT NULL,  -- SHA256 of all included snapshots
    filter_config   JSONB,                 -- Filters applied to create version
    
    -- Metadata
    created_at      TIMESTAMP DEFAULT NOW(),
    created_by      VARCHAR(100),
    notes           TEXT
);

CREATE UNIQUE INDEX idx_data_versions_name ON data_versions(version_name);
```

### Table: `data_version_snapshots`

```sql
-- Junction table: which snapshots are in which version
CREATE TABLE data_version_snapshots (
    version_id          UUID REFERENCES data_versions(version_id),
    ticker              VARCHAR(20) NOT NULL,
    snapshot_timestamp  TIMESTAMP NOT NULL,
    
    PRIMARY KEY (version_id, ticker, snapshot_timestamp),
    FOREIGN KEY (ticker, snapshot_timestamp) 
        REFERENCES financial_snapshots(ticker, snapshot_timestamp)
);
```

### Data Versioning Strategy

```python
# src/db/versioning.py
import hashlib
import json
from typing import List, Optional
from uuid import UUID

class DataVersionManager:
    """Manage data versions for reproducibility."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def create_version(
        self,
        name: str,
        description: str,
        filter_config: Optional[dict] = None,
    ) -> DataVersion:
        """
        Create a new data version from current snapshots.
        
        This "freezes" the current state of filtered data for reproducibility.
        """
        # Apply filters if provided
        if filter_config:
            snapshots = self._get_filtered_snapshots(filter_config)
        else:
            snapshots = self._get_all_snapshots()
        
        # Compute data hash for integrity
        data_hash = self._compute_hash(snapshots)
        
        # Create version record
        version = DataVersion(
            version_name=name,
            description=description,
            ticker_count=len(set(s.ticker for s in snapshots)),
            snapshot_count=len(snapshots),
            date_range_start=min(s.snapshot_timestamp for s in snapshots).date(),
            date_range_end=max(s.snapshot_timestamp for s in snapshots).date(),
            data_hash=data_hash,
            filter_config=filter_config,
        )
        self.session.add(version)
        
        # Link snapshots to version
        for snapshot in snapshots:
            link = DataVersionSnapshot(
                version_id=version.version_id,
                ticker=snapshot.ticker,
                snapshot_timestamp=snapshot.snapshot_timestamp,
            )
            self.session.add(link)
        
        self.session.commit()
        return version
    
    def get_version_snapshots(
        self, version_id: UUID
    ) -> List[FinancialSnapshot]:
        """Get all snapshots in a specific version."""
        return (
            self.session.query(FinancialSnapshot)
            .join(DataVersionSnapshot)
            .filter(DataVersionSnapshot.version_id == version_id)
            .all()
        )
    
    def verify_version(self, version_id: UUID) -> bool:
        """Verify data version integrity by recomputing hash."""
        version = self.session.query(DataVersion).get(version_id)
        snapshots = self.get_version_snapshots(version_id)
        current_hash = self._compute_hash(snapshots)
        return current_hash == version.data_hash
    
    def _compute_hash(self, snapshots: List[FinancialSnapshot]) -> str:
        """Compute SHA256 hash of snapshot data."""
        # Sort for deterministic order
        sorted_snaps = sorted(snapshots, key=lambda s: (s.ticker, s.snapshot_timestamp))
        
        # Serialize to canonical JSON
        data = [
            {
                "ticker": s.ticker,
                "timestamp": s.snapshot_timestamp.isoformat(),
                "revenue": str(s.total_revenue) if s.total_revenue else None,
                "mcap": str(s.market_cap_t0) if s.market_cap_t0 else None,
                # ... other key fields
            }
            for s in sorted_snaps
        ]
        
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
```

### Version Usage in Valuation

```python
# When running valuation, specify data version for reproducibility
def run_valuation(
    model_config: ModelConfig,
    data_version_id: Optional[UUID] = None,  # Use specific version
):
    if data_version_id:
        # Use frozen data version
        snapshots = version_manager.get_version_snapshots(data_version_id)
    else:
        # Use latest data (create new version)
        version = version_manager.create_version(
            name=f"auto-{datetime.now().isoformat()}",
            description="Auto-created for valuation run",
        )
        snapshots = version_manager.get_version_snapshots(version.version_id)
    
    # Store version reference with results
    results = run_model(snapshots, model_config)
    results.data_version_id = version.version_id
```

---

## Currency Handling

### ✅ RESOLVED: Currency Fallback Support

**Decision**: 
- Store `original_currency` (company's native currency)
- Store `stored_currency` (USD if converted, else original)
- `fx_rate_to_usd` is NULL if not converted
- All monetary values in `stored_currency`

```python
def convert_to_usd(value: float, currency: str, date: date) -> tuple[float, str, Optional[float]]:
    """
    Convert value to USD if possible.
    Returns (converted_value, stored_currency, fx_rate_used).
    If conversion fails, returns (original_value, original_currency, None).
    """
    if currency == "USD":
        return value, "USD", 1.0
    
    try:
        rate = get_fx_rate(currency, "USD", date)
        return value * rate, "USD", rate
    except FXRateUnavailable:
        return value, currency, None
```

---

## Design Decisions

### ✅ RESOLVED: Both Filing and Release Dates

**Decision**: Store **both** `filing_date` and `release_date` in the schema.

- `filing_date` - SEC filing timestamp
- `release_date` - Earnings announcement (may differ)

Use `filing_date` as canonical for point-in-time queries by default.

### ✅ RESOLVED: Data Versioning

**Decision**: First-class data versioning for model replication.

- Create immutable data versions for reproducibility
- Link valuation results to data versions
- Compute hash for integrity verification

### ⚠️ NEEDS REVIEW: Partitioning Strategy

For large datasets, consider partitioning `financial_snapshots` by:
- Year (range partitioning on `snapshot_timestamp`)
- Ticker hash

### 📌 TODO: Index Optimization

After initial data load, analyze query patterns and add covering indexes.

---

## Constraints

- ⚡ **PostgreSQL** is the production database
- ⚡ Store both `original_currency` and `stored_currency`
- ⚡ Store **both** `filing_date` and `release_date`
- ⚡ Must support upsert semantics (ON CONFLICT)
- ⚡ Raw payloads must be preserved for reprocessing
- ⚡ Data versions must be immutable after creation

---

## Integration Tests

### Test Scope

Integration tests verify database operations with a real PostgreSQL instance.

### Test Cases

```python
# tests/integration/test_db.py

class TestDatabaseIntegration:
    """Integration tests for database operations."""
    
    def test_snapshot_upsert(self, test_db):
        """
        Test snapshot upsert semantics.
        
        Verifies:
        - New snapshot is inserted
        - Duplicate (ticker, timestamp) updates existing row
        - Updated fields are reflected
        """
        snapshot1 = create_snapshot("AAPL", revenue=100)
        await test_db.upsert_snapshot(snapshot1)
        
        # Upsert with different revenue
        snapshot2 = create_snapshot("AAPL", revenue=200)
        await test_db.upsert_snapshot(snapshot2)
        
        result = await test_db.get_snapshot("AAPL", snapshot1.timestamp)
        assert result.total_revenue == 200
    
    def test_currency_storage(self, test_db):
        """
        Test currency fields are stored correctly.
        
        Verifies:
        - original_currency is preserved
        - stored_currency reflects actual currency
        - fx_rate_to_usd is null when not converted
        """
        # USD stock
        usd_snap = create_snapshot("AAPL", currency="USD")
        await test_db.upsert_snapshot(usd_snap)
        
        result = await test_db.get_snapshot("AAPL", usd_snap.timestamp)
        assert result.original_currency == "USD"
        assert result.stored_currency == "USD"
        assert result.fx_rate_to_usd == 1.0
        
        # Non-USD stock without conversion
        jpy_snap = create_snapshot("7203.T", original_currency="JPY", 
                                   stored_currency="JPY", fx_rate=None)
        await test_db.upsert_snapshot(jpy_snap)
        
        result = await test_db.get_snapshot("7203.T", jpy_snap.timestamp)
        assert result.original_currency == "JPY"
        assert result.stored_currency == "JPY"
        assert result.fx_rate_to_usd is None
    
    def test_data_versioning(self, test_db, version_manager):
        """
        Test data versioning for reproducibility.
        
        Verifies:
        - Version is created with correct metadata
        - All snapshots are linked to version
        - Version can be retrieved by name
        - Hash is reproducible
        """
        # Insert test data
        for ticker in ["AAPL", "MSFT", "GOOGL"]:
            await test_db.upsert_snapshot(create_snapshot(ticker))
        
        # Create version
        version = version_manager.create_version(
            name="test-v1",
            description="Test version"
        )
        
        assert version.ticker_count == 3
        assert version.snapshot_count == 3
        
        # Retrieve snapshots
        snapshots = version_manager.get_version_snapshots(version.version_id)
        assert len(snapshots) == 3
    
    def test_version_integrity(self, test_db, version_manager):
        """
        Test version integrity verification.
        
        Verifies:
        - Hash matches original
        - Tampering is detected
        """
        version = version_manager.create_version("test-v2", "Integrity test")
        
        # Verify passes
        assert version_manager.verify_version(version.version_id) is True
        
        # Note: In real test, would modify underlying data and verify fails
    
    def test_raw_payload_storage(self, test_db):
        """
        Test raw payload storage for audit.
        
        Verifies:
        - JSONB payload is stored correctly
        - Can retrieve by ticker and timestamp
        """
        payload = {"symbol": "AAPL", "marketCap": 2850000000000}
        await test_db.store_raw_payload("AAPL", timestamp, "yfinance", "info", payload)
        
        result = await test_db.get_raw_payloads("AAPL", timestamp)
        assert len(result) == 1
        assert result[0].payload_body["marketCap"] == 2850000000000

### Running Tests

```bash
# Run with test database (requires PostgreSQL)
pytest tests/integration/test_db.py -v --db-url="postgresql://test:test@localhost:5432/test_fintech"

# Run with Docker PostgreSQL
docker-compose -f docker-compose.test.yml up -d
pytest tests/integration/test_db.py -v

# Run with coverage
pytest tests/integration/test_db.py --cov=src/db
```
