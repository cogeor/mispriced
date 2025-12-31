# Financial Normalization & Quality

> Converts provider-specific statement schemas into canonical fields with currency conversion and quality scoring.

---

## Responsibilities

- Convert provider-specific statement schemas into canonical fields
- **Currency conversion**: Attempt USD conversion; fallback to local currency if FX unavailable
- **Always preserve original currency** field, even for USD-native stocks
- Unit normalization (thousands/millions)
- Point-in-time correctness at snapshot level:
  - Store both `filing_date` and `release_date`
  - Ensure snapshot timestamp corresponds to "known at" moment
- Data quality scoring:
  - Missing critical fields, inconsistent totals, etc.
- Compute derived ratios (margins, ROE/ROIC)

---

## Currency Handling

### ✅ RESOLVED: Attempt USD with Fallback

**Strategy**:
1. **Attempt** to convert all monetary values to USD using day's FX rate
2. **Fallback**: If FX rate unavailable, keep original currency
3. **Always store** both `original_currency` and `stored_currency` fields
4. Store numeric values in `stored_currency` (either USD or original)

### Currency Fields

| Field | Description | Example |
|-------|-------------|---------|
| `original_currency` | Native currency of the company | `"GBP"` |
| `stored_currency` | Currency values are stored in | `"USD"` or `"GBP"` |
| `fx_rate_to_usd` | Conversion rate (null if not converted) | `1.27` or `null` |

### Why Both Fields?

Even when conversion succeeds:
- `original_currency = "GBP"` (company reports in GBP)
- `stored_currency = "USD"` (we converted it)
- `fx_rate_to_usd = 1.27` (rate used)

When conversion fails:
- `original_currency = "GBP"`
- `stored_currency = "GBP"` (kept as-is)
- `fx_rate_to_usd = null` (not converted)

---

## Inputs

- Raw statement payloads (from ingestion/providers)
- Asset metadata (for native currency, sector context)
- Price context payloads (t-1, t0, t+1)
- FX rates (from provider or cached)

---

## Outputs

- Canonical `FinancialSnapshot` object (pydantic model)
- `original_currency: str` - Native currency of the company
- `stored_currency: str` - Currency values are actually in
- `fx_rate_to_usd: Optional[float]` - Conversion rate used (null if not converted)
- `data_quality_score: float` (0.0 - 1.0)
- `validation_warnings: List[str]`

---

## Dependencies

### External Packages
- `pydantic` - Data validation and models
- `pandas` (optional) - Data manipulation utilities
- `httpx` - For FX rate API calls

### Internal Modules
- `src/db/models/` - Database model definitions
- `src/providers/` - FX rate provider

---

## Folder Structure

```
src/normalize/
  __init__.py
  service.py              # Main normalization orchestrator
  schema_mapper.py        # Provider → canonical field mapping
  currency.py             # Currency conversion utilities
  unit_normalizer.py      # Unit conversion (millions → raw)
  ratio_calculator.py     # Derived ratio computation
  quality_scorer.py       # Data quality assessment
  validators.py           # Consistency validators
  models.py               # Pydantic models for canonical data
```

---

## Currency Conversion

### ✅ RESOLVED: USD Conversion with Fallback

**Decision**: 
- Attempt to convert all monetary values to USD at ingestion
- Use conversion rate from the filing/release date
- If rate unavailable, keep original currency
- Store both original currency and conversion info for auditability

```python
from datetime import date
from typing import Optional

class CurrencyConverter:
    """
    Convert monetary values to USD using historical FX rates.
    Falls back to original currency if rate unavailable.
    """
    
    def __init__(self, fx_provider: FXRateProvider):
        self.fx_provider = fx_provider
    
    def convert_to_usd(
        self, 
        value: float, 
        from_currency: str, 
        as_of_date: date
    ) -> tuple[float, str, Optional[float]]:
        """
        Attempt to convert value to USD with fallback.
        
        Returns:
            (converted_value, stored_currency, fx_rate_to_usd)
            
        If conversion fails:
            (original_value, original_currency, None)
        """
        if from_currency == "USD":
            return value, "USD", 1.0
        
        try:
            rate = self.fx_provider.get_rate(from_currency, "USD", as_of_date)
            return value * rate, "USD", rate
        except FXRateUnavailable:
            # Fallback: keep original currency
            return value, from_currency, None
    
    def convert_snapshot(
        self, 
        snapshot: RawSnapshot, 
        as_of_date: date
    ) -> CanonicalSnapshot:
        """
        Convert all monetary fields in a snapshot.
        Uses fallback if any conversion fails.
        """
        original_currency = snapshot.currency
        
        # Attempt conversion
        _, stored_currency, fx_rate = self.convert_to_usd(
            1.0, original_currency, as_of_date
        )
        
        # If we have a rate, convert all fields
        if fx_rate is not None:
            converted = {}
            for field in MONETARY_FIELDS:
                value = getattr(snapshot, field, None)
                if value is not None:
                    converted[field] = value * fx_rate
        else:
            # Keep original values
            converted = {f: getattr(snapshot, f, None) for f in MONETARY_FIELDS}
        
        return CanonicalSnapshot(
            **converted,
            original_currency=original_currency,
            stored_currency=stored_currency,
            fx_rate_to_usd=fx_rate,
        )

# Monetary fields that need conversion
MONETARY_FIELDS = [
    "total_revenue",
    "gross_profit",
    "ebitda",
    "operating_income",
    "net_income",
    "total_debt",
    "total_cash",
    "total_assets",
    "free_cash_flow",
    "operating_cash_flow",
    "capex",
    "working_capital",
    "book_value",
    "market_cap_t0",
    "price_t0",
    "price_t_minus_1",
    "price_t_plus_1",
]
```

---

## Field Mapping (yfinance → Canonical)

Based on `snp_data.py`:

| yfinance Field | Canonical Field | Source |
|----------------|-----------------|--------|
| `marketCap` | `market_cap_t0` | `info` |
| `totalRevenue` | `total_revenue` | `info` |
| `grossProfit` | `gross_profit` | `info` |
| `ebitda` | `ebitda` | `info` |
| `totalDebt` | `total_debt` | `info` |
| `totalCash` | `total_cash` | `info` |
| `freeCashflow` | `free_cash_flow` | `info` |
| `operatingCashFlow` | `operating_cash_flow` | `quarterly_cash_flow` |
| `profitMargins` | `profit_margins` | `info` |
| `returnOnAssets` | `roa` | `info` |
| `returnOnEquity` | `roe` | `info` |
| `debtToEquity` | `debt_to_equity` | `info` |
| `currentRatio` | `current_ratio` | `info` |
| `quickRatio` | `quick_ratio` | `info` |
| `dividendYield` | `dividend_yield` | `info` |
| `dividendRate` | `dividend_rate` | `info` |
| `auditRisk` | `audit_risk` | `info` |
| `boardRisk` | `board_risk` | `info` |
| `compensationRisk` | `compensation_risk` | `info` |
| `sharesOutstanding` | `shares_outstanding` | `info` |
| `floatShares` | `float_shares` | `info` |
| `sharesShort` | `shares_short` | `info` |
| `shortRatio` | `short_ratio` | `info` |
| `heldPercentInsiders` | `held_percent_insiders` | `info` |
| `heldPercentInstitutions` | `held_percent_institutions` | `info` |
| `fullTimeEmployees` | `full_time_employees` | `info` |

### Balance Sheet Fields (from quarterly_balance_sheet)

| yfinance Field | Canonical Field |
|----------------|-----------------|
| `Total Assets` | `total_assets` |
| `Total Debt` | `total_debt` |
| `Gross Profit` | `gross_profit` |

---

## Quality Scoring

```python
def compute_quality_score(snapshot: CanonicalSnapshot) -> float:
    """
    Compute data quality score (0.0 - 1.0).
    Higher score = more complete/reliable data.
    """
    score = 1.0
    warnings = []
    
    # Critical fields (high penalty)
    CRITICAL_FIELDS = ["total_revenue", "market_cap_t0", "shares_outstanding"]
    for field in CRITICAL_FIELDS:
        if getattr(snapshot, field) is None:
            score -= 0.2
            warnings.append(f"Missing critical field: {field}")
    
    # Important fields (medium penalty)
    IMPORTANT_FIELDS = ["ebitda", "free_cash_flow", "total_debt", "net_income"]
    for field in IMPORTANT_FIELDS:
        if getattr(snapshot, field) is None:
            score -= 0.1
            warnings.append(f"Missing important field: {field}")
    
    # Currency conversion penalty
    if snapshot.fx_rate_to_usd is None and snapshot.original_currency != "USD":
        score -= 0.05
        warnings.append("FX conversion failed, values in local currency")
    
    # Validation checks
    if snapshot.total_revenue and snapshot.gross_profit:
        if snapshot.gross_profit > snapshot.total_revenue:
            score -= 0.1
            warnings.append("Gross profit > revenue (inconsistent)")
    
    if snapshot.shares_outstanding and snapshot.shares_outstanding <= 0:
        score -= 0.2
        warnings.append("Invalid shares outstanding")
    
    return max(0.0, score), warnings
```

---

## Design Decisions

### ✅ RESOLVED: Currency Handling

**Decision**: 
- Attempt USD conversion at ingestion
- Fallback to local currency if FX unavailable
- Always store `original_currency` field
- Store `stored_currency` to indicate actual currency of values

### ✅ RESOLVED: Both Dates Stored

**Decision**: Store both `filing_date` and `release_date` when available.

### ✅ RESOLVED: FX Rate Source

**Decision**: Use **theratesapi.com** for currency conversion.

```python
# Example: GET https://api.theratesapi.com/v1/2024-01-15?base=EUR&symbols=USD
# Response: {"rates": {"USD": 1.0876}, "base": "EUR", "date": "2024-01-15"}
```

---

## Constraints

- ⚡ Attempt USD conversion, fallback to local currency
- ⚡ Always store `original_currency` field
- ⚡ Store `stored_currency` to indicate actual currency of values
- ⚡ Must handle missing fields gracefully
- ⚡ Quality score must be between 0.0 and 1.0
- ⚡ Must preserve raw payloads via ingestion module

---

## Integration Tests

### Test Scope

Integration tests verify normalization with real data structures and FX provider integration.

### Test Cases

```python
# tests/integration/test_normalize.py

class TestNormalizationIntegration:
    """Integration tests for normalization pipeline."""
    
    def test_usd_stock_normalization(self, mock_raw_snapshot):
        """
        Test normalizing a USD-native stock.
        
        Verifies:
        - original_currency = "USD"
        - stored_currency = "USD"  
        - fx_rate_to_usd = 1.0
        - All monetary fields preserved
        """
        snapshot = normalize(mock_raw_snapshot("AAPL", currency="USD"))
        
        assert snapshot.original_currency == "USD"
        assert snapshot.stored_currency == "USD"
        assert snapshot.fx_rate_to_usd == 1.0
    
    def test_foreign_stock_usd_conversion(self, mock_raw_snapshot, mock_fx):
        """
        Test normalizing a foreign stock with successful FX conversion.
        
        Verifies:
        - original_currency preserved (e.g., "EUR")
        - stored_currency = "USD"
        - fx_rate_to_usd is populated
        - All monetary values converted correctly
        """
        mock_fx.get_rate.return_value = 1.08  # EUR -> USD
        
        snapshot = normalize(mock_raw_snapshot("ASML.AS", currency="EUR"))
        
        assert snapshot.original_currency == "EUR"
        assert snapshot.stored_currency == "USD"
        assert snapshot.fx_rate_to_usd == pytest.approx(1.08)
        assert snapshot.total_revenue == 10000 * 1.08  # Converted
    
    def test_fx_fallback(self, mock_raw_snapshot, mock_fx_failing):
        """
        Test fallback when FX rate unavailable.
        
        Verifies:
        - original_currency preserved
        - stored_currency = original_currency (not converted)
        - fx_rate_to_usd is None
        - Values kept in original currency
        """
        snapshot = normalize(mock_raw_snapshot("7203.T", currency="JPY"))
        
        assert snapshot.original_currency == "JPY"
        assert snapshot.stored_currency == "JPY"
        assert snapshot.fx_rate_to_usd is None
    
    def test_quality_score_calculation(self, mock_raw_snapshot):
        """
        Test quality score reflects data completeness.
        
        Verifies:
        - Complete data gets high score
        - Missing critical fields reduce score significantly
        - Warnings are generated for issues
        """
        # Complete data
        complete = normalize(mock_raw_snapshot("AAPL", complete=True))
        assert complete.data_quality_score >= 0.8
        
        # Missing critical fields
        incomplete = normalize(mock_raw_snapshot("MSFT", missing=["total_revenue"]))
        assert incomplete.data_quality_score < 0.8
        assert "total_revenue" in str(incomplete.validation_warnings)
    
    def test_field_mapping(self, real_yfinance_response):
        """
        Test yfinance field mapping to canonical schema.
        
        Verifies:
        - All expected fields are mapped
        - Field names match schema
        - No data loss in mapping
        """
        snapshot = normalize_from_yfinance(real_yfinance_response)
        
        assert snapshot.total_revenue is not None
        assert snapshot.market_cap_t0 is not None
```

### Running Tests

```bash
# Run all normalization integration tests
pytest tests/integration/test_normalize.py -v

# Run with FX provider integration
pytest tests/integration/test_normalize.py -v --real-fx

# Run with coverage
pytest tests/integration/test_normalize.py --cov=src/normalize
```
