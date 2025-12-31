# External Data Providers

> Abstraction layer for external financial data APIs.

---

## Responsibilities

- Abstract external API interactions behind consistent interfaces
- Handle authentication, rate limiting, and retries
- Normalize raw responses for downstream processing
- Cache responses where appropriate

---

## Inputs

- Ticker symbols
- Date ranges
- Endpoint-specific parameters
- API configuration (keys, rate limits)

---

## Outputs

- Raw API responses (JSON)
- Provider-specific metadata (rate limit status, response headers)
- Error responses with retry information

---

## Dependencies

### External Packages
- `httpx` - Async HTTP client
- `tenacity` - Retry logic
- `pydantic` - Configuration validation

### Internal Modules
- None (infrastructure layer - no internal dependencies)

---

## Folder Structure

```
src/providers/
  __init__.py
  base.py                 # Abstract provider interface
  yahoo/
    __init__.py
    client.py             # Yahoo Finance API client
    endpoints.py          # Endpoint definitions
    models.py             # Response models
  alpha_vantage/          # Future: Alternative provider
    __init__.py
    client.py
  simfin/                 # Future: Free fundamentals
    __init__.py
    client.py
  cache.py                # Response caching
  rate_limiter.py         # Rate limit handling
  config.py               # Provider configuration
```

---

## Provider Interface

```python
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from datetime import date

class FinancialDataProvider(ABC):
    """Abstract base for all financial data providers."""
    
    @abstractmethod
    async def get_company_info(self, ticker: str) -> Dict[str, Any]:
        """Fetch company metadata (name, sector, industry, exchange)."""
        pass
    
    @abstractmethod
    async def get_income_statement(
        self, ticker: str, frequency: str = "quarterly"
    ) -> Dict[str, Any]:
        """Fetch income statement history."""
        pass
    
    @abstractmethod
    async def get_balance_sheet(
        self, ticker: str, frequency: str = "quarterly"
    ) -> Dict[str, Any]:
        """Fetch balance sheet history."""
        pass
    
    @abstractmethod
    async def get_cashflow_statement(
        self, ticker: str, frequency: str = "quarterly"
    ) -> Dict[str, Any]:
        """Fetch cash flow statement history."""
        pass
    
    @abstractmethod
    async def get_price_at_date(
        self, ticker: str, target_date: date
    ) -> Optional[float]:
        """Fetch closing price for a specific date."""
        pass
    
    @abstractmethod
    async def get_price_window(
        self, ticker: str, target_date: date, days_before: int = 1, days_after: int = 1
    ) -> Dict[str, Optional[float]]:
        """Fetch price window around a date (t-1, t0, t+1)."""
        pass
```

---

## Yahoo Finance Implementation

### ⚠️ NEEDS REVIEW: API Choice

**Options**:
1. **yfinance** - Python wrapper, unofficial, may break
2. **RapidAPI Yahoo Finance** - Paid, stable, documented
3. **Yahoo Query (unofficial)** - Free, risk of blocking

### Endpoints Used

| Data Type | Endpoint | Notes |
|-----------|----------|-------|
| Company Info | `/quoteSummary?modules=assetProfile` | Basic metadata |
| Income Statement | `/quoteSummary?modules=incomeStatementHistory` | Quarterly/Annual |
| Balance Sheet | `/quoteSummary?modules=balanceSheetHistory` | Quarterly/Annual |
| Cash Flow | `/quoteSummary?modules=cashflowStatementHistory` | Quarterly/Annual |
| Historical Price | `/chart/{ticker}?range=5d&interval=1d` | For t-1/t0/t+1 |

### Rate Limiting Strategy

```python
class YahooRateLimiter:
    """
    Yahoo Finance unofficial rate limits:
    - ~2000 requests/hour recommended
    - Implement exponential backoff on 429
    """
    
    def __init__(self, requests_per_hour: int = 1800):
        self.requests_per_hour = requests_per_hour
        self.min_interval = 3600 / requests_per_hour  # seconds
        self.last_request = 0.0
    
    async def wait_if_needed(self):
        elapsed = time.time() - self.last_request
        if elapsed < self.min_interval:
            await asyncio.sleep(self.min_interval - elapsed)
        self.last_request = time.time()
```

---

## FX Rate Provider: theratesapi.com

### ✅ RESOLVED: Use theratesapi.com

```python
# src/providers/fx_rates.py
import httpx
from datetime import date
from typing import Optional

class TheRatesAPIProvider:
    """
    FX rate provider using theratesapi.com
    Free tier: 1000 requests/month
    """
    
    BASE_URL = "https://api.theratesapi.com/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.cache = {}  # Simple in-memory cache
    
    def get_rate(
        self, 
        from_currency: str, 
        to_currency: str, 
        as_of_date: date
    ) -> float:
        """
        Get exchange rate for a specific date.
        
        Example: get_rate("EUR", "USD", date(2024, 1, 15))
        """
        cache_key = f"{from_currency}_{to_currency}_{as_of_date}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # API call
        url = f"{self.BASE_URL}/{as_of_date.isoformat()}"
        params = {
            "base": from_currency,
            "symbols": to_currency,
        }
        if self.api_key:
            params["access_key"] = self.api_key
        
        response = httpx.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        rate = data["rates"][to_currency]
        
        self.cache[cache_key] = rate
        return rate
    
    def convert(
        self, 
        amount: float, 
        from_currency: str, 
        to_currency: str, 
        as_of_date: date
    ) -> tuple[float, float]:
        """
        Convert amount and return (converted_amount, rate_used).
        """
        if from_currency == to_currency:
            return amount, 1.0
        
        rate = self.get_rate(from_currency, to_currency, as_of_date)
        return amount * rate, rate


# === ACTIVE FX PROVIDER ===
FX_PROVIDER = TheRatesAPIProvider()
```

---

## Caching Strategy

### ⚠️ NEEDS REVIEW: Cache Implementation

**Options**:
1. **In-memory (TTL)** - Fast, loses data on restart
2. **Redis** - Distributed, persistent, operational complexity
3. **SQLite cache** - Simple, persistent, single-node

### Cache Key Design

```python
def cache_key(provider: str, endpoint: str, ticker: str, params: dict) -> str:
    param_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
    return f"{provider}:{endpoint}:{ticker}:{param_hash}"
```

### TTL Strategy

| Data Type | TTL | Rationale |
|-----------|-----|-----------|
| Company Info | 24 hours | Changes infrequently |
| Financial Statements | 7 days | Quarterly updates |
| Historical Prices | 30 days | Doesn't change after fact |

---

## Error Handling

```python
class ProviderError(Exception):
    """Base exception for provider errors."""
    pass

class RateLimitError(ProviderError):
    """Raised when rate limited, includes retry-after."""
    def __init__(self, retry_after: Optional[int] = None):
        self.retry_after = retry_after

class DataNotFoundError(ProviderError):
    """Raised when ticker/data not found."""
    pass

class ProviderUnavailableError(ProviderError):
    """Raised when provider is down/unreachable."""
    pass
```

---

## Design Decisions

### ⚠️ NEEDS REVIEW: Multi-Provider Strategy

**Options**:
1. **Single provider** - Yahoo only, simpler
2. **Fallback chain** - Try Yahoo → Alpha Vantage → SimFin
3. **Best-of-breed** - Use specific provider per data type

### 📌 TODO: API Key Management

Securely store and rotate API keys:
- Environment variables
- Secrets manager integration

### 💡 ALTERNATIVE: Batch Fetching

Some providers support batch endpoints. Could reduce API calls:
- Fetch multiple tickers in one request
- Fetch all statement types in one request

---

## Constraints

- ⚡ Must respect provider rate limits
- ⚡ Must store raw responses for audit (via ingestion module)
- ⚡ Must handle provider outages gracefully
- ⚡ Must not cache indefinitely (TTL required)
- ⚡ No credentials in code - use environment/config

---

## Integration Tests

### Test Scope

Integration tests verify provider connectivity, rate limiting, and FX rate retrieval.

### Test Cases

```python
# tests/integration/test_providers.py

class TestProviderIntegration:
    """Integration tests for data providers."""
    
    async def test_yfinance_basic_fetch(self, real_network):
        """
        Test basic yfinance data fetch.
        
        Verifies:
        - Can connect to Yahoo Finance
        - Returns valid company info
        - All expected fields present
        """
        provider = YFinanceProvider()
        info = await provider.get_company_info("AAPL")
        
        assert "symbol" in info
        assert "currency" in info
        assert "marketCap" in info
    
    async def test_fx_rate_fetch(self, real_network):
        """
        Test FX rate retrieval from theratesapi.com.
        
        Verifies:
        - Can fetch historical rate
        - Rate is reasonable (e.g., EUR/USD between 0.8 and 1.5)
        - Caching works
        """
        provider = TheRatesAPIProvider()
        rate = provider.get_rate("EUR", "USD", date(2024, 1, 15))
        
        assert 0.8 < rate < 1.5
    
    async def test_rate_limiting(self, mock_yfinance):
        """
        Test rate limiting enforces delays.
        
        Verifies:
        - Sequential requests respect rate limit
        - Total time >= expected minimum
        """
        limiter = YahooRateLimiter(requests_per_hour=3600)  # 1/sec
        
        start = time.time()
        for _ in range(3):
            await limiter.wait_if_needed()
        elapsed = time.time() - start
        
        assert elapsed >= 2.0  # At least 2 seconds for 3 requests
    
    async def test_provider_error_handling(self, mock_yfinance_failing):
        """
        Test error handling for provider failures.
        
        Verifies:
        - RateLimitError raised on 429
        - DataNotFoundError raised for invalid ticker
        - ProviderUnavailableError for network issues
        """
        provider = YFinanceProvider()
        
        with pytest.raises(DataNotFoundError):
            await provider.get_company_info("INVALID_TICKER_XYZ")
```

### Running Tests

```bash
# Run provider tests (mocked)
pytest tests/integration/test_providers.py -v

# Run with real network (slow, may hit rate limits)
pytest tests/integration/test_providers.py -v --real-network

# Run with coverage
pytest tests/integration/test_providers.py --cov=src/providers
```
