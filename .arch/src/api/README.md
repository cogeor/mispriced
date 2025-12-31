# API Layer

> REST API endpoints for serving snapshots, valuations, portfolios, and evaluation results.

---

## Responsibilities

- Serve snapshots, valuation outputs, portfolio targets, evaluation results via HTTP
- Provide query interfaces for filtering and aggregation
- Authentication and rate limiting (v2)

---

## Inputs

- DB query parameters:
  - Ticker, date range, sector filter
  - Latest snapshot views
  - Evaluation run IDs
- Request headers for authentication (v2)

---

## Outputs

- JSON responses via REST endpoints
- Pydantic-validated response models

---

## Dependencies

### External Packages
- `fastapi` - REST framework
- `pydantic` - Request/response validation
- `uvicorn` - ASGI server

### Internal Modules
- `src/db/` - Repository access
- `src/valuation/` - Valuation models
- `src/strategy/` - Portfolio models
- `src/evaluation/` - Evaluation results

---

## Folder Structure

```
src/api/
  __init__.py
  main.py                 # FastAPI app entry point
  routers/
    __init__.py
    snapshots.py          # Snapshot endpoints
    valuation.py          # Valuation endpoints
    portfolio.py          # Portfolio endpoints
    evaluation.py         # Evaluation endpoints
    health.py             # Health check / status
  schemas/
    __init__.py
    snapshot.py           # Snapshot response schemas
    valuation.py          # Valuation response schemas
    portfolio.py          # Portfolio response schemas
    common.py             # Common schemas (pagination, errors)
  dependencies.py         # Shared dependencies (DB session, auth)
  config.py               # API configuration
```

---

## Endpoint Specification

### Snapshots

```
GET /snapshots
  Query params: ticker, start_date, end_date, sector, limit, offset
  Response: List[SnapshotResponse]

GET /snapshots/{ticker}/latest
  Response: SnapshotResponse

GET /snapshots/{ticker}/history
  Query params: start_date, end_date, limit
  Response: List[SnapshotResponse]
```

### Valuation

```
GET /valuation/latest
  Query params: universe (comma-separated tickers), sector
  Response: List[ValuationResponse]

GET /valuation/{ticker}
  Query params: model_version (optional)
  Response: ValuationResponse

GET /valuation/{ticker}/history
  Query params: start_date, end_date
  Response: List[ValuationResponse]
```

### Portfolio

```
GET /portfolio/latest
  Response: PortfolioResponse

GET /portfolio/runs
  Query params: limit, offset
  Response: List[PortfolioRunSummary]

GET /portfolio/runs/{run_id}
  Response: PortfolioResponse with full weights
```

### Evaluation

```
GET /evaluation/runs
  Response: List[EvaluationRunSummary]

GET /evaluation/runs/{run_id}
  Response: EvaluationResponse with all metrics

GET /evaluation/{ticker}/metrics
  Response: TickerEvaluationResponse
```

---

## Response Schemas

### SnapshotResponse

```python
class SnapshotResponse(BaseModel):
    ticker: str
    snapshot_timestamp: datetime
    period_end_date: date
    frequency: str
    
    # Key financials
    revenue: Optional[float]
    net_income: Optional[float]
    eps: Optional[float]
    free_cash_flow: Optional[float]
    
    # Price context
    price_t0: Optional[float]
    market_cap_t0: Optional[float]
    
    # Quality
    data_quality_score: float
```

### ValuationResponse

```python
class ValuationResponse(BaseModel):
    ticker: str
    snapshot_timestamp: datetime
    intrinsic_value: float
    price_t0: float
    mispricing_pct: float
    confidence_score: float
    model_version: str
```

---

## Design Decisions

### ⚠️ NEEDS REVIEW: Authentication Strategy

**Options**:
1. **No auth** - Internal/research use only
2. **API key** - Simple token-based auth
3. **OAuth2** - Full authentication flow

### ⚠️ NEEDS REVIEW: Pagination Strategy

**Options**:
1. **Offset-based** - `limit` + `offset` params (simple but poor performance)
2. **Cursor-based** - `after` param with snapshot ID (better for large datasets)

### 📌 TODO: Rate Limiting

Implement rate limiting for:
- Per-IP limits
- Per-endpoint limits
- Burst vs sustained limits

### 📌 TODO: API Versioning

**Options**:
1. **URL path** - `/v1/snapshots`, `/v2/snapshots`
2. **Header** - `Accept: application/vnd.api+json;version=1`

---

## Constraints

- ⚡ All responses must be JSON
- ⚡ All endpoints must have Pydantic validation
- ⚡ Health endpoint must be available at `/health`
- ⚡ Error responses must follow consistent schema

---

## Integration Tests

### Test Scope

Integration tests verify API endpoints with real database and HTTP client.

### Test Cases

```python
# tests/integration/test_api.py
from fastapi.testclient import TestClient

class TestAPIIntegration:
    """Integration tests for REST API."""
    
    def test_health_endpoint(self, client):
        """
        Test health check endpoint.
        
        Verifies:
        - Returns 200 OK
        - Includes status field
        """
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_snapshot_list(self, client, test_db_with_data):
        """
        Test snapshot listing endpoint.
        
        Verifies:
        - Returns list of snapshots
        - Pagination works
        - Filtering by ticker works
        """
        response = client.get("/snapshots", params={"ticker": "AAPL", "limit": 10})
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert all(s["ticker"] == "AAPL" for s in data)
    
    def test_valuation_latest(self, client, test_db_with_data):
        """
        Test latest valuation endpoint.
        
        Verifies:
        - Returns valuation data
        - Includes required fields
        - Model version is present
        """
        response = client.get("/valuation/AAPL")
        assert response.status_code == 200
        data = response.json()
        assert "intrinsic_value" in data
        assert "mispricing_pct" in data
    
    def test_portfolio_latest(self, client, test_db_with_data):
        """
        Test latest portfolio endpoint.
        
        Verifies:
        - Returns portfolio weights
        - Weights are valid
        """
        response = client.get("/portfolio/latest")
        assert response.status_code == 200
    
    def test_error_handling(self, client):
        """
        Test error response format.
        
        Verifies:
        - 404 for invalid ticker
        - Error response schema is consistent
        """
        response = client.get("/valuation/INVALID_TICKER")
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
```

### Running Tests

```bash
# Run API integration tests
pytest tests/integration/test_api.py -v

# Run with test database
pytest tests/integration/test_api.py -v --db-url="postgresql://test:test@localhost:5432/test"

# Run with coverage
pytest tests/integration/test_api.py --cov=src/api
```
