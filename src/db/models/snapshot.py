from sqlalchemy import Column, String, Integer, Numeric, Boolean, Date, DateTime, JSON, ForeignKey, func, Index
from sqlalchemy.orm import relationship
from .base import Base

class FinancialSnapshot(Base):
    __tablename__ = "financial_snapshots"
    
    ticker = Column(String(20), primary_key=True)
    snapshot_timestamp = Column(DateTime, primary_key=True)
    
    # Statement metadata
    period_end_date = Column(Date)
    filing_date = Column(Date)
    release_date = Column(Date)
    frequency = Column(String(10))  # 'quarterly' | 'annual'
    
    # Currency
    original_currency = Column(String(10), nullable=False)
    stored_currency = Column(String(10), nullable=False)
    fx_rate_to_usd = Column(Numeric)
    # False when monetary fields are NOT verified to be in USD (e.g. FX
    # provider failed at ingestion, GBp/financialCurrency yfinance quirks).
    # Downstream consumers must exclude these rows or treat them as suspect.
    currency_validated = Column(Boolean, nullable=False, server_default="1")
    
    # Financials
    total_revenue = Column(Numeric)
    gross_profit = Column(Numeric)
    ebitda = Column(Numeric)
    operating_income = Column(Numeric)
    net_income = Column(Numeric)
    eps = Column(Numeric)
    shares_outstanding = Column(Numeric)
    float_shares = Column(Numeric)
    total_debt = Column(Numeric)
    total_cash = Column(Numeric)
    total_assets = Column(Numeric)
    free_cash_flow = Column(Numeric)
    operating_cash_flow = Column(Numeric)
    capex = Column(Numeric)
    working_capital = Column(Numeric)
    book_value = Column(Numeric)
    
    # Derived ratios
    profit_margins = Column(Numeric)
    gross_margin = Column(Numeric)
    operating_margin = Column(Numeric)
    net_margin = Column(Numeric)
    roe = Column(Numeric)
    roa = Column(Numeric)
    roic = Column(Numeric)
    debt_to_equity = Column(Numeric)
    current_ratio = Column(Numeric)
    quick_ratio = Column(Numeric)
    
    # Dividend
    dividend_yield = Column(Numeric)
    dividend_rate = Column(Numeric)
    five_year_avg_div_yield = Column(Numeric)
    
    # Risk scores
    audit_risk = Column(Integer)
    board_risk = Column(Integer)
    compensation_risk = Column(Integer)
    beta = Column(Numeric)
    
    # Short interest
    shares_short = Column(Numeric)
    shares_short_prior_month = Column(Numeric)
    short_ratio = Column(Numeric)
    
    # Ownership
    held_percent_insiders = Column(Numeric)
    held_percent_institutions = Column(Numeric)
    
    # Employees
    full_time_employees = Column(Integer)
    
    # Price context
    price_t0 = Column(Numeric)
    price_t_minus_1 = Column(Numeric)
    price_t_plus_1 = Column(Numeric)
    market_cap_t0 = Column(Numeric)
    
    # Quality metadata
    data_quality_score = Column(Numeric)
    validation_warnings = Column(JSON)
    
    # Audit
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Indexes defined in __table_args__ or implicitly via create_index
    __table_args__ = (
        Index('idx_snapshots_timestamp', 'snapshot_timestamp'),
        Index('idx_snapshots_filing', 'filing_date'),
        Index('idx_snapshots_mcap', 'market_cap_t0'),
    )
