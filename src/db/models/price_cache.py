"""Price cache model for storing historical ticker prices."""

from sqlalchemy import Column, String, Date, Numeric, DateTime, Index, func
from .base import Base


class PriceCache(Base):
    """Cached historical prices for tickers.
    
    Used to avoid repeated API calls during backtesting.
    Stores daily closing prices from Yahoo Finance.
    """
    __tablename__ = "price_cache"
    
    ticker = Column(String(20), primary_key=True)
    price_date = Column(Date, primary_key=True)
    close_price = Column(Numeric, nullable=False)
    
    # Metadata
    fetched_at = Column(DateTime, server_default=func.now())
    source = Column(String(50), default="yfinance")
    
    __table_args__ = (
        Index('idx_price_cache_date', 'price_date'),
        Index('idx_price_cache_ticker', 'ticker'),
    )
