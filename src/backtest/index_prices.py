"""On-the-fly index price fetching for backtest purposes."""

import logging
from datetime import date, timedelta
from typing import Optional

import yfinance as yf

logger = logging.getLogger(__name__)


# Index symbol mappings (Yahoo Finance format)
INDEX_SYMBOLS: dict[str, str] = {
    "SP500": "^GSPC",
    "NASDAQ100": "^NDX",
    "DOWJONES": "^DJI",
    "EUROSTOXX50": "^STOXX50E",
    "CAC40": "^FCHI",
    "DAX": "^GDAXI",
    "FTSE100": "^FTSE",
    "SMI": "^SSMI",
    "NIKKEI225": "^N225",
    "NIFTY50": "^NSEI",
    "SSE50": "000016.SS",  # Shanghai
}


def get_index_price(index_id: str, target_date: date) -> Optional[float]:
    """
    Fetch index closing price on target_date.
    
    Falls back to previous trading day if market closed.
    
    Args:
        index_id: Index identifier (e.g., "SP500")
        target_date: Target date for price
        
    Returns:
        Closing price, or None if not available
        
    Raises:
        ValueError: If index_id is unknown
    """
    symbol = INDEX_SYMBOLS.get(index_id)
    if not symbol:
        raise ValueError(f"Unknown index: {index_id}. Available: {list(INDEX_SYMBOLS.keys())}")
    
    # Fetch window around target date (handles weekends/holidays)
    start = target_date - timedelta(days=10)
    end = target_date + timedelta(days=1)
    
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start.isoformat(), end=end.isoformat())
        
        if hist.empty:
            logger.warning(f"No price data for {index_id} ({symbol}) around {target_date}")
            return None
        
        # Get closest date <= target_date
        valid_dates = [d.date() for d in hist.index if d.date() <= target_date]
        if not valid_dates:
            logger.warning(f"No price data for {index_id} on or before {target_date}")
            return None
        
        closest = max(valid_dates)
        # Find the row with this date
        for idx in hist.index:
            if idx.date() == closest:
                return float(hist.loc[idx]["Close"])
        
        return None
        
    except Exception as e:
        logger.error(f"Error fetching price for {index_id}: {e}")
        return None


def get_index_return(
    index_id: str,
    from_date: date,
    to_date: date,
) -> Optional[float]:
    """
    Calculate index return over period.
    
    Args:
        index_id: Index identifier
        from_date: Start date
        to_date: End date
        
    Returns:
        Return as decimal (e.g., 0.05 = 5%), or None if prices unavailable
    """
    p_start = get_index_price(index_id, from_date)
    p_end = get_index_price(index_id, to_date)
    
    if p_start is None or p_end is None:
        return None
    
    if p_start == 0:
        logger.warning(f"Start price is zero for {index_id} on {from_date}")
        return None
    
    return (p_end - p_start) / p_start
