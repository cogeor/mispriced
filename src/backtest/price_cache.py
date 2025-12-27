"""Price caching utilities for backtesting.

Provides functions to fetch and cache historical ticker prices,
avoiding repeated API calls during backtest runs.
"""

import logging
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import yfinance as yf
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert

from src.db.models.price_cache import PriceCache

logger = logging.getLogger(__name__)


def get_cached_price(
    session: Session,
    ticker: str,
    target_date: date,
) -> Optional[float]:
    """
    Get price from cache, returns None if not cached.
    
    Args:
        session: Database session
        ticker: Ticker symbol
        target_date: Date to get price for
        
    Returns:
        Cached close price or None
    """
    result = session.query(PriceCache.close_price).filter(
        PriceCache.ticker == ticker,
        PriceCache.price_date == target_date,
    ).first()
    
    return float(result[0]) if result else None


def cache_prices(
    session: Session,
    ticker: str,
    prices: Dict[date, float],
    commit: bool = True,
) -> int:
    """
    Store prices in cache.
    
    Args:
        session: Database session
        ticker: Ticker symbol
        prices: Dict mapping date -> close price
        commit: Whether to commit transaction
        
    Returns:
        Number of prices cached
    """
    count = 0
    for price_date, close_price in prices.items():
        stmt = insert(PriceCache).values(
            ticker=ticker,
            price_date=price_date,
            close_price=close_price,
            source="yfinance",
        ).on_conflict_do_update(
            index_elements=["ticker", "price_date"],
            set_={"close_price": close_price},
        )
        session.execute(stmt)
        count += 1
    
    if commit:
        session.commit()
    
    return count


def fetch_and_cache_prices(
    session: Session,
    tickers: List[str],
    start_date: date,
    end_date: date,
    batch_size: int = 50,
) -> Dict[str, Dict[date, float]]:
    """
    Fetch prices from Yahoo Finance for multiple tickers and cache them.
    
    First checks cache, only fetches missing data.
    
    Args:
        session: Database session
        tickers: List of ticker symbols
        start_date: Start of date range
        end_date: End of date range
        batch_size: Number of tickers to fetch at once
        
    Returns:
        Dict mapping ticker -> {date -> price}
    """
    all_prices: Dict[str, Dict[date, float]] = {}
    
    # Check what's already cached
    cached = session.query(
        PriceCache.ticker,
        PriceCache.price_date,
        PriceCache.close_price
    ).filter(
        PriceCache.ticker.in_(tickers),
        PriceCache.price_date >= start_date,
        PriceCache.price_date <= end_date,
    ).all()
    
    for ticker, price_date, close_price in cached:
        if ticker not in all_prices:
            all_prices[ticker] = {}
        all_prices[ticker][price_date] = float(close_price)
    
    logger.info(f"Found {len(cached)} cached prices for {len(all_prices)} tickers")
    
    # Find tickers that need fetching (any with missing dates)
    tickers_to_fetch = [t for t in tickers if t not in all_prices or len(all_prices[t]) < 5]
    
    if not tickers_to_fetch:
        logger.info("All prices cached, no fetch needed")
        return all_prices
    
    logger.info(f"Fetching prices for {len(tickers_to_fetch)} tickers...")
    
    # Fetch in batches
    for i in range(0, len(tickers_to_fetch), batch_size):
        batch = tickers_to_fetch[i:i + batch_size]
        logger.debug(f"  Fetching batch {i // batch_size + 1}...")
        
        try:
            # yfinance can download multiple tickers at once
            data = yf.download(
                batch,
                start=start_date.isoformat(),
                end=(end_date + timedelta(days=1)).isoformat(),
                progress=False,
                threads=True,
            )
            
            if data.empty:
                continue
            
            # Handle single vs multiple tickers (yfinance returns different structures)
            if len(batch) == 1:
                ticker = batch[0]
                if "Close" in data.columns:
                    ticker_prices = {d.date(): float(p) for d, p in data["Close"].items() if pd.notna(p)}
                    all_prices[ticker] = ticker_prices
                    cache_prices(session, ticker, ticker_prices, commit=False)
            else:
                # Multi-ticker: data has multi-level columns
                if "Close" in data.columns:
                    close_data = data["Close"]
                    for ticker in batch:
                        if ticker in close_data.columns:
                            ticker_prices = {
                                d.date(): float(p) 
                                for d, p in close_data[ticker].items() 
                                if pd.notna(p)
                            }
                            if ticker_prices:
                                all_prices[ticker] = ticker_prices
                                cache_prices(session, ticker, ticker_prices, commit=False)
        
        except Exception as e:
            logger.warning(f"Error fetching batch: {e}")
            continue
    
    session.commit()
    logger.info(f"Cached prices for {len(all_prices)} tickers")
    
    return all_prices


def get_ticker_return(
    prices: Dict[date, float],
    from_date: date,
    horizon_days: int,
) -> Optional[float]:
    """
    Calculate return from cached prices.
    
    Finds closest available price on or before from_date,
    and closest price on or after from_date + horizon_days.
    
    Args:
        prices: Dict mapping date -> price for a single ticker
        from_date: Start date
        horizon_days: Number of days forward
        
    Returns:
        Return as decimal (e.g., 0.05 = 5%), or None if prices unavailable
    """
    if not prices:
        return None
    
    # Find start price (on or before from_date)
    start_candidates = [d for d in prices.keys() if d <= from_date]
    if not start_candidates:
        return None
    start_date = max(start_candidates)
    start_price = prices[start_date]
    
    # Find end price (on or after from_date + horizon_days)
    target_end = from_date + timedelta(days=horizon_days)
    end_candidates = [d for d in prices.keys() if d >= target_end]
    if not end_candidates:
        # Try closest before target
        end_candidates = [d for d in prices.keys() if d > from_date]
        if not end_candidates:
            return None
    end_date = min(end_candidates)
    end_price = prices[end_date]
    
    if start_price == 0:
        return None
    
    return (end_price - start_price) / start_price
