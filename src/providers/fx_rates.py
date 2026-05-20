from datetime import date, timedelta
from typing import Dict, Optional, Tuple

import yfinance as yf


class FXRateProvider:
    """Abstract base for FX providers."""

    def get_rate(self, from_currency: str, to_currency: str, as_of_date: date) -> float:
        raise NotImplementedError


class YFinanceFXProvider(FXRateProvider):
    """
    FX rate provider backed by yfinance currency pairs (e.g., 'EURUSD=X').

    Looks up the daily close on or before the requested date.
    Handles 'GBp' (British pence) by deriving from 'GBPUSD=X' and dividing by 100.
    Caches successful lookups in-process.
    """

    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, str, date], float] = {}

    def get_rate(self, from_currency: str, to_currency: str, as_of_date: date) -> float:
        if from_currency == to_currency:
            return 1.0

        key = (from_currency, to_currency, as_of_date)
        if key in self._cache:
            return self._cache[key]

        pair_from, scale = _normalize_currency(from_currency)
        pair_to, scale_to = _normalize_currency(to_currency)
        rate = _fetch_rate(pair_from, pair_to, as_of_date)
        rate = rate * scale / scale_to

        self._cache[key] = rate
        return rate


def _normalize_currency(code: str) -> Tuple[str, float]:
    """Map non-ISO currency codes to ISO equivalents with a scale factor.

    GBp (pence) -> GBP with scale 0.01.
    ZAc (South African cents) -> ZAR with scale 0.01.
    """
    if code == "GBp":
        return "GBP", 0.01
    if code == "ZAc":
        return "ZAR", 0.01
    return code, 1.0


def _fetch_rate(from_iso: str, to_iso: str, as_of_date: date) -> float:
    """Fetch FX rate using yfinance for the given ISO currency pair.

    Looks at a small window ending on as_of_date and uses the last available close.
    """
    if from_iso == to_iso:
        return 1.0

    pair = f"{from_iso}{to_iso}=X"
    start = as_of_date - timedelta(days=10)
    end = as_of_date + timedelta(days=2)
    hist = yf.Ticker(pair).history(start=start.isoformat(), end=end.isoformat(), interval="1d")
    if hist.empty:
        raise ValueError(f"yfinance returned no rows for {pair} around {as_of_date}")

    on_or_before = hist[hist.index.date <= as_of_date]
    if on_or_before.empty:
        raise ValueError(f"yfinance has no quotes on or before {as_of_date} for {pair}")
    return float(on_or_before["Close"].iloc[-1])


FX_PROVIDER = YFinanceFXProvider()
