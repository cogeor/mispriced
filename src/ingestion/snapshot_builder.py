"""
Builds FinancialSnapshot objects from raw provider data.

Uses SchemaMapper from normalize module for field mapping.
Handles currency conversion via FXRateProvider.
"""

from datetime import datetime, date
from typing import Dict, Any, Optional

import pandas as pd

from src.db.models.snapshot import FinancialSnapshot
from src.providers.fx_rates import FXRateProvider
from src.normalize.schema_mapper import SchemaMapper


class SnapshotBuilder:
    """
    Builds FinancialSnapshot objects from raw provider data.

    Responsibilities:
    - Map provider fields to canonical schema (via SchemaMapper)
    - Handle currency conversion
    - Merge data from income, balance, cashflow, and company info
    """

    def __init__(self, fx_provider: FXRateProvider):
        self.fx_provider = fx_provider
        self.mapper = SchemaMapper()

    def build_snapshot(
        self,
        ticker: str,
        statement_date: date,
        raw_income: Dict[str, Any],
        raw_balance: Dict[str, Any],
        raw_cashflow: Dict[str, Any],
        company_info: Dict[str, Any],
        price_context: Dict[str, Optional[float]],
    ) -> FinancialSnapshot:
        """
        Construct a FinancialSnapshot from raw data parts.

        Args:
            ticker: Stock ticker symbol
            statement_date: Date of the financial statement
            raw_income: Income statement data for this date
            raw_balance: Balance sheet data for this date
            raw_cashflow: Cashflow statement data for this date
            company_info: Company metadata and current stats from .info
            price_context: Price data dict with keys 't0', 't-1', 't+1'

        Returns:
            FinancialSnapshot ready for database insertion
        """
        # Merge all financial statement data into one dict
        financials = self._merge_financials(raw_income, raw_balance, raw_cashflow)

        # Use SchemaMapper to map provider fields to canonical schema
        mapped_data = self.mapper.map_yfinance(company_info, financials)

        # Handle currency conversion
        original_currency = company_info.get("currency", "USD")
        financial_currency = company_info.get("financialCurrency")
        
        # If financial_currency is missing or None, fallback to original_currency
        if not financial_currency:
            financial_currency = original_currency

        converted_data, stored_currency, fx_rate = self._convert_monetary_fields(
            mapped_data, 
            trading_currency=original_currency,
            financial_currency=financial_currency,
            as_of_date=statement_date
        )

        # Build the snapshot
        snapshot = FinancialSnapshot(
            ticker=ticker,
            snapshot_timestamp=datetime.combine(statement_date, datetime.min.time()),
            period_end_date=statement_date,
            original_currency=original_currency,
            stored_currency=stored_currency,
            fx_rate_to_usd=fx_rate,
            # Price context
            price_t0=price_context.get("t0"),
            price_t_minus_1=price_context.get("t-1"),
            price_t_plus_1=price_context.get("t+1"),
            # All mapped fields from converted_data
            **converted_data,
        )

        return snapshot

    def _merge_financials(
        self,
        raw_income: Dict[str, Any],
        raw_balance: Dict[str, Any],
        raw_cashflow: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Merge income, balance sheet, and cashflow data into one dict.

        Handles potential key conflicts by preferring income > balance > cashflow.
        """
        merged = {}

        # Add in reverse priority order (later overwrites earlier)
        for source in [raw_cashflow, raw_balance, raw_income]:
            if source:
                for key, value in source.items():
                    if value is not None and not pd.isna(value):
                        merged[key] = value

        return merged

    def _convert_monetary_fields(
        self,
        data: Dict[str, Any],
        trading_currency: str,
        financial_currency: str,
        as_of_date: date,
    ) -> tuple[Dict[str, Any], str, Optional[float]]:
        """
        Convert monetary fields to USD if possible.
        
        Uses trading_currency for market data (market_cap) and
        financial_currency for financial statements (revenue, debt, etc).

        Args:
            data: Mapped data dictionary
            trading_currency: Currency for stock price/market cap
            financial_currency: Currency for financial statements
            as_of_date: Date for FX rate lookup

        Returns:
            (converted_data, stored_currency, fx_rate_to_usd)
            Note: fx_rate_to_usd returned is the TRADING currency rate, 
            as that's the primary reference for the stock.
        """
        # Fields derived from stock price (Trading Currency)
        TRADING_FIELDS = {
            "market_cap_t0",
        }
        
        # Fields from financial statements (Financial Currency)
        FINANCIAL_FIELDS = {
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
            "book_value",
        }

        # 1. Get FX Rates
        trading_fx = 1.0
        if trading_currency != "USD":
            try:
                trading_fx = self.fx_provider.get_rate(trading_currency, "USD", as_of_date)
            except Exception:
                trading_fx = None # Failed
        
        financial_fx = 1.0
        if financial_currency != "USD":
            try:
                financial_fx = self.fx_provider.get_rate(financial_currency, "USD", as_of_date)
            except Exception:
                financial_fx = None # Failed

        # 2. Convert fields
        converted = {}
        for key, value in data.items():
            if value is None:
                converted[key] = None
                continue
                
            # Handle Trading Fields
            if key in TRADING_FIELDS:
                if trading_fx is not None:
                     try:
                        converted[key] = float(value) * trading_fx
                     except (TypeError, ValueError):
                        converted[key] = value
                else:
                    # Keep original if rate missing
                    converted[key] = value
            
            # Handle Financial Fields
            elif key in FINANCIAL_FIELDS:
                if financial_fx is not None:
                    try:
                        converted[key] = float(value) * financial_fx
                    except (TypeError, ValueError):
                        converted[key] = value
                else:
                     converted[key] = value
                     
            # Non-monetary or already handled
            else:
                converted[key] = value

        # We return the trading FX rate as the primary "fx_rate_to_usd" 
        # because the Snapshot's 'original_currency' is typically the trading currency.
        # Ideally we'd store both, but schema has one. Trading makes sense for Price reference.
        return converted, "USD", trading_fx
