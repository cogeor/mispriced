
from typing import Dict, Any, Optional

class SchemaMapper:
    """
    Map provider-specific data (e.g., YFinance) to canonical schema fields.
    Handles key aliases and deep fetching.
    """
    
    def map_yfinance(self, info: Dict[str, Any], financials: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map YFinance raw dictionaries to FinancialSnapshot fields.
        
        Args:
            info: YFinance .info dictionary
            financials: Flattened dictionary of various statement line items
        
        Returns:
            Dict[str, Any]: Dictionary matching FinancialSnapshot column names
        """
        data = {}
        
        # Mappings: Canonical Field -> List of possible Provider Keys
        # Order matters: first found wins
        mappings = {
            "market_cap_t0": ["marketCap"],
            "total_revenue": ["totalRevenue", "Total Revenue"],
            "gross_profit": ["grossProfit", "Gross Profit"],
            "ebitda": ["ebitda", "EBITDA"],
            "operating_income": ["operatingIncome", "Operating Income"],
            "net_income": ["netIncome", "Net Income", "netIncomeToCommon"],
            "eps": ["trailingEps", "forwardEps"], # Prefer trailing
            "shares_outstanding": ["sharesOutstanding", "impliedSharesOutstanding"],
            "float_shares": ["floatShares"],
            "total_debt": ["totalDebt", "Total Debt"],
            "total_cash": ["totalCash", "Total Cash", "Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"],
            "total_assets": ["totalAssets", "Total Assets"],
            "free_cash_flow": ["freeCashflow", "Free Cash Flow"],
            "operating_cash_flow": ["operatingCashflow", "Operating Cash Flow"],
            "capex": ["capitalExpenditures", "Capital Expenditures", "Capital Expenditure"],
            "working_capital": ["Working Capital"],
            "book_value": ["bookValue", "Stockholders Equity", "Tangible Book Value"],
            # Ratios
            "profit_margins": ["profitMargins"],
            "gross_margin": ["grossMargins"],
            "operating_margin": ["operatingMargins"],
            "roe": ["returnOnEquity"],
            "roa": ["returnOnAssets"],
            "roic": ["returnOnCapital"], # Might not be direct
            "debt_to_equity": ["debtToEquity"],
            "current_ratio": ["currentRatio"],
            "quick_ratio": ["quickRatio"],
            "dividend_yield": ["dividendYield"],
            "dividend_rate": ["dividendRate"],
            "five_year_avg_div_yield": ["fiveYearAvgDividendYield"],
            "beta": ["beta"],
            
            # Risk
            "audit_risk": ["auditRisk"],
            "board_risk": ["boardRisk"],
            "compensation_risk": ["compensationRisk"],
            
            # Short
            "shares_short": ["sharesShort"],
            "shares_short_prior_month": ["sharesShortPriorMonth", "sharesShortPreviousMonthDate"],
            "short_ratio": ["shortRatio"],
            "held_percent_insiders": ["heldPercentInsiders"],
            "held_percent_institutions": ["heldPercentInstitutions"],
            "full_time_employees": ["fullTimeEmployees"],
        }
        
        for field, keys in mappings.items():
            for key in keys:
                # check info first
                val = info.get(key)
                if val is not None:
                    data[field] = val
                    break
                
                # check financials next
                val = financials.get(key)
                if val is not None:
                    data[field] = val
                    break
                    
        return data
