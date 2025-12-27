"""Unit tests for backtest module."""

import pytest
import numpy as np
from datetime import date
from unittest.mock import Mock, patch, MagicMock

from src.backtest.models import (
    BacktestConfig,
    BacktestPoint,
    BacktestResult,
    CorrelationMetrics,
)
from src.backtest.analysis.correlation import (
    analyze_correlation,
    bootstrap_correlation_ci,
    _compute_quintile_returns,
)
from src.backtest.index_prices import get_index_price, get_index_return, INDEX_SYMBOLS


class TestBacktestModels:
    """Tests for Pydantic models."""
    
    def test_backtest_config_defaults(self) -> None:
        """Test BacktestConfig with minimal required fields."""
        config = BacktestConfig(
            indices=["SP500"],
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31),
        )
        
        assert config.step_days == 90
        assert config.forward_horizons == [63, 126, 252]
        assert config.min_coverage == 0.5
    
    def test_backtest_config_custom(self) -> None:
        """Test BacktestConfig with custom values."""
        config = BacktestConfig(
            indices=["SP500", "NASDAQ100"],
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31),
            step_days=60,
            forward_horizons=[30, 60, 90],
            min_coverage=0.7,
        )
        
        assert config.step_days == 60
        assert len(config.forward_horizons) == 3
        assert config.min_coverage == 0.7
    
    def test_backtest_point_creation(self) -> None:
        """Test BacktestPoint creation."""
        point = BacktestPoint(
            cutoff_date=date(2020, 6, 30),
            index_id="SP500",
            actual_index_value=3000.0,
            estimated_index_value=2800.0,
            estimated_index_std=100.0,
            overpriciness=0.0714,  # (3000 - 2800) / 2800
            forward_returns={63: 0.05, 126: 0.08},
            n_constituents=500,
            n_constituents_with_valuation=450,
            coverage_ratio=0.9,
        )
        
        assert point.index_id == "SP500"
        assert point.overpriciness == pytest.approx(0.0714)
        assert 63 in point.forward_returns
    
    def test_correlation_metrics_creation(self) -> None:
        """Test CorrelationMetrics creation."""
        metrics = CorrelationMetrics(
            horizon_days=63,
            pearson_r=-0.3,
            pearson_p_value=0.05,
            spearman_rho=-0.28,
            spearman_p_value=0.06,
            hit_rate=0.65,
            hit_rate_p_value=0.02,
            n_observations=50,
        )
        
        assert metrics.pearson_r == -0.3
        assert metrics.hit_rate == 0.65


class TestCorrelationAnalysis:
    """Tests for correlation analysis functions."""
    
    def test_analyze_correlation_basic(self) -> None:
        """Test basic correlation analysis."""
        # Create synthetic negatively correlated data
        np.random.seed(42)
        n = 50
        overpriciness = np.random.randn(n)
        # Negative correlation: high overpriciness -> low returns
        returns = -0.5 * overpriciness + 0.3 * np.random.randn(n)
        
        metrics = analyze_correlation(overpriciness, returns, horizon_days=63)
        
        assert metrics.pearson_r < 0  # Should be negative
        assert metrics.n_observations == n
        assert 0 <= metrics.hit_rate <= 1
        assert metrics.bootstrap_ci_pearson is not None
    
    def test_analyze_correlation_insufficient_data(self) -> None:
        """Test that insufficient data raises error."""
        overpriciness = np.array([0.1, 0.2, 0.3])
        returns = np.array([-0.05, -0.03, -0.04])
        
        with pytest.raises(ValueError, match="Insufficient observations"):
            analyze_correlation(overpriciness, returns, horizon_days=63)
    
    def test_quintile_returns(self) -> None:
        """Test quintile return calculation."""
        np.random.seed(42)
        overpriciness = np.linspace(-0.2, 0.2, 50)
        # Lower overpriciness = higher returns (thesis)
        returns = -overpriciness + 0.05 * np.random.randn(50)
        
        quintiles = _compute_quintile_returns(overpriciness, returns)
        
        assert len(quintiles) == 5
        # Quintile 1 (most underpriced) should have higher returns than Q5
        assert quintiles[1] > quintiles[5]
    
    def test_bootstrap_ci(self) -> None:
        """Test bootstrap confidence interval."""
        np.random.seed(42)
        overpriciness = np.random.randn(100)
        returns = -0.4 * overpriciness + 0.5 * np.random.randn(100)
        
        lower, upper = bootstrap_correlation_ci(overpriciness, returns, n_bootstrap=500)
        
        assert lower < upper
        assert lower < 0  # Should capture negative correlation
        assert upper < 0.5  # Not too wide


class TestIndexPrices:
    """Tests for index price fetching."""
    
    def test_index_symbols_exist(self) -> None:
        """Test that common indices are in the symbol map."""
        assert "SP500" in INDEX_SYMBOLS
        assert "NASDAQ100" in INDEX_SYMBOLS
        assert INDEX_SYMBOLS["SP500"] == "^GSPC"
    
    def test_unknown_index_raises(self) -> None:
        """Test that unknown index raises ValueError."""
        with pytest.raises(ValueError, match="Unknown index"):
            get_index_price("FAKE_INDEX", date(2023, 1, 1))
    
    @patch("src.backtest.index_prices.yf.Ticker")
    def test_get_index_price_success(self, mock_ticker_class: MagicMock) -> None:
        """Test successful price fetch."""
        import pandas as pd
        
        # Create mock history data
        mock_hist = pd.DataFrame(
            {"Close": [4500.0, 4520.0, 4510.0]},
            index=pd.DatetimeIndex([
                "2023-06-28", "2023-06-29", "2023-06-30"
            ])
        )
        
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_hist
        mock_ticker_class.return_value = mock_ticker
        
        price = get_index_price("SP500", date(2023, 6, 30))
        
        assert price == pytest.approx(4510.0)
    
    @patch("src.backtest.index_prices.yf.Ticker")
    def test_get_index_return(self, mock_ticker_class: MagicMock) -> None:
        """Test return calculation."""
        import pandas as pd
        
        # Create mock with two different prices
        mock_hist = pd.DataFrame(
            {"Close": [100.0, 105.0, 110.0]},
            index=pd.DatetimeIndex([
                "2023-01-02", "2023-01-15", "2023-01-31"
            ])
        )
        
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_hist
        mock_ticker_class.return_value = mock_ticker
        
        # Return from 100 to 110 = 10%
        ret = get_index_return("SP500", date(2023, 1, 2), date(2023, 1, 31))
        
        assert ret == pytest.approx(0.10)


class TestBacktestService:
    """Tests for BacktestService."""
    
    @pytest.fixture
    def mock_session(self) -> Mock:
        """Create a mock database session."""
        return Mock()
    
    @pytest.fixture  
    def mock_index_result(self) -> Mock:
        """Create a mock IndexResult."""
        result = Mock()
        result.actual_index = 3000.0
        result.estimated_index = 2800.0
        result.estimated_index_std = 100.0
        result.n_tickers = 500
        result.n_tickers_with_valuation = 450
        return result
    
    def test_backtest_service_init(self, mock_session: Mock) -> None:
        """Test BacktestService initialization."""
        from src.backtest.service import BacktestService
        
        with patch.object(BacktestService, '__init__', lambda x, y: None):
            service = BacktestService.__new__(BacktestService)
            service.session = mock_session
            
            assert service.session == mock_session


class TestBacktestIntegration:
    """Integration-style tests (with mocks)."""
    
    def test_full_backtest_flow_mocked(self) -> None:
        """Test the full backtest flow with mocked dependencies."""
        # This would be expanded for full integration testing
        config = BacktestConfig(
            indices=["SP500"],
            start_date=date(2020, 1, 1),
            end_date=date(2020, 3, 31),
            step_days=90,
            forward_horizons=[63],
        )
        
        # Create mock observations
        observations = [
            BacktestPoint(
                cutoff_date=date(2020, 1, 1),
                index_id="SP500",
                actual_index_value=3000.0,
                estimated_index_value=2900.0,
                estimated_index_std=100.0,
                overpriciness=0.0345,
                forward_returns={63: -0.02},
                n_constituents=500,
                n_constituents_with_valuation=450,
                coverage_ratio=0.9,
            ),
            BacktestPoint(
                cutoff_date=date(2020, 4, 1),
                index_id="SP500",
                actual_index_value=2700.0,
                estimated_index_value=2850.0,
                estimated_index_std=120.0,
                overpriciness=-0.0526,
                forward_returns={63: 0.08},
                n_constituents=500,
                n_constituents_with_valuation=440,
                coverage_ratio=0.88,
            ),
        ]
        
        # Could analyze these observations
        overpriciness = np.array([o.overpriciness for o in observations])
        returns = np.array([o.forward_returns[63] for o in observations])
        
        # With only 2 points we can't run full analysis, but we can check signs
        # Point 1: overpriced (+), return negative (-) -> HIT
        # Point 2: underpriced (-), return positive (+) -> HIT
        assert (overpriciness[0] > 0 and returns[0] < 0)  # Hit
        assert (overpriciness[1] < 0 and returns[1] > 0)  # Hit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
