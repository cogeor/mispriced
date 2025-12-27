"""BacktestService - orchestrates index overpriciness backtesting."""

import logging
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np
from sqlalchemy.orm import Session

from src.db.repositories.index_repo import IndexRepository, IndexMembershipRepository
from src.db.models.valuation import ValuationResult
from src.db.models.index import Index
from src.index.service import IndexService

from .models import BacktestConfig, BacktestPoint, BacktestResult, CorrelationMetrics
from .index_prices import get_index_return
from .analysis import analyze_correlation

logger = logging.getLogger(__name__)


class BacktestService:
    """Orchestrate index overpriciness backtesting.
    
    This service:
    1. Iterates through cutoff dates
    2. For each cutoff, computes index overpriciness using only pre-cutoff data
    3. Fetches forward returns
    4. Analyzes correlation between overpriciness and returns
    """
    
    def __init__(self, session: Session):
        """
        Initialize BacktestService.
        
        Args:
            session: SQLAlchemy session for database access
        """
        self.session = session
        self.index_repo = IndexRepository(session)
        self.membership_repo = IndexMembershipRepository(session)
        self.index_service = IndexService(session)
    
    def run_backtest(self, config: BacktestConfig) -> BacktestResult:
        """
        Run full walk-forward backtest.
        
        Args:
            config: Backtest configuration
            
        Returns:
            BacktestResult with all observations and analysis
        """
        logger.info(
            f"Starting backtest: {config.indices} from {config.start_date} to {config.end_date}"
        )
        
        observations: List[BacktestPoint] = []
        
        # Iterate through cutoff dates
        cutoff = config.start_date
        while cutoff <= config.end_date:
            for index_id in config.indices:
                try:
                    point = self._run_single_point(
                        cutoff_date=cutoff,
                        index_id=index_id,
                        horizons=config.forward_horizons,
                        model_version=config.model_version,
                        min_coverage=config.min_coverage,
                        use_latest_valuations=config.use_latest_valuations,
                    )
                    if point:
                        observations.append(point)
                        logger.debug(
                            f"  {index_id} @ {cutoff}: overpriciness={point.overpriciness:.2%}"
                        )
                except Exception as e:
                    logger.warning(f"Failed {index_id} at {cutoff}: {e}")
            
            cutoff += timedelta(days=config.step_days)
        
        logger.info(f"Collected {len(observations)} observations")
        
        # Analyze correlations for each horizon
        correlations = self._analyze_all_correlations(
            observations, config.forward_horizons
        )
        
        # Compute summary statistics
        if observations:
            overpriciness_values = [o.overpriciness for o in observations]
            coverages = [o.coverage_ratio for o in observations]
            mean_op = float(np.mean(overpriciness_values))
            std_op = float(np.std(overpriciness_values))
            mean_cov = float(np.mean(coverages))
        else:
            mean_op, std_op, mean_cov = None, None, None
        
        return BacktestResult(
            start_date=config.start_date,
            end_date=config.end_date,
            indices_tested=config.indices,
            forward_horizons=config.forward_horizons,
            model_version=config.model_version,
            observations=observations,
            n_observations=len(observations),
            correlations=correlations,
            mean_overpriciness=mean_op,
            std_overpriciness=std_op,
            mean_coverage=mean_cov,
        )
    
    def _run_single_point(
        self,
        cutoff_date: date,
        index_id: str,
        horizons: List[int],
        model_version: Optional[str],
        min_coverage: float,
        use_latest_valuations: bool = False,
    ) -> Optional[BacktestPoint]:
        """
        Run single backtest observation for one index at one cutoff.
        
        Args:
            cutoff_date: Point-in-time cutoff
            index_id: Index to evaluate
            horizons: Forward horizons in days
            model_version: Model version filter
            min_coverage: Minimum required coverage ratio
            use_latest_valuations: If True, use latest valuations ignoring cutoff
            
        Returns:
            BacktestPoint or None if insufficient data
        """
        # 1. Get index definition
        index_def = self.index_repo.get_index(index_id)
        if not index_def:
            logger.warning(f"Index not found: {index_id}")
            return None
        
        # 2. Determine as_of_time for valuation lookup
        from datetime import datetime
        
        if use_latest_valuations:
            # Use latest valuations regardless of cutoff date
            as_of_time = None
        else:
            # Strict: only use valuations from before cutoff
            as_of_time = datetime.combine(cutoff_date, datetime.min.time())
        
        # 3. Compute index values using existing index service
        try:
            index_result = self.index_service.compute_index(
                index_id=index_id,
                as_of_time=as_of_time,
                model_version=model_version,
            )
        except Exception as e:
            logger.warning(f"Could not compute index {index_id} at {cutoff_date}: {e}")
            return None
        
        if index_result is None:
            return None
        
        # Check coverage
        coverage = index_result.n_tickers_with_valuation / max(index_result.n_tickers, 1)
        if coverage < min_coverage:
            logger.debug(
                f"Insufficient coverage for {index_id} at {cutoff_date}: "
                f"{coverage:.1%} < {min_coverage:.1%}"
            )
            return None
        
        # 4. Compute overpriciness
        # overpriciness = (actual - estimated) / estimated
        if index_result.estimated_index == 0:
            return None
        
        overpriciness = (
            (index_result.actual_index - index_result.estimated_index)
            / index_result.estimated_index
        )
        
        # Z-score (if we have std)
        z_score = None
        if index_result.estimated_index_std > 0:
            z_score = (
                (index_result.actual_index - index_result.estimated_index)
                / index_result.estimated_index_std
            )
        
        # 5. Fetch forward returns for each horizon
        forward_returns: Dict[int, float] = {}
        for horizon in horizons:
            end_date = cutoff_date + timedelta(days=horizon)
            ret = get_index_return(index_id, cutoff_date, end_date)
            if ret is not None:
                forward_returns[horizon] = ret
        
        return BacktestPoint(
            cutoff_date=cutoff_date,
            index_id=index_id,
            actual_index_value=index_result.actual_index,
            estimated_index_value=index_result.estimated_index,
            estimated_index_std=index_result.estimated_index_std,
            overpriciness=overpriciness,
            overpriciness_z_score=z_score,
            forward_returns=forward_returns,
            n_constituents=index_result.n_tickers,
            n_constituents_with_valuation=index_result.n_tickers_with_valuation,
            coverage_ratio=coverage,
            model_version=model_version,
        )
    
    def _analyze_all_correlations(
        self,
        observations: List[BacktestPoint],
        horizons: List[int],
    ) -> Dict[int, CorrelationMetrics]:
        """
        Analyze correlations for all horizons.
        
        Args:
            observations: List of backtest points
            horizons: Forward horizons to analyze
            
        Returns:
            Dictionary mapping horizon -> CorrelationMetrics
        """
        correlations: Dict[int, CorrelationMetrics] = {}
        
        for horizon in horizons:
            # Extract pairs with valid data for this horizon
            pairs = [
                (o.overpriciness, o.forward_returns.get(horizon))
                for o in observations
                if horizon in o.forward_returns and o.forward_returns[horizon] is not None
            ]
            
            if len(pairs) < 10:
                logger.warning(
                    f"Insufficient data for horizon {horizon}: {len(pairs)} observations"
                )
                continue
            
            overpriciness = np.array([p[0] for p in pairs])
            returns = np.array([p[1] for p in pairs])
            
            try:
                metrics = analyze_correlation(overpriciness, returns, horizon)
                correlations[horizon] = metrics
                
                logger.info(
                    f"Horizon {horizon}d: r={metrics.pearson_r:.3f} (p={metrics.pearson_p_value:.3f}), "
                    f"hit_rate={metrics.hit_rate:.1%}"
                )
            except Exception as e:
                logger.warning(f"Correlation analysis failed for horizon {horizon}: {e}")
        
        return correlations
