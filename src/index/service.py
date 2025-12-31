"""IndexService - computes index-level values from per-ticker valuations."""

import logging
import math
from typing import Dict, List, Optional
from datetime import datetime
from sqlalchemy.orm import Session

from src.db.models.index import Index
from src.db.models.valuation import ValuationResult
from src.db.repositories.index_repo import IndexRepository, IndexMembershipRepository
from .models import IndexAnalysis, IndexResult, WeightingScheme
from .weights import (
    compute_equal_weights,
    compute_market_cap_weights,
    normalize_custom_weights,
)

logger = logging.getLogger(__name__)


class IndexService:
    """
    Compute index-level values from per-ticker valuations.

    This service:
    1. Pulls existing valuation results from DB for tickers in an index
    2. Computes estimated index value (weighted sum of predicted market caps) with std
    3. Computes actual index value (weighted sum of actual market caps)
    4. Returns results in-memory (no persistence for now)
    """

    def __init__(self, session: Session):
        """
        Initialize IndexService with database session.

        Args:
            session: SQLAlchemy session for database access
        """
        self.session = session
        self.index_repo = IndexRepository(session)
        self.membership_repo = IndexMembershipRepository(session)

    def compute_index(
        self,
        index_id: str,
        as_of_time: Optional[datetime] = None,
        model_version: Optional[str] = None,
        custom_weights: Optional[Dict[str, float]] = None,
    ) -> IndexResult:
        """
        Compute index values at a specific time.

        Args:
            index_id: Index to compute (e.g., "SP500")
            as_of_time: Snapshot time (default: use latest valuations)
            model_version: Filter valuations by model version
            custom_weights: Override index weighting scheme with custom weights

        Returns:
            IndexResult with actual, estimated, and uncertainty values

        Raises:
            ValueError: If index not found or no valuations available
        """
        logger.info(f"Computing index {index_id} as of {as_of_time}")

        # 1. Get index definition
        index_def = self.index_repo.get_index(index_id)
        if not index_def:
            raise ValueError(f"Index '{index_id}' not found")

        # 2. Get index members
        tickers = self.membership_repo.get_members(index_id, as_of_time)
        if not tickers:
            raise ValueError(f"No members found for index '{index_id}'")

        logger.info(f"Index {index_id} has {len(tickers)} members")

        # 3. Get valuations for members
        valuations = self._get_valuations(tickers, as_of_time, model_version)
        if not valuations:
            raise ValueError(
                f"No valuations found for index '{index_id}' members"
            )

        logger.info(
            f"Found valuations for {len(valuations)}/{len(tickers)} tickers"
        )

        # 4. Compute weights
        if custom_weights:
            weights = normalize_custom_weights(custom_weights)
            weighting_scheme = WeightingScheme.CUSTOM
        else:
            weighting_scheme = WeightingScheme(
                index_def.weighting_scheme or "equal"
            )
            weights = self._compute_weights(valuations, weighting_scheme)

        # 5. Aggregate into index values
        return self._aggregate_index(
            index_def=index_def,
            valuations=valuations,
            weights=weights,
            weighting_scheme=weighting_scheme,
            n_total_tickers=len(tickers),
            model_version=model_version,
        )

    def _get_valuations(
        self,
        tickers: List[str],
        as_of_time: Optional[datetime],
        model_version: Optional[str],
    ) -> List[ValuationResult]:
        """Fetch valuation results for the given tickers."""
        query = self.session.query(ValuationResult).filter(
            ValuationResult.ticker.in_(tickers)
        )

        if as_of_time:
            query = query.filter(
                ValuationResult.snapshot_timestamp <= as_of_time
            )

        if model_version:
            query = query.filter(ValuationResult.model_version == model_version)

        # Get most recent valuation per ticker
        # This is a simplified approach - for production, use window functions
        valuations = query.all()

        # Deduplicate: keep most recent per ticker
        latest_by_ticker: Dict[str, ValuationResult] = {}
        for v in valuations:
            existing = latest_by_ticker.get(v.ticker)
            if not existing or v.snapshot_timestamp > existing.snapshot_timestamp:
                latest_by_ticker[v.ticker] = v

        return list(latest_by_ticker.values())

    def _compute_weights(
        self,
        valuations: List[ValuationResult],
        scheme: WeightingScheme,
    ) -> Dict[str, float]:
        """Compute ticker weights based on scheme."""
        if scheme == WeightingScheme.EQUAL:
            return compute_equal_weights([v.ticker for v in valuations])
        elif scheme == WeightingScheme.MARKET_CAP:
            return compute_market_cap_weights(valuations)
        else:
            raise ValueError(f"Unknown weighting scheme: {scheme}")

    def _aggregate_index(
        self,
        index_def: Index,
        valuations: List[ValuationResult],
        weights: Dict[str, float],
        weighting_scheme: WeightingScheme,
        n_total_tickers: int,
        model_version: Optional[str],
    ) -> IndexResult:
        """Aggregate valuations into index-level metrics."""
        # Sum weighted values
        actual_sum = 0.0
        estimated_sum = 0.0
        variance_sum = 0.0

        # Determine as_of_time from valuations
        as_of_time = max(v.snapshot_timestamp for v in valuations)

        for v in valuations:
            w = weights.get(v.ticker, 0.0)
            if w == 0:
                continue

            actual_mcap = float(v.actual_mcap) if v.actual_mcap else 0
            predicted_mcap = float(v.predicted_mcap_mean) if v.predicted_mcap_mean else 0
            predicted_std = float(v.predicted_mcap_std) if v.predicted_mcap_std else 0

            actual_sum += w * actual_mcap
            estimated_sum += w * predicted_mcap
            variance_sum += (w * predicted_std) ** 2

        # Propagated uncertainty (assuming independent errors)
        estimated_std = math.sqrt(variance_sum)

        # Relative error
        if actual_sum != 0:
            relative_error = (estimated_sum - actual_sum) / actual_sum
        else:
            relative_error = 0.0

        return IndexResult(
            index_id=index_def.index_id,
            as_of_time=as_of_time,
            actual_index=actual_sum,
            estimated_index=estimated_sum,
            estimated_index_std=estimated_std,
            index_relative_error=relative_error,
            n_tickers=n_total_tickers,
            n_tickers_with_valuation=len(valuations),
            weights=weights,
            model_version=model_version,
            weighting_scheme=weighting_scheme,
        )

    def compute_index_series(
        self,
        index_id: str,
        start_time: datetime,
        end_time: datetime,
        model_version: Optional[str] = None,
    ) -> List[IndexResult]:
        """
        Compute index values over a time range.

        This iterates through available snapshots and computes
        index values at each point.

        Args:
            index_id: Index to compute
            start_time: Start of time range
            end_time: End of time range
            model_version: Filter by model version

        Returns:
            List of IndexResult, one per snapshot timestamp
        """
        # Get all unique snapshot timestamps in range
        timestamps = (
            self.session.query(ValuationResult.snapshot_timestamp)
            .filter(
                ValuationResult.snapshot_timestamp >= start_time,
                ValuationResult.snapshot_timestamp <= end_time,
            )
            .distinct()
            .order_by(ValuationResult.snapshot_timestamp)
            .all()
        )

        results = []
        for (ts,) in timestamps:
            try:
                result = self.compute_index(
                    index_id=index_id,
                    as_of_time=ts,
                    model_version=model_version,
                )
                results.append(result)
            except ValueError as e:
                logger.warning(f"Skipping timestamp {ts}: {e}")
                continue

        return results

    def analyze_index(
        self,
        index_id: str,
        official_count: int = 0,
        as_of_date: Optional[datetime] = None,
        model_version: Optional[str] = None,
    ) -> Optional[IndexAnalysis]:
        """
        Analyze index mispricing for dashboard display.

        Uses cap-weighted aggregation of actual vs predicted market caps.

        Args:
            index_id: Index to analyze (e.g., "SP500")
            official_count: Official number of constituents (for coverage display)
            as_of_date: If provided, only use valuations from this specific date
            model_version: If provided, only use valuations from this model version

        Returns:
            IndexAnalysis with mispricing metrics, or None if no data
        """
        from src.db.models.index import IndexMembership

        query = self.session.query(
            ValuationResult.ticker,
            ValuationResult.relative_error,
            ValuationResult.residual_error,
            ValuationResult.actual_mcap,
            ValuationResult.predicted_mcap_mean,
            ValuationResult.snapshot_timestamp,
        ).join(
            IndexMembership,
            IndexMembership.ticker == ValuationResult.ticker,
        ).filter(
            IndexMembership.index_id == index_id
        )
        
        # If as_of_date provided, filter to that specific quarter
        if as_of_date:
            query = query.filter(ValuationResult.snapshot_timestamp == as_of_date)
            
        # If model_version provided, filter to that specific version
        if model_version:
            query = query.filter(ValuationResult.model_version == model_version)

        import pandas as pd

        results = pd.read_sql(query.statement, self.session.bind)

        if results.empty:
            logger.warning(f"No valuation results for {index_id}")
            return None

        # Keep only the latest valuation per ticker
        results = results.sort_values('snapshot_timestamp', ascending=False)
        results = results.drop_duplicates(subset=["ticker"], keep='first')
        covered_count = len(results)

        total_actual = float(results["actual_mcap"].sum())
        total_predicted = float(results["predicted_mcap_mean"].sum())

        if total_actual == 0:
            logger.warning(f"Total actual market cap is 0 for {index_id}")
            return None

        mispricing = (total_predicted - total_actual) / total_actual
        
        # Compute residual (size-corrected) Index Mispricing
        # We define this as the cap-weighted average of individual residual errors
        # residual_error = relative_error - size_bias
        # Index Residual = Sum(w_i * residual_error_i) where w_i = cap_i / total_cap
        
        residual_mispricing = None
        if "residual_error" in results.columns and not results["residual_error"].isna().all():
            # Fill NA with relative_error if some are missing (partial coverage)
            # Or just drop? Let's fill with relative_error (assuming 0 bias if missing)
            res_errors = results["residual_error"].fillna(results["relative_error"])
            weights = results["actual_mcap"] / total_actual
            residual_mispricing = float((weights * res_errors).sum())

        status = "UNDERPRICED" if mispricing > 0 else "OVERPRICED"

        return IndexAnalysis(
            index=index_id,
            mispricing=mispricing,
            residual_mispricing=residual_mispricing,
            status=status,
            total_actual=total_actual,
            total_predicted=total_predicted,
            count=covered_count,
            official_count=official_count or covered_count,
        )

    def analyze_all_indices(
        self,
        index_ids: Optional[List[str]] = None,
    ) -> List[IndexAnalysis]:
        """
        Analyze mispricing for multiple indices using the latest quarter with data.

        Args:
            index_ids: List of index IDs to analyze.
                       If None, analyzes all indices in the database.

        Returns:
            List of IndexAnalysis results (excludes indices with no data)
        """
        if index_ids is None:
            # Get all indices from database
            indices = self.session.query(Index.index_id).all()
            index_ids = [idx[0] for idx in indices]

        # Find the latest quarter date with significant valuations
        from sqlalchemy import func
        from src.db.models.index import IndexMembership
        
        last_quarter = (
            self.session.query(ValuationResult.snapshot_timestamp)
            .group_by(ValuationResult.snapshot_timestamp)
            .having(func.count(func.distinct(ValuationResult.ticker)) > 1000)
            .order_by(ValuationResult.snapshot_timestamp.desc())
            .first()
        )
        
        last_quarter_date = last_quarter[0] if last_quarter else None
        
        # Find dominant model version for this quarter
        model_version = None
        if last_quarter_date:
            most_common_version = (
                self.session.query(ValuationResult.model_version)
                .filter(ValuationResult.snapshot_timestamp == last_quarter_date)
                .group_by(ValuationResult.model_version)
                .order_by(func.count(ValuationResult.ticker).desc())
                .first()
            )
            model_version = most_common_version[0] if most_common_version else None
            
        logger.info(f"Using last quarter date: {last_quarter_date}, model: {model_version}")

        # Get official counts from memberships (unique tickers per index)
        counts_query = (
            self.session.query(
                IndexMembership.index_id,
                func.count(func.distinct(IndexMembership.ticker)).label("count"),
            )
            .filter(IndexMembership.index_id.in_(index_ids))
            .group_by(IndexMembership.index_id)
        )
        official_counts = {row[0]: row[1] for row in counts_query.all()}

        results = []
        for index_id in index_ids:
            official_count = official_counts.get(index_id, 0)
            analysis = self.analyze_index(
                index_id, 
                official_count, 
                as_of_date=last_quarter_date,
                model_version=model_version
            )
            if analysis:
                results.append(analysis)

        return results
