from sqlalchemy import Column, String, Integer, Numeric, DateTime, JSON, ForeignKey, func, Index
from sqlalchemy.dialects.postgresql import UUID
from .base import Base

class ValuationResult(Base):
    __tablename__ = "valuation_results"
    
    ticker = Column(String(20), primary_key=True)
    snapshot_timestamp = Column(DateTime, primary_key=True)
    model_version = Column(String(50), primary_key=True)
    
    # Prediction outputs
    predicted_mcap_mean = Column(Numeric, nullable=False)
    # DEPRECATED: kept for backwards-compat (existing rows). Use predicted_mcap_lo/hi (CQR) for uncertainty.
    predicted_mcap_std = Column(Numeric, nullable=False)
    actual_mcap = Column(Numeric, nullable=False)
    relative_error = Column(Numeric, nullable=False)  # Raw mispricing
    residual_error = Column(Numeric, nullable=True)   # Size-corrected mispricing (nullable for backfill)
    relative_std = Column(Numeric, nullable=False)

    # CQR (Conformalized Quantile Regression) 90% prediction interval bounds in raw mcap space.
    # Populated by run_cv_plus_cqr in src/valuation/conformal.py; NULL on rows predating loop 02.
    predicted_mcap_lo = Column(Numeric, nullable=True)
    predicted_mcap_hi = Column(Numeric, nullable=True)
    conformal_alpha = Column(Numeric, nullable=True)  # e.g. 0.1 for a 90% interval
    
    # Model metadata
    model_config_hash = Column(String(64))
    n_experiments = Column(Integer, nullable=False)
    experiment_metadata = Column(JSON)
    
    # Data versioning
    data_version_id = Column(UUID(as_uuid=True), ForeignKey("data_versions.version_id"))
    
    # Audit
    computed_at = Column(DateTime, server_default=func.now())
    
    __table_args__ = (
        Index('idx_valuation_error', 'relative_error'),
        Index('idx_valuation_model', 'model_version'),
        Index('idx_valuation_data_version', 'data_version_id'),
    )
