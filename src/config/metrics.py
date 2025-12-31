"""
Mispricing metrics configuration.

Provides multiple mispricing definitions that can be switched centrally.
Each metric has a name, formula description, and computation function.
"""

from enum import Enum
from typing import Callable, NamedTuple
import numpy as np


class MispricingDefinition(NamedTuple):
    """Definition of a mispricing metric."""
    name: str
    formula: str
    description: str
    compute: Callable[[np.ndarray, np.ndarray], np.ndarray]


def _ratio_mispricing(predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """
    Standard percentage mispricing: (predicted - actual) / actual
    
    Pros: Intuitive percentage interpretation
    Cons: Unstable for small caps, asymmetric, can produce extreme values
    """
    safe_actual = np.where(actual == 0, 1e-6, actual)
    return (predicted - actual) / safe_actual


def _log_ratio_mispricing(predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """
    Log ratio mispricing: log(predicted / actual)
    
    Pros: Symmetric, stable for small caps, bounded distribution
    Cons: Less intuitive interpretation
    """
    safe_actual = np.where(actual <= 0, 1e-6, actual)
    safe_predicted = np.where(predicted <= 0, 1e-6, predicted)
    return np.log(safe_predicted / safe_actual)


# Available mispricing metrics
RATIO = MispricingDefinition(
    name="ratio",
    formula="(predicted − actual) / actual",
    description="Standard percentage mispricing. Positive = undervalued.",
    compute=_ratio_mispricing,
)

LOG_RATIO = MispricingDefinition(
    name="log_ratio", 
    formula="log(predicted / actual)",
    description="Log ratio mispricing. More stable for extreme values.",
    compute=_log_ratio_mispricing,
)


class MispricingMetric(Enum):
    """Enum for selecting mispricing calculation method."""
    RATIO = RATIO
    LOG_RATIO = LOG_RATIO
    
    @property
    def definition(self) -> MispricingDefinition:
        return self.value


# === ACTIVE METRIC SELECTION ===
# Change this to swap mispricing definition
ACTIVE_MISPRICING_METRIC: MispricingMetric = MispricingMetric.RATIO


def compute_mispricing(
    predicted: np.ndarray, 
    actual: np.ndarray,
    metric: MispricingMetric = None
) -> np.ndarray:
    """
    Compute mispricing using the specified or active metric.
    
    Args:
        predicted: Predicted values (market cap)
        actual: Actual values (market cap)
        metric: Optional metric override. Uses ACTIVE_MISPRICING_METRIC if None.
        
    Returns:
        Mispricing values as numpy array
    """
    if metric is None:
        metric = ACTIVE_MISPRICING_METRIC
    return metric.definition.compute(predicted, actual)


def get_metric_info(metric: MispricingMetric = None) -> MispricingDefinition:
    """Get information about the specified or active metric."""
    if metric is None:
        metric = ACTIVE_MISPRICING_METRIC
    return metric.definition
