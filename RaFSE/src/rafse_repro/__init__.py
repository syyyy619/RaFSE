"""RaFSE implementation for University-1652 retrieval."""

from .dynamic_k import DynamicKThresholds, assign_dynamic_k, calibrate_thresholds
from .metrics import evaluate_top200

__all__ = [
    "DynamicKThresholds",
    "assign_dynamic_k",
    "calibrate_thresholds",
    "evaluate_top200",
]

__version__ = "0.1.0"
