"""LeakAuditKit: audit ML/LLM evaluations for data leakage and shortcut learning."""

__version__ = "0.2.0"

from .detectors.base import BaseDetector, ContaminationMatch, DetectionResult
from .detectors.ngram import NgramDetector

__all__ = ["BaseDetector", "ContaminationMatch", "DetectionResult", "NgramDetector", "__version__"]