"""Minimal task-aware codec foundation for scorer, architecture, and artifact records.

This package keeps the interfaces intentionally small:

- scorer registration wraps callables with serializable metadata
- architecture registration points at existing experiment entrypoints
- quantization metadata reads best-meta and int8 artifact sidecars
- evaluation records normalize current-workflow style JSON summaries and logs
"""

from .architectures import (
    ArchitectureConfig,
    ArchitectureRegistry,
    ArchitectureSpec,
    register_default_architectures,
)
from .quantization import QuantizationMetadata
from .state import FinalMetadata, ResumeState
from .records import EvaluationRecord, ProxyEvaluationRecord
from .scorers import RegisteredScorer, ScorerRegistry, ScorerSpec

__all__ = [
    "ArchitectureConfig",
    "ArchitectureRegistry",
    "ArchitectureSpec",
    "EvaluationRecord",
    "FinalMetadata",
    "ProxyEvaluationRecord",
    "QuantizationMetadata",
    "ResumeState",
    "RegisteredScorer",
    "ScorerRegistry",
    "ScorerSpec",
    "register_default_architectures",
]
