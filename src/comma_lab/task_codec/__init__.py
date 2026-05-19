"""Legacy comma_lab.task_codec compatibility foundation for early post-filter records.

This is not a new-code namespace and it is not the expansion of TAC. New
compression primitives belong in the Task-Aware Compression library under
``src/tac``.

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
from .records import EvaluationRecord, ProxyEvaluationRecord
from .scorers import RegisteredScorer, ScorerRegistry, ScorerSpec
from .state import FinalMetadata, ResumeState

__all__ = [
    "ArchitectureConfig",
    "ArchitectureRegistry",
    "ArchitectureSpec",
    "EvaluationRecord",
    "FinalMetadata",
    "ProxyEvaluationRecord",
    "QuantizationMetadata",
    "RegisteredScorer",
    "ResumeState",
    "ScorerRegistry",
    "ScorerSpec",
    "register_default_architectures",
]
