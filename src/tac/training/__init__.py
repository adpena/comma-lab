# SPDX-License-Identifier: MIT
"""tac.training - training-side helpers shared across substrates.

This package is the canonical home for training-loop helpers (hooks,
losses, schedulers) that are reused across substrate trainers. Each
module declares its own canonical public surface; consumers import
specific symbols rather than the package wildcard.

Modules:
- ``streaming_master_gradient_hook`` - SLOT MG-5 streaming sample hook
  that registers per-N-epoch master-gradient samples in the canonical
  fcntl-locked ledger at ``.omx/state/streaming_predictions.jsonl``
- ``score_weighted_reconstruction_loss`` - SLOT MG-7 sister exploit-2
  reconstruction loss helper (sister territory; see MG-7-BUNDLE)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

_LEGACY_MODULE_NAME = "tac._training_legacy"
_LEGACY_PUBLIC = {
    "EMA",
    "KalmanWeightFilter",
    "ScorerLossConvergenceDetector",
    "SWA",
    "TrainConfig",
    "Trainer",
}
__all__ = sorted(_LEGACY_PUBLIC)

_legacy_module: ModuleType | None = None


def _load_legacy_training_module() -> ModuleType:
    """Load the sibling ``training.py`` module without shadowing this package."""

    global _legacy_module
    if _legacy_module is not None:
        return _legacy_module

    cached = sys.modules.get(_LEGACY_MODULE_NAME)
    if cached is not None:
        _legacy_module = cached
        return cached

    legacy_path = Path(__file__).resolve().parents[1] / "training.py"
    spec = importlib.util.spec_from_file_location(_LEGACY_MODULE_NAME, legacy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load legacy training module from {legacy_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[_LEGACY_MODULE_NAME] = module
    spec.loader.exec_module(module)
    _legacy_module = module
    return module


def __getattr__(name: str) -> Any:
    if name not in _LEGACY_PUBLIC:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name == "ScorerLossConvergenceDetector":
        from tac.scorer_loss_convergence_detector import (
            ScorerLossConvergenceDetector,
        )

        globals()[name] = ScorerLossConvergenceDetector
        return ScorerLossConvergenceDetector
    value = getattr(_load_legacy_training_module(), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LEGACY_PUBLIC)
