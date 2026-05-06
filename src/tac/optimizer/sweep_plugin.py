"""Plugin interface for closed-loop sweep candidate generators.

`tools/feedback_loop_sweep.py` was originally apogee_intN-specific: the
candidate generator and dispatch shell-out were both hard-coded. This module
extracts the contract so anyone can plug in a new sweep workload without
modifying the loop driver.

A sweep plugin is a `CandidateGenerator` subclass that:
  - returns a list of candidate dicts (the rank-then-dispatch contract), and
  - knows how to build a dispatch command for one candidate.

Plugins register themselves with `register_generator(name, factory)`. The
loop driver loads them by name from the registry. This makes the closed-loop
driver itself generic — it can drive any rank-then-rank-more workload.

Per CLAUDE.md non-negotiables:
  - The plugin returns candidates with `ready_for_exact_eval_dispatch=False`
    by default. Real readiness comes from exact-SHA non-proxy evidence.
  - The plugin's dispatch command runs through the same predispatch ladder.
  - Synthetic / non-comma plugins MUST mark `evidence_semantics` as a
    non-prediction value (e.g. `"synthetic_test"`) so the existing dispatch
    blockers do not silently filter them at level 1+ workloads.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# A "candidate" is a free-form dict; we document the canonical fields here.
# Required fields:
#   candidate_id        : str   — globally unique within a sweep cycle
#   archive_bytes       : int   — absolute byte size (used by ranker + reseeder)
# Optional fields used by the loop driver:
#   archive_path        : Path  — where the bytes live (for SHA verification)
#   archive_sha256      : str   — exact SHA (required for paid dispatch)
#   rel_err_pct         : float — local proxy of distortion (used for ranking)
#   n_layers            : int   — workload-specific telemetry (passed through)
#   lane_class          : str   — workload tag (passed through to anchors)
#   ready_for_exact_eval_dispatch : bool — default False, set True ONLY when
#                                          exact-SHA non-proxy evidence exists
#   evidence_semantics  : str   — one of "byte_only_forensic", "synthetic_test",
#                                 "exact_sha_anchored", etc. The loop driver
#                                 blocks dispatch for any value containing
#                                 "prediction" or "proxy".
#   dispatch_blockers   : list[str] — explicit blockers (passed through)
#   score_claim         : bool  — does this candidate claim a contest-CUDA score?
#   score_claim_verified: bool  — has it been verified?
Candidate = dict[str, Any]


@dataclass
class DispatchSpec:
    """The minimum a generator must yield to fire one dispatch."""

    label: str
    cmd: list[str]
    estimated_cost_usd: float = 0.30
    timeout_seconds: float = 1800.0
    cwd: Path | None = None
    env: dict[str, str] = field(default_factory=dict)


class CandidateGenerator(ABC):
    """Plugin contract: produce candidates and build dispatch commands.

    Subclasses MUST implement `__call__()` and `build_dispatch()`.
    """

    name: str = "unnamed"  # subclasses override

    @abstractmethod
    def __call__(self) -> list[Candidate]:
        """Return the current snapshot of candidates."""
        raise NotImplementedError

    @abstractmethod
    def build_dispatch(self, candidate: Candidate, *, label: str) -> DispatchSpec:
        """Return the dispatch spec for one candidate at a chosen label."""
        raise NotImplementedError

    # Optional hooks — base implementations are sensible defaults.
    def candidate_dispatch_blockers(self, candidate: Candidate) -> list[str]:
        """Return per-candidate blockers above the loop-driver baseline.

        The base implementation is empty. Override to add workload-specific
        gates (e.g., "missing required preprocessor output").
        """
        return []

    def expected_artifacts(self, label: str) -> list[Path]:
        """Return artifacts the harvester should look for. Optional."""
        return []


# ── Registry ──────────────────────────────────────────────────────────────

GeneratorFactory = Callable[[], CandidateGenerator]
_REGISTRY: dict[str, GeneratorFactory] = {}


def register_generator(name: str, factory: GeneratorFactory) -> None:
    """Register a generator factory under `name`. Re-registration replaces."""
    if not name or not isinstance(name, str):
        raise ValueError(f"generator name must be non-empty str, got {name!r}")
    _REGISTRY[name] = factory


def unregister_generator(name: str) -> None:
    """Remove a registered factory (no-op if absent). Mostly for tests."""
    _REGISTRY.pop(name, None)


def list_generators() -> list[str]:
    """Return registered generator names, sorted."""
    return sorted(_REGISTRY)


def load_generator(name: str) -> CandidateGenerator:
    """Instantiate the registered generator named `name`.

    Raises `KeyError` if `name` is not registered.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown candidate generator: {name!r}; registered: {list_generators()}"
        )
    return _REGISTRY[name]()


def reset_registry() -> None:
    """Clear the in-process registry. For test isolation only."""
    _REGISTRY.clear()
