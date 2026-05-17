# SPDX-License-Identifier: MIT
"""ParetoFrontTracker — rate vs distortion frontier tracker for boosting.

Per `.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md`
§I.6 "Pareto-frontier-aware stack growth": each stack increment must
Pareto-improve `(rate, distortion)`; reject increments that improve one
at the cost of the other.

The tracker maintains a list of `(rate, distortion, source)` anchors and
exposes:
  - admits(rate, distortion): True if the candidate is admissible per
    the frontier (i.e. strictly dominates ≥1 anchor OR fills a gap).
  - track_anchor(rate, distortion, source): append to the frontier.
  - for_pareto_growth_filter(): returns a callable suitable as a pipeline
    `with_pareto_growth(...)` filter — the filter returns False on a
    candidate that worsens both axes vs the prior best.

Per CLAUDE.md "Apples-to-apples evidence discipline": the tracker carries
an `axis` label (`[contest-CUDA]` / `[contest-CPU]` / `[macOS-CPU advisory]`
/ `[proxy]`) and refuses cross-axis comparison. A future ParetoFrontTracker
construction with axis="[contest-CUDA]" cannot be queried with a
`[contest-CPU]` anchor without an explicit cross-axis bridge.

PV-6 verified ParetoFrontTracker is BUILT FROM SCRATCH (no upstream sister
in tac.sensitivity_map or tac.optimization).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from tac.boosting.errors import BoostingNamespaceError

__all__ = [
    "ParetoAnchor",
    "ParetoFrontTracker",
    "ParetoFrontTrackerError",
]


class ParetoFrontTrackerError(BoostingNamespaceError):
    """Raised on cross-axis anchor mixing or other tracker invariant
    violations."""


@dataclass(frozen=True)
class ParetoAnchor:
    """A single point on the rate-distortion frontier.

    Frozen so once an anchor is added it cannot be silently mutated; the
    tracker stores immutable history per CLAUDE.md HISTORICAL_PROVENANCE
    discipline (Catalog #110/#113 sister at the in-memory layer).
    """

    rate: float
    distortion: float
    source: str
    axis: str = "[proxy]"

    def __post_init__(self) -> None:
        if self.rate < 0:
            raise ParetoFrontTrackerError(
                f"rate={self.rate} must be >= 0 (rate is bytes-or-bits, "
                "always non-negative)"
            )
        if self.distortion < 0:
            raise ParetoFrontTrackerError(
                f"distortion={self.distortion} must be >= 0"
            )
        if not isinstance(self.source, str) or not self.source.strip():
            raise ParetoFrontTrackerError(
                f"source={self.source!r} must be a non-empty string"
            )
        if not isinstance(self.axis, str) or not self.axis.strip():
            raise ParetoFrontTrackerError(
                f"axis={self.axis!r} must be a non-empty string"
            )


# Apples-to-apples axis labels per CLAUDE.md non-negotiable.
_LEGAL_AXIS_LABELS: frozenset[str] = frozenset(
    {
        "[contest-CUDA]",
        "[contest-CPU]",
        "[macOS-CPU advisory]",
        "[MPS-PROXY]",
        "[proxy]",
        "[advisory only]",
        "[prediction]",
    }
)


@dataclass
class ParetoFrontTracker:
    """Tracks the rate-distortion Pareto frontier for a single axis.

    Per CLAUDE.md "Apples-to-apples evidence discipline": one tracker per
    axis. Cross-axis comparison is FORBIDDEN — mixing `[contest-CUDA]` and
    `[contest-CPU]` anchors silently produces invalid frontier inferences.

    Usage::

        tracker = ParetoFrontTracker(axis="[contest-CUDA]")
        tracker.track_anchor(rate=180_000, distortion=0.193, source="pr101_gold")
        tracker.track_anchor(rate=183_000, distortion=0.196, source="pr102")
        if tracker.admits(rate=181_000, distortion=0.194):
            # candidate strictly improves vs the frontier
            ...

    For pipeline integration::

        pipeline = (
            ComposableBoostingPipeline()
            | "raw_decoder"
            | "cascade_pose_residual"
        ).with_pareto_growth(reject_if_worsens_axis="rate")
    """

    axis: str
    _anchors: list[ParetoAnchor] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.axis not in _LEGAL_AXIS_LABELS:
            raise ParetoFrontTrackerError(
                f"axis={self.axis!r} not in {sorted(_LEGAL_AXIS_LABELS)}. Per "
                "CLAUDE.md 'Apples-to-apples evidence discipline' non-negotiable, "
                "every score must carry one of these axis labels inline."
            )

    @property
    def anchors(self) -> tuple[ParetoAnchor, ...]:
        """Return immutable snapshot of the tracker's anchors."""
        return tuple(self._anchors)

    def track_anchor(
        self, *, rate: float, distortion: float, source: str
    ) -> ParetoAnchor:
        """Append a new anchor to the frontier.

        The tracker does NOT prune dominated anchors automatically — history
        is preserved per HISTORICAL_PROVENANCE discipline. Use
        :meth:`pareto_optimal_anchors` to get only the non-dominated subset.
        """
        anchor = ParetoAnchor(
            rate=rate, distortion=distortion, source=source, axis=self.axis
        )
        self._anchors.append(anchor)
        return anchor

    def admits(self, *, rate: float, distortion: float) -> bool:
        """Return True if a candidate (rate, distortion) is admissible per
        the current frontier.

        Admissible means:
          (a) the frontier is empty (any candidate admits the first anchor)
          (b) the candidate strictly Pareto-dominates ≥1 existing anchor
              (lower rate AND lower-or-equal distortion, OR lower-or-equal
              rate AND lower distortion)
          (c) the candidate is non-dominated by EVERY existing anchor

        A candidate that is strictly dominated (higher rate AND higher
        distortion than at least one existing anchor) is REJECTED.
        """
        if not self._anchors:
            return True
        if rate < 0 or distortion < 0:
            return False
        # Reject if any existing anchor strictly dominates the candidate
        for a in self._anchors:
            if a.rate <= rate and a.distortion <= distortion and (
                a.rate < rate or a.distortion < distortion
            ):
                return False
        return True

    def pareto_optimal_anchors(self) -> tuple[ParetoAnchor, ...]:
        """Return the non-dominated subset of the frontier (sorted by rate)."""
        non_dominated: list[ParetoAnchor] = []
        for a in self._anchors:
            dominated = False
            for b in self._anchors:
                if a is b:
                    continue
                if (
                    b.rate <= a.rate
                    and b.distortion <= a.distortion
                    and (b.rate < a.rate or b.distortion < a.distortion)
                ):
                    dominated = True
                    break
            if not dominated:
                non_dominated.append(a)
        non_dominated.sort(key=lambda a: (a.rate, a.distortion))
        return tuple(non_dominated)

    def best_on_axis(self, *, axis: str) -> ParetoAnchor | None:
        """Return the single anchor minimizing the given axis (`'rate'` or
        `'distortion'`); None if the frontier is empty.
        """
        if axis not in ("rate", "distortion"):
            raise ParetoFrontTrackerError(
                f"axis={axis!r} must be 'rate' or 'distortion'"
            )
        if not self._anchors:
            return None
        key = (lambda a: a.rate) if axis == "rate" else (lambda a: a.distortion)
        return min(self._anchors, key=key)

    def for_pareto_growth_filter(
        self, *, reject_if_worsens_axis: str = "rate"
    ) -> Callable[[float, float], bool]:
        """Return a callable filter for pipeline `with_pareto_growth(...)`.

        The returned callable accepts ``(rate, distortion)`` and returns True
        if the candidate is admissible per the configured rejection axis.

        Semantics:
          - ``reject_if_worsens_axis='rate'``: candidate must have rate
            ≤ best_rate_on_frontier (distortion can vary freely)
          - ``reject_if_worsens_axis='distortion'``: candidate must have
            distortion ≤ best_distortion_on_frontier
          - ``reject_if_worsens_axis='both'``: candidate must be admissible
            per :meth:`admits` (Pareto-non-dominated)
        """
        if reject_if_worsens_axis not in ("rate", "distortion", "both"):
            raise ParetoFrontTrackerError(
                f"reject_if_worsens_axis={reject_if_worsens_axis!r} must be "
                f"'rate', 'distortion', or 'both'"
            )

        def _filter(rate: float, distortion: float) -> bool:
            if not self._anchors:
                return True
            if reject_if_worsens_axis == "both":
                return self.admits(rate=rate, distortion=distortion)
            if reject_if_worsens_axis == "rate":
                best = min(self._anchors, key=lambda a: a.rate)
                return rate <= best.rate
            # distortion
            best = min(self._anchors, key=lambda a: a.distortion)
            return distortion <= best.distortion

        return _filter

    def to_dict(self) -> dict[str, Any]:
        """Serialize the tracker state for JSON-roundtrip / persistence."""
        return {
            "axis": self.axis,
            "anchors": [
                {
                    "rate": a.rate,
                    "distortion": a.distortion,
                    "source": a.source,
                    "axis": a.axis,
                }
                for a in self._anchors
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParetoFrontTracker:
        """Reconstruct a tracker from a dict (e.g. JSON round-trip)."""
        tracker = cls(axis=data["axis"])
        for entry in data.get("anchors", []):
            tracker._anchors.append(
                ParetoAnchor(
                    rate=entry["rate"],
                    distortion=entry["distortion"],
                    source=entry["source"],
                    axis=entry.get("axis", data["axis"]),
                )
            )
        return tracker
