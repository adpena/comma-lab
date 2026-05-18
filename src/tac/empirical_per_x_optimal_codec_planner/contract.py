# SPDX-License-Identifier: MIT
"""Typed contracts for the per-X optimal codec planner.

[verified-against: .omx/research/empirical_per_x_optimal_codec_planner_plus_duckdb_canonical_unification_20260518.md Section 5.2]
[verified-against: Catalog #265 canonical-contract tokens]
[verified-against: Catalog #287 evidence-tag discipline]
[verified-against: Catalog #323 canonical Provenance contract]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tac.empirical_per_x_optimal_codec_planner.codec_menu import CODEC_NAMES


X_GRANULARITY_VALUES: frozenset[str] = frozenset({
    "byte",
    "bit",
    "pixel",
    "region",
    "pair",
    "frame",
    "boundary",
    "latent_index",
    "channel",
    "tensor",
    "layer",
})
"""Canonical X-granularity values for the per-X planner."""


class PlannerError(RuntimeError):
    """Raised when the per-X planner cannot produce a valid plan."""


@dataclass(frozen=True)
class PerXAssignmentRow:
    """One row of the per-X codec assignment plan.

    Each row maps ONE X-unit (byte / pair / region / etc.) to ONE codec from the
    codec_menu, with predicted ΔS contribution + predicted bytes after codec.
    """
    x_index: int
    x_class: str                              # sensitivity class: top_2pct / top_5pct / top_20pct / tail
    sensitivity_score: float
    chosen_codec: str                         # must be in CODEC_NAMES
    chosen_codec_bits: int | None             # e.g. 16 for fp16, 4 for int4
    predicted_score_delta: float              # per-row predicted ΔS contribution
    predicted_bytes_after_codec: int

    def __post_init__(self) -> None:
        if self.chosen_codec not in CODEC_NAMES:
            raise PlannerError(
                f"chosen_codec {self.chosen_codec!r} not in canonical CODEC_NAMES"
            )
        if self.x_index < 0:
            raise PlannerError(f"x_index must be >= 0; got {self.x_index}")
        if self.predicted_bytes_after_codec < 0:
            raise PlannerError(
                f"predicted_bytes_after_codec must be >= 0; got {self.predicted_bytes_after_codec}"
            )


@dataclass(frozen=True)
class PerXCodecAssignmentPlan:
    """Typed assignment plan output by the per-X optimal codec planner.

    Per Catalog #287 every score_delta field is `predicted` evidence_grade UNTIL
    paired empirical anchor materializes.

    Per Catalog #323 carries canonical Provenance sub-object validated via
    `tac.provenance.validate_provenance`.
    """
    archive_sha256: str
    x_granularity: str                        # one of X_GRANULARITY_VALUES
    codec_menu: tuple[str, ...]               # subset of CODEC_NAMES
    byte_budget: int
    sensitivity_threshold_quantiles: tuple[float, ...]
    assignments: tuple[PerXAssignmentRow, ...]
    total_predicted_score_delta: float
    total_predicted_bytes: int
    total_predicted_bytes_within_budget: bool
    operating_point: dict                     # {d_seg, d_pose, rate, score}
    measurement_axis: str                     # [contest-CPU] / [contest-CUDA] / [macos-cpu-advisory]
    evidence_grade: str                       # 'predicted' (forced; cannot be promoted at construction)
    provenance: dict                          # canonical Provenance per Catalog #323
    captured_at_utc: str
    schema_version: str = "per_x_codec_assignment_plan_v1"

    def __post_init__(self) -> None:
        if self.x_granularity not in X_GRANULARITY_VALUES:
            raise PlannerError(
                f"x_granularity {self.x_granularity!r} not in {sorted(X_GRANULARITY_VALUES)}"
            )
        for codec in self.codec_menu:
            if codec not in CODEC_NAMES:
                raise PlannerError(f"codec_menu contains unknown codec {codec!r}")
        if self.evidence_grade != "predicted":
            raise PlannerError(
                f"evidence_grade must be 'predicted' at construction (got {self.evidence_grade!r}); "
                "only the operator-routable upgrade gate can flip to empirical "
                "per Catalog #287 + Catalog #323"
            )
        if self.byte_budget < 0:
            raise PlannerError(f"byte_budget must be >= 0; got {self.byte_budget}")
        # Provenance contract per Catalog #323
        for required_key in ("kind", "source_artifact_path", "captured_at_utc"):
            if required_key not in self.provenance:
                raise PlannerError(
                    f"provenance must contain {required_key!r}; got keys={sorted(self.provenance.keys())}"
                )

    def as_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON / DuckDB persistence."""
        return {
            "archive_sha256": self.archive_sha256,
            "x_granularity": self.x_granularity,
            "codec_menu": list(self.codec_menu),
            "byte_budget": self.byte_budget,
            "sensitivity_threshold_quantiles": list(self.sensitivity_threshold_quantiles),
            "assignments_count": len(self.assignments),
            "assignments_sample": [
                {
                    "x_index": r.x_index,
                    "x_class": r.x_class,
                    "sensitivity_score": r.sensitivity_score,
                    "chosen_codec": r.chosen_codec,
                    "chosen_codec_bits": r.chosen_codec_bits,
                    "predicted_score_delta": r.predicted_score_delta,
                    "predicted_bytes_after_codec": r.predicted_bytes_after_codec,
                }
                for r in self.assignments[:10]
            ],
            "total_predicted_score_delta": self.total_predicted_score_delta,
            "total_predicted_bytes": self.total_predicted_bytes,
            "total_predicted_bytes_within_budget": self.total_predicted_bytes_within_budget,
            "operating_point": self.operating_point,
            "measurement_axis": self.measurement_axis,
            "evidence_grade": self.evidence_grade,
            "provenance": self.provenance,
            "captured_at_utc": self.captured_at_utc,
            "schema_version": self.schema_version,
        }

    def class_summary(self) -> dict[str, dict[str, int | float]]:
        """Summarize assignments grouped by x_class.

        Returns: {x_class: {n_bytes, codec_bits, total_bytes_after_codec, fraction_of_archive}}
        """
        n_total = len(self.assignments) or 1
        summary: dict[str, dict[str, int | float]] = {}
        for row in self.assignments:
            cls = row.x_class
            if cls not in summary:
                summary[cls] = {
                    "n_bytes": 0,
                    "codec": row.chosen_codec,
                    "codec_bits": row.chosen_codec_bits or 0,
                    "total_bytes_after_codec": 0,
                    "fraction_of_archive": 0.0,
                }
            summary[cls]["n_bytes"] += 1
            summary[cls]["total_bytes_after_codec"] += row.predicted_bytes_after_codec
        for cls in summary:
            summary[cls]["fraction_of_archive"] = summary[cls]["n_bytes"] / n_total
        return summary


__all__ = [
    "PerXAssignmentRow",
    "PerXCodecAssignmentPlan",
    "PlannerError",
    "X_GRANULARITY_VALUES",
]
