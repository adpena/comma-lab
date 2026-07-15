# SPDX-License-Identifier: MIT
"""Typed global ``H_cov`` gluing/descent datum for D38.

This module deliberately separates three claims that earlier prose conflated:

1. local fixed-stratum semidirect products can split;
2. exact restrictions of an already-global NumPy array can agree on overlaps;
3. a receiver-complete coefficient/action atlas can descend with a charged
   global section.

Only (1) and the tautological instance of (2) are measured.  The types below
make (3) expressible and fail closed until every action, restriction,
intertwiner, cocycle, receiver section, and charge is bound.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

EQUATION_ID = "hcov_global_gluing_descent_v1"
DEFAULT_RECEIPT = ".omx/research/v9_cgauge_symmetry_homotopy_n600_receipt_20260714.json"
DEFAULT_RECEIPT_SHA256 = (
    "60dd6a4837706d100932416cf8fdf77fce0e7c171b1ef58fd3f1154021428308"
)


class HCovGluingStatus(StrEnum):
    INVALID = "INVALID_HCOV_GLUING_DATUM"
    TYPED_UNBOUND = "TYPED_LOCAL_DATA_GLOBAL_RATE_DESCENT_UNBOUND"
    GLOBAL_RATE_DESCENT_TYPED = "GLOBAL_RATE_DESCENT_TYPED"


@dataclass(frozen=True)
class HCovStratumChart:
    chart_id: str
    stratum_id: str
    kernel_group_id: str | None
    hcov_action_id: str | None
    coefficient_object_id: str | None


@dataclass(frozen=True)
class HCovOverlapRestriction:
    overlap_id: str
    left_chart_id: str
    right_chart_id: str
    left_restriction_id: str | None
    right_restriction_id: str | None
    hcov_intertwiner_2cell_id: str | None
    exact_on_bound_receipt: bool | None


@dataclass(frozen=True)
class HCovTripleCocycle:
    triple_id: str
    chart_ids: tuple[str, str, str]
    composed_restriction_id: str | None
    direct_restriction_id: str | None
    commutes_on_bound_receipt: bool | None


@dataclass(frozen=True)
class HCovGluingAtlas:
    cover_id: str
    charts: tuple[HCovStratumChart, ...]
    overlaps: tuple[HCovOverlapRestriction, ...]
    triple_cocycles: tuple[HCovTripleCocycle, ...]
    receiver_section_id: str | None
    charged_section_bits: float | None
    source_receipt: str | None = None
    source_receipt_sha256: str | None = None
    measured_overlap_points: int | None = None

    def schema_violations(self) -> tuple[str, ...]:
        """Return structural errors; absence of empirical bindings is not a schema error."""

        problems: list[str] = []
        if not self.cover_id.strip():
            problems.append("cover_id must be non-empty")
        chart_ids = [chart.chart_id for chart in self.charts]
        if not chart_ids:
            problems.append("atlas needs at least one chart")
        if len(set(chart_ids)) != len(chart_ids):
            problems.append("chart_id values must be unique")
        known = set(chart_ids)
        overlap_ids: set[str] = set()
        for overlap in self.overlaps:
            if overlap.overlap_id in overlap_ids:
                problems.append(f"duplicate overlap_id={overlap.overlap_id!r}")
            overlap_ids.add(overlap.overlap_id)
            if overlap.left_chart_id not in known or overlap.right_chart_id not in known:
                problems.append(
                    f"overlap {overlap.overlap_id!r} references an unknown chart"
                )
            if overlap.left_chart_id == overlap.right_chart_id:
                problems.append(
                    f"overlap {overlap.overlap_id!r} must join distinct charts"
                )
        triple_ids: set[str] = set()
        for cocycle in self.triple_cocycles:
            if cocycle.triple_id in triple_ids:
                problems.append(f"duplicate triple_id={cocycle.triple_id!r}")
            triple_ids.add(cocycle.triple_id)
            if len(set(cocycle.chart_ids)) != 3 or not set(cocycle.chart_ids) <= known:
                problems.append(
                    f"triple {cocycle.triple_id!r} must name three distinct known charts"
                )
        if self.charged_section_bits is not None and self.charged_section_bits < 0.0:
            problems.append("charged_section_bits must be >= 0 when bound")
        return tuple(problems)

    def unbound_global_rate_fields(self) -> tuple[str, ...]:
        """List every missing proof object that blocks a global rate-descent claim."""

        missing: list[str] = []
        for chart in self.charts:
            for name in ("kernel_group_id", "hcov_action_id", "coefficient_object_id"):
                if not getattr(chart, name):
                    missing.append(f"chart:{chart.chart_id}:{name}")
        for overlap in self.overlaps:
            for name in (
                "left_restriction_id",
                "right_restriction_id",
                "hcov_intertwiner_2cell_id",
            ):
                if not getattr(overlap, name):
                    missing.append(f"overlap:{overlap.overlap_id}:{name}")
            if overlap.exact_on_bound_receipt is not True:
                missing.append(f"overlap:{overlap.overlap_id}:exact_receipt")
        for cocycle in self.triple_cocycles:
            for name in ("composed_restriction_id", "direct_restriction_id"):
                if not getattr(cocycle, name):
                    missing.append(f"triple:{cocycle.triple_id}:{name}")
            if cocycle.commutes_on_bound_receipt is not True:
                missing.append(f"triple:{cocycle.triple_id}:commuting_receipt")
        if not self.receiver_section_id:
            missing.append("receiver_section_id")
        if self.charged_section_bits is None:
            missing.append("charged_section_bits")
        return tuple(missing)

    def status(self) -> HCovGluingStatus:
        if self.schema_violations():
            return HCovGluingStatus.INVALID
        if self.unbound_global_rate_fields():
            return HCovGluingStatus.TYPED_UNBOUND
        return HCovGluingStatus.GLOBAL_RATE_DESCENT_TYPED

    def require_global_rate_descent(self) -> None:
        """Fail closed unless this datum supports a charged receiver descent claim."""

        if self.status() is not HCovGluingStatus.GLOBAL_RATE_DESCENT_TYPED:
            raise ValueError(
                f"global H_cov rate descent unavailable: status={self.status().value}; "
                f"schema={self.schema_violations()}; unbound={self.unbound_global_rate_fields()}"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_array_quadrant_atlas_from_receipt(
    receipt_path: str | Path = DEFAULT_RECEIPT,
    *,
    expected_sha256: str = DEFAULT_RECEIPT_SHA256,
    repo_root: str | Path | None = None,
) -> HCovGluingAtlas:
    """Bind the measured exact-array overlap instance without laundering it into rate descent."""

    path = Path(receipt_path)
    if not path.is_absolute():
        root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
        path = root / path
    actual_sha = _sha256(path)
    if actual_sha != expected_sha256:
        raise ValueError(
            f"D38 receipt sha256 mismatch: expected {expected_sha256}, got {actual_sha}"
        )
    doc = json.loads(path.read_text(encoding="utf-8"))
    d38 = doc["d38"]
    if d38.get("exact_section_glues") is not True or d38.get("failed_pairs"):
        raise ValueError("D38 receipt does not certify exact array restrictions")

    chart_ids = ("quadrant_nw", "quadrant_ne", "quadrant_sw", "quadrant_se")
    charts = tuple(
        HCovStratumChart(
            chart_id=chart_id,
            stratum_id="realized_flip_array",
            kernel_group_id=None,
            hcov_action_id=None,
            coefficient_object_id=None,
        )
        for chart_id in chart_ids
    )
    pairs = (
        ("north", chart_ids[0], chart_ids[1]),
        ("west", chart_ids[0], chart_ids[2]),
        ("east", chart_ids[1], chart_ids[3]),
        ("south", chart_ids[2], chart_ids[3]),
    )
    overlaps = tuple(
        HCovOverlapRestriction(
            overlap_id=name,
            left_chart_id=left,
            right_chart_id=right,
            left_restriction_id="exact_numpy_array_slice",
            right_restriction_id="exact_numpy_array_slice",
            hcov_intertwiner_2cell_id=None,
            exact_on_bound_receipt=True,
        )
        for name, left, right in pairs
    )
    return HCovGluingAtlas(
        cover_id=str(d38["cover"]),
        charts=charts,
        overlaps=overlaps,
        triple_cocycles=(),
        receiver_section_id=None,
        charged_section_bits=None,
        source_receipt=str(receipt_path),
        source_receipt_sha256=actual_sha,
        measured_overlap_points=int(d38["total_pairwise_overlap_points"]),
    )


__all__ = [
    "DEFAULT_RECEIPT",
    "DEFAULT_RECEIPT_SHA256",
    "EQUATION_ID",
    "HCovGluingAtlas",
    "HCovGluingStatus",
    "HCovOverlapRestriction",
    "HCovStratumChart",
    "HCovTripleCocycle",
    "exact_array_quadrant_atlas_from_receipt",
]
