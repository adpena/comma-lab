# SPDX-License-Identifier: MIT
"""Canonical apparatus law for DDM scorer-metric bundle completeness.

This law grants no scientific or score authority.  It makes the conjunction
required by the MS3 custody gate executable and registered: all four named
components must be scientifically complete, freshly rehashed, n600/batch32,
and each bucket-valued surface must cover the exact 1,200-row PF2 atlas.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "ddm_metric_custody_bundle_completion_v1"
COMPONENTS = (
    "SEG_METRIC",
    "POSE_METRIC",
    "COMPOSITE_R_SECOND_ORDER",
    "DUAL_METRIC_DIAGNOSTICS",
)
REPO = Path(__file__).resolve().parents[3]
PARTIAL_MANIFEST = REPO / ".omx/research/ddm_ms3_metric_custody_bundle_20260724T035249Z/BUNDLE-PARTIAL.json"


def metric_custody_bundle_completion_law(
    component_complete: Mapping[str, bool],
    component_sha_fresh: Mapping[str, bool],
    component_pair_counts: Mapping[str, int],
    component_batch_sizes: Mapping[str, int],
    *,
    seg_bucket_count: int,
    composite_r_bucket_count: int,
    dual_bucket_count: int,
) -> bool:
    """Return the exact structural completion conjunction.

    Scientific field validation remains the responsibility of
    :func:`load_metric_custody_bundle`; this law is its portable admission
    skeleton and cannot turn arbitrary booleans into measured evidence.
    """

    expected = set(COMPONENTS)
    mappings = (
        component_complete,
        component_sha_fresh,
        component_pair_counts,
        component_batch_sizes,
    )
    if any(set(value) != expected for value in mappings):
        raise ValueError("metric custody mappings must contain exactly four components")
    if any(not isinstance(component_complete[name], bool) for name in COMPONENTS):
        raise ValueError("component_complete values must be exact booleans")
    if any(not isinstance(component_sha_fresh[name], bool) for name in COMPONENTS):
        raise ValueError("component_sha_fresh values must be exact booleans")
    if any(
        isinstance(component_pair_counts[name], bool) or not isinstance(component_pair_counts[name], int)
        for name in COMPONENTS
    ):
        raise ValueError("component_pair_counts values must be exact integers")
    if any(
        isinstance(component_batch_sizes[name], bool) or not isinstance(component_batch_sizes[name], int)
        for name in COMPONENTS
    ):
        raise ValueError("component_batch_sizes values must be exact integers")
    bucket_counts = (seg_bucket_count, composite_r_bucket_count, dual_bucket_count)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in bucket_counts):
        raise ValueError("bucket counts must be exact integers")
    return (
        all(component_complete.values())
        and all(component_sha_fresh.values())
        and all(component_pair_counts[name] == 600 for name in COMPONENTS)
        and all(component_batch_sizes[name] == 32 for name in COMPONENTS)
        and bucket_counts == (1200, 1200, 1200)
    )


def build_ddm_metric_custody_bundle_completion_v1(*, source_manifest: Path = PARTIAL_MANIFEST) -> CanonicalEquation:
    """Build the research-only completeness law from a SHA-bound manifest."""

    provenance = build_provenance_for_research_sidecar(
        source_manifest,
        reactivation_criteria=(
            "Replace BUNDLE-PARTIAL with MAIN-reviewed BUNDLE-COMPLETE only after "
            "all four exact n600 batch32 measurements pass the strict loader."
        ),
        measurement_axis="[macOS-CPU frozen-scorer advisory]",
        hardware_substrate="darwin_arm64_cpu_torch",
        captured_at_utc="2026-07-24T03:52:49Z",
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="DDM scorer-metric custody bundle completion conjunction",
        one_line_summary=(
            "Metric authority activates iff four named components are fresh, "
            "scientifically complete, n600/batch32, and PF2-bucket complete."
        ),
        latex_form=(
            r"C_{\mathrm{bundle}}=\bigwedge_{k=1}^{4}"
            r"(C_k\land H_k\land n_k=600\land b_k=32)"
            r"\land(N_{\mathrm{Seg}},N_R,N_{\mathrm{dual}})=(1200,1200,1200)"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.ddm_metric_custody_bundle_20260724:metric_custody_bundle_completion_law"
        ),
        domain_of_validity={
            "role": "structural admission apparatus only",
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
            "current_status": "PARTIAL",
            "verdict_scope": (
                "INSTANCE: current bundle lacks required measurements; no FORMULATION, FAMILY, or PARADIGM verdict"
            ),
            "scientific_validator": ("tac.optimization.ddm_metric_custody_bundle:load_metric_custody_bundle"),
        },
        units_in={
            "component_complete": "boolean",
            "component_sha_fresh": "boolean",
            "component_pair_counts": "pairs",
            "component_batch_sizes": "pairs_per_batch",
            "bucket_counts": "PF2_buckets",
        },
        units_out={"bundle_complete": "boolean"},
        empirical_anchors=(),
        predicted_vs_empirical_residual={},
        last_calibration_utc="2026-07-24T03:52:49Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.optimization.ddm_metric_custody_bundle:load_metric_custody_bundle",
            "tac.optimization.ddm_min_description_contract:build_minimum_description_headline",
        ),
        canonical_producers=("tools.materialize_ddm_metric_custody_bundle",),
        provenance=provenance,
    )


def populate_ddm_metric_custody_bundle_completion(
    *,
    source_manifest: Path = PARTIAL_MANIFEST,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Register the law through the append-only canonical registry."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_ddm_metric_custody_bundle_completion_v1(source_manifest=source_manifest)
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "MS3 custody apparatus; current bundle PARTIAL; score_claim=false; "
            "pointer unchanged; MAIN landing review required"
        ),
    )
    return equation


__all__ = [
    "COMPONENTS",
    "EQUATION_ID",
    "build_ddm_metric_custody_bundle_completion_v1",
    "metric_custody_bundle_completion_law",
    "populate_ddm_metric_custody_bundle_completion",
]
