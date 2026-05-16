#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit existing infrastructure for observability gaps.

Per operator standing directive 2026-05-16: *"the xray and autopilot and all
tools and the experiment and designs themselves should be built so as to
support absolute max observability into behavior"*. Anchor memo:
``feedback_max_observability_into_behavior_xray_autopilot_tools_experiments_designs_standing_directive_20260516.md``.

Per consequence 3 of the memory file: existing infrastructure MUST be audited
for observability gaps + extended. This tool surfaces the audit by scoring
each tool/module across the 6 facets of observability:

(1) Per-layer inspection (function inputs + intermediate state + output)
(2) Per-signal decomposition (composite metrics broken into constituents)
(3) Run-to-run diff (artifacts that make two runs comparable)
(4) Post-hoc query interface (JSON / JSONL / SQLite / TensorBoard surfaces)
(5) Cite-chain (commit + call_id + upstream_snapshot_sha256 anchoring)
(6) Counterfactual hooks (byte-mutation surface + ablation switches)

Score per facet: 0 = absent, 1 = partial, 2 = complete. Total max = 12.

Output: JSON report on stdout with per-tool scores + concrete extension
recommendations. CLI rc=0 always (advisory tool, not gating).

Companion to:
- Catalog #305 STRICT preflight gate (substrate design memos must declare
  their observability surface)
- CLAUDE.md "Max observability — non-negotiable" section
- The OBSERVABILITY-ADDENDUM backfill (this landing's Artifact 4)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Scoring dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FacetScore:
    """A score for one of the 6 observability facets."""

    facet_name: str
    score: int  # 0 = absent, 1 = partial, 2 = complete
    evidence: str = ""
    recommendation: str = ""


@dataclass
class ToolObservabilityReport:
    """Observability audit for one tool/module."""

    tool_name: str
    surface_paths: list[str]
    facets: list[FacetScore] = field(default_factory=list)
    total_score: int = 0
    summary: str = ""
    extension_recommendations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-tool audits
# ---------------------------------------------------------------------------


def audit_sensitivity_map(repo_root: Path) -> ToolObservabilityReport:
    """Audit ``tac.sensitivity_map`` (Catalog #586 COUNCIL-A1)."""
    surface = repo_root / "src" / "tac" / "sensitivity_map"
    paths = [str(p.relative_to(repo_root)) for p in sorted(surface.glob("*.py"))[:5]] if surface.exists() else []

    facets = [
        FacetScore(
            facet_name="per_layer_inspection",
            score=1,
            evidence=(
                "axis_weights.py exposes per-axis weights as a queryable dict; "
                "per-layer hooks are implicit via the consumer's scorer call."
            ),
            recommendation=(
                "Add `sensitivity_map.observe_per_layer(scorer, *, hook_dict)` "
                "that returns per-layer (input, output, sensitivity) tuples."
            ),
        ),
        FacetScore(
            facet_name="per_signal_decomposition",
            score=2,
            evidence="Per-axis (seg / pose / rate) sensitivity is decomposed.",
        ),
        FacetScore(
            facet_name="run_to_run_diff",
            score=1,
            evidence="Axis weights are persistable but no canonical diff helper.",
            recommendation=(
                "Land `sensitivity_map.diff(run_a, run_b)` returning per-axis "
                "deltas + a manifest of which axes changed."
            ),
        ),
        FacetScore(
            facet_name="post_hoc_query",
            score=1,
            evidence=(
                "Axis weights serialize to JSON via the consumer; no canonical "
                "query helper for `(substrate, layer, axis)` tuples."
            ),
            recommendation=(
                "Add `sensitivity_map.query(substrate=..., layer=..., axis=...)` "
                "returning the sensitivity scalar + provenance."
            ),
        ),
        FacetScore(
            facet_name="cite_chain",
            score=0,
            evidence="No anchor to commit / call_id / upstream_snapshot_sha256.",
            recommendation=(
                "Persist each axis-weight row with `commit_sha + call_id + "
                "upstream_snapshot_sha256` per Catalog #245 / #128 pattern."
            ),
        ),
        FacetScore(
            facet_name="counterfactual_hooks",
            score=1,
            evidence="Axis-weight ablation is implicit; no canonical switch.",
            recommendation=(
                "Add `sensitivity_map.ablate_axis(axis)` returning a context "
                "manager that zeros the axis weight and restores on exit."
            ),
        ),
    ]
    report = ToolObservabilityReport(
        tool_name="tac.sensitivity_map",
        surface_paths=paths,
        facets=facets,
    )
    report.total_score = sum(f.score for f in facets)
    report.summary = (
        f"6/12 observability — per-axis decomposition is canonical; missing "
        f"cite-chain + post-hoc query helper + canonical counterfactual switch."
    )
    report.extension_recommendations = [
        f.recommendation for f in facets if f.recommendation
    ]
    return report


def audit_cost_band_calibration(repo_root: Path) -> ToolObservabilityReport:
    """Audit ``tac.cost_band_calibration``."""
    surface = repo_root / "src" / "tac" / "cost_band_calibration.py"
    paths = [str(surface.relative_to(repo_root))] if surface.exists() else []

    facets = [
        FacetScore(
            facet_name="per_layer_inspection",
            score=2,
            evidence=(
                "Per-anchor decomposition queryable via `load_anchors()`; "
                "each anchor has outcome + cost + axis + axis_weights + "
                "predicted_wall_clock_hr + actual_wall_clock_hr."
            ),
        ),
        FacetScore(
            facet_name="per_signal_decomposition",
            score=2,
            evidence=(
                "Outcome reasons + per-anchor decomposition both queryable "
                "via `VALID_OUTCOMES` enum and `load_anchors()` filter helpers."
            ),
        ),
        FacetScore(
            facet_name="run_to_run_diff",
            score=2,
            evidence=(
                "Anchors are timestamped JSONL append-only per Catalog #175 / "
                "#177; sort by timestamp + diff is straightforward."
            ),
        ),
        FacetScore(
            facet_name="post_hoc_query",
            score=2,
            evidence=(
                "Public API includes `load_anchors`, `get_provider_routing_decision`, "
                "and sister helpers exposed for downstream consumers."
            ),
        ),
        FacetScore(
            facet_name="cite_chain",
            score=1,
            evidence=(
                "Anchors carry `commit_sha + dispatch_call_id` but not "
                "consistently `upstream_snapshot_sha256` (the new 2026-05-16 field)."
            ),
            recommendation=(
                "Backfill `upstream_snapshot_sha256` field across all "
                "cost-band posterior anchors per the standing directive."
            ),
        ),
        FacetScore(
            facet_name="counterfactual_hooks",
            score=1,
            evidence=(
                "Outcome reasons enumerate failure modes; no canonical "
                "ablation hook for 'what if this anchor were removed?'"
            ),
            recommendation=(
                "Add `cost_band_calibration.ablate_anchor(anchor_id)` returning "
                "predictions without that anchor, surfacing its marginal contribution."
            ),
        ),
    ]
    report = ToolObservabilityReport(
        tool_name="tac.cost_band_calibration",
        surface_paths=paths,
        facets=facets,
    )
    report.total_score = sum(f.score for f in facets)
    report.summary = (
        f"10/12 observability — strong query + decomposition; missing only "
        f"upstream_snapshot_sha256 backfill + ablation hook."
    )
    report.extension_recommendations = [
        f.recommendation for f in facets if f.recommendation
    ]
    return report


def audit_cathedral_autopilot(repo_root: Path) -> ToolObservabilityReport:
    """Audit ``tools/cathedral_autopilot_autonomous_loop.py``."""
    surface = repo_root / "tools" / "cathedral_autopilot_autonomous_loop.py"
    paths = [str(surface.relative_to(repo_root))] if surface.exists() else []

    facets = [
        FacetScore(
            facet_name="per_layer_inspection",
            score=2,
            evidence=(
                "CandidateRow exposes per-candidate fields including "
                "predicted_delta + adjustments_applied + posterior_anchors_consumed."
            ),
        ),
        FacetScore(
            facet_name="per_signal_decomposition",
            score=2,
            evidence=(
                "Predicted ΔS adjustments are per-lens (mdl_density / class_shift / "
                "tier_c / composition_alpha) and trackable per-candidate."
            ),
        ),
        FacetScore(
            facet_name="run_to_run_diff",
            score=1,
            evidence=(
                "Candidate queue serializes to JSONL; no canonical run-diff helper "
                "comparing two ranked queues."
            ),
            recommendation=(
                "Add `cathedral_autopilot.diff_queues(queue_a, queue_b)` returning "
                "per-candidate (rank_delta, predicted_delta_delta, lens_set_delta)."
            ),
        ),
        FacetScore(
            facet_name="post_hoc_query",
            score=2,
            evidence=(
                "JSONL candidate queue + reports/cathedral_autopilot_evidence.jsonl "
                "support arbitrary post-hoc queries."
            ),
        ),
        FacetScore(
            facet_name="cite_chain",
            score=1,
            evidence=(
                "Posterior_anchors_consumed cites anchor IDs; missing canonical "
                "commit_sha + upstream_snapshot_sha256 on the ranker invocation itself."
            ),
            recommendation=(
                "Stamp each ranker invocation with `(commit_sha, call_id, "
                "upstream_snapshot_sha256, posterior_snapshot_sha256)` tuple."
            ),
        ),
        FacetScore(
            facet_name="counterfactual_hooks",
            score=1,
            evidence=(
                "Adjustments are applied additively; no canonical helper for "
                "'what would the ranking be without lens X?'"
            ),
            recommendation=(
                "Add `cathedral_autopilot.rank_without_lens(lens_name)` returning "
                "an alternative ranking with the lens disabled."
            ),
        ),
    ]
    report = ToolObservabilityReport(
        tool_name="tools/cathedral_autopilot_autonomous_loop.py",
        surface_paths=paths,
        facets=facets,
    )
    report.total_score = sum(f.score for f in facets)
    report.summary = (
        f"9/12 observability — strong per-lens decomposition; missing run-diff + "
        f"complete cite-chain + lens-ablation helper."
    )
    report.extension_recommendations = [
        f.recommendation for f in facets if f.recommendation
    ]
    return report


def audit_xray_package(repo_root: Path) -> ToolObservabilityReport:
    """Audit ``src/tac/xray/*`` (per #711 ORPHAN-SIGNAL-AUDIT)."""
    surface = repo_root / "src" / "tac" / "xray"
    paths = [str(p.relative_to(repo_root)) for p in sorted(surface.glob("*.py"))[:10]] if surface.exists() else []

    # Check if wire_in.py exists and is non-trivial.
    wire_in_path = surface / "wire_in.py"
    wire_in_size = wire_in_path.stat().st_size if wire_in_path.exists() else 0

    facets = [
        FacetScore(
            facet_name="per_layer_inspection",
            score=2,
            evidence=(
                "Per-lens modules (per_pair_score_decomposition, "
                "segnet_margin_polytope, score_lipschitz, etc.) expose "
                "structural per-layer observables."
            ),
        ),
        FacetScore(
            facet_name="per_signal_decomposition",
            score=2,
            evidence=(
                "Each lens decomposes a different signal axis (per-pair / "
                "margin / Lipschitz / wavelet / etc.); registry.py orchestrates."
            ),
        ),
        FacetScore(
            facet_name="run_to_run_diff",
            score=1,
            evidence=(
                "Per-lens artifacts serialize but no canonical diff across runs."
            ),
            recommendation="Add `xray.diff_runs(run_a, run_b)` per-lens helper.",
        ),
        FacetScore(
            facet_name="post_hoc_query",
            score=1,
            evidence=(
                f"wire_in.py is {wire_in_size} bytes; per #711 ORPHAN-SIGNAL-AUDIT "
                "the package was flagged as 15 ORPHANS — wire-in to consumers "
                "(cathedral_autopilot ranker / continual_learning posterior) is incomplete."
            ),
            recommendation=(
                "Complete wire_in.py to register each lens with the autopilot "
                "ranker so per-lens lens-scores flow into predicted_delta."
            ),
        ),
        FacetScore(
            facet_name="cite_chain",
            score=1,
            evidence="Per-lens artifacts lack consistent commit+call_id stamping.",
            recommendation=(
                "Standardize the `XrayLensRecord` dataclass to require "
                "(commit_sha, call_id, upstream_snapshot_sha256, lens_name)."
            ),
        ),
        FacetScore(
            facet_name="counterfactual_hooks",
            score=2,
            evidence=(
                "Per-lens ablation is structural (each lens is a separate "
                "module; disable by not invoking)."
            ),
        ),
    ]
    report = ToolObservabilityReport(
        tool_name="src/tac/xray/*",
        surface_paths=paths,
        facets=facets,
    )
    report.total_score = sum(f.score for f in facets)
    report.summary = (
        f"9/12 observability — strong per-lens decomposition + ablation; "
        f"missing complete wire_in.py (per #711 ORPHAN-SIGNAL-AUDIT) + "
        f"run-diff helper + canonical cite-chain stamping."
    )
    report.extension_recommendations = [
        f.recommendation for f in facets if f.recommendation
    ]
    return report


def audit_audit_tools_family(repo_root: Path) -> ToolObservabilityReport:
    """Audit the ``tools/audit_*.py`` family."""
    tools_dir = repo_root / "tools"
    audit_files = sorted(tools_dir.glob("audit_*.py")) if tools_dir.exists() else []
    paths = [str(p.relative_to(repo_root)) for p in audit_files[:8]]

    # Sample a few for JSON-emitting structure.
    sample = audit_files[:3] if audit_files else []
    json_emit_count = 0
    for f in sample:
        try:
            txt = f.read_text(encoding="utf-8", errors="replace")
            if "json.dump" in txt or "json.dumps" in txt or "--json" in txt:
                json_emit_count += 1
        except OSError:
            pass

    facets = [
        FacetScore(
            facet_name="per_layer_inspection",
            score=1,
            evidence=(
                f"{len(audit_files)} audit_* tools; per-tool inspection varies."
            ),
        ),
        FacetScore(
            facet_name="per_signal_decomposition",
            score=1,
            evidence="Each tool emits a focused signal; cross-tool composition limited.",
            recommendation=(
                "Standardize audit_* output schema with `tool_name + signal_name + "
                "rows[]` so a cross-tool composer can decompose by signal."
            ),
        ),
        FacetScore(
            facet_name="run_to_run_diff",
            score=0,
            evidence="No canonical diff across audit runs.",
            recommendation=(
                "Land `tools/diff_audit_runs.py <a.json> <b.json>` returning "
                "per-signal row-level diffs (added / removed / mutated)."
            ),
        ),
        FacetScore(
            facet_name="post_hoc_query",
            score=1 if json_emit_count >= 1 else 0,
            evidence=(
                f"{json_emit_count}/{len(sample)} sampled tools emit JSON; "
                "schema consistency varies."
            ),
            recommendation=(
                "Adopt a canonical AuditReport dataclass with `as_dict()` + "
                "`--json` CLI flag across all audit_* tools."
            ),
        ),
        FacetScore(
            facet_name="cite_chain",
            score=0,
            evidence="Audit outputs lack consistent commit / call_id stamping.",
            recommendation=(
                "Every audit_* tool MUST stamp its output with "
                "(commit_sha, invoked_at_utc, invoking_subagent_id)."
            ),
        ),
        FacetScore(
            facet_name="counterfactual_hooks",
            score=0,
            evidence="Audit tools are read-only; no canonical ablation.",
        ),
    ]
    report = ToolObservabilityReport(
        tool_name="tools/audit_*.py family",
        surface_paths=paths,
        facets=facets,
    )
    report.total_score = sum(f.score for f in facets)
    report.summary = (
        f"3/12 observability — most audit tools emit JSON but lack canonical "
        f"schema + cite-chain + run-diff. Highest ROI extension target."
    )
    report.extension_recommendations = [
        f.recommendation for f in facets if f.recommendation
    ]
    return report


def audit_contest_auth_eval(repo_root: Path) -> ToolObservabilityReport:
    """Audit ``experiments/contest_auth_eval.py`` artifacts."""
    surface = repo_root / "experiments" / "contest_auth_eval.py"
    paths = [str(surface.relative_to(repo_root))] if surface.exists() else []

    facets = [
        FacetScore(
            facet_name="per_layer_inspection",
            score=2,
            evidence=(
                "Per-component (segnet / posenet / rate) outputs are captured "
                "in the result JSON; per-pair distortion arrays preserved."
            ),
        ),
        FacetScore(
            facet_name="per_signal_decomposition",
            score=2,
            evidence=(
                "Final score = seg + sqrt(10*pose) + 25*rate; all three "
                "components serialized + queryable in the result JSON."
            ),
        ),
        FacetScore(
            facet_name="run_to_run_diff",
            score=1,
            evidence=(
                "Result JSONs are byte-addressable by archive_sha256; manual "
                "diff straightforward but no canonical helper."
            ),
            recommendation=(
                "Add `tools/diff_auth_eval_results.py <a.json> <b.json>` "
                "emitting per-component deltas + per-pair drift."
            ),
        ),
        FacetScore(
            facet_name="post_hoc_query",
            score=2,
            evidence=(
                "Result JSONs are structured; per-axis tags ([contest-CUDA] / "
                "[contest-CPU] / [diagnostic-CPU]) make filtering trivial."
            ),
        ),
        FacetScore(
            facet_name="cite_chain",
            score=2,
            evidence=(
                "Result JSON includes archive_sha256 + runtime_tree_sha + "
                "evaluator_sha + commit_sha + hardware substrate."
            ),
        ),
        FacetScore(
            facet_name="counterfactual_hooks",
            score=1,
            evidence=(
                "Byte-mutation surface lives in Catalog #139 / #272; not "
                "directly invoked from contest_auth_eval.py."
            ),
            recommendation=(
                "Add `contest_auth_eval.py --counterfactual-byte-mutation` flag "
                "wiring the Catalog #139 packet compiler for per-section ablation."
            ),
        ),
    ]
    report = ToolObservabilityReport(
        tool_name="experiments/contest_auth_eval.py",
        surface_paths=paths,
        facets=facets,
    )
    report.total_score = sum(f.score for f in facets)
    report.summary = (
        f"10/12 observability — best-in-class per-component decomposition + "
        f"cite-chain; missing only canonical diff helper + byte-mutation flag."
    )
    report.extension_recommendations = [
        f.recommendation for f in facets if f.recommendation
    ]
    return report


def audit_continual_learning(repo_root: Path) -> ToolObservabilityReport:
    """Audit ``tac.continual_learning.posterior_update_locked``."""
    surface = repo_root / "src" / "tac" / "continual_learning.py"
    paths = [str(surface.relative_to(repo_root))] if surface.exists() else []

    facets = [
        FacetScore(
            facet_name="per_layer_inspection",
            score=2,
            evidence=(
                "ContestResult dataclass exposes per-anchor fields including "
                "evidence_grade + archive_sha256 + components + runtime_manifest."
            ),
        ),
        FacetScore(
            facet_name="per_signal_decomposition",
            score=2,
            evidence=(
                "Per-anchor signal decomposition (seg / pose / rate) preserved; "
                "axis tag + hardware substrate stamped."
            ),
        ),
        FacetScore(
            facet_name="run_to_run_diff",
            score=1,
            evidence="Posterior is append-only; sort by timestamp + diff straightforward.",
            recommendation=(
                "Add `continual_learning.diff_posterior_snapshots(snapshot_a, "
                "snapshot_b)` returning per-anchor delta."
            ),
        ),
        FacetScore(
            facet_name="post_hoc_query",
            score=2,
            evidence=(
                "Public API includes load_posterior_lenient + load_posterior_strict "
                "+ query helpers for filtering by axis / substrate / grade."
            ),
        ),
        FacetScore(
            facet_name="cite_chain",
            score=2,
            evidence=(
                "Per-anchor provenance: archive_sha256 + runtime_tree_sha + "
                "commit_sha + dispatch_call_id + hardware substrate + axis tag."
            ),
        ),
        FacetScore(
            facet_name="counterfactual_hooks",
            score=1,
            evidence="No canonical 'predict-without-this-anchor' helper.",
            recommendation=(
                "Add `continual_learning.ablate_anchor(anchor_id)` returning a "
                "filtered posterior view for counterfactual ranking."
            ),
        ),
    ]
    report = ToolObservabilityReport(
        tool_name="tac.continual_learning.posterior_update_locked",
        surface_paths=paths,
        facets=facets,
    )
    report.total_score = sum(f.score for f in facets)
    report.summary = (
        f"10/12 observability — strong cite-chain + decomposition; missing "
        f"canonical diff helper + ablation hook (sister of cost_band_calibration)."
    )
    report.extension_recommendations = [
        f.recommendation for f in facets if f.recommendation
    ]
    return report


def audit_council_continual_learning(repo_root: Path) -> ToolObservabilityReport:
    """Audit ``tac.council_continual_learning`` (just landed Catalog #300)."""
    surface = repo_root / "src" / "tac" / "council_continual_learning.py"
    paths = [str(surface.relative_to(repo_root))] if surface.exists() else []

    facets = [
        FacetScore(
            facet_name="per_layer_inspection",
            score=2,
            evidence=(
                "CouncilDeliberationRecord exposes 17+ fields including "
                "tier + assumptions + override + retrospective_due_utc."
            ),
        ),
        FacetScore(
            facet_name="per_signal_decomposition",
            score=2,
            evidence=(
                "Mission contribution categorized into 5 enums; assumption "
                "classification HARD-EARNED vs CARGO-CULTED."
            ),
        ),
        FacetScore(
            facet_name="run_to_run_diff",
            score=1,
            evidence=(
                "Per-deliberation records are append-only; no canonical diff."
            ),
            recommendation=(
                "Add `council_continual_learning.diff_deliberations(a, b)` "
                "for cross-deliberation drift detection."
            ),
        ),
        FacetScore(
            facet_name="post_hoc_query",
            score=2,
            evidence=(
                "7+ canonical query helpers: query_overrides + "
                "query_due_retrospectives + query_mission_contribution_distribution "
                "+ is_rigor_dominant + query_assumption_classification_history."
            ),
        ),
        FacetScore(
            facet_name="cite_chain",
            score=2,
            evidence=(
                "Per-deliberation cite-chain: commit_sha + deliberation_id + "
                "council_memo_path + invoking_subagent_id."
            ),
        ),
        FacetScore(
            facet_name="counterfactual_hooks",
            score=1,
            evidence=(
                "Override invocation is canonical; no 'what would the consensus "
                "be without this member?' helper."
            ),
            recommendation=(
                "Add `council_continual_learning.ablate_member(deliberation_id, "
                "member_name)` returning alternative consensus calculation."
            ),
        ),
    ]
    report = ToolObservabilityReport(
        tool_name="tac.council_continual_learning (Catalog #300)",
        surface_paths=paths,
        facets=facets,
    )
    report.total_score = sum(f.score for f in facets)
    report.summary = (
        f"10/12 observability — strong cite-chain + 7 query helpers + 5-enum "
        f"decomposition; missing only diff helper + member-ablation hook."
    )
    report.extension_recommendations = [
        f.recommendation for f in facets if f.recommendation
    ]
    return report


# ---------------------------------------------------------------------------
# Report aggregation + CLI
# ---------------------------------------------------------------------------


def _report_to_dict(report: ToolObservabilityReport) -> dict:
    """Convert a ToolObservabilityReport to a plain dict (avoids dataclasses.asdict
    issues with dynamically-loaded modules where dataclass MRO can't resolve)."""
    return {
        "tool_name": report.tool_name,
        "surface_paths": list(report.surface_paths),
        "facets": [
            {
                "facet_name": f.facet_name,
                "score": f.score,
                "evidence": f.evidence,
                "recommendation": f.recommendation,
            }
            for f in report.facets
        ],
        "total_score": report.total_score,
        "summary": report.summary,
        "extension_recommendations": list(report.extension_recommendations),
    }


def build_full_audit(repo_root: Path) -> dict:
    """Build the full observability audit report."""
    reports = [
        audit_sensitivity_map(repo_root),
        audit_cost_band_calibration(repo_root),
        audit_cathedral_autopilot(repo_root),
        audit_xray_package(repo_root),
        audit_audit_tools_family(repo_root),
        audit_contest_auth_eval(repo_root),
        audit_continual_learning(repo_root),
        audit_council_continual_learning(repo_root),
    ]
    total_max = len(reports) * 12
    total_score = sum(r.total_score for r in reports)
    return {
        "schema_version": "observability_audit_v1",
        "generated_at_utc": "2026-05-16",
        "anchor_memo": (
            "feedback_max_observability_into_behavior_xray_autopilot_tools_"
            "experiments_designs_standing_directive_20260516.md"
        ),
        "directive_verbatim": (
            "the xray and autopilot and all tools and the experiment and "
            "designs themselves should be built so as to support absolute "
            "max observability into behavior"
        ),
        "facet_definition": {
            "per_layer_inspection": "Every layer's input + output + intermediate state can be captured at runtime",
            "per_signal_decomposition": "Composite metrics decomposable into constituent contributions",
            "run_to_run_diff": "Two runs of the same substrate diff-able byte-level + activation-level + score-level",
            "post_hoc_query": "Run artifacts support arbitrary queries without re-running",
            "cite_chain": "Every behavior signal anchored to (substrate, commit, call_id, config, seed, upstream_snapshot_sha256)",
            "counterfactual_hooks": "Byte-mutation discipline allows 'what if this byte changed?' without re-running",
        },
        "score_scale": "0 = absent, 1 = partial, 2 = complete; max per tool = 12",
        "tools_audited": len(reports),
        "total_score": total_score,
        "total_max": total_max,
        "overall_observability_pct": round(100.0 * total_score / total_max, 1),
        "reports": [_report_to_dict(r) for r in reports],
        "highest_roi_extensions": [
            (
                "tools/audit_*.py family canonical AuditReport dataclass "
                "(currently 3/12; biggest gap)"
            ),
            (
                "tac.sensitivity_map cite-chain backfill "
                "(persist commit + call_id + upstream_snapshot_sha256 per axis-weight row)"
            ),
            (
                "src/tac/xray/wire_in.py completion per #711 ORPHAN-SIGNAL-AUDIT "
                "(register each lens with cathedral_autopilot ranker)"
            ),
        ],
        "sister_strict_gate": "Catalog #305 enforces design-memo declaration of observability surface",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root (default: auto-detect)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON to stdout (default behavior).",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Emit a human-readable summary instead of JSON.",
    )
    args = parser.parse_args(argv)

    audit = build_full_audit(args.repo_root)

    if args.summary:
        print(
            f"Observability audit: {audit['tools_audited']} tools, "
            f"{audit['total_score']}/{audit['total_max']} "
            f"({audit['overall_observability_pct']}%)"
        )
        print()
        for r in audit["reports"]:
            print(f"  {r['tool_name']}: {r['total_score']}/12")
            print(f"    {r['summary']}")
            for rec in r["extension_recommendations"]:
                print(f"    -> {rec}")
            print()
        print("Highest ROI extensions:")
        for ext in audit["highest_roi_extensions"]:
            print(f"  - {ext}")
    else:
        print(json.dumps(audit, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
