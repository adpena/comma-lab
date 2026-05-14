#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Rank the next non-queued public-floor breakthrough opportunities.

This planner is deliberately local-only. It reads existing C091/PR75 public
floor artifacts, excludes the exact-eval lanes the parent already queued, and
emits a no-dispatch ledger unless a new non-queued candidate is both
break-even-plausible and locally safe. It never launches GPU work and never
edits dispatch state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "public_floor_next_breakthrough_worker_plan_v1"
TOOL = "experiments/plan_public_floor_next_breakthrough_worker.py"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "experiments/results/public_floor_next_breakthrough_worker_20260503"
)
DEFAULT_LEDGER_MD = REPO_ROOT / ".omx/research/public_floor_next_breakthrough_worker_20260503.md"
TARGET_SCORE = 0.314
RATE_SCORE_PER_BYTE = 25.0 / 37_545_489.0
FRONTIER_SCORE = 0.31516575028285976
FRONTIER_BYTES = 276_481
FRONTIER_SHA256 = "03a2afd5fe92c93a9b7b7e43625158a73b455f0cfbca82d278008a728db78746"
FRONTIER_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_pr75_minp_public_replay_t4_20260503T1049Z/archive.zip"
)
FRONTIER_EVAL_JSON = FRONTIER_ARCHIVE.with_name("contest_auth_eval.adjudicated.json")
PARENT_QUEUED_IDS = {
    "c091_native_cem_pose_waterfill_top128_s025",
    "c091_pr65_pose_qp1_c089_actions_p6",
    "pr65_fix1_exact_eval",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _load_json(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _repo_rel(path: Path | str | None) -> str | None:
    if path is None:
        return None
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def strict_bytes_needed_for_score_gap(
    *,
    score: float,
    target_score: float = TARGET_SCORE,
    rate_score_per_byte: float = RATE_SCORE_PER_BYTE,
) -> int:
    """Bytes required to become strictly below target at unchanged components."""

    gap = float(score) - float(target_score)
    if gap <= 0:
        return 0
    return int(math.floor(gap / float(rate_score_per_byte))) + 1


def score_if_components_unchanged(
    *,
    frontier_score: float,
    frontier_bytes: int,
    candidate_bytes: int,
) -> float:
    return float(frontier_score) + RATE_SCORE_PER_BYTE * (
        int(candidate_bytes) - int(frontier_bytes)
    )


def break_even(
    *,
    candidate_bytes: int,
    frontier_score: float = FRONTIER_SCORE,
    frontier_bytes: int = FRONTIER_BYTES,
    target_score: float = TARGET_SCORE,
) -> dict[str, Any]:
    unchanged = score_if_components_unchanged(
        frontier_score=frontier_score,
        frontier_bytes=frontier_bytes,
        candidate_bytes=candidate_bytes,
    )
    gain_needed = max(0.0, unchanged - target_score)
    return {
        "delta_bytes_vs_frontier": int(candidate_bytes) - int(frontier_bytes),
        "rate_score_delta_vs_frontier": RATE_SCORE_PER_BYTE
        * (int(candidate_bytes) - int(frontier_bytes)),
        "score_if_components_unchanged": unchanged,
        "strict_sub314_at_unchanged_components": unchanged < target_score,
        "sub314_component_score_gain_needed": gain_needed,
        "sub314_equivalent_bytes_needed_after_candidate": strict_bytes_needed_for_score_gap(
            score=unchanged,
            target_score=target_score,
        ),
    }


def _frontier_custody() -> dict[str, Any]:
    archive = FRONTIER_ARCHIVE
    eval_json = FRONTIER_EVAL_JSON
    failures: list[str] = []
    actual_bytes = archive.stat().st_size if archive.is_file() else None
    actual_sha = _sha256_file(archive) if archive.is_file() else None
    eval_payload = _load_json(eval_json)
    if actual_bytes != FRONTIER_BYTES:
        failures.append("frontier_archive_bytes_mismatch_or_missing")
    if actual_sha != FRONTIER_SHA256:
        failures.append("frontier_archive_sha256_mismatch_or_missing")
    if eval_payload is None:
        failures.append("frontier_eval_json_missing")
        score = None
    else:
        score = float(eval_payload.get("canonical_score", math.nan))
        if not math.isfinite(score) or abs(score - FRONTIER_SCORE) > 1e-12:
            failures.append("frontier_eval_score_mismatch")
        if int(eval_payload.get("archive_size_bytes", -1)) != FRONTIER_BYTES:
            failures.append("frontier_eval_archive_size_mismatch")
    return {
        "ok": not failures,
        "failures": failures,
        "archive": {
            "path": _repo_rel(archive),
            "bytes": actual_bytes,
            "sha256": actual_sha,
            "expected_bytes": FRONTIER_BYTES,
            "expected_sha256": FRONTIER_SHA256,
        },
        "eval_json": {
            "path": _repo_rel(eval_json),
            "score": score,
            "target_score": TARGET_SCORE,
            "strict_unchanged_component_bytes_needed": strict_bytes_needed_for_score_gap(
                score=FRONTIER_SCORE
            ),
        },
    }


def _pose_safety_summary(report: Mapping[str, Any] | None, path: Path | None) -> dict[str, Any]:
    if report is None:
        return {
            "preflight_ran": False,
            "path": _repo_rel(path) if path else None,
            "safe_for_exact_eval_dispatch": False,
            "failure_class": "pose_safety_preflight_missing",
        }
    parity = report.get("output_parity")
    aggregate: Mapping[str, Any] = {}
    if isinstance(parity, Mapping) and isinstance(parity.get("aggregate"), Mapping):
        aggregate = parity["aggregate"]
    report_path = path if path is not None else report.get("path")
    return {
        "preflight_ran": True,
        "path": _repo_rel(report_path),
        "safe_for_exact_eval_dispatch": bool(report.get("safe_for_exact_eval_dispatch")),
        "failure_class": report.get("failure_class"),
        "fail_closed_reasons": report.get("fail_closed_reasons", []),
        "mean_abs_delta": aggregate.get("mean_abs_delta"),
        "rms_delta": aggregate.get("rms_delta"),
        "max_abs_delta": aggregate.get("max_abs_delta"),
        "same_uint8_hash": aggregate.get("same_uint8_hash"),
    }


def opportunity(
    *,
    opportunity_id: str,
    rank_group: int,
    title: str,
    status: str,
    archive_bytes: int | None = None,
    archive_sha256: str | None = None,
    archive_path: str | None = None,
    evidence_grade: str = "empirical_planning_only",
    component_gain_proxy: float | None = None,
    exact_score: float | None = None,
    exact_eval_path: str | None = None,
    fail_reasons: list[str] | None = None,
    next_unblock: str | None = None,
    safety: Mapping[str, Any] | None = None,
    source_artifacts: list[str] | None = None,
    parent_queued: bool = False,
) -> dict[str, Any]:
    economics = (
        break_even(candidate_bytes=archive_bytes)
        if archive_bytes is not None
        else {
            "delta_bytes_vs_frontier": None,
            "rate_score_delta_vs_frontier": None,
            "score_if_components_unchanged": None,
            "strict_sub314_at_unchanged_components": False,
            "sub314_component_score_gain_needed": None,
            "sub314_equivalent_bytes_needed_after_candidate": None,
        }
    )
    exact_eval_recommended = (
        not parent_queued
        and status == "ready_for_exact_eval_after_claim"
        and (
            economics["strict_sub314_at_unchanged_components"]
            or (component_gain_proxy is not None and component_gain_proxy >= economics["sub314_component_score_gain_needed"])
        )
        and (not safety or bool(safety.get("safe_for_exact_eval_dispatch", False)))
    )
    return {
        "opportunity_id": opportunity_id,
        "rank_group": int(rank_group),
        "title": title,
        "status": status,
        "parent_queued": parent_queued,
        "evidence_grade": evidence_grade,
        "score_claim": False,
        "promotion_eligible": False,
        "exact_eval_recommended": exact_eval_recommended,
        "archive": {
            "path": _repo_rel(archive_path),
            "bytes": archive_bytes,
            "sha256": archive_sha256,
        },
        "economics_vs_c091": economics,
        "component_gain_proxy": component_gain_proxy,
        "exact_score_if_already_measured": exact_score,
        "exact_eval_path": _repo_rel(exact_eval_path),
        "safety": dict(safety or {}),
        "fail_reasons": fail_reasons or [],
        "next_unblock": next_unblock,
        "source_artifacts": [_repo_rel(path) or path for path in (source_artifacts or [])],
    }


def _renderer_opportunities(output_dir: Path) -> list[dict[str, Any]]:
    plan_path = (
        REPO_ROOT
        / "experiments/results/c091_renderer_self_compression_bigmove_20260503_worker/plan.json"
    )
    plan = _load_json(plan_path)
    if plan is None:
        return []
    by_id = {
        str(row.get("candidate_id")): row
        for row in plan.get("candidates", [])
        if isinstance(row, Mapping)
    }

    rows: list[dict[str, Any]] = []
    for candidate_id, rank_group, label in (
        ("qzs3_b0064_c091_p3_preserved_minp_slices", 10, "near-miss C091 QZS3 block64 renderer reblock"),
        ("qzs3_b0096_c091_p3_preserved_minp_slices", 20, "byte-sufficient C091 QZS3 block96 renderer reblock"),
    ):
        row = by_id.get(candidate_id)
        if row is None:
            continue
        if candidate_id.startswith("qzs3_b0064"):
            preflight_path = output_dir / "qzs3_b0064_pose_safety_preflight.json"
            preflight = _load_json(preflight_path)
        else:
            pose_safety = row.get("pose_safety")
            preflight_path = None
            preflight = pose_safety if isinstance(pose_safety, Mapping) else None
        safety = _pose_safety_summary(preflight, preflight_path)
        archive_bytes = int(row["archive_bytes"])
        be = break_even(candidate_bytes=archive_bytes)
        fail_reasons = []
        if not be["strict_sub314_at_unchanged_components"]:
            fail_reasons.append(
                f"needs {be['sub314_equivalent_bytes_needed_after_candidate']} more byte-equivalent "
                "savings or component gain after rate"
            )
        if not safety.get("safe_for_exact_eval_dispatch", False):
            fail_reasons.append("renderer output parity failed local pose-safety")
        rows.append(
            opportunity(
                opportunity_id=candidate_id,
                rank_group=rank_group,
                title=label,
                status="blocked_pose_safety",
                archive_bytes=archive_bytes,
                archive_sha256=str(row.get("archive_sha256")),
                archive_path=str(row.get("archive")),
                evidence_grade="empirical_local_preflight_no_score",
                fail_reasons=fail_reasons,
                next_unblock=(
                    "requires a parity-aware renderer encoder or recovery training; "
                    "do not exact-eval the raw reblock"
                ),
                safety=safety,
                source_artifacts=[str(plan_path)],
            )
        )
    return rows


def _queued_opportunities() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pose_recs_path = (
        REPO_ROOT
        / "experiments/results/c091_pose_manifold_bigmove_20260503_worker/"
        "exact_eval_recommendations.json"
    )
    recs_payload = json.loads(pose_recs_path.read_text(encoding="utf-8")) if pose_recs_path.is_file() else []
    if isinstance(recs_payload, list):
        for item in recs_payload:
            if not isinstance(item, Mapping):
                continue
            if item.get("candidate_id") != "c091_native_cem_pose_waterfill_top128_s025":
                continue
            archive = item.get("archive") if isinstance(item.get("archive"), Mapping) else {}
            economics = item.get("economics") if isinstance(item.get("economics"), Mapping) else {}
            rows.append(
                opportunity(
                    opportunity_id=str(item["candidate_id"]),
                    rank_group=5,
                    title="parent-queued C091-native CEM pose/manifold top128",
                    status="excluded_parent_queued",
                    archive_bytes=int(archive.get("archive_bytes")),
                    archive_sha256=str(archive.get("archive_sha256")),
                    archive_path=str(archive.get("path")),
                    evidence_grade="empirical_proxy_plus_local_roundtrip",
                    component_gain_proxy=float(economics.get("expected_component_gain_proxy", 0.0)),
                    fail_reasons=["already queued by parent; do not duplicate"],
                    next_unblock="wait for exact T4 result and update the frontier from structured JSON",
                    source_artifacts=[str(pose_recs_path)],
                    parent_queued=True,
                )
            )

    pr65_path = (
        REPO_ROOT / "experiments/results/pr65_henosis_stream_transfer_20260503_codex/candidate_matrix.json"
    )
    pr65 = _load_json(pr65_path)
    if pr65 is not None:
        for item in pr65.get("candidates", []):
            if not isinstance(item, Mapping):
                continue
            if item.get("candidate_id") != "c091_pr65_pose_qp1_c089_actions_p6":
                continue
            rows.append(
                opportunity(
                    opportunity_id=str(item["candidate_id"]),
                    rank_group=6,
                    title="parent-queued PR65 pose fix1 stream transfer",
                    status="excluded_parent_queued",
                    archive_bytes=int(item["archive_bytes"]),
                    archive_sha256=str(item["archive_sha256"]),
                    archive_path=str(item["archive"]),
                    evidence_grade="empirical_local_roundtrip_no_score",
                    component_gain_proxy=float(
                        item.get("selected_pair_trace_positive_advantage_sum", {}).get("combined", 0.0)
                    ),
                    fail_reasons=["already queued by parent; do not duplicate"],
                    next_unblock="wait for exact T4 result; current local trace proxy is not positive",
                    source_artifacts=[str(pr65_path)],
                    parent_queued=True,
                )
            )
    return rows


def _mask_and_action_opportunities() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mask_plan_path = REPO_ROOT / "experiments/results/c091_mask_packer_bigmove_20260503_codex/plan.json"
    mask_plan = _load_json(mask_plan_path)
    if mask_plan is not None:
        candidates = [
            item for item in mask_plan.get("candidates", []) if isinstance(item, Mapping)
        ]
        for item in candidates:
            if item.get("candidate_id") == "public_renderer_c089_p6_lossless_stream_resweep":
                rows.append(
                    opportunity(
                        opportunity_id=str(item["candidate_id"]),
                        rank_group=40,
                        title="public renderer plus C089 P6 lossless stream resweep",
                        status="blocked_break_even_and_renderer_gate",
                        archive_bytes=int(item["archive_bytes"]),
                        archive_sha256=str(item["archive_sha256"]),
                        archive_path=str(item["archive_path"]),
                        evidence_grade="empirical_byte_screen_only",
                        fail_reasons=[
                            "saves only 357 bytes versus C091",
                            "still needs about 0.000928 component-score gain",
                            "renderer transplant gate remains unresolved",
                        ],
                        next_unblock="only revisit if a renderer-safety proof and component-positive trace exist",
                        source_artifacts=[str(mask_plan_path)],
                    )
                )
                break
        lossless = mask_plan.get("mask_lossless_probe")
        if isinstance(lossless, Mapping):
            savings = int(lossless.get("best_savings_bytes", lossless.get("savings_bytes", 0)))
            rows.append(
                opportunity(
                    opportunity_id="c091_mask_lossless_brotli_resweep",
                    rank_group=60,
                    title="exact-lossless PR75 mask Brotli resweep",
                    status="safe_but_too_small",
                    archive_bytes=FRONTIER_BYTES - savings,
                    archive_sha256=None,
                    evidence_grade="empirical_lossless_byte_screen",
                    fail_reasons=[
                        f"saves only {savings} mask-stream bytes",
                        "self-describing header overhead dominates standalone use",
                    ],
                    next_unblock="bundle only as polish behind a larger byte move",
                    source_artifacts=[str(mask_plan_path)],
                )
            )

    action_path = (
        REPO_ROOT / "experiments/results/c091_native_action_atoms_20260503_codex/ranked_atom_policy.json"
    )
    action_policy = _load_json(action_path)
    if action_policy is not None:
        upper = action_policy.get("policy_upper_bound")
        break_even_row = action_policy.get("break_even_for_best_byte_screen")
        dispatch = action_policy.get("dispatch_decision")
        upper_bound = (
            float(upper.get("modelled_component_improvement_upper_bound", 0.0))
            if isinstance(upper, Mapping)
            else 0.0
        )
        gain_needed = (
            float(break_even_row.get("sub314_component_score_improvement_needed", 0.0))
            if isinstance(break_even_row, Mapping)
            else 0.0
        )
        rows.append(
            opportunity(
                opportunity_id="c091_native_action_atom_upper_bound",
                rank_group=50,
                title="C091-native action atom policy upper bound",
                status="blocked_component_break_even",
                archive_bytes=FRONTIER_BYTES
                + int(break_even_row.get("archive_delta_bytes_vs_c091", 0))
                if isinstance(break_even_row, Mapping)
                else None,
                evidence_grade="exact_trace_feedback_planning_only",
                component_gain_proxy=upper_bound,
                fail_reasons=[
                    f"component proxy upper bound {upper_bound:.12f} is below needed {gain_needed:.12f}",
                    str(dispatch.get("reason")) if isinstance(dispatch, Mapping) else "no dispatch decision",
                ],
                next_unblock="needs new non-action byte move or a measured C091-native component response larger than this upper bound",
                source_artifacts=[str(action_path)],
            )
        )
    return rows


def _large_byte_negative_opportunities() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    geom_path = REPO_ROOT / "experiments/results/geometry_safe_mask_overlay_search_20260503_worker/candidate_matrix.json"
    geom = _load_json(geom_path)
    if geom is not None:
        lossy_rows = [
            row
            for row in geom.get("candidates", [])
            if isinstance(row, Mapping)
            and str(row.get("candidate_id", "")).startswith("bounded_mask_reencode")
        ]
        if lossy_rows:
            best = min(lossy_rows, key=lambda row: int(row.get("archive_bytes", 10**12)))
            rows.append(
                opportunity(
                    opportunity_id="geometry_bounded_mask_reencode_family",
                    rank_group=70,
                    title="large mask byte-cut family with decoded class flips",
                    status="blocked_geometry_safety",
                    archive_bytes=int(best["archive_bytes"]),
                    archive_sha256=str(best.get("archive_sha256")),
                    archive_path=str(best.get("archive_path")),
                    evidence_grade="empirical_geometry_negative",
                    fail_reasons=[
                        "large byte saves change decoded mask classes",
                        "exact CDO1 repair overlays cost hundreds of KB after compression",
                    ],
                    next_unblock="needs a geometry-preserving mask coder, not another lossy AV1 transcode",
                    source_artifacts=[str(geom_path)],
                )
            )

    pr77_exact = (
        REPO_ROOT
        / "experiments/results/lightning_batch/"
        "exact_eval_pr77_action_pose_fixedslice_t4_20260503T114254Z/"
        "contest_auth_eval.adjudicated.json"
    )
    pr77_payload = _load_json(pr77_exact)
    if pr77_payload is not None:
        rows.append(
            opportunity(
                opportunity_id="pr77_actions_c089_pose_fixedslice_exact_negative",
                rank_group=80,
                title="PR77 action stream with C089 pose exact-negative",
                status="measured_negative",
                archive_bytes=int(pr77_payload["archive_size_bytes"]),
                archive_sha256=str(pr77_payload.get("provenance", {}).get("archive_sha256")),
                exact_score=float(pr77_payload["canonical_score"]),
                exact_eval_path=str(pr77_exact),
                evidence_grade="A++_exact_negative",
                fail_reasons=["exact T4 score is worse than C091 by more than 0.003"],
                next_unblock="use only as pose-toxicity feedback for atom planners",
                source_artifacts=[str(pr77_exact)],
            )
        )
    return rows


def collect_opportunities(output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    queued = _queued_opportunities()
    active_ids = {row["opportunity_id"] for row in queued} | PARENT_QUEUED_IDS
    rows = (
        _renderer_opportunities(output_dir)
        + _mask_and_action_opportunities()
        + _large_byte_negative_opportunities()
    )
    rows = [row for row in rows if row["opportunity_id"] not in active_ids]
    rows.sort(key=lambda row: (int(row["rank_group"]), str(row["opportunity_id"])))
    queued.sort(key=lambda row: (int(row["rank_group"]), str(row["opportunity_id"])))
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    for index, row in enumerate(queued, start=1):
        row["excluded_rank"] = index
    return rows, queued


def recommendation_from(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    recommended = [row for row in rows if row.get("exact_eval_recommended")]
    if recommended:
        best = sorted(recommended, key=lambda row: (int(row["rank_group"]), str(row["opportunity_id"])))[0]
        return {
            "recommendation": "claim_lane_then_exact_cuda_eval",
            "reason": "a non-queued local candidate clears break-even and safety gates",
            "candidate": dict(best),
            "claim_required_before_dispatch": True,
            "remote_gpu_dispatch_performed": False,
            "dispatch_state_touched": False,
        }
    return {
        "recommendation": "do_not_dispatch",
        "reason": (
            "no new non-queued opportunity is both byte-sufficient or component-break-even "
            "and locally safe; parent-queued lanes are explicitly excluded"
        ),
        "candidate": None,
        "claim_required_before_dispatch": True,
        "remote_gpu_dispatch_performed": False,
        "dispatch_state_touched": False,
    }


def render_markdown(plan: Mapping[str, Any]) -> str:
    frontier = plan["frontier"]
    rec = plan["recommendation"]
    lines = [
        "# Public Floor Next Breakthrough Worker - 2026-05-03",
        "",
        "Evidence grade: `empirical_planning_only`.",
        "",
        "No remote GPU job was dispatched and no `.omx/state` dispatch claim was read or edited.",
        "",
        "## Anchor",
        "",
        f"- C091 score: `{FRONTIER_SCORE}`.",
        f"- C091 bytes/SHA-256: `{FRONTIER_BYTES}`, `{FRONTIER_SHA256}`.",
        f"- Strict unchanged-component bytes needed for `<0.314`: `{frontier['eval_json']['strict_unchanged_component_bytes_needed']}`.",
        "",
        "## Recommendation",
        "",
        f"`{rec['recommendation']}`: {rec['reason']}",
        "",
        "## Excluded Parent-Queued Lanes",
        "",
    ]
    for row in plan["excluded_parent_queued"]:
        econ = row["economics_vs_c091"]
        lines.append(
            f"- `{row['opportunity_id']}`: bytes `{row['archive']['bytes']}`, "
            f"needs component gain `{econ['sub314_component_score_gain_needed']}`; "
            "excluded to avoid duplicating the parent queue."
        )
    if not plan["excluded_parent_queued"]:
        lines.append("- None recorded.")
    lines.extend(["", "## Ranked Non-Queued Opportunities", ""])
    for row in plan["ranked_opportunities"]:
        econ = row["economics_vs_c091"]
        fail = "; ".join(row["fail_reasons"]) if row["fail_reasons"] else "not dispatch-ready"
        archive_bytes = row["archive"]["bytes"] if row["archive"]["bytes"] is not None else "n/a"
        archive_sha = row["archive"]["sha256"] or "n/a"
        unchanged_score = econ["score_if_components_unchanged"]
        unchanged_score_display = unchanged_score if unchanged_score is not None else "n/a"
        gain_needed = econ["sub314_component_score_gain_needed"]
        gain_needed_display = gain_needed if gain_needed is not None else "n/a"
        lines.append(
            f"{row['rank']}. `{row['opportunity_id']}` - `{row['status']}`"
        )
        lines.append(
            f"   - bytes/SHA: `{archive_bytes}`, `{archive_sha}`"
        )
        lines.append(
            f"   - unchanged-component score: `{unchanged_score_display}`; "
            f"component gain still needed: `{gain_needed_display}`"
        )
        if row.get("safety"):
            safety = row["safety"]
            lines.append(
                f"   - safety: safe=`{safety.get('safe_for_exact_eval_dispatch')}`, "
                f"mean/rms/max=`{safety.get('mean_abs_delta')}`/`{safety.get('rms_delta')}`/`{safety.get('max_abs_delta')}`"
            )
        lines.append(f"   - why not: {fail}")
    lines.extend(
        [
            "",
            "## Verification",
            "",
            "- Planner artifacts are deterministic JSON/Markdown.",
            "- Score claims remain `false`; promotion eligible remains `false`.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_plan(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    ledger_md: Path = DEFAULT_LEDGER_MD,
    force: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    ledger_md = ledger_md.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"output directory is non-empty; pass --force: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    ranked, queued = collect_opportunities(output_dir)
    plan = {
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "promotion_eligible": False,
        "remote_gpu_dispatch_performed": False,
        "dispatch_state_touched": False,
        "no_archive_emitted": True,
        "generated_date": "2026-05-03",
        "target_score": TARGET_SCORE,
        "rate_score_per_byte": RATE_SCORE_PER_BYTE,
        "frontier": _frontier_custody(),
        "excluded_parent_queued": queued,
        "ranked_opportunities": ranked,
        "recommendation": recommendation_from(ranked),
    }
    plan_path = output_dir / "next_breakthrough_plan.json"
    md_path = output_dir / "next_breakthrough_plan.md"
    markdown = render_markdown(plan)
    plan_path.write_bytes(_json_bytes(plan))
    md_path.write_text(markdown, encoding="utf-8")
    ledger_md.parent.mkdir(parents=True, exist_ok=True)
    ledger_md.write_text(markdown, encoding="utf-8")
    return plan


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ledger-md", type=Path, default=DEFAULT_LEDGER_MD)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_plan(output_dir=args.output_dir, ledger_md=args.ledger_md, force=args.force)
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
