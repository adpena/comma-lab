#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Convert a completed deep-research wave memo into a guarded intake queue.

The comprehensive research wave is valuable frontier signal, but its predicted
bands are not score evidence and not dispatch authority. This helper extracts
the TOP-5 table into a machine-readable artifact that keeps the signal alive
while preserving Pact false-authority discipline:

  * score_claim=false
  * promotion_eligible=false
  * ready_for_paid_dispatch=false
  * explicit blockers before byte-closed probes or paid provider work

Usage:

    .venv/bin/python tools/research_wave_intake_queue.py --json

    .venv/bin/python tools/research_wave_intake_queue.py --write-artifact
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

from asymptotic_pursuit_candidate_readiness_assessment import (  # noqa: E402
    assess_candidates,
)

DEFAULT_MEMO_REL = ".omx/research/comprehensive_research_wave_20260518.md"
DEFAULT_LANE_ID = "lane_deep_research_wave_20260518"
TOP5_HEADING = "### TOP-5 substrate reformulations from new evidence"

COMMON_FALSE_AUTHORITY_BLOCKERS: tuple[str, ...] = (
    "research_wave_prediction_not_score_authority",
    "requires_byte_closed_archive_or_runtime_probe_before_score_claim",
    "requires_operator_session_directive_budget_and_lane_claim_before_provider_dispatch",
    "requires_paired_contest_cuda_cpu_harvest_before_promotion_or_ranking_claim",
)


SUBSTRATE_CROSSWALK: tuple[tuple[str, str, str], ...] = (
    (
        r"\bTT5L\b|time[-_ ]traveler",
        "time_traveler_l5_autonomy",
        "resolve_modal_billing_or_lightning_doctor_env_then_stage_manifest_and_claim",
    ),
    (
        r"\bZ7\b|Mamba",
        "time_traveler_l5_z7_lstm_predictive_coding",
        "land_z7_mamba_design_memo_recipe_and_timing_smoke",
    ),
    (
        r"\bATW\b",
        "atw_codec_v2",
        "design_substrate_native_scorer_logit_sketch_or_trained_atw_residual_probe",
    ),
    (
        r"\bDP1\b",
        "dp1_pr101_composition",
        "run_full_frame_parity_or_path2_lambda_prior_disambiguator_after_l1_noop_probe",
    ),
    (
        r"lane_17_imp|Frankle|LTH",
        "lane_17_imp",
        "run_imp_cycle0_timing_smoke_after_operator_budget_and_claim",
    ),
)


def _strip_markdown(value: str) -> str:
    value = re.sub(r"\*\*([^*]+)\*\*", r"\1", value)
    value = re.sub(r"`([^`]+)`", r"\1", value)
    value = value.replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", value).strip()


def _split_markdown_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [_strip_markdown(cell) for cell in line.split("|")]


def _parse_frontmatter(text: str) -> dict[str, str]:
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---", 4)
    if end == -1:
        return {}
    out: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if ":" not in line or line.startswith("  "):
            continue
        key, value = line.split(":", 1)
        out[key.strip()] = value.strip().strip('"')
    return out


def parse_top5_rows(text: str) -> list[dict[str, str]]:
    """Extract the TOP-5 Markdown table rows from the research memo."""

    lines = text.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == TOP5_HEADING)
    except StopIteration as exc:
        raise ValueError(f"missing heading: {TOP5_HEADING}") from exc

    table_lines: list[str] = []
    collecting = False
    for line in lines[start + 1 :]:
        if line.lstrip().startswith("|"):
            collecting = True
            table_lines.append(line)
            continue
        if collecting:
            break

    rows: list[dict[str, str]] = []
    for line in table_lines:
        cells = _split_markdown_row(line)
        if len(cells) < 6:
            continue
        if cells[0] == "#" or set(cells[0]) <= {"-"}:
            continue
        if not cells[0].strip().isdigit():
            continue
        rows.append(
            {
                "rank": cells[0],
                "substrate": cells[1],
                "reformulation": cells[2],
                "predicted_delta_s_band_raw": cells[3],
                "first_principles_citation": cells[4],
                "approx_cost": cells[5],
            }
        )
    if not rows:
        raise ValueError("TOP-5 table had no parseable candidate rows")
    return rows


def _parse_band(cell: str) -> dict[str, Any]:
    ranges = re.findall(
        r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]",
        cell,
    )
    delta_band: list[float | None] = [None, None]
    frontier_band: list[float | None] = [None, None]
    if ranges:
        delta_band = [float(ranges[0][0]), float(ranges[0][1])]
    if len(ranges) > 1:
        frontier_band = [float(ranges[-1][0]), float(ranges[-1][1])]

    axis_match = re.search(r"\[(contest-[A-Za-z]+|macOS-[^\]]+)\]", cell)
    return {
        "predicted_delta_s_band": delta_band,
        "predicted_frontier_score_band": frontier_band,
        "score_axis": axis_match.group(1) if axis_match else "unknown",
    }


def _finite_pair(values: list[float | None]) -> tuple[float, float] | None:
    if len(values) != 2:
        return None
    lo, hi = values
    if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
        return None
    if not math.isfinite(float(lo)) or not math.isfinite(float(hi)):
        return None
    return float(lo), float(hi)


def _prediction_band_blockers(band: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    delta = _finite_pair(list(band["predicted_delta_s_band"]))
    frontier = _finite_pair(list(band["predicted_frontier_score_band"]))

    if delta is None:
        blockers.append("predicted_delta_s_band_missing_or_malformed")
    else:
        delta_lo, delta_hi = delta
        if delta_lo > delta_hi:
            blockers.append("predicted_delta_s_band_order_invalid")
        if delta_hi >= 0.0:
            blockers.append("predicted_delta_s_band_not_strictly_score_lowering")

    if frontier is None:
        blockers.append("predicted_frontier_score_band_missing_or_malformed")
    else:
        frontier_lo, frontier_hi = frontier
        if frontier_lo > frontier_hi:
            blockers.append("predicted_frontier_score_band_order_invalid")
        if frontier_lo <= 0.0:
            blockers.append("predicted_frontier_score_band_nonpositive")

    return blockers


def _parse_cost_band(cell: str) -> dict[str, Any]:
    # Sum explicit dollar ranges across combined stages, e.g.
    # "$5-7 CPU probe + ~$15-25 Modal A100" -> [20, 32].
    matches = list(
        re.finditer(
            r"\$\s*(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?",
            cell.replace("~", ""),
        )
    )
    low = 0.0
    high = 0.0
    for match in matches:
        first = float(match.group(1))
        second = float(match.group(2)) if match.group(2) else first
        low += min(first, second)
        high += max(first, second)
    return {
        "approx_cost_raw": cell,
        "approx_cost_low_usd": round(low, 3) if matches else None,
        "approx_cost_high_usd": round(high, 3) if matches else None,
    }


def _map_substrate(raw_substrate: str) -> tuple[str, str]:
    for pattern, substrate_id, next_gate in SUBSTRATE_CROSSWALK:
        if re.search(pattern, raw_substrate, re.IGNORECASE):
            return substrate_id, next_gate
    token = re.sub(r"[^a-z0-9]+", "_", raw_substrate.lower()).strip("_")
    return token or "unknown_substrate", "land_substrate_specific_intake_mapping"


def _readiness_lookup(repo_root: Path) -> tuple[dict[str, dict[str, Any]], str | None]:
    try:
        assessment = assess_candidates(repo_root=repo_root)
    except Exception as exc:  # pragma: no cover - defensive CLI guard.
        return {}, f"{type(exc).__name__}: {exc}"
    return {candidate.substrate_id: candidate.as_dict() for candidate in assessment.candidates}, None


def _candidate_from_row(
    row: dict[str, str],
    *,
    readiness_by_substrate: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    raw_substrate = row["substrate"]
    substrate_id, next_gate = _map_substrate(raw_substrate)
    readiness = readiness_by_substrate.get(substrate_id)

    blockers = list(COMMON_FALSE_AUTHORITY_BLOCKERS)
    current_readiness_verdict = None
    current_blocking_issues: list[str] = []
    recipe_path = None
    if readiness is None:
        blockers.append("no_current_asymptotic_readiness_entry")
    else:
        current_readiness_verdict = readiness.get("readiness_verdict")
        current_blocking_issues = list(readiness.get("blocking_issues") or [])
        recipe_path = readiness.get("recipe_path")
        if current_readiness_verdict != "READY":
            blockers.append(f"current_readiness_verdict_{current_readiness_verdict}")
        if current_blocking_issues:
            blockers.append("current_readiness_has_blocking_issues")

    band = _parse_band(row["predicted_delta_s_band_raw"])
    prediction_band_blockers = _prediction_band_blockers(band)
    blockers.extend(prediction_band_blockers)
    if band["score_axis"] != "contest-CUDA":
        blockers.append(f"prediction_axis_is_{band['score_axis']}_not_contest_cuda")

    return {
        "rank": int(row["rank"]),
        "raw_substrate": raw_substrate,
        "substrate_id": substrate_id,
        "reformulation": row["reformulation"],
        "first_principles_citation": row["first_principles_citation"],
        **band,
        "prediction_band_valid_for_queue": not prediction_band_blockers,
        "prediction_band_blockers": tuple(prediction_band_blockers),
        **_parse_cost_band(row["approx_cost"]),
        "recommended_next_gate": next_gate,
        "current_readiness_verdict": current_readiness_verdict,
        "current_readiness_blocking_issues": current_blocking_issues,
        "current_recipe_path": recipe_path,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_paid_dispatch": False,
        "evidence_grade": "research_intake",
        "provenance_kind": "RESEARCH_WAVE_PREDICTION_NOT_EMPIRICAL",
        "blockers": tuple(dict.fromkeys(blockers)),
    }


def _research_priority_order(candidates: list[dict[str, Any]]) -> list[str]:
    return [
        candidate["substrate_id"]
        for candidate in candidates
        if candidate["prediction_band_valid_for_queue"]
    ]


def _dispatch_actionable_priority_order(
    candidates: list[dict[str, Any]],
) -> list[str]:
    """Return candidates that can safely drive provider-dispatch prioritization."""

    out: list[str] = []
    for candidate in candidates:
        if not candidate["prediction_band_valid_for_queue"]:
            continue
        if candidate["score_axis"] != "contest-CUDA":
            continue
        if candidate["current_readiness_verdict"] != "READY":
            continue
        if candidate["current_readiness_blocking_issues"]:
            continue
        out.append(candidate["substrate_id"])
    return out


def build_intake_payload(
    text: str,
    *,
    repo_root: Path,
    source_memo_path: Path,
    readiness_by_substrate: dict[str, dict[str, Any]] | None = None,
    created_utc: str | None = None,
) -> dict[str, Any]:
    """Build the guarded research-wave intake payload."""

    frontmatter = _parse_frontmatter(text)
    rows = parse_top5_rows(text)
    readiness_error = None
    if readiness_by_substrate is None:
        readiness_by_substrate, readiness_error = _readiness_lookup(repo_root)
    candidates = [
        _candidate_from_row(row, readiness_by_substrate=readiness_by_substrate)
        for row in rows
    ]
    research_priority_order = _research_priority_order(candidates)
    dispatch_actionable_priority_order = _dispatch_actionable_priority_order(
        candidates
    )

    memo_bytes = text.encode("utf-8")
    return {
        "schema": "research_wave_intake_queue_v1",
        "created_utc": created_utc
        or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source_memo_path": str(source_memo_path),
        "source_memo_sha256": hashlib.sha256(memo_bytes).hexdigest(),
        "source_review_id": frontmatter.get("review_id"),
        "source_lane_id": frontmatter.get("lane_id", DEFAULT_LANE_ID),
        "source_research_only": frontmatter.get("research_only") == "true",
        "candidate_count": len(candidates),
        "research_priority_order": research_priority_order,
        "dispatch_actionable_priority_order": dispatch_actionable_priority_order,
        "actionable_priority_order": dispatch_actionable_priority_order,
        "actionable_priority_order_semantics": (
            "provider-dispatch actionable only: finite score-lowering band, "
            "contest-CUDA prediction axis, READY current readiness, and no "
            "current readiness blockers"
        ),
        "current_readiness_join_count": sum(
            1 for candidate in candidates if candidate["current_readiness_verdict"] is not None
        ),
        "current_readiness_load_error": readiness_error,
        "candidates": candidates,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_paid_dispatch": False,
        "evidence_grade": "research_intake",
        "provenance_kind": "RESEARCH_WAVE_PREDICTION_NOT_EMPIRICAL",
        "result_review_blockers": list(COMMON_FALSE_AUTHORITY_BLOCKERS),
    }


def write_artifact(payload: dict[str, Any], *, repo_root: Path) -> Path:
    artifact_dir = repo_root / ".omx" / "state" / "asymptotic_pursuit"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stamp = str(payload["created_utc"]).replace(":", "").replace("-", "")
    path = artifact_dir / f"research_wave_intake_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--memo-path", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="Emit JSON payload")
    parser.add_argument("--write-artifact", action="store_true")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    memo_path = args.memo_path or (repo_root / DEFAULT_MEMO_REL)
    text = memo_path.read_text(encoding="utf-8")
    payload = build_intake_payload(
        text,
        repo_root=repo_root,
        source_memo_path=memo_path,
    )
    if args.write_artifact:
        path = write_artifact(payload, repo_root=repo_root)
        print(f"[research-wave-intake] wrote artifact: {path}", file=sys.stderr)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("=== RESEARCH WAVE intake queue ===")
        print(f"  source: {payload['source_memo_path']}")
        print(f"  candidates: {payload['candidate_count']}")
        print(f"  joined_current_readiness: {payload['current_readiness_join_count']}")
        print("  false_authority: score_claim=false promotion_eligible=false")
        for candidate in payload["candidates"]:
            print(
                "  "
                f"#{candidate['rank']} {candidate['substrate_id']} "
                f"axis={candidate['score_axis']} "
                f"next={candidate['recommended_next_gate']} "
                f"blockers={len(candidate['blockers'])}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
