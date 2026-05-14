#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan contest-faithful counter-moves from public PR frontier signals.

This is a read-only planning tool. It can consume saved GitHub PR API JSON,
recompute public report math, classify likely innovation families, and emit
counter-design rows with explicit custody blockers. It does not build archives,
load scorer models, edit runtime paths, or dispatch GPU work.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any


ORIGINAL_VIDEO_BYTES = 37_545_489
RATE_POINTS_PER_BYTE = 25.0 / ORIGINAL_VIDEO_BYTES
SCHEMA = "nostradamus_future_frontier_v1"
TOOL = "experiments/plan_nostradamus_future_frontier.py"

DEFAULT_ANCHOR_JSON = (
    Path(__file__).resolve().parents[1]
    / "experiments/results/lightning_batch/"
    / "exact_eval_pr85_stbm1br_stbm_runtime_t4_g4dn2x_20260504T0613Z/"
    / "contest_auth_eval.adjudicated.json"
)


class NostradamusPlanError(ValueError):
    """Raised when explicit planner input is malformed."""


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise NostradamusPlanError(f"invalid JSON input: {path}") from exc


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def score_from_components(*, archive_bytes: int, seg_dist: float, pose_dist: float) -> float:
    return (
        100.0 * float(seg_dist)
        + math.sqrt(10.0 * float(pose_dist))
        + 25.0 * int(archive_bytes) / ORIGINAL_VIDEO_BYTES
    )


def rate_score_delta(delta_bytes: int) -> float:
    return int(delta_bytes) * RATE_POINTS_PER_BYTE


def bytes_for_score_delta(score_delta: float) -> int:
    """Return the strict byte count needed to cover a positive score delta."""

    if score_delta <= 0:
        return 0
    return math.ceil(float(score_delta) / RATE_POINTS_PER_BYTE)


def _extract_number(patterns: list[str], text: str) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1).replace(",", ""))
    return None


def parse_report_text(text: str) -> dict[str, Any]:
    """Extract public report fields from PR body text."""

    body = str(text or "")
    pose = _extract_number(
        [
            r"Average\s+PoseNet\s+Distortion:\s*([0-9][0-9.,eE+-]*)",
            r"PoseNet:\s*`?([0-9][0-9.,eE+-]*)`?",
            r"posenet[_\s-]*dist(?:ortion)?:\s*`?([0-9][0-9.,eE+-]*)`?",
        ],
        body,
    )
    seg = _extract_number(
        [
            r"Average\s+SegNet\s+Distortion:\s*([0-9][0-9.,eE+-]*)",
            r"SegNet:\s*`?([0-9][0-9.,eE+-]*)`?",
            r"segnet[_\s-]*dist(?:ortion)?:\s*`?([0-9][0-9.,eE+-]*)`?",
        ],
        body,
    )
    archive_bytes_raw = _extract_number(
        [
            r"Submission\s+file\s+size:\s*([0-9][0-9,]*)\s*bytes",
            r"archive\s*(?:size|bytes):\s*`?([0-9][0-9,]*)`?",
            r"claimed\s+archive\s+size:\s*`?([0-9][0-9,]*)`?",
        ],
        body,
    )
    final_score = _extract_number(
        [
            r"Final\s+score:.*?=\s*([0-9][0-9.,eE+-]*)",
            r"exact\s+score:\s*`?([0-9][0-9.,eE+-]*)`?",
            r"Final\s+promoted\s+score:\s*`?([0-9][0-9.,eE+-]*)`?",
        ],
        body,
    )
    archive_bytes = int(archive_bytes_raw) if archive_bytes_raw is not None else None
    recomputed = None
    if archive_bytes is not None and seg is not None and pose is not None:
        recomputed = score_from_components(
            archive_bytes=archive_bytes,
            seg_dist=seg,
            pose_dist=pose,
        )
    return {
        "archive_bytes": archive_bytes,
        "seg_dist": seg,
        "pose_dist": pose,
        "reported_final_score": final_score,
        "score_recomputed_from_report_components": recomputed,
        "score_claim_is_external": bool(
            archive_bytes is not None or seg is not None or pose is not None or final_score is not None
        ),
    }


def anchor_from_exact_eval(payload: dict[str, Any], *, path: str | None = None) -> dict[str, Any]:
    archive_bytes = _as_int(payload.get("archive_size_bytes"))
    seg_dist = _as_float(payload.get("avg_segnet_dist"))
    pose_dist = _as_float(payload.get("avg_posenet_dist"))
    score = _as_float(payload.get("score_recomputed_from_components"))
    if score is None:
        score = _as_float(payload.get("canonical_score"))
    if archive_bytes is None or seg_dist is None or pose_dist is None or score is None:
        raise NostradamusPlanError("anchor exact-eval JSON is missing archive/components/score")
    provenance = payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {}
    return {
        "path": path,
        "archive_bytes": archive_bytes,
        "archive_sha256": provenance.get("archive_sha256"),
        "score": score,
        "seg_dist": seg_dist,
        "pose_dist": pose_dist,
        "score_seg_contribution": 100.0 * seg_dist,
        "score_pose_contribution": math.sqrt(10.0 * pose_dist),
        "score_rate_contribution": 25.0 * archive_bytes / ORIGINAL_VIDEO_BYTES,
        "evidence_grade": "A++" if provenance.get("device") == "cuda" else "A_or_lower",
        "device": provenance.get("device"),
        "gpu_model": provenance.get("gpu_model"),
        "n_samples": payload.get("n_samples"),
    }


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _pr_text(pr: dict[str, Any], files: list[dict[str, Any]] | None = None) -> tuple[str, str]:
    primary = " ".join([str(pr.get("title", "")), str(pr.get("body", ""))]).lower()
    filenames = " ".join(str(f.get("filename", "")) for f in files or []).lower()
    return primary, filenames


def classify_families(pr: dict[str, Any], files: list[dict[str, Any]] | None = None) -> list[str]:
    primary_text, file_text = _pr_text(pr, files)
    text = " ".join([primary_text, file_text])
    families: list[str] = []
    primary_checks = [
        ("source_embedded_payload_loophole", ("100_bytes", "base85", "dummy archive")),
        ("hpm1_hpac_mask_entropy", ("hpac", "hpm1", "pr86_hpac", "constriction")),
        ("semantic_geometry_mask_recode", ("stbm", "topband", "qrepro", "semantic")),
        ("qzs3_range_joint_sidechannel", ("qzs3", "range", "joint", "adaptive masking")),
        ("pose_manifold_qpose_action", ("qpose", "poseq", "tile-action", "tile_action")),
        ("low_frequency_bias_or_qrgb", ("qrgb", "final_bias", "bias", "low-frequency")),
        ("quantizr_flatpack", ("quantizr", "flatpup", "crf50")),
    ]
    for family, needles in primary_checks:
        if _contains_any(primary_text, needles):
            families.append(family)
    # File names are a weaker signal than the PR body/title because late public
    # PRs sometimes include neighboring historical submissions in the diff.
    # Only let file paths add families when no primary family was identified.
    if not families:
        for family, needles in primary_checks:
            if _contains_any(file_text, needles):
                families.append(family)
    return families or ["unknown_public_pr_family"]


def public_pr_row(
    pr: dict[str, Any],
    *,
    files: list[dict[str, Any]] | None = None,
    anchor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = parse_report_text(str(pr.get("body") or ""))
    families = classify_families(pr, files)
    number = pr.get("number")
    url = pr.get("html_url") or (
        f"https://github.com/commaai/comma_video_compression_challenge/pull/{number}"
        if number is not None
        else None
    )
    score_delta = None
    bytes_delta = None
    strict_bytes_needed_to_match_anchor = None
    if anchor is not None:
        candidate_score = report.get("score_recomputed_from_report_components")
        if candidate_score is not None:
            score_delta = candidate_score - anchor["score"]
            strict_bytes_needed_to_match_anchor = bytes_for_score_delta(score_delta)
        candidate_bytes = report.get("archive_bytes")
        if candidate_bytes is not None:
            bytes_delta = candidate_bytes - anchor["archive_bytes"]
    non_faithful = "source_embedded_payload_loophole" in families
    score_device = "external_unverified"
    body_lower = str(pr.get("body") or "").lower()
    if "device: cuda" in body_lower:
        score_device = "public_report_cuda_unverified"
    elif "device: mps" in body_lower:
        score_device = "public_report_mps_invalid_for_promotion"
    elif report["score_claim_is_external"]:
        score_device = "public_report_device_unknown_unverified"
    return {
        "pr": number,
        "url": url,
        "title": pr.get("title"),
        "state": pr.get("state"),
        "merged": bool(pr.get("merged_at")),
        "author": (pr.get("user") or {}).get("login") if isinstance(pr.get("user"), dict) else None,
        "head_sha": (pr.get("head") or {}).get("sha") if isinstance(pr.get("head"), dict) else None,
        "created_at": pr.get("created_at"),
        "updated_at": pr.get("updated_at"),
        "families": families,
        "report": report,
        "score_device_class": score_device,
        "delta_vs_anchor": {
            "score": score_delta,
            "archive_bytes": bytes_delta,
            "neutral_rate_score_delta": rate_score_delta(bytes_delta) if bytes_delta is not None else None,
            "strict_bytes_needed_to_match_anchor": strict_bytes_needed_to_match_anchor,
        },
        "score_claim_status": "invalid_external_loophole"
        if non_faithful
        else "external_until_local_exact_cuda_replay",
    }


def _countermove_catalog(anchor: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        "hpm1_hpac_mask_entropy": {
            "move": "Competitors compress PR85-style semantic masks with learned HPAC/HPM1 entropy models.",
            "counter_design": (
                "Recover full HPM1 decode/reencode parity, then train or refit an Apogee-owned "
                "mask entropy coder over PR85/STBM token fields. Dispatch only after typed mask "
                "segment parity and runtime output parity."
            ),
            "existing_files_tools": [
                "src/tac/pr86_hpac_codec.py or runtime-rs/crates/hpac-codec",
                "experiments/profile_pr85_residual_sufficient_program.py",
                "experiments/preflight_pr85_fixed_runtime_readiness.py",
                "experiments/contest_auth_eval.py",
            ],
            "implementation_status": "blocked_on_hpm1_full_decode_reencode_parity",
            "custody_gates": [
                "no local score claim from PR91 until HPM1 entropy decode succeeds",
                "candidate must preserve archive/member SHA custody and runtime tree hash",
                "Level-2 lane claim required before exact CUDA dispatch",
            ],
        },
        "semantic_geometry_mask_recode": {
            "move": "Competitors decompose the semantic mask into geometry priors plus sparse residuals.",
            "counter_design": (
                "Extend the STBM1BR positive pure-rate lane into geometry-aware mask coding while "
                "keeping SegNet/PoseNet components unchanged at JSON precision."
            ),
            "existing_files_tools": [
                "src/tac/stbm1br_mask_codec.py",
                "experiments/build_pr85_stbm1br_mask_recode_candidate.py",
                "experiments/profile_pr85_residual_sufficient_program.py",
                "scripts/pre_submission_compliance_check.py",
            ],
            "implementation_status": "ready_for_planning_and_local_parity_not_dispatch",
            "custody_gates": [
                "mask decode parity before archive build",
                "pre-submission compliance pass",
                "exact T4 eval required for any score/rank claim",
            ],
        },
        "qzs3_range_joint_sidechannel": {
            "move": "Competitors keep the PR85/QZS3 basin and search tiny side channels around it.",
            "counter_design": (
                "Treat PR92-style joint/range submissions as near-neighbor probes: compare exact "
                "segments against PR85/STBM, mine changed side-channel bytes, and only lower atoms "
                "that are non-noop and break-even against exact component risk."
            ),
            "existing_files_tools": [
                "src/tac/pr85_bundle.py",
                "experiments/plan_pr85_full_stack_opportunity_matrix.py",
                "experiments/plan_frontier_stack_reconstruction.py",
                "experiments/preflight_pr85_fixed_runtime_readiness.py",
            ],
            "implementation_status": "ready_for_static_diff_after_archive_intake",
            "custody_gates": [
                "public archive must be downloaded by SHA before comparison",
                "raw wire slices, not metadata-order slices, define charged deltas",
                "exact eval only if formula math can beat the STBM anchor",
            ],
        },
        "pose_manifold_qpose_action": {
            "move": "Competitors spend small bytes on qpose/tile-action pose manifold tweaks.",
            "counter_design": (
                "Use exact pose-action deltas only as signed training labels until a charged "
                "candidate beats the byte/component break-even under local parity."
            ),
            "existing_files_tools": [
                "experiments/build_qp1_pose_active_subspace_candidates.py",
                "experiments/optimize_poses.py",
                "experiments/build_pr85_pair_action_candidates.py",
                "experiments/contest_auth_eval.py",
            ],
            "implementation_status": "blocked_without_grounded_action_value_evidence",
            "custody_gates": [
                "MPS/CPU pose reports are invalid for promotion",
                "QZS3 CUDA batch safety applies to any pose-regeneration support tool",
                "do not dispatch broad qpose grids without charged archive deltas",
            ],
        },
        "low_frequency_bias_or_qrgb": {
            "move": "Competitors add tiny RGB/bias residual controls for hard pairs.",
            "counter_design": (
                "Preserve QRGB/final-bias builders as transfer machinery, but require a new "
                "target basin or exact positive singleton before stacking."
            ),
            "existing_files_tools": [
                "experiments/build_pr85_qrgb_pair_atom_archive_candidates.py",
                "experiments/build_pr85_qrgb_pair_atom_combo_candidates.py",
                "experiments/build_pr85_final_bias_stack_candidates.py",
            ],
            "implementation_status": "measured_negative_on_pr85_singletons",
            "custody_gates": [
                "do not queue combo archives after negative singleton exact evidence",
                "new basin requires fresh local parity and exact singleton evidence",
            ],
        },
        "quantizr_flatpack": {
            "move": "Competitors package older Quantizr/flatpack basins cleanly for public intake.",
            "counter_design": (
                "Use these as regression baselines and public anatomy fixtures, not as the "
                "next promotion lane unless their exact CUDA components beat STBM."
            ),
            "existing_files_tools": [
                "experiments/reverse_engineer_top_submissions.py",
                "experiments/plan_frontier_stack_reconstruction.py",
                "scripts/pre_submission_compliance_check.py",
            ],
            "implementation_status": "ready_for_forensic_intake_low_frontier_priority",
            "custody_gates": [
                "archive payload must be charged, not embedded in source",
                "external reports remain non-promotable until local exact CUDA replay",
            ],
        },
        "source_embedded_payload_loophole": {
            "move": "Competitors may exploit source-embedded payloads or dummy tiny archives.",
            "counter_design": (
                "Keep loophole submissions quarantined as harness forensics and preserve the "
                "source-embedded payload guard for internal exact-eval submitters."
            ),
            "existing_files_tools": [
                "scripts/launch_lightning_batch_job.py",
                "src/tac/tests/test_public_replay_exact_eval_hardening.py",
            ],
            "implementation_status": "guarded_invalid_external_only",
            "custody_gates": [
                "not contest-faithful under local Apogee rules",
                "never use for rank/promotion or paper evidence",
            ],
        },
    }


def anticipated_moves(anchor: dict[str, Any], pr_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    catalog = _countermove_catalog(anchor)
    family_hits: dict[str, list[dict[str, Any]]] = {key: [] for key in catalog}
    for row in pr_rows:
        for family in row["families"]:
            if family in family_hits:
                family_hits[family].append(row)
    priority = [
        "hpm1_hpac_mask_entropy",
        "semantic_geometry_mask_recode",
        "qzs3_range_joint_sidechannel",
        "pose_manifold_qpose_action",
        "low_frequency_bias_or_qrgb",
        "quantizr_flatpack",
        "source_embedded_payload_loophole",
    ]
    rows: list[dict[str, Any]] = []
    for family in priority:
        base = catalog[family]
        public_hits = family_hits[family]
        best_external_score = None
        for hit in public_hits:
            score = hit["report"].get("score_recomputed_from_report_components")
            if score is not None:
                best_external_score = score if best_external_score is None else min(best_external_score, score)
        row = {
            "family": family,
            "public_prs": [hit["pr"] for hit in public_hits],
            "public_urls": [hit["url"] for hit in public_hits if hit.get("url")],
            "best_external_score_from_report": best_external_score,
            "score_gap_vs_anchor_if_report_true": (
                None if best_external_score is None else best_external_score - anchor["score"]
            ),
            **base,
        }
        rows.append(row)
    return [{**row, "rank": idx + 1} for idx, row in enumerate(rows[:5])]


def build_plan(
    *,
    anchor_payload: dict[str, Any],
    public_prs: list[dict[str, Any]],
    pr_files: dict[int, list[dict[str, Any]]] | None = None,
    anchor_path: str | None = None,
) -> dict[str, Any]:
    anchor = anchor_from_exact_eval(anchor_payload, path=anchor_path)
    pr_files = pr_files or {}
    rows = [
        public_pr_row(pr, files=pr_files.get(int(pr.get("number", -1)), []), anchor=anchor)
        for pr in public_prs
        if isinstance(pr, dict)
    ]
    rows.sort(key=lambda row: (row.get("pr") is None, -(row.get("pr") or 0)))
    moves = anticipated_moves(anchor, rows)
    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "score_formula": "100*seg_dist + sqrt(10*pose_dist) + 25*archive_bytes/37545489",
        "score_truth": "planning_only; exact score truth remains archive.zip -> inflate.sh -> upstream/evaluate.py on CUDA",
        "anchor": anchor,
        "public_pr_rows": rows,
        "anticipated_moves_top5": moves,
        "highest_ev_concrete_tool": {
            "built": TOOL,
            "why": (
                "It turns fast-moving PR92-PR100 public claims into score math, family classification, "
                "counter-designs, and exact-eval custody gates before implementation or dispatch."
            ),
            "default_side_effects": "stdout only unless --json-out is supplied",
        },
    }


def _load_public_prs(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = _read_json(path)
        if isinstance(payload, list):
            rows.extend(item for item in payload if isinstance(item, dict))
        elif isinstance(payload, dict):
            rows.append(payload)
        else:
            raise NostradamusPlanError(f"PR JSON must be object or list: {path}")
    return rows


def _fetch_github_pr(number: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base = "https://api.github.com/repos/commaai/comma_video_compression_challenge/pulls"
    with urllib.request.urlopen(f"{base}/{number}", timeout=20) as handle:
        pr = json.load(handle)
    with urllib.request.urlopen(f"{base}/{number}/files?per_page=100", timeout=20) as handle:
        files = json.load(handle)
    if not isinstance(pr, dict) or not isinstance(files, list):
        raise NostradamusPlanError(f"unexpected GitHub API payload for PR{number}")
    return pr, [item for item in files if isinstance(item, dict)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-json", type=Path, default=DEFAULT_ANCHOR_JSON)
    parser.add_argument(
        "--public-pr-json",
        type=Path,
        action="append",
        default=[],
        help="Saved GitHub PR API JSON object or list. May be passed multiple times.",
    )
    parser.add_argument(
        "--github-pr",
        type=int,
        action="append",
        default=[],
        help="Fetch a live public GitHub PR and file list read-only.",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)

    anchor_payload = _read_json(args.anchor_json)
    if not isinstance(anchor_payload, dict):
        raise NostradamusPlanError("--anchor-json must contain a JSON object")
    public_prs = _load_public_prs(args.public_pr_json)
    pr_files: dict[int, list[dict[str, Any]]] = {}
    for number in args.github_pr:
        pr, files = _fetch_github_pr(number)
        public_prs.append(pr)
        pr_files[number] = files

    plan = build_plan(
        anchor_payload=anchor_payload,
        public_prs=public_prs,
        pr_files=pr_files,
        anchor_path=str(args.anchor_json),
    )
    text = _json_text(plan)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
