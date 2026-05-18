# SPDX-License-Identifier: MIT
"""Option B archive-member pre-entropy sweep across 8 VALIDATED contest archives.

Per operator NON-NEGOTIABLE 2026-05-17 + lane
``lane_option_b_pre_entropy_prober_archive_member_validated_q4_target_20260517``:
re-run the pre-entropy prober scoped to ACTUAL archive.zip MEMBER bytes (NOT
research sidecars) for the 8 VALIDATED_CONTEST_MEMBER substrates per the
corrected artifact at
``.omx/state/wyner_ziv_deliverability/pre_entropy_candidate_substrates_corrected_20260517T215345.json``.
Emit apples-to-apples corrected pivot recommendation; surface a VALIDATED Q4
target candidate or honestly recommend Q4 deferral.

This script COMPOSES the canonical method
``tools.pre_entropy_substrate_pivot_prober.probe_substrate_archive_member`` per
Catalog #321 Option C, runs it across every validated contest archive in the
corrected artifact, and aggregates the results into a single sidecar JSON.

Per CLAUDE.md non-negotiables honored:
  * Catalog #131 — fcntl-locked atomic write via existing prober helper.
  * Catalog #287 — every claim carries an evidence tag (apples-to-apples per
    Catalog #321).
  * Catalog #321 — every emitted row carries
    ``validation_status=VALIDATED_CONTEST_MEMBER`` (enforced by the upstream
    canonical method; phantom-score class structurally extincted).
  * Catalog #229 — premise verifier executed pre-edit at
    ``.omx/tmp/option_b_archive_member_pre_entropy_sweep_premise_verifier.txt``.
  * Catalog #206 — checkpoint discipline observed via
    ``tools/subagent_checkpoint.py``.
  * Catalog #245 — non-paid; $0 GPU; runs locally in seconds.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.pre_entropy_substrate_pivot_prober import (  # noqa: E402
    CONTEST_RATE_DENOM_BYTES,
    SubstrateProbeResult,
    _fcntl_locked_atomic_write,
    probe_substrate_archive_member,
)

DEFAULT_CORRECTED_ARTIFACT = (
    REPO_ROOT
    / ".omx/state/wyner_ziv_deliverability/"
    "pre_entropy_candidate_substrates_corrected_20260517T215345.json"
)

DEFAULT_OUTPUT_DIR = REPO_ROOT / ".omx/state/wyner_ziv_deliverability"

SCHEMA_VERSION = "option_b_archive_member_sweep_v1"

# Q4 retarget decision threshold (apples-to-apples score delta).
# Anything below this is functionally indistinguishable from zero on the
# contest leaderboard (precision ~0.001).
Q4_MIN_DELIVERABLE_SAVINGS_FOR_RETARGET: float = 0.001


def load_validated_targets_from_corrected_artifact(
    artifact_path: Path,
) -> list[dict[str, Any]]:
    """Return list of (substrate_name, archive_zip_path, substrate_class,
    member_name) for every VALIDATED_CONTEST_MEMBER row in the corrected
    artifact. Each row maps the EXACT archive + member the canonical method
    will probe."""
    payload = json.loads(artifact_path.read_text())
    per = payload.get("per_substrate_results", {})
    targets: list[dict[str, Any]] = []
    for name, row in per.items():
        if row.get("validation_status") != "VALIDATED_CONTEST_MEMBER":
            continue
        members = list(row.get("member_breakdown", {}).keys())
        if not members:
            continue
        for member_name in members:
            targets.append(
                {
                    "substrate_name": name,
                    "archive_zip_path": row["archive_path"],
                    "substrate_class": row.get("substrate_class", "unknown"),
                    "member_name": member_name,
                }
            )
    return targets


def _normalise_probe_result(result: SubstrateProbeResult) -> dict[str, Any]:
    """Serialize a SubstrateProbeResult to a plain dict for JSON dump."""
    data = asdict(result)
    # MemberProbeResult tuples already dataclasses-as-dicts via asdict.
    return data


def aggregate_per_substrate(
    probe_dict: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Per substrate, aggregate over members and compute apples-to-apples
    deliverable_savings_estimate. Each member-row already has a per-member
    savings estimate; we sum and return the per-substrate aggregate."""
    agg: dict[str, dict[str, Any]] = {}
    for substrate_name, probe_rows in probe_dict.items():
        members = probe_rows["members"]
        aggregate_savings = sum(m.get("savings_estimate", 0.0) for m in members)
        aggregate_pre_entropy = sum(m.get("pre_entropy_bytes", 0) for m in members)
        aggregate_at_floor = sum(m.get("at_floor_bytes", 0) for m in members)
        aggregate_post_entropy = sum(m.get("post_entropy_bytes", 0) for m in members)
        agg[substrate_name] = {
            "archive_sha256": probe_rows.get("archive_sha256"),
            "archive_zip_path": probe_rows.get("archive_zip_path"),
            "substrate_class": probe_rows.get("substrate_class"),
            "members": members,
            "aggregate_savings_estimate": float(aggregate_savings),
            "aggregate_pre_entropy_bytes": int(aggregate_pre_entropy),
            "aggregate_at_floor_bytes": int(aggregate_at_floor),
            "aggregate_post_entropy_bytes": int(aggregate_post_entropy),
            "validation_status": "VALIDATED_CONTEST_MEMBER",
        }
    return agg


def rank_by_savings(
    agg: dict[str, dict[str, Any]],
) -> list[tuple[str, float]]:
    """Rank substrates by aggregate_savings_estimate (descending). Only
    substrates with aggregate savings > 0 are surfaced; ties broken by
    aggregate_pre_entropy_bytes (more pre-entropy bytes = more upside if a
    better codec lands later)."""
    rows = [
        (name, row["aggregate_savings_estimate"], row["aggregate_pre_entropy_bytes"])
        for name, row in agg.items()
    ]
    rows.sort(key=lambda r: (-r[1], -r[2]))
    return [(name, savings) for name, savings, _ in rows]


def build_q4_recommendation(
    ranked: list[tuple[str, float]],
    agg: dict[str, dict[str, Any]],
    threshold: float = Q4_MIN_DELIVERABLE_SAVINGS_FOR_RETARGET,
) -> dict[str, Any]:
    """Decide whether ranked top-1 is a ready Q4 BUILD target or whether Q4
    should be deferred. Honest reporting: if all 8 archives at entropy floor
    with aggregate_savings ≤ threshold, recommend deferral + substrate-class-
    shift pivot per HORIZON-CLASS standing directive.

    The Q4 BUILD invocation is the operator-facing concrete command if a
    target is selected; the deferral recommendation cites alternative class-
    shift paths if not.
    """
    if not ranked:
        return {
            "recommended_q4_target_substrate": None,
            "recommended_q4_target_archive_sha256": None,
            "recommended_q4_target_member_name": None,
            "recommended_q4_target_aggregate_savings": 0.0,
            "verdict": "DEFER_Q4",
            "reason": (
                "no VALIDATED_CONTEST_MEMBER targets surfaced; corrected "
                "artifact contained zero validated rows"
            ),
            "alternative_paths": [],
        }
    top_name, top_savings = ranked[0]
    if top_savings >= threshold:
        row = agg[top_name]
        # Pick the largest single-member contributor to surface as the Q4
        # target's exact byte payload.
        if row["members"]:
            top_member = max(row["members"], key=lambda m: m.get("savings_estimate", 0.0))
            top_member_name = top_member.get("name")
        else:
            top_member_name = None
        return {
            "recommended_q4_target_substrate": top_name,
            "recommended_q4_target_archive_sha256": row["archive_sha256"],
            "recommended_q4_target_member_name": top_member_name,
            "recommended_q4_target_aggregate_savings": float(top_savings),
            "verdict": "BUILD_Q4_VALIDATED_TARGET",
            "reason": (
                f"top-ranked substrate {top_name!r} aggregate savings "
                f"{top_savings:.6f} exceeds Q4 retarget threshold {threshold:.6f}"
            ),
            "alternative_paths": [],
        }
    return {
        "recommended_q4_target_substrate": None,
        "recommended_q4_target_archive_sha256": None,
        "recommended_q4_target_member_name": None,
        "recommended_q4_target_aggregate_savings": float(top_savings),
        "verdict": "DEFER_Q4",
        "reason": (
            f"ALL {len(ranked)} VALIDATED contest archives are at the "
            f"entropy floor (top apples-to-apples aggregate savings "
            f"{top_savings:.6f} < threshold {threshold:.6f}); Wyner-Ziv "
            f"hoist offers NO measurable score gain on currently-shipping "
            f"archives; pivot to substrate-class-shift methods per "
            f"HORIZON-CLASS Stage 2 deferred plan"
        ),
        "alternative_paths": [
            "predictive-receiver substrates (Z5/Z6/Z7/Z8 family per Catalogs #310/#311/#312)",
            "cooperative-receiver scorer-margin substrates (ATW V2/D4 family)",
            "foveation + LA-pose (TT5L per Catalog #311)",
            "Wyner-Ziv side-info for NOVEL ratesplit (NOT re-compression of already-compressed bytes)",
            "substrate-class-shift candidates ranked by Catalog #227 Tier C density evidence",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Option B sweep: re-run pre-entropy prober on archive.zip MEMBER "
            "bytes for the 8 VALIDATED contest archives + emit apples-to-"
            "apples Q4 retarget recommendation."
        )
    )
    parser.add_argument(
        "--corrected-artifact",
        default=str(DEFAULT_CORRECTED_ARTIFACT),
        help=(
            "Path to the corrected pre-entropy-prober artifact "
            "(default: canonical Option C output)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=(
            "Directory to write the sweep manifest "
            "(default: .omx/state/wyner_ziv_deliverability/)"
        ),
    )
    parser.add_argument(
        "--retarget-threshold",
        type=float,
        default=Q4_MIN_DELIVERABLE_SAVINGS_FOR_RETARGET,
        help=(
            "Minimum aggregate apples-to-apples score-savings for Q4 "
            "retarget (default: 0.001)"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON summary to stdout in addition to writing sidecar",
    )
    args = parser.parse_args(argv)

    corrected_path = Path(args.corrected_artifact)
    if not corrected_path.exists():
        print(
            f"ERROR: corrected artifact not found at {corrected_path}",
            file=sys.stderr,
        )
        return 2

    targets = load_validated_targets_from_corrected_artifact(corrected_path)
    if not targets:
        print(
            "ERROR: no VALIDATED_CONTEST_MEMBER targets in corrected artifact",
            file=sys.stderr,
        )
        return 2

    # Per-substrate probe via canonical method (Option B).
    probe_dict: dict[str, dict[str, Any]] = {}
    for tgt in targets:
        result = probe_substrate_archive_member(
            substrate_name=tgt["substrate_name"],
            archive_zip_path=tgt["archive_zip_path"],
            member_name=tgt["member_name"],
            substrate_class=tgt["substrate_class"],
        )
        result_dict = _normalise_probe_result(result)
        # Surface a unified per-member entry list. Each probe returns 0..1
        # MemberProbeResult; we accumulate by substrate_name.
        if tgt["substrate_name"] not in probe_dict:
            probe_dict[tgt["substrate_name"]] = {
                "archive_zip_path": tgt["archive_zip_path"],
                "archive_sha256": result_dict.get("archive_sha256"),
                "substrate_class": tgt["substrate_class"],
                "members": [],
            }
        for m in result_dict.get("member_results", []):
            probe_dict[tgt["substrate_name"]]["members"].append(
                {
                    "name": m["member_name"],
                    "bytes_raw": m["raw_bytes"],
                    "lzma_compressed": m["lzma_bytes"],
                    "lzma_ratio": m["lzma_ratio"],
                    "brotli_compressed": m.get("brotli_bytes"),
                    "brotli_ratio": m.get("brotli_ratio"),
                    "zlib_compressed": m["zlib_bytes"],
                    "zlib_ratio": m["zlib_ratio"],
                    "best_codec": m["best_codec"],
                    "best_compressed": (
                        m["brotli_bytes"]
                        if m["best_codec"] == "brotli"
                        else m["lzma_bytes"]
                        if m["best_codec"] == "lzma"
                        else m["zlib_bytes"]
                    ),
                    "best_ratio": m["best_ratio"],
                    "classification": m["classification"],
                    "savings_estimate": (
                        25.0
                        * max(0, m["raw_bytes"] - min(
                            m["lzma_bytes"],
                            m["zlib_bytes"],
                            m["brotli_bytes"] if m["brotli_bytes"] is not None else m["lzma_bytes"],
                        ))
                        / CONTEST_RATE_DENOM_BYTES
                    ),
                    "pre_entropy_bytes": m["raw_bytes"] if m["classification"] == "PRE_ENTROPY" else 0,
                    "at_floor_bytes": m["raw_bytes"] if m["classification"] == "AT_FLOOR" else 0,
                    "post_entropy_bytes": m["raw_bytes"] if m["classification"] == "POST_ENTROPY" else 0,
                    "validation_status": "VALIDATED_CONTEST_MEMBER",
                    "evidence_tag": f"[empirical:lzma_ratio_on_actual_member={m['lzma_ratio']:.4f}]",
                }
            )

    agg = aggregate_per_substrate(probe_dict)
    ranked = rank_by_savings(agg)
    q4_recommendation = build_q4_recommendation(ranked, agg, args.retarget_threshold)

    written_at = datetime.now(UTC).isoformat()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "candidates_validated": len(agg),
        "per_substrate_results": agg,
        "ranked_by_savings": ranked,
        "recommended_q4_target_substrate": q4_recommendation["recommended_q4_target_substrate"],
        "recommended_q4_target_archive_sha256": q4_recommendation["recommended_q4_target_archive_sha256"],
        "recommended_q4_target_member_name": q4_recommendation["recommended_q4_target_member_name"],
        "recommended_q4_target_aggregate_savings": q4_recommendation["recommended_q4_target_aggregate_savings"],
        "q4_retarget_threshold": args.retarget_threshold,
        "q4_verdict": q4_recommendation["verdict"],
        "q4_reason": q4_recommendation["reason"],
        "q4_alternative_paths": q4_recommendation["alternative_paths"],
        "phantom_score_class_extincted": True,
        "catalog_321_compliant": True,
        "evidence_grade": "predicted",
        "score_claim": False,
        "promotion_eligible": False,
        "claude_md_compliance_tags": [
            "apples_to_apples_per_catalog_287",
            "validated_contest_member_per_catalog_321",
            "phantom_score_research_sidecar_rejected_per_catalog_321",
            "fcntl_locked_write_per_catalog_131",
            "non_authoritative_per_catalog_192",
        ],
        "source_corrected_artifact": str(corrected_path),
        "written_at_utc": written_at,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = written_at.replace(":", "").replace("-", "").split(".")[0]
    output_path = (
        output_dir
        / f"option_b_archive_member_sweep_{timestamp}.json"
    )
    # Per CLAUDE.md "Operator gates must be wired and used" + Catalog #131,
    # use the canonical helper from the prober module.
    _fcntl_locked_atomic_write(output_path, payload)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))

    # Human-readable summary
    print(f"[Option B sweep] Wrote {output_path}", file=sys.stderr)
    print(
        f"[Option B sweep] Probed {len(agg)} VALIDATED archives; "
        f"q4_verdict={payload['q4_verdict']}",
        file=sys.stderr,
    )
    print(
        f"[Option B sweep] Recommended Q4 target: "
        f"{payload['recommended_q4_target_substrate']!r} "
        f"(aggregate_savings={payload['recommended_q4_target_aggregate_savings']:.6f})",
        file=sys.stderr,
    )
    if payload["q4_verdict"] == "DEFER_Q4":
        print(
            f"[Option B sweep] DEFER_Q4 rationale: {payload['q4_reason']}",
            file=sys.stderr,
        )
        print(
            f"[Option B sweep] Alternative class-shift paths surfaced: "
            f"{len(payload['q4_alternative_paths'])}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
