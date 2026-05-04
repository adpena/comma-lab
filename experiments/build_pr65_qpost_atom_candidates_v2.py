#!/usr/bin/env python3
"""Expand PR65 qpost atom-subset candidates for the C-089/QP1 frontier.

This is a local planner/builder only.  It delegates byte-closed archive
construction to ``build_pr65_qpost_atom_candidates.py`` and adds a deterministic
v2 screen over more PR65 qpost atom subsets.  The output archives contain all
score-affecting qpost bytes in ``archive.zip`` but carry no score claim until
exact CUDA auth eval is run on the exact bytes after a lane claim.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CORE_BUILDER_PATH = REPO_ROOT / "experiments/build_pr65_qpost_atom_candidates.py"
PRODUCER = "experiments/build_pr65_qpost_atom_candidates_v2.py"
SCHEMA_VERSION = 2
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/results/pr65_qpost_atom_v2_worker_20260503"
DEFAULT_TARGET_SCORE = 0.314
DEFAULT_MIN_PRIMARY_SLACK = 0.00020
DEFAULT_PRIMARY_PAIR_CAP = 64
ORIGINAL_VIDEO_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25.0 / ORIGINAL_VIDEO_BYTES

_CANDIDATE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_]{2,120}$")
_DEFAULT_PATH = object()


class QPostAtomV2Error(ValueError):
    """Raised when v2 planning fails a deterministic guard."""


@dataclass(frozen=True)
class ExpansionFamily:
    family_id: str
    streams: tuple[str, ...]
    top_pairs: tuple[int, ...]
    risk_family: str
    dispatch_class: str


DEFAULT_FAMILIES: tuple[ExpansionFamily, ...] = (
    ExpansionFamily(
        "bias_poseadv",
        ("bias",),
        (24, 28, 32, 40, 48, 56, 64, 80, 96, 128),
        "low_byte_bias_only_trace_positive",
        "low_bias_only",
    ),
    ExpansionFamily(
        "bias_region_poseadv",
        ("bias", "region"),
        (16, 24, 32, 40, 48, 64, 96),
        "medium_region_bias_trace_positive",
        "diagnostic_region",
    ),
    ExpansionFamily(
        "post_bias_poseadv",
        ("post", "bias"),
        (8, 16, 24, 32),
        "high_post_bias_trace_positive",
        "diagnostic_post",
    ),
    ExpansionFamily(
        "post_region_bias_poseadv",
        ("post", "region", "bias"),
        (8, 16, 24),
        "high_post_region_bias_trace_positive",
        "diagnostic_post_region",
    ),
    ExpansionFamily(
        "shift_frac_poseadv",
        ("shift", "frac", "frac2", "frac3"),
        (8, 16, 24, 32),
        "very_high_motion_fraction_trace_positive",
        "diagnostic_motion",
    ),
    ExpansionFamily(
        "bias_shift_poseadv",
        ("bias", "shift"),
        (16, 24, 32),
        "high_bias_shift_trace_positive",
        "diagnostic_motion",
    ),
)


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _load_core_builder():
    spec = importlib.util.spec_from_file_location("pact_pr65_qpost_atoms_core", CORE_BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise QPostAtomV2Error(f"cannot load core builder: {CORE_BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _candidate_id(family_id: str, top_pairs: int) -> str:
    return f"v2_pr65_qpost_{family_id}_top{top_pairs:03d}"


def _validate_family(family: ExpansionFamily) -> None:
    if not _CANDIDATE_ID_RE.match(family.family_id):
        raise QPostAtomV2Error(f"invalid family id: {family.family_id!r}")
    if not family.streams:
        raise QPostAtomV2Error(f"family {family.family_id!r} has no qpost streams")
    if any(count <= 0 or count > 600 for count in family.top_pairs):
        raise QPostAtomV2Error(f"family {family.family_id!r} has invalid top-pair counts")
    if tuple(sorted(set(family.top_pairs))) != family.top_pairs:
        raise QPostAtomV2Error(
            f"family {family.family_id!r} top-pair counts must be unique and ascending"
        )


def expand_specs(core: Any, families: Sequence[ExpansionFamily]) -> tuple[Any, ...]:
    specs: list[Any] = []
    seen: set[str] = set()
    for family in families:
        _validate_family(family)
        for top_pairs in family.top_pairs:
            candidate_id = _candidate_id(family.family_id, int(top_pairs))
            if candidate_id in seen:
                raise QPostAtomV2Error(f"duplicate candidate id: {candidate_id}")
            seen.add(candidate_id)
            specs.append(
                core.CandidateSpec(
                    candidate_id,
                    tuple(family.streams),
                    int(top_pairs),
                    family.risk_family,
                )
            )
    return tuple(specs)


def _family_by_candidate_id(families: Sequence[ExpansionFamily]) -> dict[str, ExpansionFamily]:
    out: dict[str, ExpansionFamily] = {}
    for family in families:
        for top_pairs in family.top_pairs:
            out[_candidate_id(family.family_id, int(top_pairs))] = family
    return out


def _exact_eval_command(archive: Path, output_dir: Path, candidate_id: str) -> str:
    work_dir = output_dir / "exact_eval_work" / candidate_id
    return (
        ".venv/bin/python -u experiments/contest_auth_eval.py "
        f"--archive {archive} "
        "--inflate-sh submissions/robust_current/inflate.sh "
        "--upstream-dir upstream --device cuda --keep-work-dir "
        f"--work-dir {work_dir}"
    )


def _candidate_pair_count(row: Mapping[str, Any]) -> int:
    pairs = row.get("selected_pairs") or []
    if isinstance(pairs, list):
        return len(pairs)
    return 0


def _candidate_screen(
    row: Mapping[str, Any],
    *,
    family: ExpansionFamily | None,
    baseline_score: float,
    target_score: float,
    primary_pair_cap: int,
) -> dict[str, Any]:
    if not row.get("built"):
        return {
            "candidate_id": row.get("candidate_id"),
            "built": False,
            "recommendation_class": "not_built",
            "reason": row.get("skip_reason", "not built by core builder"),
        }

    rate_delta = float(row.get("formula_rate_score_delta", 0.0))
    opportunity = float(row.get("public_trace_opportunity_bound", 0.0))
    break_even = (baseline_score - target_score) + rate_delta
    slack = opportunity - break_even
    pair_count = _candidate_pair_count(row)
    streams = tuple(str(v) for v in row.get("include_streams", []))
    blockers = [str(v) for v in row.get("dispatch_blockers", [])]
    low_risk_bias_only = streams == ("bias",)
    class_name = family.dispatch_class if family is not None else "unknown"

    if blockers:
        recommendation = "do_not_dispatch"
        reason = "core risk classifier has blockers"
    elif not low_risk_bias_only:
        recommendation = "diagnostic_only_after_operator_review"
        reason = "non-bias qpost streams inherit prior exact-negative risk"
    elif slack <= 0.0:
        recommendation = "do_not_dispatch"
        reason = "public-trace opportunity does not clear sub-0.314 break-even"
    elif pair_count > primary_pair_cap:
        recommendation = "aggressive_bias_only_after_primary_screen"
        reason = "bias-only clears break-even but exceeds primary pair-count cap"
    else:
        recommendation = "primary_bias_only_exact_screen_candidate"
        reason = "bias-only clears break-even inside primary pair-count cap"

    return {
        "candidate_id": row["candidate_id"],
        "built": True,
        "dispatch_class": class_name,
        "recommendation_class": recommendation,
        "recommendation_reason": reason,
        "score_claim": False,
        "evidence_grade": "byte_trace_planning_only_until_exact_cuda",
        "target_score": target_score,
        "baseline_score": baseline_score,
        "archive_bytes": int(row["archive_bytes"]),
        "archive_sha256": str(row["archive_sha256"]),
        "archive": str(row["archive"]),
        "archive_byte_delta": int(row.get("archive_byte_delta", 0)),
        "qpost_bytes": int(row.get("qpost_bytes", 0)),
        "selected_pair_count": pair_count,
        "selected_pairs": list(row.get("selected_pairs", [])),
        "selected_active_atoms_total": int(row.get("selected_active_atoms_total", 0)),
        "include_streams": list(streams),
        "public_trace_opportunity_bound": opportunity,
        "rate_score_delta": rate_delta,
        "break_even_component_gain_for_target": break_even,
        "trace_slack_vs_target": slack,
        "core_dispatch_recommendation": row.get("dispatch_recommendation"),
        "core_dispatch_blockers": blockers,
    }


def _sort_primary_key(row: Mapping[str, Any], *, min_primary_slack: float) -> tuple[int, int, float, int]:
    slack = float(row.get("trace_slack_vs_target", 0.0))
    pair_count = int(row.get("selected_pair_count", 0))
    enough_slack = slack >= min_primary_slack
    return (0 if enough_slack else 1, pair_count, -slack, int(row.get("archive_bytes", 0)))


def choose_recommendation(
    screens: Sequence[Mapping[str, Any]],
    *,
    min_primary_slack: float = DEFAULT_MIN_PRIMARY_SLACK,
) -> dict[str, Any]:
    primary = [
        row
        for row in screens
        if row.get("recommendation_class") == "primary_bias_only_exact_screen_candidate"
    ]
    aggressive = [
        row
        for row in screens
        if row.get("recommendation_class") == "aggressive_bias_only_after_primary_screen"
    ]
    primary_sorted = sorted(primary, key=lambda row: _sort_primary_key(row, min_primary_slack=min_primary_slack))
    aggressive_sorted = sorted(
        aggressive,
        key=lambda row: (-float(row.get("trace_slack_vs_target", 0.0)), int(row.get("selected_pair_count", 0))),
    )
    best = primary_sorted[0] if primary_sorted else (aggressive_sorted[0] if aggressive_sorted else None)
    return {
        "remote_dispatched": False,
        "score_claim": False,
        "target_score": DEFAULT_TARGET_SCORE,
        "recommendation": "no_remote_dispatch_from_worker",
        "primary_candidate_id": best.get("candidate_id") if best else None,
        "primary_archive": best.get("archive") if best else None,
        "primary_archive_bytes": best.get("archive_bytes") if best else None,
        "primary_archive_sha256": best.get("archive_sha256") if best else None,
        "primary_trace_slack_vs_target": best.get("trace_slack_vs_target") if best else None,
        "primary_reason": (
            "Run at most one exact CUDA screen after claiming a dispatch lane; "
            "this worker produced only planning artifacts and did not dispatch."
            if best
            else "No v2 candidate cleared the planning-only bias-stream screen."
        ),
        "aggressive_followup_candidate_id": aggressive_sorted[0].get("candidate_id") if aggressive_sorted else None,
    }


def _write_overlay_manifests(
    *,
    screens: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    for row in screens:
        if not row.get("built"):
            continue
        candidate_dir = Path(str(row["archive"])).resolve().parent
        overlay = {
            "schema_version": SCHEMA_VERSION,
            "tool": PRODUCER,
            "candidate_id": row["candidate_id"],
            "score_claim": False,
            "evidence_grade": "byte_trace_planning_only_until_exact_cuda",
            "archive": row["archive"],
            "archive_bytes": row["archive_bytes"],
            "archive_sha256": row["archive_sha256"],
            "v2_screen": row,
            "exact_eval_command_template": _exact_eval_command(
                Path(str(row["archive"])), output_dir, str(row["candidate_id"])
            ),
            "remote_dispatch": {
                "dispatched": False,
                "requires_lane_claim": True,
                "note": "No remote GPU, training, or eval job was dispatched by this worker.",
            },
        }
        _write_json(candidate_dir / "v2_manifest.json", overlay)


def build_expansion(
    *,
    source_archive: Path | None = None,
    pr65_archive: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    anatomy_json: Path | None | object = _DEFAULT_PATH,
    c089_trace: Path | None | object = _DEFAULT_PATH,
    pr65_trace: Path | None | object = _DEFAULT_PATH,
    families: Sequence[ExpansionFamily] = DEFAULT_FAMILIES,
    target_score: float = DEFAULT_TARGET_SCORE,
    min_primary_slack: float = DEFAULT_MIN_PRIMARY_SLACK,
    primary_pair_cap: int = DEFAULT_PRIMARY_PAIR_CAP,
    positive_trace_only: bool = True,
    allow_source_sha_mismatch: bool = False,
) -> dict[str, Any]:
    core = _load_core_builder()
    output_dir = output_dir.resolve()
    candidate_root = output_dir / "candidates"
    specs = expand_specs(core, families)
    family_lookup = _family_by_candidate_id(families)

    matrix = core.build_matrix(
        source_archive=source_archive or core.DEFAULT_SOURCE_ARCHIVE,
        pr65_archive=pr65_archive or core.DEFAULT_PR65_ARCHIVE,
        output_dir=candidate_root,
        anatomy_json=core.DEFAULT_ANATOMY_JSON if anatomy_json is _DEFAULT_PATH else anatomy_json,
        c089_trace=core.DEFAULT_C089_TRACE if c089_trace is _DEFAULT_PATH else c089_trace,
        pr65_trace=core.DEFAULT_PR65_TRACE if pr65_trace is _DEFAULT_PATH else pr65_trace,
        specs=specs,
        positive_trace_only=positive_trace_only,
        expected_source_sha256=None if allow_source_sha_mismatch else core.EXPECTED_C089_SHA256,
        expected_pr65_sha256=None if allow_source_sha_mismatch else core.EXPECTED_PR65_SHA256,
        expected_pr65_head_sha=None if allow_source_sha_mismatch else core.EXPECTED_PR65_HEAD_SHA,
    )

    baseline_score = float((matrix.get("baseline") or {}).get("score", core.BASELINE_SCORE))
    screens = [
        _candidate_screen(
            row,
            family=family_lookup.get(str(row.get("candidate_id"))),
            baseline_score=baseline_score,
            target_score=target_score,
            primary_pair_cap=primary_pair_cap,
        )
        for row in matrix.get("candidate_summary", [])
        if isinstance(row, dict)
    ]
    recommendation = choose_recommendation(screens, min_primary_slack=min_primary_slack)
    recommendation["target_score"] = target_score
    recommendation["min_primary_slack"] = min_primary_slack
    recommendation["primary_pair_cap"] = primary_pair_cap

    _write_overlay_manifests(screens=screens, output_dir=output_dir)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "tool": PRODUCER,
        "score_claim": False,
        "evidence_grade": "byte_trace_planning_only_until_exact_cuda",
        "target_score": target_score,
        "source_core_tool": str(CORE_BUILDER_PATH.relative_to(REPO_ROOT)),
        "candidate_root": str(candidate_root),
        "core_candidate_summary": str(candidate_root / "candidate_summary.json"),
        "baseline": matrix.get("baseline"),
        "source_custody": matrix.get("source_custody"),
        "runtime_hook": matrix.get("runtime_hook"),
        "ranking_inputs": matrix.get("ranking_inputs"),
        "families": [
            {
                "family_id": family.family_id,
                "streams": list(family.streams),
                "top_pairs": list(family.top_pairs),
                "risk_family": family.risk_family,
                "dispatch_class": family.dispatch_class,
            }
            for family in families
        ],
        "candidate_screens": screens,
        "dispatch_summary": recommendation,
        "remote_dispatch": {
            "dispatched": False,
            "requires_lane_claim_before_any_eval": True,
        },
    }
    _write_json(output_dir / "candidate_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    core = _load_core_builder()
    parser.add_argument("--source-archive", type=Path, default=core.DEFAULT_SOURCE_ARCHIVE)
    parser.add_argument("--pr65-archive", type=Path, default=core.DEFAULT_PR65_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--anatomy-json", type=Path, default=core.DEFAULT_ANATOMY_JSON)
    parser.add_argument("--c089-trace", type=Path, default=core.DEFAULT_C089_TRACE)
    parser.add_argument("--pr65-trace", type=Path, default=core.DEFAULT_PR65_TRACE)
    parser.add_argument("--target-score", type=float, default=DEFAULT_TARGET_SCORE)
    parser.add_argument("--min-primary-slack", type=float, default=DEFAULT_MIN_PRIMARY_SLACK)
    parser.add_argument("--primary-pair-cap", type=int, default=DEFAULT_PRIMARY_PAIR_CAP)
    parser.add_argument(
        "--allow-negative-trace-pairs",
        action="store_true",
        help="Allow PR65-active pairs even when public traces do not show a positive pair opportunity.",
    )
    parser.add_argument(
        "--allow-source-sha-mismatch",
        action="store_true",
        help="Planning-only override for fixture/forensic sources; default fails closed on source SHA drift.",
    )
    args = parser.parse_args(argv)
    summary = build_expansion(
        source_archive=args.source_archive,
        pr65_archive=args.pr65_archive,
        output_dir=args.output_dir,
        anatomy_json=args.anatomy_json,
        c089_trace=args.c089_trace,
        pr65_trace=args.pr65_trace,
        target_score=args.target_score,
        min_primary_slack=args.min_primary_slack,
        primary_pair_cap=args.primary_pair_cap,
        positive_trace_only=not args.allow_negative_trace_pairs,
        allow_source_sha_mismatch=args.allow_source_sha_mismatch,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
