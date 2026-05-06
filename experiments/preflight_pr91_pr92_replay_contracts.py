#!/usr/bin/env python3
"""Fail-closed PR91/HPM1 and PR92/RMB1 replay contract preflight.

This tool is local-only. It does not load scorers, dispatch remote jobs, or
make score claims. It records whether PR91's HPM1 entropy stream is still
blocked by the probability/range contract and whether a PR92/RMB1 exact replay
artifact is internally consistent enough to cite as existing evidence.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.pr91_hpm1_codec import DEFAULT_PR91_ARCHIVE, run_pr91_hpm1_probability_variant_matrix  # noqa: E402

TOOL = "experiments/preflight_pr91_pr92_replay_contracts.py"
SCHEMA = "pr91_pr92_replay_contract_preflight_v1"
SCORE_DENOMINATOR = 37_545_489

DEFAULT_PR91_PROBABILITY_REPORT = (
    REPO_ROOT
    / "experiments/results/public_pr91_intake_20260504_codex/diagnostics/"
    "pr91_hpm1_probability_variant_matrix_frame0_20260504_current_codex.json"
)
DEFAULT_PR92_MANIFEST = (
    REPO_ROOT
    / "experiments/results/pr85_stbm1br_pr92_rmb1_randmulti_20260504_worker/"
    "pr85_stbm1br_plus_pr92_rmb1_randmulti_recode/manifest.json"
)
DEFAULT_PR92_EXACT_JSON = (
    REPO_ROOT
    / "experiments/results/lightning_batch/exact_eval_pr85_stbm1br_pr92_rmb1_t4_20260504T082220Z/"
    "contest_auth_eval.adjudicated.json"
)
DEFAULT_OUTPUT_JSON = (
    REPO_ROOT / "experiments/results/pr91_pr92_replay_contract_preflight_20260504_codex/preflight.json"
)
DEFAULT_LEDGER = REPO_ROOT / ".omx/research/pr91_pr92_replay_contract_preflight_20260504_codex.md"


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(_rel(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {_rel(path)}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_text(payload), encoding="utf-8")


def _failed_checks(checks: Mapping[str, bool]) -> list[str]:
    return [name for name, passed in checks.items() if not passed]


def _read_or_run_pr91_probability_report(
    *, archive: Path, report_path: Path, rerun: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    if rerun:
        report = run_pr91_hpm1_probability_variant_matrix(
            archive,
            variants=None,
            write_json=False,
        )
        source = {
            "mode": "rerun_local_prefix",
            "archive": _rel(archive),
            "report_path": None,
            "score_claim": False,
            "dispatch_performed": False,
        }
        return report, source
    report = _load_json(report_path)
    source = {
        "mode": "read_existing_report",
        "report_path": _rel(report_path),
        "report_sha256": _sha256_file(report_path),
        "archive": _rel(archive),
        "score_claim": False,
        "dispatch_performed": False,
    }
    return report, source


def validate_pr91_hpm1_contract(
    *, archive: Path, probability_report: Path, rerun: bool
) -> dict[str, Any]:
    report, source = _read_or_run_pr91_probability_report(
        archive=archive,
        report_path=probability_report,
        rerun=rerun,
    )
    variants = report.get("variant_results", [])
    if not isinstance(variants, list):
        variants = []
    passed_variants = [row for row in variants if isinstance(row, dict) and row.get("status") == "passed"]
    failed_variants = [
        str(row.get("variant") or row.get("name") or "unknown")
        for row in variants
        if isinstance(row, dict) and row.get("status") != "passed"
    ]
    static = report.get("hpm1_static_contract", {})
    checks = {
        "archive_exists": archive.is_file(),
        "probability_report_has_variants": bool(variants),
        "static_contract_passed": not isinstance(static, dict) or static.get("status") in (None, "passed"),
        "no_variant_passed": not passed_variants,
        "not_a_score_claim": report.get("score_claim") is False,
        "not_dispatch_allowed": report.get("dispatch_allowed") is False,
    }
    status = (
        "blocked_hpm1_probability_range_contract_mismatch"
        if not _failed_checks(checks)
        else "failed_closed"
    )
    return {
        "status": status,
        "score_claim": False,
        "dispatch_allowed": False,
        "source": source,
        "checks": checks,
        "failed_checks": _failed_checks(checks),
        "classification": {
            "bug_class": "hpm1_probability_range_contract_mismatch",
            "failure_reason": "no tested PR91 HPM1 probability/range variant is byte-closed",
            "failed_variants": failed_variants,
            "passed_variants": [str(row.get("variant") or row.get("name")) for row in passed_variants],
        },
        "safe_next_action": (
            "Recover byte-exact HPM1 range/probability replay before any derived PR91 exact-eval dispatch."
        ),
    }


def _score_from_components(payload: Mapping[str, Any]) -> float:
    return (
        100 * float(payload["avg_segnet_dist"])
        + math.sqrt(10 * float(payload["avg_posenet_dist"]))
        + 25 * int(payload["archive_size_bytes"]) / SCORE_DENOMINATOR
    )


def _artifact_recomputed_score(payload: Mapping[str, Any]) -> float:
    if "score_recomputed_from_components" in payload:
        return float(payload["score_recomputed_from_components"])
    return _score_from_components(payload)


def validate_pr92_rmb1_contract(*, manifest_path: Path, exact_json_path: Path) -> dict[str, Any]:
    try:
        manifest = _load_json(manifest_path)
        exact = _load_json(exact_json_path)
    except FileNotFoundError as exc:
        return {
            "status": "failed_closed_missing_pr92_artifact",
            "score_claim": False,
            "dispatch_allowed": False,
            "missing": str(exc),
            "manifest_path": _rel(manifest_path),
            "exact_json_path": _rel(exact_json_path),
            "next_safe_build_command": "Rebuild PR92/RMB1 manifest before claiming replay evidence.",
            "next_safe_exact_eval_command_if_rebuilt": "Run exact CUDA auth eval on the rebuilt archive.",
        }

    candidate = manifest.get("candidate_archive", {})
    runtime = manifest.get("exact_eval_runtime_contract", {})
    readiness = manifest.get("dispatch_readiness", {})
    parity = manifest.get("randmulti_decoded_row_parity", {})
    non_noop = manifest.get("non_noop_byte_change", {})
    provenance = exact.get("provenance", {})
    runtime_manifest = provenance.get("inflate_runtime_manifest", {})
    recomputed = _artifact_recomputed_score(exact)
    rounded_component_recompute = _score_from_components(exact)
    required_inflate_sh = runtime.get("required_inflate_sh")
    runtime_root = str(runtime_manifest.get("runtime_root", ""))
    archive_sha = str(candidate.get("sha256") or provenance.get("archive_sha256") or exact.get("archive_sha256", ""))
    exact_archive_sha = str(provenance.get("archive_sha256") or exact.get("archive_sha256", ""))
    checks = {
        "candidate_archive_sha_matches_exact": bool(archive_sha) and archive_sha == exact_archive_sha,
        "candidate_archive_bytes_match_exact": int(candidate.get("bytes", exact.get("archive_size_bytes", -1)))
        == int(exact.get("archive_size_bytes", -2)),
        "runtime_required_inflate_sh_recorded": bool(required_inflate_sh),
        "runtime_root_matches_required": not required_inflate_sh or str(required_inflate_sh) in runtime_root,
        "dispatch_readiness_passed": readiness.get("ready") is True or readiness.get("ready_for_exact_eval_dispatch") is True,
        "randmulti_decoded_rows_match": parity.get("decoded_rows_match") is True
        or parity.get("parity_status") == "passed",
        "non_noop_byte_change": non_noop.get("changed") is True or non_noop.get("non_noop") is True,
        "score_recompute_matches": abs(float(exact.get("score_recomputed_from_components", recomputed)) - recomputed) < 1e-9,
    }
    status = "passed_t4_exact_pr92_rmb1_stack_validated" if not _failed_checks(checks) else "failed_closed"
    return {
        "status": status,
        "score_claim": status.startswith("passed_"),
        "dispatch_allowed": False,
        "evidence_grade": "A++" if status.startswith("passed_") else "invalid",
        "checks": checks,
        "failed_checks": _failed_checks(checks),
        "manifest_path": _rel(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "exact_json_path": _rel(exact_json_path),
        "exact_json_sha256": _sha256_file(exact_json_path),
        "exact_eval": {
            "score": recomputed,
            "rounded_component_recompute": rounded_component_recompute,
            "archive_bytes": int(exact["archive_size_bytes"]),
            "archive_sha256": exact_archive_sha,
            "avg_segnet_dist": float(exact["avg_segnet_dist"]),
            "avg_posenet_dist": float(exact["avg_posenet_dist"]),
            "runtime_tree_sha256": runtime_manifest.get("runtime_tree_sha256"),
        },
        "next_safe_build_command": manifest.get(
            "next_safe_build_command",
            "Rebuild PR92/RMB1 candidate with decoded-row parity manifest.",
        ),
        "next_safe_exact_eval_command_if_rebuilt": manifest.get(
            "next_safe_exact_eval_command_if_rebuilt",
            "Run experiments/contest_auth_eval.py --device cuda on rebuilt archive.",
        ),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    pr91 = validate_pr91_hpm1_contract(
        archive=args.pr91_archive,
        probability_report=args.pr91_probability_report,
        rerun=args.rerun_pr91_prefix,
    )
    pr92 = validate_pr92_rmb1_contract(
        manifest_path=args.pr92_manifest,
        exact_json_path=args.pr92_exact_json,
    )
    overall_status = (
        "passed_pr92_a_plus_plus_pr91_fail_closed"
        if pr91["status"] == "blocked_hpm1_probability_range_contract_mismatch"
        and pr92["status"] == "passed_t4_exact_pr92_rmb1_stack_validated"
        else "failed_closed"
    )
    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "recorded_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "status": overall_status,
        "score_claim": overall_status.startswith("passed_") and bool(pr92.get("score_claim")),
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "pr91_hpm1": pr91,
        "pr92_rmb1_stack": pr92,
        "next_actions": [
            pr91["safe_next_action"],
            pr92["next_safe_build_command"],
            pr92["next_safe_exact_eval_command_if_rebuilt"],
        ],
    }


def render_ledger(report: Mapping[str, Any]) -> str:
    pr91 = report["pr91_hpm1"]
    pr92 = report["pr92_rmb1_stack"]
    lines = [
        "# PR91/PR92 Replay Contract Preflight - 2026-05-04",
        "",
        f"- tool: `{TOOL}`",
        f"- status: `{report['status']}`",
        "- dispatch_performed: `false`",
        "- remote_jobs_dispatched: `false`",
        "",
        "## PR91 HPM1",
        "",
        f"- status: `{pr91['status']}`",
        f"- dispatch_allowed: `{pr91['dispatch_allowed']}`",
        f"- bug_class: `{pr91['classification']['bug_class']}`",
        f"- failure_reason: `{pr91['classification']['failure_reason']}`",
        f"- failed_variants: `{', '.join(pr91['classification']['failed_variants'])}`",
        "",
        "PR91 remains fail-closed until a byte-exact HPM1 probability/range contract decodes the stream.",
        "",
        "## PR92 RMB1 Stack",
        "",
        f"- status: `{pr92['status']}`",
        f"- evidence_grade: `{pr92.get('evidence_grade', 'invalid')}`",
    ]
    if "exact_eval" in pr92:
        exact = pr92["exact_eval"]
        lines.extend(
            [
                f"- score: `{exact['score']}`",
                f"- archive bytes: `{exact['archive_bytes']}`",
                f"- archive sha256: `{exact['archive_sha256']}`",
                f"- avg_segnet_dist: `{exact['avg_segnet_dist']}`",
                f"- avg_posenet_dist: `{exact['avg_posenet_dist']}`",
                f"- runtime_tree_sha256: `{exact['runtime_tree_sha256']}`",
            ]
        )
    else:
        lines.append(f"- missing: `{pr92.get('missing', 'unknown')}`")
    lines.extend(
        [
            "",
            "## Next Safe Commands",
            "",
            "```bash",
            str(pr92["next_safe_build_command"]),
            "```",
            "",
            str(pr92["next_safe_exact_eval_command_if_rebuilt"]),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr91-archive", type=Path, default=DEFAULT_PR91_ARCHIVE)
    parser.add_argument("--pr91-probability-report", type=Path, default=DEFAULT_PR91_PROBABILITY_REPORT)
    parser.add_argument("--rerun-pr91-prefix", action="store_true")
    parser.add_argument("--pr92-manifest", type=Path, default=DEFAULT_PR92_MANIFEST)
    parser.add_argument("--pr92-exact-json", type=Path, default=DEFAULT_PR92_EXACT_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--ledger-md", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--stdout", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    _write_json(args.output_json, report)
    args.ledger_md.parent.mkdir(parents=True, exist_ok=True)
    args.ledger_md.write_text(render_ledger(report), encoding="utf-8")
    if args.stdout:
        sys.stdout.write(_json_text(report))
        return 0
    print(_json_text({"status": report["status"], "output_json": _rel(args.output_json)}), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
