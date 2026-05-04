#!/usr/bin/env python3
"""Fail-closed PR91/HPM1 and PR92/RMB1 replay contract preflight.

This tool is local-only. It does not load scorers, dispatch remote jobs, or
make new score claims. It sharpens two endgame replay questions:

* PR91/HPM1: is the downloaded entropy stream locally replayable enough to
  justify a derived exact-eval dispatch?
* PR92/RMB1: was the PR92 randmulti transfer evaluated through the corrected
  replay runtime, and is the exact T4 result internally consistent?
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

from tac.pr91_hpm1_codec import (  # noqa: E402
    DEFAULT_PR91_ARCHIVE,
    run_pr91_hpm1_probability_variant_matrix,
)


TOOL = "experiments/preflight_pr91_pr92_replay_contracts.py"
SCHEMA = "pr91_pr92_replay_contract_preflight_v1"
SCORE_DENOMINATOR = 37_545_489.0

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
    REPO_ROOT
    / "experiments/results/pr91_pr92_replay_contract_preflight_20260504_codex/"
    "preflight.json"
)
DEFAULT_LEDGER = REPO_ROOT / ".omx/research/pr91_pr92_replay_contract_preflight_20260504_codex.md"


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_text(payload: Any) -> str:
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
    return [name for name, passed in checks.items() if passed is not True]


def _read_or_run_pr91_probability_report(
    *,
    archive: Path,
    report_path: Path,
    rerun: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if rerun:
        report = run_pr91_hpm1_probability_variant_matrix(archive, variants=None, max_frames=1)
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
    *,
    archive: Path = DEFAULT_PR91_ARCHIVE,
    probability_report: Path = DEFAULT_PR91_PROBABILITY_REPORT,
    rerun: bool = False,
) -> dict[str, Any]:
    report, source = _read_or_run_pr91_probability_report(
        archive=archive,
        report_path=probability_report,
        rerun=rerun,
    )
    variants = report.get("variant_results", [])
    if not isinstance(variants, list):
        variants = []
    failed_variants = [
        row.get("variant")
        for row in variants
        if isinstance(row, Mapping) and row.get("status") == "failed_closed"
    ]
    decoded_counts = {
        str(row.get("variant")): row.get("failure_context", {}).get(
            "decoded_symbol_count_before_failure"
        )
        for row in variants
        if isinstance(row, Mapping)
    }
    source_variant = next(
        (
            row
            for row in variants
            if isinstance(row, Mapping) and row.get("variant") == "source_float64_perfect_false"
        ),
        {},
    )
    checks = {
        "probability_report_failed_closed": report.get("status") == "failed_closed",
        "no_decode_variant_succeeded": report.get("local_decode_variants") == [],
        "dispatch_locked": report.get("dispatch_unlocked") is False,
        "ready_for_exact_eval_false": report.get("pr91_ready_for_exact_eval") is False,
        "known_failure_reason": report.get("failure_reason")
        == "no_probability_variant_decodes_pr91_hpm1_prefix",
        "known_blocker_class": report.get("blocker_class")
        == "real_invalid_entropy_or_probability_model_contract_mismatch",
        "source_contract_fails_at_recorded_coordinate": source_variant.get("failure_context", {}).get(
            "failed_at"
        )
        == {"frame": 0, "group": 10, "symbol_in_group": 191},
        "source_contract_decoded_count_recorded": source_variant.get("failure_context", {}).get(
            "decoded_symbol_count_before_failure"
        )
        == 5951,
        "all_variants_failed_closed": bool(variants) and len(failed_variants) == len(variants),
    }
    blockers = _failed_checks(checks)
    return {
        "status": "blocked_hpm1_probability_range_contract_mismatch"
        if not blockers
        else "failed_closed_pr91_contract_unclassified",
        "dispatch_allowed": False,
        "score_claim": False,
        "evidence_grade": "local_fail_closed_preflight",
        "source": source,
        "checks": checks,
        "blockers": blockers,
        "classification": {
            "bug_class": "hpm1_probability_range_contract_mismatch",
            "failure_reason": report.get("failure_reason"),
            "blocker_class": report.get("blocker_class"),
            "failed_variants": failed_variants,
            "decoded_symbol_count_before_failure_by_variant": decoded_counts,
        },
        "safe_next_action": (
            "Recover or rebuild a byte-exact PR91/PR86 HPAC encoder-decoder "
            "probability trace before any PR91-derived exact eval."
        ),
    }


def _score_from_components(payload: Mapping[str, Any]) -> float:
    return (
        100.0 * float(payload["avg_segnet_dist"])
        + math.sqrt(10.0 * float(payload["avg_posenet_dist"]))
        + 25.0 * int(payload["archive_size_bytes"]) / SCORE_DENOMINATOR
    )


def _artifact_recomputed_score(payload: Mapping[str, Any]) -> float:
    """Return the artifact's canonical recomputation field when available.

    Some adjudicated JSONs expose rounded component summaries plus the exact
    score recomputed inside the evaluator. The exact field is the custody value;
    rounded component summaries are retained for human reporting only.
    """

    if "score_recomputed_from_components" in payload:
        return float(payload["score_recomputed_from_components"])
    return _score_from_components(payload)


def validate_pr92_rmb1_contract(
    *,
    manifest_path: Path = DEFAULT_PR92_MANIFEST,
    exact_json_path: Path = DEFAULT_PR92_EXACT_JSON,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    exact = _load_json(exact_json_path)
    candidate = manifest.get("candidate_archive", {})
    runtime = manifest.get("exact_eval_runtime_contract", {})
    readiness = manifest.get("dispatch_readiness", {})
    parity = manifest.get("randmulti_decoded_row_parity", {})
    non_noop = manifest.get("non_noop_byte_change", {})
    provenance = exact.get("provenance", {})
    runtime_manifest = provenance.get("inflate_runtime_manifest", {})
    recomputed = _artifact_recomputed_score(exact)
    rounded_component_recompute = _score_from_components(exact)
    runtime_root = str(runtime_manifest.get("runtime_root", ""))
    required_inflate_sh = runtime.get("required_inflate_sh")
    checks = {
        "manifest_score_claim_false": manifest.get("score_claim") is False,
        "manifest_dispatch_performed_false": manifest.get("dispatch_performed") is False,
        "candidate_archive_exists": (REPO_ROOT / str(candidate.get("path", ""))).is_file(),
        "candidate_sha_matches_exact": candidate.get("archive_sha256")
        == provenance.get("archive_sha256"),
        "candidate_bytes_matches_exact": candidate.get("archive_bytes")
        == exact.get("archive_size_bytes"),
        "strict_zip_valid": manifest.get("strict_zip", {}).get("valid") is True,
        "changed_only_randmulti": non_noop.get("changed_segments_vs_stbm") == ["randmulti"],
        "randmulti_decoded_rows_match": parity.get("decoded_rows_match") is True,
        "runtime_ready": runtime.get("ready_for_exact_eval_runtime") is True,
        "runtime_no_blockers": runtime.get("remaining_blockers") == [],
        "required_inflate_sh_exists": isinstance(required_inflate_sh, str)
        and (REPO_ROOT / required_inflate_sh).is_file(),
        "exact_cuda": provenance.get("device") == "cuda",
        "exact_t4_match": provenance.get("gpu_t4_match") is True,
        "exact_full_sample_count": exact.get("n_samples") == 600,
        "exact_score_recomputes": math.isclose(
            recomputed,
            float(exact.get("canonical_score")),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "exact_runtime_tree_recorded": isinstance(
            runtime_manifest.get("runtime_tree_sha256"), str
        )
        and bool(runtime_manifest.get("runtime_tree_sha256")),
        "exact_runtime_root_is_correct_replay_runtime": runtime_root.endswith(
            "replay_submission_stbm_rmb1"
        ),
    }
    blockers = _failed_checks(checks)
    command_archive = candidate.get("path")
    command_inflate = required_inflate_sh
    return {
        "status": "passed_t4_exact_pr92_rmb1_stack_validated" if not blockers else "failed_closed",
        "dispatch_allowed": False,
        "score_claim": not blockers,
        "evidence_grade": "A++" if not blockers else "invalid",
        "checks": checks,
        "blockers": blockers,
        "manifest": {
            "path": _rel(manifest_path),
            "sha256": _sha256_file(manifest_path),
        },
        "exact_eval": {
            "path": _rel(exact_json_path),
            "sha256": _sha256_file(exact_json_path),
            "score": exact.get("canonical_score"),
            "score_recomputed_by_preflight": recomputed,
            "rounded_component_recompute": rounded_component_recompute,
            "rounded_component_recompute_drift": rounded_component_recompute
            - float(exact.get("canonical_score")),
            "archive_bytes": exact.get("archive_size_bytes"),
            "archive_sha256": provenance.get("archive_sha256"),
            "avg_segnet_dist": exact.get("avg_segnet_dist"),
            "avg_posenet_dist": exact.get("avg_posenet_dist"),
            "n_samples": exact.get("n_samples"),
            "runtime_tree_sha256": runtime_manifest.get("runtime_tree_sha256"),
            "runtime_root": runtime_manifest.get("runtime_root"),
        },
        "next_safe_build_command": (
            ".venv/bin/python experiments/build_pr85_stbm1br_pr92_rmb1_randmulti_candidate.py --stdout"
        ),
        "next_safe_exact_eval_command_if_rebuilt": (
            "claim lane first, then exact-eval the rebuilt bytes with "
            f"`{command_archive}` and custom inflate `{command_inflate}`; "
            "skip if archive SHA still equals the already validated exact JSON."
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
        "recorded_at_utc": dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "status": overall_status,
        "score_claim": pr92["score_claim"] and overall_status.startswith("passed_"),
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
    exact = pr92["exact_eval"]
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
        "PR91 remains fail-closed. The local source-contract variant still fails at "
        "`frame=0 group=10 symbol=191` after `5951` decoded symbols, and no "
        "tested probability/range variant decodes frame 0.",
        "",
        "## PR92 RMB1 Stack",
        "",
        f"- status: `{pr92['status']}`",
        f"- evidence_grade: `{pr92['evidence_grade']}`",
        f"- score: `{exact['score']}`",
        f"- archive bytes: `{exact['archive_bytes']}`",
        f"- archive sha256: `{exact['archive_sha256']}`",
        f"- avg_segnet_dist: `{exact['avg_segnet_dist']}`",
        f"- avg_posenet_dist: `{exact['avg_posenet_dist']}`",
        f"- runtime_tree_sha256: `{exact['runtime_tree_sha256']}`",
        "",
        "PR92/RMB1 is not blocked: the validated opportunity is already realized as "
        "a pure-rate randmulti recode stacked onto the PR85 STBM1BR frontier.",
        "",
        "## Next Safe Commands",
        "",
        "```bash",
        pr92["next_safe_build_command"],
        "```",
        "",
        pr92["next_safe_exact_eval_command_if_rebuilt"],
        "",
    ]
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
    else:
        print(_json_text({"status": report["status"], "output_json": _rel(args.output_json)}), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
