#!/usr/bin/env python3
"""Read-only static preflight for ADMM no-dead-K CPU build artifacts.

This helper verifies the generated candidate archive/runtime closure without
claiming score, mutating the archive, touching `.omx/state`, or dispatching
exact-eval work.

Usage:

    .venv/bin/python tools/run_admm_no_dead_k_static_preflight.py \
        --build-manifest experiments/results/admm_x_lossy_coarsening_path_b_step6_no_dead_k_20260508T064711Z/build_manifest.json \
        --json-out experiments/results/admm_x_lossy_coarsening_path_b_step6_no_dead_k_20260508T064711Z/static_preflight_no_score.json \
        --fail-if-static-closure-broken

The report has two separate verdicts:

* ``static_archive_runtime_closure_passed`` means archive bytes/SHA, ZIP member
  safety, inflate runtime manifest computation, and shell syntax checked out.
* ``ready_for_exact_eval_dispatch`` remains ``False`` for CPU artifacts and
  carries the build manifest's score/dispatch blockers forward.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import json_text, read_json, repo_relative, sha256_file  # noqa: E402

SCHEMA = "admm_no_dead_k_static_preflight_v1"
BUILD_SCHEMA = "admm_x_lossy_coarsening_path_b_step6_no_dead_k_build.v1"
LANE_ID = "admm_x_lossy_coarsening_path_b_step6_no_dead_k"
EXPECTED_MEMBER = "x"
REQUIRED_CPU_BUILD_BLOCKERS = {
    "cpu_build_rel_err_proxy_not_score_evidence",
    "exact_cuda_auth_eval_not_yet_harvested",
    "requires_contest_auth_eval_json_before_score_promotion_rank_or_kill",
    "apogee_int6_contest_cuda_anchor_required_first",
}
ACCEPTED_RELEASE_PACKET_GAP_CHECKS = {
    "required_file_present:archive.zip": "release_packet_archive_missing_inside_submission_dir",
    "required_file_present:report.txt": "release_packet_report_missing",
    "report_exists": "release_packet_report_missing",
}


def _load_pre_submission_module() -> Any:
    script = REPO_ROOT / "scripts" / "pre_submission_compliance_check.py"
    spec = importlib.util.spec_from_file_location("pre_submission_compliance_check", script)
    if spec is None or spec.loader is None:  # pragma: no cover - sanity
        raise RuntimeError(f"could not load {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _check(name: str, passed: bool, details: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "details": details}


def _resolve_artifact_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _string_list(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    if not all(isinstance(item, str) for item in value):
        return None
    return list(value)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _reject_omx_state_path(path: Path, *, label: str) -> None:
    parts = path.parts
    for index, part in enumerate(parts[:-1]):
        if part == ".omx" and parts[index + 1] == "state":
            raise ValueError(f"refusing to touch .omx/state for {label}: {path}")


def _manifest_checks(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    dispatch_blockers = _string_list(manifest.get("dispatch_blockers"))
    score_blockers = _string_list(manifest.get("score_claim_blockers"))
    dispatch_set = set(dispatch_blockers or [])
    score_set = set(score_blockers or [])
    return [
        _check(
            "build_manifest_schema",
            manifest.get("schema_version") == BUILD_SCHEMA,
            str(manifest.get("schema_version")),
        ),
        _check("build_manifest_lane_id", manifest.get("lane_id") == LANE_ID, str(manifest.get("lane_id"))),
        _check("build_manifest_score_claim_false", manifest.get("score_claim") is False, repr(manifest.get("score_claim"))),
        _check(
            "build_manifest_promotion_eligible_false",
            manifest.get("promotion_eligible") is False,
            repr(manifest.get("promotion_eligible")),
        ),
        _check(
            "build_manifest_rank_or_kill_eligible_false",
            manifest.get("rank_or_kill_eligible") is False,
            repr(manifest.get("rank_or_kill_eligible")),
        ),
        _check(
            "build_manifest_ready_for_exact_eval_dispatch_false",
            manifest.get("ready_for_exact_eval_dispatch") is False,
            repr(manifest.get("ready_for_exact_eval_dispatch")),
        ),
        _check(
            "build_manifest_dispatch_attempted_false",
            manifest.get("dispatch_attempted") is False,
            repr(manifest.get("dispatch_attempted")),
        ),
        _check(
            "build_manifest_evidence_grade_cpu_build",
            manifest.get("evidence_grade") == "[CPU-build]",
            str(manifest.get("evidence_grade")),
        ),
        _check(
            "build_manifest_no_dead_k_wire_format",
            manifest.get("section_K_bytes_in_wire_format") == 0,
            repr(manifest.get("section_K_bytes_in_wire_format")),
        ),
        _check(
            "build_manifest_smoke_decoded_600_latent_pairs",
            manifest.get("smoke_n_latent_pairs_decoded") == 600,
            repr(manifest.get("smoke_n_latent_pairs_decoded")),
        ),
        _check(
            "build_manifest_dispatch_blockers_string_list",
            dispatch_blockers is not None,
            repr(manifest.get("dispatch_blockers")),
        ),
        _check(
            "build_manifest_score_claim_blockers_string_list",
            score_blockers is not None,
            repr(manifest.get("score_claim_blockers")),
        ),
        _check(
            "build_manifest_required_cpu_dispatch_blockers_present",
            REQUIRED_CPU_BUILD_BLOCKERS.issubset(dispatch_set),
            ",".join(sorted(REQUIRED_CPU_BUILD_BLOCKERS - dispatch_set)),
        ),
        _check(
            "build_manifest_required_cpu_score_blockers_present",
            REQUIRED_CPU_BUILD_BLOCKERS.issubset(score_set),
            ",".join(sorted(REQUIRED_CPU_BUILD_BLOCKERS - score_set)),
        ),
    ]


def _bash_syntax_check(submission_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    inflate_sh = submission_dir / "inflate.sh"
    if not inflate_sh.is_file():
        return (
            {
                "command": ["bash", "-n", repo_relative(inflate_sh, REPO_ROOT)],
                "returncode": None,
                "stdout": "",
                "stderr": "inflate.sh missing",
            },
            _check("inflate_sh_bash_syntax", False, repo_relative(inflate_sh, REPO_ROOT)),
        )
    result = subprocess.run(
        ["bash", "-n", str(inflate_sh)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return (
        {
            "command": ["bash", "-n", repo_relative(inflate_sh, REPO_ROOT)],
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        },
        _check(
            "inflate_sh_bash_syntax",
            result.returncode == 0,
            result.stderr.strip() or repo_relative(inflate_sh, REPO_ROOT),
        ),
    )


def _runtime_cache_checks(submission_dir: Path) -> list[dict[str, Any]]:
    if not submission_dir.is_dir():
        return [_check("submission_runtime_python_caches_absent", False, str(submission_dir))]
    cache_paths = sorted(
        repo_relative(path, submission_dir)
        for path in submission_dir.rglob("*")
        if path.name == "__pycache__" or path.suffix == ".pyc"
    )
    return [
        _check(
            "submission_runtime_python_caches_absent",
            not cache_paths,
            ",".join(cache_paths),
        )
    ]


def build_static_preflight(build_manifest_path: Path) -> dict[str, Any]:
    """Return a deterministic no-score static preflight report."""

    _reject_omx_state_path(build_manifest_path, label="build manifest")
    payload = read_json(build_manifest_path)
    if not isinstance(payload, dict):
        raise ValueError(f"{build_manifest_path} must contain a JSON object")
    manifest = payload

    archive_path = _resolve_artifact_path(manifest.get("archive_relpath"))
    submission_dir = _resolve_artifact_path(manifest.get("submission_dir_relpath"))
    if archive_path is None:
        archive_path = Path("__missing_archive_relpath__")
    if submission_dir is None:
        submission_dir = Path("__missing_submission_dir_relpath__")

    expected_sha = manifest.get("archive_sha256")
    expected_bytes = manifest.get("archive_bytes")
    psc = _load_pre_submission_module()
    pre_args = [
        "--submission-dir",
        str(submission_dir),
        "--archive",
        str(archive_path),
        "--expect-single-member",
        EXPECTED_MEMBER,
    ]
    if isinstance(expected_sha, str):
        pre_args.extend(["--expected-archive-sha256", expected_sha])
    if isinstance(expected_bytes, int):
        pre_args.extend(["--expected-archive-size-bytes", str(expected_bytes)])
    pre_report = psc.build_report(psc.build_arg_parser().parse_args(pre_args))

    pre_failed = [check for check in pre_report["checks"] if not check["passed"]]
    accepted_release_gaps = [
        check for check in pre_failed if check["name"] in ACCEPTED_RELEASE_PACKET_GAP_CHECKS
    ]
    unexpected_pre_failures = [
        check for check in pre_failed if check["name"] not in ACCEPTED_RELEASE_PACKET_GAP_CHECKS
    ]

    checks = _manifest_checks(manifest)
    if archive_path.is_file() and isinstance(expected_sha, str):
        checks.append(
            _check("build_manifest_archive_sha_matches_file", sha256_file(archive_path) == expected_sha.lower(), str(archive_path))
        )
    else:
        checks.append(_check("build_manifest_archive_sha_matches_file", False, str(archive_path)))
    if archive_path.is_file() and isinstance(expected_bytes, int):
        checks.append(
            _check("build_manifest_archive_bytes_matches_file", archive_path.stat().st_size == expected_bytes, str(archive_path))
        )
    else:
        checks.append(_check("build_manifest_archive_bytes_matches_file", False, str(archive_path)))
    bash_syntax, bash_check = _bash_syntax_check(submission_dir)
    checks.append(bash_check)
    checks.extend(_runtime_cache_checks(submission_dir))

    manifest_blockers = [check["name"] for check in checks if not check["passed"]]
    pre_submission_blockers = [
        f"pre_submission_static_unexpected_failure:{check['name']}"
        for check in unexpected_pre_failures
    ]
    static_blockers = _dedupe([*manifest_blockers, *pre_submission_blockers])
    release_gap_blockers = _dedupe(
        [ACCEPTED_RELEASE_PACKET_GAP_CHECKS[check["name"]] for check in accepted_release_gaps]
    )
    manifest_dispatch_blockers = _string_list(manifest.get("dispatch_blockers")) or []
    readiness_blockers = _dedupe(
        [
            *manifest_dispatch_blockers,
            *release_gap_blockers,
            "contest_auth_eval_json_missing",
            "archive_manifest_json_missing",
            "active_dispatch_claim_required_before_dispatch",
        ]
    )
    static_passed = not static_blockers

    return {
        "schema": SCHEMA,
        "build_manifest": {
            "path": repo_relative(build_manifest_path, REPO_ROOT),
            "sha256": sha256_file(build_manifest_path),
            "schema_version": manifest.get("schema_version"),
            "lane_id": manifest.get("lane_id"),
            "archive_relpath": manifest.get("archive_relpath"),
            "archive_bytes": manifest.get("archive_bytes"),
            "archive_sha256": manifest.get("archive_sha256"),
            "submission_dir_relpath": manifest.get("submission_dir_relpath"),
            "evidence_grade": manifest.get("evidence_grade"),
            "dispatch_blockers": manifest_dispatch_blockers,
        },
        "pre_submission_static": {
            "passed": pre_report["passed"],
            "archive": pre_report["archive"],
            "submission_runtime": pre_report["submission_runtime"],
            "accepted_release_packet_gap_checks": accepted_release_gaps,
            "unexpected_failed_checks": unexpected_pre_failures,
        },
        "bash_syntax": bash_syntax,
        "checks": checks,
        "static_blockers": static_blockers,
        "readiness_blockers": readiness_blockers,
        "static_archive_runtime_closure_passed": static_passed,
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "dispatch_attempted": False,
        "archive_mutation_attempted": False,
        "omx_state_touched": False,
        "score_promotion_attempted": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-manifest", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--fail-if-static-closure-broken", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.json_out is not None:
            _reject_omx_state_path(args.json_out, label="json output")
        report = build_static_preflight(args.build_manifest)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"admm no-dead-k static preflight failed: {exc}") from None

    text = json_text(report)
    if args.json_out is None:
        print(text, end="")
    else:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    if args.fail_if_static_closure_broken and not report["static_archive_runtime_closure_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
