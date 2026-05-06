#!/usr/bin/env python3
"""Build a deterministic WR01/HNeRV exact-eval operator packet."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_JOB_NAME = "exact_eval_wr01_apply_pr106x_half_20260506"
DEFAULT_LANE_ID = "wr01_apply_pr106x_half"
DEFAULT_ARCHIVE = Path(
    "experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/"
    "hnerv_wavelet_apply_transform_candidate.zip"
)
DEFAULT_BASELINE_JSON = Path(
    "experiments/results/lightning_batch/"
    "exact_eval_public_pr106_belt_and_suspenders_xrepack_t4_20260504T1342Z/"
    "contest_auth_eval.adjudicated.json"
)
DEFAULT_INFLATE_SH = Path(
    "experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/"
    "source/submissions/belt_and_suspenders/inflate.sh"
)
DEFAULT_UPSTREAM_DIR = Path(
    "experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/source"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/results/lightning_batch/exact_eval_wr01_apply_pr106x_half_20260506"
)
DEFAULT_RESULT_DIR = Path(
    "experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex"
)
REQUIRED_ENV = (
    "LIGHTNING_SSH_TARGET",
    "LIGHTNING_REMOTE_PACT",
    "LIGHTNING_UPSTREAM_DIR",
    "LIGHTNING_TEAMSPACE",
    "LIGHTNING_STUDIO",
    "LIGHTNING_SDK_USER",
)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads((REPO_ROOT / path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _q(value: object) -> str:
    text = str(value)
    if text.startswith("$") or text.startswith("${"):
        return text
    return shlex.quote(text)


def _one_liner(cmd: list[str]) -> str:
    return " ".join(_q(item) for item in cmd)


def _artifact_status(path: Path) -> dict[str, Any]:
    full = REPO_ROOT / path
    return {
        "path": path.as_posix(),
        "exists": full.is_file(),
        "bytes": full.stat().st_size if full.is_file() else None,
    }


def build_packet(args: argparse.Namespace) -> dict[str, Any]:
    result_dir = args.result_dir
    archive = args.archive
    preflight = result_dir / "public_replay_preflight.json"
    compliance = result_dir / "pre_submission_compliance.json"
    dry_run = result_dir / "lightning_exact_eval_dry_run.json"
    payload_diff = result_dir / "payload_section_diff_vs_pr106x.json"
    strength_summary = Path("experiments/results/hnerv_wavelet_apply_transform_wr01_strength_summary_20260506_codex.json")
    manifest = result_dir / "manifest.json"

    preflight_payload = _read_json(preflight)
    compliance_payload = _read_json(compliance)
    payload_diff_payload = _read_json(payload_diff)
    dry_run_payload = _read_json(dry_run)
    missing_env = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    artifacts = [
        archive,
        args.baseline_json,
        manifest,
        preflight,
        compliance,
        payload_diff,
        dry_run,
        strength_summary,
    ]
    missing_artifacts = [row["path"] for row in map(_artifact_status, artifacts) if not row["exists"]]
    preflight_ready = preflight_payload.get("ready_for_exact_eval_dispatch") is True
    compliance_ok = not [
        check for check in compliance_payload.get("checks", []) if isinstance(check, dict) and check.get("ok") is False
    ]
    payload_diff_ready = (
        payload_diff_payload.get("ready_for_archive_preflight") is True
        and payload_diff_payload.get("changed_section_count") == 1
        and not payload_diff_payload.get("blockers")
    )
    dry_run_ready = (dry_run_payload.get("submit_readiness") or {}).get("ok") is True

    notes = (
        f"WR01 half exact CUDA eval; archive_sha256={args.archive_sha256} "
        f"bytes={args.archive_bytes}; static_preflight={preflight.as_posix()}"
    )
    claim_cmd = [
        ".venv/bin/python",
        "tools/claim_lane_dispatch.py",
        "claim",
        "--lane-id",
        args.lane_id,
        "--platform",
        "lightning",
        "--instance-job-id",
        args.job_name,
        "--agent",
        args.agent,
        "--predicted-eta-utc",
        args.predicted_eta_utc,
        "--status",
        "active_exact_eval",
        "--notes",
        notes,
    ]
    submit_cmd = [
        ".venv/bin/python",
        "scripts/lightning_exact_eval_repro.py",
        "--job-name",
        args.job_name,
        "--stage-workspace",
        "--submit",
        "--archive",
        archive.as_posix(),
        "--baseline-json",
        args.baseline_json.as_posix(),
        "--inflate-sh",
        args.inflate_sh.as_posix(),
        "--upstream-dir",
        "$LIGHTNING_UPSTREAM_DIR",
        "--remote",
        "$LIGHTNING_SSH_TARGET",
        "--remote-pact",
        "$LIGHTNING_REMOTE_PACT",
        "--studio",
        "$LIGHTNING_STUDIO",
        "--teamspace",
        "$LIGHTNING_TEAMSPACE",
        "--sdk-user",
        "$LIGHTNING_SDK_USER",
        "--machine",
        "${LIGHTNING_MACHINE:-T4}",
        "--predicted-band",
        "0.18",
        "0.25",
        "--regression-threshold",
        "0.02",
        "--max-posenet-relative",
        "1.25",
        "--max-segnet-relative",
        "1.10",
        "--max-sane-score",
        "1.0",
        "--component-trace",
        "--queue-metadata",
        f"lane={args.lane_id}",
        "--queue-metadata",
        f"archive_manifest={manifest.as_posix()}",
        "--queue-metadata",
        f"public_preflight={preflight.as_posix()}",
        "--queue-metadata",
        f"payload_section_diff={payload_diff.as_posix()}",
        "--extra-artifact",
        manifest.as_posix(),
        "--extra-artifact",
        preflight.as_posix(),
        "--extra-artifact",
        compliance.as_posix(),
        "--extra-artifact",
        payload_diff.as_posix(),
        "--extra-artifact",
        strength_summary.as_posix(),
    ]
    harvest_cmd = [
        ".venv/bin/python",
        "scripts/launch_lightning_batch_job.py",
        "harvest-ssh",
        "--state-path",
        ".omx/state/lightning_batch_jobs.json",
        "--job-name",
        args.job_name,
        "--ssh-target",
        "$LIGHTNING_SSH_TARGET",
        "--expected-archive-sha256",
        args.archive_sha256,
        "--expected-archive-size-bytes",
        str(args.archive_bytes),
        "--require-adjudication",
    ]
    blockers = []
    if missing_artifacts:
        blockers.append("missing_artifacts")
    if not preflight_ready:
        blockers.append("public_replay_preflight_not_ready")
    if not compliance_ok:
        blockers.append("pre_submission_compliance_failed")
    if not payload_diff_ready:
        blockers.append("payload_section_diff_not_ready")
    if not dry_run_ready:
        blockers.append("lightning_dry_run_not_ready")
    if missing_env:
        blockers.append("missing_lightning_environment")
    return {
        "schema_version": 1,
        "tool": "tools/build_wr01_exact_eval_packet.py",
        "recorded_at_utc": _utc_now(),
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_submit": not blockers,
        "blockers": blockers,
        "missing_env": missing_env,
        "missing_artifacts": missing_artifacts,
        "lane_id": args.lane_id,
        "job_name": args.job_name,
        "archive_sha256": args.archive_sha256,
        "archive_bytes": args.archive_bytes,
        "preflight_ready": preflight_ready,
        "compliance_ok": compliance_ok,
        "payload_diff_ready": payload_diff_ready,
        "dry_run_ready": dry_run_ready,
        "artifacts": [_artifact_status(path) for path in artifacts],
        "commands": {
            "claim": _one_liner(claim_cmd),
            "submit": _one_liner(submit_cmd),
            "harvest": _one_liner(harvest_cmd),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-name", default=DEFAULT_JOB_NAME)
    parser.add_argument("--lane-id", default=DEFAULT_LANE_ID)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--baseline-json", type=Path, default=DEFAULT_BASELINE_JSON)
    parser.add_argument("--inflate-sh", type=Path, default=DEFAULT_INFLATE_SH)
    parser.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--archive-sha256", default="d2208ffa41297c40ce5f0bdbbe4767a9831301e382522afd2f6acf455a6b1628")
    parser.add_argument("--archive-bytes", type=int, default=186222)
    parser.add_argument("--agent", default="codex:gpt-5.5")
    parser.add_argument("--predicted-eta-utc", default="2026-05-06T07:30Z")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    payload = build_packet(args)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.json_out:
        (REPO_ROOT / args.json_out).parent.mkdir(parents=True, exist_ok=True)
        (REPO_ROOT / args.json_out).write_text(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
