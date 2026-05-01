#!/usr/bin/env python3
"""Credential-safe reproducible Lightning exact-eval orchestrator.

This wrapper composes the two lower-level Lightning tools:

* ``scripts/lightning_repro_workspace.py`` for manifest-based Studio staging.
* ``scripts/launch_lightning_batch_job.py exact-eval`` for strict CUDA auth eval.

It intentionally does not contain SSH users, key paths, tokens, or Studio
credentials. Operators provide those via environment variables, SSH config
aliases, or CLI flags at runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STATE_DIR = REPO_ROOT / ".omx/state"
DEFAULT_REMOTE_PACT = "/teamspace/studios/this_studio/pact"
DEFAULT_UPSTREAM_DIR = "/teamspace/studios/this_studio/upstream"
REQUIREMENTS_MODES = ("uv-sync", "verify-only", "no-install")


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _env_first(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _repo_rel(path: str | Path, *, repo_root: Path = REPO_ROOT) -> Path:
    raw = Path(path)
    resolved = (repo_root / raw).resolve() if not raw.is_absolute() else raw.resolve()
    try:
        return resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(f"path must stay inside repo for reproducible staging: {path}") from exc


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _archive_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"archive not found: {path}")
    return {"sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _finite_number(payload: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out = float(value)
            if math.isfinite(out):
                return out
    return None


def _load_baseline(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    score = _finite_number(payload, "score_recomputed_from_components", "final_score")
    archive_bytes = payload.get("archive_size_bytes")
    if not isinstance(archive_bytes, int):
        archive_bytes = provenance.get("archive_size_bytes")
    if not isinstance(archive_bytes, int):
        archive_bytes = None
    return {
        "score": score,
        "archive_size_bytes": archive_bytes,
        "avg_posenet_dist": _finite_number(payload, "avg_posenet_dist", "pose_dist"),
        "avg_segnet_dist": _finite_number(payload, "avg_segnet_dist", "seg_dist"),
        "archive_sha256": provenance.get("archive_sha256"),
        "device": provenance.get("device"),
        "gpu_t4_match": provenance.get("gpu_t4_match"),
    }


def _parse_metadata(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--queue-metadata requires KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--queue-metadata key is empty in {item!r}")
        out[key] = value.strip()
    return out


def _fmt_number(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return format(float(value), ".17g")


def _append_optional(cmd: list[str], flag: str, value: str | int | float | None) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def _dedupe_preserve(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--archive", required=True, help="Local repo-relative archive to stage and evaluate.")
    parser.add_argument("--extra-artifact", action="append", default=[], help="Additional repo artifact to stage.")
    parser.add_argument("--baseline-json", default=None, help="Baseline contest_auth_eval.json for adjudication defaults.")
    parser.add_argument("--remote", default=_env_first("LIGHTNING_SSH_TARGET", "LIGHTNING_REMOTE", "REMOTE"))
    parser.add_argument("--remote-pact", default=os.environ.get("LIGHTNING_REMOTE_PACT", DEFAULT_REMOTE_PACT))
    parser.add_argument("--remote-archive-path", default=None, help="Override archive path visible inside the Batch job.")
    parser.add_argument("--upstream-dir", default=os.environ.get("LIGHTNING_UPSTREAM_DIR", DEFAULT_UPSTREAM_DIR))
    parser.add_argument("--studio", default=os.environ.get("LIGHTNING_STUDIO"))
    parser.add_argument("--image", default=os.environ.get("LIGHTNING_IMAGE"))
    parser.add_argument("--teamspace", default=os.environ.get("LIGHTNING_TEAMSPACE"))
    parser.add_argument("--org", default=os.environ.get("LIGHTNING_ORG"))
    parser.add_argument("--sdk-user", dest="sdk_user", default=os.environ.get("LIGHTNING_SDK_USER"))
    parser.add_argument("--machine", default=os.environ.get("LIGHTNING_MACHINE", "T4"))
    parser.add_argument("--requirements-mode", choices=REQUIREMENTS_MODES, default=os.environ.get("LIGHTNING_REQUIREMENTS_MODE", "uv-sync"))
    parser.add_argument("--python-bin", default=os.environ.get("LIGHTNING_PYTHON_BIN"))
    parser.add_argument("--batch-python-bin", default=os.environ.get("LIGHTNING_BATCH_PYTHON_BIN"))
    parser.add_argument("--source", action="append", default=None, help="Override staged source path; repeatable.")
    parser.add_argument("--stage-workspace", action="store_true", help="Run manifest staging over SSH before queuing.")
    parser.add_argument("--require-stage-cuda", action="store_true", help="Require CUDA during SSH staging runtime check.")
    parser.add_argument("--stage-only", action="store_true", help="Stage and verify workspace, but do not queue a Batch job.")
    parser.add_argument("--submit", action="store_true", help="Submit the Lightning Batch Job. Default is dry-run queue only.")
    parser.add_argument(
        "--allow-unstaged-submit",
        action="store_true",
        help="Allow --submit without --stage-workspace when remote custody was verified separately.",
    )
    parser.add_argument(
        "--allow-skip-remote-preflight-reason",
        default=None,
        help=(
            "Break-glass reason passed to launch_lightning_batch_job.py when "
            "a Studio submit cannot run SSH remote preflight."
        ),
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--manifest-out", default=None)
    parser.add_argument("--plan-out", default=None)
    parser.add_argument("--queue-record-out", default=None)
    parser.add_argument("--state-path", default=None)
    parser.add_argument("--local-artifact-dir", default=None)
    parser.add_argument("--output-dir", default=None, help="Writable remote output dir. Defaults under remote pact results.")
    parser.add_argument("--plan-only", action="store_true", help="Write/print plan JSON without executing commands.")
    parser.add_argument("--queue-metadata", action="append", default=[])

    parser.add_argument("--baseline-score", type=float)
    parser.add_argument("--baseline-archive-bytes", type=int)
    parser.add_argument("--predicted-band", nargs=2, type=float, metavar=("LOW", "HIGH"))
    parser.add_argument("--regression-threshold", type=float)
    parser.add_argument("--delta-key", default="score_delta_vs_baseline")
    parser.add_argument("--max-posenet-dist", type=float)
    parser.add_argument("--max-segnet-dist", type=float)
    parser.add_argument("--baseline-posenet-dist", "--reference-posenet-dist", dest="baseline_posenet_dist", type=float)
    parser.add_argument("--baseline-segnet-dist", "--reference-segnet-dist", dest="baseline_segnet_dist", type=float)
    parser.add_argument("--max-posenet-relative", type=float)
    parser.add_argument("--max-segnet-relative", type=float)
    parser.add_argument("--component-reference-label", default=None)
    parser.add_argument("--max-sane-score", type=float, default=10.0)
    return parser


def _resolve_plan_paths(args: argparse.Namespace, *, repo_root: Path) -> dict[str, str]:
    run_id = args.run_id or args.job_name
    return {
        "run_id": run_id,
        "manifest_out": args.manifest_out or f".omx/state/{run_id}_manifest.json",
        "local_supply_chain_scan": f".omx/state/{run_id}_local_lightning_supply_chain_scan.json",
        "plan_out": args.plan_out or f".omx/state/{run_id}_lightning_exact_eval_repro_plan.json",
        "queue_record_out": args.queue_record_out or f".omx/state/{run_id}_lightning_batch_record.json",
        "local_artifact_dir": args.local_artifact_dir or f"experiments/results/lightning_batch/{args.job_name}",
    }


def build_plan(args: argparse.Namespace, *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    if args.stage_only and not args.stage_workspace:
        raise ValueError("--stage-only requires --stage-workspace")
    if args.studio and args.image:
        raise ValueError("--studio and --image are mutually exclusive")
    if args.submit and not args.stage_workspace and not args.allow_unstaged_submit:
        raise ValueError("--submit requires --stage-workspace or --allow-unstaged-submit")
    if args.submit and not (args.studio or args.image):
        raise ValueError("--submit requires --studio or --image to avoid SDK-default ambiguity")
    if args.stage_workspace and not args.remote:
        raise ValueError("--stage-workspace requires --remote or LIGHTNING_SSH_TARGET")

    archive_rel = _repo_rel(args.archive, repo_root=repo_root)
    archive_path = repo_root / archive_rel
    identity = _archive_identity(archive_path)
    paths = _resolve_plan_paths(args, repo_root=repo_root)

    baseline_rel: Path | None = None
    baseline_payload: dict[str, Any] = {}
    if args.baseline_json:
        baseline_rel = _repo_rel(args.baseline_json, repo_root=repo_root)
        baseline_payload = _load_baseline(repo_root / baseline_rel)

    queue_needed = not args.stage_only
    baseline_score = args.baseline_score if args.baseline_score is not None else baseline_payload.get("score")
    baseline_archive_bytes = (
        args.baseline_archive_bytes
        if args.baseline_archive_bytes is not None
        else baseline_payload.get("archive_size_bytes")
    )
    baseline_posenet_dist = (
        args.baseline_posenet_dist
        if args.baseline_posenet_dist is not None
        else baseline_payload.get("avg_posenet_dist")
    )
    baseline_segnet_dist = (
        args.baseline_segnet_dist
        if args.baseline_segnet_dist is not None
        else baseline_payload.get("avg_segnet_dist")
    )
    component_reference_label = args.component_reference_label
    if component_reference_label is None:
        component_reference_label = baseline_rel.as_posix() if baseline_rel else "baseline"

    if queue_needed:
        missing = []
        if baseline_score is None:
            missing.append("--baseline-score or --baseline-json with score")
        if args.predicted_band is None:
            missing.append("--predicted-band LOW HIGH")
        if args.regression_threshold is None:
            missing.append("--regression-threshold")
        if args.max_posenet_relative is not None and baseline_posenet_dist is None:
            missing.append("--baseline-posenet-dist or --baseline-json with avg_posenet_dist")
        if args.max_segnet_relative is not None and baseline_segnet_dist is None:
            missing.append("--baseline-segnet-dist or --baseline-json with avg_segnet_dist")
        if missing:
            raise ValueError("exact-eval queue requires " + ", ".join(missing))

    artifact_rels = [archive_rel.as_posix()]
    for item in args.extra_artifact:
        artifact_rels.append(_repo_rel(item, repo_root=repo_root).as_posix())
    if baseline_rel is not None:
        artifact_rels.append(baseline_rel.as_posix())
    artifact_rels = _dedupe_preserve(artifact_rels)

    local_scan_cmd = [
        sys.executable,
        "scripts/scan_lightning_supply_chain.py",
        "--json-out",
        paths["local_supply_chain_scan"],
        "--strict",
        "--quiet",
    ]

    stage_cmd: list[str] | None = None
    if args.stage_workspace:
        stage_cmd = [
            sys.executable,
            "scripts/lightning_repro_workspace.py",
            "--remote",
            args.remote,
            "--remote-pact",
            args.remote_pact,
            "--run-id",
            paths["run_id"],
            "--manifest-out",
            paths["manifest_out"],
            "--requirements-mode",
            args.requirements_mode,
        ]
        if args.python_bin:
            stage_cmd.extend(["--python-bin", args.python_bin])
        if args.require_stage_cuda:
            stage_cmd.append("--require-cuda")
        if args.source is not None:
            for source in args.source:
                stage_cmd.extend(["--source", source])
        for artifact in artifact_rels:
            stage_cmd.extend(["--artifact", artifact])

    batch_python_bin = args.batch_python_bin
    if batch_python_bin is None:
        batch_python_bin = args.python_bin if args.requirements_mode == "verify-only" and args.python_bin else ".venv/bin/python"

    remote_archive = args.remote_archive_path or f"{args.remote_pact.rstrip('/')}/{archive_rel.as_posix()}"
    remote_output_dir = args.output_dir
    if remote_output_dir is not None and remote_output_dir.rstrip("/") == "/teamspace/jobs":
        raise ValueError("--output-dir must be writable; /teamspace/jobs is a read-only artifact view")
    if remote_output_dir is not None and remote_output_dir.startswith("/teamspace/jobs/"):
        raise ValueError(
            "--output-dir must not target /teamspace/jobs/...; that read-only artifact view "
            "is not a writable Studio workspace path"
        )

    metadata = {
        "wrapper": "scripts/lightning_exact_eval_repro.py",
        "run_id": paths["run_id"],
        "archive_rel": archive_rel.as_posix(),
        "source_manifest": paths["manifest_out"],
        "local_supply_chain_scan": paths["local_supply_chain_scan"],
    }
    if baseline_rel is not None:
        metadata["baseline_json"] = baseline_rel.as_posix()
    metadata.update(_parse_metadata(args.queue_metadata))

    queue_cmd: list[str] | None = None
    if queue_needed:
        assert baseline_score is not None
        assert args.predicted_band is not None
        assert args.regression_threshold is not None
        queue_cmd = [
            sys.executable,
            "scripts/launch_lightning_batch_job.py",
            "exact-eval",
            "--job-name",
            args.job_name,
            "--archive",
            remote_archive,
            "--repo-dir",
            args.remote_pact,
            "--upstream-dir",
            args.upstream_dir,
            "--machine",
            args.machine,
            "--python-bin",
            batch_python_bin,
            "--expected-archive-sha256",
            identity["sha256"],
            "--expected-archive-size-bytes",
            str(identity["size_bytes"]),
            "--local-artifact-dir",
            paths["local_artifact_dir"],
            "--adjudicate",
            "--baseline-score",
            _fmt_number(float(baseline_score)),
            "--predicted-band",
            _fmt_number(args.predicted_band[0]),
            _fmt_number(args.predicted_band[1]),
            "--regression-threshold",
            _fmt_number(args.regression_threshold),
            "--delta-key",
            args.delta_key,
            "--component-reference-label",
            component_reference_label,
            "--max-sane-score",
            _fmt_number(args.max_sane_score),
        ]
        if not args.submit:
            queue_cmd.append("--dry-run")
        if args.submit:
            queue_cmd.extend(["--source-manifest", paths["manifest_out"]])
            if args.remote:
                queue_cmd.extend(["--remote-preflight-ssh-target", args.remote])
            elif args.allow_skip_remote_preflight_reason:
                queue_cmd.extend(
                    [
                        "--allow-skip-remote-preflight-reason",
                        args.allow_skip_remote_preflight_reason,
                    ]
                )
        _append_optional(queue_cmd, "--state-path", args.state_path)
        _append_optional(queue_cmd, "--output-dir", remote_output_dir)
        _append_optional(queue_cmd, "--studio", args.studio)
        _append_optional(queue_cmd, "--image", args.image)
        _append_optional(queue_cmd, "--teamspace", args.teamspace)
        _append_optional(queue_cmd, "--org", args.org)
        _append_optional(queue_cmd, "--user", args.sdk_user)
        _append_optional(queue_cmd, "--baseline-archive-bytes", baseline_archive_bytes)
        _append_optional(queue_cmd, "--max-posenet-dist", args.max_posenet_dist)
        _append_optional(queue_cmd, "--max-segnet-dist", args.max_segnet_dist)
        _append_optional(queue_cmd, "--baseline-posenet-dist", baseline_posenet_dist)
        _append_optional(queue_cmd, "--baseline-segnet-dist", baseline_segnet_dist)
        _append_optional(queue_cmd, "--max-posenet-relative", args.max_posenet_relative)
        _append_optional(queue_cmd, "--max-segnet-relative", args.max_segnet_relative)
        for key in sorted(metadata):
            queue_cmd.extend(["--queue-metadata", f"{key}={metadata[key]}"])

    commands = {
        "local_supply_chain_scan": local_scan_cmd,
        "stage_workspace": stage_cmd,
        "queue_exact_eval": queue_cmd,
    }
    return {
        "schema_version": 1,
        "tool": "scripts/lightning_exact_eval_repro.py",
        "generated_at_utc": _utc_now(),
        "job_name": args.job_name,
        "submit": bool(args.submit),
        "stage_workspace": bool(args.stage_workspace),
        "stage_only": bool(args.stage_only),
        "archive": {
            "local_path": archive_rel.as_posix(),
            "remote_path": remote_archive,
            "sha256": identity["sha256"],
            "size_bytes": identity["size_bytes"],
        },
        "baseline": {
            "json_path": baseline_rel.as_posix() if baseline_rel else None,
            "loaded": baseline_payload,
            "baseline_score": baseline_score,
            "baseline_archive_bytes": baseline_archive_bytes,
            "baseline_posenet_dist": baseline_posenet_dist,
            "baseline_segnet_dist": baseline_segnet_dist,
            "component_reference_label": component_reference_label,
        },
        "artifacts": artifact_rels,
        "paths": paths,
        "queue_metadata": metadata,
        "commands": commands,
        "command_strings": {
            key: shlex.join(value) if value else None
            for key, value in commands.items()
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _run(cmd: list[str], *, cwd: Path, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("+ " + shlex.join(cmd))
    if capture:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="", file=sys.stderr)
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr)
        return proc
    return subprocess.run(cmd, cwd=cwd, text=True, check=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = build_plan(args)
    plan_out = REPO_ROOT / plan["paths"]["plan_out"]
    _write_json(plan_out, plan)
    print(json.dumps({"status": "PLAN_READY", "plan": str(plan_out), "submit": plan["submit"]}, indent=2, sort_keys=True))
    if args.plan_only:
        return 0

    _run(plan["commands"]["local_supply_chain_scan"], cwd=REPO_ROOT)
    if plan["commands"]["stage_workspace"]:
        _run(plan["commands"]["stage_workspace"], cwd=REPO_ROOT)
    if args.stage_only:
        return 0

    queue_cmd = plan["commands"]["queue_exact_eval"]
    if queue_cmd is None:
        raise RuntimeError("internal error: queue command is missing")
    queue_proc = _run(queue_cmd, cwd=REPO_ROOT, capture=True)
    try:
        queue_record = json.loads(queue_proc.stdout)
    except json.JSONDecodeError:
        queue_record = {"raw_stdout": queue_proc.stdout}
    queue_record_out = REPO_ROOT / plan["paths"]["queue_record_out"]
    _write_json(queue_record_out, queue_record)
    print(json.dumps({"status": "QUEUED" if args.submit else "DRY_RUN_RECORDED", "record": str(queue_record_out)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
