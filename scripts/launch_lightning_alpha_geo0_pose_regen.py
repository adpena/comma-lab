#!/usr/bin/env python3
"""Queue Alpha-Geo-0 pose regeneration as a reproducible Lightning Batch job."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("LIGHTNING_DISABLE_VERSION_CHECK", "1")

from tac.deploy.lightning.batch_jobs import (  # noqa: E402
    LightningAdjudicationSpec,
    LightningBatchJobsClient,
    LightningBatchJobSpec,
    _runner_preflight_command,
)


DEFAULT_REMOTE_REPO = "/teamspace/studios/this_studio/pact"
DEFAULT_LOCAL_ARTIFACT_ROOT = "experiments/results/lightning_batch"
DEFAULT_STATE_PATH = ".omx/state/lightning_batch_jobs.json"
DEFAULT_CANDIDATE_ARCHIVE = (
    "experiments/results/lane_12_nerv_20260430_codex_jsonfix40/archive_lane_12_nerv.zip"
)
DEFAULT_BASELINE_ARCHIVE = "experiments/results/lane_g_v3_pfp16/final_deploy_bundle_20260430/archive/archive.zip"
DEFAULT_WARM_POSES = "experiments/results/lane_g_v3_landed/iter_0/optimized_poses.pt"
DEFAULT_GT_POSE_TARGETS = "experiments/results/lane_a_landed/gt_pose_targets.pt"


SSH_AUTH_OPTIONS = (
    "-o",
    "BatchMode=yes",
    "-o",
    "PasswordAuthentication=no",
    "-o",
    "KbdInteractiveAuthentication=no",
    "-o",
    "ServerAliveInterval=15",
    "-o",
    "ServerAliveCountMax=4",
    "-o",
    "TCPKeepAlive=yes",
    "-o",
    "ConnectionAttempts=3",
    "-o",
    "ConnectTimeout=20",
)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    return {"sha256": _sha256_path(path), "bytes": path.stat().st_size}


def _repo_rel(path: str | Path) -> str:
    raw = Path(path)
    resolved = raw.resolve() if raw.is_absolute() else (REPO_ROOT / raw).resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise SystemExit(f"path must stay inside repo for reproducible staging: {path}") from exc


def _remote_repo_path(repo_dir: str, rel_path: str) -> str:
    return f"{repo_dir.rstrip('/')}/{rel_path}"


def _q(value: str | Path) -> str:
    return shlex.quote(str(value))


def _adjudication_spec() -> LightningAdjudicationSpec:
    return LightningAdjudicationSpec(
        baseline_score=1.043987524793892,
        baseline_archive_size_bytes=686635,
        predicted_band_low=0.70,
        predicted_band_high=1.30,
        regression_threshold=1.30,
        delta_key="score_delta_vs_pfp16_a_plus_plus",
        max_sane_score=100.0,
        allow_component_gate_forensic_success=True,
    )


def _metadata_writer_command(
    *,
    python_bin: str,
    output_dir: str,
    job_name: str,
    queue_metadata: dict[str, Any],
    adjudication: LightningAdjudicationSpec,
) -> str:
    payload = {
        "schema_version": 1,
        "job_name": job_name,
        "role": "exact_cuda_eval",
        "queue_metadata": queue_metadata,
        "adjudication": adjudication.asdict(),
        "score_source": "contest_auth_eval.json:score_recomputed_from_components",
        "status_source": "lightning_sdk_job_attributes",
    }
    payload_arg = _q(json.dumps(payload, sort_keys=True))
    out = _q(output_dir)
    py = _q(python_bin)
    return (
        f"{py} - {out}/lightning_queue_metadata.json {out}/archive.zip {payload_arg} <<'PY'\n"
        "import hashlib, json, pathlib, sys, time\n"
        "metadata_path = pathlib.Path(sys.argv[1])\n"
        "archive_path = pathlib.Path(sys.argv[2])\n"
        "payload = json.loads(sys.argv[3])\n"
        "h = hashlib.sha256()\n"
        "with archive_path.open('rb') as f:\n"
        "    for chunk in iter(lambda: f.read(1024 * 1024), b''):\n"
        "        h.update(chunk)\n"
        "payload['expected_archive_sha256'] = h.hexdigest()\n"
        "payload['expected_archive_size_bytes'] = archive_path.stat().st_size\n"
        "payload['artifact_recorded_at_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())\n"
        "metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\\n')\n"
        "print('LIGHTNING_ALPHA_GEO0_METADATA_OK')\n"
        "print(json.dumps({'archive_sha256': payload['expected_archive_sha256'], 'archive_size_bytes': payload['expected_archive_size_bytes']}, sort_keys=True))\n"
        "PY"
    )


def _contest_json_guard_command(*, python_bin: str, output_dir: str) -> str:
    py = _q(python_bin)
    out = _q(output_dir)
    return (
        f"{py} - {out}/contest_auth_eval.json {out}/archive.zip <<'PY'\n"
        "import hashlib, json, math, pathlib, sys\n"
        "contest = json.loads(pathlib.Path(sys.argv[1]).read_text())\n"
        "archive = pathlib.Path(sys.argv[2])\n"
        "h = hashlib.sha256()\n"
        "with archive.open('rb') as f:\n"
        "    for chunk in iter(lambda: f.read(1024 * 1024), b''):\n"
        "        h.update(chunk)\n"
        "archive_sha = h.hexdigest()\n"
        "archive_bytes = archive.stat().st_size\n"
        "prov = contest.get('provenance') or {}\n"
        "score = contest.get('score_recomputed_from_components')\n"
        "assert contest.get('n_samples') == 600, contest.get('n_samples')\n"
        "assert prov.get('device') == 'cuda', prov.get('device')\n"
        "assert prov.get('archive_sha256') == archive_sha, (prov.get('archive_sha256'), archive_sha)\n"
        "assert contest.get('archive_size_bytes') == archive_bytes, (contest.get('archive_size_bytes'), archive_bytes)\n"
        "assert isinstance(score, (int, float)) and math.isfinite(score), score\n"
        "print('LIGHTNING_EXACT_CUDA_EVAL_JSON_OK')\n"
        "print(json.dumps({'score_recomputed_from_components': score, 'archive_sha256': archive_sha, 'archive_size_bytes': archive_bytes, 'gpu_model': prov.get('gpu_model'), 'gpu_t4_match': prov.get('gpu_t4_match')}, sort_keys=True))\n"
        "PY"
    )


def _build_command(args: argparse.Namespace, queue_metadata: dict[str, Any]) -> str:
    repo = args.remote_repo.rstrip("/")
    out = args.output_dir or f"{repo}/{DEFAULT_LOCAL_ARTIFACT_ROOT}/{args.job_name}"
    py = args.python_bin
    adjudication = _adjudication_spec()
    candidate_rel = _repo_rel(args.candidate_archive)
    baseline_rel = _repo_rel(args.baseline_archive)
    warm_rel = _repo_rel(args.warm_poses)
    targets_rel = _repo_rel(args.gt_pose_targets)
    pieces = [
        "set -euo pipefail",
        f"cd {_q(repo)}",
        "test -f env.sh && source env.sh || true",
        f"mkdir -p {_q(out)}",
        f"rm -f {_q(out)}/lightning_queue_metadata.json",
        _runner_preflight_command(python_bin=py, output_dir=out),
        f"export UV_PROJECT_ENVIRONMENT={_q(out)}/uv_project_env",
        "export UV_LINK_MODE=${UV_LINK_MODE:-copy}",
        f"export PYTHONPATH={_q(repo + '/src:' + repo + '/upstream:' + repo)}",
        (
            f"{_q(py)} -u experiments/alpha_geo0_pose_regen.py "
            f"--candidate-archive {_q(_remote_repo_path(repo, candidate_rel))} "
            f"--baseline-archive {_q(_remote_repo_path(repo, baseline_rel))} "
            f"--warm-poses {_q(_remote_repo_path(repo, warm_rel))} "
            f"--gt-pose-targets {_q(_remote_repo_path(repo, targets_rel))} "
            f"--output-dir {_q(out)} "
            "--repo-root . "
            "--upstream-dir upstream "
            "--inflate-sh submissions/robust_current/inflate.sh "
            "--device cuda "
            f"--pose-steps {_q(str(args.pose_steps))} "
            f"--pose-batch-pairs {_q(str(args.pose_batch_pairs))} "
            f"--pose-lr {_q(str(args.pose_lr))} "
            f"--pose-seg-weight {_q(str(args.pose_seg_weight))} "
            f"--pose-weight {_q(str(args.pose_weight))} "
            f"--inflate-timeout {_q(str(args.inflate_timeout))} "
            f"--evaluate-timeout {_q(str(args.evaluate_timeout))} "
            f"--max-seconds {_q(str(args.max_seconds))} "
            f"2>&1 | tee {_q(out)}/alpha_geo0_pose_regen.log"
        ),
        "test -f " + _q(f"{out}/archive.zip"),
        "test -f " + _q(f"{out}/contest_auth_eval.json"),
        "test -f " + _q(f"{out}/contest_auth_eval.adjudicated.json"),
        _metadata_writer_command(
            python_bin=py,
            output_dir=out,
            job_name=args.job_name,
            queue_metadata=queue_metadata,
            adjudication=adjudication,
        ),
        _contest_json_guard_command(python_bin=py, output_dir=out),
    ]
    return "\n".join(pieces)


_BARE_LIGHTNING_HOST_SENTINEL = "ssh." + "lightning.ai"


def _remote_supply_chain_preflight(args: argparse.Namespace) -> None:
    if not args.remote_preflight_ssh_target:
        raise SystemExit("--remote-preflight-ssh-target is required for non-dry-run Studio submit")
    if args.remote_preflight_ssh_target == _BARE_LIGHTNING_HOST_SENTINEL:
        raise SystemExit("bare Lightning host is not acceptable; use a configured SSH alias or user-qualified target")
    remote = (
        "set -euo pipefail; "
        f"cd {_q(args.remote_repo)}; "
        "if [ -f env.sh ]; then source env.sh; fi; "
        f"{_q(args.python_bin)} scripts/scan_lightning_supply_chain.py --quiet --strict"
    )
    subprocess.run(
        ["ssh", *SSH_AUTH_OPTIONS, args.remote_preflight_ssh_target, "bash", "-lc", _q(remote)],
        check=True,
        text=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--candidate-archive", default=DEFAULT_CANDIDATE_ARCHIVE)
    parser.add_argument("--baseline-archive", default=DEFAULT_BASELINE_ARCHIVE)
    parser.add_argument("--warm-poses", default=DEFAULT_WARM_POSES)
    parser.add_argument("--gt-pose-targets", default=DEFAULT_GT_POSE_TARGETS)
    parser.add_argument("--remote-repo", default=os.environ.get("LIGHTNING_REMOTE_PACT", DEFAULT_REMOTE_REPO))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--local-artifact-dir", default=None)
    parser.add_argument("--state-path", default=DEFAULT_STATE_PATH)
    parser.add_argument("--machine", default=os.environ.get("LIGHTNING_MACHINE", "T4"))
    parser.add_argument("--studio", default=os.environ.get("LIGHTNING_STUDIO", "lossy-compression-challenge"))
    parser.add_argument("--teamspace", default=os.environ.get("LIGHTNING_TEAMSPACE", "comma-lab"))
    parser.add_argument("--org", default=os.environ.get("LIGHTNING_ORG"))
    parser.add_argument("--sdk-user", dest="sdk_user", default=os.environ.get("LIGHTNING_SDK_USER", "adpena"))
    parser.add_argument("--python-bin", default=os.environ.get("LIGHTNING_BATCH_PYTHON_BIN", ".venv/bin/python"))
    parser.add_argument("--max-runtime", type=int, default=6 * 3600)
    parser.add_argument("--pose-steps", type=int, default=500)
    parser.add_argument("--pose-batch-pairs", type=int, default=8)
    parser.add_argument("--pose-lr", type=float, default=0.01)
    parser.add_argument("--pose-seg-weight", type=float, default=100.0)
    parser.add_argument("--pose-weight", type=float, default=10.0)
    parser.add_argument("--inflate-timeout", type=int, default=1800)
    parser.add_argument("--evaluate-timeout", type=int, default=1800)
    parser.add_argument("--max-seconds", type=int, default=6 * 3600)
    parser.add_argument("--source-manifest", default=None)
    parser.add_argument("--queue-metadata", action="append", default=[])
    parser.add_argument("--remote-preflight-ssh-target", default=os.environ.get("LIGHTNING_SSH_TARGET"))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _parse_metadata(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--queue-metadata requires KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def main() -> int:
    args = build_parser().parse_args()
    candidate_rel = _repo_rel(args.candidate_archive)
    baseline_rel = _repo_rel(args.baseline_archive)
    warm_rel = _repo_rel(args.warm_poses)
    targets_rel = _repo_rel(args.gt_pose_targets)
    for rel in (candidate_rel, baseline_rel, warm_rel, targets_rel):
        path = REPO_ROOT / rel
        if not path.is_file():
            raise SystemExit(f"required artifact not found: {path}")

    queue_metadata: dict[str, Any] = {
        "schema_version": 1,
        "queued_by": "scripts/launch_lightning_alpha_geo0_pose_regen.py",
        "queued_at_utc": _utc_now(),
        "rationale": "lane12_alpha_geo0_stale_pose_isolation_lightning_cuda",
        "candidate_archive_rel": candidate_rel,
        "candidate_archive_identity": _identity(REPO_ROOT / candidate_rel),
        "baseline_archive_rel": baseline_rel,
        "baseline_archive_identity": _identity(REPO_ROOT / baseline_rel),
        "warm_poses_rel": warm_rel,
        "warm_poses_identity": _identity(REPO_ROOT / warm_rel),
        "gt_pose_targets_rel": targets_rel,
        "gt_pose_targets_identity": _identity(REPO_ROOT / targets_rel),
        "source_manifest": args.source_manifest,
        "no_retraining": True,
        "canonical_path": "archive.zip -> inflate.sh -> upstream/evaluate.py --device cuda",
    }
    queue_metadata.update(_parse_metadata(args.queue_metadata))
    command = _build_command(args, queue_metadata)
    remote_output_dir = args.output_dir or f"{args.remote_repo.rstrip('/')}/{DEFAULT_LOCAL_ARTIFACT_ROOT}/{args.job_name}"
    local_artifact_dir = args.local_artifact_dir or f"{DEFAULT_LOCAL_ARTIFACT_ROOT}/{args.job_name}"
    spec = LightningBatchJobSpec(
        name=args.job_name,
        machine=args.machine,
        command=command,
        studio=args.studio,
        teamspace=args.teamspace,
        user=args.sdk_user,
        interruptible=False,
        max_runtime=args.max_runtime,
        reuse_snapshot=False,
        role="alpha_geo0_exact_eval",
        queue_metadata=queue_metadata,
        local_artifact_dir=local_artifact_dir,
        remote_output_dir=remote_output_dir,
        adjudication=_adjudication_spec(),
    )
    spec.validate()
    if not args.dry_run:
        _remote_supply_chain_preflight(args)
    client = LightningBatchJobsClient(state_path=Path(args.state_path))
    record = client.submit(spec, dry_run=args.dry_run)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
