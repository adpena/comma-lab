#!/usr/bin/env python3
"""Dispatch Phase A1 (track1_phase_a1_score_gradient) to a remote GPU.

Phase A1: score-gradient supervision PR101 fine-tune. Council priority: 22/22
ENDORSE, UNANIMOUS HIGHEST PRIORITY (Quantizr/Carmack/Hinton). Reference:
``.omx/research/grand_council_extreme_rigor_track_1_20260508.md``.

This is the ACTUATOR per CLAUDE.md "parallel-dispatch is a FIRST-CLASS
DELIVERABLE". It runs the full chain on the remote GPU:

    1. Stage workspace via lightning_repro_workspace.py (or scp on Vast.ai)
    2. Open lane claim via tools/claim_lane_dispatch.py
    3. Submit dispatch via Lightning SSH (calling
       scripts/remote_track1_phase_a1_score_gradient_pr101.sh) — that script
       chains: bootstrap → train → build_archive → contest_auth_eval --device cuda
    4. Write heartbeat scaffold + manifest stub at
       experiments/results/<lane_id>_<timestamp>/

Returns when the job is FIRED (does not wait for completion). Harvest via
launch_lane_lightning.py status / harvest commands using the printed session_id.

Cost: Lightning T4 ~$0.66/hr × ~3.5h = ~$2.30. Hard cap: $8 per --cost-cap.

Usage:
    .venv/bin/python tools/dispatch_phase_a1_score_gradient_pr101.py \\
        --lane-id track1_phase_a1_score_gradient \\
        --pr101-archive experiments/results/public_pr_intake_full/public_pr101_intake_20260505_auto/archive.zip \\
        --video-path upstream/videos/0.mkv \\
        --pr101-source-dir experiments/results/public_pr_intake_full/public_pr101_intake_20260505_auto/source/submissions/hnerv_ft_microcodec/src \\
        --provider lightning \\
        --gpu-tier T4 \\
        --epochs 200 \\
        --cost-cap 8.0
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.deploy.lightning.batch_jobs import (  # noqa: E402
    LightningBatchJobsClient,
    LightningBatchJobSpec,
)
from tac.deploy.lightning.defaults import (  # noqa: E402
    DEFAULT_LIGHTNING_REMOTE_PACT,
    default_studio,
    default_teamspace,
    default_user,
)


# ---------------------------------------------------------------------------
# Cost gates
# ---------------------------------------------------------------------------

HOURLY_RATES = {
    "lightning_t4": 0.66,
    "lightning_a10g": 1.10,
    "lightning_l40s": 1.95,
    "lightning_a100": 3.00,
    "vastai_4090": 0.30,
    "vastai_a100": 1.40,
}


def estimated_cost_usd(provider: str, gpu_tier: str, hours: float) -> float:
    key = f"{provider}_{gpu_tier.lower()}"
    rate = HOURLY_RATES.get(key)
    if rate is None:
        # Fallback to Lightning T4 rate if unknown.
        rate = HOURLY_RATES["lightning_t4"]
    return float(rate) * float(hours)


# ---------------------------------------------------------------------------
# Lane claim
# ---------------------------------------------------------------------------

def claim_lane(
    *,
    lane_id: str,
    platform: str,
    instance_job_id: str,
    predicted_eta_utc: str,
    notes: str = "",
    status: str = "active_dispatching",
    force: bool = False,
) -> int:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "claim_lane_dispatch.py"),
        "claim",
        "--lane-id", lane_id,
        "--platform", platform,
        "--instance-job-id", instance_job_id,
        "--agent", "codex:dispatch_phase_a1_score_gradient_pr101",
        "--predicted-eta-utc", predicted_eta_utc,
        "--status", status,
    ]
    if notes:
        cmd.extend(["--notes", notes])
    if force:
        cmd.append("--force")
    print(f"[claim] {' '.join(shlex.quote(c) for c in cmd)}")
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
    return proc.returncode


# ---------------------------------------------------------------------------
# Provenance + heartbeat
# ---------------------------------------------------------------------------

def write_heartbeat(lane_dir: Path, message: str) -> None:
    lane_dir.mkdir(parents=True, exist_ok=True)
    with (lane_dir / "heartbeat.txt").open("a") as f:
        f.write(f"{dt.datetime.now(tz=dt.UTC).isoformat(timespec='seconds').replace('+00:00', 'Z')} {message}\n")


def write_dispatch_manifest(
    lane_dir: Path,
    *,
    lane_id: str,
    instance_job_id: str,
    provider: str,
    gpu_tier: str,
    estimated_cost_usd: float,
    predicted_low: float,
    predicted_high: float,
    pr101_archive: str,
    video_path: str,
    pr101_source_dir: str,
    epochs: int,
    timeout_hours: float,
    session_id: str | None,
    dispatch_attempted: bool,
    dispatch_status: str,
    started_at_utc: str,
    predicted_eta_utc: str,
    args_dict: dict,
) -> Path:
    in_flight = dispatch_attempted and dispatch_status == "fired" and bool(session_id)
    manual_verification = dispatch_status == "fired_no_session_id_verify_manually"
    manifest = {
        "lane_id": lane_id,
        "schema_version": "phase_a1_dispatch_manifest_v1",
        "council_memo_ref": ".omx/research/grand_council_extreme_rigor_track_1_20260508.md",
        "council_decision": "A1 — score-gradient supervision (UNANIMOUS HIGHEST PRIORITY)",
        "instance_job_id": instance_job_id,
        "session_id": session_id,
        "provider": provider,
        "gpu_tier": gpu_tier,
        "estimated_cost_usd": estimated_cost_usd,
        "estimated_duration_hours": timeout_hours,
        "predicted_band": [predicted_low, predicted_high],
        "started_at_utc": started_at_utc,
        "predicted_eta_utc": predicted_eta_utc,
        "remote_lane_script": "scripts/remote_track1_phase_a1_score_gradient_pr101.sh",
        "training_script": "experiments/train_score_gradient_pr101_finetune.py",
        "archive_build_tool": "tools/build_pr101_finetuned_archive.py",
        "pr101_archive": pr101_archive,
        "video_path": video_path,
        "pr101_source_dir": pr101_source_dir,
        "epochs": epochs,
        "evidence_grade": (
            "[advisory only — dispatch in flight]"
            if in_flight
            else "[advisory only — no dispatch]"
        ),
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "dispatch_blockers": ["dispatch_in_flight"] if in_flight else [dispatch_status],
        "dispatch_attempted": dispatch_attempted,
        "dispatch_status": dispatch_status,
        "harvest_command_hint": (
            f".venv/bin/python scripts/launch_lane_lightning.py harvest "
            f"--session-id {session_id} --local-dir {lane_dir}/harvested"
            if session_id
            else (
                "verify platform state manually with the instance_job_id; "
                "lane claim was closed terminally because session_id was missing"
                if manual_verification
                else (
                    "use platform-specific status command with the instance_job_id"
                    if in_flight
                    else None
                )
            )
        ),
        "args": args_dict,
    }
    manifest_path = lane_dir / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


# ---------------------------------------------------------------------------
# Lightning dispatch
# ---------------------------------------------------------------------------

def dispatch_lightning(
    *,
    lane_id: str,
    instance_job_id: str,
    pr101_archive: Path,
    video_path: Path,
    pr101_source_dir: Path,
    epochs: int,
    predicted_low: float,
    predicted_high: float,
    estimated_cost_usd: float,
    gpu_tier: str,
    allow_gpu_mismatch: bool,
    print_only: bool,
) -> tuple[str | None, bool]:
    """Submit the lane to Lightning via launch_lane_lightning.py dispatch.

    The remote script reads PR101_ARCHIVE_PATH / VIDEO_PATH / etc. from env.
    Returns (session_id, fired_ok). ``session_id`` may be None when the
    underlying launcher succeeded but did not print a parseable session id.
    """
    lane_script = "scripts/remote_track1_phase_a1_score_gradient_pr101.sh"

    env_args: list[str] = []
    env_args += ["--env", f"LANE_ID={lane_id}"]
    env_args += ["--env", f"DISPATCH_INSTANCE_JOB_ID={instance_job_id}"]
    env_args += ["--env", "DISPATCH_CLAIMS_PATH=.omx/state/active_lane_dispatch_claims.md"]
    env_args += ["--env", f"PR101_ARCHIVE_PATH={pr101_archive.relative_to(REPO_ROOT)}"]
    env_args += ["--env", f"PR101_SOURCE_DIR={pr101_source_dir.relative_to(REPO_ROOT)}"]
    env_args += ["--env", f"VIDEO_PATH={video_path.relative_to(REPO_ROOT)}"]
    env_args += ["--env", f"EPOCHS={epochs}"]
    env_args += ["--env", f"PRED_LOW={predicted_low}"]
    env_args += ["--env", f"PRED_HIGH={predicted_high}"]

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "launch_lane_lightning.py"),
        "dispatch",
        "--lane-script", lane_script,
        "--label", instance_job_id,
        "--predicted-band", str(predicted_low), str(predicted_high),
        "--estimated-cost", str(estimated_cost_usd),
        "--gpu-tier", gpu_tier,
        "--kill-criteria",
        f"max_cost_usd={estimated_cost_usd*1.2:.2f}; "
        f"min(seg_delta_pct,pose_delta_pct)<5%→retire; "
        f"NaN_in_loss→abort",
        *env_args,
    ]
    if allow_gpu_mismatch:
        cmd.append("--allow-gpu-mismatch")

    print(f"[dispatch] {' '.join(shlex.quote(c) for c in cmd)}")
    if print_only:
        print("[dispatch] --print-only: not actually launching")
        return None, False

    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        print(f"[dispatch] FAILED rc={proc.returncode}", file=sys.stderr)
        return None, False

    # Parse session_id from JSON output.
    try:
        # launch_lane_lightning.py prints JSON via _print_json; find first {...} block
        out = proc.stdout
        first_brace = out.find("{")
        last_brace = out.rfind("}")
        if first_brace >= 0 and last_brace > first_brace:
            payload = json.loads(out[first_brace : last_brace + 1])
            return payload.get("session_id"), True
    except Exception as exc:
        print(f"[dispatch] WARN: failed to parse session_id: {exc}", file=sys.stderr)
    return None, True


# ---------------------------------------------------------------------------
# Lightning Batch Jobs dispatch (CPU-only Studio fallback / GPU-on-demand)
# ---------------------------------------------------------------------------

def build_batch_command(args: argparse.Namespace, instance_job_id: str) -> str:
    """Return a single shell script the Lightning Batch worker can run.

    Mirrors the canonical pattern in
    ``tools/lightning_dispatch_pr106_yshift_score_table.py::build_batch_command``:
    switch to the pact repo, create OUT_DIR, set WORKSPACE/TAC_UPSTREAM_DIR/
    PYTHONPATH, run a CUDA preflight that fails-loud if torch.cuda is missing,
    execute the canonical remote script, then harvest contest_auth_eval.json.
    """
    pr101_archive_rel = args.pr101_archive_rel
    pr101_source_rel = args.pr101_source_rel
    video_path_rel = args.video_path_rel
    lane_id = args.lane_id

    env_exports = [
        f"export PYBIN={args.python_bin}",
        "export PYTHONUNBUFFERED=1",
        f"export LANE_ID={lane_id}",
        f"export DISPATCH_INSTANCE_JOB_ID={instance_job_id}",
        'export DISPATCH_CLAIMS_PATH="$PWD/.omx/state/active_lane_dispatch_claims.md"',
        f"export PR101_ARCHIVE_PATH={pr101_archive_rel}",
        f"export PR101_SOURCE_DIR={pr101_source_rel}",
        f"export VIDEO_PATH={video_path_rel}",
        f"export EPOCHS={args.epochs}",
        f"export PRED_LOW={args.predicted_low}",
        f"export PRED_HIGH={args.predicted_high}",
    ]
    return "\n".join(
        [
            "set -euo pipefail",
            (
                f"if [ -d pact ] && [ -f pact/pyproject.toml ]; then cd pact; "
                f"elif [ -d {args.remote_pact} ]; then cd {args.remote_pact}; "
                "else echo FATAL: pact repo not found >&2; exit 2; fi"
            ),
            f'export OUT_DIR="$PWD/experiments/results/lightning_batch/{instance_job_id}"',
            'mkdir -p "$OUT_DIR"',
            'export WORKSPACE="$PWD"',
            'export TAC_UPSTREAM_DIR="$PWD/upstream"',
            'export PYTHONPATH="$PWD/src:$PWD/upstream:$PWD"',
            f'export LOG_DIR="$OUT_DIR"',
            *env_exports,
            "mkdir -p .omx/state",
            (
                f"{args.python_bin} tools/claim_lane_dispatch.py claim "
                f"--lane-id {shlex.quote(lane_id)} "
                "--platform lightning "
                f"--instance-job-id {shlex.quote(instance_job_id)} "
                "--agent codex:remote_track1_phase_a1_score_gradient_pr101 "
                "--status active_dispatching "
                "--force "
                "--notes remote_mirror_of_local_dispatch_claim_for_in-script_verification"
            ),
            (
                "TERMINAL_CLAIM_CLOSED=0\n"
                "close_remote_claim() {\n"
                "  local status=\"$1\"\n"
                "  local notes=\"$2\"\n"
                "  if [ \"$TERMINAL_CLAIM_CLOSED\" = \"1\" ]; then return 0; fi\n"
                f"  {args.python_bin} tools/claim_lane_dispatch.py claim "
                f"--lane-id {shlex.quote(lane_id)} --platform lightning "
                f"--instance-job-id {shlex.quote(instance_job_id)} "
                "--agent codex:phase_a1_lightning_batch_worker "
                "--predicted-eta-utc \"$(date -u +%FT%TZ)\" "
                "--status \"$status\" --notes \"$notes\" --force >/dev/null || true\n"
                "  TERMINAL_CLAIM_CLOSED=1\n"
                "}\n"
                "cleanup_remote_claim() {\n"
                "  local rc=$?\n"
                "  if [ \"$rc\" -ne 0 ]; then\n"
                "    close_remote_claim \"failed_batch_worker_rc_${rc}\" "
                "\"Phase A1 batch worker exited rc=${rc}; no score claim unless remote manifest proves otherwise\"\n"
                "  fi\n"
                "}\n"
                "trap cleanup_remote_claim EXIT"
            ),
            (
                f"{args.python_bin} - <<'PY' > \"$OUT_DIR/lightning_runner_preflight.json\"\n"
                "import subprocess\n"
                "import torch\n"
                "from tac.repo_io import json_text\n"
                "gpu = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], "
                "text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)\n"
                "payload = {\n"
                "  'cuda_available': bool(torch.cuda.is_available()),\n"
                "  'device_count': int(torch.cuda.device_count()),\n"
                "  'torch_version': torch.__version__,\n"
                "  'torch_cuda': getattr(torch.version, 'cuda', None),\n"
                "  'nvidia_smi_returncode': gpu.returncode,\n"
                "  'gpu_names': gpu.stdout.strip().splitlines(),\n"
                "}\n"
                "if not payload['cuda_available']:\n"
                "    raise SystemExit(json_text({'LIGHTNING_RUNNER_CUDA_PREFLIGHT_OK': False, **payload}))\n"
                "print(json_text({'LIGHTNING_RUNNER_CUDA_PREFLIGHT_OK': True, **payload}), end='')\n"
                "PY"
            ),
            (
                "set +e\n"
                "bash scripts/remote_track1_phase_a1_score_gradient_pr101.sh "
                "2>&1 | tee \"$OUT_DIR/batch_run.log\"\n"
                "REMOTE_SCRIPT_RC=${PIPESTATUS[0]}\n"
                "set -e\n"
                "export REMOTE_SCRIPT_RC"
            ),
            (
                f"{args.python_bin} - <<'PY'\n"
                "import os, pathlib, shutil\n"
                "from tac.repo_io import write_json\n"
                "out = pathlib.Path(os.environ['OUT_DIR'])\n"
                "lane_dir = out\n"
                "candidate_eval_dirs = [lane_dir / 'eval_work', lane_dir]\n"
                "score_json = None\n"
                "for d in candidate_eval_dirs:\n"
                "    candidate = d / 'contest_auth_eval.json'\n"
                "    if candidate.is_file():\n"
                "        score_json = candidate\n"
                "        break\n"
                "summary = {\n"
                "  'score_claim': False,\n"
                "  'promotion_requires_adjudication': True,\n"
                "  'lane_dir': str(lane_dir),\n"
                "  'remote_script_rc': int(os.environ.get('REMOTE_SCRIPT_RC') or 0),\n"
                "  'contest_auth_eval_json': str(score_json) if score_json else None,\n"
                "  'contest_auth_eval_json_exists': bool(score_json and score_json.is_file()),\n"
                "}\n"
                "if score_json and score_json.is_file() and score_json.parent != lane_dir:\n"
                "    shutil.copy2(score_json, lane_dir / 'contest_auth_eval.json')\n"
                "    summary['copied_contest_auth_eval_json'] = True\n"
                "write_json(lane_dir / 'track1_phase_a1_batch_summary.json', summary)\n"
                "if not (score_json and score_json.is_file()):\n"
                "    raise SystemExit('FATAL: phase A1 batch did not produce contest_auth_eval.json')\n"
                "if int(os.environ.get('REMOTE_SCRIPT_RC') or 0) != 0:\n"
                "    raise SystemExit(f\"FATAL: phase A1 remote script rc={os.environ.get('REMOTE_SCRIPT_RC')}\")\n"
                "PY"
            ),
        ]
    )


def build_batch_spec(args: argparse.Namespace, instance_job_id: str) -> LightningBatchJobSpec:
    """Build the LightningBatchJobSpec for the A1 lane."""
    output_dir = (
        f"{args.remote_pact}/experiments/results/lightning_batch/{instance_job_id}"
    )
    return LightningBatchJobSpec(
        name=instance_job_id,
        machine=args.machine,
        command=build_batch_command(args, instance_job_id),
        studio=args.studio or None,
        teamspace=args.teamspace or None,
        user=args.user or None,
        cloud_account=args.cloud_account or None,
        max_runtime=args.max_runtime_seconds,
        reuse_snapshot=False,
        role="track1_phase_a1_score_gradient_cuda",
        local_artifact_dir=f"experiments/results/lightning_batch/{instance_job_id}",
        remote_output_dir=output_dir,
        queue_metadata={
            "lane": args.lane_id,
            "predicted_band": [args.predicted_low, args.predicted_high],
            "score_claim": "false",
            "promotion_gate": "requires contest_auth_eval_json adjudication",
            "council_decision": "A1 — score-gradient supervision (UNANIMOUS HIGHEST PRIORITY)",
        },
    )


def _batch_submit_failure_claim(exc: BaseException) -> tuple[str, str]:
    """Classify Lightning submit exceptions for dispatch-claim hygiene."""

    message = str(exc).lower()
    terminal_markers = (
        "couldn't resolve teamspace",
        "could not resolve teamspace",
        "lightning_sdk is required",
    )
    if any(marker in message for marker in terminal_markers):
        return (
            "failed_batch_submit",
            "Phase A1 Lightning Batch Job submit failed before a provider job "
            f"handle was returned: {type(exc).__name__}",
        )
    return (
        "submit_status_unknown_reconcile_before_refire",
        "Phase A1 Lightning Batch Job submit raised "
        f"{type(exc).__name__}; provider state may be ambiguous after the "
        "Job.run boundary. Reconcile Lightning Batch Jobs state before re-fire.",
    )


def submit_batch(
    args: argparse.Namespace,
    *,
    lane_id: str,
    instance_job_id: str,
    predicted_eta_utc: str,
) -> tuple[bool, str | None, str]:
    """Submit the A1 lane via Lightning Batch Jobs.

    Returns ``(fired_ok, batch_record_str_or_None, dispatch_status)``.
    Exceptions from the provider submit call are billing-ambiguous, so the lane
    claim remains non-terminal until an operator reconciles provider state.
    """
    spec = build_batch_spec(args, instance_job_id)
    if args.print_only:
        print("=== batch spec ===")
        print(json.dumps(spec.asdict(), indent=2, default=str))
        return False, None, "print_only_no_dispatch"
    try:
        record = LightningBatchJobsClient().submit(spec, dry_run=args.dry_run_batch)
    except Exception as exc:
        print(f"[submit-batch] FAILED: {exc!r}", file=sys.stderr)
        status, notes = _batch_submit_failure_claim(exc)
        claim_lane(
            lane_id=lane_id,
            platform="lightning",
            instance_job_id=instance_job_id,
            predicted_eta_utc=predicted_eta_utc,
            notes=notes,
            status=status,
            force=True,
        )
        return False, None, status
    if args.dry_run_batch:
        # Close the lane claim terminally for dry-run so it does not block re-fire.
        claim_lane(
            lane_id=lane_id,
            platform="lightning",
            instance_job_id=instance_job_id,
            predicted_eta_utc=predicted_eta_utc,
            notes=(
                "Phase A1 Lightning Batch dry-run only; no CUDA work dispatched"
            ),
            status="completed_dry_run",
            force=True,
        )
        return False, json.dumps(record, default=str, indent=2), "completed_dry_run"
    return True, json.dumps(record, default=str, indent=2), "fired"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--lane-id", default="track1_phase_a1_score_gradient",
                   help="Base lane id (a timestamp suffix is appended for instance_job_id)")
    p.add_argument("--pr101-archive", type=Path, required=True,
                   help="Repo-relative path to PR101 source archive.zip")
    p.add_argument("--pr101-source-dir", type=Path, required=True,
                   help="Repo-relative path to PR101 hnerv_ft_microcodec/src dir")
    p.add_argument("--video-path", type=Path, required=True,
                   help="Repo-relative path to upstream/videos/0.mkv")
    p.add_argument("--provider", choices=["lightning", "vastai"], default="lightning")
    p.add_argument("--gpu-tier", default="T4",
                   help="GPU tier (T4/A10G/L40S/A100/H100). Lightning g4dn.2xlarge maps to T4.")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--predicted-low", type=float, default=0.150,
                   help="Predicted score band lower bound [contest-CUDA]")
    p.add_argument("--predicted-high", type=float, default=0.220,
                   help="Predicted score band upper bound [contest-CUDA]")
    p.add_argument("--timeout-hours", type=float, default=4.0,
                   help="Wall-clock cap for the remote chain (default 4h)")
    p.add_argument("--cost-cap", type=float, default=8.0,
                   help="Hard cap on estimated $; abort dispatch if exceeded")
    p.add_argument("--allow-gpu-mismatch", action="store_true",
                   help="Allow dispatch even if Lightning Studio's GPU doesn't match")
    p.add_argument("--print-only", action="store_true",
                   help="Print resolved invocation without claiming or dispatching")
    p.add_argument("--allow-legacy-studio", action="store_true",
                   help="Allow the legacy Lightning Studio tmux path. Default is "
                        "fail-closed because that path cannot prove a mirrored "
                        "remote dispatch claim before CUDA work in all environments; "
                        "prefer --submit-batch.")
    p.add_argument("--force-claim", action="store_true",
                   help="Pass --force to lane claim (only for explicit conflict resolution)")
    p.add_argument("--output-root", type=Path,
                   default=REPO_ROOT / "experiments" / "results")
    # Lightning Batch Jobs mode (CPU-only Studio fallback / GPU-on-demand).
    p.add_argument("--submit-batch", action="store_true",
                   help="Switch from Studio-SSH dispatch to Lightning Batch Job submission. "
                        "The batch worker requests its own GPU at job-runtime so the "
                        "currently-attached Studio machine does not need a GPU.")
    p.add_argument("--machine", default="g4dn.2xlarge",
                   help="Lightning machine identifier (T4 = g4dn.2xlarge); only used with --submit-batch")
    p.add_argument("--studio", default=default_studio(),
                   help="Lightning Studio name (env LIGHTNING_STUDIO); only used with --submit-batch")
    p.add_argument("--teamspace", default=default_teamspace(),
                   help="Lightning teamspace (env LIGHTNING_TEAMSPACE); only used with --submit-batch")
    p.add_argument("--user", default=default_user(),
                   help="Lightning user (env LIGHTNING_USER); only used with --submit-batch")
    p.add_argument("--cloud-account", default=None,
                   help="Lightning cloud account override; only used with --submit-batch")
    p.add_argument("--max-runtime-seconds", type=int, default=4 * 60 * 60,
                   help="Wall-clock cap inside the Batch Job (default 4h); only used with --submit-batch")
    p.add_argument("--dry-run-batch", action="store_true",
                   help="Submit the batch spec via LightningBatchJobsClient.submit(dry_run=True). "
                        "Records a DRY_RUN row in .omx/state/lightning_batch_jobs.json and does not "
                        "request a GPU.")
    p.add_argument("--remote-pact", default=DEFAULT_LIGHTNING_REMOTE_PACT,
                   help="Remote pact repo path on the Lightning Batch worker; "
                        "only used with --submit-batch")
    p.add_argument("--python-bin", default=".venv/bin/python",
                   help="Python interpreter on the Lightning Batch worker; only used with --submit-batch")
    args = p.parse_args(argv)

    # Path resolution.
    pr101_archive = args.pr101_archive
    if not pr101_archive.is_absolute():
        pr101_archive = REPO_ROOT / pr101_archive
    if not pr101_archive.is_file():
        print(f"FATAL: --pr101-archive not found: {pr101_archive}", file=sys.stderr)
        return 2
    try:
        pr101_archive.relative_to(REPO_ROOT)
    except ValueError:
        print(f"FATAL: --pr101-archive must be inside repo root: {pr101_archive}", file=sys.stderr)
        return 2

    video_path = args.video_path
    if not video_path.is_absolute():
        video_path = REPO_ROOT / video_path
    if not video_path.is_file():
        print(f"FATAL: --video-path not found: {video_path}", file=sys.stderr)
        return 2
    try:
        video_path.relative_to(REPO_ROOT)
    except ValueError:
        print(f"FATAL: --video-path must be inside repo root: {video_path}", file=sys.stderr)
        return 2

    pr101_source_dir = args.pr101_source_dir
    if not pr101_source_dir.is_absolute():
        pr101_source_dir = REPO_ROOT / pr101_source_dir
    if not pr101_source_dir.is_dir():
        print(f"FATAL: --pr101-source-dir not found: {pr101_source_dir}", file=sys.stderr)
        return 2
    try:
        pr101_source_dir.relative_to(REPO_ROOT)
    except ValueError:
        print(f"FATAL: --pr101-source-dir must be inside repo root: {pr101_source_dir}", file=sys.stderr)
        return 2

    # Stash repo-relative posix path strings on args for build_batch_command.
    args.pr101_archive_rel = pr101_archive.relative_to(REPO_ROOT).as_posix()
    args.pr101_source_rel = pr101_source_dir.relative_to(REPO_ROOT).as_posix()
    args.video_path_rel = video_path.relative_to(REPO_ROOT).as_posix()

    # Cost gate.
    cost = estimated_cost_usd(args.provider, args.gpu_tier, args.timeout_hours)
    print(f"[cost-gate] estimated ${cost:.2f} for {args.provider} {args.gpu_tier} × {args.timeout_hours}h "
          f"(cap ${args.cost_cap:.2f})")
    if cost > args.cost_cap:
        print(f"FATAL: estimated cost ${cost:.2f} exceeds cap ${args.cost_cap:.2f}; abort.", file=sys.stderr)
        return 3

    # Lane dir and instance_job_id.
    timestamp = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    instance_job_id = f"{args.lane_id}_{timestamp}"
    lane_dir = args.output_root / instance_job_id
    lane_dir.mkdir(parents=True, exist_ok=True)

    started_at_utc = dt.datetime.now(tz=dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    predicted_eta_utc = (
        dt.datetime.now(tz=dt.UTC) + dt.timedelta(hours=args.timeout_hours)
    ).isoformat(timespec="seconds").replace("+00:00", "Z")

    # Heartbeat (pre-dispatch).
    write_heartbeat(lane_dir, f"DISPATCH_START provider={args.provider} gpu={args.gpu_tier} "
                              f"eta={predicted_eta_utc} cost=${cost:.2f}")

    if args.print_only:
        print("[print-only] skipping lane claim and remote dispatch")
        # Still write a manifest so operator can audit the resolved invocation.
        write_dispatch_manifest(
            lane_dir,
            lane_id=args.lane_id,
            instance_job_id=instance_job_id,
            provider=args.provider,
            gpu_tier=args.gpu_tier,
            estimated_cost_usd=cost,
            predicted_low=args.predicted_low,
            predicted_high=args.predicted_high,
            pr101_archive=str(pr101_archive),
            video_path=str(video_path),
            pr101_source_dir=str(pr101_source_dir),
            epochs=args.epochs,
            timeout_hours=args.timeout_hours,
            session_id=None,
            dispatch_attempted=False,
            dispatch_status="print_only_no_dispatch",
            started_at_utc=started_at_utc,
            predicted_eta_utc=predicted_eta_utc,
            args_dict={k: str(v) for k, v in vars(args).items()},
        )
        if args.submit_batch:
            # Print the resolved batch spec (validates the heredocs locally).
            submit_batch(
                args,
                lane_id=args.lane_id,
                instance_job_id=instance_job_id,
                predicted_eta_utc=predicted_eta_utc,
            )
        else:
            # Still print the invocation for audit.
            dispatch_lightning(
                lane_id=args.lane_id,
                instance_job_id=instance_job_id,
                pr101_archive=pr101_archive,
                video_path=video_path,
                pr101_source_dir=pr101_source_dir,
                epochs=args.epochs,
                predicted_low=args.predicted_low,
                predicted_high=args.predicted_high,
                estimated_cost_usd=cost,
                gpu_tier=args.gpu_tier,
                allow_gpu_mismatch=args.allow_gpu_mismatch,
                print_only=True,
            )
        return 0

    if args.provider != "lightning":
        print(
            f"FATAL: provider {args.provider!r} not yet supported in this dispatcher; "
            "use --provider lightning",
            file=sys.stderr,
        )
        return 5

    if not args.submit_batch and not args.allow_legacy_studio:
        print(
            "FATAL: legacy Lightning Studio dispatch is disabled for A1 readiness. "
            "Use --submit-batch so the worker mirrors and verifies the dispatch "
            "claim before CUDA work, or pass --allow-legacy-studio only after "
            "manually confirming the remote .omx/state claim ledger is current.",
            file=sys.stderr,
        )
        return 7

    # Open lane claim BEFORE GPU spend (CLAUDE.md NON-NEGOTIABLE).
    platform_for_claim = "lightning"
    claim_rc = claim_lane(
        lane_id=args.lane_id,
        platform=platform_for_claim,
        instance_job_id=instance_job_id,
        predicted_eta_utc=predicted_eta_utc,
        notes=(
            f"Phase A1 score-gradient supervision PR101 fine-tune; "
            f"council UNANIMOUS HIGHEST PRIORITY; predicted=[{args.predicted_low}, {args.predicted_high}]; "
            f"cost=${cost:.2f}; archive={pr101_archive.name}"
        ),
        force=args.force_claim,
    )
    if claim_rc != 0:
        print(f"FATAL: lane claim failed rc={claim_rc}; aborting before GPU spend", file=sys.stderr)
        return 4

    # Submit dispatch.
    batch_dispatch_status = ""
    if args.submit_batch:
        fired_ok, batch_record_str, batch_dispatch_status = submit_batch(
            args,
            lane_id=args.lane_id,
            instance_job_id=instance_job_id,
            predicted_eta_utc=predicted_eta_utc,
        )
        if args.dry_run_batch:
            write_heartbeat(lane_dir, "BATCH_DRY_RUN_COMPLETED no GPU work dispatched")
            manifest_path = write_dispatch_manifest(
                lane_dir,
                lane_id=args.lane_id,
                instance_job_id=instance_job_id,
                provider=args.provider,
                gpu_tier=args.gpu_tier,
                estimated_cost_usd=cost,
                predicted_low=args.predicted_low,
                predicted_high=args.predicted_high,
                pr101_archive=str(pr101_archive),
                video_path=str(video_path),
                pr101_source_dir=str(pr101_source_dir),
                epochs=args.epochs,
                timeout_hours=args.timeout_hours,
                session_id=None,
                dispatch_attempted=False,
                dispatch_status="dry_run_batch_no_dispatch",
                started_at_utc=started_at_utc,
                predicted_eta_utc=predicted_eta_utc,
                args_dict={k: str(v) for k, v in vars(args).items()},
            )
            if batch_record_str:
                print("[submit-batch] record:")
                print(batch_record_str)
            print(f"[dispatch] DRY_RUN completed manifest={manifest_path}")
            return 0
        if not fired_ok:
            write_heartbeat(lane_dir, f"BATCH_SUBMIT_NOT_FIRED status={batch_dispatch_status}")
            write_dispatch_manifest(
                lane_dir,
                lane_id=args.lane_id,
                instance_job_id=instance_job_id,
                provider=args.provider,
                gpu_tier=args.gpu_tier,
                estimated_cost_usd=cost,
                predicted_low=args.predicted_low,
                predicted_high=args.predicted_high,
                pr101_archive=str(pr101_archive),
                video_path=str(video_path),
                pr101_source_dir=str(pr101_source_dir),
                epochs=args.epochs,
                timeout_hours=args.timeout_hours,
                session_id=None,
                dispatch_attempted=True,
                dispatch_status=batch_dispatch_status or "failed_batch_submit",
                started_at_utc=started_at_utc,
                predicted_eta_utc=predicted_eta_utc,
                args_dict={k: str(v) for k, v in vars(args).items()},
            )
            return 6
        # Lightning Batch Jobs do not expose a tmux session_id; the
        # instance_job_id IS the canonical handle for status/harvest.
        session_id = instance_job_id if fired_ok else None
        dispatch_ok = fired_ok
        if fired_ok and batch_record_str:
            print("[submit-batch] record:")
            print(batch_record_str)
    else:
        session_id, dispatch_ok = dispatch_lightning(
            lane_id=args.lane_id,
            instance_job_id=instance_job_id,
            pr101_archive=pr101_archive,
            video_path=video_path,
            pr101_source_dir=pr101_source_dir,
            epochs=args.epochs,
            predicted_low=args.predicted_low,
            predicted_high=args.predicted_high,
            estimated_cost_usd=cost,
            gpu_tier=args.gpu_tier,
            allow_gpu_mismatch=args.allow_gpu_mismatch,
            print_only=False,
        )
    if not dispatch_ok:
        claim_lane(
            lane_id=args.lane_id,
            platform=platform_for_claim,
            instance_job_id=instance_job_id,
            predicted_eta_utc=started_at_utc,
            notes="Phase A1 dispatch submission failed before remote job fired",
            status="failed_dispatch_submission",
            force=True,
        )
        write_heartbeat(lane_dir, "DISPATCH_SUBMISSION_FAILED")
        write_dispatch_manifest(
            lane_dir,
            lane_id=args.lane_id,
            instance_job_id=instance_job_id,
            provider=args.provider,
            gpu_tier=args.gpu_tier,
            estimated_cost_usd=cost,
            predicted_low=args.predicted_low,
            predicted_high=args.predicted_high,
            pr101_archive=str(pr101_archive),
            video_path=str(video_path),
            pr101_source_dir=str(pr101_source_dir),
            epochs=args.epochs,
            timeout_hours=args.timeout_hours,
            session_id=None,
            dispatch_attempted=True,
            dispatch_status="failed_dispatch_submission",
            started_at_utc=started_at_utc,
            predicted_eta_utc=predicted_eta_utc,
            args_dict={k: str(v) for k, v in vars(args).items()},
        )
        return 6

    write_heartbeat(lane_dir, f"DISPATCH_FIRED session_id={session_id or 'unknown'}")

    # R3-3 fix (A1 review 2026-05-08): if dispatch_ok=True but session_id=None,
    # the launcher returned success (SSH worked) but we couldn't parse a
    # session_id from its output — usually means the remote script exited
    # before reaching the session-id stamp (e.g., GPU not attached, conda
    # torch CUDA mismatch). Without explicit closure here the lane claim
    # would remain in `active_dispatching` and block any re-fire by raising
    # "REFUSING_DISPATCH: active claim(s) already exist" until manually
    # closed via --force. Close it terminally with a status that signals
    # manual verification is required.
    if not session_id:
        claim_lane(
            lane_id=args.lane_id,
            platform=platform_for_claim,
            instance_job_id=instance_job_id,
            predicted_eta_utc=started_at_utc,
            notes=(
                "Phase A1 dispatch SSH submitted successfully but session_id "
                "could not be parsed from launcher output. Likely cause: "
                "remote script exited before stamping session_id (no GPU / "
                "torch CUDA mismatch). Verify remote state manually before "
                "re-firing; this status is non-active so re-fire is unblocked."
            ),
            status="fired_no_session_id_verify_manually",
            force=True,
        )
        sys.stderr.write(
            "[dispatch] WARNING: session_id was not parsed from launcher "
            "output. Lane claim closed as 'fired_no_session_id_verify_manually' "
            f"so re-fire is unblocked. Verify remote state for "
            f"instance_job_id={instance_job_id} via platform-specific tooling "
            "before assuming the dispatch is live.\n"
        )

    manifest_path = write_dispatch_manifest(
        lane_dir,
        lane_id=args.lane_id,
        instance_job_id=instance_job_id,
        provider=args.provider,
        gpu_tier=args.gpu_tier,
        estimated_cost_usd=cost,
        predicted_low=args.predicted_low,
        predicted_high=args.predicted_high,
        pr101_archive=str(pr101_archive),
        video_path=str(video_path),
        pr101_source_dir=str(pr101_source_dir),
        epochs=args.epochs,
        timeout_hours=args.timeout_hours,
        session_id=session_id,
        dispatch_attempted=True,
        dispatch_status=(
            "fired" if session_id else "fired_no_session_id_verify_manually"
        ),
        started_at_utc=started_at_utc,
        predicted_eta_utc=predicted_eta_utc,
        args_dict={k: str(v) for k, v in vars(args).items()},
    )
    print(f"[dispatch] FIRED session_id={session_id} manifest={manifest_path}")
    if session_id:
        print(f"[harvest-hint] .venv/bin/python scripts/launch_lane_lightning.py status --session-id {session_id}")
        print(f"[harvest-hint] .venv/bin/python scripts/launch_lane_lightning.py harvest --session-id {session_id} --local-dir {lane_dir}/harvested")
    else:
        print(f"[harvest-hint] session_id was not parsed; use instance_job_id={instance_job_id} with platform status tooling")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
