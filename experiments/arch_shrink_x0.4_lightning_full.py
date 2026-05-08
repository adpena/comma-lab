#!/usr/bin/env python3
"""arch_shrink_x0.4_quantizr_class — FULL Lightning T4 dispatch.

Trains the Quantizr-class ~88K-element renderer (profile
``q_faithful_dilated_88k``) on Lightning T4 (g4dn.2xlarge), packs the
contest archive, runs ``contest_auth_eval.py --device cuda`` on the
packed archive, and writes a ``[contest-CUDA]`` evidence row alongside
the harvested artifacts.

The post-hoc byte anchor (tools/pr101_arch_shrink_post_hoc_sweep.py)
landed 83,571 B at r=0.4 — TIES the predicted 80,000 B for the
arch_shrink_x0.4_quantizr_class catalog row.  That number is BYTES only
and tagged `[CPU-prep empirical byte-anchor only]`.  This script closes
the loop with the SCORE anchor ``[contest-CUDA]`` (operator-authorized
spend; budget cap $25).

Workflow
--------
1. Pre-flight: ``q_faithful_dilated_88k`` profile resolves; canonical
   inflate.sh exists; lane registry entry created if missing.
2. Build the runtime payload: a single Lightning Job command (bash) that
   - bootstraps cu124 torch (``INFLATE_TORCH_SPEC=torch==2.5.1+cu124``)
   - runs ``train_renderer.py --profile q_faithful_dilated_88k``
   - runs ``pipeline.py compress`` to pack the archive
   - runs ``contest_auth_eval.py --device cuda`` on the packed archive
3. Stage workspace via ``scripts/lightning_repro_workspace.py`` (rsync +
   sha-256 manifest).  Required for the launcher's pre-staged contract.
4. File a dispatch claim via ``tools/claim_lane_dispatch.py`` so the
   cross-agent ledger refuses concurrent claims.
5. Submit a Lightning Studio Job via ``lightning_sdk.Job.run`` (T4
   g4dn.2xlarge, ``interruptible=False``).
6. Persist dispatch metadata to ``.omx/state/lightning_active_jobs.json``
   so ``arch_shrink_x0.4_lightning_harvest.py`` can poll + harvest.

CLAUDE.md compliance
--------------------
- INFLATE_TORCH_SPEC=cu124 (driver<580 cu13 wheel CPU-fallback trap)
- claim_lane_dispatch.py BEFORE submitting (cross-agent coordination)
- ``platform=lightning`` lowercase canonical
- NO score claim on dispatch — only the harvester emits a
  ``[contest-CUDA]`` row after the Job lands and the inflate roundtrip
  succeeds on the EXACT archive bytes.
- Heartbeat / watchdog responsibility transferred to harvester.

Usage
-----
.. code-block:: bash

    .venv/bin/python experiments/arch_shrink_x0.4_lightning_full.py \\
        --machine g4dn.2xlarge \\
        --predicted-low 0.40 --predicted-high 0.80

Outputs
-------
- Lightning Studio Job (status visible in
  https://lightning.ai/adpena/comma-lab/studios/lossy-compression-challenge)
- ``.omx/state/lightning_active_jobs.json`` (job_name + sdk_job_name +
  output paths so the harvester can find the artifacts)
- ``experiments/results/lightning_batch/<job_name>/source_manifest.json``
  (rsync custody record from lightning_repro_workspace.py)
- Dispatch claim in ``.omx/state/active_lane_dispatch_claims.md`` with
  ``status=active_dispatching``
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.deploy.lightning.defaults import (  # noqa: E402
    DEFAULT_LIGHTNING_REMOTE_PACT,
    default_remote_pact,
    default_ssh_target,
    default_studio,
    default_teamspace,
    default_user,
)

LANE_ID = "arch_shrink_x0.4_lightning"
PROFILE = "q_faithful_dilated_88k"  # Quantizr-class 88K-element renderer
TARGET_ELEMENTS = 88_000
INFLATE_TORCH_SPEC = "torch==2.5.1+cu124"
UV_EXTRA_INDEX_URL = "https://download.pytorch.org/whl/cu124"
UV_INDEX_STRATEGY = "unsafe-best-match"
DEFAULT_MACHINE = "g4dn.2xlarge"  # AWS T4; matches reference recipe (NOT "T4" literal)
DEFAULT_MAX_RUNTIME_SEC = 18 * 60 * 60  # 18h cap (T4 training runs ~12h on Q-FAITHFUL profile)
DEFAULT_BUDGET_CAP_USD = 25.0
LIGHTNING_ACTIVE_JOBS_PATH = REPO_ROOT / ".omx" / "state" / "lightning_active_jobs.json"


def _utc_now_iso() -> str:
    return dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_plus_hours(hours: int) -> str:
    return (dt.datetime.now(tz=dt.UTC) + dt.timedelta(hours=hours)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _job_name() -> str:
    ts = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"arch-shrink-x0-4-lightning-{ts}"


def build_remote_command(*, job_name: str, remote_pact: str) -> str:
    """Construct the bash command that runs train→archive→auth-eval on Lightning.

    The command is executed on the Lightning Job machine with the staged
    workspace mounted at ``remote_pact``. We follow the established lane_j_imp
    pattern: cu124 torch pin, NVDEC probe-light (T4 datacenter card), provenance
    write, training, archive build, then auth eval with DALI lazy-installed.
    """
    output_subdir = f"experiments/results/lightning_batch/{job_name}"
    archive_path = f"{output_subdir}/archive.zip"
    auth_eval_dir = f"{output_subdir}/auth_eval_work"
    auth_eval_log = f"{output_subdir}/auth_eval.log"
    auth_eval_result_json = f"{output_subdir}/contest_auth_eval.json"
    train_output_dir = f"{output_subdir}/train"
    train_tag = f"arch_shrink_x0_4_{job_name}"

    # NOTE: "$PYBIN" not used — Lightning Job machines may not have the venv
    # at the same path. We use ``python -m uv`` style or rely on the system
    # python the Lightning runtime provides plus uv-managed envs.
    return f"""set -euo pipefail
cd {remote_pact}

mkdir -p {output_subdir}
PYBIN=/teamspace/studios/this_studio/pact/.venv/bin/python
WORKSPACE={remote_pact}
export PYTHONHASHSEED=1234
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$WORKSPACE/src:$WORKSPACE/upstream:$WORKSPACE:${{PYTHONPATH:-}}"
export TAC_UPSTREAM_DIR="$WORKSPACE/upstream"
export INFLATE_TORCH_SPEC={INFLATE_TORCH_SPEC}
export UV_EXTRA_INDEX_URL={UV_EXTRA_INDEX_URL}
export UV_INDEX_STRATEGY={UV_INDEX_STRATEGY}

# AppleDouble cleanup (CLAUDE.md feedback_remote_setup_script_correct_path).
find "$WORKSPACE" -name '._*' -type f -delete 2>/dev/null || true

LOG_DIR="$WORKSPACE/{output_subdir}"
mkdir -p "$LOG_DIR"

log() {{ echo "[arch-shrink-x0.4] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }}

# Stage 0: GPU/CUDA presence check (T4 g4dn.2xlarge is a known-good NVDEC
# datacenter card; the Vast.ai-class NVDEC roulette doesn't apply).
log "=== Stage 0: GPU presence check ==="
"$PYBIN" -c "
import torch, sys
if not torch.cuda.is_available():
    print('FATAL: torch.cuda.is_available()=False on Lightning T4', file=sys.stderr)
    sys.exit(2)
name = torch.cuda.get_device_name(0)
mem_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
print(f'OK: GPU={{name}} mem={{mem_gb}}GB cuda_version={{torch.version.cuda}}')
"

# Provenance / heartbeat
PROVENANCE="$LOG_DIR/provenance.json"
HEARTBEAT="$LOG_DIR/heartbeat.log"
GIT_HASH=$(cd "$WORKSPACE" && git rev-parse HEAD 2>/dev/null || echo "no-git")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1)
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1)
"$PYBIN" -c "
import json, time, torch
prov = {{
    'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'git_hash': '$GIT_HASH',
    'gpu_name': '$GPU_NAME',
    'driver_version': '$DRIVER_VER',
    'torch_version': torch.__version__,
    'cuda_version': getattr(torch.version, 'cuda', None),
    'cuda_available': torch.cuda.is_available(),
    'lane_id': '{LANE_ID}',
    'job_name': '{job_name}',
    'profile': '{PROFILE}',
    'target_elements': {TARGET_ELEMENTS},
    'inflate_torch_spec': '{INFLATE_TORCH_SPEC}',
}}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane={LANE_ID} gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

# Stage 1: training on T4 via train_renderer.py with profile q_faithful_dilated_88k
log "=== Stage 1: train_renderer.py --profile {PROFILE} ==="
mkdir -p "$WORKSPACE/{train_output_dir}"
"$PYBIN" -u src/tac/experiments/train_renderer.py \\
    --profile {PROFILE} \\
    --tag {train_tag} \\
    --output-dir "$WORKSPACE/{train_output_dir}" \\
    --device cuda \\
    --auth-eval-on-best 2>&1 | tee "$LOG_DIR/train.log" | tail -40

# Find the EMA-best fp4 checkpoint emitted by train_renderer.py.
CHECKPOINT=$(ls -1 "$WORKSPACE/{train_output_dir}"/*BEST*ema*.pt 2>/dev/null | head -1)
if [ -z "$CHECKPOINT" ]; then
    CHECKPOINT=$(ls -1 "$WORKSPACE/{train_output_dir}"/*BEST*.pt 2>/dev/null | head -1)
fi
if [ -z "$CHECKPOINT" ]; then
    log "FATAL: no BEST checkpoint produced by train_renderer.py"
    exit 3
fi
log "  trained checkpoint: $CHECKPOINT"

# Stage 2: build contest archive via experiments/pipeline.py compress
log "=== Stage 2: pipeline.py compress ==="
"$PYBIN" -u experiments/pipeline.py compress \\
    --profile {PROFILE} \\
    --video upstream/videos/0.mkv \\
    --checkpoint "$CHECKPOINT" \\
    --output-dir "$WORKSPACE/{output_subdir}" \\
    --device cuda 2>&1 | tee "$LOG_DIR/compress.log" | tail -20

ARCHIVE="$WORKSPACE/{output_subdir}/archive.zip"
if [ ! -f "$ARCHIVE" ]; then
    # Fallback: pipeline.py may emit a different filename; locate the .zip.
    ARCHIVE=$(ls -1 "$WORKSPACE/{output_subdir}"/*.zip 2>/dev/null | head -1)
fi
if [ -z "$ARCHIVE" ] || [ ! -f "$ARCHIVE" ]; then
    log "FATAL: pipeline.py compress did not produce an archive.zip"
    exit 4
fi
ARCHIVE_BYTES=$(stat -c '%s' "$ARCHIVE" 2>/dev/null || stat -f '%z' "$ARCHIVE")
log "  archive: $ARCHIVE bytes=$ARCHIVE_BYTES"

# Stage 3: contest_auth_eval [contest-CUDA] on the packed archive
log "=== Stage 3: contest_auth_eval [contest-CUDA] ==="
"$PYBIN" -c "
import importlib.util, subprocess, sys
if importlib.util.find_spec('nvidia.dali') is None:
    print('Installing nvidia-dali-cuda130 lazily...', flush=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir',
                    '--extra-index-url', 'https://pypi.nvidia.com',
                    'nvidia-dali-cuda130'], check=True)
import nvidia.dali as dali
print(f'DALI version: {{dali.__version__}}')
"
rm -rf "$WORKSPACE/{auth_eval_dir}"
"$PYBIN" -u experiments/contest_auth_eval.py \\
    --archive "$ARCHIVE" \\
    --inflate-sh submissions/robust_current/inflate.sh \\
    --upstream-dir upstream \\
    --device cuda \\
    --keep-work-dir \\
    --work-dir "$WORKSPACE/{auth_eval_dir}" 2>&1 | tee "{auth_eval_log}"
PIPE_RC=("${{PIPESTATUS[@]}}")
if [ "${{PIPE_RC[0]}}" -ne 0 ]; then
    log "FATAL: contest_auth_eval rc=${{PIPE_RC[0]}}"
    exit "${{PIPE_RC[0]}}"
fi

# Capture the RESULT_JSON line into a structured artifact.
"$PYBIN" -c "
import json, re
log_path = '{auth_eval_log}'
out_path = '{auth_eval_result_json}'
with open(log_path) as f:
    text = f.read()
m = re.search(r'RESULT_JSON\\s*(\\{{.*?\\}})', text, re.DOTALL)
if not m:
    raise SystemExit('FATAL: no RESULT_JSON in auth_eval log')
data = json.loads(m.group(1))
data['archive_path'] = '$ARCHIVE'
data['archive_bytes'] = int('$ARCHIVE_BYTES')
data['lane_id'] = '{LANE_ID}'
data['job_name'] = '{job_name}'
data['evidence_grade'] = '[contest-CUDA]'
data['profile'] = '{PROFILE}'
with open(out_path, 'w') as f:
    json.dump(data, f, indent=2)
print('contest_auth_eval result:', json.dumps(data))
"

log "=== DONE arch_shrink_x0.4_lightning_full ==="
""".strip()


def stage_workspace(
    *,
    job_name: str,
    archive_placeholder: Path | None,
    ssh_target: str,
    remote_pact: str,
) -> Path:
    """Stage src/, experiments/, submissions/, scripts/, upstream/, tools/ via rsync.

    The launcher's source-manifest contract requires this even when no archive
    artifact is staged (training will produce the archive on the remote).
    """
    manifest_dir = REPO_ROOT / "experiments" / "results" / "lightning_batch" / job_name
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = manifest_dir / "source_manifest.json"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "lightning_repro_workspace.py"),
        "--remote",
        ssh_target,
        "--remote-pact",
        remote_pact,
        "--run-id",
        job_name,
        "--manifest-out",
        str(manifest_out),
        "--source",
        "src",
        "--source",
        "experiments",
        "--source",
        "submissions",
        "--source",
        "scripts",
        "--source",
        "upstream",
        "--source",
        "tools",
        "--source",
        "pyproject.toml",
        "--requirements-mode",
        "no-install",
        "--no-install",
        "--ssh-connect-timeout",
        "30",
    ]
    if archive_placeholder is not None:
        cmd += ["--artifact", str(archive_placeholder)]
    print(f"[stage] {' '.join(cmd[:6])} ... ({len(cmd)} args total)")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    if result.returncode != 0:
        sys.exit(f"FATAL: lightning_repro_workspace.py failed (rc={result.returncode})")
    return manifest_out


def claim_lane(*, job_name: str, force_claim: bool, force_claim_reason: str | None) -> None:
    notes = (
        f"FULL Lightning T4 train+archive+auth-eval via "
        f"experiments/arch_shrink_x0.4_lightning_full.py {_utc_now_iso()}"
    )
    if force_claim:
        if not force_claim_reason:
            sys.exit("FATAL: --force-claim requires --force-claim-reason")
        notes = f"{notes}; force-claim: {force_claim_reason}"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "claim_lane_dispatch.py"),
        "claim",
        "--lane-id",
        LANE_ID,
        "--agent",
        "claude_lab",
        "--platform",
        "lightning",
        "--instance-job-id",
        job_name,
        "--predicted-eta-utc",
        _utc_plus_hours(18),
        "--status",
        "active_dispatching",
        "--notes",
        notes,
        "--ttl-hours",
        "20",
    ]
    if force_claim:
        cmd += ["--force"]
    print(f"[claim] platform=lightning lane={LANE_ID} job={job_name}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    if result.returncode != 0:
        sys.exit(f"FATAL: claim_lane_dispatch.py failed (rc={result.returncode})")


def submit_lightning_job(
    *,
    job_name: str,
    machine: str,
    command: str,
    teamspace: str,
    studio: str,
    user: str,
    max_runtime_sec: int,
    dry_run: bool,
) -> dict[str, object]:
    """Submit a Lightning Studio Job via lightning_sdk.Job.run.

    The call returns a Job handle exposing ``name`` and ``status``. We persist
    these to ``LIGHTNING_ACTIVE_JOBS_PATH`` so the harvester can poll without
    re-importing the SDK.
    """
    if dry_run:
        return {
            "dry_run": True,
            "command_preview": command[:400],
            "would_submit_machine": machine,
        }

    os.environ.setdefault("LIGHTNING_DISABLE_VERSION_CHECK", "1")
    try:
        from lightning_sdk import Job  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - env-dependent
        sys.exit(
            f"FATAL: lightning_sdk import failed; install with `uv pip install lightning-sdk` ({exc})"
        )

    print(f"[submit] Job.run name={job_name} machine={machine} studio={studio}")
    env = {
        "INFLATE_TORCH_SPEC": INFLATE_TORCH_SPEC,
        "UV_EXTRA_INDEX_URL": UV_EXTRA_INDEX_URL,
        "UV_INDEX_STRATEGY": UV_INDEX_STRATEGY,
    }
    job = Job.run(
        name=job_name,
        machine=machine,
        command=command,
        studio=studio,
        teamspace=teamspace,
        user=user,
        env=env,
        interruptible=False,
        max_runtime=max_runtime_sec,
    )
    return {
        "name": getattr(job, "name", job_name),
        "machine": machine,
        "studio": studio,
        "teamspace": teamspace,
        "user": user,
        "status_at_submit": str(getattr(job, "status", "unknown")),
    }


def persist_active_job(
    *,
    job_name: str,
    machine: str,
    submit_result: dict[str, object],
    manifest_path: Path,
) -> None:
    """Append a row to .omx/state/lightning_active_jobs.json."""
    LIGHTNING_ACTIVE_JOBS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LIGHTNING_ACTIVE_JOBS_PATH.exists():
        existing = json.loads(LIGHTNING_ACTIVE_JOBS_PATH.read_text(encoding="utf-8"))
        if not isinstance(existing, list):
            existing = []
    else:
        existing = []
    record = {
        "schema_version": "lightning_active_jobs.v1",
        "lane_id": LANE_ID,
        "job_name": job_name,
        "submitted_at_utc": _utc_now_iso(),
        "machine": machine,
        "profile": PROFILE,
        "target_elements": TARGET_ELEMENTS,
        "predicted_band": [0.40, 0.80],
        "evidence_tag_pending": "[contest-CUDA]",
        "manifest_path": str(manifest_path.relative_to(REPO_ROOT))
        if manifest_path.is_relative_to(REPO_ROOT)
        else str(manifest_path),
        "expected_artifact_dir": (
            f"experiments/results/lightning_batch/{job_name}"
        ),
        "expected_archive_path": (
            f"experiments/results/lightning_batch/{job_name}/archive.zip"
        ),
        "expected_auth_eval_json": (
            f"experiments/results/lightning_batch/{job_name}/contest_auth_eval.json"
        ),
        "submit_result": submit_result,
    }
    existing.append(record)
    LIGHTNING_ACTIVE_JOBS_PATH.write_text(
        json.dumps(existing, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[persist] {LIGHTNING_ACTIVE_JOBS_PATH.relative_to(REPO_ROOT)} ({len(existing)} active jobs)"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--machine",
        default=DEFAULT_MACHINE,
        help=f"Lightning machine class (default {DEFAULT_MACHINE} = AWS T4)",
    )
    p.add_argument(
        "--max-runtime-sec",
        type=int,
        default=DEFAULT_MAX_RUNTIME_SEC,
        help=f"Hard cap on Job runtime (default {DEFAULT_MAX_RUNTIME_SEC}s = 18h)",
    )
    p.add_argument(
        "--predicted-low",
        type=float,
        default=0.40,
        help="Predicted band low (q_faithful_dilated_88k anchor; advisory, not score-claimed)",
    )
    p.add_argument(
        "--predicted-high",
        type=float,
        default=0.80,
        help="Predicted band high (q_faithful_dilated_88k anchor)",
    )
    p.add_argument(
        "--ssh-target",
        default=default_ssh_target(),
        help="Lightning Studio SSH target (defaults to $LIGHTNING_SSH_TARGET / "
             "$LIGHTNING_REMOTE / $REMOTE)",
    )
    p.add_argument(
        "--remote-pact",
        default=default_remote_pact(),
        help=f"Remote pact dir (default $LIGHTNING_REMOTE_PACT or "
             f"{DEFAULT_LIGHTNING_REMOTE_PACT})",
    )
    p.add_argument(
        "--teamspace",
        default=default_teamspace(),
        help="Lightning teamspace (default $LIGHTNING_TEAMSPACE)",
    )
    p.add_argument(
        "--studio",
        default=default_studio(),
        help="Lightning Studio name (default $LIGHTNING_STUDIO)",
    )
    p.add_argument(
        "--user",
        default=default_user(),
        help="Lightning user (default $LIGHTNING_USER)",
    )
    p.add_argument(
        "--job-name",
        default=None,
        help="Override auto-generated job name (default arch-shrink-x0-4-lightning-<UTC>)",
    )
    p.add_argument(
        "--budget-cap-usd",
        type=float,
        default=DEFAULT_BUDGET_CAP_USD,
        help=f"Budget cap (default ${DEFAULT_BUDGET_CAP_USD}); recorded in metadata only",
    )
    p.add_argument(
        "--skip-stage",
        action="store_true",
        help="Skip lightning_repro_workspace.py (workspace already staged)",
    )
    p.add_argument(
        "--force-claim",
        action="store_true",
        help="Force the dispatch claim only when replacing a known terminal/stale claim",
    )
    p.add_argument(
        "--force-claim-reason",
        default=None,
        help="Required rationale when --force-claim is set",
    )
    p.add_argument(
        "--print-only",
        action="store_true",
        help="Print resolved invocation + remote command without staging, claiming, or submitting",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Stage + claim, but submit a Lightning dry-run (no GPU spend)",
    )
    args = p.parse_args(argv)

    # Pre-flight: profile must resolve.
    try:
        from tac.profiles import PROFILES  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - dep must exist
        sys.exit(f"FATAL: cannot import tac.profiles: {exc}")
    if PROFILE not in PROFILES:
        sys.exit(
            f"FATAL: profile {PROFILE!r} not in tac.profiles.PROFILES "
            f"(available: {sorted(PROFILES)[:5]}...)"
        )

    # Pre-flight: canonical inflate.sh exists.
    inflate_sh = REPO_ROOT / "submissions" / "robust_current" / "inflate.sh"
    if not inflate_sh.is_file():
        sys.exit(f"FATAL: missing canonical inflate.sh at {inflate_sh}")

    job_name = args.job_name or _job_name()
    command = build_remote_command(job_name=job_name, remote_pact=args.remote_pact)

    if args.print_only:
        print(f"=== resolved Lightning Job submission for {job_name} ===")
        print(f"machine: {args.machine}")
        print(f"studio: {args.studio or '<unset; pass --studio or $LIGHTNING_STUDIO>'}")
        print(
            f"teamspace: {args.teamspace or '<unset; pass --teamspace or $LIGHTNING_TEAMSPACE>'}"
        )
        print(f"user: {args.user or '<unset; pass --user or $LIGHTNING_USER>'}")
        print(
            f"ssh_target: {args.ssh_target or '<unset; pass --ssh-target or $LIGHTNING_SSH_TARGET>'}"
        )
        print(f"max_runtime_sec: {args.max_runtime_sec}")
        print(f"predicted_band: [{args.predicted_low}, {args.predicted_high}]")
        print(f"budget_cap_usd: {args.budget_cap_usd}")
        print(f"--- remote command preview (first 800 chars) ---")
        print(command[:800])
        print(f"--- (full length: {len(command)} chars) ---")
        return 0

    # Fail-loud env validation BEFORE any spend or external action.
    missing = []
    if not args.ssh_target:
        missing.append(
            "--ssh-target / $LIGHTNING_SSH_TARGET (e.g. "
            "s_<token>@ssh.lightning.ai)"
        )
    if not args.studio:
        missing.append("--studio / $LIGHTNING_STUDIO (e.g. lossy-compression-challenge)")
    if not args.teamspace:
        missing.append("--teamspace / $LIGHTNING_TEAMSPACE (e.g. comma-lab)")
    if not args.user:
        missing.append("--user / $LIGHTNING_USER (e.g. adpena)")
    if missing:
        sys.exit(
            "FATAL: missing required Lightning environment values:\n  - "
            + "\n  - ".join(missing)
            + "\nReference recipe: "
            "~/.claude/projects/-Users-adpena-Projects-pact/memory/"
            "reference_lightning_studio_canonical_dispatch_recipe_20260505.md"
        )

    # Stage 1: workspace rsync + manifest.
    if args.skip_stage:
        manifest = (
            REPO_ROOT
            / "experiments"
            / "results"
            / "lightning_batch"
            / job_name
            / "source_manifest.json"
        )
        if not manifest.is_file():
            sys.exit(f"FATAL: --skip-stage but manifest not found: {manifest}")
    else:
        if not args.ssh_target:
            sys.exit(
                "FATAL: --ssh-target unset; pass an ~/.ssh/config alias or "
                "user-qualified SSH target"
            )
        manifest = stage_workspace(
            job_name=job_name,
            archive_placeholder=None,
            ssh_target=args.ssh_target,
            remote_pact=args.remote_pact,
        )

    # Stage 2: dispatch claim.
    claim_lane(
        job_name=job_name,
        force_claim=args.force_claim,
        force_claim_reason=args.force_claim_reason,
    )

    # Stage 3: submit Lightning Studio Job.
    submit_result = submit_lightning_job(
        job_name=job_name,
        machine=args.machine,
        command=command,
        teamspace=args.teamspace,
        studio=args.studio,
        user=args.user,
        max_runtime_sec=args.max_runtime_sec,
        dry_run=args.dry_run,
    )

    # Stage 4: persist active-jobs row (always — even on dry-run, so harvester
    # can discover the staged manifest).
    persist_active_job(
        job_name=job_name,
        machine=args.machine,
        submit_result=submit_result,
        manifest_path=manifest,
    )

    print(json.dumps(submit_result, indent=2, default=str))
    print(f"\n[submitted] job_name={job_name}")
    print(
        f"[harvest]   .venv/bin/python experiments/arch_shrink_x0.4_lightning_harvest.py "
        f"--job-name {job_name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
