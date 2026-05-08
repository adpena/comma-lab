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
   follows the canonical Q-FAITHFUL pipeline from
   ``scripts/remote_lane_q_faithful_jointgen.sh`` because
   ``q_faithful_dilated_88k`` uses ``variant=quantizr_faithful`` (NOT in
   ``_VARIANTS_BUILD_RENDERER_FP4A_OK``):

   - bootstraps cu124 torch (``INFLATE_TORCH_SPEC=torch==2.5.1+cu124``)
   - builds half-frame masks.mkv seed (``build_baseline_archive.py``)
   - runs ``train_renderer.py --profile q_faithful_dilated_88k``
     **with ``--no-auth-eval-on-best`` and
     ``--qfaithful-training-poses experiments/results/lane_a_landed/optimized_poses.pt``**
   - manually exports JointFrameGenerator state_dict via
     ``tac.quantizr_faithful_export.save_qfai`` to ``renderer.bin``
   - assembles archive (renderer.bin + masks.mkv + optimized_poses.pt)
     with deterministic ZIP dating
   - runs ``contest_auth_eval.py --device cuda`` on the EXACT archive
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

from tac.deploy.lightning.defaults import (
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
    workspace mounted at ``remote_pact``.

    BUG-FIX 2026-05-08 (prior dispatch arch-shrink-x0-4-lightning-20260508T010514Z
    FAILED at startup):
      ``q_faithful_dilated_88k`` profile sets ``variant=quantizr_faithful`` which
      flows through ``build_quantizr_faithful_renderer()`` → JointFrameGenerator
      (NOT ``build_renderer()`` / AsymmetricPairGenerator). The
      ``--auth-eval-on-best`` early-fail gate in train_renderer.py
      (``_VARIANTS_BUILD_RENDERER_FP4A_OK``) correctly rejects this combination
      because the standard FP4A export path does not understand the
      JointFrameGenerator state_dict layout.

    Fix: follow the canonical Q-FAITHFUL pattern from
    ``scripts/remote_lane_q_faithful_jointgen.sh``:
      Stage 0: GPU presence + provenance + heartbeat.
      Stage 1: build masks.mkv seed (build_baseline_archive.py --half-frame
               matches the q_faithful_dilated_88k profile's
               mask_half_sim_prob=1.0).
      Stage 2: ``train_renderer.py --no-auth-eval-on-best
               --qfaithful-training-poses experiments/results/lane_a_landed/optimized_poses.pt``.
      Stage 3+4: export QFAI binary (``tac.quantizr_faithful_export.save_qfai``)
                 from the JointFrameGenerator state_dict.
      Stage 5: assemble archive (renderer.bin + masks.mkv + optimized_poses.pt)
               with deterministic ZIP dating.
      Stage 6: ``contest_auth_eval.py`` against the EXACT archive bytes,
               emitting ``[contest-CUDA]`` once RESULT_JSON parses.

    This still satisfies the CLAUDE.md ``Auth eval EVERYWHERE`` rule: the run
    ends with a CUDA auth eval whose RESULT_JSON is captured into
    ``contest_auth_eval.json``, the same artifact the harvester expects.
    """
    output_subdir = f"experiments/results/lightning_batch/{job_name}"
    auth_eval_dir = f"{output_subdir}/eval_work"
    auth_eval_log = f"{output_subdir}/auth_eval.log"
    auth_eval_result_json = f"{output_subdir}/contest_auth_eval.json"
    train_output_dir = f"{output_subdir}/train"
    train_tag = f"arch_shrink_x0_4_{job_name}"

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

# Required Q-FAITHFUL inputs (per scripts/remote_lane_q_faithful_jointgen.sh).
ANCHOR_LANE_A_POSES="experiments/results/lane_a_landed/optimized_poses.pt"
for f in "$ANCHOR_LANE_A_POSES" \\
         upstream/videos/0.mkv \\
         upstream/models/segnet.safetensors \\
         upstream/models/posenet.safetensors \\
         src/tac/quantizr_faithful_renderer.py \\
         src/tac/quantizr_faithful_export.py; do
    [ -f "$f" ] || {{ log "FATAL: missing required input: $f"; exit 1; }}
done

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
    'variant': 'quantizr_faithful',
    'target_elements': {TARGET_ELEMENTS},
    'inflate_torch_spec': '{INFLATE_TORCH_SPEC}',
    'fix_note': 'Q-FAITHFUL eval path; --no-auth-eval-on-best + manual QFAI export + separate contest_auth_eval',
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

# Stage 1: build masks.mkv seed via build_baseline_archive.py --half-frame
# (q_faithful_dilated_88k profile uses mask_half_sim_prob=1.0; the renderer
# is half-frame-aware so the mask seed MUST be half-frame to avoid the
# score-17.55 catastrophe per feedback_half_frame_breaks_posenet).
log "=== Stage 1: build half-frame masks.mkv seed ==="
"$PYBIN" -u experiments/build_baseline_archive.py \\
    --device cuda --crf 50 --half-frame \\
    --output "$LOG_DIR/archive_masks_seed.zip" 2>&1 | tee "$LOG_DIR/build_masks.log" | tail -5
PIPE_RC=("${{PIPESTATUS[@]}}")
if [ "${{PIPE_RC[0]}}" -ne 0 ]; then
    log "FATAL: build_baseline_archive failed rc=${{PIPE_RC[0]}}"
    exit "${{PIPE_RC[0]}}"
fi
mkdir -p "$LOG_DIR/extracted"
( cd "$LOG_DIR/extracted" && unzip -o "$LOG_DIR/archive_masks_seed.zip" 2>&1 | tail -3 )
[ -f "$LOG_DIR/extracted/masks.mkv" ] || {{ log "FATAL: masks.mkv extract failed"; exit 2; }}
log "  masks.mkv extracted ($(stat -c '%s' "$LOG_DIR/extracted/masks.mkv" 2>/dev/null || stat -f '%z' "$LOG_DIR/extracted/masks.mkv") bytes)"

# Stage 2: training (Q-FAITHFUL JointFrameGenerator path).
# --no-auth-eval-on-best is REQUIRED — variant=quantizr_faithful is not in
# _VARIANTS_BUILD_RENDERER_FP4A_OK; the FP4A export path can't serialise the
# JointFrameGenerator state_dict. We do the eval manually after Stage 5.
# --qfaithful-training-poses is REQUIRED — JointFrameGenerator's FiLM head
# learns pose -> frame mapping; it needs scorer-measured Lane A poses.
log "=== Stage 2: train_renderer.py --profile {PROFILE} (--no-auth-eval-on-best) ==="
mkdir -p "$WORKSPACE/{train_output_dir}"
"$PYBIN" -u src/tac/experiments/train_renderer.py \\
    --profile {PROFILE} \\
    --device cuda \\
    --seed 1234 \\
    --tag {train_tag} \\
    --qfaithful-training-poses "$ANCHOR_LANE_A_POSES" \\
    --no-auth-eval-on-best \\
    --output-dir "$WORKSPACE/{train_output_dir}" 2>&1 | tee "$LOG_DIR/train.log" | tail -50
PIPE_RC=("${{PIPESTATUS[@]}}")
if [ "${{PIPE_RC[0]}}" -ne 0 ]; then
    log "FATAL: train_renderer.py rc=${{PIPE_RC[0]}}"
    exit "${{PIPE_RC[0]}}"
fi

BEST_CKPT=$(ls -t "$WORKSPACE/{train_output_dir}"/*BEST*.pt 2>/dev/null | head -1)
if [ -z "$BEST_CKPT" ]; then
    BEST_CKPT=$(ls -t "$WORKSPACE/{train_output_dir}"/*.pt 2>/dev/null | head -1)
fi
[ -f "$BEST_CKPT" ] || {{ log "FATAL: train_renderer didn't produce any .pt checkpoint"; exit 3; }}
log "  best checkpoint: $BEST_CKPT ($(stat -c '%s' "$BEST_CKPT" 2>/dev/null || stat -f '%z' "$BEST_CKPT") bytes)"

# Stage 3+4: export the JointFrameGenerator state_dict to the QFAI binary.
# QFAI format = [b"QFAI"][header_len][JSON header][torch.save(state_dict)].
# The contest inflate_renderer.py dispatches QFAI/QZS3 by file magic.
log "=== Stage 3+4: export QFAI binary from JointFrameGenerator state_dict ==="
"$PYBIN" -u -c "
import sys
sys.path.insert(0, 'src')
import torch, brotli
from pathlib import Path
from tac.quantizr_faithful_renderer import build_quantizr_faithful_renderer
from tac.quantizr_faithful_export import save_qfai

ckpt = torch.load('$BEST_CKPT', map_location='cpu', weights_only=False)
sd_raw = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
sd = {{}}
for k, v in sd_raw.items():
    if k.startswith('gen.'):
        sd[k[len('gen.'):]] = v
    else:
        sd[k] = v
gen = build_quantizr_faithful_renderer()
gen.load_state_dict(sd, strict=True)
gen.eval()
n_params = sum(p.numel() for p in gen.parameters())
print(f'JointFrameGenerator loaded: {{n_params:,}} params')

# Promotable training_pose_contract is required for save_qfai (per
# scripts/remote_lane_q_faithful_jointgen.sh:317-318).
training_pose_contract = None
for key in ('qfaithful_training_pose_contract', 'training_pose_contract'):
    value = ckpt.get(key)
    if isinstance(value, dict):
        training_pose_contract = value
        break
if training_pose_contract is None:
    meta = ckpt.get('__meta__') or ckpt.get('arch_meta') or {{}}
    if isinstance(meta, dict):
        for key in ('qfaithful_training_pose_contract', 'training_pose_contract'):
            value = meta.get(key)
            if isinstance(value, dict):
                training_pose_contract = value
                break
if not isinstance(training_pose_contract, dict) or training_pose_contract.get('training_pose_contract_promotable') is not True:
    raise SystemExit('FATAL: checkpoint missing promotable Q-FAITHFUL training_pose_contract')

qfai_path = Path('$WORKSPACE/{train_output_dir}/renderer.bin')
n_bytes = save_qfai(gen, qfai_path, extra_meta={{'training_pose_contract': training_pose_contract}})
print(f'QFAI raw renderer.bin: {{n_bytes:,}} bytes')

raw = qfai_path.read_bytes()
br = brotli.compress(raw, quality=11)
br_path = Path('$WORKSPACE/{train_output_dir}/renderer.qfai.bin.br')
br_path.write_bytes(br)
print(f'QFAI brotli sidecar q=11: {{len(br):,}} bytes ({{100*len(br)/len(raw):.1f}}% of raw)')
"
EXPORT_BIN="$WORKSPACE/{train_output_dir}/renderer.bin"
[ -f "$EXPORT_BIN" ] || {{ log "FATAL: QFAI export failed"; exit 4; }}
log "  renderer.bin: $(stat -c '%s' "$EXPORT_BIN" 2>/dev/null || stat -f '%z' "$EXPORT_BIN") bytes"

# Stage 5: assemble archive (renderer.bin + masks.mkv + Lane A poses).
# Deterministic ZIP per Codex R5-r6 #5 (check_archive_builders_use_deterministic_zip).
log "=== Stage 5: build Q-FAITHFUL archive ==="
mkdir -p "$LOG_DIR/iter_0"
cp "$EXPORT_BIN" "$LOG_DIR/iter_0/renderer.bin"
cp "$LOG_DIR/extracted/masks.mkv" "$LOG_DIR/iter_0/masks.mkv"
cp "$ANCHOR_LANE_A_POSES" "$LOG_DIR/iter_0/optimized_poses.pt"
find "$LOG_DIR/iter_0" -name '._*' -delete 2>/dev/null || true
find "$LOG_DIR/iter_0" -name '.DS_Store' -delete 2>/dev/null || true

ARCHIVE="$WORKSPACE/{output_subdir}/archive.zip"
"$PYBIN" -c "
import zipfile, os
src = '$LOG_DIR/iter_0'
dst = '$ARCHIVE'
det_dt = (1980, 1, 1, 0, 0, 0)
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for n in ('renderer.bin', 'masks.mkv', 'optimized_poses.pt'):
        p = os.path.join(src, n)
        assert os.path.isfile(p), f'missing {{p}}'
        info = zipfile.ZipInfo(filename=n, date_time=det_dt)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        info.create_system = 3
        with open(p, 'rb') as f:
            z.writestr(info, f.read(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
size = os.path.getsize(dst)
print(f'archive {{dst}}: {{size}} bytes')
assert 100_000 < size < 1_500_000, f'archive size {{size}} outside sane band [100K, 1.5M]'
"
ARCHIVE_BYTES=$(stat -c '%s' "$ARCHIVE" 2>/dev/null || stat -f '%z' "$ARCHIVE")
log "  archive: $ARCHIVE bytes=$ARCHIVE_BYTES"

# Stage 6: contest_auth_eval [contest-CUDA] on the EXACT archive bytes.
# BUG-FIX 2026-05-08 (companion to lossy_coarsening fix): use the canonical
# hash-pinned DALI bootstrap helper which auto-detects pipless venvs and
# runs `python -m ensurepip --upgrade` before any `pip install`. CLAUDE.md
# memory `feedback_remote_archive_only_eval_self_bootstraps_all_deps_20260501`
# pins this as the canonical pattern; do NOT copy-paste install commands inline.
log "=== Stage 6a: hash-pinned DALI bootstrap ==="
"$PYBIN" scripts/bootstrap_dali_hash_pinned.py \\
    --json-out "$LOG_DIR/lightning_dali_bootstrap.json" \\
    --requirements-out "$LOG_DIR/lightning_dali_requirements.txt" \\
    --timeout 900

log "=== Stage 6b: contest_auth_eval [contest-CUDA] ==="
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
data['variant'] = 'quantizr_faithful'
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

    BUG-FIX 2026-05-08 (companion to train_renderer ego_flow+pose forward fix):
      ``experiments/results/lane_a_landed/optimized_poses.pt`` is the
      load-bearing pose stream the Q-FAITHFUL training requires
      (``--qfaithful-training-poses``). However ``experiments/results/`` is in
      ``lightning_repro_workspace.EXCLUDED_PREFIXES`` (line 88) so the rsync
      source-traversal silently drops it. Without an explicit ``--artifact``
      override the file never lands on Lightning, ``_load_qfaithful_training_poses``
      raises ``FileNotFoundError``, and the dispatch wastes a round-trip.
      Stage the pose file as an artifact (``include_excluded=True``) so it
      survives EXCLUDED_PREFIXES.
    """
    manifest_dir = REPO_ROOT / "experiments" / "results" / "lightning_batch" / job_name
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = manifest_dir / "source_manifest.json"
    pose_artifact = REPO_ROOT / "experiments" / "results" / "lane_a_landed" / "optimized_poses.pt"
    if not pose_artifact.is_file():
        sys.exit(
            f"FATAL: required Q-FAITHFUL training pose artifact missing: {pose_artifact}\n"
            f"  This is the load-bearing pose stream cited in CLAUDE.md "
            f"`project_baseline_poses_load_bearing` and required by the "
            f"`scripts/remote_lane_q_faithful_jointgen.sh` canonical pattern."
        )
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
        # Force-stage the load-bearing Q-FAITHFUL pose artifact past
        # EXCLUDED_PREFIXES (`experiments/results/`).
        "--artifact",
        str(pose_artifact),
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
        print("--- remote command preview (first 800 chars) ---")
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
