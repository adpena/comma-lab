"""Auto-generate ``scripts/remote_lane_substrate_<id>.sh`` from a contract.

Pure function: ``generate_driver_shell(contract) -> str`` returns the full
remote driver shell script bytes. The output honors the canonical
``c6_e4_mdl_ibps`` template invariants:

  - ``set -euo pipefail`` (per Catalog #2).
  - ``REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1`` sentinel for the canonical
    bootstrap source (per Catalog #163).
  - 5-min heartbeat (per CLAUDE.md "Remote code parity — non-negotiable").
  - Dispatch-claim verification + terminal-status append (per CLAUDE.md
    "CROSS-AGENT DISPATCH COORDINATION").
  - Modal-results path remap when ``MODAL_RUNTIME=1`` (per Catalog #204).
  - Auth-eval validation gate before emitting ``[contest-CUDA]`` completion
    marker (per CLAUDE.md "Auth eval EVERYWHERE").
  - 3-arg archive-grammar handoff (per Catalog #146).

Generators are PURE FUNCTIONS — same contract → same bytes. The migration
subagent diffs ``generate_driver_shell(extracted_contract)`` against the
existing remote driver to surface drift.
"""

from __future__ import annotations

from tac.substrate_registry.contract import SubstrateContract

__all__ = [
    "generate_driver_shell",
    "default_driver_relpath",
]


def default_driver_relpath(contract: SubstrateContract) -> str:
    """Canonical driver path under ``scripts/``."""
    return f"scripts/remote_lane_substrate_{contract.id}.sh"


def _env_var_for(substrate_id: str, suffix: str) -> str:
    """Convert (id, suffix) → ENV_VAR token, e.g. (c6_e4_mdl_ibps, VIDEO_PATH) → C6_E4_MDL_IBPS_VIDEO_PATH."""
    return f"{substrate_id.upper()}_{suffix}"


def generate_driver_shell(contract: SubstrateContract) -> str:
    """Emit canonical remote driver shell script bytes for ``contract``.

    Output is deterministic: same contract → same bytes (no clock, no host).
    The generated driver is reviewable in 30 seconds (~150 LOC, single
    purpose, no hidden state).
    """
    sid = contract.id
    sid_upper = sid.upper()
    lane_id = contract.lane_id
    tag = f"substrate_{sid}"
    log_dir_var = f"$WORKSPACE/lane_substrate_{sid}_results"

    e_video = _env_var_for(sid, "VIDEO_PATH")
    e_output = _env_var_for(sid, "OUTPUT_DIR")
    e_epochs = _env_var_for(sid, "EPOCHS")
    e_device = _env_var_for(sid, "DEVICE")
    e_dispatch = _env_var_for(sid, "DISPATCH_INSTANCE_JOB_ID")
    e_claims_path = _env_var_for(sid, "DISPATCH_CLAIMS_PATH")

    return f"""#!/bin/bash
# Remote lane script: substrate {sid} (auto-generated).
#
# Trainer: experiments/train_substrate_{sid}.py
# Lane: {lane_id}
# Recipe: .omx/operator_authorize_recipes/substrate_{sid}_{contract.cost_band_platform_key}_{contract.cost_band_gpu_key.lower()}_dispatch.yaml
#
# AUTO-GENERATED FROM SubstrateContract via
# src/tac/substrate_registry/driver_generator.py. DO NOT EDIT BY HAND --
# regenerate via the driver_generator if the contract changes. Catalog #242
# (check_register_substrate_contract_fields_canonical) catches drift at
# preflight time.
#
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline" this
# script DELEGATES bootstrap to scripts/remote_archive_only_eval.sh's
# bootstrap_runtime_deps() function (Catalog #163 sentinel
# REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 prepended).
#
# Heartbeat: every 5 min per CLAUDE.md "Remote code parity".
# Auth-eval gate per CLAUDE.md "Auth eval EVERYWHERE".

set -euo pipefail

WORKSPACE="${{WORKSPACE:-/workspace/pact}}"
PYBIN="${{PYBIN:-}}"
LANE_ID="{lane_id}"
TAG="${{TAG:-{tag}}}"
LOG_DIR="${{LOG_DIR:-{log_dir_var}}}"
OUTPUT_DIR="${{OUTPUT_DIR:-$LOG_DIR/output}}"

# Trainer-flag env-var ladder (Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS).
{e_video}="${{{e_video}:-$WORKSPACE/upstream/videos/0.mkv}}"
{e_output}="${{{e_output}:-$OUTPUT_DIR}}"
{e_epochs}="${{{e_epochs}:-{contract.cost_band_epochs}}}"
{e_device}="${{{e_device}:-cuda}}"

DISPATCH_INSTANCE_JOB_ID="${{{e_dispatch}:-${{DISPATCH_INSTANCE_JOB_ID:-}}}}"
DISPATCH_CLAIMS_PATH="${{{e_claims_path}:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}}"
DISPATCH_PLATFORM="${{DISPATCH_PLATFORM:-{contract.cost_band_platform_key}}}"
HEARTBEAT_PID=""
CLAIM_VERIFIED=0

log() {{ echo "[lane-{sid}] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }}

# Stage 0: dispatch claim verification (CLAUDE.md "CROSS-AGENT DISPATCH COORDINATION").
if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
    log "FATAL: {e_dispatch} or DISPATCH_INSTANCE_JOB_ID is required for active lane-claim verification"
    exit 21
fi

# Catalog #204: Modal-results path remap when MODAL_RUNTIME=1.
if [ "${{MODAL_RUNTIME:-0}}" = "1" ] && [ -d "/modal_results" ]; then
    case "${e_output}" in
        "$WORKSPACE"/*|/tmp/*|/workspace/*)
            LOG_DIR="/modal_results/${{DISPATCH_INSTANCE_JOB_ID}}"
            OUTPUT_DIR="$LOG_DIR/output"
            {e_output}="$OUTPUT_DIR"
            ;;
    esac
fi
PROVENANCE="$LOG_DIR/provenance.json"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

CLAIM_PYTHON="${{PYBIN:-}}"
if [ -z "$CLAIM_PYTHON" ] && [ -x "$WORKSPACE/.venv/bin/python" ]; then
    CLAIM_PYTHON="$WORKSPACE/.venv/bin/python"
fi
if [ -z "$CLAIM_PYTHON" ]; then
    CLAIM_PYTHON="python3"
fi

append_terminal_claim() {{
    local rc="$1"
    if [ ! -f "$WORKSPACE/tools/claim_lane_dispatch.py" ]; then
        log "WARN: claim helper missing; cannot append terminal dispatch claim"
        return 0
    fi
    local status
    if [ "$rc" -eq 0 ]; then
        status="completed_{sid}_remote_driver"
    elif [ "${{CLAIM_VERIFIED:-0}}" != "1" ]; then
        status="failed_{sid}_claim_verification_rc_${{rc}}"
    else
        status="failed_{sid}_remote_driver_rc_${{rc}}"
    fi
    "$CLAIM_PYTHON" "$WORKSPACE/tools/claim_lane_dispatch.py" claim \\
        --claims-path "$DISPATCH_CLAIMS_PATH" \\
        --force \\
        --lane-id "$LANE_ID" \\
        --platform "$DISPATCH_PLATFORM" \\
        --instance-job-id "$DISPATCH_INSTANCE_JOB_ID" \\
        --agent "remote_lane_substrate_{sid}" \\
        --status "$status" \\
        --notes "remote_driver_terminal rc=$rc output_dir=${e_output}" \\
        >> "$LOG_DIR/run.log" 2>&1 || {{
        log "WARN: failed to append terminal dispatch claim status=$status"
    }}
}}

cleanup() {{
    local rc="$?"
    if [ -n "$HEARTBEAT_PID" ]; then
        kill "$HEARTBEAT_PID" 2>/dev/null || true
    fi
    append_terminal_claim "$rc"
    exit "$rc"
}}
trap cleanup EXIT

# Stage 1: bootstrap remote runtime deps via canonical sourced helper
# (Catalog #163 sentinel + Catalog #189 path-resolution invariants).
if [ -f "$WORKSPACE/scripts/remote_archive_only_eval.sh" ]; then
    log "stage_1_bootstrap_via_canonical_sourced_helper"
    # shellcheck disable=SC1091
    REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 source "$WORKSPACE/scripts/remote_archive_only_eval.sh"
    bootstrap_runtime_deps || {{
        log "FATAL: bootstrap_runtime_deps failed; refusing dispatch"
        exit 22
    }}
else
    log "WARN: canonical bootstrap script missing; assuming runtime deps present"
fi

# Stage 2: heartbeat (every 5 min per CLAUDE.md "Remote code parity").
(
    while true; do
        date -u +%FT%TZ > "$LOG_DIR/heartbeat.log"
        sleep 300
    done
) &
HEARTBEAT_PID=$!

# Stage 3: write provenance.
cat > "$PROVENANCE" <<EOF
{{
  "lane_id": "$LANE_ID",
  "tag": "$TAG",
  "trainer": "experiments/train_substrate_{sid}.py",
  "recipe": ".omx/operator_authorize_recipes/substrate_{sid}_{contract.cost_band_platform_key}_{contract.cost_band_gpu_key.lower()}_dispatch.yaml",
  "video_path": "${e_video}",
  "output_dir": "${e_output}",
  "epochs": "${e_epochs}",
  "device": "${e_device}",
  "dispatch_instance_job_id": "$DISPATCH_INSTANCE_JOB_ID",
  "started_at_utc": "$(date -u +%FT%TZ)"
}}
EOF

# Stage 4: invoke trainer.
log "stage_4_trainer_begin epochs=${e_epochs}"
TRAINER_PY="$WORKSPACE/experiments/train_substrate_{sid}.py"
if [ ! -f "$TRAINER_PY" ]; then
    log "FATAL: trainer missing at $TRAINER_PY"
    exit 23
fi

PYBIN_RESOLVED="$CLAIM_PYTHON"

"$PYBIN_RESOLVED" "$TRAINER_PY" \\
    --video-path "${e_video}" \\
    --output-dir "${e_output}" \\
    --epochs "${e_epochs}" \\
    --device "${e_device}" \\
    2>&1 | tee -a "$LOG_DIR/trainer.log"

# Auth-eval validation per CLAUDE.md "Auth eval EVERYWHERE" + Catalog #226.
"$PYBIN_RESOLVED" - "${e_output}/stats.json" <<'PY'
import json
import sys
from pathlib import Path

stats_path = Path(sys.argv[1])
if not stats_path.is_file():
    raise SystemExit(f"missing {sid} stats.json: {{stats_path}}")
stats = json.loads(stats_path.read_text(encoding="utf-8"))
if stats.get("auth_eval_score_claim_valid") is not True:
    raise SystemExit(
        "{sid} stats missing valid auth_eval_score_claim_valid=true; "
        f"blockers={{stats.get('result_review_blockers')!r}}"
    )
if stats.get("auth_eval_score_axis") != "contest_cuda":
    raise SystemExit(
        f"{sid} stats not on contest_cuda axis: {{stats.get('auth_eval_score_axis')!r}}"
    )
if stats.get("auth_eval_exact_cuda_complete") is not True:
    raise SystemExit("{sid} stats missing auth_eval_exact_cuda_complete=true")
print(
    "{sid_upper}_AUTH_EVAL_VALIDATED "
    f"score={{stats.get('auth_eval_score')}} "
    f"path={{stats.get('auth_eval_result_path')}}"
)
PY

# Stage 5: emit completion marker (operator + autopilot consume).
log "LANE_{sid_upper}_DONE [contest-CUDA] output_dir=${e_output}"
cat >> "$LOG_DIR/completion.log" <<EOF
LANE_{sid_upper}_DONE [contest-CUDA] $LANE_ID $(date -u +%FT%TZ)
EOF
"""
