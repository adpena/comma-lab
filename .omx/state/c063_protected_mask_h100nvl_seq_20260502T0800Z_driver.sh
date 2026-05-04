#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace/pact}"
cd "$WORKSPACE"
export PYTHONPATH="$WORKSPACE/src:$WORKSPACE/upstream:$WORKSPACE:${PYTHONPATH:-}"

ROOT="experiments/results/vast_live_harvest/c063_protected_mask_h100nvl_seq_20260502T0800Z"
mkdir -p "$ROOT"
log() { echo "[c063-protected-mask-h100nvl] $(date -u +%FT%TZ) $*" | tee -a "$ROOT/driver.log"; }

cat > "$ROOT/source_sha256s.expected" <<'SHA'
submissions/robust_current/inflate.sh=86449a1f52ac6b2be120d47287b8410f915dce7e562c69f480103f6e527c6017
submissions/robust_current/inflate_renderer.py=1bf64e9f055c88438c854d1e09f048c07c359177494da8e636079c62706b6472
submissions/robust_current/unpack_renderer_payload.py=cac8cde654f2d875d4567c18b77d573af91c29dbb0b05b7934dc7e019ae66f49
experiments/contest_auth_eval.py=8d9dd3e1e9f97245623c938aa9b28a41fdd3e41752208b16b218131906f7ef53
scripts/remote_archive_only_eval.sh=8d1069de4f2426108beb44e519f83428608c1354ea2752b426cddd1f5ab815f8
SHA

run_one() {
  local label="$1"
  local archive="$2"
  local low="$3"
  local high="$4"
  test -f "$archive"
  log "running $label archive=$(sha256sum "$archive" | cut -d' ' -f1) bytes=$(stat -c '%s' "$archive")"
  export ARCHIVE_PATH="$WORKSPACE/$archive"
  export ARCHIVE_LABEL="$label"
  export LOG_DIR="$WORKSPACE/$ROOT/$label"
  export PREDICTED_LOW="$low"
  export PREDICTED_HIGH="$high"
  export CONTROLLED_BASELINE="C-063 protected/foveated mask reencode diagnostic; exact H100 screen only, T4 required for promotion"
  export REQUIRED_SOURCE_SHA256S="$(cat "$ROOT/source_sha256s.expected")"
  bash scripts/remote_archive_only_eval.sh
}

run_one \
  "archive_eval_c063_protected_boundary1_horizon_pose_top16_crf56_h100_20260502" \
  "$ROOT/protect_boundary1_horizon_pose_top16_crf56_rawzip.final_archive.zip" \
  "0.18" "2.50"

run_one \
  "archive_eval_c063_protected_boundary2_horizon_pose_top32_crf60_h100_20260502" \
  "$ROOT/protect_boundary2_horizon_pose_top32_crf60_rawzip.final_archive.zip" \
  "0.18" "2.80"

log "done"
