#!/usr/bin/env bash
# ddm_v4c STAGED realized gate — task #776 / QA59+QA60+QA62.  DO NOT self-fire.
# MAIN fires ONE candidate at a time when the n600 scorer slot is idle.
#
# WHAT IT MEASURES: the exact composed row (SegNet d_seg + PoseNet d_pose + rate)
# of the v4c byte-closed archive (cell_drop50 base tokens + RE-SOLVED-through-
# static two-plane pose field + per-pair photometric (a,b) + lossless trio),
# apples-to-apples with the MEASURED pfs1 D1 reference row S=2.256641 and the
# measured v4b gate S=1.534258 (same eval_root, same 0.mkv, same device).
# Axis: [macOS-CPU advisory - real evaluator, real bytes]; contest-CPU authority
# is a later Modal flight (operator-GO). Pointer 0.1910828242 UNMOVED.
#
# MECHANISM (no upstream edits): surgical archive.zip swap into a COPY of the
# pfs1 D1 submission dir, with the v4c receiver (inflate_runner_v4c.py) +
# vendored decode deps, then the stock eval_root/evaluate.sh.
#
# PATH EXPORT is REQUIRED (the bare-python death): evaluate.sh + inflate.sh call
# bare `python`; without the venv on PATH they die "command not found".
#
# Usage:  bash experiments/stage_v4c_realized_gate.sh [device cpu|cuda] [tag]
set -euo pipefail

export PATH="/Users/adpena/Projects/pact/.venv/bin:$PATH"

DEVICE="${1:-cpu}"
TAG="${2:-static_photo_celldrop50}"
V4C_DIR="/Volumes/VertigoDataTier/pact/ddm_v4c_20260730"
EVAL_ROOT="/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/eval_root"
TEMPLATE_SUB="${EVAL_ROOT}/submissions/pfs1"
ARCHIVE="${V4C_DIR}/v4c_composed_${TAG}_archive.zip"
RECEIVER="${V4C_DIR}/inflate_runner_v4c.py"

[ -f "$ARCHIVE" ] || { echo "missing byte-closed archive: $ARCHIVE" >&2; exit 1; }
[ -f "$RECEIVER" ] || { echo "missing v4c receiver: $RECEIVER" >&2; exit 1; }
[ -d "$TEMPLATE_SUB" ] || { echo "missing pfs1 template sub: $TEMPLATE_SUB" >&2; exit 1; }

RUN_SUB="${EVAL_ROOT}/submissions/v4c_${TAG}"
rm -rf "$RUN_SUB"
mkdir -p "$RUN_SUB"
# stock inflate.sh (calls inflate_runner.py) + vendored decode deps; the v4c
# receiver replaces inflate_runner.py (single/two-plane static compose + photo).
for f in inflate.sh pfs1_warp_receiver.py ddm_r7_token_coder.py \
         ddm_tr1_runtime.py repair_entropy_coder_runtime_adapters.py; do
  cp "${TEMPLATE_SUB}/${f}" "${RUN_SUB}/${f}"
done
cp "$RECEIVER" "${RUN_SUB}/inflate_runner.py"
cp "$ARCHIVE" "${RUN_SUB}/archive.zip"

echo "[v4c gate] archive=$(basename "$ARCHIVE") device=${DEVICE}"
echo "[v4c gate] archive.zip bytes: $(stat -f%z "${RUN_SUB}/archive.zip" 2>/dev/null || stat -c%s "${RUN_SUB}/archive.zip")"
echo "[v4c gate] receiver=inflate_runner_v4c (frame0_policy=warp_two_plane_static_photo_v4c)"

time bash "${EVAL_ROOT}/evaluate.sh" \
  --submission-dir "$RUN_SUB" \
  --video-names-file "${EVAL_ROOT}/public_test_video_names.txt" \
  --device "$DEVICE"

echo "[v4c gate] report:"
cat "${RUN_SUB}/report.txt"
echo "[v4c gate] receipt schema: ddm_v4c_realized_gate.v1"
echo "  REF pfs1 D1: d_seg 0.00389011 d_pose 0.22144216 S 2.256641"
echo "  v4b MEASURED: d_seg 0.00553676 d_pose 0.06365131 S 1.534258 (Knee-A base)"
echo "  v4c base=cell_drop50 (seg+rate -0.066 vs Knee-A); see build receipt for"
echo "    predicted decomposition. ACCEPT/verify: realized d_seg ~0.00431"
echo "    (cell_drop50 tokens; gr1 n600 evaluate band +-2.8e-5) ;"
echo "    realized d_pose per the v4c solve+photo advisory ;"
echo "    realized S << v4b 1.5343."
