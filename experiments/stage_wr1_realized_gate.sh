#!/usr/bin/env bash
# ddm_wr1 STAGED realized gate — task #766 / QA06.  DO NOT self-fire.
# MAIN fires ONE candidate at a time when the n600 scorer slot is idle
# (sb1 owns the current slot; QA43 pose tail is queued ahead of this).
#
# WHAT IT MEASURES: the exact composed row (SegNet d_seg + PoseNet d_pose +
# rate) of a wr1 byte-closed archive, apples-to-apples with the MEASURED
# pfs1 D1 reference row S=2.256641 (same eval_root, same 0.mkv, same device).
# Axis: [macOS-CPU advisory - real evaluator, real bytes]; the contest-CPU
# authority is the later Modal flight (operator-GO). Pointer UNMOVED.
#
# MECHANISM (no source edits): the pfs1 inflate_runner does NOT enforce
# manifest tokens_sha256, so the gate is a surgical archive.zip swap into a
# COPY of the pfs1 D1 submission dir, then the stock eval_root/evaluate.sh.
#
# Usage:
#   bash experiments/stage_wr1_realized_gate.sh \
#       <candidate: kneeA|kneeB> [device cpu|cuda]
#
# Projected wall: ~17 min/candidate on macOS-CPU (pfs1 D1 was 1006 s, full
# n600 inflate+SegNet+PoseNet). Well inside the 30-min decode budget.
set -euo pipefail

CAND="${1:-kneeA}"
DEVICE="${2:-cpu}"
WR1_DIR="/Volumes/VertigoDataTier/pact/ddm_wr1_20260729"
EVAL_ROOT="/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/eval_root"
TEMPLATE_SUB="${EVAL_ROOT}/submissions/pfs1"

case "$CAND" in
  kneeA) ARCHIVE="${WR1_DIR}/wr1_kneeA_safe_274k_archive.zip" ;;
  kneeB) ARCHIVE="${WR1_DIR}/wr1_kneeB_subbar_173k_archive.zip" ;;
  *) echo "unknown candidate: $CAND (kneeA|kneeB)" >&2; exit 2 ;;
esac

[ -f "$ARCHIVE" ] || { echo "missing byte-closed archive: $ARCHIVE" >&2; exit 1; }
[ -d "$TEMPLATE_SUB" ] || { echo "missing pfs1 template sub: $TEMPLATE_SUB" >&2; exit 1; }

RUN_SUB="${EVAL_ROOT}/submissions/wr1_${CAND}"
rm -rf "$RUN_SUB"
# copy decode deps (inflate.sh, inflate_runner.py, vendored coders/receiver);
# skip the template's own archive/inflated scratch.
mkdir -p "$RUN_SUB"
for f in inflate.sh inflate_runner.py pfs1_warp_receiver.py \
         ddm_r7_token_coder.py ddm_tr1_runtime.py \
         repair_entropy_coder_runtime_adapters.py; do
  cp "${TEMPLATE_SUB}/${f}" "${RUN_SUB}/${f}"
done
cp "$ARCHIVE" "${RUN_SUB}/archive.zip"

echo "[wr1 gate] candidate=${CAND} archive=$(basename "$ARCHIVE") device=${DEVICE}"
echo "[wr1 gate] archive.zip bytes: $(stat -f%z "${RUN_SUB}/archive.zip" 2>/dev/null || stat -c%s "${RUN_SUB}/archive.zip")"

time bash "${EVAL_ROOT}/evaluate.sh" \
  --submission-dir "$RUN_SUB" \
  --video-names-file "${EVAL_ROOT}/public_test_video_names.txt" \
  --device "$DEVICE"

echo "[wr1 gate] report:"
cat "${RUN_SUB}/report.txt"
echo "[wr1 gate] parse report.txt -> ${WR1_DIR}/wr1_${CAND}_realized_gate_receipt.json"
echo "  schema: ddm_wr1_realized_gate.v1"
echo "  compare d_seg vs REF 0.00389011 ; d_pose vs REF 0.22144216 (pose-safety) ;"
echo "  final S vs REF 2.256641 ; break-even: accept iff realized dS < 25*dB/37545489."
