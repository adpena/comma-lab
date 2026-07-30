#!/usr/bin/env bash
# ddm_ck1 STAGED composed realized gate — QA06 composed decision cell.  DO NOT
# self-fire.  MAIN fires ONE candidate when the n600 scorer slot is idle.
#
# WHAT IT MEASURES: the exact composed row (SegNet d_seg + PoseNet d_pose + rate)
# of the COMPOSED candidate = Knee-A byte-closed token base + the pose field
# RE-SOLVED on the Knee-A base (grammar v4a, single-plane 6-DOF, s_r=1.0),
# apples-to-apples with the MEASURED pfs1 D1 reference row S=2.256641 and the
# REJECTED Knee-A-standalone row S=2.4097 (same eval_root, same 0.mkv, same dev).
# Axis: [macOS-CPU advisory - real evaluator, real bytes]; contest-CPU authority
# is a later Modal flight (operator-GO). Pointer 0.1910828242 UNMOVED.
#
# PRECONDITION: the composed archive + v4a inflate_runner must be built first:
#   .venv/bin/python experiments/ddm_ck1_build_composed_archive.py \
#       --solve-jsonl /Volumes/VertigoDataTier/pact/ddm_ck1_20260729/ck1_solve.partial.jsonl \
#       --pose-key p_single_kneeA --tag single_kneeA
# (that writes ck1_composed_single_kneeA_archive.zip + inflate_runner_v4a.py)
#
# MECHANISM: surgical submission-dir copy of the pfs1 D1 template, then SWAP
# (a) archive.zip -> the composed archive, (b) inflate_runner.py -> the v4a runner
# (s_r=1.0 warp of the re-solved pose). All other decode deps are the pfs1 bytes.
#
# Usage:  bash experiments/stage_ck1_composed_gate.sh <tag: single_kneeA|...> [cpu|cuda]
# Projected wall: ~17 min/candidate on macOS-CPU (full n600 inflate+SegNet+PoseNet).
set -euo pipefail

TAG="${1:-single_kneeA}"
DEVICE="${2:-cpu}"
CK1_DIR="/Volumes/VertigoDataTier/pact/ddm_ck1_20260729"
EVAL_ROOT="/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/eval_root"
TEMPLATE_SUB="${EVAL_ROOT}/submissions/pfs1"
ARCHIVE="${CK1_DIR}/ck1_composed_${TAG}_archive.zip"
V4A_RUNNER="${CK1_DIR}/inflate_runner_v4a.py"

[ -f "$ARCHIVE" ] || { echo "missing composed archive: $ARCHIVE (build it first)" >&2; exit 1; }
[ -f "$V4A_RUNNER" ] || { echo "missing v4a runner: $V4A_RUNNER (build it first)" >&2; exit 1; }
[ -d "$TEMPLATE_SUB" ] || { echo "missing pfs1 template sub: $TEMPLATE_SUB" >&2; exit 1; }

RUN_SUB="${EVAL_ROOT}/submissions/ck1_${TAG}"
rm -rf "$RUN_SUB"
mkdir -p "$RUN_SUB"
# decode deps: the pfs1 receiver + coders + inflate.sh (verbatim); the runner is SWAPPED.
for f in inflate.sh pfs1_warp_receiver.py ddm_r7_token_coder.py ddm_tr1_runtime.py \
         repair_entropy_coder_runtime_adapters.py; do
  cp "${TEMPLATE_SUB}/${f}" "${RUN_SUB}/${f}"
done
cp "$V4A_RUNNER" "${RUN_SUB}/inflate_runner.py"   # v4a (s_r=1.0) replaces s_r=0
cp "$ARCHIVE" "${RUN_SUB}/archive.zip"

echo "[ck1 gate] tag=${TAG} archive=$(basename "$ARCHIVE") device=${DEVICE}"
echo "[ck1 gate] archive.zip bytes: $(stat -f%z "${RUN_SUB}/archive.zip" 2>/dev/null || stat -c%s "${RUN_SUB}/archive.zip")"

time bash "${EVAL_ROOT}/evaluate.sh" \
  --submission-dir "$RUN_SUB" \
  --video-names-file "${EVAL_ROOT}/public_test_video_names.txt" \
  --device "$DEVICE"

echo "[ck1 gate] report:"
cat "${RUN_SUB}/report.txt"
echo "[ck1 gate] REF pfs1 D1: S 2.256641 (seg 0.00389011, pose 0.22144216, rate 0.379537)"
echo "[ck1 gate] Knee-A standalone (REJECT): S 2.4097 (seg 0.00553676, pose 0.28002128, rate 0.182667)"
echo "[ck1 gate] EXPECT: seg ~0.00553676 (Knee-A tokens unchanged), pose = the ck1 re-solve,"
echo "           rate ~0.18268; accept iff composed S < 2.256641 (advisory win over the ref row)."