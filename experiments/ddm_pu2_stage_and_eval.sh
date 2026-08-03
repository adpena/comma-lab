#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# ddm_pu2 — stage the rebuilt cx1 archive as a submission and run the EXACT
# end-to-end upstream evaluation on it (inflate.sh -> evaluate.py, n600).
#
# This is the ONLY step that closes ddm_pu2 §6 R1-b (the surrogate gap): every
# d_pose in the probe came from our own frozen-PoseNet instantiation, which
# agrees with the authority to 1.6e-5 in AGGREGATE but is not bit-exact.  A knob
# set optimized against that instrument must be re-scored by the authority
# itself before any aggregate number is quoted.
#
# Axis: [macOS-CPU advisory] — NOT contest-CPU (that requires Linux x86_64).
# NON-PROMOTABLE.  The exact contest pointer 0.1910828242 is UNMOVED by this.
set -euo pipefail

UP=/Volumes/VertigoDataTier/pact/molab_witness_machine_upstream_20260709
SRC=/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/eval_root/submissions/v4d_cx1_pj2ix2
WORK=/Volumes/VertigoDataTier/pact/ddm_pu2_20260803
DST="$WORK/submission_pu2"
NAMES=/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/eval_root/public_test_video_names.txt
VIDEOS=/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/eval_root/videos

REBUILT="$WORK/verify_archive/0.bin"
if [ ! -f "$REBUILT" ]; then
  echo "ERROR: rebuilt 0.bin not found — run --mode bytes first" >&2; exit 1
fi

rm -rf "$DST"; mkdir -p "$DST"
# runtime tree: byte-identical to the shipped cx1 receiver (generic code, rule-118)
for f in inflate.sh inflate_runner.py pfs1_warp_receiver.py ddm_tr1_runtime.py \
         ddm_r7_token_coder.py ddm_ix2_archive_container.py \
         repair_entropy_coder_runtime_adapters.py; do
  cp "$SRC/$f" "$DST/$f"
done
# archive.zip is rebuilt from the probe's winning knobs by the SAME deterministic
# single-member STORED packer the encoder uses.
.venv/bin/python - "$REBUILT" "$DST/archive.zip" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "src")
from tac.optimization.ddm_ix2_archive_container import build_single_member_zip
payload = Path(sys.argv[1]).read_bytes()
Path(sys.argv[2]).write_bytes(build_single_member_zip(payload))
print("archive.zip bytes:", Path(sys.argv[2]).stat().st_size)
PY

echo "=== runtime-tree + archive custody ==="
shasum -a 256 "$DST/archive.zip" | tee "$DST/archive.sha256"
ls -l "$DST/archive.zip"

echo "=== inflate (exact contest entry point) ==="
rm -rf "$DST/archive" "$DST/inflated"
mkdir -p "$DST/archive"
unzip -o -q "$DST/archive.zip" -d "$DST/archive"
bash "$DST/inflate.sh" "$DST/archive" "$DST/inflated" "$NAMES"

echo "=== evaluate (n600, exact upstream) ==="
cd "$UP"
.venv/bin/python "$UP/evaluate.py" \
  --submission-dir "$DST" \
  --uncompressed-dir "$VIDEOS" \
  --video-names-file "$NAMES" \
  --device cpu --batch-size 16 --num-threads 2 \
  --report "$DST/report.txt"
echo "=== report ==="
cat "$DST/report.txt"
