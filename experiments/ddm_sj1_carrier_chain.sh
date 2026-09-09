#!/bin/bash
# ddm_sj1 -- the POINTER-DEPENDENT half of a pass: stale pose, carrier re-solve, resolved
# pose.  Sister of ddm_sj1_successor_price.sh, which holds the pointer-INDEPENDENT half.
#
# The split is not cosmetic.  `render-edits` touches the pointer ZERO times, while `pose`
# goes through `load_pose_instrument` -> `assert_carrier_is_pointer` and BINDS the run to
# whichever row is live at that instant.  So the encodes and renders can run while a
# sister arm's row is in flight, and everything below runs ONCE, against the final
# pointer.  Appending a new PointerRow mid-chain makes every step here refuse -- which is
# correct, and is what stops a carrier solved from one pointer's coefficients being
# spliced onto another pointer's member -- but the partial work is then wasted.
#
# Nothing here names the pointer: `up2.load_carrier_state` READS the basis, the
# coefficient scales and the codes out of the live pointer tree's own archive, using that
# tree's own runtime reader.  A re-base is therefore an appended row, never an edit here.
#
# Env: OVERLAY (the candidate's odd frames), FIELD (the pass's edited npz),
#      BASE_POSE (the live row's own resolved pose), OUT_POSE, OUT_REFINE,
#      SHARDS (default 6), THREADS (default 2), POSE_THREADS (default 6).
set -euo pipefail

REPO="/Users/adpena/Projects/pact"
OVERLAY="${OVERLAY:?set OVERLAY}"
FIELD="${FIELD:?set FIELD}"
BASE_POSE="${BASE_POSE:?set BASE_POSE}"
OUT_POSE="${OUT_POSE:?set OUT_POSE}"
OUT_REFINE="${OUT_REFINE:?set OUT_REFINE}"
SHARDS="${SHARDS:-6}"
THREADS="${THREADS:-2}"
POSE_THREADS="${POSE_THREADS:-6}"

mkdir -p "$OUT_POSE" "$OUT_REFINE"
cd "$REPO"

# 1. STALE -- the live pointer's OWN carrier on the NEW renders, i.e. the pose damage this
#    pass's edits do BEFORE any re-solve.  The BASE leg is not re-measured: it is the live
#    row's own pose on its own renders, already on disk.
"$REPO/.venv/bin/python" experiments/ddm_sj1_joint_admission.py pose \
    --tag stale --overlay "$OVERLAY" --out "$OUT_POSE/pose_stale.npy" \
    --threads "$POSE_THREADS" > "$OUT_POSE/pose_stale.log" 2>&1

# 2. RE-SOLVE -- jg5's refine_pair on the candidate's own renders, seeded from the LIVE
#    pointer's coefficients and lattice (enforced inside the module, not remembered here).
OVERLAY="$OVERLAY" FIELD="$FIELD" BASE_POSE="$BASE_POSE" OUT="$OUT_REFINE" \
  SHARDS="$SHARDS" THREADS="$THREADS" \
  bash experiments/ddm_sj1_refine_shards.sh > "$OUT_REFINE/refine.log" 2>&1

# 3. RESOLVED -- the pose the candidate would actually ship, on the new codes.
"$REPO/.venv/bin/python" experiments/ddm_sj1_joint_admission.py pose \
    --tag resolved --overlay "$OVERLAY" --codes "$OUT_REFINE/codes_resolved.npy" \
    --out "$OUT_POSE/pose_resolved.npy" --threads "$POSE_THREADS" \
    > "$OUT_POSE/pose_resolved.log" 2>&1

echo "ddm_sj1 carrier chain complete: stale + re-solve + resolved"
