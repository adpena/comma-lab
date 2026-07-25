---
title: DDM MR1 independent-approver review of J12 and inherited PC1
date_utc: 2026-07-25
reviewer: mr1-independent-approver
reviewed_tip: 0fc6659e2e4846ba1966ec129ae7cb860a49bb39
main_landing_review_required: true
score_claim: false
pointer_moved: false
---

# Verdict

`MERGE-WORTHY AFTER FIXES`. J12’s realized forward secants, exact adapter
parse-back, full-column-rank implication, complete-Gram null projectors, and
pure-priced objective authority survive independent rederivation. This
credential does not authorize the merged-main/worst-geometry reseal, READY,
FIRE, training, promotion, or a pointer move.

# Mandated critical-entity verdicts

1. `serialize_pc1_packet`: CLEAN. Typed immutable arrays, bounded packet
   geometry, canonical zlib bytes.
2. `parse_pc1_packet`: CLEAN. Exact header/length validation and canonical
   re-emission.
3. `receive_pc1_camera_pairs`: CLEAN AFTER FIX. Exact inactive identity and
   deterministic uint8 realization remain unchanged; the finite macOS
   NumPy/Accelerate matmul warning is contained locally and all real
   nonfiniteness now refuses.
4. `null_projector_from_full_column_rank_sketch`: CLEAN AFTER FIX. The
   implication `rank(LJ)=n => ker(J)={0}` is valid; coordinate IDs must now be
   unique nonempty strings.
5. `null_projector_from_receiver_gram`: CLEAN AFTER FIX. The complete
   `J.T@J` eigenspace construction remains symmetric/idempotent and now has the
   same coordinate-ID custody.
6. `objective_gate_contradiction`: CLEAN AFTER FIX. Only a literal boolean
   auxiliary decision is admitted; it cannot override the realized joint
   objective.
7. `_measure_jacobians`: CLEAN AFTER FIX. Resumed source/proposal NPZ arrays
   now revalidate pair IDs, exact shapes, and finiteness before Gram/digest
   accumulation.
8. `_rehomed_endpoint`: CLEAN AFTER FIX. Resumed chunks now revalidate schema,
   endpoint/packet/parent/archive identities, pair range, site/pose counts,
   per-class totals, and finite nonnegative Pose SSE.

The three source-preserving adapter helpers
`build_source_preserving_pc1_adapter_archive`,
`parse_source_preserving_pc1_adapter_archive`, and
`receive_source_preserving_pc1_camera_pairs` are also clean after zero-effect
packet type validation was made mandatory rather than bypassed by raw-parent
canonicalization.

# PC1 warning disposition

The reported warnings were reproduced under `PYTHONWARNINGS=error` at the
finite `rotation.T @ points` call: divide-by-zero, overflow, and invalid
warnings were emitted together even though every input and output was finite.
A direct `einsum` comparison was byte-equal, confirming a backend warning
artifact. The fix deliberately preserves the original matmul and its output,
uses a narrow `np.errstate` only around the known operation, then fails closed
unless the transform, inverse projection, and float32 grid are finite and
representable. An extreme transform regression test proves the refusal path.

# Clean passes

Pass 1 rederived packet/adapter byte custody and call sites. Pass 2 rederived
rank/null-space logic and pure-objective authority. Pass 3 attacked resumed
NPZ/chunk geometry, malformed zero-effect packets, nonboolean gates, and
nonfinite warp/scorer states after the fix reset.

### src/tac/optimization/ddm_pc1_pose_stream.py — CLEAN

### tools/tests/test_ddm_pc1_pose_stream.py — CLEAN

### src/tac/optimization/ddm_j11_opening_proposal_decomposition.py — CLEAN

### src/tac/optimization/tests/test_ddm_j11_opening_proposal_decomposition.py — CLEAN

### tools/run_ddm_j12_receiver_coordinate_custody.py — CLEAN

### tools/tests/test_run_ddm_j12_receiver_coordinate_custody.py — CLEAN

# Verification

- Four touched/consumer suites under `PYTHONWARNINGS=error`: 30 passed, zero
  warnings.
- Five preserved J12 PC1 endpoints: 95/95 chunks passed the new semantic
  validator without scorer execution or artifact mutation.
- Ruff check/format, `py_compile`, and `git diff --check`: clean.

# Authority boundary

No scorer run, training, paid dispatch, exact contest evaluation, archive
promotion, reseal, FIRE decision, or frontier-pointer mutation occurred.
Landing this merge is only the prerequisite for the separately governed J12
merged-main/worst-geometry reseal. MAIN must review the merge commit and this
credential before landing.
