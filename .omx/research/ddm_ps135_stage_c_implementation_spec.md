# ddm_ps135 Stage C implementation spec

## Objective

Execute the charter's adaptive semantic mixed-precision x pose-compensation
stage on the LC2 vehicle after Leg A converges.  Produce a cumulative q4->q3
drop trace for the four retained SD1 tensor rungs, with each rung followed by
receiver-realized signed-int12 coefficient compensation, complete archive
charging, and full-n600 macOS-CPU advisory components.  This is a separate
sequential consumer of Leg A, not a proxy and not a second simultaneous scorer.

## Ground truth and reusable apparatus

- LC2 archive/container/runtime: `/Volumes/VertigoDataTier/pact/ddm_lc2_20260810/`.
- Leg-A result/state: `/Volumes/VertigoDataTier/pact/ddm_ps135_20260810/`.
- Bulk target/decode state: `/Volumes/APDataStore/pact/ddm_ps135_20260810/`.
- Retained exact LC2 token tensor:
  `/Volumes/APDataStore/pact/ddm_lc2_20260810/cold_decode/checkpoint/tokens.npz`.
- Mixed-precision packer/reference:
  `experiments/ddm_sd1_semantic_rd_curve.py::pack_semantic_state`.
- Shipped LC2 runtime already parses counted `SD1M` mixed allocations through
  `submission/inflate.py::semantic_allocation`; do not claim a missing selector.
- Cumulative measured SD1 rung order:
  `blocks.3.film.weight`, `blocks.2.film.weight`, `frame_embed.weight`,
  `blocks.1.film.weight`, using the retained `greedy_prefix_01..04` payloads.
- Reuse the exact LC2 container mutation, renderer, PoseNet, retention, resume,
  storage, lock, and pass machinery in `experiments/ddm_ps135_pose_resolve.py`.

## Required behavior

1. Implement a separate driver at
   `experiments/ddm_ps135_stage_c_mixed_precision.py` and focused tests.  Import
   the Leg-A module; avoid duplicating scorer/container laws.
2. Verify every source by byte count/SHA and prove the LC2 q4 semantic equals
   the SD1/PR130 q4 semantic before transferring a mixed allocation.
3. For each cumulative rung, build the actual LC2 CX2/Brotli archive with the
   rung's counted semantic bytes, the preceding rung's compensated carrier,
   frozen HPAC/tokens/temporal bytes, and strict parse-back.  Retain every
   semantic blob, carrier, compressed stream, archive, repeat archive,
   allocation, and receipt under
   `/Volumes/VertigoDataTier/pact/ddm_ps135_20260810/stage_c/`.
4. Materialize and retain the exact semantic master-frame bank from the shipped
   LC2 runtime and retained exact token tensor.  Use `frame_utils.yuv420_to_rgb`
   only for GT; generated masters come from the actual receiver renderer.
5. Warm-start each rung from the preceding compensated coefficient state.  Run
   the same full-n600 CPU-torch fp32 signed-int12 GN/exact-refresh search until
   three dry passes; no inherited eight-pass minimum is required for these
   continuation rungs.  Preserve every pass/candidate payload and atomic resume
   state.  The four rungs are cumulative, never independent marginal sums.
6. Globally exact-refresh every eligible rate variant before accepting a rung
   pass.  Keep archive bytes <=187,226 and admit only a strict full-S
   improvement relative to the preceding compensated rung; retain losing rungs.
7. Emit per-rung bytes, d_seg, d_pose, S, delta S, accepted rows, passes,
   elapsed time, S/hour, semantic allocation, and all payload hashes.  The final
   retained candidate must pass actual LC2 decode plus untouched
   `upstream/evaluate.py` at n600.  Label all local numbers
   `[macOS-CPU advisory]`; never move the contest pointer or dispatch Modal.
8. Acquire the same fleet scorer lock and require the sole ps135 lane claim.
   Chunk <=120, pin batch16/threads, and preserve the final unpadded batch8.
9. Consume the #453/#460 JRD policy as data: retain source/hash and compile its
   no-confirmation state.  The PR110 plane ranges are dormant n=1 ordering data,
   not LC2 precision assignments.  Keep JRD exact click polish terminal after
   each compensated convergence only if it remains a distinct exact lattice
   traversal.

## Acceptance criteria

- `python -m py_compile` passes for both new files.
- Focused pytest covers q4 identity, cumulative allocation order, LC2 mixed
  parse-back, state/payload mutation refusal, resume bindings, batch geometry,
  and a mocked rung FIRED/dry continuation path.
- A scorer-free `preflight` materializes q4 and four mixed LC2 archives,
  verifies complete parse-back/repeat identity, and retains them before any
  full-n600 launch.
- Two review-tracker passes and serializer commit occur before launch.

## Do not touch

- `upstream/`.
- The three protected files named by the common contract.
- Unrelated dirty files or the existing staged index.
- Existing SD1 retained payloads; consume them read-only.
- Modal, MPS authority, subset verdicts, or scalar-only payload measurements.
