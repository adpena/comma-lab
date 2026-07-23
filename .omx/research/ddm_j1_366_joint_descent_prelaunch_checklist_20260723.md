---
title: DDM J1 #366 fail-closed prelaunch checklist
utc: 2026-07-23T00:32:10Z
ticket: .omx/research/configs/ddm_j1_366_joint_descent_witness_program_20260723.json
status: ALL_ROWS_REQUIRED_CURRENTLY_BLOCKED
execution_allowed: false
main_landing_review_required: true
---

# Fire rule

Do not fire on this branch or from this ticket alone. MAIN must review and land the missing
consumer apparatus, then a separate operator GO must bind the exact compiled hash. Any missing,
stale, mismatched, advisory, or WARN-only row below is a REFUSE.

## A. Build and custody gates — currently RED

- [ ] `DirectDescriptionJointDescentTypedConfigV1` exists and compiles the ticket's semantic
  program without hand-written trainer flags.
- [ ] Recomputed RFC8785/SHA-256 of `semantic_program` equals the sealed ticket hash.
- [ ] v15 archive SHA is
  `759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df`.
- [ ] Stage-00 adapter loads the archive, re-emits the same counted parameters, and renders
  byte-identical camera output before an optimizer step.
- [ ] The 29,810-byte G1 payload is lifted into equivalent trainable coordinates without hidden
  state; current zero track/knot and lane-program/knot counts are handled by an explicit typed
  zero-state or immediately counted seed.
- [ ] A mutation of every trainable group changes its owned receiver output and its counted archive
  bytes; an unowned or no-op trainable group REFUSES.
- [ ] No scorer weights, GT table, decoded frame/mask plane, or post-hoc pose payload reaches the
  archive.
- [ ] MAIN round-1 review explicitly approves the warm-start adapter and counted-payload boundary.

## B. Resumability and storage gates — currently RED

- [ ] Fresh SSD-tier out-dir under `/Volumes/VertigoDataTier/pact`, falling back to
  `/Volumes/APDataStore/pact` only through the storage preflight.
- [ ] Same-outdir guard finds no live trainer/launcher/daemon process and no incompatible prior run.
- [ ] Every stage writes a distinct immutable EMA-shadow checkpoint atomically; periodic intra-stage
  checkpoints are enabled and prior stages remain loadable.
- [ ] A bounded crash/resume test proves exact stage/step, optimizer, EMA, RNG, and typed-hash
  continuation.
- [ ] Cleanup is success-only and provenance-certified; otherwise bytes remain in place and cleanup
  blocks.

## C. DSL, schedule, and freshness gates — currently RED

- [ ] Governed named config `ddm_j1_366_joint_descent` is accepted by
  `tools/launch_witness_run.py`; raw direct trainer invocation is forbidden.
- [ ] DSL/hash verifier accepts the exact compiled artifact (missing or mismatched binding is rc=8).
- [ ] Every transition is event-governed or a LawRef-tagged fail-safe cap.
- [ ] Config freshness is green. A stale schedule/config is rc=6 and blocks fire; never downgrade it
  with a skip flag.
- [ ] #378 amber resolves to grad clip 0.5, pose coefficient cap 25, normalized gradients, and
  per-group clip.
- [ ] #383 conditioning gate is present; sigma-star remains advisory; disengaged fallback is loud.

## D. Memory, governor, and timing gates — currently RED

- [ ] `witness_memory_preflight.py` parses the **real J1 emitted config**, not the historical R1
  level-set surrogate.
- [ ] Strict n600 projection records peak GiB, live free GiB, total GiB, operator ceiling, and tensor
  breakdown.
- [ ] `--system-aware` sums current system use and all active governed jobs and returns ADMIT.
- [ ] Real-config bounded dry start measures peak memory and reconciles projection versus actual.
- [ ] Storage waterfall has enough free space for checkpoints, verdict caches, and cold-store move.
- [ ] Timing smoke supplies measured seconds/step, seconds/exact-verdict, and projected stage/run
  wall-clock; the provisional 17–30 hour surrogate is not admission authority.
- [ ] Baseline resolves to MLX-GPU with `TAC_MLX_CUSTOM_GROUPED_BACKWARD=1` and fused
  differentiable-R. If timing exceeds 30 hours or memory binds 116 GiB, any proposed raster/R or
  YUV6/Pose kernel is measured by equal-config smoke and parity tests; speed never becomes authority.
- [ ] Governor returns ADMIT. A governor REFUSE is information and ends the fire attempt.

## E. Startup telemetry gates — currently RED

- [ ] Startup telemetry echoes source archive hash, semantic-program hash, seed, deterministic mode,
  EMA decay, stage id, checkpoint cadence, exact evaluator hashes, hardware axis, and out-dir.
- [ ] First receiver replay exactly matches d_seg 0.027470296224 / Movable 0.291615222639 / Lane
  0.435195521828 / 133,941 bytes before training.
- [ ] Chunked n600 exact-verdict telemetry is enabled for all stage exits.
- [ ] Telemetry reports global/per-class d_seg, official-YUV6 d_pose, bytes, score-unit value per
  byte, Fisher margin, secant prediction error, amber effective-step sensors, peak memory, and
  parse-back mutation.
- [ ] Any missing startup field or first-replay mismatch terminates before stage 01.

## F. Fire and promotion gates — currently RED

- [ ] Dry-run argv in the sealed ticket passes every gate without spawning.
- [ ] Bounded real-n600 dry start steps, checkpoints, resumes, and exits cleanly.
- [ ] MAIN review has landed the reviewed exact ticket hash.
- [ ] Separate operator GO binds that exact hash and fresh out-dir.
- [ ] Real fire uses the ticket's governed argv exactly; no paid dispatch is authorized here.
- [ ] Box fork uses one receiver-closed artifact: d_seg <=0.00116, <=200,000 bytes, Pose present.
- [ ] Exact contest-CPU and contest-CUDA are measured separately before any score/pointer claim.

Current verdict: `REFUSE_NO_JOINT_CONSUMER_OR_REAL_MEMORY_RECEIPT`.

Craft authority: `docs/operating_manual_craft_handoff.md`. Pointer
`0.1910828242 [contest-CPU]` unchanged.
