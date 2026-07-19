# M1 / Task #575 C2 banded-generator glue rebuild — implementation spec

**Date:** 2026-07-19 UTC
**Lane:** `lane_m1_c2_banded_generator_glue_rebuild_20260719`
**Authority:** delegated BUILD + bounded local verify only. The governed full-n600
training launch, paid dispatch, exact contest eval, promotion, and pointer mutation
are out of scope. MAIN must independently review and merge the isolated branch.

## Outcome

Replace the three missing integration pieces named by
`m1_c2_banded_generator_launch_blocker_and_recovery_spec_20260719.md`:

1. a streamed n600-capable band trainer for the existing C2 quotient residual;
2. an argv-effective `IntegerPlaneEmitterPolicy` DSL path consumed by that exact
   trainer (no hand-invented launch flag and no routing through the level-set
   coordinate-INR trainer);
3. a counted, parse-back-verified byte-close adapter that combines a counted base
   generator description with the C2 pair codes/shared head and optional certified
   band-slack repair, then reproduces the emitted uint8 planes/frames exactly.

The implementation rebuilds the lost glue but promotes only if every later operator
gate below is present. The bounded control proved the structural path; it did not
make the vehicle ready to fire. The contest-CPU pointer stays `0.1910828242`.

## Superseding operator gates consumed during implementation

Two directives arrived after this spec was frozen and are binding:

1. **PDW2 consumption:** the counted adapter must carry strict canonical #553
   gauge-fixed target bytes, not reinvent/raw frozen-head coefficients, and may
   claim receiver closure only when #543 causally expands that target into the
   spatial/RGB witness. Current #553 custody is explicitly
   `TARGET_ONLY_VS_REALIZATION_NON_EQUIVALENT`; #543 supplies the exact factor-2
   lattice but no scorer-free feature-to-spatial pullback. The adapter therefore
   counts and parse-checks the 138-byte PDW2 margin packet as a training-only
   target certificate and refuses `receiver_consumed` authority.
2. **Surgical Fisher/EV allocation:** no blanket band is admissible. A positive
   n600 band must bind the measured 38,077-candidate EV field, rank in the
   Fisher/top1-top2-margin metric, use the first-order+secant+QP inner-Jacobian
   realization law, target cheapest Road-Lane/resize-realizable edges first,
   use curvelet/shearlet carrier custody, factor pose through one SE(3) xi twist,
   and stop when marginal value falls below `25/37,545,489` score units per byte.
   Unselected pixels use the sealed radius 255 and therefore exert no blanket
   source-matching pressure. The positive-band manifest hash-binds every input
   artifact and named canonical law.

The current coordinate topology is the pre-existing polynomial control, not a
receiver-bound curvelet/shearlet implementation. No executable measured EV-field
artifact or positive full-n600 band was found. Those are exact promotion blockers,
not permission to substitute Euclidean/Fourier or a proxy.

## Existing authorities to reuse, not fork

- Forward / fixed capacity / exact factor-2 lattice:
  `src/tac/boundary_math/integer_plane_emitter.py`.
- Source-centered band law and exact interval solve:
  `src/tac/optimization/joint_seg_pose_rate.py`. Positive bands must enter as
  custodied winner/rival pullback bands produced by
  `derive_hyperplane_channel_band`; the legacy isotropic positive band remains
  forbidden. A zero-radius control must be labeled as such.
- Policy and complete checkpoint envelope:
  `src/tac/witness_dsl/integer_plane_emitter_policy.py`. Reuse its canonical,
  distinct per-stage, EMA/live/optimizer/RNG-bearing checkpoint format. Do not
  invent a second incompatible resume format.
- DSL factory: `IntegerPlaneEmitter(...)` in
  `src/tac/witness_dsl/curriculum_dsl.py`.
- Real GT/scorer custody and source-plane projection helpers may be factored from
  `tools/measure_c2_integer_plane_emitter.py`; do not silently import an unpinned
  scorer from an arbitrary `sys.path` entry.
- Counted compact base anchor for the integration smoke:
  `/Volumes/VertigoDataTier/pact/yhat_rd_ladder_20260719/prepare/full_n600_packet/`
  (`archive.zip` SHA-256
  `149fefd097c1fa85c4afb6cb2d8ab20311035d7ba8063f1e72137b843a9b89f3`).
  It is read-only. Any derived bulk scratch belongs on the SSD and is success-cleaned
  only after a machine-readable reproducibility record exists.

## Required implementation contracts

### A. Active typed DSL consumer

- The policy must gain an explicit band-training compilation surface without
  granting launch, score, promotion, or pointer authority.
- `IntegerPlaneEmitter(policy=...)` must emit non-empty typed overrides and a
  runtime receipt schema. Its policy hash/capacity/basis/STE identity must cross-bind
  to the trainer and every checkpoint.
- The dedicated trainer's real argparse parser must accept the emitted argv. Add a
  behavioral test that compiles the Lever and parses it with the exact trainer
  parser, then proves removing or mutating the policy flag refuses.
- Preserve a clearly named inactive/build-only policy mode if compatibility needs
  it; an "ON" integration policy must no longer compile baseline-identically.

### B. Streamed band trainer

- Add a dedicated C2 trainer entrypoint (do not modify or invoke
  `train_levelset_witness_realized_through_R_mlx.py`). It must operate at the real
  `[600,2,384,512,3]` logical geometry while streaming bounded pair batches; it may
  not allocate the complete dense base/source tensor in RAM.
- Train only `pair_plane_codes` plus `shared_rgb_head`. The base generator is
  immutable and source/GT bytes are read-only. Use deterministic seed-derived pair
  order and deterministic NumPy/Torch behavior.
- Loss must be source-centered band violation plus an explicit rate/parameter
  pressure. Candidate forward uses saturation-aware uint8 semantics. Positive band
  mode requires a validated custodied anisotropic band artifact; zero-band is a
  separately labeled control, never presented as a positive-band result.
- Support at least `warmup`, `band_fit`, and `rate_polish` stages. Save a distinct
  atomic checkpoint at every stage end and bounded periodic checkpoints for long
  stages. Preserve prior checkpoints. Save live parameters, EMA shadow, optimizer,
  RNG, stage/epoch/global step/next-pair, policy/config/capacity hashes, and the
  source/base/band identities. `--resume-from` must resume the next unfinished unit
  and reject config/data/policy drift.
- Include a storage waterfall preflight and an automatic success-only scratch
  cleanup/certification path.

### C. Counted byte-close

- Add a dedicated sibling adapter rather than pretending the existing level-set
  tool already consumes C2. Its input is a counted base generator packet plus a C2
  stage checkpoint (EMA is the default authority).
- The counted archive must include every video-derived byte required to reproduce
  the result: base generator description, quantized pair codes/shared head, and any
  band-slack repair. Generic decoder/lattice algorithms may remain rule-118 free.
- It must additionally include strict #553 PDW2 bytes and identify the C2 residual
  RGB factor separately from the frozen segmentation-head target. A no-op PDW2
  section is not receiver closure; promotion must fail until a scorer-free spatial
  consumer exists.
- Serialize deterministically, parse back exactly, reject trailing/unknown sections,
  report actual `archive.zip` bytes/SHA-256 and per-section byte counts, and prove
  decode equals the canonical NumPy uint8 emitter before any hard-oracle metric is
  reported. No dense source/base sidecar may be hidden outside the counted archive.
- The adapter must support a bounded pair cap for smoke while still reporting the
  actual full archive byte count separately from capped decode bytes. A capped
  result is explicitly non-n600/non-score evidence.
- Hard CPU-Torch d_seg/d_pose uses the real cache and frozen scorer hashes; rate is
  the actual archive size. It must report pre-training and post-training rows so
  "training moved d_seg" is measured, not inferred.

### D. Governed fire surface

- Land a deterministic config/argv materializer for the future full run. It binds
  the DSL policy hash, base archive/checkpoint/cache/band hashes, seed, stage plan,
  output/cold-store paths, and exact trainer argv.
- Preflight must return `4` for insufficient storage and `6` for stale/mismatched
  config custody. It must refuse a full fire without a positive-band artifact and
  without a byte-close-compatible base packet.
- This task does not run that command. The materializer intentionally returns rc=6
  while PDW2 spatial consumption, curvelet/shearlet carrier custody, or the measured
  Fisher/secant/QP EV artifact is absent.

## Tests and bounded verification

At minimum cover:

- DSL ON is argv-effective and exact-parser-consumed; OFF compatibility is explicit.
- positive isotropic bands refuse; positive anisotropic artifact custody is required;
- only the two quotient-residual arrays are trainable;
- deterministic fresh vs resume equality; config/base/source/band/policy tamper
  refusals; distinct stage filenames; EMA reload is byte-closeable;
- archive deterministic double-build, strict parse-back/trailing-byte refusal,
  counted-section sum, decoder/NumPy uint8 equality, and changed pair codes change
  decoded bytes;
- full logical n600 state with streamed batches and a measured peak-RSS ceiling;
- a real-cache, real-base, all-levers-ON bounded smoke followed by a small byte-close
  and hard-oracle comparison. It must not be a one-pair/tiny-batch false green.

The smoke may use a bounded number of optimization steps and decoded/scored pairs,
but it must instantiate all 600 pair codes, use real 384x512 planes, exercise every
stage/checkpoint/resume/byte-close path, record peak RSS, and label its capped metric
scope. If the host cannot honestly execute one of these surfaces, land the exact
machine-readable blocker and leave Task #575 blocked; do not substitute a proxy.

## Durable landing and review

- Append a dated DAG FEED with DSL/equation/pointer legs and the measured smoke row
  (or exact blocker).
- Update canonical Task #575 from `blocked` to `in_progress` during build, and only
  to `ready_to_fire`/the nearest valid enum after all required smoke gates pass.
- Record the lane-dispatch claim and terminal status; pointer remains unmoved.
- Every changed Python file receives two clean review-tracker passes and the whole
  branch receives a fresh adversarial round-1 review. Commit via
  `tools/subagent_commit_serializer.py` with expected content hashes.
- Final handoff names commits, verification, honest verdict, remaining governed
  launch blocker, and says explicitly: **awaiting MAIN landing review/merge**.
