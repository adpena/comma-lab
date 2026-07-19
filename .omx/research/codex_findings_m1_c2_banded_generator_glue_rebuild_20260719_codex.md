# Codex findings — M1 / Task #575 C2 banded-generator glue rebuild

**Date:** 2026-07-19 UTC
**Lane:** `lane_m1_c2_banded_generator_glue_rebuild_20260719`
**Verdict:** `IMPLEMENTATION_REBUILT_BOUNDED_CONTROL_PASS_FULL_FIRE_BLOCKED`
**Authority:** local BUILD plus capped real-GT macOS-CPU hard-oracle only; non-n600,
non-score, non-promotion
**Pointer:** `0.1910828242 [contest-CPU Linux x86_64]` **UNMOVED**

## Outcome first

The lost M1 glue is rebuilt: the active `IntegerPlaneEmitterPolicy` now emits argv
consumed by the dedicated parser; the dedicated streamed trainer owns only 600-pair
codes plus the shared residual head and preserves atomic EMA/live/optimizer/RNG
checkpoints at every stage; and the counted adapter strictly parses its archive,
measures exact ZIP bytes, and proves NumPy-emitter/factor-2 receiver equality before
hard scoring.

It is **not ready to fire**. Three later operator directives expose real dependencies
that are absent rather than configuration details:

1. no real full-n600 positive anisotropic band with a hash-bound 38,077-candidate
   Fisher-margin EV field, corrected inner-Jacobian/secant/QP prediction, and KKT
   break-even stop;
2. #553 PDW2 is a canonical counted target but #543 has no scorer-free
   coefficients-to-spatial/RGB pullback, so it remains
   `TARGET_ONLY_VS_REALIZATION_NON_EQUIVALENT`;
3. the pre-existing emitter topology is a four-term polynomial coordinate control,
   not a receiver-bound curvelet/shearlet carrier. No Fourier/Euclidean proxy was
   substituted.

The governed materializer returns rc=6 under those conditions. The durable blocked
config is
`.omx/research/m1_c2_banded_generator_governed_blocked_config_20260719.json`.

## What landed

- `IntegerPlaneEmitterPolicy` has compatibility-OFF and argv-effective
  `banded_training` modes. Launch, paid dispatch, score, promotion, and pointer
  authorities remain sealed false. The policy hash is consumed by the exact trainer
  parser and every checkpoint.
- `experiments/train_c2_integer_plane_emitter_banded.py` delegates to the dedicated
  streamed implementation. Base/source/band stores remain read-only memmaps; batches
  are bounded; only `pair_plane_codes` and `shared_rgb_head` receive gradients.
- `warmup`, `band_fit`, and `rate_polish` each preserve distinct atomic stage-end
  checkpoints. Periodic checkpoints are additive, and resume rejects config/data/
  policy drift while restoring live, EMA, optimizer, RNG, stage, epoch, step, and
  next-pair state.
- Positive bands now require an EV-selection record that binds the measured law
  family, 38,077 full-run candidates, Fisher/top1-top2-margin metric, highest-EV-first
  reverse-waterfill order, curvelet carrier identity, one SE(3) xi pose factorization, and the exact
  `25/37,545,489` stop price. Unselected pixels carry radius 255 and exert no blanket
  source-matching force.
- The byte-close archive counts the compact base packet, the strict #553 PDW2 target,
  quantized pair codes, a separately named C2 RGB residual factor, and optional dense
  repair. Unknown, duplicate, encrypted, truncated, noncanonical, and trailing bytes
  refuse. `receiver_consumed` PDW2 authority refuses until the missing spatial pullback
  exists.
- The #543 receiver contract/arithmetic identifiers are cross-bound to the exact
  factor-2 lattice path. This is lattice reuse, not a false PDW2 spatial-consumption
  claim.
- Large capped raw scorer inputs are success-cleaned only after a fsynced,
  machine-readable reproducibility authorization. The one completed run's two
  36,624,096-byte raws were separately certified and removed; counted archives and
  every checkpoint remain on the SSD.

## MEASURED bounded real control

Durable summary:
`.omx/research/m1_c2_banded_generator_glue_rebuild_receipt_20260719.json`.
Full external receipt:
`/Volumes/VertigoDataTier/pact/evidence/m1_c2_glue_rebuild_20260719/receipt.json`,
SHA-256 `d28db2cdc80b3514c12ab6f81b5b66a3bf53014ef12669a3ecc6492278b39dd1`.

| field | pre | post | delta | label |
|---|---:|---:|---:|---|
| archive bytes | 94,344 | 94,352 | +8 | **MEASURED exact ZIP** |
| `d_seg` | 0.003132290323264897 | 0.003132290323264897 | 0 | **MEASURED prefix n6 hard CPU scorer** |
| `d_pose` | 150.166140238444 | 150.166140238444 | 0 | **MEASURED prefix n6 hard CPU scorer** |

Additional custody:

- logical codes: 600 pairs; executed/scored prefix: 6 pairs;
- stages/steps: 3/9; distinct preserved checkpoints: 8;
- peak RSS: 580,861,952 bytes;
- SSD storage preflight: 786,854,711,296 free versus 2,000,000,000 required;
- PDW2 margin packet: 138 bytes, SHA-256
  `93c0d3320e6673aed1975426a6c8c1bbc41475f295ea62b357ad7a6bf9427568`;
- parsed scorer bytes equal canonical NumPy bytes; exact factor-2 lattice proof passes;
- positive band active: false; EV metric active: false; carrier is explicitly the
  polynomial control; `training_moved_d_seg=false`.

The 138-byte PDW2 value follows the canonical #553 memo/receipt and source. The
operator message's transient 142-byte wording is not used to overwrite that settled
artifact.

## Verdict ladder

- **CONFIRMED:** active typed DSL -> exact trainer parser; streamed 600-code state;
  deterministic resume; complete stage checkpoint envelope; exact counted/archive
  parse-back; exact factor-2 lattice; real-cache hard-scorer control; disk hygiene.
- **MEASURED:** bounded zero-band control did not move `d_seg` or `d_pose` and cost
  eight archive bytes. This is a structural control, not a C2 efficacy result.
- **BLOCKED:** full positive-band fire, PDW2 receiver closure, curvelet/shearlet
  carrier closure, measured Fisher/secant/QP EV field, and model-factorized gauge
  covariance.
- **NO VERDICT:** whether a correctly custodied surgical C2 positive arm lowers the
  full-n600 score. The absent inputs cannot be replaced by a zero band, Euclidean
  ranking, Fourier carrier, or the prefix control.

Round-1 fresh-eyes review is durably recorded in
`.omx/research/m1_c2_banded_generator_glue_rebuild_round1_review_20260719.json`.
It closed three implementation defects (artifact-byte custody, radius/count
cross-checks, and cleanup certification ordering) and kept the three authority
blockers fail-closed.

## Triality and system wire-in

- **DSL:** argv-effective `IntegerPlaneEmitter` active mode plus compatibility OFF;
  compiled policy/hash and runtime schema bind parser/checkpoints.
- **Equations consumed, not re-registered:**
  `realization_necessity_preimage_per_stratum_v1`,
  `resize_exploit_flip_fix_frontier_v1`,
  `segnet_head_rank4_linear_flipdist_v1`, Fisher/margin laws,
  `flip_margin_step_law_v1`, `instant_projected_input_adjoint_v1`,
  curvelet/shearlet laws, scorer/xi factorization laws,
  `witness_measured_reverse_waterfill_v1`, the KKT ranker, and
  `cgauge_master_action_v1`.
- **DAG:** Task #575 rebuild -> bounded control pass -> three exact fire blockers ->
  future real EV/band plus PDW2 spatial receiver plus curvelet carrier -> governed
  full run -> exact CPU/CUDA replay. No downstream edge is authorized today.
- **Sensitivity/Pareto/bit allocator/autopilot:** positive admission consumes the
  measured EV/KKT manifest; no empirical positive row exists to append. The capped
  zero-control is deliberately not admitted as a promotion signal.

## STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; v7.5/v8/v10 specs; Task #575 recovery spec; canonical
task/lane/subagent state; latest C2/PDW2/#543 memos and sources; measured necessity,
resize, Fisher, flip, curvelet/shearlet, xi, reverse-waterfill, KKT, and master-action
law surfaces; real compact base archive; real n600 GT cache; frozen CPU scorer source;
the task inbox through `2026-07-19T19:48:01Z`.

## MAIN landing requirement

This branch is not repository authority. MAIN must review the whole diff, rerun the
focused tests, verify the durable/external receipt and cleanup hashes, and decide
whether to merge. Task #575 remains blocked after merge until the named positive
inputs and receiver/carrier consumers exist.
