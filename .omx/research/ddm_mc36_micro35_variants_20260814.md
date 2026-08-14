# ddm_mc36 MICRO35 variants: C clears every local gate

Variant C, `successor_drop532_pair105`, is the only admitted successor. It is a
real 186,269-byte archive with 37 net Seg flips, `delta d_pose =
-1.4632967835484165e-10`, exact receiver parse-back, and a byte-identical
repeat. All four unchanged mc35 gates pass on the local admission surface. One
dual-axis T4 fire order is sealed but not fired. This arm did not run a full
n600 authority scorer, `upstream/evaluate.py`, Metal, MPS, Modal, or any remote
worker, and it did not move a frontier pointer.

## Measured rows

The baseline is cp135 at 34,970 flips and 186,252 bytes. The Pose gate is the
delta against the same retained local baseline, with cap
`5.9739759814e-10`. Every row uses all changed pairs after one composed receiver
closure, never a prefix.

| variant | Seg net flips (gate >=35) | archive / delta bytes (gate <=+29) | delta d_pose (gate <=5.9739759814e-10) | parse-back + repeat | disposition |
|---|---:|---:|---:|---|---|
| A `successor_pair105` | 35, PASS | 186,313 / +61, FAIL by 32 B | `-3.2393930715483334e-10`, PASS | PASS | FOLDED; rate failure |
| B `successor_drop532` | 37, PASS | 186,292 / +40, FAIL by 11 B | `+5.4409295585687925e-9`, FAIL by `4.843531960428793e-9` (9.11x cap) | PASS | FOLDED; rate and Pose failures |
| C `successor_drop532_pair105` | 37, PASS | 186,269 / +17, PASS with 12 B headroom | `-1.4632967835484165e-10`, PASS | PASS | QUEUED-WITH-A-FIRE-ORDER |

Axes:

- A: `[macOS-CPU advisory frozen CPU-torch SegNet/PoseNet; 8 changed pairs over n600; successor_pair105] NON-PROMOTABLE`.
- B and C: `[macOS-CPU advisory frozen CPU-torch SegNet/PoseNet; 7 changed pairs over n600; <variant>] NON-PROMOTABLE`.
- Denominators: 600 pairs, 117,964,800 Seg pixels, and 3,600 Pose scalars.

The local Pose baseline recomputed as `0.0001474653494795297`; it is not
substituted for cp135's pinned contest instrument component. Only the matched
candidate-minus-base delta is used by this local gate. CPU recomputation of the
affected base Seg fields differed from the retained T4 base fields at zero
pixels for A, B, and C.

The non-promotable local component recomposition is `delta S =
+1.0905350235622721e-5` for A, `-4.0225051272743045e-6` for B, and
`-2.006473916964026e-5` for C. These are admission diagnostics, not contest
scores.

## Built objects and receiver closure

Variant A kept the eight mc35 token objects byte-identically and replaced only
pair 105's compensation. The solver first ran QS5's exact-object DLS/int12
solve, then admitted only signed-int12 neighbors satisfying the full composed
n600 Pose cap. Four strict feasible descent passes were accepted before a
complete non-improving pass. The other seven compensation rows were reused only
after their semantic-token, rendered-master, and compensation fingerprints
matched.

Variant B removed pair 532, rematerialized the seven-object entropy stream, and
fresh-solved compensation on all seven final rendered objects. Its primary and
repeat RC64 token streams are each 115,238 bytes with SHA-256
`b44367b18f5f625834ef04239c4a5850fe9a9ec129e685b4c3609496f6f8c98c`.

A passed Pose; B was 21 bytes smaller than A and gained two more Seg flips.
That measured complementarity justified C. C combines B's seven-object token
stream with A's constrained pair-105 compensation and B's other six exact
compensation rows, after rechecking every object fingerprint. Its archive is:

- path: `/Volumes/VertigoDataTier/pact/ddm_mc35_successor_drop532_pair105/micro35_candidate/archive.zip`
- bytes: 186,269
- SHA-256: `f0ba4bb41d55fff85542f2a17dfe682508aa4f9ab50ef51cda573d79f0c4b1de`
- repeat: same bytes and SHA-256
- runtime parse-back: exact 115,238-byte token stream; exact compensation pairs
  `[7, 96, 105, 176, 178, 517, 523]`; receiver-identical HP4 container

`unzip -tqq` passed for all three archives. `cmp` passed for every
archive/repeat pair and for C's archive against the sealed fire input.

## Fire order

Exactly one request is sealed at
`/Volumes/VertigoDataTier/pact/ddm_mc35_successor_drop532_pair105/SEALED_FIRE_ORDER.json`.
Its request SHA-256 is
`c73a72d902b81358090ae697e3caa0e524fccde64bdfbc01d326ba9573d6b5bd`.
The sealed archive is the exact C archive above. The deterministic runtime zip
is 238,713 bytes with SHA-256
`64e4642d30b436e6393d5573efcb579a13f922726566790efad40bc2ca117545`;
`unzip -tqq` passed.

Disposition is `QUEUED-WITH-A-FIRE-ORDER`. Owner is MAIN, the sole scorer-lane
router. Consumer store is
`/Volumes/VertigoDataTier/pact/ddm_mc35_successor_drop532_pair105/dispatch/ddm_mc36_dual_axis_t4_r1`.
The fire trigger is: MAIN verifies no active full-n600 scorer lane, claims
`ddm_mc36_dual_axis_t4_r1`, and verifies the sealed archive/runtime SHAs. The
worker is the proven RE1T/JS1B dual-axis T4 chain, estimated at about $0.16 with
scorer chunks no larger than 120. The arm did not preclaim the lane or dispatch
the request.

Required returns are the retained decoded `0.raw`, candidate/GT/base argmax
fields, candidate/base/GT first-six PoseNet vectors, and the exact upstream
evaluator receipt on the same archive bytes.

## Payload custody and resumability

All materialized payloads were retained. Logical stores are:

- `/Volumes/VertigoDataTier/pact/ddm_mc35_successor_pair105`
- `/Volumes/VertigoDataTier/pact/ddm_mc35_successor_drop532`
- `/Volumes/VertigoDataTier/pact/ddm_mc35_successor_drop532_pair105`
- coordinator: `/Volumes/VertigoDataTier/pact/ddm_mc36_20260814`

Bulky retained state was routed by immutable receipts to APDataStore's second
SSD tier under `/Volumes/APDataStore/pact/ddm_mc36_20260814/`. Current physical
sizes are about 3.0 GiB for A, 13 GiB for B, and 229 MiB for C. No retained
payload was deleted or moved.

B exposed a JO1 hard-link device constraint after an initial AP compile
workspace had materialized about 455 MiB. That failed-attempt payload remains
under `compile_workspace_cross_device_attempt`; an immutable storage addendum
records it, and the replacement compile workspace stays on Vertigo with the
source probabilities. Resume then reused all completed compensation and coder
stages.

Two immutable-receipt namespace collisions were also found and fixed: the
variant wrapper now leaves the solver's `FRESH_COMPENSATION.json` intact and
writes `VARIANT_COMPENSATION.json`; the sealed status leaves the local-gate
`FINAL_RESULT.json` intact and writes `FINAL_RESULT_SEALED.json`. The first,
overly strict gate-only C adjudication remains preserved at coordinator
`FINAL_RESULT.json` (SHA-256
`fcf99b7594c51c338e59906e0b1ea2b8c297777d4b9677c903227ae370c9d650`)
and is explicitly superseded by `FINAL_RESULT_V2.json` (SHA-256
`0ebde5a957a0698e06d48d198b97193970460a0fce8a9ae4ce15013c6acbea83`).

## RECALL EVIDENCE

Searched before adjudication:

- full content under `.omx/research/` for `MICRO35`, `pair 105`, `drop 532`,
  `fresh compensation`, `in-compile compensation`, `quantize then compensate`,
  `stale fit`, and `pose safety`;
- `.omx/research/CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*`, the live hot state,
  task ledgers, queues, and task lists for the same terms;
- `.venv/bin/python tools/list_canonical_equations.py --json`, filtered for
  MICRO35, the two pair numbers, compensation, quantization, and pose safety;
- the complete mc35 and QS5 memos/builders and all retained mc35 support,
  compensation, coder, archive, and scorer receipts.

Beyond the charter seeds, `ddm_rvs1_realization_survival_harvest_20260811.md`
requires hard-rounding the final camera lattice before re-solving nearby legal
DOF; `ddm_sf1_stale_fit_genus_sweep_and_structural_fix_20260802.md` says a
moved partner invalidates the old solve and re-solving is the only cure; and
`ddm_wr1_reverse_waterfill_20260729.md` supplies the pose-safety-constrained
variant precedent. Those findings changed A from a post-hoc rejection/filter
into a constraint inside the exact-object solve, forced B to fresh-solve every
final object, and required C to prove fingerprints before reuse. No direct
MICRO35/pair-105/drop-532 canonical-equation, index, DAG, or ledger row beyond
the mc35/QS5 seeds was found in those bounded scopes.

## Verification and boundaries

- 46 focused tests passed across the mc35/mc36 builder, QS5 exact-object solve,
  QS1 Schur solve, and QS2 overlay runtime.
- Python bytecode compilation, Ruff, `git diff --check`, and the payload
  measure-and-discard detector passed.
- Both modified Python files received two review-tracker passes; policy check
  reports zero violations.
- Every archive/repeat payload, scorer input/logit/argmax/Pose field, solve
  batch, final code/vector, token stream, runtime, and fire input is retained
  with bytes and SHA-256.
- No exact contest score is claimed. The exact frontier pointer is unchanged.

## NEXT_IF_RESUMED

- `QUEUED-WITH-A-FIRE-ORDER` - owner: MAIN sole scorer-lane router; consumer store: `/Volumes/VertigoDataTier/pact/ddm_mc35_successor_drop532_pair105/dispatch/ddm_mc36_dual_axis_t4_r1`; fire trigger: no active full-n600 scorer lane, MAIN has claimed `ddm_mc36_dual_axis_t4_r1`, and archive SHA `f0ba4bb4...` plus runtime SHA `64e4642d...` match the sealed request; action: fire the proven RE1T/JS1B dual-axis T4 worker and retain every required return.

## LIVE-HYPOTHESES

- C's local gain may survive on T4 because the exact receiver object closes,
  all seven base Seg fields match the retained T4 base at pixel level, and its
  local component recomposition is negative. This is plausible but untested;
  CPU-to-T4 candidate scorer transfer remains the decisive uncertainty.
- The drop-532 and constrained-pair-105 mechanisms interact favorably in the
  real container: C is 23 bytes smaller than B and 44 bytes smaller than A,
  despite changing only compensation relative to B. The nonadditive HP4 result
  is measured for this object, but its exact contest score effect is untested.

## DEAD-ENDS

- A alone is closed for this instance: it passes Seg and Pose but misses the
  unchanged rate gate by 32 bytes.
- B alone is closed for this instance: it gains 37 flips but misses rate by 11
  bytes and Pose by `4.843531960428793e-9`.
- Deciding C from gate booleans alone is closed: A/B metric-level
  complementarity justified a real build, and the built C passed every local
  gate.
- Additive QS2 + RE1 + HP4 projections and stale compensation transfer remain
  closed as admission evidence; only the built, parsed, recounted object was
  admitted.
- Local admission is closed as promotion authority. Own-vehicle frontier
  remains LC2 `S = 0.16959899569230852 @ 187,226 B [contest-CUDA T4,
  adjudicated, n600]`.
