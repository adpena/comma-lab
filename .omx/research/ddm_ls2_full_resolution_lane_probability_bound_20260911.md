# LS2 — compact corrections retain the prior but recover only a small gain

**Measurement complete; exact frontier unchanged.** Both declared models have positive net
integer-frequency log-loss gains. The stronger joint model gives **385.550480 byte-equivalents net**,
only **1.4887%** of the **25,899 B** rate demand. No candidate fire is admitted: native receiver cost
and live complete-evaluation timing have not been measured.

Axis throughout the new measurement: **[macOS-CPU advisory / scorer-free n600 receiver probabilities]**.
`research_only=true`, `score_claim=false`, `promotable=false`. These are likelihood prices through
actual RC64 frequency conversion, not physical compressed stream lengths or a new exact score.

Consumer store: `/Volumes/VertigoDataTier/pact/ddm_ls2_full_resolution_lane_probability_bound/`.
Primary receipts: `RESULT.json`, `FIRE_OR_REFUSAL.json`, `PRICING.json`, `VALIDATION.json`,
`fit/{separate,joint}/{train,full}/RESULT.json`, and the actual `parameters.bin` files beside them.

## What was measured

| Full600 fitted model | Coefficients | Header + weights B | Rounded gross B | Rounded net B | Numerical optimistic net upper B |
|---|---:|---:|---:|---:|---:|
| Separate distance features + HPAC phase | 15 | 23 | 289.162261 | 266.162261 | 348.307699 |
| Joint distance feature + HPAC phase | 10 | 18 | 403.550480 | 385.550480 | 468.726421 |

All **117,964,800** symbols are included in rounded pricing, including correct predictions and
negative gains. The full fits are in-sample and the stronger model is selected from two declared
families; selection optimism is not hidden. Every proposed continuous trial, accepted vector and
all four rounded payloads are retained, including both training-only models.

The source is move44's archive, **180,406 B**, SHA
`04758c0dfb8d94ebe801602aac96d93a4f260aad24f58b6b9cdbbe2270ad460e`.
The encoder-bound shipped field is `subset6.u8`, **117,964,800 B**, SHA
`a92e7d902a4498961217f02c2b90d3fb9025901ba6d047201ff3bf297fa2f7a8`.
LS1 atlas RESULT SHA consumed:
`ddf6a80bd9d49e6fe1ace7c227650694c9c2af7178c97830b8fd57d98e231bb9`.
The retained actual decoder rows, exact context assignments and complete receiver checkpoints
were reused. No full decoder trace was repeated, and no scorer, renderer, archive or paid run was launched.

## Model and charged boundary

`PROTOCOL.md` was written and hashed before fitting. Each feature is the generic Q10 clipped
log-ratio `(earlier count + 0.5)/(earlier shipped expected mass + 0.5)`. Count banks are crossed
with the **shipped coding-row winning class** and contain all five possible outcomes. They update
only after an entire plane finishes, using unchanged shipped integer frequencies for expectations.
All inherited model weights, counted configuration and adaptive histories stay fixed.

The two LS1 distance codes are recovered exactly from its retained `receiver_lane` keys: prior
frame horizontal Lane distance and group-causal preceding-row Lane distance. Phase is the actual
HPAC group `x % 64 + 2*(y % 64)`, all 190 phases. Separate uses three features; joint uses the
64-way distance crossing plus phase. Each has one coefficient per feature per winning class.
The coefficients are signed int8 divided by 32, constrained to `[-4,127/32]`. The 8-byte `LS2P` header
counts format, family choice, scale and the selected Lane class. No empirical table or fitted
spatial selector is free. Count tables rebuild causally from decoded symbols.

The zero correction returns the exact original integer probability row without a float roundtrip.
Nonzero corrections reuse TC1's fixed power-table calculation, float64 normalization followed by float32 conversion and the
**actual shipped RC64 frequency conversion**. The inherited prior retains its full integer
resolution; it is never replaced by a constant-cell empirical distribution.

## Numerical lower-loss bounds

The TC1 C categorical objective, gradient and Hessian are reused with zero padding to seven
features. Only the declared 15 or 10 coefficients vary; inherited coefficients are never refit.
For the box, `F(v) >= F(w) + grad(F(w)) dot (v-w)` supplies a numerical lower loss. The reported
upper saving subtracts the exact parameter charge and grants every omitted symbol zero loss.

The KEEP objective retains **17,723,417** uncertain or wrong-winner symbols. The omitted
**100,241,383** correct/high-confidence symbols receive **85.722116 B** optimistic credit in each
full bound. This is explicit optimism, not measured attainable saving. Every rounded price then
includes these omitted symbols through RC64 conversion.

| Full model | Kept optimum gain B | Lower retained loss, nats | Numerical gap B | Omitted credit B |
|---|---:|---:|---:|---:|
| Separate |285.585580|661966.606445949|0.000002207|85.722116|
| Joint |401.004161|661326.589147558|0.000143789|85.722116|

These are floating-point supporting-plane receipts, not interval-arithmetic proofs. They bound
these continuous fixed-feature boxes, not arbitrary context architectures, online weight schedules,
carriers, generators, integer rounding discontinuities, or physical archive lengths.

## Held-out validation and signed attribution

Seed **20260911** chose 120 pair IDs uniformly without replacement from the entire 600-pair population;
480 pairs fit each validation model. Earlier decoded held-out outcomes may enter later receiver
count banks, as a real causal decoder would observe them, but never enter coefficient fitting.
This is validation of fixed coefficients under causal online state, not an independent-label
confidence guarantee. Full600-fit slices are not called held-out validation.

| Validation model | Fit gain B /480 | Held-out gain B /120 | Fit B/pair | Held-out B/pair | Held-out gain less entire parameter charge B |
|---|---:|---:|---:|---:|---:|
| Separate |238.131644|48.622033|0.496108|0.405184|25.622033|
| Joint |335.708295|67.049582|0.699392|0.558747|49.049582|

Held-out gains per pair are smaller. Training-only coefficients applied to all 600 pairs yield
263.753677 B and 384.757877 B net respectively. Full refitting adds only 2.408584 B and 0.792603 B;
these differences are descriptive, not an uncertainty estimate.

| Full-fit signed gross B | Road | Lane | Undrivable | Movable | MyCar |
|---|---:|---:|---:|---:|---:|
| Separate |267.748643|-88.535929|57.694425|-44.617355|96.872477|
| Joint |291.982212|47.158295|50.259610|-47.225163|61.375527|

The gain is primarily **Road**. Calling the separate model's total gain a Lane improvement would
be false: it worsens Lane. The joint model's Lane gain is only 47.158295 B.
Patch-row boundaries are exactly `y % 64 == 0`, six rows and 1,843,200 symbols over n600.
Full-fit boundary/other-row gains are 20.201660/268.960601 B (separate) and 29.550088/374.000392 B
(joint). Training-only held-out boundary gains are 3.151504 B and 4.628027 B. Complete class, row,
pair and boundary-by-class arrays, with signed losses preserved, are retained in `tables/*.npz`
and `VALIDATION.json`. No ranked spatial selector is granted in these models.

## Receiver work and fire decision

The source-based operation/state table is in `ddm_ls2_20260911/RECALL.md` and consumer `RECALL.md`.
Even the joint model entails roughly 1.180 billion feature multiplies,1.180 billion expected-mass
increments and 589.824 million power-table lookups, plus normalization, frequency conversion and
causal geometry. Its count/expectation state is 101,600 B despite its 18-byte parameter payload.
The separate model has more per-symbol work but less state. No receiver timing follows from these counts.
The compiled C loss instrument is an analysis tool, not a native receiver implementation.

Live move44 T4 inflate is **1232.418725255 s**, leaving only **27.581274745 s** below the source-defined
**1260 s** decode gate. Evaluation is another 50.726310111 s; the complete-job diagnostic is **WARN**.
The 1800 s contract is job-wide. TC4's actual later inflate timed out at 1800 s, so its prior local
projection and tiny weights cannot authorize this changed receiver.

**Typed decision: REFUSED_CANDIDATE_FIRE — timing admission absent.** Both families are positive
at their exact parameter charges, so the no-positive-gain stop rule does not apply. The joint model
is the sole retained lead for a MAIN-owned native timing feasibility check. No lossless recode is
queued or fired now. Only MAIN's separately bound build charter may authorize one after admission;
that build must retain twins, prove full public-field identity, charge every fitted byte and satisfy
complete-evaluation timing before exact dispatch.

The live component rederivation gives seg 0.010345 + pose0.00677495387438173 + rate0.12012495029695844
= S 0.13724490417134017, from the existing report's 8-decimal distortions and exact archive bytes.
The strict sub-0.12 integer cap is 154,507 B; demanded saving 25,899 B. Joint net likelihood gain leaves
25,513.449520 byte-equivalents of demand. Its conditional rate contribution is 0.000256722239 score
units at 6.658589531221714e-7 S/B; this is not a newly measured score or archive saving.

## RECALL EVIDENCE

Full search queries, sources beyond seeds, plan changes, bounded negatives and source exceptions
are in [RECALL.md](ddm_ls2_20260911/RECALL.md). DCC1 strengthened exact causality controls;
predictor R2 established prior charged-correction design; BND1 prevented log-loss being called a
physical stream; DWC1 and TC4's actual timeout established the runtime refusal. HC1/HC2/MI1 and
canonical equation exclusions prevented old magnitudes or constant-cell ceilings being transferred.
No additional current-field finite correction result was found in the recorded search scope.

## Verification, retention and landing

Causality controls pass 32 seeded pairs, including 176,640 queried symbol positions under seven
future/same-group masks per pair. Zero corrections preserve integer rows. Independent NumPy
loss/gradient/Hessian controls agree with TC1 C to at most 4.84e-13/8.89e-16/5.78e-15 respectively.
A separate audit imported the shipped codec and recomputed all four models on real frames 0,299,599:
all 12 class/row/boundary arrays match exactly, maximum error 0. Codec functions are AST-identical.

Independent LS1 custody audit passed 49 receiver-checkpoint entries, 5 resume entries, 100 runtime
source/copy pins, encoder receipt, field and 8 retained source-release files. Two live LS1 sources
have import-format drift; imported helpers remain AST-identical to retained release. This explicit
exception is recorded; no claim of complete live byte identity is made.

Every plane has a complete atomic feature/state checkpoint; every fit trial and accepted vector is
retained. Final stage payload manifest contains **653,146,212 B** of NPZ/bin evidence. Automatic
stage manifests preserve hashes and refuse storage below a 16 GiB free reserve. Nothing was deleted
or moved. Full scientific producer/runtime/input/seed/argv custody is in `BINDING.json`, launch and
stage receipts, `ENVIRONMENT.json`, and the final manifest.

A concurrent startup exposed a shared binding temporary-file race; one child stopped before fitting.
The corrected whole-stage lock prevents shared output/native-build races. The old source/binding and
failed log remain retained. Only `main` changed (independent AST verification); all 15 scientific
functions are unchanged. `SOURCE_TRANSITION.json` records why earlier valid fits remain usable.
Two final code review-tracker passes, Ruff and compilation passed. Reviewed source was frozen before
full fits and pricing. The shared index and unrelated dirty work remain outside this unit.

Serializer runs LAST with per-file post-edit hashes and the required tokens. If shared Git object
writes are denied, the verified bundle/patch is handed to MAIN; a fallback commit is not shared HEAD.
Final landing state is recorded separately in `ddm_ls2_20260911/LANDING_RECEIPT.json`.

<!-- # FORMALIZATION_PENDING: finite new-field probability measurement through existing TC1 log-loss instrument; no universal law or production receiver actuator claimed. -->

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER:** owner MAIN; consumer store `/Volumes/VertigoDataTier/pact/ddm_ls2_full_resolution_lane_probability_bound/handoff/landing/`; trigger verified final manifest/bundle and MAIN Git access; land the exact reviewed LS2 unit.
- **QUEUED-WITH-A-FIRE-ORDER:** owner MAIN; consumer store `/Volumes/VertigoDataTier/pact/ddm_ls2_full_resolution_lane_probability_bound/handoff/joint_native_timing/`; trigger MAIN harvests the positive net receipt and verifies current archive/field and timing contract; measure one native joint-correction feasibility implementation before considering a separately authorized recode.

## LIVE-HYPOTHESES

- Native incremental geometry and fused probability arithmetic could make the joint model's small
  positive gain affordable. The real rounded prior-preserving model pays in likelihood, but the
  narrow remaining timing budget has not been tested. This is the sole named measurement follow-on.
- Other probability architectures or positioned Lane carriers remain untested by these two finite
  boxes. LS1 identified context information outside these compact count-ratio parameterizations,
  which makes richer models plausible; their parameter and receiver costs remain unknown. No new
  price or future launch is assigned to them by this measurement.

## DEAD-ENDS

- The continuous boxes and tested rounded models as a 25,899 B solution: continuous numerical upper bounds are below 469 B and tested rounded net gains below 386 B. No exhaustive integer-rounding bound is claimed.
- Separate-feature gain described as Lane improvement: its Lane contribution is negative 88.535929 B.
- Small coefficient size as timing evidence: decoder work and TC4's actual timeout refute that shortcut.
- Summing context gains, transferring PC2 pose prices, or inheriting changed-code timing: wrong object
  or unsupported composition. This measurement uses the actual joint model and current source boundary.

composition S 0.1372449041713402 @ 180,406 B [contest-CUDA T4 n600] (move 44)
