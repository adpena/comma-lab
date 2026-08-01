# Codex finding: functional-quotient compression needs a receding-horizon controller

Date: 2026-07-26  
Lane: `lane_original_taskspace_inverse_witness_codec_capstone_20260726`  
Scope: original Pact mechanisms only; external work supplies mechanism lessons, never payloads,
weights, archives, selectors, or video-derived constants.

## Verdict

The HOPE paper is useful, but its highest-value contribution here is not ordinary neural-network
pruning. It exposes a control-law gap in our current stack:

1. compare representation atoms by the *function they induce*, after quotienting parameterization
   and gauge symmetries;
2. allow atom pruning, atom merging, and whole-family eviction to compete on one surface; and
3. execute exactly one globally best current action, then rebuild the physical object and recompute
   every remaining action from the new state.

That is the missing bridge between our micro costates and the macro codec. Fixed independent
segment, pose, and rate targets are mathematically wrong because the acceptable value of each is
conditional on the other two through

```text
S(x) = 100*d_seg(x) + sqrt(10*d_pose(x)) + 25*archive_bytes(x)/37_545_489.
```

The same is true of action marginals: ZIP context, decoder placement, scorer-cell crossings, and
mutually exclusive transforms make an action's value state-dependent. No marginal survives a
committed state transition without remeasurement.

## Premise verification against the live code

`taskspace_whole_archive_allocator.allocate_taskspace_whole_archive` correctly rebuilds each
complete archive, selects the actual ZIP encoding, double-decodes, measures the nonlinear score,
and remeasures an accepted prefix. However, it consumes proposals in caller order and accepts every
negative exact delta. Therefore two improving but interacting proposals can reach different final
states when their input order changes. The allocator is exact *conditional on an arbitrary order*;
it is not yet a global receding-horizon action controller.

G19 already preserves nonadditivity and whole-object remeasurement, but its acquisition surface is
calibration over supplied effects with unit probe cost. It does not certify that the supplied set is
the complete current micro/macro action catalog, globally select one action, invalidate every stale
trial after the commit, and request a fresh catalog at the new exact base.

G25 closes lossless population-global same-state recoding and reveals a valuable physical
factorization: 64 latent-coordinate trajectories over the 1,200 chronological frames. It does not
yet perform a lossy functional merge, prune, or macro eviction. Those 64 trajectories are the first
concrete carrier family on which to test the functional-quotient controller.

## Exact Pact adaptation of the HOPE mechanism

The paper's functional identity and receding-horizon compression mechanism is described at
<https://arxiv.org/abs/2607.21366>. Pact has a stronger task-specific measure than the paper's
maximum-entropy surrogate because the contest information space is frozen and the encoder has
unlimited time:

```text
carrier c_i
  -> deterministic public receiver R
  -> realized uint8 pair video
  -> frozen evaluator sufficient statistics
  -> local/global score effect phi_i.
```

Define each carrier's identity by its realized evaluator-effect function `phi_i`, not by a latent
index, raw coefficient norm, tensor spelling, or pre-compression byte count. On one exact operating
point, construct the costate-weighted Gram matrix

```text
K_ij = <phi_i, phi_j>_Lambda,
```

where `Lambda` is built from actual Seg cell, Pose pair, and whole-object rate observations. The
finite score law remains verdict authority; `K` is only an encoder-side proposal generator.

Candidate operations are:

- **PRUNE:** project one low-capacity carrier to the decoder's lawful identity/zero path;
- **MERGE:** replace two near-collinear functional carriers with one fitted parent plus, only if
  valuable, a counted quotient residual;
- **MACRO_EVICT:** delete an entire representation pathway when a generic decode-time solve/repair
  path preserves its obligations;
- **MIGRATE:** move video-specific state from stored coefficients to a smaller counted sufficient
  statistic expanded by generic `inflate` machinery;
- **TRAIN_QUOTIENT:** train only the telemetry-proven residual outside the inverse-solvable span.

Every proposal must produce a complete archive and public decode receipt. The controller ranks
realized endpoint score, not the Gram proxy. Unlike HOPE's static parameter-yield denominator,
Pact must use the exact rebuilt ZIP size at every trial because compression interactions violate
item independence.

## Required controller contract

One controller step must consume:

- one exact base archive/output/measurement/runtime identity;
- the dynamic frontier pointer artifact and target score;
- a content-addressed action-universe closure proving the current scan includes both micro and
  macro actions;
- one complete rebuilt archive, public double-decode receipt, realized Seg/Pose measurement, exact
  ZIP size, payload-placement proof, and decode resource observation per admissible action; and
- explicit blockers for unmeasured or resource-infeasible actions.

It must then:

1. reject every trial bound to another base object;
2. reject component thresholds as an admission rule;
3. compute the exact nonlinear endpoint score for every admissible action;
4. select at most one globally lowest-score action with deterministic tie-breaking;
5. emit a per-component explanation and a whole-object finite action costate;
6. preserve macro and micro actions on the same comparison surface;
7. treat decode time, memory, determinism, recursive dependency closure, and hidden-data absence as
   hard constraints rather than score terms;
8. mark every nonselected trial stale immediately after commit; and
9. require the encoder to regenerate the full action universe at the committed exact base.

If the action universe is incomplete, the controller may prioritize acquisition but must not claim
global selection. If no exact action improves score, it returns a scoped local fixed-point verdict,
not a family-level negative.

## Encode/decode placement law

The terminal codec has two asymmetric machines:

- the encoder/compiler may use the source video, frozen scorer, unlimited search, inverse solves,
  functional Gram construction, joint descent, and exact archive remeasurement;
- `inflate` may run deterministic generic solvers, rasterizers, postfilters, optimizers, or a generic
  network/fitter within the official resource envelope.

All video-specific weights, coefficients, targets, selectors, thresholds, exceptions, programs, and
initialization state remain counted. A decode-time fitter is lawful only when every fitting target
descends from counted payload plus generic code and the observed execution graph proves no access to
the original video, scorer, teacher, GT cache, hidden prior output, ambient file, or network.

Compute is therefore a constrained free reservoir. The placement controller compares:

```text
STORE state
DERIVE state analytically
REPAIR a coarse realization
FIT from counted constraints at decode time
```

by realized score and exact archive bytes, subject to hard runtime/memory/determinism gates. Generic
decoder source size is a guardrail and custody surface, not a contest-rate charge.

## First production probe

After the G25 public LVPG2 receiver and G28 same-object score close, the first non-toy probe should
operate on the real n600 `code_quantized.reshape(600, 2, 32)` state:

1. obtain full-population carrier effect vectors through the actual receiver/R/scorer path or exact
   sensitivity sidecars;
2. build the 64-carrier functional Gram and identify the best prune, merge, and macro-evict actions;
3. fit physical parents at encode time, quantize them, and materialize complete archives;
4. public-double-decode and exact-measure every action from the same base;
5. globally choose one action, checkpoint it, invalidate the other trials, and repeat; and
6. hand the surviving irreducible quotient—not the full original state—to terminal joint descent.

The probe must be resumable per horizon, preserve every committed horizon checkpoint, spill large
raw/scorer artifacts to the SSD tier, and certify cleanup. No action is admitted from a proxy loss.

## Falsifiers and stop conditions

- Functional Gram rank does not predict exact whole-object score ordering beyond chance.
- Candidate merges save no exact post-compression bytes after parent/residual metadata.
- Receding-horizon recomputation repeatedly reverses actions, indicating an unstable proposal
  generator or insufficient trust region.
- Generic decode repair exceeds the single official runtime/memory envelope or is nondeterministic.
- CPU and CUDA axes disagree enough that one shared counted statistic is not robust.
- The apparent quotient requires video-specific state hidden in free decoder code.
- Full-population effects reveal no low-rank carrier structure; the verdict is scoped to this EP725
  carrier representation, not to functional-quotient compression generally.

## Pointer delta and triality

Pointer delta: none; this is a controller/representation finding, not a score.

- DSL: complete current action universe; PRUNE/MERGE/MACRO_EVICT/MIGRATE/TRAIN_QUOTIENT; one commit
  per horizon.
- DAG: exact base -> functional proposals -> complete archives -> public double decode -> exact
  nonlinear score -> select one -> checkpoint -> regenerate.
- Equations: evaluator-induced Gram for proposal generation; exact `S` for verdict; hard constrained
  decoder resources; no independent component thresholds.

