# ADVISORY — v8 per-class/per-edge vehicle audit — 2026-07-10

`research_only=true`

**Disposition:** **HOLD any v8 increment-1a training EVENT.** Permit only blocker-closing builds,
receipt hardening, and read-only measurements. v8 contains useful L1 components and a serious design
hypothesis; it is not yet a launchable vehicle.

This is an advisory means artifact. No training, dispatch, pointer move, or code mutation was
performed.

## Answer first

v8 correctly attacks the deepest single-trunk failure mode: class competition through shared
parameters can erase a rare class even when aggregate loss improves. Its proposed cure—separate
class/edge carriers with explicit reconciliation—is plausible. The current apparatus cannot test or
ship that cure. The pre-registered inequality is reversed, the verdict accepts naked d_seg numbers,
the architecture alternates between E edge fields and K class fields without an integrability rule,
and the “byte-closed” screen is only a boolean assertion over an in-memory GT-derived analytic
construction.

The current analytic row is useful solely as a high harness floor. It is not the increment-1a A/B,
not byte-closed, and not evidence for or against trained decoupling.

## P0 findings

### P0-1 — the pre-registered kill inequality is reversed

Lower d_seg is better. The evaluator correctly defines
`improvement = control.agg_dseg - decoupled.agg_dseg` and confirms when that quantity exceeds the
floor (`src/tac/inc1a_harness/decoupling_screen.py:222-231,297-308`). The sealed/configured contract
instead says:

```text
decoupled_mask_dseg > control_mask_dseg + delta_mask
```

at:

- `.omx/research/t5_crucible3/SPEC_v8.1_20260709.md:99`
- `.omx/research/t5_crucible3/SYNTHESIS_v3_v8_20260709.md:295`
- `src/tac/witness_autoconfig.py:3485-3491`
- `src/tac/tests/test_crucible3_v8_inc1a_config.py:151-155`

That predicate confirms the worse arm. The focused suite passes because the config test preserves
the same reversed sign, and `Inc1aScreenConfig.validate()` does not derive or check the semantic
predicate.

**Required gate:** one callable verdict owns the inequality. The spec/config manifest serializes a
stable predicate ID, not handwritten algebra. Add metamorphic cases with an obvious winner and
loser, and refuse any manifest whose predicate ID/hash differs.

### P0-2 — the kill gate cannot establish a matched-compute experiment

`ControlArmSpec` describes parameter matching, same seed, same epochs, and same curriculum at
`src/tac/inc1a_harness/decoupling_screen.py:65-119`. `ArmResult` carries only a name, frame count,
aggregate/per-class d_seg, and toy bit (`:174-193`). `evaluate_kill` checks only toy status and n600
(`:274-295`).

It does not verify:

- parameter/FLOP/optimizer-state equivalence;
- paired seeds, frame IDs, or curriculum;
- config, checkpoint, GT-cache, and source hashes;
- carrier serialization, parse-back, or decoded payload identity;
- byte closure or archive hash.

Arbitrary naked d_seg numbers can therefore receive a confirming verdict.

**Required gate:** a receipt-backed arm manifest containing all of the above, plus an exact decoded
partition hash. Any missing or mismatched receipt must return REFUSED before numerical comparison.

### P0-3 — declared byte closure is not executable byte closure

`Inc1aScreenConfig.measure_byte_closed_composite=True` at
`src/tac/witness_autoconfig.py:3447-3452`; validation merely checks that boolean at `:3300-3302`.
The actual analytic runner derives polynomial curves directly from full `L*` with `np.polyfit`
(`src/tac/inc1a_harness/analytic_smoke.py:44-109`), composes them in memory (`:166-210`), and emits
no quantized coefficient payload, archive section, decoder, or parse-back receipt.

The preserved run at `experiments/results/inc1a_killgate_20260710/` is honest about the rest:

- decoupled analytic analog d_seg = 0.10040343390570747 at n600;
- matched control arm absent;
- verdict `REFUSED-arm-missing-or-toy`;
- three identical reruns establish determinism, not training-seed variance.

This is a useful harness validation and floor. It is not a codec and cannot satisfy a
`measure_byte_closed_composite` requirement.

### P0-4 — edge-centric and class-centric architectures are unresolved

The base v8 spec first defines global class potentials `argmax_c phi_c`, then mandates one field per
RAG edge and calls a naive five-class-field build a violation
(`.omx/research/SPEC_v8_perclass_decomposition_20260708.md:13-27,101-106`). The actual assembler
accepts one `CarrierField.class_id` per class and rejects duplicate carriers for a class
(`src/tac/through_r/scaffold_assembler.py:62-78,104-148`). The next-build memo pins five independent
class heads (`.omx/research/v8_unlock_398a_20260710.md:45-47`).

Independent pairwise edge preferences `g_ij` admit a global class-potential argmax only when every
cycle is integrable:

```text
g_ij + g_jk + g_ki = 0
```

for every RAG cycle, up to a consistent gauge. No cycle-consistency penalty, potential
reconstruction, or alternative graph-labeling decoder currently exists.

**Required design decision:** either:

1. ship K class potentials and describe edge-centricity as loss/rate ownership; or
2. ship E edge fields with explicit cycle-integrability and a globally defined decoder.

The minimal probe computes fundamental-cycle holonomy, reconstructs least-squares class potentials,
and reports reconstruction residual plus partition drift.

## P1 findings

### P1-1 — five heads on a trainable shared trunk do not make theft impossible

The design requires `partial phi_c / partial theta_c' = 0` for c != c'. A trainable shared feature
trunk violates that condition even when each class has a distinct head: a loss step for c' changes
the common basis and therefore every phi_c. The current control language itself describes a common
trunk with different heads (`decoupling_screen.py:68-73`).

**Falsifier:** take one optimizer step using one class loss only and measure every other field before
and after, including all upstream trainable parameters. Either freeze a shared deterministic basis
and train isolated adapters, or build genuinely disjoint paths. Match the control on FLOPs,
activations, optimizer state, and paired seeds—not parameter count alone.

### P1-2 — aggregate d_seg is insufficient for the promised guardrails

Tie-flicker is config prose at `src/tac/witness_autoconfig.py:3145-3155`, and its test checks strings
at `src/tac/tests/test_crucible3_v8_inc1a_config.py:165-175`. `evaluate_kill` ignores the supplied
per-class d_seg. A candidate can pass on aggregate while damaging Lane, Movable, Road/Undrivable
topology, or temporal stability.

Confirmation must be conjunctive: paired aggregate gain, required Road/Undrivable gain, per-class
non-inferiority bounds, executable tie-flicker bound, and valid byte-closed receipts.

### P1-3 — several carrier byte claims are description lengths, not receiver sections

Horizon/lateral routines compress temporary arrays and report sizes without a serialized payload,
decoder, or parse-back proof (`src/tac/boundary_math/road_undriv_bulk_field.py:470-555,583-688`).
Movable accounting likewise reports bytes without a container/decoder
(`src/tac/boundary_math/movable_site_coder.py:240-311`). The geocoder measures original extracted
sites rather than decoded quantized bytes (`experiments/measure_v8_geocoder_close.py:226-245`), and
its through-R arm composes the oracle GT partition rather than the decoded site geometry (`:194-208`).

Texture “15.6 exact bytes” is also a description-length calculation: it prices 5-bit colors in
`src/tac/through_r/stem_perception.py:329-352`, while the planner retains and renderer consumes
floating colors (`src/tac/through_r/roadlane_texture_generator.py:292-333,353-387`).

Every carrier needs `encode -> exact bytes -> fresh-process decode -> render`, with headers and
quantizers counted and the decoded output used for the score measurement.

### P1-4 — class IDs are hard-coded in new components

`roadlane_texture_generator.py:70-76` and `movable_site_coder.py:56-72` reintroduce fixed label order,
despite the reusable self-detector at `movable_deshare.py:60-92`. Current-cache measurements remain
instance-valid, but the implementation is not permutation-safe.

**Falsifier:** randomly permute the five label IDs, run extraction/composition, inverse-permute, and
require identical payloads, geometry, and metrics.

### P1-5 — the rate ledger mixes Movable coder generations

The current component reports 6,289 B (`.omx/research/v8_geocoder_close_20260710.md:65-68`), while a
spec addendum pairs 6,289 B with score-rate 0.00344 (`SPEC_v8.1_20260709.md:421`). That rate belongs
to the older 5,161 B coder in
`src/tac/canonical_equations/v8_geometric_rate_decomposition_20260709.py:29-34,157-163`.

At the frozen 37,545,489-byte denominator, 6,289 B costs 0.00418759, a +0.00075109 correction. This
does not reverse the WASH verdict; it proves the macro row is not generated from one exact selected
payload manifest.

### P1-6 — the seed-spread rule is not a statistical decision rule

`operative_delta_mask` accepts an undefined scalar `seed_spread`
(`decoupling_screen.py:139-168`). It does not define range, standard deviation, SEM, confidence
bound, pairing, or decoupled-arm variance. Repeating a deterministic analytic program three times
and observing zero spread is not a substitute for training seeds.

Use paired per-seed differences, preregister the estimator and confidence interval, and confirm only
when the lower confidence bound exceeds a practical-effect floor. If the interval crosses the floor,
return INCONCLUSIVE. Three seeds are a smoke floor, not stable inference.

### P1-7 — “power diagram/Laguerre” is overclaimed for arbitrary fields

The assembler calls arbitrary scalar-field argmax a power diagram
(`src/tac/through_r/scaffold_assembler.py:2-20`). A classical power/Laguerre diagram is induced by
weighted-point squared power distances; see the
[CGAL regular-triangulation definition](https://doc.cgal.org/latest/Triangulation_2/index.html).
Arbitrary learned SDF/logit fields are a generalized tropical upper envelope. They need not inherit
convex-cell, weighted-site, or regular-triangulation properties.

Use “tropical class-potential argmax” generically and reserve “power diagram” for an explicit
weighted-site parameterization.

## Deep-math/additive directions

1. **Potential/edge Hodge decomposition.** Decompose learned edge preferences into an integrable
   gradient part plus cyclic residual. Charge or suppress the cyclic component; it cannot be
   represented by global class potentials without ambiguity.
2. **Frozen shared dictionary + disjoint modulations.** A counted/frozen basis with class-owned
   low-dimensional modulations can preserve exact cross-class isolation while exploiting common
   geometry. [COIN++](https://openreview.net/forum?id=NXB0rEM2Tq) and its
   [official MIT implementation](https://github.com/EmilienDupont/coinpp) provide a useful bounded
   quantized-modulation control, not a shipping dependency.
3. **Isolated SDF controls.** [SIREN](https://github.com/vsitzmann/siren) is a useful initialization
   and derivative-quality baseline for truly separate fields.
4. **PCGrad only as a control.** [PCGrad](https://proceedings.neurips.cc/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf)
   can reduce shared-trunk interference; it cannot establish zero cross-class derivatives.
5. **Lossless binary residual challenge.** An offline JBIG2-style generic-region/symbol/refinement
   baseline can test whether coordinate entropy is really the right conclusion. Any shipping use
   requires a separate runtime/license/compliance audit.

## Smallest convincing proof matrix

1. Kill-sign invariant across evaluator, spec-derived manifest, and config.
2. Receipt-backed matched-compute refusal tests.
3. Cross-class one-step/Jacobian isolation test.
4. Edge-cycle integrability or an explicit K-potential ownership decision.
5. Class-permutation metamorphic test.
6. Paired-seed aggregate, per-class, topology, and tie-flicker gate.
7. Exact per-carrier serialization and fresh-process parse-back.
8. Full decoded composite through R at n600.
9. Exact contest CPU and CUDA only after rows 1-8 pass.

## Scorer-derived additions — frozen evaluator/video pass

The measured video/scorer geometry supports edge-focused rate ownership, but makes exact independent
edge optimization less defensible.

1. **Cycles are present in the actual partition.** The frozen n600 RAG has at least nine observed
   unordered class adjacencies and 6,703 two-by-two triple-or-higher junction blocks. An E-edge
   representation must enforce `g_ij=-g_ji` plus zero cycle holonomy after quantize/parse-back, or
   reconstruct globally consistent K class potentials before argmax. The high-order junction count
   makes this a live decoder requirement, not an abstract corner case.
2. **Sparse boundaries do not imply local scorer ownership.** One-sided right/down boundary
   incidence is 1.2388865%, symmetric four-neighbor support is 2.1628333%, and scored-frame turnover
   is 1.24564%, so sparse carriers are rate-plausible. SegNet has source-level global paths; a
   recovered margin-Jacobian summary reports nonzero full-input support and remote tail energy, but
   its raw receipt is missing. Reproduce `J(edge e <- region e')` for same, adjacent, and remote
   regions and measure the nonlinear decoded composite. Edge fields are an empirical error
   representation, not a proven factorization of SegNet.
   The one-sided stencil is class-biased (MyCar 0.0201% versus 1.0266% symmetric), so size and gate
   undirected edge carriers from explicitly symmetric support, not the smaller forward statistic.
3. **Rare-class gates are mandatory.** Lane occupies only 0.58546% of cells and 74.31% of its cells
   have top-two margin below 1; Movable is 1.23793% with 17.69% below 1. Aggregate d_seg can hide
   exactly the theft v8 promises to eliminate. Confirmation must include per-class non-inferiority,
   Road/Undrivable movement, topology/junction drift, and tie-flicker.
   Of 1,619,917 unlike horizontal/vertical adjacencies, Road-Lane alone contributes 814,066
   (50.2536%) and all Road-incident edges contribute 93.3526%. Moreover 75.0433% of Lane cells lie
   on symmetric boundaries and 59.6014% of Lane labels turn over between consecutive 10-Hz scored
   frames. Prioritize explicit Road-Lane ownership and a Lane turnover/topology gate; do not infer
   scorer independence from that concentration.
4. **Frame0 is a Pose-only substrate.** SegNet selects frame1; PoseNet consumes both. v8 must not
   carry a frame0 Seg partition/texture merely because the renderer is symmetric. Frame0 should be
   reduced to a measured Pose carrier, while frame1 bears partition and texture obligations.
5. **Rate and Pose must be priced on exact decoded bytes.** One byte costs `6.6585895312e-7`; one
   corrected Seg cell buys 1.273108 bytes absent Pose. Pose has no fixed linear exchange rate because
   its score term is a square root. Each carrier receipt must report before/after
   `sqrt(10*d_pose)`, exact bytes, per-class Seg deltas, and the fresh-process receiver output.
6. **Carrier geometry should target scorer footprints, not all camera pixels.** The shared resize
   reads only 786,432 of 1,017,336 camera pixels; 230,904 per frame are exact joint-preprocess blind
   coordinates, and each scored RGB sample has a disjoint 2x2 camera footprint. This can simplify
   receiver realization, but does not excuse a short raw: require exactly 3,662,409,600 bytes and
   1,200 frames before any v8 verdict.

**Scorer-derived launch disposition:** **HOLD increment-1a unchanged.** The reversed kill predicate,
receipt-less matched-arm gate, declarative byte closure, and unresolved E-edge/K-potential decoder
remain exact blockers.

## Already settled — do not overread or reopen

- The analytic horizon-only harness runs deterministically at n600 and measures 0.100403; it is a
  high floor and harness receipt, not a trained-arm result.
- The naive lateral polynomial envelope was measured worse. That kills that formulation, not
  margin-aware lateral carriers or the v8 family.
- No matched trained control exists; the REFUSED verdict is the correct present verdict.
- The local frontier pointer remains unchanged.

## Stores consulted

- `.omx/research/SPEC_v8_perclass_decomposition_20260708.md`
- `.omx/research/t5_crucible3/SPEC_v8.1_20260709.md`
- `.omx/research/t5_crucible3/SYNTHESIS_v3_v8_20260709.md`
- `.omx/research/v8_unlock_398a_20260710.md`
- `.omx/research/v8_geocoder_close_20260710.md`
- `src/tac/witness_autoconfig.py`
- `src/tac/inc1a_harness/*`
- `src/tac/through_r/scaffold_assembler.py`
- v8 boundary-math carrier and geocoder modules
- `experiments/results/inc1a_killgate_20260710/*`
- `.omx/research/ADVISORY_evaluator_video_geometry_20260710.md`
- focused v8 config, harness, assembler, and carrier tests
- the primary sources linked above

**Pointer delta:** none. v8 remains L0/L1 and has no receiver-closed candidate.
