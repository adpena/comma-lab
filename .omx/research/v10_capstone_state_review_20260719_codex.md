# V10 capstone state review — canonical settlement, unmerged work, and launch debt

Date: 2026-07-19 UTC
Evidence snapshot: 2026-07-19T02:55:30Z
Role: Codex delegated fresh-eyes state review
Authority: read/derive/review only; no launch, paid dispatch, score, pointer, live-run, or MAIN mutation
Review branch: `codexwt/v10_capstone_state_review_20260719_20260719T023923Z`
Canonical pointer: `0.1910828242 [contest-CPU Linux x86_64]` **UNMOVED**

## 1. Executive digest

**Verdict: `BUILDABLE_RESEARCH_PROGRAM / NOT_LAUNCH_READY / NOT_SEALED`.** V10 is no longer merely a prose idea: MAIN contains an exact bounded-uint8 lattice primitive, a structural seven-section compiler/receiver, quotient-`T` custody, a cold/fork head split, and several measured factor probes. It still has no compact receiver-closed program, no complete n600 realization receipt, no measured shared-plane rate-distortion ladder, no production renderer/inflate path, no full interaction/KKT solution, and no exact contest-CPU/CUDA row. The pointer therefore does not move. Evidence: `.omx/research/inverse_solve_completeness_matrix_20260718.md`; `.omx/research/v10_compiler_receiver_fresh_eyes_20260718.md`; `.omx/research/v10_lattice_rate_verdict_and_composition_20260719.md`.

The strongest new result is **MEASURED** `[macOS-CPU advisory n6]`: one exact uint8 realization of the shared scorer plane produced `d_seg=0.0` and mean full-DistortionNet `d_pose=9.3e-10` on pairs `{90,175,277,381,424,573}`. This answers the narrow frame-1 Seg/Pose feasibility question and collapses the representation search to a compact description of shared `y_hat`; it does **not** answer rate, frame 0, n600, receiver, archive, or contest-axis custody. Directly storing arbitrary/minimum-norm solved frames is **DEAD** (verdict_scope: formulation — raw-frame-payload only; feasibility/MDL family OPEN): `1.70 MB/frame` Brotli-Q11, about `1.02 GB` for n600 and rate term about `680`. The feasibility/MDL family remains open. Evidence: `.omx/research/v10_lattice_rate_verdict_and_composition_20260719.md`; `.omx/research/v10_uint8_lattice_feasibility_receipt_20260718.{md,json}`.

The live n600 lattice campaign is **INCOMPLETE**. At the evidence snapshot it had eight sealed receipts covering pairs `0..95`: `56,623,104/56,623,104` exact blocks, zero budget/heuristic blocks, `17/18,874,368` frozen-SegNet mismatches (`d_seg=9.006924099392361e-7`). Chunk 08 (`96..107`) was in flight. A transient resume refusal at chunk 06 was recovered into a sealed receipt; it is not a current blocker and must not be confused with full-n600 completion. Evidence: `/Volumes/VertigoDataTier/pact/evidence/v10_uint8_lattice_n600_20260719/chunks/receipt_chunk_00.json` through `receipt_chunk_07.json`; `chunked_run.log`.

The required SSoT is itself genuinely unmerged: branch `claude/p0_521_spec_v10_capstone_20260717` at `9495e2fe95` adds the 971-line SPEC and its old blocker-skeleton DSL, neither present on MAIN. Meanwhile MAIN has superseding implementation work, especially structural compiler commit `c2f866da8f`, lattice harvest `fae46ae58a`, and equation ratification `bce6010c17`. MAIN must reconcile the SSoT with those later facts rather than merging the branch mechanically. Evidence: `claude/p0_521_spec_v10_capstone_20260717:.omx/research/SPEC_v10_capstone_cold_start_seeded_20260717.md`; `src/tac/witness_dsl/v10_compiler_receiver.py`; git commits named above.

The shortest honest path is: finish/receipt n600 lattice -> measure compact `y_hat` R-D rows -> declare deterministic float32 receiver arithmetic -> build production renderer/archive receiver -> run coherent train-side/constrained phase and P1/P2/P3 probes -> measure residual pool/channel curves -> solve the revised shared-fidelity KKT -> prove per-stage resumability/parse-back -> exact same-byte contest CPU/CUDA replay -> operator GO. Paid Modal authority remains a later, explicit operator decision, capped at `<= $20` under Task `#381`; this review grants none.

## 2. Vehicle and authority lock

| name | settled meaning | state at this review |
|---|---|---|
| v7.5 / v8 | historical parent contracts; their operating contract remains binding | not the current run |
| v9.CGauge `v9c2` | warm-started development vehicle using a v7.5-donor trunk | terminal/dead as a training run; banked ep725 best retained at advisory n600 `d_seg=0.003458` |
| `v9c3` | warm fixed-events development successor | development only; no V10 cold-start equivalence |
| v10 | cold-start, fully seeded capstone child; no warm weights from v9c2 | buildable research program, never launched |

Sources: `.omx/research/vehicle_naming_resolution_v10_capstone_20260717.md`; `.omx/research/phase_stack_efficacy_probe_v10_gate_20260718.md`; `claude/p0_521_spec_v10_capstone_20260717:.omx/research/SPEC_v10_capstone_cold_start_seeded_20260717.md` §§1, 14.3.

The borrowed `0.1880443979880752 [contest-CPU]` snapshot named in the factor matrix is non-submission comparison custody, not the SUBMITTABLE pointer and not a V10 input. The canonical SUBMITTABLE pointer remains `0.1910828242`. Source: `.omx/research/inverse_solve_completeness_matrix_20260718.md`.

## 3. SETTLED / MEASURED facts

Labels here are strict: **MEASURED** is receipt-backed observation; **DERIVED** is algebra/source consequence; **INFERRED** is a scoped mechanistic reading still requiring the named test.

| label | settled fact | exact value / scope | consequence |
|---|---|---|---|
| MEASURED | measured generator-chain description ratio | `K/H=0.47` for the tested generator-chain versus entropy-chain codec pair | useful rate prior, **not** Kolmogorov complexity or global optimality; use `L_G/H_chain=0.47` scope |
| MEASURED | SegNet centered final head | rank `4` | exact final-head quotient is four-dimensional; nonlinear upstream preimage remains nonconvex |
| MEASURED | static hood geometry | IoU `0.993` hood and `0.976` sky | strong structured seed; not receiver or rate closure |
| MEASURED | hood texture | `d_seg 0.04538 -> 0.01328` for `1,759 B` | texture is necessary in at least this measured stratum |
| MEASURED | flat versus textured realization | `0.0416` versus `0.0048`, an `8.7x` gap | generic `GEOM-ONLY`/flat-cell theorem is falsified at generic-formulation scope; explicit quotient residual `T` is required |
| MEASURED | shared-resize blind rows | `22.6969%` exact blind-row fraction | real null coordinates exist; not automatically free under a concrete uint8/archive grammar |
| MEASURED | rendered energy in `ker(A)` | about `52.4%..52.9%` | capacity-allocation signal, not blob-rate savings by itself |
| MEASURED | gauge burden | `52.36%` head-weight norm, `69.34%` bias, `97.27%` palette norm | gauge fixing is a precision lever |
| MEASURED | gauge quantization | int8 scale `0.028416 -> 0.022088` (`22.3%` finer); score-relevant quantization error `25.0%` lower | rate-neutral on fixed-shape int8; n600 sign confirmation still owed |
| MEASURED | palette gauge error | about `3.3x` inflation | canonicalize before quantization, then measure through receiver |
| DERIVED | frame-0 Seg obligation | exactly zero; Pose still consumes frame 0 | spend frame-0 freedom only under a joint Pose/rate guard |
| MEASURED | camera support | strict support `1.66%` | local camera actuation is sparse but nonlinear scorer coupling remains |
| MEASURED | saddle fragility | `29.2%` of sampled saddle flips are sub-LSB | precision/birth/phase surfaces remain load-bearing |
| MEASURED | phase carrier rate | `29,958 B` excluding `xi`; `11.3x` below `338,523 B` raster but `16.6x` above the `0.9..1.8 KB` anchor budget | STORE primitive is real; adoption not earned |
| MEASURED | factor-2 n6 affine solve | all `3,538,944` blocks exact, max float `A` residual `8.526512829121202e-14`, zero Seg mismatches | bounded Diophantine law is valid for the measured canonical factor |
| MEASURED | n6 clip/round comparator | `520` mismatches, `d_seg=0.00044080946180555556`, max `A` discrepancy `63.824981689453125` | `clip(round(real solve))` is not a realization solver |
| MEASURED | n6 exact Seg/Pose composition | `d_seg=0.0`; mean `d_pose=9.3e-10`; Pose score contribution about `0.0001` | one frame-1 shared-plane realization satisfies both scorers on n6 |
| DERIVED | shared-plane composition | payload should describe compact `y_hat` at `384x512`; the exact lattice solve is generic/free decode work | rate search moves from solved RGB frames to a compact shared sufficient statistic |
| MEASURED | raw solved-frame rate | n6 sidecar `11,346,894 B`; direct frames about `1.70 MB/frame`, n600 about `1.02 GB`, rate term about `680` | `DEAD` (verdict_scope: formulation — `RAW_ARBITRARY_OR_MINIMUM_NORM_SOLVED_FRAME_PAYLOAD`); MDL-steered feasible points remain OPEN |

Primary sources: `claude/p0_521_spec_v10_capstone_20260717:.omx/research/SPEC_v10_capstone_cold_start_seeded_20260717.md` §§2–4, 13–14; `.omx/research/sol_ultra_v10_true_final_form_review_20260717.md`; `.omx/research/alternative_forms_conv_wall_20260718.md`; `.omx/research/null_subspace_rate_measure_20260717.md`; `.omx/research/v10_uint8_lattice_feasibility_receipt_20260718.{md,json}`; `.omx/research/v10_lattice_rate_verdict_and_composition_20260719.md`.

### Current n600 lattice tally

All rows are `[macOS-CPU advisory subset]`, non-promotable. Every sealed chunk has `7,077,888` exact blocks, zero budget blocks, and zero heuristic blocks.

| sealed receipt | pairs | exact-candidate mismatches | `d_seg` | clip/round mismatches | runtime s |
|---|---:|---:|---:|---:|---:|
| `receipt_chunk_00.json` | 0–11 | 1 | `4.238552517361111e-7` | 386 | `170.57295879209414` |
| `receipt_chunk_01.json` | 12–23 | 2 | `8.477105034722222e-7` | 238 | `169.38525508414023` |
| `receipt_chunk_02.json` | 24–35 | 4 | `1.6954210069444444e-6` | 228 | `167.18821112508886` |
| `receipt_chunk_03.json` | 36–47 | 0 | `0.0` | 262 | `167.7221590001136` |
| `receipt_chunk_04.json` | 48–59 | 0 | `0.0` | 399 | `167.4271731248591` |
| `receipt_chunk_05.json` | 60–71 | 4 | `1.6954210069444444e-6` | 542 | `168.56721391691826` |
| `receipt_chunk_06.json` | 72–83 | 3 | `1.2715657552083333e-6` | 715 | `168.663386750035` |
| `receipt_chunk_07.json` | 84–95 | 3 | `1.2715657552083333e-6` | 899 | `169.4326372500509` |
| sealed aggregate | 96 pairs | 17 / 18,874,368 | `9.006924099392361e-7` | 3,669 (`0.00019439061482747397`) | `1,348.9589950433` |

Chunk 08 (`96..107`) had begun but had no receipt at the snapshot. The table is therefore a sealed-prefix tally, not an n600 verdict. Sources: `/Volumes/VertigoDataTier/pact/evidence/v10_uint8_lattice_n600_20260719/`.

The Yousfi c2 reducibility lane is also in flight. Its durable directory exists and its log had rendered `576/600` pairs at the evidence snapshot; no terminal reducibility receipt existed. Sources: `/Users/adpena/Projects/pact/.omx/research/yousfi_c2_reducibility_n600_20260719/render.log`; sibling `partition.py` and checkpoint arrays.

## 4. BUILT, PARTIAL, AND BUILT-UNMERGED map

### What MAIN already owns

- **Factor 2:** exact bounded uint8 lattice code and tests landed via `fae46ae58a`; the factor-2 law was ratified in the canonical equation registry via `bce6010c17`. Sources: `src/tac/optimization/uint8_lattice_feasibility.py`; `src/tac/tests/test_uint8_lattice_feasibility.py`; `.omx/state/canonical_equations_registry.jsonl`.
- **Structural compiler/receiver:** commit `c2f866da8f` emits seven counted sections consumed exactly once: `CountedGenerator`, `Frame0PoseSixCarrier`, `InitHeadSolve`, `SharedResizePreimage`, `RgbYuv6Projection`, `BlindFillRateGrammar`, and `QuotientResidualT`. It passed 60 local structural tests and keeps `launch_ready=false`. It is not a production renderer, inflate archive, or score result. Sources: `src/tac/witness_dsl/v10_compiler_receiver.py`; `.omx/research/v10_compiler_receiver_fresh_eyes_20260718.md`.
- **SOL fixes `#528..#532`:** pose-marginal composition refusal (`3de9067479`), cold/fork split plus quotient `T` in `c2f866da8f`, and structured diagnostic scaling/role tests (`eb3d8da627`). These close named implementation defects, not V10 launch readiness.
- **Power-diagram containment:** source/tools/findings are on MAIN; exact prefix measurement is preserved, while the historical mutable executable is tombstoned. Source: `.omx/research/v10_power_diagram_byteclose_findings_20260718.md`.
- **Other harvested research payloads:** alternative-forms, completeness-coupling, constructive inverse-solve, power-diagram witness, MDL lower-bound, pool/channel harness, and campaign review payloads are byte-identical on MAIN even though their branch tips are not ancestors. Their divergent registry/audit/DAG tails are not evidence that the scientific payload is missing.

### Branch custody

| branch / head | relationship to MAIN | payload disposition | MAIN action |
|---|---|---|---|
| `claude/p0_521_spec_v10_capstone_20260717` / `9495e2fe95` | not ancestor | **genuinely unmerged** SPEC and old DSL skeleton; both absent on MAIN | reconcile SPEC with later compiler/lattice/phase facts; do not mechanically land stale DSL |
| `claude/p0_v10_buildable_components_20260717` / `35dc3b70f3` | not ancestor | memo, `range_a_projection.py`, `content_priced_coder.py`, structured-init defaults/tests absent; trainer and curriculum files diverged | review primitives individually; port semantic payload through current DSL/compiler, not whole-branch merge |
| `claude/p0_518_resume_warmup_geometry_20260717` / `ead2a13760` | not ancestor | warmup equation/tests absent; trainer, DSL, launcher, registry diverged | keep fork/resume-only semantics out of cold birth; rebase/port after exact 8-vs-27 A/B and current resume schema review |
| `codexwt/v10_uint8_lattice_...` / `28f0b5e464` | not ancestor | core code, receipts, specs, tests identical on MAIN; completeness/registry tails diverged | branch is harvested/superseded; no raw merge |
| `codexwt/v10_power_diagram_byteclose_...` / `45cef2d010` | not ancestor | core tools/final receipts/findings identical on MAIN; preseal evidence and storage-plan files remain branch-only | retain as forensic custody; MAIN review whether branch-only preseal evidence merits archival landing |
| `codexwt/v10_kkt_waterfill_...` / `8f3dfb9813` | not ancestor | blocked receipt/review/DAG identical on MAIN; equation-candidate row branch-only | preserve blocked verdict; do not register equation without scientific rows |
| `codexwt/v10_compiler_receiver_...` / `4bb2efb7e0` | not ancestor | compiler, tests, memos byte-identical on MAIN | harvested; branch itself need not merge |
| `codexwt/v10_A2_profiler_...` / `f3a1e678f2` | ancestor of MAIN | fully landed | no action |
| `codexwt/sol_ultra_v10_true_final_form_...` / `f3366e3a2f` | ancestor of MAIN | fully landed; three older duplicate branch labels still point to `6427ab9ede` | preserve latest; retire duplicate labels only under separate branch-hygiene authority |
| campaign / alternative / completeness / constructive / power-witness / MDL / pool-channel branches | not ancestors | scientific payloads identical on MAIN; branch registry/audit or shared-DAG tails diverge | treat as harvested; reconcile only specific shared-state deltas after ownership review |

The branch classification above compares each branch's changes from its merge base against the MAIN blob for every changed path. It does not infer merge status from branch ancestry alone.

### Structural forms that are real but non-authorizing

- **Alternative forms:** exact rank-4 head is useful, but the frozen feature extractor still contains `68` SiLU modules and `10` ReLU modules. The Gibbs/logsumexp target measured `2.5133 s` for n600, mean target top-1 probability `0.9958448`, and `2,629` fp16-logit/cache argmax differences (`2.2286e-5`); no RGB preimage was built. Cole-Hopf remains a candidate, not an executed cure. Source: `.omx/research/alternative_forms_conv_wall_20260718.md`.
- **Power diagram:** frames `0..194` produce feature-pullback `d_seg=0.023459097055288463`; generator bytes are `314` raw / `306` Brotli / `257` optimistic ideal. This is **NON-EQUIVALENT** because the spatial quotient field, RGB inverse, uint8 receiver, Pose, and archive are absent. At frame 195, CPU-Torch/native-f32 chooses class 0 with class-0/class-1 logit margin `4.76837158203125e-7`; generic float64 instead puts class 1 over class 0 by `2.5277826765e-7`, while native float32 arithmetic ties and selects class 0. Verdict scope: `SERIALIZED_RECEIVER_ARITHMETIC_FORMULATION/INSTANCE`; power-diagram family remains open. Sources: `.omx/research/v10_power_diagram_byteclose_findings_20260718.md`; `.omx/research/v10_power_diagram_frame195_diagnostic_20260718.json`.
- **Factor 10:** preflight admitted 25 GiB but executed zero scorer cells; `0/48` scientific cells and null operating point. The harness lacks exact skip/deep and range/kernel interventions, byte-closed curves, a Pose/KKT consumer, and stable resume authority. Source: `.omx/research/factor10_kkt_waterfill_blocked_receipt_20260718.json`.

## 5. OPEN / OWED, deduplicated

The following list merges SPEC §6, §14.5, §14.12, the 11-leaf matrix, five lattice receipt blockers, Tasks `#535/#536/#539/#540/#541/#543/#547/#548`, and P1/P2/P3. Each item appears once at its strongest gate.

The factor-2 receipt's five blockers resolve as follows: **(1)** full n600 is still open (`#547`); **(2)** frame-1 Pose interaction is answered only on n6, while n600 and any compact-frame-0 interaction remain open; **(3)** complete receiver/archive custody is open (`#543`); **(4)** contest-CPU and contest-CUDA exact replay are open; **(5)** independent MAIN review is open. Source: `.omx/research/v10_uint8_lattice_feasibility_receipt_20260718.json`; updated disposition in `.omx/research/v10_lattice_rate_verdict_and_composition_20260719.md`.

### BUILD

1. **Merge/rewrite the SSoT.** Land a reviewed new SPEC version on MAIN that consumes the SOL corrections, exact lattice/pose result, one-shared-plane composition, structural compiler v2, power-diagram blocker, phase-posthoc negative, and updated manifest. Concrete source: `claude/p0_521_spec_v10_capstone_20260717:.omx/research/SPEC_v10_capstone_cold_start_seeded_20260717.md`.
2. **Compact `y_hat` descriptor and decoder.** Build the actual video-derived representation for the shared `384x512` scorer plane; exact lattice solve stays generic receiver work. Owner: Task `#548`. Gate: full counted bytes, deterministic decode, reconstructed `y_hat`, and both scorer responses.
3. **Production receiver/archive.** Replace structural instruction semantics with a renderer/inflate implementation that consumes every counted section, reconstructs both uint8 frames, declares deterministic float32 arithmetic/tie policy, and emits parse-back-stable bytes. Owner: `#543`. Structural `launch_ready=false` is binding.
4. **Factor-2 production integration.** Bind `bounded_uint8_resize_preimage_cell_feasibility_v1` into the production receiver; the structural compiler currently has no factor-2 section/receipt.
5. **Factor-specific residuals.** Port/review the range(A), content-priced coder, and structured-init primitives from `claude/p0_v10_buildable_components_20260717`; preserve explicit unique-home `T` so the residual cannot relearn or double-pay solved terms.
6. **Coherent phase actuator.** Build the through-R coherent object-level phase carrier or constrained joint solve and the train-side joint-loss A/B. The post-hoc per-pixel amplitude formulation is closed negative, not the family. Source: `.omx/research/phase_stack_efficacy_probe_v10_gate_20260718.md`.
7. **Birth-before-phase stage.** Implement the island-birth source and its stage boundary before phase conditioning, with complete resume state. SPEC §14.2 remains an ordering hypothesis, not a measured efficacy result.
8. **Full resumability.** Task `#537`: atomic per-stage checkpoints, all preserved; optimizer, EMA, controller modes/buffers/latches, event/ramp position, authoritative d_pose source/epoch, and interrupted-versus-continuous equality. The #518 branch is not sufficient for cold v10.
9. **Complete equation/DSL legs.** `#540` has ratified only factor 2. Rate-ladder, power-diagram, and revised KKT/Jacobian-Lagrangian laws still need real evaluators; the DSL must compile a production program rather than merely hold a structural reference.

### MEASURE

10. **Seal the n600 lattice replay.** Continue after the sealed `0..95` prefix, complete all 50 chunks, validate aggregate custody, and measure full DistortionNet plus any frame-0 substitution used by the compact program. Owner: `#547`. No partial tally is an n600 result.
11. **Measure the `y_hat` R-D ladder — COMPLETED at advisory n24 / production adoption OPEN.** Task `#548` landed `.omx/research/yhat_rd_ladder_20260719_codex.{md,json,csv}` with four actual-byte rungs, two disjoint real n12 receipts, both frozen-DistortionNet terms, reconstructed-plane error, source-frame0 policy, exact/repair counts, runtime custody, and the byte-closed ep725 interaction gap. All `56,623,104` rung-block solves were exact with zero repair/infeasible/error cells. The decisive measured separation is `83,838 B` for the full-n600 generator archive versus `42,051,900 B` for direct Brotli-Q11 descriptions of its n24 planes. `verdict_scope`: this closes the n24 measurement item only; compact frame0, production V10 receiver, full-n600 compact-ladder measurement, contest axes, and MAIN landing review remain open. Pointer unchanged.
12. **P1 terminal decomposition.** Per class/cell, remove or replace `T`; report geometry/texture/phase residual and full receiver effect. This decides which rows may honestly become `GEOM-ONLY`.
13. **P2 mirror transport.** Compare light-anchored mirror/specular transport bytes against residual hood-rim texture, including Pose effect.
14. **P3 chroma margin Jacobian.** Full n600 per-pair channel-plane sensitivity at the actual post-`A` YUV6 surface, including clamp/tie behavior.
15. **Phase efficacy.** Run coherent object-level and train-side/constrained arms through the real receiver. The existing n24 post-hoc oracle worsened `d_seg` by `+56.6%..+161.1%`; that is a formulation negative, not authority to delete phase training.
16. **Old findings and warm controls.** Finish p0_497 matched-byte Fourier/curvelet/shearlet/step-native receiver A/B; isolate cold-versus-warm Muon `#270`; measure #518 8-vs-27 warmup on fork semantics. These inform laws but cannot seed v10 with warm weights.
17. **Pool/channel and MDL measurements.** `#535/#536/#541`: populate the real path x resize x channel rows, intrinsic-complexity/K lower bounds, and residual bytes. Structural 48-row plans with `0` scientific cells do not authorize waterfill.
18. **Interactions.** For all eleven leaves, measure isolated and composed Seg/Pose/rate effects, resume custody, and adoption/exclusion. The current matrix has zero `COMPLETE` leaves.

### DERIVE

19. **Revised optimization object.** The exact lattice result couples Seg and Pose through one `y_hat`; derive waterfill over bytes allocated among `y_hat` representation components with joint distortions `(d_seg(y_hat), d_pose(y_hat, y0))`. This collapses the old three-independent-axis picture, but it does not eliminate the two distortion terms or rate.
20. **Receiver arithmetic law.** Specify exact CPU/native float32 ordering and tie behavior so frame-195-class ULP differences cannot change semantics across generator, serialized program, or evaluator.
21. **MDL form choice.** Derive the shortest admissible hybrid per factor—power diagram/head, channel/chroma, antialias/preimage, lattice, null-space fill, and residual `T`—from measured R-D rows. No codec-pair ratio proves global Kolmogorov optimality.
22. **Fresh-init laws.** Distinguish `InitHeadSolve` and cold EMA/optimizer initialization from `ForkHeadSolve`, LR rewarm, fork clearance, and restored moments. Only the former belongs at v10 birth.

### CONFIRM

23. **Receiver byte-close:** same counted program -> deterministic parse-back -> same uint8 frames; every paid byte has exactly one consumer and every video-derived input is counted.
24. **Runtime/storage:** decode under 30 minutes; storage waterfall, per-stage checkpoint preservation, automatic certified cleanup, source/runtime hashes, and reproducible command/env custody.
25. **Exact evaluator closure:** same final `archive.zip` bytes through `upstream/evaluate.py` on contest-CPU Linux x86_64 **and** contest-CUDA; record both axes without inferred equivalence.
26. **Score/pointer:** only a receiver-closed exact row may be compared against `0.1910828242`; the provisional `0.118` floor and the `0.1880443979880752` borrowed snapshot have no pointer authority.

### SEAL

V10 may be called launch-ready only when: the reconciled SPEC is on MAIN; every 11-leaf manifest row is derived+built+compiled+consumed+resume-certified+measured+interaction-audited+adopted/scoped-excluded; compiler `launch_ready=true`; all §6/§14.12 gates validate receipt bytes rather than presence; n600 and production receiver receipts are complete; and operator GO is explicit. Any paid Modal execution additionally requires the governed launcher, lane claim, explicit operator authorization, and `<= $20` cap under `#381`.

## 6. Contradictions and risk register

| risk / contradiction | adjudication |
|---|---|
| SSoT branch is unmerged while MAIN has later implementation truth | write a reconciled successor; never merge the old DSL skeleton as launch authority |
| SSoT's early stored-pose/three-axis framing versus one shared exact `y_hat` solve | shared frame-1 plane is the representation axis; score still has Seg and Pose distortions, plus frame-0 and rate |
| completeness matrix says factor 2 is unconsumed/no equation while MAIN later ratified its law | equation existence is settled; production receiver consumption and R-D/adoption remain open |
| Task `#543` is pending while a structural receiver exists | no conflict if scope is explicit: structural reference built; production inflate/renderer/evaluator receiver pending |
| structural compiler routes factors `1,3a,3b,4,5,6,7,8,9` but no factor 2 or 10 section | add factor-2 consumer receipt and a separate factor-10 decision receipt; never invent counted sections for equations |
| power diagram reports `306 B` while factor 6 is open | bytes cover only affine generator parameters; missing spatial field and receiver make it non-equivalent |
| exact rational `A` equality still yields rare Seg mismatches | fp32 network arithmetic/ties, not lattice failure; preserve as measured noise until arithmetic semantics are sealed |
| post-hoc phase is strongly harmful while phase headroom is large | negative scope is summed per-pixel amplitude edit only; coherent/train-side/constrained formulations remain open |
| `K/H=0.47` called Kolmogorov-optimal | relabel as measured codec-pair description ratio; global optimality unproved |
| direct solved-frame payload called DEAD | verdict_scope: formulation — raw arbitrary/minimum-norm frame storage only; compact `y_hat`/MDL feasibility family remains open |
| #518 says 27-epoch law is built and merge-ready | it is fork-specific, its coefficient is provisional, 8-vs-27 efficacy is unmeasured, and its hot files diverge from MAIN |
| n600 driver transiently refused a chunk-06 resume, then produced sealed chunks 06/07 | fail-closed recovery worked, but only the sealed prefix is evidence; full aggregate authority remains open |
| Yousfi directory was initially absent, then appeared and reached 576/600 during review | treat as live/incomplete; no terminal claim may be inferred from directory existence |
| Claude Tasks `#528..#548` are absent from `.omx/state/canonical_task_status.jsonl` at review time | violates the shared-status mirror contract; actual task JSON and P0 prose can drift from dashboards until mirrored |
| operator P0 v10 row still describes the early 357-line/13-blocker state | stale relative to 971-line SSoT, structural compiler, factor-2 law, and lattice/Pose result; refresh after MAIN review |
| task labels `completed` can overstate scientific closure | `#528..#532` implementation defects may be complete while launch, production, interaction, rate, and axis certificates remain open |

Task-state sources: `/Users/adpena/.claude/tasks/89ff112f-013d-43b5-b949-2a6d43b650c3/{521,528,529,530,531,532,535,536,539,540,541,543,547,548}.json`; `.omx/state/canonical_task_status.jsonl`; `.omx/state/operator_p0_ledger.jsonl`.

## 7. Immediate dependency path

| order | verb | concrete owner / artifact | pass condition |
|---:|---|---|---|
| 1 | CONFIRM | `#547`, `/Volumes/VertigoDataTier/pact/evidence/v10_uint8_lattice_n600_20260719/` | all 50 chunk receipts plus aggregate; full scorer/frame interaction; no unsealed state |
| 2 | MEASURE | `#548`, new `y_hat` R-D receipt | at least three compact byte rungs with both distortions, exact decode, and runtime custody |
| 3 | DERIVE | `#540`, equation registry + reconciled SPEC | shared-fidelity KKT and float32 receiver law have executable evaluators and honest evidence status |
| 4 | BUILD | `#543`, `src/tac/witness_dsl/v10_compiler_receiver.py` successor + inflate renderer | production program parses, consumes factor 2, emits deterministic uint8 pairs, and stays `launch_ready=false` until gates close |
| 5 | MEASURE | P1/P2/P3, coherent phase, p0_497, #270, #518 A/B | named formulation decisions have isolated through-R Seg/Pose/byte receipts |
| 6 | MEASURE/DERIVE | `#535/#536/#541`, pool-channel/MDL/KKT | scientific rows populated; operating point non-null; non-additive interactions represented |
| 7 | CONFIRM | `#537`, receiver/archive gate | continuous==resume, all stage checkpoints preserved, parse-back byte-close, cleanup certified |
| 8 | CONFIRM | governed exact evaluator | same archive passes contest-CPU and contest-CUDA, decode budget, compliance, custody |
| 9 | SEAL | MAIN review + operator GO; paid route only via `#381 <=$20` | reviewed MAIN SSoT/compiler manifest green; explicit authority exists |

## 8. MAIN landing review required

This memo is the only deliverable from the isolated review branch. MAIN must independently review:

1. every numeric label and verdict scope, especially n6 versus n600, sealed receipt versus checkpoint state, and raw-frame DEAD versus family OPEN;
2. branch custody classification, particularly the genuinely unmerged SPEC/buildable/#518 payloads and branch-only power preseal evidence;
3. the proposed shared-`y_hat` reformulation of factor 10;
4. the distinction between structural compiler/receiver and production archive receiver;
5. task/P0/canonical-mirror drift and the exact owner/status before changing shared ledgers;
6. whether to land this memo on MAIN. Its presence on this branch grants no launch, paid dispatch, score, pointer, equation promotion, or task completion authority.

## Stores consulted

`CLAUDE.md`; `AGENTS.md`; `PROGRAM.md`; V7.5 §8 and V8 SPEC; full V10 SPEC/compiler branch; naming resolution; SOL-ultra review; alternative-forms, completeness-coupling, lattice, power-diagram, compiler/receiver, phase, MDL, pool/channel, KKT, and campaign memos/receipts; exact upstream scorer factorization references; canonical equation/task/P0/DAG surfaces; all named branch diffs and worktree heads; live read-only lattice SSD receipts/checkpoints/log; live read-only Yousfi render log; delegation inbox and broadcast ledger through the final review snapshot.

**Final scoped verdict:** V10 has a credible exact-realization core and a coherent train-least architecture, but it is not a complete system, not a production receiver, not launch-ready, and not promotion-grade. The decisive remaining science is the compact shared-plane R-D curve and its production receiver; the decisive remaining engineering is n600/parse-back/resume/axis custody. Pointer delta: `0`.
