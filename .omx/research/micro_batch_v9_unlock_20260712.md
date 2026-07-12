# V9 CGauge MLX micro-batch unlock receipt

Date: 2026-07-12
Lane: `lane_micro_batch_v9_unlock_mlx_20260712`
Scope: BUILD + local verification only; no paid dispatch; live V9 process untouched
Verdict scope: implementation/config compatibility. The canonical V9 argv is cleared at B=2; runtime
functional-parity and throughput admission remain fail-closed until an in-process real-Metal/full-step
measurement executes on an uncontended device.

## Outcome

The typed `v9_cgauge_432` program now compiles with `MicroBatch(2)`, the real trainer parser accepts
the result, and the governed launcher dry-run validates all 199 emitted flags without spawning. The
two V9-active explicit refusals (phase advection and chroma boundary) have real batched twins, so their
guards were removed. The audit also found and routed semantics that had no refusal and therefore could
have been silently omitted: temporal screw, Chan-Vese area, live birth-completion logit/amplify/
persistence state, and the persistence skeleton cache. Analytic lane render-band already executes in
the shared render composition and is protected by a parity regression rather than duplicated as a
loss term.

This is not a score claim. The canonical contest-CPU pointer in `reports/latest.md` is **UNMOVED**.
Only an exact, byte-closed `upstream/evaluate.py` row can move it.

## Canonical argv derivation and complete refusal inventory

Authority was the typed compiler
`tac.witness_dsl.spec_v9_cgauge.compile_v9_cgauge_432_launch_config`, composed with DSL
`MicroBatch(2)`, then parsed with the real trainer parser. No hand-authored flag surrogate and no
`launch.sh` eyeballing controlled this classification.

The pre-landing trainer contained six actual `B>1` refusal predicates. Comments and argparse help
that merely contained the words "fails closed" are not counted as executable guards.

| Actual refusal predicate | Canonical V9 value | V9 blocker? | Landing |
|---|---:|---|---|
| margin-saliency reachability with positive margin-saliency weight | `False`, weight `0.0` | No | Guard retained |
| `--seg-spike-reweight` | `False` | No | Guard retained |
| `--seg-subpix-boundary-weight > 0` | `0.0` | No | Guard retained |
| `--seg-phase-advect-weight > 0` | `0.4` | **Yes** | Batched twin + provider stack + fused Metal phase map; guard removed |
| `--seg-chroma-boundary-weight > 0` | `0.1` | **Yes** | Batched twin + GT-chroma/annulus providers + fused Metal chroma map; guard removed |
| normalized StEik with `B>1` | `False` | No | Guard retained |

Compiler receipt after the build: 364 argv tokens including `MicroBatch(2)`; `program.validate() == []`;
the six predicates classify `{False, False, False, True, True, False}` in the table order. The DSL test
pins both the active and inactive sides so a future V9 edit cannot silently activate a scattered guard.
The other trainer grep hits are comments/argparse help for these predicates or already-routed #D15
surfaces, not additional executable refusal branches; the direct `micro_batch_pairs` condition scan is
also pinned in the audit receipt.

## V9-active semantic audit beyond explicit guards

| Active V9 surface | Pre-landing risk | Batched disposition |
|---|---|---|
| temporal screw `0.1` | Silently absent from the twin | One raw-witness frame-0 render per pair, one batched frame-0 SegNet call, batched homography warp, fused Metal residual map |
| lane band `1.0` + analytic render-band | Already in shared renderer | No duplicate term; B=2/B=4 composition parity test |
| area constraint | Silently absent | Per-pair soft-mass hinge vectorized across B, then mean over pairs; witness-alone logits retained |
| logit adjustment `1.0` + birth ramp | Static construction-time snapshot could go stale | Live offset cell reread each loss call |
| amplify/persistence birth ramp | Live per-class state absent | By-reference ramp state, disjoint masks, persistence recall scale, and shared skeleton cache |
| seed islands + witness-alone island loss | Previously routed by #313 | Preserved; dual batched co-gradient path unchanged |
| pose carrier + temporal screw | Tempting but wrong reuse of carrier frame 0 | PoseNet retains carrier-composed frame 0; temporal SegNet separately receives raw witness frame 0 |
| unify-tau, focal, boundary distance, eik-stab, weight entropy, AA/render composition | Already routed/shared | Preserved and covered by existing or combined-stack tests |

After routing, the only remaining executable micro-batch refusals are the four dormant V9 predicates:
margin reachability, spike reweight, subpixel boundary, and normalized StEik. There is no remaining
compiler-active refusal or identified silent active-V9 omission.

## New Metal paths

`src/tac/local_acceleration/metal_micro_batch_v9_levers.py` adds three real
`mx.fast.metal_kernel` forwards plus fused theta-bearing Metal VJPs, each vectorized over `(B,H,W)`
with no Python loop over B in the pixel math:

1. `micro_batch_v9_chroma_map`: BT.601 luma removal plus three-channel chroma squared residual.
2. `micro_batch_v9_phase_map`: positive margin, directional partner lookup, tie coordinate, and
   squared phase residual.
3. `micro_batch_v9_temporal_map`: selected-ground-channel temporal probability residual.

Pair-local weighted denominators are reduced outside the pixel kernels, then averaged across B;
there is no invalid global batch denominator. Chroma's frame VJP, phase's signed-margin gather VJP,
and temporal's `g1`/`g0_warped` VJPs are fused; gradients of static provider fields use safe generic
MLX expressions. The pure-MLX implementation remains the functional oracle and CPU/non-f32 fallback.
Shape/rank/channel guards refuse malformed buffers before raw Metal indexing.

The receipt state is lazy-execution honest: graph construction records `metal_planned`, and only
`verify_v9_lever_backend_execution()` evaluating the module-owned forward plus VJP arrays can promote
it to verified `metal`; compile/runtime failure records `metal_failed`. Temporal homography now uses
the existing cached `mx.compile` batch-native warp, and the two 3-of-5 softmax fields are packed in one
explicit contiguous allocation. Provider rows are host-packed from the trainer's legacy per-pair
store, but pixel math, scorer calls, warp math, and kernel dispatch do not loop over B.

A final carrier audit also closed a dormant accepted-mode defect in both serial and batched temporal
screw: `carrier_live` now always receives `model.code[2*pi]`. Table carriers ignore that argument;
FiLM carriers require it. A K=2 regression pins the accepted FiLM/live-xi composition.

## Functional-parity and admission receipts

The test/probe surfaces now cover:

- K=2 and K=4 loss/gradient parity for chroma, phase, temporal, and area;
- independent B=1 serial-form value/gradient oracles for each routed term;
- gate-off, zero-mask, missing-provider, and active-term sensitivity;
- live logit/amplify/persistence mutation without rebuilding `LeverConfig`;
- lane render composition and the combined V9 stack;
- B=2 carrier/raw-temporal authority separation for both SegNet and PoseNet;
- pure-MLX versus fused forward/all-primal VJP checks plus mandatory evaluated-Metal backend receipts;
- a B=2, 384x512 synthetic map diagnostic using typed V9 weights/classes and production value domains;
  temporal differentiates both theta-bearing inputs (`g1`, `g0_warped`).

Persisted functional/timing JSON is deliberately only schema-validated *reported telemetry*. It is
allowlisted to durable Pact result roots, rehashed/reparsed, checked against module-owned tolerances,
the canonical typed V9+`MicroBatch(K)` identity, and the fixed scorer hashes, but bytes cannot prove
that Metal or the frozen scorer executed. Consequently neither disk rows nor caller booleans/scalars
can set authoritative `functional_parity_passed` or training GO; both remain hard-false. The in-process
384x512 diagnostic is also explicitly synthetic/no-scorer/no-warp and cannot stand in for full V9.

Local verification completed:

- `102 passed` — typed DSL plus the new canonical-equation tests.
- `52 passed, 12 deselected` — the complete runtime-independent probe/admission-custody slice.
- `py_compile`, critical Ruff (`E9,F63,F7,F82`), and `git diff --check`: clean.
- Six `mx.fast.metal_kernel` forward/VJP objects construct under installed MLX 0.31.2; independent
  static/finitary audits found no formula, index, shape, dtype, or API blocker.
- Real compiled launcher dry-run: `expected-active-lever manifest: OK`, `199/199 flags`, DSL gate OK,
  exactly one `--micro-batch-pairs 2`, exit 0, and **NOT spawning**.

The machine-readable evidence is durable under
`experiments/results/v9_cgauge_micro_batch_dryrun_20260712/`: `launch.sh`
SHA-256 `3cb7dc4c565f45748ae9e5f4e0a8bca7a93471396fdbcc169e121ebb2399ca72`, constants manifest
SHA-256 `81dc4bdae4ac7c21ab4f1e6d08c1859b3f6fb58dfc86c2c470cc1e13d1c28388`, DSL/equation JUnit
SHA-256 `09a1177730d6d5406dc978f2833ce52c78a4321a7135f1e2c9a24d2dfb09728e`, probe/admission JUnit
SHA-256 `60cd6ade552dbfbf0a7ba0c1ff29d86c2be0b598d595ec0ec8c55f8ebff98d1a`, explicit
environment-blocked MLX JUnit SHA-256 `77b9c0881e532369d8f7427d60dffbe88c193ff08cbbe43872b65f6d8fa00f90`,
and fail-closed synthetic-map probe JSON SHA-256
`e44521accc6637da6a7a8d3d7c37b446fbed2fe649646f62eddfe1bc2395ce64`.
`verification_receipt.json` binds the typed identities, complete refusal classification, source and
scorer hashes, test/JUnit hashes, dry-run disposition, audit verdicts, and explicitly owed gates.

The current managed sandbox cannot initialize a Metal device; even an MLX CPU allocation terminates
with `[metal::load_device] No Metal device available`. Therefore the executable MLX loss/gradient and
real-kernel receipt tests are landed but cannot truthfully be called runtime-green here. The admission
surface consequently remains **REFUSE**, not inferred from static tests. The governed launcher also
reported system-admission REFUSE under current concurrent memory pressure; that refusal was respected.

## Speed evidence and owed validation

Existing measured isolated-scorer anchors remain GPU **1.56x** and CPU **1.75x** at K=8, and the reused
compiled batch warp carries its existing approximately **1.8x over eager** component measurement.
These establish the useful component speed class but are not additive and are not a faithful full-V9
training-step measurement. The operator target/hypothesis is approximately 2-4x; no such full-step
number is claimed by this landing.

Owed to the operator-fired fast arm on an uncontended real Metal device:

1. faithful B=1 versus B=2 full-V9 loss/gradient receipt with the real frozen scorers, raw/carrier
   temporal surfaces, and evaluated fused forward/VJPs;
2. end-to-end multi-step wall-clock median and backend receipts proving all three kernels fired;
3. n600 training/score validation on the produced checkpoint/archive through untouched
   `upstream/evaluate.py`.

No heavy n600 training was launched by this build task.

## Triality and custody

- DSL: `MicroBatch` now validates positive B, composes with seed islands, and stamps the training-only
  drift waiver/no-score-authority contract. The real V9 compile/parse inventory is a regression test.
- DAG: `FEED-microbatch-v9-unlock-landing` in
  `.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`.
- Equations: `micro_batch_functional_parity_training_admission_v1`; existing chroma, logit-adjust,
  scorer-batch-dependence, and fp-reorder laws reconciled with the training-only override. The new law
  is intentionally not registered as an empirical speed law until the faithful Metal receipt exists.

Pointer delta: **zero** against the canonical `reports/latest.md` contest-CPU frontier. The
implementation moves wall-clock capability only.

## STORES CONSULTED

- `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, `docs/operating_manual_craft_handoff.md`;
- `.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md` and
  `.omx/research/SPEC_v8_perclass_decomposition_20260708.md`;
- operator memory `max_throughput_over_bit_identity_operator_override_20260712.md`;
- `reports/latest.md`, `.omx/state/lane_registry.json`, `.omx/state/subagent_progress.jsonl`,
  `.omx/state/master_gradient_anchors.jsonl`, `.omx/state/modal_call_id_ledger.jsonl`,
  `.omx/state/cost_band_posterior.jsonl`, and `.omx/state/continual_learning_posterior.jsonl`;
- latest sister Codex findings/session summary and current council/design/directive memos;
- the typed V9 compiler/program, the real trainer parser, the serial loss body, the batched twin, and
  the existing #D15/test/probe/equation/DAG receipts.
