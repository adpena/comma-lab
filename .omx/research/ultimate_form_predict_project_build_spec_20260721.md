# Ultimate-form PREDICT -> PROJECT receiver — build specification

`task=#597` · `lane_id=lane_ultimate_form_predict_project_597_20260721` · `$0 local` · `research_only=true` · `[macOS-CPU advisory]` · pointer `0.1910828242 [contest-CPU] UNMOVED` · MAIN landing review required

## Objective

Build a deterministic, NumPy-portable receiver primitive that stores the scorer-equivalence problem instead of camera-frame solutions:

1. **PREDICT** a scorer-space cell field from one spacetime object: one static ground-frame chart, one smooth `xi(t)` trajectory, sparse movable tracks, boundary-normal jitter/phase residuals, and sparse critical events.
2. **PROJECT** only violated constraints into the intersection of declared argmax cells, a Pose tube, exact factor-2 resize reachability, and the uint8 lattice. Projection is deterministic, uses a declared first-max tie policy, and may run full Dykstra/POCS to convergence. Decode wall-clock is engineering telemetry only and can never reject a small-distortion/small-rate candidate.
3. Emit a canonical constraint-seed schema v0 and measurements B1–B5. A projection row is not a score row; only a future exact archive replay on contest hardware can move the pointer.

## Binding corrections and settled corpus

- The temporal seed is **not** a stream of xi-transported labels. The measured screw amortization ratio is resolved through LawRef ID `partition_temporal_transport_amortization_jitter_bound_v1`, using its canonical builder anchor and `amortization_ratio` evaluator; it is not copied into this spec. That law says naive transport loses because cell bulk is already cheap and the residual is boundary jitter. Bulk may ride xi for free, but the counted temporal stream is arc-length-indexed boundary-normal offsets, sparse events, and, where cheaper, the existing phase carrier.
- The single-object/per-frame comparison must decompose chart, trajectory, bulk, jitter, tracks, events, and container/header bytes separately. The worldsheet may win on chart/trajectory/bulk; both formulations honestly pay the jitter residual.
- Reuse the settled factor-2 receiver contract, R1b4 zero-search/section-custody policy, `joint_seg_pose_rate` interval solve and measured waterfill, the #580 full `ker(A)` callable, support-fill, `INFLATE_WORKERS`, and plane-cache conventions. Do not reimplement scorers or store scorer weights.
- B1 is an engineering ledger: record stage time, peak/output scale, worker count, and the optimization ladder `algorithmic cut -> vectorize -> parallel workers -> Rust port`. Rust hooks are interface notes only in this landing.

## Binding additions received after initial build

These directives supersede any narrower raster/two-curve reading of this specification:

1. **Morse–Smale chart and vineyard.** The normative static partition description is critical points, separatrix arcs, cell adjacency, and a canonical traversal of the MS 1-skeleton—not a label raster. Persistence orders features from stable bulk to low-persistence dash/event structure. Time-varying birth/death/split/merge events are persistence-pair vineyard lifecycle symbols. A raster may exist only as a deterministically derived decode product or a clearly labeled fixture compatibility surface.
2. **Causal jitter ladder.** Boundary jitter is modeled cause, not irreducible noise. The measurement API must distinguish R0 raw normal offsets; R1 phase-conditioned residuals; and R2 static ground-appearance chart plus `xi` plus a generic response surface plus sparse exceptions. Each rung is measured in exact canonical bytes at equal represented constraints. The remaining exception stream is reported by stratum.
3. **Global joint waterfill.** B4 is no longer a pair of independent Seg/Pose curves. It must accept jointly decoded, custody-bound sweeps across chart precision, sites/cells, jitter, events, pose tightening, response parameters, and the eat-the-flip option. Curves are constructed inside the joint solve, use exact `(delta score, delta bytes)` points, enforce the timestamp-free LawRef-derived reciprocal of `realization_breakeven_bytes_v1` for a provenance-bound unit score recovery, credit overlapping flips once, preserve composition order, and report the pairwise interaction matrix. Until such a sweep is supplied, B4 remains fail-closed and inconclusive.
4. **Pose-blind decode by constraint tightening.** The archive cannot load PoseNet. Encoder-side hard-oracle work must translate the Pose tube into shippable pixel-space boxes/linear constraints such that every admitted decoder choice is verified to remain inside the tube. Merely naming a Pose projection is not executable authority.
5. **Complete receiver stages.** The schema/receiver must make structural: camera-resolution inverse-R uint8 realization; frame-0 pose-only versus frame-1 Seg+Pose asymmetry; integer/fixed-order cross-host byte identity; and a versioned, length-delimited, fail-closed #402-style section container. The full `ker(A)` remains a callable degree of freedom, never serialized.
6. **Exact interaction rollout.** All stream compositions are evaluated jointly at encode; no independence assumption. Where order matters, expose commutator-aware ordering and record delta-of-joint versus sum-of-singles. This lane consumes existing dynamics/costate/collateral-coupling surfaces by ID and must not fork sibling #535.

Acceptance is staged honestly: schema/receiver interfaces and strict guards may land before real scorer rows, but no receipt may call the ultimate representation, G1–G5, global waterfill, or the MS-native codec complete until the corresponding measured leg exists.

## Binding additions through 2026-07-21T05:37:49Z

These additions consume the verified M1 receipt at commit
`9bd01e2232f6898c2564ab8bb7254609c1ebf645`, whose committed
`rep_mine_solved_binary_20260721T045500Z.json` bytes have SHA-256
`265302908fd7c4789891ab0d3b0f8aacaf9f178ea8e40f8737ed5f4fcd55b368`.

1. **Exact M1 anchors and caveats.** Preserve 45.1668 percent actual M2 kernel energy, which is not byte savings; 31.1071 percent measured teacher-logit gauge, explicitly rejecting the old approximately-52-percent premise and never subtracting this different surface from camera-array bytes or context; the centered rank re-derived from the canonical `segnet_head_rank4_linear_flipdist_v1` builder anchor and `head_difference_rank`; 83.1564 percent quotient energy explained by cell constants; 21,304 digital cells, which are not a classical Morse-Smale certificate; 0.0150856545 bit/cell and 222,447.0271 bytes for the optimistic empirical position/adjacency/pose-proxy context estimate. That estimate excludes Pose side information, model, and header; it is not a lower bound. The target is 216,222 bytes. The G3 census contains exactly 17,926 flips. The isolated r2b marginal is 6.806 percent of the LawRef-resolved lambda and is eaten absent a measured positive interaction.
2. **Per-flip sellback fixed point.** A strict evidence object binds the M1 commit/SHA, all 17,926 sorted unique flip IDs, each receiver stratum and operator price stratum, exact positive-pixel counts, exact integer #557 coded bits, score value `positive_pixels*100/(600*512*384)`, score per coded byte, and keep/eat decision at the timestamp-free LawRef-derived `realization_breakeven_bytes_v1` reciprocal. It binds r1b7 survival custody and an iterative recode chain with exact input/output kept sets, context hashes, coded-stream hashes, and a final stable fixed point. Repricing may be measured nonmonotone and must be recorded honestly; monotonicity is never assumed. Reject threshold inversions, broken chains, missing flips, inconsistent ledgers, admitted/eaten overlap, or disagreement with the global eaten decomposition.
3. **Pose-tube knee.** A separate strict evidence object binds the same joint decode and M1 receipt over increasing tube relaxation. Each point carries exact `d_pose`, archive bytes and byte savings, the derived nonlinear `sqrt(10*d_pose)` score delta, and marginal score per byte. The selected point is the measured crossing at lambda; no KKT point may be invented.
4. **Global integration and equation stop.** Global waterfill evidence must include both strict objects and reconcile its admitted/eaten flip sets with the per-flip fixed point. Missing sellback or Pose-knee evidence is explicitly inconclusive in the measurement receipt and equation gate. Equation registration remains fail-closed until the real B2, decoder gates, measured global joint sweep, per-flip fixed point, and Pose knee all agree under byte-identical custody.

## Binding additions through 2026-07-21T05:51:18Z

1. **Action-level ladder, not pixel-actuator inference.** The individual 17,926-flip ledger prices value and is currency only. For every homogeneous flip family, separately measure all five canonical actuator rungs under the same joint decode: `L1_geometry_chart`, `L2_channel`, `L3_hyperplane_feature`, `L4_regional_plane`, and `L5_pixel_write`. Each rung binds exact coded bits and actual coded bytes, the same positive family benefit, archive/decoded/context/coded-stream hashes, and ERF collateral flip IDs/count/through-R `d_seg`. Select the minimum coded-bytes-per-net-score valid rung with canonical-rung tie breaking. `L5_pixel_write` is valid only for an explicitly isolated singleton.
2. **Exact census reconciliation.** Families are sorted, nonoverlapping, stratum-homogeneous, decision-homogeneous, membership-hashed, and must partition all 17,926 M1 flips. The chosen-rung distribution is exact for every receiver-stratum/operator-price-stratum pair and reconciles family/flip counts, kept/eaten fixed-point sets, candidate/admitted/avoided bits and bytes, ERF collateral union-once counts/`d_seg`, and net benefit. Per-flip and global evidence carry byte-identical ladder objects; omission or drift is inconclusive/refused.
3. **Boundary-inverse scoped policy.** Bind commit `e2f679755fea09e4c55a12592db1bc615373c6a8`, receipt `.omx/research/boundary_inverse_custody_20260721T052100Z.json`, SHA-256 `2c7c091c61d1676c80b5db1772a29d3b2f73934398966c8566b6175abc4021e3`. Its authority is mask fidelity only, without through-R score. No lossless spatial phase-atom budget is allowed; the eight-bin spatial phase arm is a formulation-scoped negative, while temporal phase through causal R1/R2 remains live. `generic_2d_k4` is only an L4 candidate: 586 counted sidecar bytes, four atoms, `+0.0046822634384159` mask F1, and 313,271 remaining mask false negatives. It is never admitted until same-joint-decode through-R pricing exists.
4. **Source-bound resumability.** Every new measurement config hashes the exact receiver, schema, measurement tool, and equation consumer source bytes and records an aggregate implementation SHA. Each stage and checkpoint binds that aggregate plus the full config hash. Any configuration or implementation-source drift refuses existing stages. Old-code SSD stages that lack this custody are deliberately non-resumable under the new implementation; they are preserved, never rewritten or adopted.
5. **Registration remains blocked.** Absence status is exactly `INCONCLUSIVE_NO_MEASURED_ACTION_LEVEL_LADDER`; it is surfaced by B4, the measurement gate, the dated receipt, and the equation gate. Structural fixtures may exercise the contract but are never measurements or registration authority.

## Binding additions through 2026-07-21T06:21:21Z

1. **Native grammar is wire structure.** `predict_project_constraint_seed.v0` carries the canonical nonterminals `SceneChart`, `XiSpline`, `CriticalPoint`, `SeparatrixArc`, `VineyardSymbol`, `PoseBox`, `ExceptionRun`, `LadderEdit`, and optional-default-absent `LearnedTailGenerator`. Each production names an existing/free interpreter procedure and one #557 grammar-position arithmetic context. Unknown/reordered symbols, productions, procedure references, or context gaps fail closed. Exact consumption, final byte, final SHA-256, and atomic decode are mandatory. No grammar production carries raster, scorer, weights, or hidden video bytes.
2. **Surgical attribution and edits.** A strict importable receipt binds #350 exact attribution, #404 telemetry binding, and #420 artifact custody from chart coefficient through channel, the canonical rank-4 hyperplane, regional values, and realized pixels. Every action-family L1--L5 rung has an edit request and a deterministic same-joint-decode through-R response with exact `dS`, bytes, ERF collateral, and before/after hashes. The aggregate binds every M1 flip attribution receipt by content SHA-256 and every complete edit receipt byte-identically to its action-rung bytes. The per-flip ledger remains currency, never an actuator.
3. **Learned-tail three-way race.** For every global stream, the strict race compares literal exceptions, Rule-118-counted S3 generator weights plus instance seeds plus its own exceptions, and the eaten-flip score cost at the timestamp-free LawRef-derived `realization_breakeven_bytes_v1` reciprocal. Literal and generator must have equal realized hard-oracle fidelity, all exact bytes are counted, and the unique strict Lagrangian minimum wins. Learned tails are absent by default and are admitted only where the generator strictly wins. This build performs no training or launch and supplies no measured race.
4. **No duplicate source of truth.** The reuse manifest below is the canonical Task #597 reuse table. Dated JSON/memo/DAG artifacts reference this table; they do not define a competing component inventory. `NEW-with-justification` was considered only after repository search. All new Task #597 work extends its owned schema/receiver/equation surfaces; no parallel implementation of a settled component was created.
5. **Completed B1 engineering run truth.** The read-only SSD receipt at `/Volumes/VertigoDataTier/pact/evidence/predict_project_20260721/b1_full600_scorer_geometry_fixture_20260721T050000Z/measurement/receipt.json` has SHA-256 `1ea0e4bb571cf99f539ef18cf6210f622bc72968ff54e3e810d2ac0576a5575d`; its custody manifest has SHA-256 `4d6aeab38cdb20e008934d041c4f626465160b529e7495be9536bfe151f7b141`. It is a 600-stage receiver-only scorer-geometry fixture on `[macOS-CPU advisory]`: summed stage seconds `879.9565938347951`, peak RSS `218087424`, output bytes `117964800`, and double decode equality true. It contains no hard oracle, B2, score, contest authority, or promotion claim. Because the original config lacked implementation-source hashes, the completed old-code run is preserved and is not resumable under current source. SSD bytes are immutable and were not touched.

## Canonical-law resolution custody

Task #597 consumes numeric laws only through `src/tac/witness_dsl/lawref.py`. The receiver installs idempotent in-process evaluator adapters under the already-registered IDs and never writes the persistent registry. `realization_breakeven_bytes_v1` evaluates the canonical `breakeven_bytes` function for an explicitly provenance-bound unit score recovery, then derives lambda as its reciprocal. `partition_temporal_transport_amortization_jitter_bound_v1` reads byte totals from its canonical builder anchor and evaluates `amortization_ratio`. `segnet_head_rank4_linear_flipdist_v1` reads singular values from its canonical builder anchor and evaluates `head_difference_rank`. Their timestamp-free, fallback-free runtime custody is bound by `CANONICAL_LAW_RESOLUTION_SHA256=291230db665c1bfd30ce0a7619314af0a9b6419367c7de812952d3c7714ed2c0` and is labeled `DERIVED`, not measurement or contest authority. `rate_law_ladder_v1` and Catalog #318 are recorded only as available reused identities; Task #597 consumes no numeric value from either.

### Canonical Task #597 reuse manifest

| Inventory | Component | Exact source path | Classification | Disposition / search record |
|---|---|---|---|---|
| owned | constraint-seed native grammar | `src/tac/optimization/predict_project_schema.py` | extended | Existing Task #597 schema remains the sole wire SoT. |
| owned | attribution/edit and learned-race validators | `src/tac/optimization/predict_project_receiver.py` | extended | Existing Task #597 receiver remains the sole evidence-contract SoT. |
| owned | equation registration blockers | `src/tac/canonical_equations/predict_project_receiver_20260721.py` | extended | Existing unregistered Task #597 equation surface. |
| equations | canonical equation registry | `src/tac/canonical_equations/registry.py` | reused-as-is | Registration API is referenced, never mutated here. |
| equations | canonical evaluators | `src/tac/canonical_equations/evaluators.py` | reused-as-is | Existing evaluator dispatch. |
| equations | LawRef compiler | `src/tac/witness_dsl/lawref.py` | reused-as-is | Grammar procedure/law references only. |
| equations | rank-4 flip-distance law | `src/tac/canonical_equations/segnet_head_rank4_flipdist_20260715.py` | reused-as-is | Attribution names its canonical law ID. |
| equations | rate-law ladder | `src/tac/canonical_equations/rate_law_ladder_20260713.py` | reused-as-is | Action-price semantics only. |
| equations | temporal amortization law | `src/tac/canonical_equations/partition_temporal_transport_amortization_20260715.py` | reused-as-is | Ratio is LawRef-resolved from the canonical builder anchor through `amortization_ratio`; no copied numeric constant. |
| equations | full-kernel structure law | `src/tac/canonical_equations/resize_full_kernel_structure_20260720.py` | reused-as-is | Full kernel remains callable/nonserialized. |
| task-space | witness DSL schedules | `src/tac/witness_dsl/curriculum_dsl.py` | reused-as-is | No invented flags or schedule fork. |
| task-space | witness gauge | `src/tac/witness_dsl/gauge.py` | reused-as-is | No alternative gauge SoT. |
| task-space | lever registry | `src/tac/witness_dsl/lever_registry.py` | reused-as-is | No new lever registry. |
| task-space | task-space spine audit | `tools/audit_task_space_levelset_spine.py` | reused-as-is | Canonical support-fill evidence consumer. |
| task-space | AA-SDF observation/render | `src/tac/boundary_math/aa_sdf_observation_render.py` | reused-as-is | Native grammar procedure reference. |
| task-space | AA-SDF Rust decode | `runtime-rs/crates/tac-boundary-decode/src/contour.rs` | reused-as-is | Rust hook/reference only; no port or launch. |
| task-space | S3 banded plane trainer | `src/tac/boundary_math/integer_plane_banded_trainer.py` | reused-as-is | Learned race binds reuse ID; no training. |
| task-space | SE(3) spline | `src/tac/lie/se3_bspline.py` | reused-as-is | XiSpline interpreter procedure. |
| task-space | Fisher natural solver | `src/tac/information_geometry/fisher_natural_solver.py` | reused-as-is | Reference-only attribution/training inventory. |
| task-space | Metal site/sparse ranking | `src/tac/local_acceleration/metal_sparse_adjoint.py` | reused-as-is | No Metal work launched. |
| inverse | #547 uint8 lattice feasibility | `src/tac/optimization/uint8_lattice_feasibility.py` | reused-as-is | Exact uint8/factor-2 realization. |
| inverse | #549 joint Seg/Pose/rate | `src/tac/optimization/joint_seg_pose_rate.py` | reused-as-is | Existing interval solver imported. |
| inverse | #580 full resize kernel | `src/tac/optimization/resize_full_kernel.py` | reused-as-is | Existing callable kernel imported. |
| inverse | support fill | `src/tac/optimization/resize_null_preimage.py` | reused-as-is | Reuses `apply_tier1_zero_weight_fill`. |
| inverse | S2 partition seed/#557 context | `src/tac/optimization/s2_partition_seed.py` | reused-as-is | Exists on `main`; absent only from this older isolated worktree, so referenced and not copied. |
| inverse | tie-aware preimage | `src/tac/optimization/tie_aware_preimage.py` | reused-as-is | No alternate preimage solver. |
| inverse | null preimage compiler | `src/tac/optimization/resize_null_preimage.py` | reused-as-is | No null-compiler fork. |
| receiver | R1b4 section receiver | `src/tac/boundary_math/r1b4_section_receiver.py` | reused-as-is | ABI/custody imported. |
| receiver | v10 production receiver | `src/tac/witness_dsl/v10_production_receiver.py` | reused-as-is | Tie/arithmetic/receiver IDs imported. |
| attribution | #350 exact A/B attribution | `tools/witness_exact_ab.py` | reused-as-is | Receipt binds exact source path. |
| attribution | #404 telemetry binding | `src/tac/witness_control/telemetry_binding.py` | reused-as-is | Receipt binds proof custody. |
| attribution | #420 artifact contract | `src/tac/witness_run_artifacts.py` | reused-as-is | Receipt binds artifact SHA. |
| coding | STC-Dasher codec | `src/tac/codecs/stc_dasher/encoder.py` | reused-as-is | No entropy-coder fork. |
| motion | homography implementation | `src/tac/contrib/homography_motion.py` | reused-as-is | No motion implementation fork. |
| training | governed witness launcher | `tools/launch_witness_run.py` | reused-as-is | No launch performed. |
| training | witness autoconfig | `tools/witness_autoconfig.py` | reused-as-is | No config fork. |
| training | storage/memory preflight | `tools/witness_memory_preflight.py` | reused-as-is | No bulk run performed. |
| training | resume registry | `src/tac/witness_control/resume_registry.py` | reused-as-is | Task measurement uses source-bound local stages; no registry edit. |
| training | stage checkpoint primitive | `src/tac/checkpoint.py` | reused-as-is | No checkpoint implementation fork. |
| training | costate SENSE | `tools/costate_digest.py` | reused-as-is | Reference-only; no shared state mutation. |

No row is `vendored`, `refactored`, or `NEW-with-justification`. Repository search covered the requested equation, task-space, inverse-solving, attribution, coding, and training inventories before this table was sealed.

## Owned surfaces

- `src/tac/optimization/predict_project_schema.py` — canonical importable `predict_project_constraint_seed.v0` schema, strict canonical bytes/parser, and single-spacetime-object invariants.
- `src/tac/optimization/predict_project_receiver.py` — deterministic predictor, violation extractor, projection primitives, double-decode hash, B3 stratification, B4 waterfill adapter, and existing receiver/kernel composition metadata.
- `src/tac/tests/test_predict_project_schema.py`
- `src/tac/tests/test_predict_project_receiver.py`
- `tools/measure_predict_project_receiver.py` — resumable/chunked B1–B5 measurement runner with atomic per-stage receipts and SSD-first bulk outputs.
- `src/tac/canonical_equations/predict_project_receiver_20260721.py` and focused test, but only register the equation after a real B2 anchor exists.
- Dated JSON/Markdown receipt, DAG FEED, and final findings memo.

Do not touch sibling-owned `s2_compose_full_partition` or `rep_mine_solved_binary` outputs, upstream scorer files, pinned C2 decoder/parser bytes, frontier pointers, provider dispatch, live runs, unrelated trainer/DSL files, or other worktrees.

## Schema v0 contract

The canonical schema must be one object with explicit units and integer/fixed-point quantization:

- static ground chart encoded once, including scorer geometry and a content hash;
- one 600-step xi trajectory represented by spline/AR controls plus residuals, never 600 independent pose records masquerading as the single-object form;
- sparse movable tracks with unique IDs and ordered knots;
- boundary-normal offset stream indexed by `(curve_id, time, arc_index)` with a declared subpixel quantum;
- sparse event alphabet for birth, death, split, merge, occlusion, and phase reset;
- constraint seeds containing only predictor violations, with sorted unique coordinates/cell IDs and optional pose-tube/projector metadata;
- declared receiver ABI, tie policy, projection policy, seed, pair count, no-scorer/no-search assertions, and false authority flags.

Reject unknown fields, booleans laundered as integers, non-finite values, unsorted/duplicate records, out-of-range classes/coordinates/times, inconsistent pair counts, hidden per-frame chart arrays, noncanonical JSON, hash drift, trailing bytes, or an absent jitter disposition.

## Projection contract

- Provide exact box/half-space projection plus Dykstra/POCS for a finite linear cell/tube problem. Use stable lexicographic constraint order, an explicit tolerance and iteration cap, deterministic cycle detection, and fail closed on nonconvergence.
- Quantize into uint8 by deterministic nearest-lattice search with lexicographic ties. A floating feasible point is not authority; final cell/tube checks run on the quantized point.
- For factor-2 image projection, call the existing exact integer interval solve and expose the full-kernel/support-fill composition hook; do not serialize `ker(A)`.
- Only predictor violations become constraint seeds. Report already-satisfied fraction by class and by `cell_interior`, `boundary_codim1`, `movable_track`, and `critical_event` strata.
- The module remains scorer-agnostic. Compress-time hard-oracle callbacks live only in the measurement tool; no scorer or weights enter schema/receiver bytes.

## Measurements and acceptance

### B1 — full n600 timing ledger

- Deterministically decode/project 600 pairs or, when the hard B2 campaign is still running, provide both a measured prefix and an explicit linear/worker extrapolation labeled `DERIVED` until the real n600 row lands.
- Record predict, violation scan, projection, realization, verification, serialization, and total wall time; record hardware/runtime, worker count, plane-cache hits, deterministic output hash, and double-decode equality.
- Record one-shot versus pose-tolerance sweep timings. A slow exact row is `KEEP + engineering follow-on`, never a rejection.

### B2 — hard cell/tube verification

- First pass a deterministic prefix with exact cell equality, Pose within declared tube, uint8/factor-2 exactness, and two byte-identical decodes.
- Then run the real n600 CPU-Torch hard oracle at seed `1234`, batch `16`, using exact cache/archive bytes. Report `d_seg` against the declared cell description and `d_pose` against the declared tube. If n600 cannot finish in this landing, preserve resumable stages and report the literal incomplete scope; never extrapolate distortion or call the prefix n600.
- Register `predict_project_cell_tube_uint8_projection_v1` only when a real measured B2 anchor exists, with the exact authority scope.

### B3 — predictor quality to seed size

- Report numerator/denominator/fraction of already-satisfied cells and emitted violation records by class and stratum. Report canonical raw and compressed seed bytes; never infer archive bytes from entropy alone.

### B4 — Pose tube and waterfill

- Evaluate actual hard PoseNet values for increasing same-joint-decode tube relaxations and derive the nonlinear `sqrt(10*d_pose)` marginal from exact adjacent points. Select only the measured lambda crossing; do not fabricate a KKT knee.
- Bind the exact M1 all-17,926-flip iterative #557 recode fixed point, threshold decisions, r1b7 survival custody, per-stratum counts/bytes/`d_seg`, and observed context monotonicity. Integrate admitted/eaten sets into the global same-joint-decode allocation. Missing evidence is inconclusive.

### B5 — single object versus per-frame schema

- Canonically serialize both forms at equal represented constraints. Report raw and deterministic compressed bytes by component.
- Quantify homography-proxy refinement by measured boundary-normal residual/error and event counts. Keep exact within-pair xi custody separate from cross-pair proxy custody.
- Score the result against the LawRef-resolved `partition_temporal_transport_amortization_jitter_bound_v1` ratio and state the narrow verdict scope; do not kill the worldsheet family from one chart realization.

## Verification and landing

- Focused pytest for schema, receiver, equation, and tool; Ruff; `py_compile`; JSON parse; `git diff --check`; deterministic rerun.
- Every changed Python file receives two clean `review_tracker` passes after its final edit.
- Commit through `tools/subagent_commit_serializer.py` with one expected-content SHA-256 per path. No direct commit fallback and no co-author trailer.
- The final memo must carry STORES CONSULTED, labeled `MEASURED`/`DERIVED`/`SPECULATIVE` rows, verdict scope, sensitivity/Pareto/bit-allocation/autopilot/continual-learning/probe-disambiguator hooks, triality, pointer honesty, exact commit proof, and `MAIN_REVIEW_REQUIRED=true`.

## Exact initial test command

```bash
PYTHONPATH=src /Users/adpena/Projects/pact/.venv/bin/python -m pytest -q \
  src/tac/tests/test_predict_project_schema.py \
  src/tac/tests/test_predict_project_receiver.py
```

## STORES CONSULTED

Delegated Task #597 authority; `CLAUDE.md`; `AGENTS.md`; `PROGRAM.md`; craft handoff manual; v7.5 §8; v8; v10 §§W/X; #547/#549/#580 receipts/code; shared R1 and R1b4 receiver receipts/code; G1/G3 n600 receipt; #180 Morse–Smale partition codec; #452 tube-algebra boundary code; #425 phase carrier; #402 container grammar; flicker/phase-carrier/dash-codec/dynamics corpus; current frontier/lane/progress/council/probe surfaces; M1 receipt at commit `9bd01e2232f6898c2564ab8bb7254609c1ebf645`; boundary-inverse receipt at commit `e2f679755fea09e4c55a12592db1bc615373c6a8`; exact read-only B1 SSD receipt/custody manifests; canonical reuse-manifest source paths; per-arm directives through `2026-07-21T06:21:21Z`.
