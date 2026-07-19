# C2 integer-plane emitter build and bounded advisory receipt

**UTC:** 2026-07-19
**Lane:** `lane_c2_integer_plane_emitter_build_20260719`
**Verdict:** **BUILD COMPLETE / LOCAL NUMPY+TORCH+LATTICE+HARD-ORACLE LOOP MEASURED; MLX EXECUTION PROOF BLOCKED BY NO METAL DEVICE**
**Verdict scope:** C2 source, deterministic fixtures, and six real pairs from the frozen n24 selection only. This is not a trained vehicle, byte-closed archive, contest score, basis winner, promotion gate, or family verdict.
**Pointer delta:** **NONE.** `0.1910828242 [contest-CPU Linux x86_64]` remains unmoved.

## Stores consulted

- `.omx/research/SPEC_v10_integer_plane_vehicle_20260719.md`, especially §§3.2, 4.1, 4.2, and row C2.
- `.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md` §8 and `.omx/research/SPEC_v8_perclass_decomposition_20260708.md`.
- `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, `.omx/research/t5_crucible2/VEHICLE_OS.md`, and `docs/operating_manual_craft_handoff.md`.
- `src/tac/optimization/uint8_lattice_feasibility.py`, `src/tac/optimization/v10_constructive_solver.py`, `src/tac/quantization.py`, the #531 quotient-coordinate law, the structured-init/ELM corpus, and existing MLX scorer/custom-kernel surfaces.
- Frozen upstream `modules.py` SHA `065961ba97023e393e27818760b0dc8efaa8dd53c5d4cc70a2db8ee1b3cf49aa`, `frame_utils.py` SHA `d689aca7d263997cb2fb980d6098d503f955e56e8642cd0a04cc437f0ffdab90`, SegNet SHA `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6`, and PoseNet SHA `0f3a0874c5c387f990d7b88bd1d7e1f6de35d98b45f2a289989db2c77b9b6576`.
- Real cache `gt_n600.npz` SHA `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`, primary VJP manifest SHA `3d1218a52ededc4b347ae94c5c2bf58d06d70dd8f530bec67bf9cab36ee00694`, and secondary VJP manifest SHA `200e8cfa375cbdb8154777156441ae6adadf33e75668c86cc52b816f79488e94`.
- MAIN inbox directives through `2026-07-19T14:21:03Z`: pair-parallel deterministic expansion and a measured MLX/Metal local-loop twin.

## What landed

### Core vehicle surface

`src/tac/boundary_math/integer_plane_emitter.py` now owns the strict two-plane ABI:

- scorer output `[N,2,384,512,3] uint8`; camera realization `[N,2,874,1164,3] uint8`;
- immutable structured base and coordinate topology, plus one separately named quotient-residual group consisting only of independent `[pair,plane]` codes and a shared RGB head;
- deterministic, shape-parallel expansion with no cross-pair recurrence;
- NumPy-fp32 reference, Torch `Uint8STE`, and lazy MLX clip-round STE;
- refusal of wrong dtype, rank, geometry, nonfinite state, float scorer handoff, and copied/collapsed planes when strict independence is requested;
- independent exact factor-2 realization and numerator proof for every pair and both planes;
- sign-fixed class/logit `U4`, all ten coupled pair margins, fixed-capacity basis A/B, encode-only hash-bound VJP metadata, and the exact 2x2 Pose-visible YUV6 transform.

The train-least deletion test is executable: setting the two quotient-residual arrays to zero returns the immutable solved base exactly. There is no learned camera-null field. The learned parameter count is explicit in `CapacitySignature`.

### Exact head-basis custody

The production API derives the SVD from the frozen float32 head tensor, centers the five head rows in float64, fixes each SVD sign by the first maximum-magnitude coordinate of the associated right vector, and then names the corresponding **left** `(5,4)` class basis `U4`. A tool-local right-vector basis is not substituted.

- **MEASURED:** singular values are `3.128376325627011`, `2.1542713872702617`, `2.024707869857505`, and `1.796263835653701`.
- **MEASURED:** raw float64 `U4` SHA is `1d62c1fe316214dd7b370e52c0927015c8ca4dca91bde26a2c88c68bdb6b3f62`.
- **MEASURED:** raw float64 ten-by-four pair-coefficient SHA is `86b784d04062249530499d8f6a4cc05a3423eddf0d85321ff690673053443f51`.
- **DERIVED AND TESTED:** `q = centered_logits @ U4`, `centered_logits = q @ U4.T`, and all ten margins are `q @ (D10x5 @ U4).T`.
- Singular-value division is absent and explicitly refused.

Raw and U4/pair-margin arms share the same emitter state, seed, topology, parameter count, emitted bytes, and hard-oracle duty. The DSL seals residual width at four while the basis verdict is unresolved. **No basis winner was measured in this build.**

### Typed DSL and resume surface

`IntegerPlaneEmitterPolicy` is default-OFF and sealed against trainer activation, launch, payment, score, promotion, and pointer mutation. It types geometry, basis, STE, capacity lock, pair-parallel expansion, and the no-autoregression invariant. The required-policy `IntegerPlaneEmitter(...)` factory lives in `curriculum_dsl.py`, is registry/activation visible, emits no invented trainer flag, and compiles to baseline-identical argv.

Four measured-anchor LawRefs resolve the U4 singular values from SHA-pinned `.omx/research/v10_power_diagram_frame195_diagnostic_20260718.json`. The standalone checkpoint envelope includes live residual state, EMA, optimizer, RNG, stage/epoch/pair position, and topology/discrete/event/dual hashes. Publication uses a same-directory, no-clobber hard link from an fsynced temporary, verifies exact bytes and inode/path identity around both durability barriers and source cleanup, and refuses concurrent winners or substitutions. Its guarantee is point-in-call; it does not claim control over an uncooperative process after return. It deliberately does not pretend a trainer-side ResumeRegistry controller exists yet.

## Fixture proof

Authoritative current receipt: `.omx/research/c2_integer_plane_emitter_fixture_20260719_v3.json`
Receipt SHA: `7dab522735ab85a5c9ad8f9386954a709b6ffee5932560c15102cf654c37ba08`
Receipt bytes: `5,117`

The earlier unversioned and `_v2.json` fixture receipts are retained as
**SUPERSEDED HISTORICAL EVIDENCE**. They predate the final custody/wording
closure and are not current authority receipts.

All statements below are **MEASURED on the local fixture**:

- NumPy and Torch integer bytes match exactly: SHA `ac317fcdc19b738a6681b0f8fcb193d38aa8c86bf84ad59eee7bdde160a83b9d`.
- In-range gradients are nonzero; gradients outside `[0,255]` are zero.
- Plane 0 and plane 1 differ.
- Both full-geometry planes pass exact numerator and canonical realization proof: `589,824/589,824` scorer values per plane, denominator `786,432`, two distinct camera hashes.
- Fixed-capacity raw/U4 build arms emit the same exact bytes; all ten pair margins are present; sigma division is false.
- Fixture capacity is 20 learned scalars: eight pair-plane code values plus twelve shared-head values.

**BLOCKED_ENVIRONMENT_NO_METAL:** MLX imports, but any array evaluation raises `metal::load_device: No Metal device available`. Therefore MLX forward-byte and gradient parity are **NOT MEASURED** here. The implemented MLX leg is not promoted by inference from Torch.

## Bounded real-pair loop

Authoritative current receipt: `.omx/research/c2_integer_plane_emitter_n24_advisory_20260719_v3.json`
Receipt SHA: `adc63a815f96938ecfe1014f9f13caaee800ae69d1696838c6b74dc106ab2491`
Receipt bytes: `38,707`
Axis: **`[macOS-CPU advisory, untrained]`**
Completed selection: pair IDs `0,1,2,3,4,5` from the frozen n24 set.

The earlier unversioned and `_v2.json` advisory receipts are retained as
**SUPERSEDED HISTORICAL EVIDENCE**. The unversioned receipt predates canonical
selected-NPZ tensor/custody revalidation and complete transitive code custody.
The v2 receipt revalidated the selected rows but falsely described both frozen-
n24 manifests as checked while recording only the selected primary manifest.
Neither may be cited as the current measurement receipt.

The loop was fresh deterministic init → two scorer planes → independent factor-2 camera realization → frozen CPU-Torch Seg/Pose hard oracle. All six selected VJP sidecars were file-hashed and then re-opened through the canonical `load_vjp_pair_row` path, which rechecked embedded pair identity, `custody_json`, tensor dtype/shape/finiteness, every tensor byte hash, active-arrangement invariants, and VJP factorization. They remained encode-only proposal metadata with no candidate-admission or decoder role.

| pair | d_seg | d_pose | full pair iteration s |
|---:|---:|---:|---:|
| 0 | 0.0001475016 | 0.0000100950 | 1.1315 |
| 1 | 0.0001068115 | 0.0000827029 | 1.4607 |
| 2 | 0.0000966390 | 0.0000394448 | 1.3918 |
| 3 | 0.0001271566 | 0.0000183378 | 1.0599 |
| 4 | 0.0001525879 | 0.0000260173 | 1.0915 |
| 5 | 0.0000762939 | 0.0000098116 | 1.0604 |

- **MEASURED:** 12/12 independently emitted planes have zero numerator error at denominator `786,432`.
- **MEASURED:** both frozen-n24 manifests plus canonical selected-sidecar revalidation took `2.7554 s` before scorer setup; the receipt records primary selected rows `0..5` and the checked secondary route with zero selected rows.
- **MEASURED:** CPU setup `2.9640 s`, scorer-loop total `10.1838 s`, median `1.1115 s/pair`, p95 `1.4435 s/pair` with four CPU threads.
- **DERIVED from the six recorded hard rows:** mean d_seg `0.0001178318`, maximum d_seg `0.0001525879`; mean d_pose `0.0000310682`, maximum d_pose `0.0000827029`.
- **MEASURED:** the n6 residual has 60 learned scalars at fixed width four; pair expansion is independent and no cross-pair autoregression exists.

These small distortions are **not an untrained-model score result**. The smoke consumes a full source-derived exact-projection base that is not serialized or rate-counted in C2; it proves the intervention/realization/oracle plumbing only. Receiver grammar, shortest-program deletion, counted base generation, parse-back, archive bytes, and contest replay remain later gates. Any use of these values as a score, basis win, or rate claim is false authority.

The MLX full-loop twin is implemented to batch the emitter, require exact byte parity to NumPy, independently rerun the exact CPU lattice inside each timed pair iteration, then use the existing MLX scorer adapter. It explicitly refuses to substitute fused-R for the distinct factor-2 lattice operator. On a Metal host it compares both candidate and reference scorer outputs against CPU-Torch on identical scorer-input tensors through the repository's canonical `MLXTorchParityThresholds`: zero SegNet argmax-different pixels and PoseNet component absolute delta at most `2e-5` are charged gates. Final d_seg/d_pose differences are diagnostic-only and cannot mint a parity PASS. **No MLX seconds/iteration or scorer parity were measured on this no-Metal host.** MAIN owes the Metal rerun.

## Frozen-line and six-binding reconciliation

1. **modules.py anchored:** frozen `modules.py:130-158` defines the composed Pose/Seg hard oracle and last-frame Seg behavior; frozen `frame_utils.py:51-78` defines the four luma samples plus 2x2-averaged U/V. Those exact hashes and roles are recorded above.
2. **Intrinsic complexity:** deleting the quotient residual returns the solved base; VJP guidance has no decoder payload or admission method. The real-smoke base itself is explicitly not byte-closed and cannot survive the later shortest-program gate merely because C2 plumbing works.
3. **Pose is 2x2:** `rgb_pair_to_yuv6` reproduces the four luma sites and block-averaged chroma. Tests show within-block chroma permutations with preserved averages are Pose-invisible while luma sites remain visible.
4. **Weight level:** U4 and all ten margins come from the frozen head bytes. Six custodied Seg/Pose VJPs are hash-bound and encode-only; hard inference, not a VJP proposal, produced every reported distortion.
5. **Train nothing if possible:** the smoke base is exact numerator projection; the module consumes solved base/topology and exposes only the quotient residual as trainable. No training occurred.
6. **Surgical any-point:** one typed output contract accepts scorer-plane bases and codes, then feeds the same exact lattice and hard-oracle boundary. Pre-A camera or margin-level producers can route through it without duplicating receiver logic.

## Verification

- **MEASURED:** 228 tests passed in the final affected matrix: core emitter, DSL/resume policy, tool/custody, lever registry, typed-config schema/migration, and both active spec adapters. This includes explicit half-even tie/clip-edge parity, empty-batch refusal, and checkpoint publication race regressions.
- **MEASURED:** independent fresh review passed 161 DSL/checkpoint tests and re-falsified the no-clobber, source-substitution, target-substitution, durability-order, and typed-adapter surfaces; the independent v3 receipt audit also re-derived both receipt files and their custody claims without edits.
- **MEASURED:** Ruff clean on all six new Python files; all eleven changed Python files compile; `git diff --check` clean.
- **MEASURED:** the review tracker records two clean passes across every changed Python file (`586` entity marks per pass).
- **MEASURED:** a broader adapter group separately passed 112 tests; four pre-existing `test_typed_launcher_dsl_composition.py` fixtures fail the already-existing curriculum epoch-budget gate because they request `epochs=5`, before any C2 adapter code executes.
- No training, governed launch, paid dispatch, archive build, score computation, pointer update, or sacred-run mutation occurred.

## Triality and pointer honesty

- **DSL leg:** default-OFF `IntegerPlaneEmitterPolicy` plus required-policy Lever factory and standalone resume envelope.
- **DAG leg:** structured base → quotient residual → exact clip-round scorer planes → independent factor-2 preimages → hard oracle; VJP remains proposal-only.
- **Equation leg:** `q = centered(logits) U4`, `margins = q (D U4)^T`, saturation-aware clip-round STE, and exact integer `A_num x = D y` verification.
- **Pointer leg:** no delta; the active pointer is unchanged.

## Remaining blockers and MAIN landing review

1. **MLX execution blocker:** run the focused MLX forward/gradient fixture and the `mlx-metal-iteration` n24 twin on an actual Metal host; require exact emitted bytes and record real seconds/iteration.
2. **Basis verdict owed:** compare raw versus U4/pair-margin under genuinely identical optimization/capacity and hard rows before any width increase.
3. **Rate/receiver custody owed:** replace or compile the source-derived full base into a counted generic program, then prove intrinsic deletion, parse-back, archive bytes, and receiver timing. The current smoke is not byte-closed.
4. **Contest axes owed:** CPU Linux x86_64 and CUDA remain separate future receipts under explicit authority.
5. **MAIN must review before landing:** complete base-to-head diff; frozen head/U4 hashes and SVD convention; hot `curriculum_dsl.py` hunk; policy-contained LawRefs; standalone checkpoint semantics; MLX no-Metal classification; exact-lattice versus native-f32 authority separation; and the explicit non-authority of the source-derived smoke base.
