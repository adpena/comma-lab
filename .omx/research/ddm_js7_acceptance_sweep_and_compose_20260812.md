# DDM JS7 acceptance sweep and compose receipt — 2026-08-12

## VERDICT

`QUEUED-WITH-A-FIRE-ORDER` for one MAIN-owned n600 scorer pass, followed by exact contest-CUDA only if that measured n600 row remains negative.

JS7 completed the remaining 198 EC1 singleton measurements and combined them with JS6's two retained rows into the full 200-proposal census. Sixty-five singleton proposals passed the strict pose gate, improved robust flips, and beat the exact CP135 marginal bar on their bare packet. Exact current-stack remeasurement admitted 44 of those 65: 20 were rejected by the joint pose gate and one by zero marginal robust gain. The selected stack is receiver-real and byte-closed, but it is not an n600 scorer row and not a contest score.

The retained complete container is 186,575 B, or +323 B versus the exact 186,252 B CP135 archive. On the sealed stratified-random n32 CPU gauge, it removes 61 robust sampled flips (weighted projection: 1,133 robust n600 flips) with realized pose delta +1.3798191585101111e-6. The resulting projected score change is -0.0005811215635310277: seg -0.0009604560004340278, pose +0.0001642619950445387, and rate +0.00021507244185846135. Axis: `[macOS-CPU advisory, stratified-random n32, instrument floor 0.0131 S]`; `score_claim=false`.

The effective and own-vehicle frontier pointers are unchanged.

## FULL ACCEPTANCE TABLE

The 198 new rows consumed the existing proposal-owned payloads; EC1 generation did not run. JS6's first two rows were verified and reused. All 200 rows are present in `/Volumes/APDataStore/pact/ddm_js7_20260812/ACCEPTANCE_TABLE.json` (86,806 B, SHA-256 `6274e6aa7d85b0600a57ca6c0f848d80d0af8c91e3ec5d1db8b2bf6fd9152290`).

| Family | Measured | Pose accepted | Pose rate | Robust improving | Bare admissions | Bare yield | Best B/projected robust flip |
|---|---:|---:|---:|---:|---:|---:|---:|
| boundary | 151 | 107 | 70.86% | 69 | 48 | 31.79% | 0.2407407407 |
| lane | 48 | 29 | 60.42% | 27 | 17 | 35.42% | 0.2916666667 |
| island | 1 | 1 | 100.00% | 0 | 0 | 0.00% | n/a |
| **all** | **200** | **137** | **68.50%** | **96** | **65** | **32.50%** | **0.2407407407** |

The exact marginal bar was 1.2731082153 B/robust flip. F1 did not fire because useful pose-accepted singleton admissions exist. F2 did not fire because the population contains points below the exact bar. The one island event is an instance-level negative: it passed pose but added 19 projected robust flips, so this does not kill unrepresented island formulations.

## JOINT COMPOSED STACK

Singleton values were never summed. Each ranked trial was applied to the current accepted stack and the full n32 scorer state was remeasured.

| Selected | Families | Sites | Sample robust delta | Projected robust n600 delta | Pose delta | Packet | Stop |
|---:|---|---:|---:|---:|---:|---|---|
| 44/65 | 34 boundary offsets; 10 lane-program deltas | 58 across 6 frames | -61 | -1,133 | +1.3798191585101111e-6 | LZMA1-raw, 311 B | ranking exhausted |

Twenty otherwise admissible singletons failed only after joint pose remeasurement, and one had no marginal robust gain. This directly confirms that singleton admission is not a composition receipt on this surface.

The selected packet is `/Volumes/APDataStore/pact/ddm_js7_20260812/compose/selected/retained/coders/stack.lzma1_raw.ec1overlay` (311 B, SHA-256 `8b0531565993fb33d54e80ba893fa587936e65738bf600050bf7af4f1f5e3f9b`). Raw, Brotli-q11, and LZMA1-raw payloads were all persisted with decode receipts; LZMA1-raw was the measured winner.

## COMPLETE CONTAINER AND RECEIVER PROOF

The selected overlay is inside the counted `p` member and is consumed by the adapted runtime parser. The runtime strips the self-delimiting overlay before parsing the unchanged CP135 residual sections, decodes the real LZMA1-raw event packet, validates source-class preconditions, and applies the 44 events before rendering. It is not inert archive padding.

- Base archive: `/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime/archive.zip`, 186,252 B, SHA-256 `6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6`.
- Composed archive: `/Volumes/APDataStore/pact/ddm_js7_20260812/container/retained/archive.zip`, 186,575 B, SHA-256 `465d3c584c54c55d161c6411cfbd34c4bf3c4c546c189ddcbfbb1a2a941e9af4`.
- Complete-container growth: +323 B. The 12 B beyond the 311 B packet is the counted overlay footer.
- Independent decode: the expected and decoded 600-frame token planes are byte-identical at 117,964,928 B each, SHA-256 `46c7bdafd130ec2cedf78f7c4d9478425e951def1e549a921ccae9383a0b9e87`.
- Base member prefix and every CP135 residual section remained byte-identical.
- A separately built repeat archive is byte-identical to the primary archive.
- A fully independent second sweep/compose/container run at `/Volumes/APDataStore/pact/ddm_js7_20260812_repeat` reproduced the acceptance table, selected IDs, joint metrics, packet, archive, decoded token plane, and projected score component-for-component. The cross-run receipt is `/Volumes/APDataStore/pact/ddm_js7_20260812/DETERMINISM_REPEAT.json`.
- Adapted runtime: `/Volumes/APDataStore/pact/ddm_js7_20260812/container/retained/adapted_runtime`.

All newly materialized singleton, joint-trial, coder, camera, scorer-input, logits, argmax, pose, token-plane, member, archive, and runtime payloads were retained with byte counts and SHA-256 records. The primary run occupies 2.1 GiB across 4,751 files under the APDataStore run root. It is complete and resumable; the run stayed below the 40-minute bound.

## PROJECTED ECONOMICS AND AUTHORITY BOUNDARY

| Component | Projected delta |
|---|---:|
| robust seg | -0.0009604560004340278 |
| pose | +0.0001642619950445387 |
| exact rate from +323 B | +0.00021507244185846135 |
| **total** | **-0.0005811215635310277** |

This projection combines the sealed n32 stratified robust-flip weighting, the realized n32 pose term, and measured complete-container bytes. It is not an n600 scorer measurement. JS7 did not use MPS or Metal, did not run `upstream/evaluate.py`, did not run a full n600 scorer, and did not touch Modal. No pointer moved.

## RECALL EVIDENCE

I searched the full `.omx/research/` content and arm receipts for `EC1`, `acceptance`, `joint remeasure`, `complete container`, `receiver`, `CP135`, and `1.273108`; queried `tools/list_canonical_equations.py --json`; searched `CANONICAL_RESEARCH_INDEX*`, the `sub015_DAG_*` FEED blocks, design/SPEC surfaces, the task ledgers, harness bridge, and `.omx/state/main_hot_state.md`.

Beyond the charter seeds:

- `.omx/research/ddm_v19b_joint_remeasure_stack_DAG_FEED_20260723.md` and the corresponding DAG FEED establish current-stack remeasurement as the only admissible composition law. This changed JS7 from a singleton sum into 65 retained joint trials with 21 typed rejections.
- `.omx/research/ddm_tf1_theoretical_floor_and_beyond_20260812.md` provides the exact CP135 bar 1.2731082153 B/robust flip. This replaced the charter's rounded comparison in every admission decision.
- `.omx/research/ddm_t1r1_container_build_rehearsal_20260812.md` and `.omx/research/ddm_hc1_hy1_container_push_20260812.md` distinguish packet bytes from complete archive bytes and require independent parser closure. This changed pricing to the +323 B whole-container delta and added exact 600-frame token-plane parse-back.
- `.omx/state/main_hot_state.md` supersedes the common contract's stale frontier paragraph: the live effective frontier is CP135 at 0.16195513827824176 and the own-vehicle frontier is LC2 at 0.16959899569230852. JS7 therefore reports no pointer move rather than inheriting the obsolete 0.75398 row.
- The canonical partition-transport/correspondence surfaces and `.omx/research/ddm_se1_shipping_axis_survival_resolve_20260812.md` preserve EC1 as content-distinct from the failed class-scale raster grammar. This kept the island negative scoped to the one measured event instance.

## IMPLEMENTATION AND VERIFICATION

- `experiments/ddm_js7_acceptance_sweep_and_compose.py`: resumable 198-row harvest, full table, current-stack joint scorer, complete-container builder, independent decode, projection, recipe, and queue annex.
- `experiments/ddm_js7_ec1_overlay_runtime.py`: self-delimiting overlay format; raw, Brotli-q11, and LZMA1-raw real coders; strict decoder; token application for NumPy and CPU Torch.
- `experiments/tests/test_ddm_js7_acceptance_sweep_and_compose.py`: coder, corruption, archive, application, projection, runtime wiring, and deterministic ZIP tests.

Verification before serialization: Python compilation, Ruff, and the JS7+JS6+EC1 test set passed (22 tests). `git diff --check` passed. Two review-tracker passes are required and recorded after the final source hashes are fixed.

## SEALED MAIN RECIPE AND FIRE ORDER

The negative projected delta sealed `/Volumes/APDataStore/pact/ddm_js7_20260812/SEALED_MAIN_N600_RECIPE.json` (2,818 B, SHA-256 `4a9759d643f23326b70f9a1288fd22e39826e439a2cf328fbafbebad9de70793`).

- **Action:** measure the retained JS7 composed EC1 overlay on the full n600 scorer and exact contest-CUDA chain. **Disposition:** QUEUED-WITH-A-FIRE-ORDER. **Owner:** MAIN scorer-lane router. **Consumer store:** `/Volumes/APDataStore/pact/ddm_js7_20260812/main_n600_and_exact`. **Fire trigger:** MAIN owns the sole n600 scorer lane, confirms it is idle, revalidates the retained archive/runtime SHA records, then runs n600 in chunks no larger than 120 before any exact dispatch.

## LIVE HYPOTHESES

- The 44-event stack will remain net-negative at n600 because its repeated n32 projection has total headroom of 0.0005811216 and removes robust cells across six separated sampled frames. This is plausible but untested because the 1,133-flip value is a stratified projection, not an n600 scorer result.
- A composition-aware ordering that explicitly reserves pose headroom may admit a different or larger stack than singleton B/flip order. This is plausible because 20 economical singletons were rejected only by the evolving joint pose term.
- The unused boundary and lane events may contain a second mutually exclusive stack worth racing after n600 truthing. This is plausible because 21 current-order rejections do not prove those events fail against a different accepted prefix.

## DEAD ENDS

- Summing singleton benefits is closed: joint remeasurement rejected 21/65 singleton admissions.
- Bare event-packet pricing is closed as a shipping receipt: the selected packet is 311 B, but the only relevant measured growth is the +323 B complete container.
- The available island-death event is closed at INSTANCE scope: it passed pose but worsened projected robust flips by 19. The island family is not globally killed by one event.
- Claiming JS7 as a score or frontier move is closed: no n600 scorer or exact contest evaluation ran.
