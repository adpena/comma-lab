# DDM V9 carrier composition — receiver-closed advisory finding

**Lane:** `ddm_v9_carrier_compose_byteclose`  
**Tasks:** #603 feeding #613  
**Evidence axis:** `[macOS-CPU frozen-scorer advisory]`  
**Authority:** `score_claim=false`, `d_seg_claim=false`, `d_pose_claim=false`, `research_only=true`  
**Pointer:** `0.1910828242 [contest-CPU]` **UNCHANGED**  
**MAIN landing review:** **REQUIRED**

## Outcome first

| bridge | exact archive bytes | d_seg | d_pose | advisory objective | n600 wall-clock projection |
|---|---:|---:|---:|---:|---:|
| `[448,512)` n64 | **51,668** | **0.045286496480** | **159.104827981350** | **44.450999796357** | 278.72 s |
| `[344,600)` n256 | **72,397** | **0.040169219176** | **157.798907948748** | **43.789038785395** | 240.13 s |

> **BOXED VERDICT — MEASURED:** `ADVISORY_INSTANCE_FAILS_SUB015_BOX_FORMULATION_OPEN`. The bytes are well below both the 154.6 KB sub-0.15 box and the ~216.3 KB pointer-knee box, but d_seg is 34.6× (n256) above the 0.00116 gate and Pose6 is catastrophically outside a viable joint witness. This closes only the exact inherited-v6-fixed carrier composition with no admitted G2CS1 symbol; it does **not** close joint chart-symbol, xi-event, corrected-inner-Jacobian, or broader V9 formulations.

## What landed

- A deterministic outer `ZIP_STORED` grammar that binds a custodied five-role DDM predictor and optional `G2CS1` Lane centerline-coefficient symbols.
- A receiver that applies counted chart symbols **before** generic region-coherent Lane rerasterization and canonical merge. There is no pixel-coordinate/RGB correction grammar.
- Strict parse/re-encode identity, deterministic replay, exact outer unique-home closure, recursive per-stratum byte attribution, and sampled mutation refusal.
- The existing counted Pose6 member remains the **sole** Pose6 home. It is not duplicated by the outer archive.
- Atomic preserved checkpoints at receiver-build and frozen-scorer-measurement stage boundaries.

## First non-null per-stratum accounting

The d_seg column is conditional error on the named target class under the composite receiver. The d_pose column is the measured shared-composite PoseNet error, not a dishonest leave-one-out attribution.

| stratum | n64 nested bytes | n64 d_seg | n256 nested bytes | n256 d_seg | n256 shared d_pose |
|---|---:|---:|---:|---:|---:|
| Road | 3,383 | 0.093452173388 | 9,716 | 0.082664837473 | 157.798907948748 |
| Lane | 36,313 | 0.490127010580 | 38,122 | 0.453456535463 | 157.798907948748 |
| Undrivable | 1,471 | 0.002706123838 | 5,526 | 0.004584219637 | 157.798907948748 |
| Movable | 119 | 0.999979319017 | 118 | 0.993213496226 | 157.798907948748 |
| MyCar | 193 | 0.000088321008 | 193 | 0.001108472649 | 157.798907948748 |
| xi/Pose6 | 645 | 0.045286496480 composite | 1,749 | 0.040169219176 composite | 157.798907948748 |

**MEASURED diagnosis:** MyCar is locally close to the d_seg gate and Undrivable is comparatively small. Lane and Movable are the dominant semantic failures. The inherited Pose6 stream is receiver-consumed and single-owned, but its realized control is unusable; it is not yet the triple-use transport variable required by the V9 design.

## Correction contract and blocker delta

The archive supports only `G2CS1` chart symbols addressing Lane centerline coefficients. The receiver decodes these symbols, changes the low-dimensional chart, and rerasterizes a coherent band. The measured archives intentionally carry zero symbols because no hard-oracle improvement with pose-tube custody was established in this run. Shipping an arbitrary nonzero symbol would be fake admission.

**Discharged:** receiver closure; exact archive bytes; all-five-role consumption; one Pose6 home; G2CS1 parse/apply/rerasterize path; per-stratum byte/d_seg/d_pose rows; n64/n256 bridges; n600 wall-clock projection; resumable stage receipts.

**Remaining primary solve DOF:** a Fisher-margin/curvature-ranked joint multi-coefficient `G2CS1` solve plus xi-transported Lane-dash/Movable birth-death event insertion, admitted only when hard semantic cells and the Pose tube improve through the same receiver. The corrected inner Jacobian must predict realization; no blanket or pixel-domain fix is authorized.

## Re-derive in under ten minutes

```bash
/Users/adpena/Projects/pact/.venv/bin/python tools/run_ddm_v9_carrier_compose.py --config .omx/research/configs/ddm_v9_carrier_compose_n64_20260722.json --output-directory .omx/research/ddm_v9_carrier_compose_n64_603_613_20260722T122800Z
/Users/adpena/Projects/pact/.venv/bin/python tools/run_ddm_v9_carrier_compose.py --config .omx/research/configs/ddm_v9_carrier_compose_n256_20260722.json --output-directory .omx/research/ddm_v9_carrier_compose_n256_603_613_20260722T123300Z
/Users/adpena/Projects/pact/.venv/bin/python -m pytest -q src/tac/optimization/tests/test_direct_description_carrier_compose.py
```

Completed receipts validate bound hashes and the receiver, then return without recomputation.

## STORES CONSULTED

- `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, `docs/operating_manual_craft_handoff.md`
- `.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md`
- `.omx/research/SPEC_v8_perclass_decomposition_20260708.md`
- `.omx/research/recursive_fractal_optimal_representation_v9_build_spec_20260714.md`
- `.omx/research/direct_description_minimizer_PRIMARY_SPEC_20260721T214800Z.md`
- `.omx/research/g2g_chart_symbol_receiver_20260721T161523Z.md`
- v6 n64/n256 receipts and exact fixed-AR1 archives named in the typed configs
- frozen target receipt/cache and upstream SegNet/PoseNet custody named in each result receipt
- `reports/latest.md`, `.omx/state/lane_registry.json`, `.omx/state/canonical_task_status.jsonl`

`0.1910828242 [contest-CPU]` remains the pointer. These are advisory local rows, not contest scores, candidates, promotions, or launch authority.
