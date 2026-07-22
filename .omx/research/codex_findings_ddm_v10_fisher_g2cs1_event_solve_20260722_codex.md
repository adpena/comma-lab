# DDM V10 Fisher G2CS1 + transported-event solve — advisory finding

**Lane:** `lane_ddm_v10_fisher_g2cs1_event_solve_20260722`  
**Tasks:** #603 / #613 on master #578  
**Evidence:** `[macOS-CPU frozen-scorer advisory]`  
**Authority:** `research_only=true`, `score_claim=false`, `d_seg_claim=false`, `d_pose_claim=false`  
**Pointer:** `0.1910828242 [contest-CPU]` **UNCHANGED**  
**MAIN landing review:** **REQUIRED**

## Outcome first

| window | requested added bytes | realized added bytes | exact total bytes | d_seg | official YUV6 d_pose | advisory objective |
|---|---:|---:|---:|---:|---:|---:|
| n64 `[448,512)` base | 0 | 0 | 51,668 | 0.045286496480 | 159.104827981350 | 44.450999796357 |
| n64 `[448,512)` | 5,120 | **1,353** | **53,021** | **0.042511622111** | **159.093118922196** | **44.172945495577** |
| n64 `[448,512)` | 15,360 / 40,960 / 102,400 | **1,353** | **53,021** | **0.042511622111** | **159.093118922196** | **44.172945495577** |
| n256 `[344,600)` base | 0 | 0 | 72,397 | 0.040169219176 | 157.798907948748 | 43.789038785395 |
| n256 `[344,600)` | 5,120 / 15,360 / 40,960 / 102,400 | **0** | **72,397** | **0.040169219176** | **157.798907948748** | **43.789038785395** |

**MEASURED verdict:** `ADVISORY_INSTANCE_VOCABULARY_EXPRESSIVENESS_BOUND_BEFORE_REQUESTED_BUDGET`.
The n64 joint search improves both frozen metrics with only 1,353 added bytes, then exhausts its
bounded semantic inventory. The n256 scale check admits nothing: its three Road proposals reduce
Seg errors, but each measurably worsens official Pose, and the strict containment gate refuses them.
No measured row approaches `d_seg <= 0.00116`. The ~200 KB falsifier remains armed but untriggered
because neither exact archive reaches 180,000 bytes; the earlier expressiveness plateau binds first.
This is an **INSTANCE-vocabulary negative**, not a G2CS1, event, or structured-carrier family negative.

## What became executable

- The existing V9 archive/receiver now accepts three strict semantic correction packets: Lane
  `G2CS1` centerline coefficients, a cubic normalized-x Road-boundary displacement chart, and
  Lane/Movable birth/death bbox primitives.
- Multi-pair events advect their primitive with differences in the already-counted Pose6 ordinal
  codes. They add no Pose stream, coordinate list, pixel residual, RGB patch, scorer weights, or GT.
- Every packet has a versioned magic, CRC, sorted unique addresses, canonical parse/re-encode, exact
  ZIP home, strict window checks, and an individual receiver no-op refusal.
- Fisher/head geometry proposes candidates. Admission is not closed form: the receiver rerenders the
  impacted canonical scorer batch and accepts only a measured Seg reduction whose full objective gain
  exceeds exact marginal rate cost while Pose remains inside the baseline tube.
- Candidate and budget checkpoints are immutable and preserved. Identical budget archives reuse the
  exact already-measured bridge and state the originating rung.

## Measured mechanism disposition

The corrected n64 inventory evaluated 32 proposals: 20 Road-boundary, 4 Lane-G2CS1, 4 Movable-event,
and 4 Lane-event. It admitted 15 Road candidates, two Movable births, and one Lane death. Both admitted
Movable births have `lifetime=2`; their nonzero `(gain_x_q4,gain_y_q4)` values consume Pose6 transport.
All four Lane-G2CS1 centerline candidates and all three Lane births failed exact admission.

At the final n64 rung, target-class d_seg is Road `0.083038967493`, Lane `0.464687018660`, Movable
`0.980890771318`, Undrivable `0.003038389402`, MyCar `0.000089892556`. Thus the improvement is real,
but Lane/Movable remain the dominant vocabulary debt. The n256 target-class rows remain Road
`0.082664837473`, Lane `0.453456535463`, Movable `0.993213496226`, Undrivable `0.004584219637`, and
MyCar `0.001108472649`.

## Round-1 falsification and fix

The first n64 search used a global top-32 Fisher cutoff. Road candidates occupied every slot, so the
run improved d_seg to `0.041097561518` at 53,130 bytes but never tested Lane or event vocabulary. That
receipt is preserved only as a diagnostic at
`.omx/research/ddm_v10_fisher_event_n64_20260722T130403Z/` (receipt SHA
`ab7d27d717fe943d9db8bb345e182d975ed87d1fc1e42fcd227e894609ac8f1a`). It is superseded for the joint
claim. The runner now preregisters a minimum from Road, Lane-chart, Lane-event, and Movable-event
families before filling remaining slots by Fisher priority; a regression prevents recurrence.

## Exact blocker delta

**Discharged:** executable Road-boundary and birth/death vocabularies; real Pose6-transport use;
mechanism-diverse Fisher acquisition; exact canonical-batch measured admission; rate break-even;
pose containment; 0/5/15/40/100 KiB exact ladders; n64 and n256 official bridges; resume checkpoints.

**Remaining:** the current bbox/cubic/c3 INSTANCE lacks enough expressiveness. The next formulation
must expand low-dimensional boundary/event shape and phase DOFs or use a scorer-obligation-derived
structured carrier. Any residual field must use the governed curvelet/shearlet basis; this landing
introduces no residual stream. A bound n600 v6 predictor/archive is also absent, so no n600 row was
manufactured from the n256 window.

## Reproduce each measurement in under ten minutes

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/adpena/Projects/pact/.venv/bin/python tools/run_ddm_v9_carrier_compose.py --config .omx/research/configs/ddm_v10_fisher_event_n64_20260722.json --output-directory .omx/research/ddm_v10_fisher_event_n64_20260722T130403Z_round2
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/adpena/Projects/pact/.venv/bin/python tools/run_ddm_v9_carrier_compose.py --config .omx/research/configs/ddm_v10_fisher_event_n256_20260722.json --output-directory .omx/research/ddm_v10_fisher_event_n256_20260722T130403Z_round2
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/adpena/Projects/pact/.venv/bin/python -m pytest -q src/tac/optimization/tests/test_direct_description_carrier_compose.py
```

Fresh measured wall clocks were 414.69 s (n64) and 390.35 s (n256). Completed receipts validate
their hashes/archives and return immediately.

## STORES CONSULTED

- `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, `docs/operating_manual_craft_handoff.md`
- `.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md`
- `.omx/research/SPEC_v8_perclass_decomposition_20260708.md`
- `.omx/research/direct_description_minimizer_PRIMARY_SPEC_20260721T214800Z.md`
- V6 n64/n256 receipts and fixed-AR1 archives hash-bound by the typed configs
- Frozen target receipt/cache plus upstream SegNet/PoseNet custody bound by the result receipts
- `reports/latest.md`, `.omx/state/lane_registry.json`, `.omx/state/canonical_task_status.jsonl`
- Arm inbox through `2026-07-14T12:48:10Z`; fleet inbox through `2026-07-21T13:15:53Z`

`0.1910828242 [contest-CPU]` remains unchanged. These archives are `.not_a_candidate` advisory
receipts, not contest scores, promotion evidence, or execution authority.
