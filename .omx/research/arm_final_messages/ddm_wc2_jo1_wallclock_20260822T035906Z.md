Implemented and landed as commit `20de6132a7`. The sacred r8 run remained untouched.

Latest read-only observation at `03:58:37Z`:

- Target-birth: 511/600 pairs complete.
- Recent median: 13.664 s/pair, n=64.
- Remaining: **5.96–8.81 hours**.
- Endpoint: **09:56–12:47 UTC**.
- Liveness is artifact-fresh, not process-table verified.
- Receiver/coder tail remains a derived allowance, not measured r8 work.

| Measured surface | Result |
|---|---:|
| Live training step | 1.413 s |
| Offline fresh-pair median | 26.483 s |
| PoseNet share | 24.385 s, about 92% |
| Three whole-pair workers | **3.627× gross speedup** |
| Serial identity | **9/9** |
| Concurrent identity | **3/3** |
| Real coder race | 2.543 s, 144/144 payloads retained |

Lever verdicts:

- **READY** — whole-pair subprocess parallelism.
- **RIDE-R8** — thread tuning; one thread changes Pose6, two threads are slower.
- **REFUTED** for this hot-swap — MLX/Metal, ANE/CoreML, and their heterogeneous composition. These are formulation-scoped, not family-wide kills.

The conservative swap calculation assumes r9 must redo all 600 current-stage pairs:

- r8 median remaining: 7.445 h.
- r9 remaining before swap cost: 4.292 h.
- Build/proof plus measured reseal analog: 0.531 h.
- Net saving: **2.621 h**.
- Required `2 × swap cost`: 1.062 h.

The arithmetic passes, but MAIN must measure the actual r9 integration/reseal/resume cost. At the observed lower pair rate, that remaining work must finish within approximately **34.4 minutes**.

Artifacts:

- [Full memo](/Users/adpena/Projects/pact/.omx/research/ddm_wc2_jo1_wallclock_20260821.md)
- [ETA instrument](/Users/adpena/Projects/pact/experiments/ddm_wc2_jo1_wallclock.py)
- [Pair worker](/Users/adpena/Projects/pact/experiments/ddm_wc2_jo1_pair_parallel.py)
- [Retained profiler](/Users/adpena/Projects/pact/experiments/ddm_wc2_jo1_step_profile.py)
- [Primary retained receipt](/Volumes/VertigoDataTier/pact/ddm_wc2_jo1_wallclock/profile_r3/STEP_COST_PROFILE.json)

Verification: ruff and compilation passed; payload-retention gate passed with 42 tests; Python files received two review passes; commit-scoped preflight passed. The broader developer preflight remains red on eight unrelated, pre-existing repository paths.

**GESTALT-DELTA:** JO1 changed from a static 21.7–36.2-hour projection into a live receipt-derived curve with one exact, source-pinned actuator on its dominant PoseNet wall.

Own-vehicle frontier remains **S=0.14821987563243377 @ 180,368 B [contest-CUDA T4 n600]**. This arm did not move it.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN r9 sealer; consumer store: `/Volumes/VertigoDataTier/pact/ddm_wc2_jo1_wallclock/r9_seal/`; fire trigger: bind and integrate the reviewed pair-worker source, migrate the latest checkpoint without restarting training, measure actual swap cost, and confirm `net_saved >= 2 × swap_cost`.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN/JO1 endpoint harvester; consumer store: `/Volumes/VertigoDataTier/pact/ddm_nr1_taskcell_body_rebase/retained/`; fire trigger: r8 or admitted r9 emits a sealed endpoint archive/render/component/fresh-carrier receipt.

## LIVE-HYPOTHESES

- Three or four subprocess workers will retain most of the measured 3.627× speedup at n600 because independent PoseNet work owns roughly 92% of pair wall.
- r9 remains economical even if it must redo all current-stage pairs, provided its actual integration and reseal stay below the measured fire ceiling.
- PoseNet-specific ANE offload remains plausible, but needs its own placement, fidelity, regeneration, and latency receipts.

## DEAD-ENDS

- Changing exploration batch shape or order: Pose values change.
- One-thread execution: slower and Pose6 differed on 3/3 pairs.
- Two-thread execution: slower than the four-thread incumbent and lacks repeat proof.
- `ProcessPoolExecutor` in this sandbox: semaphore creation is denied; ordinary subprocesses work.
- Immediate MLX/Metal or ANE hot-swap: required device access and PoseNet-specific exactness evidence are absent.