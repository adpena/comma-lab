# ddm_wc2 JO1 wall-clock instrument and lever adjudication

Date: 2026-08-21

Disposition: **PAIR-PROCESS LEVER READY FOR AN r9 RESEAL; LIVE r8 LEFT UNTOUCHED; MAIN OWNS FIRE**

Authority: `[macOS-CPU offline wall-clock probe; no score authority]` plus read-only timestamps from the live r8 run

## Outcome

The live curve is tracking the low half of the original 21.69--36.22 hour seal. At
`2026-08-22T03:53:11Z`, target-birth had completed 487/600 fresh-Schur pairs. The
recent 64-receipt throughput band was 13.5049/14.1369/16.8137 seconds per pair
(p10/median/p90), giving **6.21--9.42 hours remaining** and a derived endpoint band
of **2026-08-22 10:06--13:18 UTC**. The median-hybrid estimate was 7.45 hours.

Process-table access is sandbox-blocked, so process liveness is not claimed. The
latest pair receipt was 7.03 seconds old at observation time; the status is therefore
artifact-fresh, not process-verified. The receiver/coder tail has not executed under
r8 and remains a sealed 0--7,200 second allowance, not a measured current-run rate.

The critical path is PoseNet inside fresh same-object compensation. A retained
three-pair profile measured a 26.4829 second median offline pair solve, of which the
independently aggregated PoseNet median was 24.3850 seconds (about 92%). Whole-pair
subprocess concurrency preserved the exact per-pair batches and measured **3.627303x
gross speedup**, including spawn and model load, with 3/3 concurrent identities.
The serial reference passed 9/9 identities (3 pairs x 3 repeats).

The shipped ETA command is:

```bash
.venv/bin/python -m experiments.ddm_wc2_jo1_wallclock
```

Add `--json` for the full rates, denominators, liveness boundary, tail allowance,
remaining work, and endpoint band.

**GESTALT-DELTA:** the JO1 wall is no longer a 21.7--36.2 hour static projection; it
is a receipt-derived live curve whose dominant PoseNet pair work has one exact,
measured, source-pinned parallel actuator.

## Step-cost profile

Primary retained receipt:
`/Volumes/VertigoDataTier/pact/ddm_wc2_jo1_wallclock/profile_r3/STEP_COST_PROFILE.json`
(320,222,548 retained bytes beneath the profile root; receipt SHA-256
`5ee94401fb52a87e6b29384e663158c44b5d2fef63166a1a3aa65ed7c269dba0`).
Every copied input, explored camera certificate, full winner, training tensor, and
all 144 real-coder candidates remains retained. Nothing under the live run was opened
for write.

| surface | measured wall | denominator / interpretation |
|---|---:|---|
| live r8 training step | 1.413278 s | one real sealed preflight step; current-run configuration |
| offline training step | 2.520313 s median | 3 repeats; concurrent-probe wall, so component attribution only, not an absolute replacement for the live rate |
| SegNet forward in offline step | 0.754464 s | independent median |
| PoseNet forward in offline step | 0.330628 s | independent median |
| render/residual/R in offline step | 0.371668 s | independent median |
| backward | 0.464605 s | independent median |
| retention/cert I/O in offline step | 0.617291 s | independent median |
| fresh pair solve | 26.482920 s median | 3 pairs x 3 repeats at the pinned 4 threads |
| PoseNet forwards in fresh pair | 24.384968 s median | about 92% of pair wall; dominant lever |
| Schur/render/orchestration remainder | 1.020676 s median | independently aggregated per-row remainder |
| retention/cert I/O in fresh pair | 1.110559 s median | all outputs retained |
| carrier linear solve | 0.001247 s median | not the wall |
| real Brotli coder race | 2.542862 s | 144/144 candidates retained; control winner 181,472 B, SHA-256 `7d80cff46a4973a03d52a8579a5720d25129bb2bd64f6ae164f26a4f26c3b812` |

The 181,472-byte coder winner is a timing control made from the current target-birth
residual and exact fx5 base codes. It is not a completed fresh-solve package, not a
score row, and not a frontier candidate.

## Lever verdicts

| lever | verdict | evidence and boundary |
|---|---|---|
| whole-pair subprocess parallelism | **READY** | built in `experiments/ddm_wc2_jo1_pair_parallel.py`; each worker owns a complete pair, retains the same candidates/certs, and parent merge is numeric. Reference identity 9/9; concurrent identity 3/3; serial same-thread sum 86.5373 s versus 23.8572 s wall including spawn/model load = 3.627303x. No n600 production run or r9 seal was fired. |
| thread/BLAS count | **RIDE-R8** | 1 thread was slower and Pose6 differed on 3/3 pairs despite identical final codes. Two threads matched 3/3 but were slower than the 4-thread incumbent (27.6727 versus 26.4829 s median) and lack a 3x repeat proof. No thread-config reseal is justified. |
| MLX/Metal heavy forward | **REFUTED** for this live hot-swap formulation | `torch.backends.mps.is_available()` was false in this process; the L70/fp-reorder bit-identity wall remains binding; Metal controls are MAIN-fire-only, so zero Metal fires were used. This does not kill an explicit fixed-order future backend. |
| ANE/CoreML Pose6 forward | **REFUTED** for this live hot-swap formulation | `coremltools` is absent, and the recalled correction ladder is frozen-SegNet evidence, not PoseNet transfer. No PoseNet-specific conversion, placement, exact tuple regeneration, or latency receipt exists in the searched scope. This is not a family-wide ANE kill. |
| CPU x ANE x Metal composition | **REFUTED** for current r9 | CPU pair workers are ready, but neither accelerator component passed its mechanism/identity gate, so a composed speed claim would be invented. CPU-only pair process parallelism is the measured optimum. |

Python `ProcessPoolExecutor` is closed only for this managed-sandbox launch form: its
semaphore/IPC setup failed with `Operation not permitted`. Ordinary subprocess workers
ran successfully and are the retained implementation. Changing what is batched remains
closed by the jo5 batch-shape law; only whole-pair ownership was changed.

## Swap economics

This calculation uses the `03:53:11Z` live observation and charges a deliberately
conservative r9: all 600 current-stage pairs are re-solved plus 600 pairs for each of
the two future stages. It assigns **zero credit** to migration of r8's active pair
receipts and adds 197.227 seconds to rematerialize the current candidate/master fields.

| term | seconds | hours |
|---|---:|---:|
| measured r8 median-hybrid remaining ETA | 26,801.143 | 7.445 |
| r8 remaining pair wall, 1,313 x 14.136891 s | 18,561.738 | 5.156 |
| conservative r9 pair wall, 1,800 x 14.136891 / 3.627303 | 7,015.241 | 1.949 |
| r9 ETA excluding swap cost, with 197.227 s rematerialization | 15,451.873 | 4.292 |
| actual wc2 build + frozen-source re-proof | 1,541.416 | 0.428 |
| measured jo5 r7-to-r8 reseal/resume analog | 370.959 | 0.103 |
| illustrative total swap cost | **1,912.375** | **0.531** |
| r8 - swap cost - r9 | **9,436.895 saved** | **2.621 saved** |
| required `2 x swap cost` | 3,824.750 | 1.062 |
| headroom over binding bar | **5,612.145** | **1.559** |

Thus the measured-rate arithmetic passes with the prior reseal/resume analog. That
analog is not silently promoted to an actual r9 cost. At the observed lower pair rate,
the same inequality allows at most about **34.4 minutes** for the still-actual r9
integration + reseal + resume after charging the already measured 25.7-minute build
and proof. MAIN may fire only after measuring that actual cost, binding the pair-worker
source in the r9 seal, migrating the latest r8 checkpoint without training restart,
and rerunning this live arithmetic. Until then r8 continues uninterrupted.

## ES1 route wall budgets

These are fail-closed budgets for the next governed gate, not convergence promises.
They reuse the measured 1.413278 s live step, 13.5049--16.8137 s live pair band, the
3.627303x pair-worker receipt only after r9 admission, and the 30-minute evaluator wall.
Unknown new-representation training remains labeled as such.

| ES1 stage | wall budget | dominant driver / disposition |
|---:|---:|---|
| 0 JO1 endpoint harvest | **6.21--9.42 h remaining** | live fresh PoseNet pair work; artifact endpoint 10:06--13:18 UTC at observation; conditional input |
| 1 one-member grammar + reference receiver | **<=1 h scorer-free gate** | derived cap: parse/no-op/corruption/roundtrip suite plus current 40-minute-per-stage sealed receiver allowance; actual new receiver unmeasured |
| 2 worldsheet + quotient semantic body | **<=1 h first stratified n32 gate; 2.8--3.5 h maximum per admitted n600 stage-equivalent** | representation training is unmeasured; scale only after serialized negative complete action |
| 3 cumulative JO1 solve | **2.8--3.5 h per n600 stage-equivalent on r8; rederive after r9** | fresh PoseNet compensation dominates; cumulative object only |
| 4 terminal xi/pose re-solve | **2.25--2.80 h r8 pair band; about 0.62--0.77 h if r9 speed transfers** | fresh same-object PoseNet pair solve; never stale carrier |
| 5 sparse exceptions | **<=1 h per two-prefix n32 gate** | repeated scorer/pose closure; stop above 3,000 B or nonnegative complete score |
| 6 factorize/distill/entropy race | **<=1 h per born-stream byte-only race** | real coder itself is 2.543 s/144 controls; model emission/parseback is the unknown. Any retraining routes back to stage 2's budget |
| 7 optional DC1 certificate | **0 h folded; <=1 h only after trigger** | scorer-free certificate build; fire only with >=3,000 B pre-header saving and decoder under wall |
| 8 full authority closure | **<=90 min** | deterministic repeat plus contest CPU and CUDA closures, each under the 30-minute evaluation contract; actual paired wall unmeasured |

## RECALL EVIDENCE

### Stores and queries

- Governing surfaces: `PROGRAM.md`, `CLAUDE.md`, `AGENTS.md`, craft handoff, hot state,
  canonical pointer, charter, common contract, jo5 seal/migration receipts, and ES1 stages.
- Full-corpus content searches covered `#509|wallclock|epoch time|component timer`,
  `thread|BLAS|one-thread|2.96`, `pair process|parallel workers|batch identity`,
  `#477|precision backend`, `#482|ANE|CoreML|precision split|W8A8`, and
  `PoseNet|Pose6` crossed with `ANE|CoreML` across research, state, code, specs, DAG,
  index, and task surfaces.
- Canonical equations were enumerated with `tools/list_canonical_equations.py --json`
  and searched for wall/runtime/throughput/heterogeneous/bit-identity laws.

### Findings beyond the charter seeds that changed the plan

1. `verdict_parallel_workers_speedup_v1` measured 5.686x at eight CPU workers with
   identical advisory values on another n600 scorer-forward workload. It supported
   building process ownership, but its absolute rate was not transferred; JO1 was
   measured directly and landed at 3.627303x for three workers.
2. `witness_fp_reorder_transform_bit_identity_wall_v1` and jo5's same-batch cure
   jointly ruled out microbatch/compile restructuring. The implementation therefore
   copies the serial pair body and changes ownership only.
3. PLAN15 measured thread-pin -3% and batch32 -11% on a later vehicle, refuting transfer
   of the old 2.96x one-thread story. That changed the profile from presumed thread tuning
   to an explicit 1/2/4-thread identity and wall matrix.
4. The ANE correction ladder found no label-grade >=10x frozen-SegNet stack; full fp32
   CoreML reached fidelity but only 3.609x, while fp16 failed fidelity. Because JO1 is
   PoseNet-dominated, those receipts prevent a borrowed ANE claim rather than supplying one.
5. #509's telemetry-first campaign required disjoint real-component timing before a
   lever verdict. That caused the retained PoseNet/retention/carrier/coder decomposition
   to precede swap pricing.

## Custody and non-claims

- The sacred r8 entrypoint remains SHA-256
  `766d3494751b27343df8904db2b74fd21e3d7804274a7e3931316ae11736bcdd`;
  the receiver remains `0ceba3210fdabe504d975d80ad4f480a7f4e53392cda9db07854713763870931`.
- The live run was read only. No process signal, cursor edit, archive mutation, scorer
  launch, Metal fire, ANE fire, n600 production parallel solve, or evaluator ran.
- `smoke_r1`, `profile_r1`, and source-pre-freeze `profile_r2` remain retained. They
  are excluded from primary arithmetic: pool IPC failed in the first, the original
  thread gate stopped the second, and the third predates the final profile-source
  receipt. No evidence was deleted.
- The final profile's pair-worker source is
  `9d7af52f40f62a79164347f9cc6851af5039ebc7471524eff03994888ca143d1`;
  profiler source is `258aeec51ea31aeebc9f3e547567c560080a080605ad48b8aa2a3e7f5420f192`.
- The pair-process result is an offline three-pair mechanism proof. It is not an n600
  throughput guarantee, a score, or permission to bypass r9 source/config sealing.
- Current own-vehicle frontier remains
  **S=0.14821987563243377 @ 180,368 B [contest-CUDA T4 n600]**, archive SHA-256
  `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner MAIN r9 sealer; consumer store `/Volumes/VertigoDataTier/pact/ddm_wc2_jo1_wallclock/r9_seal/`; fire trigger: bind the reviewed pair-worker SHA in a compiled r9 config, integrate `solve_fresh_compensation_parallel`, migrate the latest r8 checkpoint with no training restart, measure actual integration+reseal+resume wall, rerun the live ETA, and fire only if the exact `net_saved >= 2 x swap_cost` inequality still passes (34.4-minute actual-cost ceiling at the observed lower rate).
- **QUEUED-WITH-A-FIRE-ORDER** — owner MAIN/JO1 endpoint harvester; consumer store `/Volumes/VertigoDataTier/pact/ddm_nr1_taskcell_body_rebase/retained/`; fire trigger: r8 or admitted r9 emits a sealed endpoint archive/render/component/fresh-carrier receipt; hand that exact object to ES1/NR1 stage 1 without interrupting the current run.

## LIVE-HYPOTHESES

- Three or four whole-pair CPU subprocesses will retain most of the measured 3.627x
  speedup at n600 because 92% of pair wall is independent PoseNet forward work and the
  three-worker proof already includes model load, retention, live-r8 contention, and
  exact batch replay. The n600 rate and memory curve are still unmeasured.
- An admitted r9 can finish materially earlier even if it re-solves all 600 current-stage
  pairs, because the measured pair-wall reduction is much larger than the current-stage
  redo and prior reseal costs. This remains contingent on actual seal/migration wall.
- A PoseNet-specific ANE port could eventually compose with CPU pair ownership because
  the forward owns almost all pair wall. It is plausible from hardware availability and
  prior CoreML speed, but no PoseNet placement/fidelity/regeneration receipt exists.

## DEAD-ENDS

- Changing exploration batch shape/order is closed by exact evidence: singleton/batch
  and reordered reductions change Pose values. Only complete-pair ownership is admissible.
- One-thread execution is closed as an r8/r9 speed lever on this formulation: it was
  slower and changed Pose6 on 3/3 sampled pairs.
- Two-thread execution is closed as a swap lever on this profile: it matched 3/3 once
  but was slower than the pinned four-thread reference and has no repeat proof.
- `ProcessPoolExecutor` is closed in this managed sandbox because semaphore creation is
  denied. Ordinary subprocess workers are the working replacement; the pair-parallel
  family is not closed.
- MLX/Metal and ANE are closed only as immediate JO1 r9 hot swaps: device/control access,
  PoseNet-specific fidelity, and exact regeneration proof are absent. Accelerator
  families remain open behind those explicit gates.
