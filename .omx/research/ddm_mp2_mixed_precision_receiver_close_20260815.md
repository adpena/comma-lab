# DDM MP2 mixed-precision receiver close: terminal measurement and handoff

## Outcome

The exact contest pointer did not move. Three of the seven original MZ2 candidates were measured through the complete HV1 receiver and the frozen n600 CPU evaluator. All three were rejected on the same advisory axis because PoseNet damage overwhelmed their rate savings. MAIN then stopped the unscreened fanout after the third dose-response point. The four deeper nested-prune archives remain receiver-closed and retained, but were not scored.

The keep75-versus-keep87 dose response exposed one new candidate worth preserving: prune only the 23 rows per FiLM tensor that keep75 removes after keep87. That differential generation is receiver-closed and queued, but not scored because MAIN owns the scorer lane for the WD3 teacher-cache build. Its exact standalone archive saving is only 25 B, not the 341 B conditional archive difference between keep75 and keep87.

Admission required exact recomputed `Delta S < -3.5e-6` versus the same-axis HV1 control:

`S = 100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37_545_489`

| Candidate | Archive SHA-256 | Bytes | Delta bytes | d_seg | d_pose | Delta d_seg | Delta d_pose | Exact recomputed S | Net Delta S | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| HV1 control | `80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e` | 182,759 | 0 | 0.00042714 | 0.00014747 | 0 | 0 | 0.20280753928705508 | 0 | same-axis control |
| mixed q3/q4 | `6c46ff65c55bf09745e05846d51b094b2e9fb69327f096fb615aac393305f1b5` | 181,936 | -823 | 0.00042828 | 0.00073123 | +0.00000114 | +0.00058376 | 0.24948370195898223 | +0.046676162671927146 | REJECT, INSTANCE |
| FiLM keep87 | `efa5febeb7ca82aea4b2f9dcedc86e7b83b211c54f190a4a1fb40e4bda998880` | 182,629 | -130 | 0.00042778 | 0.00068390 | +0.00000064 | +0.00053643 | 0.24708140140589041 | +0.044273862118835328 | REJECT, INSTANCE |
| FiLM keep75 | `ddf26d22e15425c681a3903959d8de7edd640f7471ec3d808088c5f7fb75dfc0` | 182,288 | -471 | 0.00042821 | 0.00063959 | +0.00000107 | +0.00049212 | 0.24417346774141535 | +0.041365928454360268 | REJECT, INSTANCE |
| FiLM keep62 | `659e3b575179d4da43e00eeaed6dffc08911509de6199f0bc285a9f6dda45d44` | 182,011 | -748 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | HELD, pose-field prerequisite |
| FiLM keep50 | `45cbd8a4a4b4cf16afe26ae5120a7ab4d3ae7ec19021d44c42cd086b8deaefd9` | 181,694 | -1,065 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | HELD, pose-field prerequisite |
| FiLM keep37 | `71932b11d1101147a40a016f73759dd8e7a632627c0d226593ba6715aa1c227d` | 181,235 | -1,524 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | HELD, pose-field prerequisite |
| FiLM keep25 | `0359fd3370e91057b17edfcf387b47d6af8c999f74bc6e4cf718b27326fc15b0` | 180,708 | -2,051 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | HELD, pose-field prerequisite |
| FiLM marginal differential | `37194782ed5c01bea33f1514684e529a1b082327ef109ba4c37d7e188c0525fa` | 182,734 | -25 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | QUEUED, receiver-closed |

All measured score rows are `[macOS-CPU env-mismatch advisory]`, n600, exact archive bytes through `experiments/contest_auth_eval.py` and the mirror's frozen `evaluate.py`. The mirror content SHA was `fa7c4bf51d47a6140ec0f95275ebf86b0e6c3c1dc00caff03a417ee989645799` after the required contamination cleanup. The evaluator environment reports `uv_group_not_declared`; these rows are not contest-CPU or contest-CUDA authority and cannot move the pointer.

The component arithmetic, not the evaluator's rounded display, decided every verdict. For example, keep75's score change decomposes as:

- Seg: `+0.0001069999999999988`
- Pose: `+0.04157254802128083`
- Rate: `-0.0003136195669205427`
- Net: `+0.04136592845436027`

No original candidate was admitted, so there is no sealed T4 fire order and no Modal dispatch. This arm spent $0.

## Receiver closure and retained custody

The original build produced 8/8 complete generations: one HV1 control and seven MZ2 candidates. The later marginal differential is a ninth complete generation. Every generation has its own runtime tree with exact `ARCHIVE_SHA256` and `ARCHIVE_BYTES` pins, an exact archive repeat, retained model/member/semantic/carrier/tail payloads, retained outer sections, and 38/38 independently decoded semantic tensors equal to the packer's intended state.

Primary retained root:

`/Volumes/APDataStore/pact/ddm_mp2_mixed_precision_receiver_close_20260815`

Original receiver-close receipts:

- `BUILD_RESULT.json`: SHA-256 `8135647d4ef7dcfde309682b80afd8c616bf730ee2fd40daec98deb5997c6f36`, 8/8 complete, 7/7 original candidates receiver-closed.
- `FINAL_RESULT.json`: complete scorer-free receiver-close inventory.
- MZ2 source custody: `/Volumes/VertigoDataTier/pact/ddm_mz2_frozen_section_representation/RETENTION_INVENTORY.json`, SHA-256 `156112d0a0b8caeec0f0a6eaedd3bc1d24e2d389b199dad2495324ebd6c2dbcc`.

Differential receipts:

- `DIFFERENTIAL_RESULT.json`: SHA-256 `adc3e97b4e216dd375cd2cf9fdfa897ace0a8225139cb3c80978a57cc63cbf4f`.
- Generation receipt: SHA-256 `3b53b2310ef23d6b934fe8793392873c523672ff58ab412b1ed9a0ba99b2794f`.
- Archive and retained repeat: both SHA-256 `37194782ed5c01bea33f1514684e529a1b082327ef109ba4c37d7e188c0525fa`, 182,734 B.
- Selection map: SHA-256 `70c1bd37d9308c7a76536dd78f97a8ae50a98e30c534c39210873604cb2ecd3e`, with all 23 pruned row indices for each of `blocks.1.film.weight`, `blocks.2.film.weight`, and `blocks.3.film.weight`.
- The selection map reverified the exact retained keep75 raw packet `abf8c75a...` and keep87 raw packet `ed7b166b...` before deriving the set difference.
- Differential receiver pins: `ARCHIVE_SHA256=37194782...`, `ARCHIVE_BYTES=182_734`; 38/38 semantic tensors equal; deterministic seed 20260815.

The differential's 25 B rate credit is `-0.000016646473828054283 S`. At unchanged d_seg it permits only about `+1.01e-7` d_pose before missing the admission bar. That is a tight hurdle, but the conditional keep75-minus-keep87 comparison measured a pose improvement of `-4.43e-5 d_pose`, so one exact advisory row remains justified. The conditional signal does not prove the standalone effect because the bottom-row set is restored in the differential candidate.

Measured work directories, candidate checkpoints, and 3,662,409,600-byte raw outputs remain retained for all three scored candidates under `advisory_n600_cpu/<candidate>/attempt_0000/work`. Result SHA-256 values are:

- mixed q3/q4: `8dd50626a93aa692ebd34d7aa9300b4f49e3d1e0c28c9f35d0875454802ff99a`
- keep87: `7934a4c6bced41b036d0ffc84bd7ca3bba8e9d90521941559b7ac508f49009b7`
- keep75: `f2a0919d11c7187eba48d7078c10c4504d4ccce2387ed9e38d0e6811bfe58fb0`

## Queue boundary

The original serial queue exposed one apparatus defect: `launch_detached_process.py` appends `.done`, while the first queue revision supplied a name already ending in `.done`. This produced real receipts named `.done.done` and left the queue waiting on the wrong path. Commit `0286280f95` fixes the class for future launches; exact hard-link/symlink aliases preserved the already-running receipts without altering scored payloads.

After keep87 was rejected, MAIN Relay 6 required keep75 to finish and prohibited further unscreened n600 fanout. The managed sandbox denied a targeted `SIGTERM` to queue PID 77093, so the keep75 expected-receipt symlink was moved to a preserved pause name. The real keep75 rc=0 `.done.done` receipt landed, but the queue could not observe it or launch keep62. The queue state therefore still labels keep75 `RUNNING`; the terminal evaluator JSON and rc=0 receipt are the authority. No keep62 attempt directory was created.

The differential advisory row has a separate sealed fire order:

`/Volumes/APDataStore/pact/ddm_mp2_mixed_precision_receiver_close_20260815/DIFFERENTIAL_N600_FIRE_ORDER.json`

Its disposition is `QUEUED-WITH-A-FIRE-ORDER`; owner `MAIN`; consumer store `.../advisory_n600_cpu/score_gated_film_row_prune_keep75_minus_keep87/attempt_0000/contest_auth_eval.json`; fire trigger: MAIN confirms the WD3 teacher-cache scorer lane has released, no full-n600 scorer is active, all archive/runtime hashes match, and `/usr/bin/find` has deleted every `._*` file from the generation, attempt directory, and pinned mirror. The row must retain its complete work directory and is admitted only at exact recomputed `Delta S < -3.5e-6`.

## Stage 2: first measurement and sealed design

The first carrier measurement is complete. The physical carrier body is 22,219 raw bytes; the logical basis-plus-coefficient pool routed by RFO2 is 22,032 B; the shipped Brotli carrier is 22,161 B. A deterministic Brotli quality 0 through 11 race retained every payload and repeat and proved exact decode for 12/12 cells:

| Cell | Exact bytes | Delta versus incumbent | Verdict |
|---|---:|---:|---|
| q0 through q9 | 22,223 | +62 | worse |
| q10 | 22,170 | +9 | worse |
| q11 | 22,161 | 0 | exact tie |

Receipt: `/Volumes/APDataStore/pact/ddm_mp2_mixed_precision_receiver_close_20260815/carrier_rank/CARRIER_EXACT_BYTE_RACE.json`. Source raw SHA-256 is `065fce08fc3d44e49d29ad624561cbef86d01282cc73dcd32533b5d63115bd9f`; incumbent compressed SHA-256 is `fd14aabcb9daa5f1dd1c9c6e63e745a88f2978766e3129b184dd3a9ac7334de0`. The q11 challenger is equal-sized and exact, not a saving. This closes only the exact Brotli-quality instance, not carrier rank/refit.

The remaining rank/refit design is sealed as follows:

1. Measure a per-row/per-coefficient finite-difference pose-sensitivity field at the actual candidate quantum through the real receiver, R/uint8 path, and frozen scorer. Use strided or stratified pairs, never a prefix. Persist every perturbation payload and its selection coordinates.
2. Race global carrier rank `r10/r8/r6/r4` with coefficient refit against adaptive per-cell/per-coefficient sub-int16 quantization. Rank-only atom dropping without refit is not the mechanism.
3. Put exact nonlinear pose contribution, d_seg, and real coded bytes in the selection metric. A bytes-plus-seg waterfill repeats the measured MP2 failure by construction.
4. Persist the adaptive depth map and all per-rank checkpoints, candidate streams, archives, repeats, receiver parse-backs, and complete work directories. Re-price every selection after joint remeasurement rather than assuming independent credits.
5. Promote only receiver-closed survivors into serial n600 advisory rows after the scorer lane is free. The same `Delta S < -3.5e-6` admission rule applies.

Disposition: `QUEUED-WITH-A-FIRE-ORDER`. Owner: `MAIN` / MZ2 carrier-structure successor. Consumer store: `/Volumes/APDataStore/pact/ddm_mp2_mixed_precision_receiver_close_20260815/carrier_rank/`. Fire trigger: the differential row is terminal or folded, the WD3 scorer lane is released, the exact output bank is under custody, and the FD pose field has closed before any lossy selection.

The lossy rank ladder itself was not built or measured in this arm. The charter explicitly allowed stopping Stage 2 at design plus first measurement when wall clock demanded; the Stage 1 n600 chain and Relay-7 differential consumed that budget.

## RECALL EVIDENCE

Sources and queries searched:

- `rg -n -i "mixed[- ]precision|q3/q4|film[-_ ]row|row prune|rank[- ]reduced|rank/refit|22,0(32|19|61)|carrier pool|pose[-_ ]sensitivity"` over the MZ2, RFO2, HV1, WD2, and MP2 relay memos.
- The same surface queries over `.omx/research/CANONICAL_RESEARCH_INDEX*`, `.omx/research/sub015_DAG_*`, and `.omx/state/main_hot_state.md`.
- `.venv/bin/python tools/list_canonical_equations.py --json`, filtered for contest score, rate, pose, quantization, and waterfill laws.
- Governing specs and task surfaces reached from those hits, including the WD3 operator amendments for adaptive quantum, regions/cells, selective placement, and surgical targeting.

Findings beyond the charter seeds and how they changed the plan:

- MZ2's retained archives were built on the older e480 surface. They could not be scored as HV1 candidates by byte arithmetic. This forced a new candidate-bound HV1 runtime tree and exact receiver parse-back for every candidate.
- RFO2 forbids adding mixed-q3/q4 and FiLM byte credits because they overlap. It also requires rank reduction to include coefficient refit. Those constraints removed an invalid additive projection and shaped Stage 2.
- The existing `pz4a` pose-coefficient sensitivity coarsening is a dead end at +2,232 B. The open field is semantic/FiLM sensitivity through the render, using the `ms6` probe pattern. This prevented repeating the wrong coefficient surface.
- Prefix pose probes are anti-conservative by 2.5 to 4.2 times on this population. The follow-on therefore requires strided or stratified pairs.
- MAIN Relay 2 added adaptive per-cell/sub-int16 quantization; Relay 3 required persistent per-row selection attribution; Relay 4 and 6 made pose sensitivity a prerequisite; Relay 7 identified the conditional pose-improving marginal set. These changed the run from a seven-row blind sweep into a bounded three-row dose response plus one receiver-closed differential candidate.
- The canonical contest score and precision-waterfill laws confirmed that all decisions must use exact nonlinear pose contribution and exact coded bytes. The exact carrier race then closed the lossless-coder instance at a tie.

## Implementation and verification

Scoped commits:

- `13dca094ca`: receiver-close MZ2 semantic candidates.
- `d56b829a29`: ExFAT-resumable generation copy.
- `088d7dee08`: tagged semantic outer receiver closure.
- `319b16dd30`: serial retained advisory queue.
- `0286280f95`: detached receipt suffix class fix.
- `e090423ac1` and `253f0e8648`: retained, crash-resumable carrier exact-byte race.
- `34f0cc9a0a`: receiver-close pose-positive FiLM differential.

Focused verification after the final code edit:

- 11/11 receiver tests passed.
- Ruff passed on all three touched Python files.
- Python compilation and `git diff --check` passed.
- Two clean `review_tracker.py mark-file --status reviewed` passes covered each touched Python file after the last fix.
- Differential materialization repeated deterministically with unchanged result, fire-order, archive, and repeat hashes.

## LIVE-HYPOTHESES

- The 23-row marginal differential may improve pose despite its tiny 25 B rate credit. It is plausible because adding those exact removals to keep87 reduced d_pose by `4.43e-5`; the standalone interaction remains untested because the bottom 25 rows are restored.
- A sparse pose-null or pose-positive subset may exist inside the three FiLM matrices. It is plausible because more pruning improved pose across keep87 to keep75 instead of worsening it monotonically.
- Carrier rank/refit may save real bytes after exact recoding tied. It is plausible because rank reduction changes the representation, while the completed test changed only the lossless Brotli quality.
- Adaptive per-cell depth may dominate global rank on the carrier pool. It is plausible because the scorer reads cells and axes unevenly, but it must include the measured pose field or it repeats the mixed-q3/q4 failure.

## DEAD-ENDS

- Reusing the retained MZ2 archives directly on HV1 is closed: they are not candidate-bound to the HV1 archive/runtime pins.
- Treating mixed q3/q4 as a rate-only win is closed at INSTANCE scope: its real n600 row was `+0.0466761627 S`, driven by pose.
- Treating keep87 or keep75 as rate-only wins is closed at INSTANCE scope: their real n600 rows were `+0.0442738621 S` and `+0.0413659285 S`.
- Continuing the blind keep62/50/37/25 n600 sweep is closed until the pose-sensitivity prerequisite is met: the first three rows already exposed a shared pose-toxic surface, and MAIN explicitly stopped unscreened fanout.
- Exact Brotli quality search on the current carrier is closed at INSTANCE scope: 12/12 exact rows produced a q11 tie and no byte saving.
- Re-running `pz4a` on the pose coefficients is closed: the prior sensitivity-allocated representation cost +2,232 B; MP2 needs the semantic/FiLM-through-render field instead.
- Transferring the conditional `-341 B` keep75-minus-keep87 difference as the differential's standalone rate prize is closed: the receiver-closed differential saves exactly 25 B.

Vehicle frontier unchanged: **HV1 ep0634 S = 0.15959729295498598 @ 182,759 B [contest-CUDA T4, n600]**, archive SHA-256 `80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e`.
