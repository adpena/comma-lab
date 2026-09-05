# ddm_ane1 — ANE placement is PROVED, and it buys nothing on the pose axis: the fp16 screen ranks at chance

Arm: `ddm_ane1_ane_screening` (2026-09-05). Tokens: `[no-triality] [p0-ledger-ok]`.
Lane: `lane_ddm_ane1_ane_screening_20260905`. Craft contract: `docs/operating_manual_craft_handoff.md`.
Axis of every row below: **`[macOS-CPU/ANE advisory]`**, frozen scorers, real n600 inputs.
`score_claim=false`, `promotable=false`. **Pointer: UNMOVED. This arm bought no exact row.**

## ANSWER FIRST

1. **Placement is PROVED, per operation, for the first time.** `MLComputePlan` reports the compute
   device of every op. On `CPU_AND_NE`: **SegNet fp16 = 298/298 ops (100.0%) on
   `MLNeuralEngineComputeDevice`**, **PoseNet fp16 = 287/287 (100.0%)**. The 2026-07-13 parent lane
   could only ever say "we requested `CPU_AND_NE`"; it is now a measurement.

2. **fp32 can NEVER reach the ANE — 0.0% of ops, both scorers, proved per-op.** `segnet_fp32` on
   `CPU_AND_NE` = 297/297 on the CPU; `posenet_fp32` = 286/286 on the CPU. Requesting the ANE for an
   fp32 graph is worse than not asking: SegNet fp32 `CPU_AND_NE` medians **96.81 ms** against
   **85.77 ms** for `CPU_ONLY`. This closes the parent lane's open question — its FLOAT32 rung was
   never going to be an ANE rung, structurally.

3. **The charter's prior-law prediction is INVERTED, and the inversion is the finding.** It predicted
   the discrete argmax would be the fragile axis and the smooth 6-dim regression the tolerant one.
   MEASURED at n600 on real frames: SegNet fp16 misses its authority bar by **1.46×**; PoseNet fp16
   misses its axis by **1,448×** (against the T4 exact d_pose 7.77e-06; 1,767× against pr1's advisory
   base 6.366e-06). A discrete read spends its top-2 MARGIN; an MSE against a target
   the model already nearly hits has **no slack at all**.

4. **The pose screen is dead, and not by a little.** Replaying pr1's 39-point selector sweep with all
   8 modes confirmed on `cpu_torch`: argmin agreement **4/39 = 10.26%** against a chance rate of
   12.5%, Kendall tau-b median **0.0714**, **34 of 39** screened picks are *worse* than the shipped
   mode once confirmed, and adopting the screen's choices would move total d_pose by
   **−4.728e-02** where pr1's CPU sweep gains **+1.208e-04** — the wrong direction by **391×**. The
   charter's falsifier ("PoseNet fp16 per-pair drift ≥ the sweep's adoption deltas") **fired**.

5. **Even a perfect screen could only buy 1.36×, because 48% of the sweep's cost never leaves the
   CPU.** MEASURED inside one run: 312 screened forwards in 24.30 s (77.90 ms each) against 312
   confirmed forwards in 50.01 s (160.28 ms each) — **2.06×**, not the 74× the trunk latency
   advertises. The residue is `render_frame0` + `preprocess_input`, which stay in torch. The 74× is
   real and irrelevant; Amdahl eats it.

6. **The acceleration that IS admissible is not the ANE — it is CoreML fp32 on the CPU.**
   `coreml_cpu_fp32` gives **0 argmax flips in 117,964,800 pixels** (bit-exact against 1-thread
   CPU-torch across all 600 pairs) at **3.28×**, and a pose drift of **2.43e-12** median self-MSE —
   **3.1e-07** of the live d_pose — at **5.12×**. It needs no screening contract because it is not a
   screen.

7. **The exact-SegNet hybrid prices well on paper and is blocked by a measured negative.** A band
   containing every fp16 flip is **0.357% of pixels** (max flip margin 0.4456, area read off the
   n600 all-pixel margin census). The 3× bar tolerates **31.77%** recompute, so the arithmetic clears
   with **89× of headroom**. But the arithmetic assumes recompute cost is proportional to pixel area,
   and the parent lane MEASURED the opposite: its 5%-band tile recompute cost **1,091.6 ms** against
   a 255.4 ms dense CPU pass — **4.27× the dense pass it was replacing**. So: **priced GO on area,
   measured NO-GO on realization.** Not built. The next step is a sparse-recompute kernel measurement,
   not more arithmetic.

## PRIOR-LAW PREDICTION vs OUTCOME (the owed line, counted plainly)

| charter claim | outcome |
|---|---|
| SegNet fp16 INADMISSIBLE for d_seg, flip rate ≥ 1e-2 at n600 | **half right.** Inadmissible: yes (1.46× the bar). Rate ≥ 1e-2: **NO** — measured 4.818e-05, **208× lower than the floor the charter predicted** and 513× below the parent lane's n24 number. |
| PoseNet fp16 fp32-vs-fp16 MSE delta lands 1e-8–1e-7 per pair | **FALSIFIED by 5 orders.** Measured median self-MSE **1.125e-02**. |
| PoseNet fp16 ADMISSIBLE for screening, ≥95% rank agreement on the adopted set | **FALSIFIED.** 10.26% argmin agreement — at chance. |
| ≥10× forward speedup for the screen | true for the trunk (**74.14×**), **false end-to-end (2.06×)**. |
| FALSIFIER: placement cannot be proved → rename the lane honestly | **did not fire.** Placement is proved per-op. |
| FALSIFIER: PoseNet fp16 drift ≥ the sweep's adoption deltas | **FIRED.** |

Two of the charter's own falsifiers were available; the one that fired is the one that decided the
lane. That is the falsifier working as designed, and it cost about four minutes of ANE time to fire.

## 1. PLACEMENT — MEASURED per-op, not inferred from latency

`coremltools 9.0`, deployment target `iOS26` (the macOS 26 target; coremltools 9 drops the `macOS*`
aliases, so the ladder is read off `ct.target` and never guessed), macOS 26.4 build 25E246, Apple M5
Max. `MLComputePlan.load_from_path` needs a compiled `.mlmodelc`, so each `.mlpackage` is compiled
once beside itself and the compiled tree is retained.

| model | precision | shape | `CPU_AND_NE` device split | ANE op fraction |
|---|---|---|---|---:|
| SegNet | fp16 | (1,3,384,512) | NeuralEngine 298 | **100.0%** |
| SegNet | fp32 | (1,3,384,512) | CPU 297 | **0.0%** |
| PoseNet | fp16 | (1,12,192,256) | NeuralEngine 287 | **100.0%** |
| PoseNet | fp32 | (1,12,192,256) | CPU 286 | **0.0%** |

Latency, median of 30 reps after 3 warmups, same input each time:

| model | `CPU_ONLY` | `CPU_AND_GPU` | `CPU_AND_NE` | `ALL` | cpu-torch fp32 1-thread | ANE speedup |
|---|---:|---:|---:|---:|---:|---:|
| SegNet fp16 | 58.78 ms | 49.22 ms | **4.91 ms** | 4.37 ms | 313.74 ms | **63.84×** |
| SegNet fp32 | 85.77 ms | 54.90 ms | 96.81 ms | 59.45 ms | 317.18 ms | 3.28× |
| PoseNet fp16 | 9.03 ms | 10.48 ms | **1.08 ms** | 1.32 ms | 79.96 ms | **74.14×** |
| PoseNet fp32 | 12.50 ms | 13.21 ms | 14.98 ms | 16.85 ms | 76.64 ms | 5.12× |

Two things the per-op census says that a latency triad alone could not. First, `ALL` and
`CPU_AND_NE` are the same placement for fp16 — the planner does not split. Second, the fp32 rows'
`CPU_AND_NE` latency being *worse* than `CPU_ONLY` is not noise: the plan shows zero ANE ops, so the
request buys a partition attempt and nothing else.

**A shape correction worth recording.** PoseNet's trunk does not see the eval lattice. Its
`preprocess_input` resizes to (384,512) and *then* applies `rgb_to_yuv6`, which folds each 2×2 luma
block into 4 channels and halves both spatial dims — so the trunk input is **(1,12,192,256)**, a
quarter of the area. The first conversion in this arm used (1,12,384,512) and its placement and
latency rows were discarded. Every PoseNet number above is at the shape the real preprocess emits,
verified by running that preprocess on a real pair.

## 2. FIDELITY at n600 on real frames

Inputs are the **shipped body's own decode** — `.../ddm_to1/advisory/attempt_0002/work/inflated/0.raw`,
2×600 frames at 874×1164 — i.e. the exact frames the pose instrument scores. Reference is 1-thread
CPU-torch fp32, the authority form. Frozen weight custody: SegNet sha
`68956e328d4c…`, the same digest the 2026-07-13 lane recorded — the difference below is not the
weights.

### SegNet — argmax disagreement, bar 3.3e-05

| route | flips / px | flip rate | × bar | per-pair median | p95 | max |
|---|---:|---:|---:|---:|---:|---:|
| `ane_fp16` | 5,684 / 117,964,800 | **4.818387e-05** | **1.4601** | 4.069e-05 | 9.155e-05 | 4.425e-04 |
| `coreml_cpu_fp32` | **0** / 117,964,800 | **0.0** | 0.0 | 0.0 | 0.0 | 0.0 |

Every one of the 600 pairs has at least one fp16 flip; the median pair loses 8 pixels of 196,608 and
the worst loses 87. The flip sites are not razor ties: the top-2 margin at a flip has median
**0.01523**, q90 **0.05963**, q99 **0.20900**, max **0.44560**.

**The 513× gap to the parent lane, and what it is not.** 2026-07-13 measured 2.4746e-02 on 24 GT
frames; this arm measures 4.8184e-05 at n600. Three things differ, and I can rule out exactly one.
*Ruled out — sample selection:* the n24 contiguous prefix of **my own** run gives 3.7935e-05, only
0.787× my n600 rate, so [[m88]] prefix bias explains none of it. *Not separated:* (a) the inputs —
GT frames there, the shipped decode here; (b) the toolchain — coremltools 9.0 / iOS26 / macOS 26.4
here, against a build whose own memo records an E5RT cache failure, an unconfirmed placement, and a
fallback graph that scored 3.4667e-02. I will not pick between (a) and (b). The experiment that
separates them is one run of this instrument's `fidelity` over `gt_n600.npz['gt_f1']`, which needs a
frames-source flag this arm did not add.

### PoseNet — 6-dim pose drift

Two denominators, both stated: the T4 exact **d_pose = 7.77e-06**
(`ddm_up2_shipping_pose_solve.py:92`) and pr1's advisory n600 base **6.366e-06**. The `×`-the-axis
claims below use the T4 exact value; against the advisory base every ratio is 1.22× larger.

| route | \|Δ\| median | \|Δ\| p95 | \|Δ\| max | rel median | rel p95 | self-MSE median | self-MSE max |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ane_fp16` | 1.2914e-03 | 3.3036e-01 | 8.4889e-01 | 9.08% | 142% | **1.1250e-02** | 1.2010e-01 |
| `coreml_cpu_fp32` | 4.4703e-08 | 7.6294e-06 | 3.0518e-05 | 3.0e-06 | 7.1e-05 | **2.4348e-12** | 1.5522e-10 |

**599 of 600 pairs** carry an fp16 self-MSE larger than the entire d_pose being measured
(median self-MSE **1,448×** the T4 exact d_pose).

**The whole failure is one output dimension, and that is the mechanism.** Per-dim:

| dim | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| \|ref\| median | **31.16** | 2.07e-02 | 1.91e-02 | 3.58e-03 | 3.72e-03 | 1.28e-02 |
| \|Δfp16\| median | 0.2598 | 2.3e-03 | 2.7e-03 | 5e-04 | 6e-04 | 9e-04 |
| relative | **0.83%** | 11.2% | 13.8% | 13.3% | 15.0% | 8.8% |
| share of fp16 MSE | **99.96%** | 0.02% | 0.01% | ~0 | ~0 | ~0 |

Dimension 0 has the **best** relative accuracy of the six and causes **99.96%** of the damage,
because its magnitude is 31 and the residual the axis must resolve is √(7.77e-06) ≈ **2.79e-03**
(2.52e-03 against the advisory base).
fp16 carries ~1e-3 relative precision; on a value of 31 that is 0.03 absolute, ten times the entire
residual. **fp16's error is relative; d_pose is absolute.** No head-precision split rescues this on a
frozen graph, because the fp16 trunk has already spent the precision before the head sees it.

## 3. SCREENING — wired with a contract, replayed, and refused by the measurement

`tac.ane_screening` holds the contract; `--scorer-backend {cpu_torch,coreml_cpu_fp32,ane_fp16_screen}`
is on `ddm_pr1 selector` (which RANKS) and on `ddm_fs1 measure` (which PRICES, and therefore
**refuses** every non-authority backend before it touches disk — the guard is the first statement in
`run_measure`, ahead of the archive read and the instrument build, and a test proves the ordering by
running both paths against a nonexistent runtime). The contract:
`assert_cpu_confirm_contract` raises unless every ADOPTED pair was re-measured on `cpu_torch`, and
`screening_receipt` emits both values, the backend name, the `.mlpackage` sha256, the coremltools
version, and `score_claim=false`.

`--confirm-all-modes` exists because a partial confirm cannot state rank agreement: with 2 of 8
modes confirmed, six entries of the comparison are the screen compared against itself. The default
production path reports `screened_picks_that_survive_confirmation` instead and says so in the field.

### The replay — pr1's 39 pairs (ratio > 1.01), all 8 modes confirmed

| quantity | value | bar |
|---|---:|---|
| argmin agreement, screen vs `cpu_torch` | **4 / 39 = 10.26%** | ≥ 95% (charter) |
| chance rate (8 modes) | 12.5% | — |
| Kendall tau-b, median over pairs | **0.0714** | — |
| screened `best_mode` equal to pr1's CPU sweep | 4 / 39 | — |
| screened picks with a positive confirmed gain | **5 / 39** | — |
| screened picks strictly worse than shipped | **34 / 39** | — |
| total confirmed gain if the screen were adopted | **−4.728e-02** | pr1 CPU sweep **+1.208e-04** |
| worst single confirmed gain | −7.599e-03 | — |
| screen never picks identity (mode 0) | 0 / 39 | pr1 CPU picks it 1 / 39 |

The last row is the tell: fp16 noise of order 1e-2 sits on top of between-mode differences of order
1e-6, so *some* perturbation always looks better than doing nothing. The screen is not a noisy
oracle. It is a random one that is biased against the correct answer.

### The wall-clock, measured inside one run — 74× becomes 2.06×

| leg | forwards | seconds | per forward |
|---|---:|---:|---:|
| screen (`ane_fp16_screen`) | 312 | 24.304 | **77.90 ms** |
| confirm (`cpu_torch`) | 312 | 50.006 | **160.28 ms** |

Subtracting the 1.08 ms ANE trunk leaves **76.82 ms** of torch `render_frame0` + `preprocess_input`
per forward that no backend touches. That implies a CPU trunk of 160.28 − 76.82 = **83.46 ms**,
against **79.96 ms** measured standalone — a 4.4% cross-check that the decomposition is real. The
trunk is **52.07%** of a forward, so a *free* trunk caps the sweep at **2.086×**, and the production
scheme (8 screened + 2 confirmed per pair) caps at **1.359×**.

**Say it plainly: the 74× is a property of the trunk, not of the instrument.** Anyone reading the
placement table and budgeting a 74× sweep would be wrong by 54×.

## 4. THE EXACT-SegNet HYBRID — PRICED, NOT BUILT

Bit-exact argmax requires an fp32 recompute band that contains **every** fp16 flip, so its width is
the max flip margin (0.44560) and its area is read from the n600 all-pixel top-2 margin census
(600 pairs, 117,964,800 pixels, 122 log-spaced bins).

| band width | flip coverage | pixel fraction |
|---:|---:|---:|
| 0.01523 (q50 of flips) | 50.0% | 0.0066% |
| 0.05963 (q90) | 90.0% | 0.0264% |
| 0.08812 (q95) | 95.0% | 0.0405% |
| 0.20900 (q99) | 99.0% | 0.1207% |
| **0.44560 (max)** | **100%** | **0.3570%** |

Cost model, proportional recompute (`T = T_ANE + f · T_cpu_dense`):

| recompute fraction | hybrid total | speedup vs cpu-torch 1-thread | ≥ 3× |
|---:|---:|---:|---|
| **0.357% (bit-exact)** | **6.03 ms** | **51.99×** | yes |
| 5% | 20.60 ms | 15.23× | yes |
| **31.77% (break-even at 3×)** | 104.58 ms | 3.00× | boundary |
| 50% | 161.78 ms | 1.94× | no |

The area half of the bar clears with **89×** of headroom (0.357% needed against 31.77% allowed).

**Why this is not a GO.** The proportional model is a lower bound, and the parent lane measured the
true cost of exactly this shape: its R3 rung recomputed a **5%-area** band as 64×64 cores with a
32 px halo and donated SE gates, and it cost **1,091.6 ms** against a **255.4 ms** dense CPU pass —
**4.27× the pass it replaced** — because a spatially dispersed band lights up a median 22.5 of 48
tiles. A 0.357% band is more dispersed, not less: 5,684 flips over 600 frames is **9.5 flips per
frame**, scattered. The binding unknown is tile occupancy, and a proportional-area model cannot see
it. Two further gaps, named rather than assumed: the band must be selected from the **fp16** margin
at inference (whether the fp16 margin selects the same band is UNMEASURED), and a U-Net "pixel"
recompute is really a tile-plus-halo recompute whose halo the parent lane measured as changing local
activations even with donated global gates.

**Verdict: the pixel-area half is GO with large headroom; the compute-realization half is the only
open question and the only measurement of it is NO-GO.** The next unit is a measured sparse
recompute kernel, not another price.

## 5. WHAT IS ACTUALLY WORTH TAKING FROM THIS LANE

**`coreml_cpu_fp32`, not the ANE.** It is bit-exact on SegNet's argmax across 600 pairs (0 flips in
117,964,800 pixels) at **3.28×**, and 3.1e-07 of d_pose on PoseNet at **5.12×**, against the
1-thread CPU-torch authority form. It is not a screen and needs no confirm. The instruments already
accept `--scorer-backend coreml_cpu_fp32`; `fs1 measure` refuses it today only because the refusal is
written against `backend_is_authority`, which is the correct conservative default until an arm
measures a full byte-closed row through it. That is the cheapest real speedup this lane found, and it
is on the CPU.

Two successor experiments, both cheap, neither run here:

1. **d_seg screening on SegNet fp16.** Not for a d_seg *value* — that is closed — but for RANKING two
   candidate archives. The fp16 flip set is largely determined by the image, so the *difference*
   between two candidates' flip sets may be far smaller than 4.8e-05. Between-candidate d_seg
   differences in this campaign are ~1e-05, so the question is whether the common mode cancels. One
   run of `fidelity` on two archives' decodes answers it. The pose replay is the cautionary control:
   common-mode cancellation is exactly what failed there.
2. **The GT-frame control** that separates inputs from toolchain in §2's 513× gap. One flag, one run.

## Equations leg (`tac.canonical_equations`)

Registered **`scorer_fp16_drift_by_axis_v1`** — *fp16 scorer drift is priced by the READING AXIS, not
by the architecture*. `A_axis = ε₁₆·‖y‖∞ / slack_axis`, with `slack_argmax = m_top2` and
`slack_MSE = √d_pose`. Two empirical anchors from this arm's n600 rows:
`ane1_segnet_fp16_argmax_flip_n600_20260905` (predicted 750× the bar, measured **1.4601×**, residual
513.66) and `ane1_posenet_fp16_pose_mse_n600_20260905` (predicted 0.013× its bar, measured
**1448×**, residual 111,371.56). Producers: `experiments/ddm_ane1_ane_screening.py`. Consumers:
`tac.ane_screening`, `ddm_pr1 selector`, `ddm_fs1 measure`.

I did **not** hang these on `mps_drift_architecture_class_dependent_v1`, whose
`domain_of_validity` literally reserves `segnet_class_pending` and `posenet_class_pending`. Its output
unit is a *drift-reduction factor* from Kahan/softmax corrections; mine is raw drift against an axis
bar. Forcing an anchor across that unit mismatch would have been a tidy lie. The new equation names
it as `parent_family` instead, and its residuals are the honest statement that both predictions were
wrong in opposite directions.

## Apparatus

- `src/tac/ane_screening.py` — the contract: backend roster, `assert_cpu_confirm_contract`,
  `screening_receipt`, `mlpackage_provenance`, `sha256_tree` (content-addressed, directory-aware),
  `rank_agreement` (SciPy-free, so it imports from the main venv). `coremltools` is imported lazily;
  the main venv does not carry it and importing `tac.ane_screening` must never break a CPU caller.
- `experiments/ddm_ane1_ane_screening.py` — `convert` / `placement` / `fidelity` / `margins` / `price`.
- **28 tests**, `src/tac/tests/test_ane_screening.py`: backend roster and authority naming; the
  confirm contract (no confirm, partial confirm — the failure names the missing pairs — full confirm,
  superset); receipt content and `score_claim=false`; provenance (file hash, tree hash is
  content-addressed not mtime-addressed, changes on any member byte, refuses a missing path); loader
  identity for `cpu_torch` and refusal without a package; rank agreement; and the wiring — both CLIs'
  flags, the default being the authority backend, an invented backend name rejected, and a runtime
  proof that fs1's refusal fires before any disk access while the authority path proceeds.
- Runbook: `/Volumes/VertigoDataTier/pact/ddm_ane1_ane_screening/replay/run_selector_replay.sh`
  (`validate` | `screen` | `cpu`); `docs/runbook_ane_screening_20260905.md`.

## Receipts

All under `/Volumes/VertigoDataTier/pact/ddm_ane1_ane_screening/`.

| artifact | sha256 (prefix) |
|---|---|
| `convert_manifest.json` | `43cff8c937dc428d…` |
| `mlpackages/segnet_b1_fp16.mlpackage` | `7f9537928046ae6a…` |
| `mlpackages/segnet_b1_fp32.mlpackage` | `696b503ed00a7240…` |
| `mlpackages/posenet_b1_fp16.mlpackage` | `183c7cdc81de57f0…` |
| `mlpackages/posenet_b1_fp32.mlpackage` | `2f823a19220651c6…` |
| `placement_v2.json` | `a3c18880f118dfa5…` |
| `fidelity/fidelity_n600.json` + `segnet_per_pair_flip_rate.npz` + `segnet_fp16_flip_margins.npy` + `posenet_poses.npz` | in-file |
| `fidelity/margin_census_n600.json` + `segnet_margin_histogram.npz` | in-file |
| `replay/ane1_selector_replay_validate39.json` | in-file |
| `hybrid_price.json` | in-file |

Payloads, not lengths: every per-pair flip rate, every flip-site margin, every pose vector from all
three routes, and the full margin histogram are on disk with their digests in the reports.

**One environment change:** `Brotli==1.2.0` was installed into `.venv_executorch_spike` (the private
research venv the 2026-07-13 lane already used), because the shipped runtime's `residual_archive`
refuses a split Brotli model without it. The shared `.venv` was not touched, and `upstream/` was
never modified — both scorers were converted from copies held in memory.

## Own-vehicle frontier

**fs2 S 0.14784474152757654 @ 180,023 B `[contest-CUDA T4 n600]` — UNMOVED by this arm.**
