# DDM NX1 — next-object route after the wall verdict

**Date:** 2026-08-31  
**Arm:** `ddm_nx1_next_object_route` / task #1368  
**Mode:** `$0`, scorer-free, derivation only; no Metal, scorer, Modal, or contest-eval launch  
**Verdict:** **FEASIBLE-CANDIDATE(S)-NAMED.** Rank 1 is the already-retained QBT/QBZ continuous-implicit object, whose first rung is the queued n600 real-render scorer realization. Rank 2 is QX1, a new continuous-implicit-partition × two-plane task-space-preimage object. Rank 3 is a joint scorer/coder redesign of the AFR1 RC64 field. No candidate is a score, seal, or frontier move yet.

## Authority and target

The live authority at derivation time is AFR1: archive `cbb8e900b6d4accb2ec84506a247826502021698f37c13229f441385085c4b3d`, 180,002 B, `d_seg=0.00020139`, `d_pose=0.00000637`, and `S=0.14797617125559104` on `[contest-CUDA T4, n600]`. Its terms are rate `0.11985594327989708`, Seg `0.020139`, and Pose `0.007981227975693965`. The exact sub-0.12 condition is

`100*d_seg + sqrt(10*d_pose) + 25*B/37,545,489 < 0.12`.

At AFR1 distortion, the archive must be at most 137,986 B, a 42,016 B cut. At AFR1 bytes, even zero distortion leaves only 216.347 B of rate margin. These are mutually exclusive corners, not composable wins. The pointer did not move in NX1.

## Stage 0 — complete closure-law and floor table

| Law / family | Number or exact condition | Evidence path | Axis | Verdict scope and consequence |
|---|---:|---|---|---|
| AFR1 live point | 180,002 B; `d_seg=2.0139e-4`; `d_pose=6.37e-6`; `S=0.14797617125559104` | `.omx/research/ddm_afr1_pointer_move_and_no_toy_erratum_20260831.md` | `[contest-CUDA T4, n600]` | INSTANCE; the only frontier authority used here |
| Fixed-distortion rate corner | `B <= 137,986`; cut `>=42,016 B` | `.omx/research/ddm_zdc1_zero_distortion_corner_reopens_20260831.md` | arithmetic from exact AFR1 | SCORE-LAW; every retained-distortion object must meet it |
| Fixed-rate distortion corner | non-rate term `<=0.000144056720103`; current `0.0281202279757`, a 195.2x reduction | same | arithmetic from exact AFR1 | SCORE-LAW; distortion-only mechanisms cannot use the current archive |
| Free seam at zero distortion | 216.347 B | current frontier hot state; `.omx/research/ddm_nu1_signal_in_nuance_20260831.md` gives the earlier-body 211.13 B seam | arithmetic | SCORE-LAW; container/coder crumbs cannot cross 0.12 |
| Current-body measured rate mechanisms | at most 88 composable B versus 238 B needed even at zero distortion in the earlier r012 body | `.omx/research/ddm_r012_rate_representation_20260821.md` | measured byte-closed body | CURRENT-BODY; object change is required |
| Pose absolute budget | useful edit budget `d_pose <=1.25e-4`; pose marginal near AFR1 is about 626.5 score units per unit `d_pose` | `.omx/research/ddm_hdp1_hidden_dynamics_gestalt_20260831.md` | derived from frozen score law | MECHANISM LAW; pose must be null, bit-identical, or priced exactly |
| Admission corridor | targeted × address-free × (`pose-null` or `bit-identical` or `pose-priced-exactly`) | `.omx/research/ddm_nu1_signal_in_nuance_20260831.md` | census over 23 measured objects | CURRENT EVIDENCE; a candidate outside the corridor has no admissible precedent |
| Closed-leg composition | a closed rate/distortion leg can re-enter only if another leg changes its object | `.omx/research/ddm_hdp1_hidden_dynamics_gestalt_20260831.md`; `.omx/research/ddm_rt3_route_rederivation_20260831.md` | synthesis | OBJECT LAW; stacking independently measured legs is not a candidate |
| Current renderer affine floor | `d_seg ~= 1.8479e-4 + 1.14078*token_error`; above about 140,477 B no token accuracy can cross 0.12 | `.omx/research/ddm_gestalt_the_chasm_not_the_cross_20260831.md` | two-point `[macOS-CPU advisory]` fit | CURRENT-RENDERER FORMULATION; not a theorem about a changed renderer |
| Cheap-body accuracy chasm | HG1/born-small own-rate bodies miss their token bars by about 16–36x; correction cost rises from 0.220 to 3.639 bits/error | same | real-coder/advisory synthesis | CURRENT CHEAP-BODY FORMS; capacity must be paid or geometry changed |
| Generate/serialize pincer | HG1 needs 0.0178% error, 78.5x below its measured 1.4007%; exact QBW2 quotient is 188,860 B, 2.224x its 84,910 B allowance; generate+exact residual is 352,525 B, 4.15x | `.omx/research/ddm_gestalt_generate_vs_serialize_pincer_20260831.md` | real-coder synthesis | MEASURED REPRESENTATIONS; do not serialize exact addresses and do not append their exact residual |
| QBW explicit quotient | best full-n600 logical quotient 188,860 B | `.omx/research/ddm_qbw2_temporal_bound_verdict_20260827.md` | `[macOS-CPU, n600]`, real coder | CURRENT-GB1 / MEASURED REPRESENTATION; QBW/QBMIX/QBCERT explicit-address forms are closed |
| HG1 analytic generator | packet 47,603 B; 1,325,033 mismatches (1.12324%); exact residual 385,448 B; total 433,051 B, 5.09x the 85,020 B bar | `.omx/research/ddm_gf1_generator_form_capacity_verdict_20260830.md` | real generic coder | FORMULATION; the 0.2909 B/correction rate cannot transfer to a structured successor |
| HG1+RC64 hybrids | temporal routing needs about 64% of pairs; spatial area-priced lower bound is 1.72x but is toy/sign-only | `.omx/research/ddm_hyb1_routing_needs_a_tail_20260831.md` | mixed measured/toy, explicitly labelled | FORMULATION (`HG1+RC64`, all partitions); not a family theorem |
| Lane exact/parametric carriage | cropped SMEVR 137,670 B; best exact D3B packet 64,276 B; lossy GF1 Lane 36,044 B versus 21,699 B target | `.omx/research/ddm_lc3_lane_carriage_rung_20260831.md` | scorer-free, real coder | REPRESENTATIONS; no composition or score transfer |
| Finite topology generator | 11,148 B joint packet + 221,717 B exact shape + 397 B overhead = 233,262 B | `.omx/research/ddm_ltg1_lane_topology_generator_floor_20260831.md` | scorer-free, real coder | FORMULATION; commit `bd0c9fa80e` pin |
| Born lane predictor | r10 transitive class-logit predictor weights alone 60,191 B before residual | `.omx/research/ddm_blp1_born_lane_predictor_20260831.md` | scorer-free, real coder | INSTANCE; no checkpoint/receiver/logit-residual proof; commit `8ce32946a6` pin |
| Born-small capacity ceiling | direct native fit improves only 0.6376%; pair holdout is 1.4362x worse than ancestor | `.omx/research/ddm_bz2_capacity_ceiling_free_closure_20260829.md` | scorer-free n600 | BORN-SMALL FAMILY under tested allocation |
| Born-small realized distortion | 100,862 B; `d_seg=0.0129952`; `d_pose=1.574`; `S=5.334`; 99.68x refused | `.omx/research/ddm_bz2d_distortion_verdict_20260830.md` | `[macOS-CPU advisory, n600]` | INSTANCE; proves rate birth did not provide a scorer-feasible object |
| Whole-body lossy NR1 | realized distortion 247.69x the allowed distortion budget | `.omx/research/ddm_ni1_247x_erratum_20260822.md`; `.omx/research/ddm_ni1r_nr1_k32_distortion_measured_20260830.md` | `[macOS-CPU advisory]` | INSTANCE / NR1 formulation |
| BO2 cheap body | about 209x distortion refusal | `.omx/research/ddm_bo2_born_small_distortion_row_20260824.md` | `[macOS-CPU advisory]` | INSTANCE |
| Trained-renderer W72 | about 46.3x refusal; diagonal control about 686x | `.omx/research/ddm_w72_distortion_advisory_20260823.md` | `[macOS-CPU advisory]` | INSTANCE |
| R+P W96 renderer | two seeds deliver only 1.186x and 1.5796x versus `>=5x` gate | `.omx/research/ddm_w96b_seed20260815_aligned_verdict_20260827.md`; `.omx/research/ddm_w96b_seed20260816_aligned_verdict_and_family_closure_20260827.md` | `[macOS-CPU advisory]` | MATCHED FORMULATION; closed without a different object |
| RB1 pose wall | W96 `d_pose=0.00117924`, 7.14x above its zero-seg ceiling; even zero seg gives `S=0.187947` | `.omx/research/ddm_rb1_pose_arithmetic_closure_and_storage_no_consumer_20260831.md` | arithmetic + `[macOS-CPU advisory]` | BORN-SMALL × WD3 TRAINED-RENDERER FORMULATION |
| Four-body cross | 0/4 objects satisfy both rate and distortion; qbt rate half + LB1 distortion half is cross-object and forbidden | `.omx/research/ddm_xo1_cross_successor_object_20260830.md`; `.omx/research/ddm_xo1_MAIN_adjudication_20260830.md` | mixed retained receipts | MEASURED SET OF FOUR; it does not span changed-renderer or task-space objects |
| QBT2B r10 born object | `B_hat=121,928`; `d_seg_hat=0.002518336`; `d_pose_hat=0.000575746`; `S_hat=0.408898`; tail exponent `-0.602178` | `.omx/research/ddm_qbt2b_r10_third_doubling_verdict_20260829.md` | `[macOS-MPS/CPU advisory, n32]` projection | INSTANCE; byte-feasible, trajectory/chase stopped, not a score |
| QBZ1 fitted n600 schema | `B_hat=122,062`; native spatial holdout error 0.0141554381 versus train 0.0141491818; 23,591,640/94,373,160 pixels | `.omx/research/ddm_qbz1_descent_rate_configuration_20260829.md` | `[macOS-CPU scorer-free native-field advisory, n600]` | INSTANCE; real realization is unmeasured and queued |
| WWC1 label/rate cone | FCD3 saves 2,940 B but worsens `d_seg` 0.0003474→0.000387463, net `+0.0019433 S`; joint FCD1+JF2+OE1 is 27 B larger than JF2-only | `.omx/research/ddm_wwc1_winwin_cone_sweep_20260831.md` | retained scorer/rate synthesis | CURRENT SELECTORS; only a scorer-native selector on a changed joint object remains untested |
| Task-space witness substrate | `research_only=true`; `receiver_closed_n600_archive=false`; old AA-SDF component `d_seg=0.000859858` is not a complete object | `.omx/research/original_taskspace_inverse_witness_codec_20260725/roadmap.json` | research-only, mixed old component receipts | PROPOSED ARCHITECTURE; old byte and directional-basis projections are retired, not evidence for a row |

## Stage 1 — objects that survive all laws

The four-body cross does **not** span the remaining space. It spans cheap/inherited bodies and inherited realizers. Three hybrids lie outside it: continuous implicit partition × real scorer realization; continuous implicit partition × a two-plane task-space preimage; and a jointly changed RC64 field × coder × exact pose constraint. Exact-address Lane/topology tails, post-hoc selectors, and an independently trained renderer are not new objects and remain closed.

| Rank | Object | Why it satisfies the laws on paper | Inherited negative and scope | Disposition |
|---:|---|---|---|---|
| 1 | **QBT/QBZ continuous-implicit born object, realized at n600** | Generated/address-free partition; real packet is 122,062 B; full n600 native field exists; realization consumes the same packet through R/uint8/frozen scorers; exact score law is the terminal. It changes the representation, not a closed leg. | r10 only stops the n32 training chase; QBZ1 does not measure scorer realization. | `QUEUED_WITH_FIRE_ORDER`; feasible candidate, not sealed |
| 2 | **QX1: continuous-implicit partition × joint two-plane task-space preimage** | QBT geometry generates topology without exact addresses; the counted state jointly determines Y0/Y1 through R; Seg and Pose are optimized in the same preimage, so pose is priced exactly; any learned residual is terminal, real-coded, and part of the same object. Generic solver/code remains free; all video-derived state is counted. | QBW2/LTG1/LC3 close explicit masks and topology; old C0B is research-only. QX1 replaces those representations and therefore does not inherit their byte numbers. | `HOLD_CONDITIONAL_ON_RANK1`; paper-feasible new object |
| 3 | **AFR-RC64-JC: joint scorer/coder field redesign on the AFR1 realization stack** | Changes the token field and its probability model together under exact coder stages and exact pose constraint; address-free whole-field update; current renderer remains byte-closed. | It inherits the affine floor. With AFR1 pose, that floor permits at most 140,479.86 B, so it must cut at least 39,522.14 B while preserving pose and reaching the floor. WWC1 found only 2,940–4,210 B scale effects and one harmed Seg; CM1 found no cheap differentiable rate surrogate. | `HOLD_CONDITIONAL_ON_RANKS1_2`; paper-feasible but weakly priced |
| — | Procedural task-space base + terminal learned residual | This is the only unmeasured hybrid outside the n=4 cross that obeys generate-not-serialize and pose-priced-exactly. | An exact-address residual would inherit the 352,525 B pincer. | Folded into QX1; not a separate fire order |
| — | QBW/QBMIX/QBCERT explicit quotient, Lane exact tail, topology events | Exact addresses or shapes are counted. | 188,860 B QBW2; 64,276/137,670 B Lane; 233,262 B topology before a compliant joint realization. | `REFUSED_MEASURED_REPRESENTATION` |
| — | HG1+RC64 routing / post-hoc scorer selector | Leaves a systematic tail; selection is not scorer-native and additive savings fail. | 64% temporal routing; spatial formulation closed; WWC1 exact selections do not transfer. | `REFUSED_FORMULATION` |
| — | BZ2/RB1 born-small trained renderer | Rate birth survives but scorer distortion and especially Pose do not. | BZ2 `S=5.334`; RB1 zero-Seg `S=0.187947`. | `REFUSED_FORMULATION` |
| — | Current-body distortion-only or coder/container cleanup | Cannot cross either exact corner. | 195.2x distortion demand or 42,016 B rate demand; current coder axes exhausted. | `REFUSED_SCORE_LAW` |

### Rank 1 capacity accounting

QBZ1's verdict is recalled verbatim: **“That is not yet the charter's capacity ceiling. No frozen scorer ran in this arm. Therefore no capacity/optimization fork is claimed.”** NX1 therefore does not call QBT capacity-limited.

At `B_hat=122,062`, rate is `0.081276075535998` and the total distortion allowance is `0.038723924464002`. With AFR1 pose, `d_seg` must be below `0.000307426965`; if pose is merely at the absolute `1.25e-4` budget, `d_seg` must be below `0.0000336858540`. With zero Seg, the pose ceiling is `0.000149954233`. The r10 pose estimate `0.000575746` is therefore impossible even with zero Seg, but it is an n32 advisory projection, not the queued n600 realization.

The fitted packet's real-coded sections are model 79,894 B and latents 26,122 B. The 137,986 B absolute cap leaves only 15,924 B above `B_hat`: at most a 1.199x model-only expansion, a 1.610x latent-only expansion, or a 1.150x proportional expansion of model+latents. Those are capacity ceilings, not predicted gains. Any larger capacity must buy back bytes elsewhere or improve distortion enough to move its own cap.

### Rank 2 compliant task-space form

QX1 is not the old directional-basis proxy or an HNeRV reskin. Its counted sufficient statistic is a continuous implicit partition plus joint two-plane preimage coefficients. The planes produce full RGB; chroma remains active as a Seg actuator and as part of PoseNet's YUV6 input. A deterministic generic decoder/solver expands the statistic; it stores no scorer weights, GT labels, per-pixel addresses, or video-derived table in code. Training and selection occur through camera R, uint8, frozen SegNet/PoseNet, and the real final coder. The terminal residual, if any, is learned jointly and retained; no exact mask residual may be appended.

Paper budget examples expose the required accuracy: at 120,000 B, AFR1-level pose permits `d_seg <0.000321156976`; at 100,000 B it permits `d_seg <0.000454328767`. If pose is only `1.25e-4`, those limits tighten to `0.000047415866` and `0.000180587656`. The old AA-SDF component's `0.000859858` proves none of these cells and cannot be promoted.

### Geometry-forced new object

QX1 is forced by the intersection of the measured walls: explicit topology/address storage is too large; cheap inherited renderers have an affine floor; independent renderer births fail Pose; and post-hoc selection is not scorer-transferable. The surviving geometry must generate the partition implicitly **and** choose its RGB/YUV preimage jointly. That is a different object from both QBT's inherited palette realization and the old research-only witness sketch.

## Stage 2 — `$0` first rungs, falsifiers, cost, and seal readiness

| Rank | `$0` first rung | Hard falsifier | Cost / wall | Seal-readiness |
|---:|---|---|---|---|
| 1 | Fire the already-written QBZ1 n600 `realize` order after MAIN claims the one local scorer lane. It consumes the retained fitted packet and writes every frame/logit/argmax/pose/target. | Realized `100*d_seg + sqrt(10*d_pose) + 0.081276075536 >= 0.12`; if it fails, do not call capacity versus optimization until the preregistered native-coefficient A/B distinguishes them. | `$0` local CPU scorer; no new training. Wall time is to be measured by the existing runner. | Inputs and retention PASS; lane and free-space checks must be repeated immediately before fire. No exact-eval seal until a same-object receiver archive exists. |
| 2 | Scorer-free transitive section census over the retained QBT packet and current task-space roadmap: classify each section as reused, replaced, or forbidden, and require a real-coder envelope below 137,986 B before any implementation. | Nonreplaceable QBT topology state plus the minimum counted two-plane/preimage state exceeds 137,986 B, or the design requires an explicit address/GT/scorer-weight stream. | `$0`; source/retained-payload analysis only. | No fire until rank 1 is terminal and a section-complete schema, decoder contract, per-stage retention plan, and score-law budget exist. |
| 3 | Reprice CM1's retained perturbations with restartable exact coder state and the exact affine-floor constraint; require a stratified `n>=32` rate-direction screen before scorer work. | No exact-rate surrogate/outer-loop with positive held-out rank correlation, or projected cut below 39,522 B at AFR1 pose. | `$0` source/cached-payload work; CM1's current exact coder is about 897.7 s/eval and prefix128 about 167.9 s/eval, so an uncached training loop is refused. | No fire until ranks 1–2 are terminal and exact-incremental coder cost is bounded. |

### Honest n600 Metal projection for a QBT retrain

R10 measured 2.135 s/step at n32 with canonical config content SHA `36a40bdf…`. Linear sample scaling to n600 is `2.135*(600/32)=40.03125 s/step`. This is a **projection**, not measured n600 Metal timing:

| Steps | Projected n600 wall time |
|---:|---:|
| 40,020 | 18.542 days |
| 400,000 | 185.330 days |
| 1,000,000 | 463.325 days |

The r10 optimistic fit placed a tie around 400k steps and sub-0.12 around 1M steps while its later local exponent was only `-0.602178`. Therefore a full n600 Metal retrain is **not seal-ready**. It may be reconsidered only after the retained n600 scorer realization and, on failure, the native-coefficient A/B show optimization rather than capacity/representation as the live limiter.

### Rank 1 fire-order audit

- `FIT_RESULT.json` SHA-256 is `69b33e5d393deff7f1fcd76844cf524d7c19691f431aa399a876b2ad1ce227bf`: **MATCH**.
- `final_reencode/reencode_payloads.tar` SHA-256 is `4c16e6c045768b2dee62f59ac9a2a27b7386280dfccff3dd5331a8d9509d95f7`: **MATCH**.
- Inner `archive.zip` and `archive.repeat.zip` are both 106,832 B with SHA-256 `0e2ffdfaa5fe481d481dd70a9672a67f80b9aad7648f0c775fe2956dd3a4841d`: **DETERMINISM PASS**.
- All 600 native fields, checkpoints, packet sections, archive payloads, and repeat payloads are retained: **RETENTION PASS**.
- APDataStore free space observed by NX1 was 6,142,427,136 B, only 1,142,427,136 B above the 5,000,000,000 B fire gate: **PASS-BUT-FRAGILE; RECHECK AT FIRE**.
- A fresh unique local scorer claim and newest-terminal check are not owned by NX1: **MAIN GATE**.
- Existing command: `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 .venv/bin/python experiments/ddm_qbz1_descent_rate_configuration.py realize --scorer-claim-id ddm_qbz1_scorer_20260829 --launch-authorized`. NX1 did not execute it.

## Stage 3 — ranked fire order and dispositions

| Order | Disposition | Owner | Consumer store | Fire trigger |
|---:|---|---|---|---|
| 1 | `QUEUED_WITH_FIRE_ORDER` | MAIN local scorer scheduler | `/Volumes/APDataStore/pact/ddm_qbz1_descent_rate_configuration/REALIZED_RESULT.json`; retained payloads under `realized_n600/` | Verify newest relevant scorer row terminal; append a fresh unique active local-scorer claim; verify no newer active scorer claim within 24 h; AP free bytes `>=5,000,000,000`; input hashes still match |
| 2 | `HOLD_CONDITIONAL_ON_RANK1` | successor QX1 derivation arm assigned by MAIN | proposed `/Volumes/VertigoDataTier/pact/ddm_qx1/SECTION_CENSUS.json` | Rank 1 terminal and fails sub-0.12; MAIN assigns owner; scorer-free section census first; no implementation until real-coder envelope `<137,986 B` |
| 3 | `HOLD_CONDITIONAL_ON_RANKS1_2` | successor AFR-RC64-JC arm assigned by MAIN | proposed `/Volumes/VertigoDataTier/pact/ddm_afr_rc64_jc/RATE_DIRECTION.json` | Ranks 1 and 2 terminal/refused; MAIN assigns owner; exact-incremental coder screen has `n>=32` and a preregistered positive rate-direction gate |

**One next MAIN action:** execute order 1 only. Do not buy n600 Metal, implement QX1, or reopen the current-body selector cone before that retained realization returns.

## RECALL EVIDENCE

Recall covered the current frontier/hot-state authority, task ledger/status surfaces, canonical index and DAG, current-branch research memos, arm final messages, and APDataStore QBT/QBZ receipts. Bounded searches included `qbt2b|qbflow|born object|capacity ceiling|n600`, `QBW|QBMIX|QBCERT|quotient`, `task.space|witness|preimage|two-plane`, `Lane|topology|residual|generator`, `pose-null|bit-identical|pose-priced`, and `score_marginal|procedural_predictor_plus_residual_correction_savings_v1`. The canonical-equation registry was queried for procedural-predictor/residual, score-marginal, compensated-semantic, token-rate-direction, and greedy-marginal equations. No NX1-specific prior memo was found. Upstream was read-only.

Provenance pins used: AFR1 archive `cbb8e900…`; LTG1 commit `bd0c9fa80e`; BLP1 commit `8ce32946a6`; QBT r10 config content `36a40bdf…`; QBZ1 result and payload hashes listed above. The working tree already contained unrelated modified and untracked files; NX1 changed only this memo.

## Denominator and authority labels

- Closure table: 29 rows, all with a number/condition, evidence path, axis, and verdict scope.
- Candidate census: 9 named rows — 3 paper-feasible, 1 folded into QX1, 5 refused by a scoped measured/formulation/score law.
- Cross claim: 0/4 measured cross objects satisfy both rate and distortion; this is not global nonexistence.
- New measurements: 0 scorer, 0 Metal, 0 Modal, 0 contest eval, 0 new payload materializations.
- Authority: all projections are labelled; only AFR1 is `[contest-CUDA T4, n600]`. MacOS and scorer-free rows are advisory and do not move the pointer.

## Own frontier

NX1 produced **no new score and no pointer movement**. It narrowed the next-object search to three paper-feasible objects and made the retained QBT/QBZ n600 realization the sole immediate fire action. The exact frontier remains AFR1 at `S=0.14797617125559104`.

## NEXT_IF_RESUMED

- **Disposition `QUEUED_WITH_FIRE_ORDER`; owner MAIN local scorer scheduler; consumer `/Volumes/APDataStore/pact/ddm_qbz1_descent_rate_configuration/REALIZED_RESULT.json` plus `realized_n600/`; fire trigger:** reverify the newest relevant scorer row is terminal, claim the unique local scorer lane, confirm no newer active scorer claim within 24 hours, confirm AP free bytes remain at least 5,000,000,000, and re-match both declared input hashes; then run the existing QBZ1 realization command.

## LIVE-HYPOTHESES

- **The retained QBT/QBZ rate-feasible object may realize much better than its n32 advisory projection.** This is plausible because the full n600 fit is already byte-closed at 122,062 B with a tiny spatial train/holdout gap, while the required R/uint8/frozen-scorer transfer has never been measured.
- **QX1 may escape both sides of the pincer.** This is plausible because it generates the partition continuously without exact address storage and changes the RGB/YUV preimage jointly, directly addressing the explicit-quotient byte wall and independent-renderer Pose wall rather than stacking them.
- **A joint RC64 field/model redesign may find a large coder-friendly scorer cell that post-hoc selectors missed.** This is plausible because WWC1 tested fixed labels/selectors, not a field trained under exact score and coder constraints; it remains low-ranked because the required 39.5 KB cut is far beyond the observed selector effects.

## DEAD-ENDS

- **Do not continue the qbt2b n32 doubling chase or extrapolate it into an n600 Metal buy.** The local exponent is `-0.602178`, its target receded, and honest n600 scaling prices even 40,020 steps at 18.542 days.
- **Do not call QBZ1 capacity-limited before realization and the coefficient A/B.** Its own verdict says no frozen scorer ran and no capacity/optimization fork is claimed.
- **Do not serialize exact masks, Lane addresses, topology events, or append an exact residual.** QBW2, LC3, LTG1, GF1, and the pincer price those representations above their lawful budgets.
- **Do not stack qbt's rate half with LB1's distortion half or reuse the four-object cross as a global theorem.** That score is cross-object and fake; the measured cross covers only four inherited/current bodies.
- **Do not reopen born-small trained renderers, W96 R+P, HG1+RC64 routing, or WWC1's post-hoc selector cone without changing the object.** Their measured failures are respectively scorer/Pose, insufficient matched gain, systematic tail, and non-transfer/additivity failure.
- **Do not use the old task-space directional-basis or byte projections as evidence.** The roadmap is research-only and not n600 receiver-closed; the production directional-basis premise was superseded.

---

## ADDENDUM (ddm_eq1, 2026-09-04) — the equations leg

**Law:** `procedural_predictor_plus_residual_correction_savings_v1` — `tac.canonical_equations.procedural_predictor_residual_savings` (`tac.canonical_equations`). **Relation:** CONSULTED, NOT ANCHORED — nx1 is a route memo; it queried the registry for this form and found no NX1-specific prior.

Recorded honestly: nx1 measures nothing and adds no anchor. It names three feasible objects and prices them at the canonical rate constants (25.0 / 37,545,489) this law carries. Its rank-3 candidate — a joint scorer/coder redesign of the AFR1 RC64 field — is the object `ddm_jc1` then adjudicated under `decoder_causal_condition_transport_v1`, so the route's downstream equations leg is that law, not this one.

This memo's Catalog #344 trigger was the word **stratified** — `"ratified"` is a substring of it, and the gate matched plainly. MEASURED by this arm: 16 of the 29 live memos (55.2%) tripped the gate ONLY that way, i.e. the gate was flagging the memos that did their sampling right. Fixed in the same batch (`(?<!st)ratified`); the disposition above stands on its own merit, not on the misfire.
