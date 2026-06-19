# SESSION SYNTHESIS — canonical source-of-truth (2026-06-17 → 2026-06-18)

**Operator: "review all design/research memos + conversation from today; ensure optimal + no signal loss."**
This is the SINGLE entry point for the next session. It resolves every supersession/correction so the
record is coherent (several of today's memos corrected each other). All `[contest-CPU advisory]` unless
noted; **exact pointer UNMOVED at 0.19110** (G3 confirmed; this session did NOT move it — stated plainly per
the GOAL firewall). The win this session is structural: a measured contest-grade pipeline + a concrete,
grounded three-lever path to sub-0.15 on the ranking axis, + recovery of two wrongly-buried levers.

## ⚠️ INFLECTION CORRECTION (2026-06-18, late — SUPERSEDES the S≈0.127 headline below)
The "three composable levers → S≈0.127" headline below was written mid-campaign on EXTRAPOLATIONS. The
QAT-pivot cycle then MEASURED each near-term sub-0.15 path and they are now all capped. Read this first;
the headline below is the historical reasoning trail (preserved per HISTORICAL_PROVENANCE), not the current
verdict. Pointer still UNMOVED 0.19110. All `[contest-CPU advisory]`.

**Sub-0.15 mechanism status — every byte-cheap-d_seg family measured:**
| # | mechanism (family) | status | evidence |
|---|---|---|---|
| 1 | bc20 d_seg long-train (epochs) | CAPPED — capacity-floored ~0.0022 (50k run, best S 0.401 > basin 0.378) | the 50k margin-hinge run, stopped |
| 2 | Path A — fresh higher-capacity decoder | DOMINATED — desk-calc best S_QAT 0.241 (bc36) > 0.191 frontier | `capacity_rd_score_aware_qat_pivot_*` |
| 3 | Path B — QAT-shrink the frontier | CAPPED S=0.483 — d_pose recovers, **d_seg walls ~0.0035** under int5 | `frontier_int5_score_aware_qat_finetune_*` |
| 4 | concentrated-saliency regularizer | NO-GO — can't redistribute criticality off a dense shared path | `probe_concentrated_saliency_feasibility` |
| 5 | factored cheap LF core (capacity-axis) | RED — **d_seg ∼ 29.3·params^−0.71**; frontier-grade needs ~10.7M params | `factored_lf_core_capacity_gate_*` |
| 6 | byte-neutral taper realloc #121 (allocation-axis) | **WEAK (downgraded from "optimal")** — frontier per-tensor saliency only **5.54× flat**, decision-band mass 44%, geometry-ordering FALSE, d_seg-blind final stage already starved (~1.75% params) → little to water-fill | `frontier_margin_saliency_qat_bitalloc_prior_*` |
| 7 | differentiable curve-core (geometry-axis) | **RED — SURVIVAL WALL** (geometry fits: geo_recon→0.00106, but realized d_seg floors ~0.0067 = 2.62× frontier = 1.05× the static-store wall; differentiable color/offset pre-comp did NOT beat the roundtrip's boundary-band mixing) | `curve_core_gate_RED_survival_wall_and_the_pincer_20260618.md` |
| 8a | generative/iterated **FLAT-PARTITION** (NCA) | **RED but MIS-EXECUTED** — the probe grew a fuzzy per-class flat-colour partition (geo_seg 0.019, WORSE than the static curve's 0.0067), re-measuring the texture wall; its "all 4 families capped" claim is OVER-STATED (it tested the wrong representation) | `generative_axis_nca_dseg_core_gate_*` (over-claim CORRECTED here) |
| 8b | generative/iterated **CONTINUOUS-TEXTURE** (latent-conditioned NCA) | **OPEN — RE-TESTING faithfully** — a few-KB shared iterated rule + small per-frame latent → CONTINUOUS RGB (not flat fill); the real prize: replace the 161KB feedforward decoder → rate ~0.013 → S≈0.086 IF it matches frontier d_seg | re-launched gate |

**The unifying physics → the PINCER (now measured on both corners):** d_seg = a perimeter integral over the
frozen SegNet's *learned, high-dimensional* decision boundary. Two measured REDs close a pincer:
(a) **flat-region representations** (partition-store, curve-core) are **survival-walled at realized d_seg
~0.0067** (the eval roundtrip's boundary-band color mixing — representation-agnostic for flat painting);
(b) **continuous-texture representations** (learned pixel decoders) are **capacity-walled at d_seg∼params^−0.71**
(factored-LF; frontier-grade needs ~4–10M params = bytes). The only unwalled corner: a representation that
paints continuous texture AND is byte-cheap = breaks params^−0.71. The generative/iterated axis (family 8,
running) is the sole remaining candidate. **The strategic fork:** GREEN (NCA) ⇒ the one escape ⇒ spec the
build. RED (NCA) ⇒ the airtight terminal finding — sub-0.15 d_seg byte-cheaply unreachable across static AND
generative families; frontier ~0.19 near the real floor; CLAUDE.md S_floor=0.11797 over-counts (it assumed
d_seg→0 byte-cheaply, which the pincer falsifies) → a goal/floor re-frame the operator should weigh in on.
Inflection memo: `campaign_inflection_three_paths_capped_concentrated_saliency_20260618.md`; curve verdict:
`curve_core_gate_RED_survival_wall_and_the_pincer_20260618.md`.

## HEADLINE (HISTORICAL — superseded by the inflection correction above): the concrete sub-0.15(CPU) path (S ≈ 0.127 projected, three composable levers)
`S_CPU = 100·d_seg + sqrt(10·d_pose) + 25·bytes/37.5M`. G3 proved local-CPU advisory ≈ exact contest-CPU to
~0.001%, so these advisory levers project onto the ranking axis faithfully:
1. **d_seg → 0.000322** (from 0.00260) — running margin-hinge 50k long-train. Feasible at ~14.5k ep per the
   stretched-exp model (NOT the power-law's "999k/infeasible" — wrong model class). RUNNING.
2. **rate → ~0.037** (from 0.0594) — FP-shrink. **VERDICT (smoke #136, 2b227d27a): naive PTQ COLLAPSES**
   (int4 cuts 47.7% bytes = Δrate −0.0283, the −0.022/−0.029 confirmed, BUT S rises monotone — int4 d_pose
   explodes 322×; int8 is the post-hoc S-optimal corner). The rate win is REAL but needs **score-aware QAT**
   (train the decoder quantization-robust; allocate quant-error via the #141 margin-saliency map; break-even:
   hold d_seg within ~+0.0003). NOT a free recode — a real training effort. NEXT GATED BUILD.
3. **d_pose held** at 0.000342 — basin value; FiLM carrier + the 1-DOF radial-zoom codec (#140).
Composed: 0.0322 + 0.0585 + 0.0366 ≈ **0.127**. Each lever is running / being tested / proven-faithful.
CAVEATS: each is unverified-at-target (extrapolation; PTQ-hold unproven; pose-variance risk); CUDA axis adds
d_pose ×1.41 (but contest ranks on CPU).
**BC20-CAPACITY-CONDITIONAL (label-noise floor resolution, `label_noise_floor_RESOLUTION_frontier_existence_proof_20260618.md`):**
the d_seg→0.000322 lever is reachable IN PRINCIPLE (the 0.19110 frontier vehicle already has d_seg ~0.0002-0.0006,
proving the proxy "label-noise floor" too pessimistic + S_floor=0.118 stands) but bc20's small basis may be
CAPACITY-walled above frontier-grade d_seg — the running margin-hinge long-train IS that capacity test (does bc20
get below ~0.001?). If bc20 walls high, the sub-0.15 vehicle needs more capacity OR the rate hedge (FP-shrink)
carries more. The path is alive, not a fundamental wall; the open question is bc20-capacity vs vehicle-choice.
**THE CRUX (after both decisive reads): capacity↔rate tension.** Frontier = d_seg ~0.0003 @ 177KB (rate 0.118);
bc20 = d_seg 0.0026 @ 89KB (rate 0.0594). Sub-0.15 needs frontier-grade d_seg AT well-below-frontier bytes =
a HIGHER-capacity decoder + aggressive score-aware FP-shrink QAT (the bridge; PTQ-collapse proved the shrink
must be trained, not post-hoc). The honest path = TWO real training efforts (d_seg long-train [capacity-gated]
+ score-aware FP-shrink QAT [rate]) + the small pose primitive — NOT "free levers compose to 0.127." Early
signal: margin-hinge ep1800 d_seg 0.003228 is already below the CE control's ep2000 (0.003251) — the lever is
winning the A/B; the ep2000 trigger confirms (not fires).

## VERIFIED insights (current status — corrections resolved)
| insight | status | score-EV | consumer | source memo |
|---|---|---|---|---|
| margin-hinge BENDS d_seg exponent (0.787 vs CE 0.608 vs soft_cosine 0.646-WORST); gradient non-vanishing on flips | VERIFIED (small-slice; rel. bend trustworthy) | the d_seg lever | running run + #136-sister | accel1_margin_hinge_flip_targeting_dseg_exponent_20260617.md |
| **margin-hinge THROUGHOUT** is the canonical seg lever (replaces CE+soft_cosine all stages) | LANDED + RUNNING | — | launch_bind_all `--seg-margin-hinge-throughout` | track_a_canonical_config_final_reconciliation_20260617.md |
| local-CPU advisory ≈ exact contest-CPU (~0.001%) | VERIFIED (G3) | de-risks whole campaign | all advisory work | g3_torch_vehicle_bc20_first_exact_row_20260618T0135Z.md |
| CPU→CUDA d_pose drift +41% | VERIFIED (G3) | budget ×1.41 on CUDA | CUDA projections | g3 memo |
| d_seg is the irreducible BOUNDARY residual (882× margin<0.5-concentrated, 0% interior-avoidable) | VERIFIED | the boundary is the whole d_seg game | margin-hinge + #137/#138 | yousfi_detector_cost_blindspot_b_verdict_20260617.md |
| d_seg = 64% road↔lane markings | VERIFIED | #137/#138 | E-probe | (probe E commit 470d54828) |
| d_seg wall is REAL (not EMA-shadow artifact); pose eval-noise = FiLM carrier | VERIFIED | trust the d_seg signal | — | blindspot_probe_c_measurement_trust_dseg_dpose_20260617.md |
| non-neural partition STORE realizes at S≈0.84 (24% boundary-survival flip even w/ μ-optimal colors); d_seg=0-store +0.0083 over frontier | SOUND-KILL | d_seg belongs in TRAINING, not a store | margin-hinge | partition_store_realization_gate_DEFER_20260617T024639Z.md |
| sufficient-statistic store floor S_floor=0.2429 (276,996 B = 3.11× the 89KB basis) → the learned decoder IS the cheaper SS carrier (not anchored on a non-minimal rep) | VERIFIED | confirms small-basis is right | — | sufficient_statistic_floor_probe_20260617.md |
| frontier rate axis is AT its entropy floor (~8.0 bits/byte every section; lossless recode recovers 0 of the 61,725 B needed) → sub-0.15 rate REQUIRES a LOSSY model-side change | VERIFIED | qualifies lever-2: FP-shrink MUST be lossy/QAT, not a recode | #136 | frontier_rate_cut_vs_small_basis_anchoring_probe_20260617T155535Z.md |
| half-res store induces d_seg 0.00554 (7× budget) — spatial downsampling is NOT free (d_seg is not HF-blind) | VERIFIED | don't downsample for bytes | — | minimal_store_lagrangian_and_compression_is_intelligence_20260617.md |
| compress-time SEED+SOLVE is NOT faster than descent (latent-solve ≤0.75% & 3.7× slower; residual pixel-solve fails roundtrip +2.5 S) | VERIFIED | don't pivot to a solver finisher; descent is right | the running run | compress_time_seed_and_solve_dseg_verdict_20260617.md |
| probe-E STC coder = 0.523 B/flip (beats the 0.749 per-flip floor 30%); witness total 121.7KB < 177KB frontier BUT 37% survival wall holds (basin-relative) | VERIFIED | reusable coder for #137 lever-D; survival-gated | #137 | (probe E: reports/probe_yousfi_filler_flip_structure_stc.json + commit 470d54828) |
| pose is ~free on the byte axis: pose-H = 5,208 B = 1.9% of the SS store | VERIFIED | #140 EV basis; pose carrier tac.optimization.pose_trajectory_entropy EXISTS | #140 | sufficient_statistic_floor_probe_20260617.md |
| stretched-exp d_seg model (16× better fit) → sub-0.15 feasible ~14.5k ep | VERIFIED (model-fit; extrapolation) | reopens long-train thesis | the running run | closure_reaudit_round2_synthesis_audited_20260618.md |
| pose is ~1-DOF radial-zoom (Jacobian rank≈1; stored-pose SVD rank-2=99.97%) | VERIFIED | #140 pose codec | #140 | g3 + round2 synthesis |

## SUPERSESSION LEDGER (what CHANGED today — do NOT cite the old verdicts)
1. **#1 pose-low-rank: "FALSIFIED" → REOPENED → SEALED with a CORRECTED (smaller) win (3 corrections).**
   (a) my inline falsification used the wrong fidelity (over-provisioned MSE 2.9e-5); (b) R1 "corrected" it to
   "rank-2/254 wins 2.7× at MSE≤d_pose" — but that was ALSO wrong (the "free headroom" fallacy: it ignored
   the nonlinear pose term √(10·d_pose), ∂/∂d_pose≈85.8, so rank-2/254 is NET-NEGATIVE, +0.020 pose cost vs
   −0.0013 bytes); (c) the #140 build's full-score math found the honest answer: the ONLY win is the
   Pareto-dominant **rank-4/511** (better bytes AND better MSE → strictly dominant, **~−0.0004**), now the
   codec default, opt-in DEFAULT-OFF. SEALED (89c692692). The pose section is a SMALL lever (~0.5 KB). Anchor
   lesson: even the "audited correction" (R1) was fallible — the invariant (the actual nonlinear score fn) is
   the truth. Memos: `lowrank_pose_section_codec_landed_20260617.md` (final) supersedes
   `pose_lowrank_CORRECTED_fidelity_20260617.json` (R1's flawed-headroom intermediate).
2. **Ego-hood: "FALSIFIED as free lever" → REGION-CORRECTED, REOPENED for re-measurement.** I measured the
   all-frame static-core (0.038%) when the mechanism's region is the per-frame class-4 mask (7.4% of flips).
   NET win still survival-gated (likely folds into #137). → task #139 reopened.
3. **d_seg projection: power-law point estimates → stretched-exp / window-range.** The power law was the wrong
   MODEL CLASS (slope steepening 0.24→0.64). My earlier S(50k)=0.177-0.226 point estimates SUPERSEDED.
4. **Floor 0.1178 → 0.1179** (rate must use archive.zip 89,274 B, not 0.bin 89,136).
5. **lane-geometric-solve JSON labels: "EUREKA_HOLDS" (LIED) → corrected to FALSIFIED/quasi-stationary**
   (NO-FAKE fix, commit 04f60aef7; the data/memo were always right, only the auto-label was inverted).
6. **RA-2 R-3 "lossless recode beats frontier −0.00092": REJECTED** (stale baseline; S=0.191117 > 0.19110;
   already banked in #64). RA-2 R-5 "store partition = d_seg=0 by construction": REJECTED (survival wall).

## RANKED REOPEN LEDGER (audited; the recovered signal)
1. **FP-shrink QAT (#136)** — TOP, −0.022 to −0.029 S; killed on naive-PTQ (wrong op point), QAT never tried. $0 smoke RUNNING.
2. **long-train d_seg feasibility (stretched-exp)** — reopens the sub-0.15-via-epochs thesis; the running run IS the test.
3. **pose 1-DOF radial-zoom (#140)** — ~0.0013 byte + d_pose-floor; $0 test.
4. **native-grid in-cell repair / boundary sidecar (#137)** — roundtrip-real (unlike camera-res), <1.27 B/flip; needs 600-verify.
5. **ego-hood per-frame re-measurement (#139)** — survival-gated; folds into #137.
6. lane-poly geometric prior (#138, IoU-gated); STC mask-delta w/ detector cost map; Cool-Chic (28.8× faster now); AC/rANS configs; HiNeRV bilinear-skip.
SOUND-KILL (do NOT reopen): rel_err² objective, pixel-blur preprocessing, per-pixel RGB seg-correction SIDECAR (36.9% survival wall — d_seg must move via the DECODER), pure-symbol AV1 mask coding.

## META-PATTERN (the durable lesson — RA-1, audited-sound)
**Closures grounded in operating-point-INVARIANT quantities held (entropy floor, geometric smear-wall).
Closures grounded in a CHOICE (region / model-class / fidelity / config) are where errors concentrate.**
The remaining ledger fallibility is un-EXECUTED reactivations, not mislabeled kills. Apply this lens to every
future closure: is the verdict invariant, or does it depend on a chosen operating point?

## CANONICAL CONFIG + registered control (drift-prevention)
The running 50k run: `experiments/launch_bind_all_taper_ab.py --arm arm_b --seg-margin-hinge-throughout
--pose-film-v2 --pose-equimarginal --pose-dim-weights-auto --split-by-head --pose-grad-on-train-device
--train-device mps --async-eval --rate-attack --ema-warmup --oomph-seg-weight-mult 1.0 --kd-warm-epochs 300
--total-epoch-budget 50000` (supervisor `.omx/tmp/arm_b_canonical/supervise.sh`, fresh out-dir
`bindall_arm_b_canonical50k_mh_n600`). REGISTERED CONTROL: the CE baseline log `run_CE_baseline_ep3700.log`
(milestones ep1000=0.004355, ep2000=0.003251, ep3700=0.002370). REVERT TRIGGER: if margin-hinge d_seg ≥ CE
at matched ep~2000, revert + run the proper hinge-vs-soft_cosine A/B. best-checkpoint selects on full S
(pose-spike-safe, verified driver.py:2596).

## PIPELINE READINESS (proven this session)
G3 proved the bc20 vehicle byte-closes → inflate.sh → dual CPU+CUDA exact eval at contest grade
(`tools/build_torch_vehicle_g3_contest_packet.py`). Any converged checkpoint → exact row immediately.
The actuator (#107) + byte-close (#125/G2) + exact-row (#127/G3) chain is complete.

## COMPLETE cross-link index (corrected after a completeness audit 2026-06-18)
**Correction:** an earlier draft falsely claimed "all 22 memos cross-linked" while listing only 6. The full
today-memo set (every file a future agent should be able to find from here):
- **Strategy/config:** SESSION_SYNTHESIS_SoT (this) · track_a_canonical_config_final_reconciliation_20260617 ·
  minimal_store_lagrangian_and_compression_is_intelligence_20260617
- **d_seg levers/probes:** accel1_margin_hinge_flip_targeting_dseg_exponent_20260617 (+ _exponent_random JSON) ·
  yousfi_detector_cost_blindspot_b_verdict_20260617 (+ JSON) ·
  blindspot_probe_c_measurement_trust_dseg_dpose_20260617 (+ JSON) ·
  compress_time_seed_and_solve_dseg_verdict_20260617 · (probe E: reports/probe_yousfi_filler_flip_structure_stc.json)
- **store/floor negatives:** partition_store_realization_gate_DEFER_20260617T024639Z ·
  sufficient_statistic_floor_probe_20260617 · frontier_rate_cut_vs_small_basis_anchoring_probe_20260617T155535Z
- **geometry/openpilot:** yousfi_road_lane_geometric_solve_probe_20260617 (+ NO-FAKE-corrected JSON) ·
  yousfi_road_lane_exploitation_research_20260617T184704Z · openpilot_comma_repo_wider_exploit_sweep_pose_cereal_hood_20260617T192718Z
- **review/re-audit:** recursive_adversarial_review_round1_20260617 (CR-A/B/C dispositions; CE-control milestones
  ep500=0.004783…ep3700=0.002370 + revert trigger) · closure_reaudit_reopen_ledger_20260618T012325Z (the R-1..R-12
  tiered table) · closure_reaudit_round2_synthesis_audited_20260618 · pose_lowrank_CORRECTED_fidelity_20260617 JSON
- **exact row:** g3_torch_vehicle_bc20_first_exact_row_20260618T0135Z (+ runtime-closure + dispatch-plan JSON)

## Captured from the openpilot/comma exploit memos (were absent; now indexed)
The #1+#2 combo (openpilot lane-poly spatial prior #138 + Jacobian-KKT coverage render); pose-trajectory
low-rank coding of the FiLM-STORE section (the named "decisive next $0", now #140); road-edge 0↔2 geometric
prior (#4); comma10k turn-arrow/crosswalk = ROAD-not-lane label idiosyncrasy (cheap d_seg); cereal/rednose
"ego trajectory is a low-rank EKF/MSCKF output" (the 1-DOF radial-zoom basis for #140); comma-4B = DEFER (nothing public).

## NO-SIGNAL-LOSS / wire-in status (post-audit)
- Code landed: `--seg-margin-hinge-throughout` (+tests), ego-hood probe, G3 packet builder (+tests), pose-low-rank
  corrected JSON, the lane-geometric JSON NO-FAKE fix (04f60aef7).
- Tasks: #136 (FP-shrink TOP, smoke running), #137 (boundary sidecar), #138 (lane prior IoU gate), #139 (ego-hood
  reopened), #140 (pose low-rank, build running), #127 (G3 DONE), #134 (final fine-tune).
- **WIRING-IN-PROGRESS (completeness-critic gaps being closed by a dedicated wiring pass):**
  (a) register #127/#134/#136-140 in `canonical_task_status.jsonl` (ledger frozen 2026-06-10 — prose-only today);
  (b) register the stretched-exp d_seg model `d=0.00566·exp(−(ep/4263)^0.860)` in the canonical_equations registry
  (the "sub-0.15 feasible ~14.5k ep" thesis rests on it; currently prose-only = tribal-knowledge violation);
  (c) register the decisive probe outcomes (B/C/D/A/compress-solve/partition-store) in `probe_outcomes.jsonl`
  (only partition-store is registered today; the rest claim "continual-learning ACTIVE" but aren't queryable).
