# SESSION SYNTHESIS — canonical source-of-truth (2026-06-17 → 2026-06-18)

**Operator: "review all design/research memos + conversation from today; ensure optimal + no signal loss."**
This is the SINGLE entry point for the next session. It resolves every supersession/correction so the
record is coherent (several of today's memos corrected each other). All `[contest-CPU advisory]` unless
noted; **exact pointer UNMOVED at 0.19110** (G3 confirmed; this session did NOT move it — stated plainly per
the GOAL firewall). The win this session is structural: a measured contest-grade pipeline + a concrete,
grounded three-lever path to sub-0.15 on the ranking axis, + recovery of two wrongly-buried levers.

## 🏁 TERMINAL FINDING (2026-06-19, late — READ FIRST) — representation-axis sub-0.15 is EXHAUSTED; frontier ~0.19110 near the real floor
The operator-directed VCM/coding-for-machines wave + every feasibility gate resolved: **sub-0.15 is not reachable for ANY known representation family.** All measured-closed above 0.19110: rate-shrink 3-way (re-pack DEAD #152 / bit-shrink CAPS #157 / deletion ~0 #153) + factored capacity-wall + static-geometry survival-wall + sub-pixel/warp RED (#149/#148) + task-space #155 NO-GO (pose bundled #158 / d_seg texture-walled) + generative amortized NCA RED (#146 — iteration BEATS the one-shot wall 0.31% [sic 0.31×] but PYRRHIC: sub-0.15 d_seg needs ~628K params → rate 0.230 > frontier). Binding walls: **SegNet interior TEXTURE-dependence + capacity d_seg∼29.3·params^−0.71 + rate/d_seg tension.** CLAUDE.md **S_floor 0.11797 REFUTED** as realizable (rate-only bound assuming d_seg→0 byte-cheap; the texture+capacity walls falsify it; realizable floor ~0.19). NOT a kill (Forbidden-premature-KILL): reactivate on a genuinely-DIFFERENT axis (NOT a frame-rep) or a measured byte-closed S<0.19110. Canonical: `TERMINAL_FINDING_representation_axis_sub015_exhausted_20260619.md` + memory `project_representation_axis_sub015_exhausted_frontier_near_floor_20260619`. The NEW DIRECTION below is the PATH that led here (preserved); the families it proposed are the ones now closed.

## 🧭 NEW DIRECTION (2026-06-19) — the contest IS indirect-RD / coding-for-machines; pivot to TASK-SPACE coding on the BINDING rate term
The VCM research program (theory #151 + methods #150, both committed; coding-primitive #152 + domain-tricks #145/#156 + the P-SUFF probe #153 running) converged on a reframe that supersedes the d_seg-vehicle framing below:
- **Our contest = an INDIRECT (remote) rate-distortion problem** (CEO problem / coding-for-machines): distortion is on `f(X)=(SegNet-argmax, PoseNet-6dim)`, NOT pixels. The rate hierarchy `R_Y ≤ R_X ≤ R_Ỹ` puts **EVERY vehicle we've built (PR95/HNeRV/bc20/0.19110 frontier) on the DOMINATED rung** — they pay bits to reconstruct RGB the frozen scorer IGNORES. The unbuilt **task-space / quotient code** (code only the scorer-sufficient statistic) is the prize.
- **RATE is the binding term (62%)** — 6 of the survey's top-8 levers attack rate; the campaign over-indexed on d_seg.
- **RATE-axis correction (#152, coding-primitive layer):** the decoder weights (91%) are ALREADY constriction-range-coded at the lossless symbol-entropy floor (~5.63 b/param; recompress GROWS them) → **RE-PACK with any learned/ELIC/MLIC entropy model is DEAD** (our `balle_hyperprior_codec` STATIC_WINS_FALLBACK proves it). The **ONLY live rate lever = change the SYMBOLS** (lossy task-RD quant surviving the frozen scorers): "5.63 b/param is the floor of the int8 symbols WE CHOSE, not of the weights." → W1 per-tensor Δ-search (numpy $0 decisive) · AdaRound (decode-unchanged, no retrain) · exact-sensitivity KKT allocation (#157) · codec co-design only carries per-channel SCALES as cheap side-info (not a better coder). Genuine OSS to lift: CompressAI `Elic2022Chandelier`. This is WHY the int5 "structural" wall was un-nuanced (it kept uniform-int5 symbols + a fixed codec).
- **#157 DONE (the exact-KKT solve, byte-closed CPU-authority) — the SMART version, and it gives a now-LEGITIMATE structural finding:** the exact non-uniform allocation DOMINATES generic uniform int5 (mb5.5 S=0.5411@127.8KB < generic 0.5593@142.9KB — lower S AND fewer bytes; rgb-heads int8-protected, stem coarsened; water-fill works). BUT the RGB-decoder bit-shrink CAPS above the pointer, monotone toward int8 baseline 0.1965: at mb6.0 **d_seg=0.00259 → its term 0.259 ALONE > pointer 0.191** → no allocation point crosses. **Bit-shrinking the EXISTING RGB-reconstruction decoder is CLOSED (proven the smart way, survived the exact-solve bar): it is task-DENSE.** → the ONLY remaining rate path is the **task-space code #155** (a structurally different representation, not shrinking the RGB renderer), gated on #153's invariant-mass. #157's reusable exact-sensitivity producer + KKT allocator = the bit-allocator HOOK for #155/the own-vehicle.
- **#153 P-SUFF DECISIVE VERDICT (the 3rd rate confirmation): `RED_FRONTIER_NEAR_TASK_RD_FLOOR_LITTLE_INVARIANT_MASS`.** The frontier has ~ZERO jointly-exploitable scorer-invariant byte mass — per-tensor bit-floors look like 0.65% savings but DON'T COMPOSE (jointly: d_seg 0.000538→0.000749, S 0.184→0.209 WORSE; super-additive interactions). So **re-pack DEAD (#152) + bit-shrink CAPS (#157) + deletion ~0 (#153) = all three ways to extract free rate from the existing frontier are closed by measurement. The frontier is TASK-DENSE / near the task-RD floor for reconstruct-RGB.** ⇒ **"RGB is wasteful" is true in PRINCIPLE (indirect-RD) but our frontier — a good codec — already captured ~all the waste; trimming/shrinking it for sub-0.15 is DEAD.** The remaining sub-0.15 path (NOT bounded by #153, which only measured the frontier) = a **fundamentally-different, geometry-anchored, FROM-SCRATCH task-space rep (#155+#158)** that BEATS a near-floor 177KB frontier by exploiting the KNOWN scene geometry (exact homography) + downloadable pose GT the frontier never used — a high bar against the measured walls (capacity ∼params^−0.71, survival ~0.0067, generative-fragility). This is the genuine strategic fork surfaced to the operator.
- **#155 FEASIBILITY GATE RESOLVED → NO-GO (2026-06-19; operator-approved $0 gate, all components measured RED):** pose-side (#158) RED — pose is BUNDLED with the full-frame recon, no separable cheap code; the comma2k19 GT is the SMOOTH PHYSICAL trajectory but the d_pose target is PoseNet's JITTER-dominated output (corr 0.72), so the GT is NOT a useful answer key. d_seg-side (#149/#148) RED — camera-res sub-pixel placement is a real 12× boundary-band lever BUT floors ABOVE the frontier (flat interiors lose texture; d_seg-term > 0.056 even if the boundary code were free) + byte-explodes (~68KB/frame); cross-frame warp closes <15% of drift (net loss). rate RED (3-way). **Every geometric/structural/task-space trick helps the WHERE (layout) but not the binding WHAT (interior boundary TEXTURE), which the frontier already codes near-minimally.** ⇒ **the from-scratch task-space rep is NOT a cheap win; do NOT build it.** The ONLY known sub-frontier path is the frontier's OWN move — a better LEARNED CONTINUOUS-TEXTURE decoder (generative #146 best-shot / capacity-RD own-vehicle #78), the capacity-walled + fragile path. **Honest comprehensive conclusion: the frontier ~0.191 is near the real achievable floor across ALL tested+theorized families (learned-decoder, static-geometry, generative, task-space); the binding wall everywhere is SegNet interior texture-dependence + capacity.** NOT a unilateral kill (Forbidden-premature-KILL): reactivation = a measured better-than-frontier continuous-texture decoder. Reusable bytes banked: #157 KKT bit-allocator + #149 sub-pixel boundary-band top-up (for any future textured vehicle). REMAINING $0 confirmation in flight: #144 polynomial (the continuous-interior-fill d_seg floor).
- **Floor:** S_floor 0.118 is a loose rate-only bound; true task-RD S* is strictly in **(0.118, 0.191)**, reachable ONLY by the task-space rep.
- **The FROZEN-INSTANCE exploit:** the field optimizes task-RD in expectation; ours is ONE frozen known instance → compute the EXACT per-instance optimum (exact histograms, exact sufficient statistic, exact polytopes/null-spaces) — provably ≤ any learned-general codec → catch up to, then SURPASS, the field.
- **Theory-grounded campaign facts:** convex-IB explains why margin-hinge works; deterministic-IB explains lossless-recode exhaustion; RDC-not-RDP means pixel/perceptual fidelity is a tax we don't owe.
- **The probe queue (no signal loss):** #153 P-SUFF/task-ablation (the decisive dominated-rung measurement — RUNNING) · #154 rate-axis $0 queue (weight-entropy/latent-AR/sensitivity-bit-alloc/entropy-penalized) · #155 level-set/fiber QUOTIENT codec (the paradigm prize) · #147 int5 best-shot (NeuroQuant-grounded, RUNNING) · #144 polynomial (RUNNING) · #156 domain-tricks (RUNNING).
- **DOMAIN UNLOCK (#156, VERIFIED):** the contest video IS a known public **comma2k19 RAV4 segment** (segment ID confirmed in `upstream/public_test_segments.txt`; camera K=[[910,0,582],[0,910,437]] verified) → its ego-motion GT (`comma2k19 global_pos/`) is the DOWNLOADABLE d_pose answer key (~1-2 DOF, deg-≤4 poly → hundreds of bytes); the exact homography pins static-class geometry (horizon row 437, road trapezoid, hood, sky) = near-zero-byte priors. **CAVEAT (NO-FAKE): GT pose is a PRIOR/oracle, NOT a drop-in** — eval scores the frozen PoseNet on our RECONSTRUCTED frames, never a stored vector; archive stays self-contained. → probes #158.
- Canonical docs: `frozen_instance_exploit_catch_up_then_surpass_vcm_20260619.md` (the lens), `vcm_theory_primitive_layer_*` (#151 math+floor), `vcm_taskaware_compression_sota_survey_*` (#150 ranked OSS), `vcm_coding_primitive_layer_*` (#152 re-pack-dead/change-symbols), `comma_openpilot_domain_tricks_*` (#156 provenance), memory `project_contest_is_indirect_rate_distortion_task_space_coding_20260619` + `project_contest_source_is_known_comma2k19_rav4_segment_*`.
- HONEST CAVEAT: the task-sufficient code still must be REALIZED as a frame the frozen SegNet reproduces through the roundtrip (the survival/texture wall, measured). "Store the argmax map" = the RED partition-store. P-SUFF (#153) measures whether the dominated-rung gap is actually large for THIS vehicle before any build.

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
| 8b | generative/iterated **CONTINUOUS-TEXTURE** (latent-conditioned NCA) | **AMBER but a FRAGILE, NON-REPRODUCIBLE SPARK — RE-OPENED with heavy caveats** — *when it converges* (~2/8 runs; the headline h128/seed1234 config COLLAPSED to 0.549 on re-run; daemon h128 0/3 converged) continuous RGB trained THROUGH the scorer reaches realized d_seg **0.00337 (1.31× frontier), interior 0.0, boundary 0.079** on ONE frame; beats polynomial-LS (0.00609) + flat (0.0067) — continuity thesis real, twice-measured. BUT: (i) **typical run COLLAPSES** (~0.5; MPS non-determinism/Muon-chaos; the canonical Mordvintsev POOL-replay was NOT used → NOT given its best shot); (ii) measured **S=0.415 (NOT sub-0.15; n_sub015=0)**; (iii) rate 0.0191 assumes a 33K SHARED rule amortizes over 600 frames — **UNTESTED (fresh rule per frame)**; sub-0.15 is a PROJECTION (cut boundary 5× AND sharing holds). **DECISIVE next $0 test (#146): convergence-robust (POOL-replay) shared-rule generalization (n=1→16-48 frames, ONE rule, AVERAGE d_seg, TRUE amortized rate)** — gates any multi-day build | `generative_axis_continuous_texture_nca_AMBER_*` |
| — | **continuity thesis VALIDATED** | trained-through-scorer 0.00337 < polynomial-LS 0.00609 < flat 0.0067 → continuity + training-through-the-roundtrip is the d_seg lever; the open question is now AMORTIZATION, not whether continuous beats flat | — |

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

## RECURSIVE ADVERSARIAL REVIEW outcome (2026-06-19, `recursive_adversarial_review_recent_negatives_*`, commit 401a2abaa)
Adversarial review (FALSIFY mandate, bare-math vs upstream, no-fake, best-shot) of the recent negatives:
- **4 structural REDs SURVIVE on their d_seg-axis conclusions** (curve-core, flat-partition-NCA, capacity-wall qualitative, eval-roundtrip math — every roundtrip operator re-derived against `upstream/` ✓). No fakes (classes 1-8); 2 auto-label bugs were CAUGHT+corrected (discipline working).
- **1 genuinely UNDER-POWERED negative → RE-TEST (highest-EV):** int5 Path-B cap used per-tensor abs-max quant — NO per-channel / NO LSQ / NO outlier-handling (the exact canonical low-bit fixes). "STRUCTURAL to int5" is over-claimed (it's structural to per-tensor-abs-max). **Path-B attacks the frontier RATE (binding 62%); a best-shot int5 (per-channel+LSQ) that recovers d_seg → rate ~0.07 → S~0.14 sub-0.15.** RE-TESTING (#147).
- **Over-claims corrected:** AMBER title ("strongest sub-0.15 candidate" → fragile non-reproducible RED-on-re-run; bnd_flip 0.079 is the lucky converged frame, polynomial's 0.15 is robust); curve docstring ("curves+colours through roundtrip" → only colours; geometry is frozen DP-decimation); factored-LF "10.7M params" is a 2-point extrapolation ±a decade.
- **Generalization caveat:** all d_seg gates measure on 3 CONSECUTIVE frames of ONE segment (n=3≈n=1) → REDs MORE trustworthy (walls worsen on diverse scenes), AMBER optimism triply suspect.
- **2 UNTESTED pincer corners (the pincer is NOT airtight):** (1) cross-frame keyframe+tiny-warp (pay the boundary ONCE; consecutive last-frames drift ~0.33px; the frontier already exploits temporal) → #148; (2) camera-res sub-pixel boundary placement (the entire remaining d_seg localizes to the 1px band at 384; placing it sub-pixel at 874×1164 BEFORE the downsample D averages is closed-form, $0) → #149. Both stay inside "render an RGB frame" (no legal non-frame path — `TensorVideoDataset` requires raw uint8 camera-res).

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
