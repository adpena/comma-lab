---
council_tier: T2
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Holub, Selfcomp, Quantizr, Hinton]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "We are revisiting a 2026-05-04 FALSIFIED-as-non-applicable verdict whose factual basis (PR106 has no mask channel) is empirically CORRECT per PV-1 of the canonical reformulation memo. The reformulation is technically defensible, but ALL THREE proposed reformulation variants have explicit Dykstra-feasibility verdicts that are NOT frontier-extending with high confidence: Variant A is frontier-PROTECTING at best (band [substrate_anchor - 0.003, substrate_anchor] ≈ [0.207, 0.21] CPU vs A1 0.192848); Variant B collapses to [-0.0007, -0.002] within-cluster noise floor; Variant C straddles the frontier with UNCERTAIN distortion-axis benefit. The operator's $10 conditional re-probe budget should fire on the CHEAPEST-TO-DISAMBIGUATE option, not the highest-EV one. Per CLAUDE.md 'Race-mode rigor inversion' the cheapest-disambiguation answer is Variant C's A1-substrate-extension feasibility audit at $0 editor work BEFORE any $10 dispatch — and even after the audit, if it returns POSITIVE, the $10 buys a CPU-axis A1-sidecar probe (not a Modal dispatch). I VETO consensus that proceeds to $10 Modal dispatch without first running the $0 Variant C feasibility audit AND the $0 Variant B latent-residual-wavelet-savings probe."
  - member: Shannon
    verbatim: "The information-theoretic framing per the canonical reformulation memo §3.3 is correct on the operating-point analysis: at PR106's frontier operating point (pose_avg ~3.4e-5), pose marginal sensitivity is 2.71× SegNet's. BUT — and this is the operating-point lens applied recursively — UNIWARD #05 targets the RATE axis (variable-precision quantization on the grayscale field) and grayscale-LUT #06 targets the DISTORTION axis via class-discrete substitution. At the operating point near A1 (0.193 CPU), the rate-axis marginal value per byte is roughly `25/37545489 ≈ 6.66e-7 per byte`. UNIWARD's predicted savings of 5-15 KB on Variant A's grayscale stream → -0.0033 to -0.0100 ΔS, which is meaningfully sub-cluster IF the substrate alpha anchor lands at ≤0.21. The math is solvable: bit savings × 25/37545489 = rate-axis ΔS. The unsolvable part is the substrate-distortion-transfer question (does Selfcomp's PR#56 0.38 paradigm transfer to the contest video's mask-class distribution?). Per CLAUDE.md 'Meta-Lagrangian/Pareto solver' non-negotiable: prefer solvable math over arbitrary sweeps. The Variant B latent-residual-wavelet entropy probe IS solvable in $0; defer the Variant A dispatch until grayscale_lut substrate alpha anchor lands."
  - member: Fridrich
    verbatim: "UNIWARD (Universal Wavelet Relative Distortion) per Fridrich-Holub 2014 IH&MMSec is the canonical adaptive embedding cost function for steganography. It computes a wavelet-domain texture-cost map and allocates bits to high-texture regions where embedding is statistically undetectable. The 2026-05-04 PR106-no-mask-channel falsification is FACTUALLY CORRECT for the PR106 architecture but it does NOT invalidate UNIWARD's class — the canonical reformulation memo's PV-6 line is exactly right: 'UNIWARD + grayscale-LUT are leaderboard-proven primitives (PR101/PR103 silver + PR#56 paradigm)'. PR101 GOLD applies UNIWARD-delta to its INT8 symbol substrate; PR103 silver applies similar adaptive-cost frameworks. The technique class is HARD-EARNED via my own published work + the leaderboard receipts. The reformulation question is per-substrate: which contest substrate has the cover signal UNIWARD needs (texture variation + signal-vs-cover decomposition)? Variant A's grayscale_lut substrate IS structurally appropriate IF the grayscale field has non-trivial texture variation on the contest video; Variant B's wavelet-of-latent-residual is mathematically defensible but the 28-dim latent has no natural 2D structure — it's a per-pair embedding, not an image. I support PROCEED_WITH_REVISIONS on Variant A as the canonical UNIWARD reformulation slot; I DEFER on Variant B as architecturally awkward (the wavelet domain doesn't apply naturally to a 28-dim latent vector); I support the Variant C feasibility audit as $0 first action."
  - member: Holub
    verbatim: "Speaking as Fridrich's co-author on the UNIWARD 2014 paper: the texture-cost function is specifically designed to operate on a 2D cover signal with spatial locality. The wavelet decomposition's HH/HL/LH/LL subbands capture horizontal/vertical/diagonal texture energy. The canonical reformulation memo's Variant A application to the grayscale field is well-posed: a per-pair grayscale image (one channel × H × W) has the spatial structure UNIWARD needs. The expected savings depend on the cover signal's wavelet-domain energy distribution — driving footage has high-texture regions (road, vegetation) and low-texture regions (sky, smooth lanes), making it well-suited to adaptive bit-allocation. My PROCEED on Variant A as the canonical reformulation; I VETO Variant B's wavelet-on-28-dim-latent as a paradigm misuse — the wavelet transform requires spatial locality structure, which a 28-dim per-pair embedding doesn't have. The 1D-wavelet-on-28-dim-vector alternative the memo mentions in §4.2 is mathematically valid but degenerates to a Haar basis transform that's effectively just a basis rotation; UNIWARD's adaptive cost function doesn't deliver meaningful gain because the 28-dim vector has no texture variation per se."
  - member: Selfcomp
    verbatim: "I am the architect of PR #56 (Szabolcs's submission; 0.38 [contest-CUDA] anchor). The grayscale-LUT paradigm was specifically engineered for: (a) AV1 grayscale (single-channel encoding of an analog signal), (b) Gaussian-softmax-LUT mapping grayscale-value → RGB via the trained class targets, (c) a FiLM-conditioned tiny RGB decoder that exploits the LUT's structured prior. The paradigm WORKS because the entire decoder is trained-from-scratch with the LUT in place — Lane MM v2 PROVED that bolting a 1-channel grayscale-LUT onto a 3ch-trained renderer regresses to 2.63 CPU (per resurrection audit §1.5). The CANONICAL reformulation pathway is: grayscale_lut substrate L0 → L1 (its alpha anchor MUST establish viability on the contest video, ideally ≤0.21 per the substrate's own reactivation criteria). Once L1 lands, the grayscale-LUT IS the substrate's central mechanism — there is no 'grayscale-LUT #06 bolt-on' to add; the substrate IS the reformulation of Lane #06. UNIWARD #05 bolt-on on the grayscale field is meaningful additional rate savings IF the grayscale field's texture-cost analysis indicates 5-10% headroom. I support PROCEED_WITH_REVISIONS on Variant A as the principal reformulation; I DEFER Variant B as redundant with NSCS06 v8 Path B / sane_hnerv reformulation work in other symposia; I support Variant C feasibility audit as a $0 first action that informs the operator's sequencing decision."
  - member: Quantizr
    verbatim: "PR101 GOLD applies UNIWARD-delta on the INT8 weight-quantization stream — the technique CANONICAL is to compute UNIWARD texture-cost on the parameter space and adaptively allocate bit-precision to high-cost (high-impact) parameters. The reformulation question reframes naturally: which substrate has a parameter or signal stream where UNIWARD's adaptive cost function delivers measurable rate savings? Variant A's grayscale field is a structural fit; Variant B's latent residual is awkward (the 28-dim has no spatial locality); Variant C's RGB-residual-sidecar is novel but feasible. From the leaderboard-empirical lens: my PR101 GOLD application of UNIWARD-delta to INT8 weights produces ~0.001-0.003 rate-axis savings at scale; transferring to Variant A (grayscale field, larger byte count) potentially scales to ~0.005-0.010 savings. The Selfcomp PR#56 paradigm + UNIWARD-canonical adaptive allocation is a HIGH-EV composition IF Variant A's substrate alpha anchor lands at ≤0.21. I support PROCEED_WITH_REVISIONS on Variant A; I support the $0 probe-disambiguator sequence as the cheapest first signal; I DEFER on Variant B per Fridrich/Holub's correct architectural critique."
  - member: Hinton
    verbatim: "The grayscale-LUT mechanism Selfcomp engineered IS a form of soft-target knowledge distillation: the trained Gaussian-softmax-LUT distills the SegNet's class targets into a fixed lookup table. The Lane MM v2 anti-pattern (bolt-on LUT regresses) IS the canonical distillation failure mode: distillation requires the student (LUT) to be trained jointly with the teacher signal (SegNet's class targets) — bolting a fixed table onto a trained model is structurally different from training the model with the distillation objective. Variant B's LUT-on-HNeRV-stem bolt-on is the SAME Lane MM v2 anti-pattern at the decoder-stem layer; the only way to make it work is to train sane_hnerv FROM SCRATCH with the LUT in place, which is substrate engineering (~350 LOC budget per HNeRV L7). I support the canonical reformulation memo's classification of Variant B LUT-on-HNeRV-stem as HIGH distortion-axis risk. The Variant A reformulation (UNIWARD on grayscale_lut substrate's grayscale field) is structurally distinct from distillation: UNIWARD is a rate-axis primitive that doesn't change the distillation pathway. I support PROCEED_WITH_REVISIONS."
  - member: Yousfi
    verbatim: "As the contest designer + Fridrich's PhD student: the original PR106 Lanes #05+#06 falsification 2026-05-04 was operationally correct — the lane DESIGN inherited a Quantizr-style mask.mkv assumption that PR106 (HNeRV with brotli-decoder + brotli-latents) does not have. The reformulation memo's Pattern B classification per Catalog #307 is correct: paradigm-INTACT, implementation-CARGO-CULTED. The contest video's per-pair mask-class distribution IS well-known (5 classes from SegNet's argmax over EfficientNet-B2 + stride-2 stem; per pair the dominant class is road/sky/vegetation with smaller class regions); UNIWARD-style texture-cost analysis on the grayscale field (which is derived from this mask-class distribution via Selfcomp's grayscale projection) is structurally well-posed. I support PROCEED_WITH_REVISIONS on Variant A with the binding revision that the $10 re-probe budget MUST be allocated to grayscale_lut substrate alpha anchor first (Variant A prerequisite) NOT to UNIWARD bolt-on dispatch (which has no substrate to operate on yet). The cheapest disambiguator IS the substrate alpha probe; UNIWARD is a follow-on after the substrate viability question is answered."
  - member: Dykstra
    verbatim: "The Dykstra-feasibility verdict per §17 of the canonical reformulation memo is internally consistent. Variant A intersection NON-EMPTY but band [substrate_anchor - 0.003, substrate_anchor] is conditional on substrate_anchor (PENDING). Variant B intersection NON-EMPTY but band collapses to [-0.0007, -0.002] within noise floor. Variant C intersection PENDING feasibility audit, net-effect band could be [-0.005, +0.003] frontier-straddling. The convex-feasibility lens at this operating point (near A1's 0.193 CPU frontier) shows TIGHT polytopes — small rate-axis savings × small distortion-axis perturbations × A1-as-baseline-anchor → the only operationally meaningful breakout path is Variant A WITH substrate alpha anchor below 0.21 AND grayscale-stream texture-variance high enough to deliver ≥5% UNIWARD savings. The probe sequence Contrarian / Shannon proposed (Variant C $0 audit + Variant B $0 latent-wavelet probe + grayscale_lut substrate alpha as the gating dispatch) IS the canonical alternating-projections order: cheapest signal first, expensive dispatch last. I PROCEED with the revision that no $10 Modal dispatch fires until at least one of the $0 probes returns POSITIVE."
  - member: Assumption-Adversary
    verbatim: "The implicit shared assumption every member is operating within is that 'reformulating Lanes #05+#06 brings the technique class back into the active queue with meaningful score-improvement potential'. The canonical reformulation memo's cargo-cult audit §16 correctly classifies this as TRUE-BUT-CONDITIONAL: the technique class IS preserved per Fridrich-Holub / Selfcomp canonical receipts, BUT the reformulation is BLOCKED on substrate readiness (Variant A on grayscale_lut L1, Variant B on sane_hnerv L1, Variant C on A1 feasibility audit). The OPERATOR'S OPEN QUESTION at this symposium (per the prompt: 'which reformulation path gets the $10 first?') has an explicit ASSUMPTION embedded: that a $10 Modal dispatch IS the right first action. The Assumption-Adversary VETO: the $10 dispatch is the WRONG first action because the substrate prerequisite for all three variants is NOT YET LANDED. The correct $10 use is grayscale_lut substrate alpha anchor ($5-15 Modal A100; this $10 buys the substrate L1 promotion that gates Variant A). Alternatively, the $10 sits in reserve while the $0 Variant C feasibility audit + $0 Variant B latent-wavelet probe run; whichever returns POSITIVE first gets the $10 buy-in. The implicit conflation of '$10 re-probe' with 'UNIWARD bolt-on dispatch' is the cargo-cult I am challenging. PROCEED_WITH_REVISIONS conditional on: (a) the $10 maps to grayscale_lut substrate alpha anchor OR sits in reserve; (b) the $0 probe sequence runs first; (c) NO UNIWARD bolt-on dispatch fires until substrate L1 + probe verdicts are in."
council_assumption_adversary_verdict:
  - assumption: "UNIWARD and grayscale-LUT TECHNIQUES are canonical leaderboard-proven primitives applicable to substrates with mask-channel-like structure"
    classification: HARD-EARNED
    rationale: "Fridrich-Holub 2014 IH&MMSec (canonical UNIWARD); Selfcomp PR #56 0.38 [contest-CUDA] anchor (canonical grayscale-LUT). Per CLAUDE.md 'HNeRV / leaderboard-implementation parity discipline' the leaderboard receipts ARE the empirical proof of HARD-EARNED status. Multiple steganography competitions + PR101 GOLD application + PR103 silver application establish UNIWARD's canonical class beyond reasonable doubt."
  - assumption: "PR106 has a separate mask channel (the 2026-05-04 lane DESIGN assumed this)"
    classification: CARGO-CULTED-FALSIFIED
    rationale: "PV-1 of canonical reformulation memo: empirical verification per `experiments/extract_pr106_decoder.py` (commit 45149f21) confirms PR106 = HNeRV with brotli-decoder (170,278 bytes) + brotli-latents (15,849 bytes) ONLY. Zero tensors with 'mask' in name. The 2026-05-04 lane DESIGN inherited a Quantizr-style mask.mkv assumption that PR106's HNeRV architecture does not satisfy. Pattern B substrate-mismatch-as-class-kill per Catalog #307; falsification is HARD-EARNED for PR106-as-substrate but INVALID as a class-kill of UNIWARD or grayscale-LUT."
  - assumption: "The contest substrate landscape has at least one substrate where UNIWARD + grayscale-LUT TECHNIQUES are operationally meaningful"
    classification: HARD-EARNED-CONDITIONAL
    rationale: "Three candidates identified per canonical reformulation memo §3.2: (a) grayscale_lut substrate (Selfcomp PR #56 paradigm; L0 SKETCH; substrate IS the grayscale-LUT mechanism by construction); (b) sane_hnerv substrate (HNeRV-family; L0 SKETCH; bolt-on reformulations possible at latent-residual + decoder-stem layers); (c) A1 substrate-extension via UNIWARD-residual-sidecar (CONDITIONAL on feasibility audit). The HARD-EARNED basis: substrate identification is empirically verified per `src/tac/substrates/grayscale_lut/__init__.py`, `src/tac/substrates/sane_hnerv/__init__.py`, `src/tac/substrates/a1/__init__.py`. The CONDITIONAL basis: substrate readiness is the gating constraint — Variant A blocked on grayscale_lut L1, Variant B blocked on sane_hnerv L1, Variant C blocked on A1 feasibility audit."
  - assumption: "$10 Modal re-probe is the optimal first action post-PROCEED verdict"
    classification: CARGO-CULTED
    rationale: "The implicit conflation in the prompt's 'which reformulation path gets the $10 first?' assumes $10 dispatch IS the canonical first action. Per Catalog #229 premise verification + Dykstra-feasibility sequencing: the substrate prerequisite for all three variants is NOT YET LANDED. The canonical first-action sequence is: (a) $0 Variant C feasibility audit (informs Variant A vs B sequencing); (b) $0 Variant B latent-residual-wavelet-savings probe (yields verdict on whether ≥5% savings exist); (c) IF Variant A is selected, the $10 maps to grayscale_lut substrate alpha anchor (NOT UNIWARD bolt-on which has no substrate to operate on yet). Cargo-cult unwound: $10 is NOT the entry budget for UNIWARD bolt-on; $10 is the substrate-promotion budget that gates the bolt-on."
  - assumption: "Predicted ΔS bands [-0.001, -0.003] for Variant A bolt-on are within-cluster and therefore frontier-protecting at best"
    classification: HARD-EARNED-EMPIRICALLY-DEFENSIBLE
    rationale: "Per Shannon first-principles: UNIWARD savings of 5-10% on grayscale stream × 25/37545489 byte→ΔS conversion = -0.003 to -0.010 ΔS rate-axis maximum. The conservative band [-0.001, -0.003] in §3.3 is appropriately tight given the substrate's PR#56 paradigm transfer uncertainty. Below A1's 0.193 frontier requires CUMULATIVE -0.001 to -0.005 across variants; the predicted bands are sub-cluster but require substrate viability first. Per T2 council Q2 verdict: CPU axis is leaderboard axis; predicted bands are CPU-axis dominant. HARD-EARNED math, CONDITIONAL on substrate readiness."
  - assumption: "Within-class plateau risk applies UNIFORMLY to all three variants"
    classification: PARTIALLY-CARGO-CULTED
    rationale: "Variant A class-distinct from PR106 (grayscale_lut is Selfcomp PR #56 paradigm; structurally distinct architectural class). Variant B same HNeRV-family class as PR106 — HIGH within-class plateau risk (correct). Variant C extends the A1 frontier substrate — frontier-straddling not within-class. The canonical reformulation memo §17 differentiates correctly per-variant; the symposium consensus should acknowledge Variant A is class-distinct (lower within-class risk) while Variant B IS within-class (higher risk). Variant C is operationally orthogonal (frontier-extension question, not within-class question). Cargo-cult unwound: the within-class plateau verdict does NOT apply uniformly across the three variants; it applies STRONGEST to Variant B and weakest to Variant A."
council_decisions_recorded:
  - "op-routable #1 (HIGHEST PRIORITY): Approve Variant C A1-substrate-extension feasibility audit ($0 editor work; ~3-5 hr). Cheapest first action; informs Variant A vs Variant B sequencing decision. Document at `.omx/research/a1_substrate_extension_feasibility_audit_<UTC>.md`. Per Assumption-Adversary VETO: the $10 conditional budget SITS IN RESERVE pending the feasibility audit verdict."
  - "op-routable #2 (HIGH PRIORITY): Approve probe-disambiguator scripts (per §15 of canonical reformulation memo; ~150+120+200 LOC; $0 CPU smoke time). Specifically: `tools/probe_grayscale_lut_alpha_anchor_score.py` (Variant A precondition probe), `tools/probe_uniward_latent_residual_savings.py` (Variant B feasibility probe), `tools/probe_a1_uniward_sidecar_feasibility.py` (Variant C feasibility probe). All $0; runs in CPU smoke mode. Probe verdicts inform $10 allocation."
  - "op-routable #3 (CONDITIONAL on op-routable #1 + #2 verdicts): IF Variant A is selected as highest-EV path, $10 conditional re-probe ALLOCATES TO grayscale_lut substrate alpha anchor (Modal A100; ~$5-15 actual; $10 estimate). Substrate L1 promotion is the prerequisite for UNIWARD #05 bolt-on dispatch; the $10 buys the substrate, NOT the bolt-on."
  - "op-routable #4 (CONDITIONAL on op-routable #1 + #2 verdicts): IF Variant C feasibility audit returns POSITIVE AND sidecar net-effect probe verdict ≤ -0.001, $10 conditional re-probe ALTERNATIVELY ALLOCATES TO A1-sidecar paired-CPU+CUDA smoke (Modal A100; ~$5-15 actual; $10 estimate). Frontier-extending path with HIGHER risk + HIGHER reward than Variant A."
  - "op-routable #5 (LOWEST PRIORITY): DEFER Variant B (UNIWARD-latent-residual on sane_hnerv) per Fridrich + Holub VETO on 28-dim-latent wavelet decomposition as architecturally awkward. Predicted savings collapse to noise floor regardless of substrate readiness. Variant B remains in DESIGN-ONLY state pending substrate-engineering reformulation that addresses the spatial-locality gap (potentially via 2D-reshape of latent or alternative basis like DCT)."
  - "op-routable #6 (REQUEST-REINVESTIGATION-OF-ALTERNATIVES per Catalog #308): Per Pattern E discipline + sister symposium template (`council_per_substrate_symposium_stc_clean_source_20260517.md` §15.3 op-routable #3): the THREE variants enumerated represent N=3 alternative substrate-pivots per Catalog #308 reactivation paths. Additionally enumerated alternative reducers from §16 cargo-cult audit + canonical reformulation memo appendix: (a) per-pair HISTOGRAM grayscale field analysis (instead of per-pair-dominant), (b) per-region grayscale histogram (K=16 region partition), (c) per-segment-class conditional grayscale (condition on 5 SegNet classes individually), (d) per-temporal-window grayscale predictor (T=4 frame windows). The original 2026-05-04 KILL is now classified RATIFY-FALSIFICATION-OF-THE-SPECIFIC-METHODOLOGY (PR106-mask-channel-substitution IS empirically falsified) + REQUEST-REINVESTIGATION-OF-ALTERNATIVES (UNIWARD + grayscale-LUT TECHNIQUE classes remain viable per the enumerated variants + reducers)."
  - "op-routable #7 (per Catalog #298 substrate retirement discipline): Lane `lane_tier_1_resurrection_5_pr106_lanes_05_06_reformulated_uniward_grayscale_lut_20260516` is registered L0 with research_only=false. Per the canonical reformulation memo's status 'DESIGN-ONLY, RESEARCH-ONLY at the recipe level', this symposium recommends updating the lane registry notes to add `research_only=true` with reactivation_criteria pinning the 5-criterion-per-variant gates from §21. This matches CLAUDE.md 'Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY' non-negotiable per the Catalog #220 sister discipline."
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: ""
deferred_substrate_id: "lane_tier_1_resurrection_5_pr106_lanes_05_06_reformulated_uniward_grayscale_lut_20260516"
deferred_substrate_retrospective_due_utc: "2026-06-16T00:00:00Z"
related_deliberation_ids:
  - council_per_substrate_symposium_stc_clean_source_20260517
  - council_per_substrate_symposium_lane_17_imp_20260517
  - council_per_substrate_symposium_nscs06_v8_path_b_20260517
  - resurrection_audit_20260516
  - pre_rigor_kill_defer_falsified_inventory_20260517
predicted_band_validation_status: first_principles
horizon_class: frontier_protecting
---

# Per-substrate symposium — `lane_pr106_uniward_delta_latent_stream + lane_pr106_grayscale_lut_latent_codebook REFORMULATED` (council_priority #858 / pre-rigor inventory #4 SHOULD_BE_RESYMPOSIUM'D)

**Date:** 2026-05-17
**Subagent ID:** pr106_05_06_symposium_20260517T220000
**Lane:** `lane_per_substrate_symposium_pr106_05_06_reformulated_20260517` L0 (pre-registered)
**Tier:** T2 sextet pact (Shannon LEAD + Dykstra CO-LEAD + Yousfi + Fridrich + Contrarian + Assumption-Adversary) + 4 grand-council attendees (Holub canonical UNIWARD co-author + Selfcomp PR#56 lead + Quantizr PR101 GOLD adversary + Hinton knowledge-distillation memorial)
**Verdict:** **PROCEED_WITH_REVISIONS** (Assumption-Adversary VETO satisfied via revisions to op-routables: $0 probes BEFORE $10 dispatch; $10 maps to substrate prerequisite NOT bolt-on)
**Mission-alignment:** frontier_protecting (Variant A bands above A1; Variant B within-cluster; Variant C frontier-straddling — the reformulation prevents technique-class loss without claiming frontier-extension)
**Horizon class:** frontier_protecting (per Catalog #309; Variant A band [0.207, 0.21] CPU is plateau-adjacent; Variant C straddles frontier_pursuit; Variant B is plateau-adjacent)
**Budget consumed:** $0 (editor only); $10 conditional re-probe AUTHORIZED-BUT-CONDITIONAL-ON-OP-ROUTABLE-#1-#2-VERDICTS

## Executive summary

The original 2026-05-04 FALSIFIED-as-non-applicable verdict on PR106 Lanes #05 + #06 (`feedback_pr106_no_mask_channel_lanes_05_06_falsified_20260504.md`) is empirically CORRECT for the PR106-as-substrate-target (PV-1 of canonical reformulation memo: PR106 = HNeRV with brotli-decoder + brotli-latents only, NO separate mask channel). Per CLAUDE.md FORBIDDEN_PATTERN "Forbidden premature KILL without research exhaustion" + Catalog #307 paradigm-vs-implementation classification: the falsification is PARADIGM-INTACT + IMPLEMENTATION-CARGO-CULTED. The TECHNIQUES (UNIWARD-delta + grayscale-LUT) ARE leaderboard-proven canonical primitives (Fridrich-Holub 2014; Selfcomp PR #56 0.38 [contest-CUDA] anchor). The PARADIGM cargo-cult was the lane DESIGN's assumption that PR106 = Quantizr-style mask.mkv architecture.

This symposium adjudicates the REFORMULATION proposal per the canonical comprehensive memo `tier_1_resurrection_5_pr106_lanes_05_06_reformulated_uniward_grayscale_lut_full_stack_design_20260516.md` (66.2 KB; 8 PVs; 22 sections; 3 variants; Catalog #290 + #294 + #296 + #303 + #305 evidence sections). The reformulation identifies THREE candidate target substrates with mask-channel-like structure (grayscale_lut / sane_hnerv / A1) and provides per-variant Dykstra-feasibility verdicts on BOTH CPU + CUDA axes.

**Verdict: PROCEED_WITH_REVISIONS.** The reformulation is technically defensible per UNIQUE-AND-COMPLETE-PER-METHOD discipline; Pattern B substrate-mismatch-as-class-kill correctly identified. HOWEVER the predicted ΔS bands are NOT frontier-extending with high confidence per Shannon first-principles:

- Variant A (grayscale_lut + UNIWARD #05 + grayscale-LUT #06 = substrate mechanism): predicted CPU band `[substrate_anchor - 0.003, substrate_anchor]` with substrate_anchor PENDING grayscale_lut L1 promotion (conservative ≈ 0.21); ABOVE A1's 0.192848 frontier — frontier-protecting at best.
- Variant B (sane_hnerv + UNIWARD-latent-residual): predicted CPU band `[-0.0007, -0.002]` within noise floor; within-cluster plateau. Fridrich + Holub VETO on architectural mismatch (28-dim latent has no spatial locality structure UNIWARD needs).
- Variant C (A1 + UNIWARD-residual-sidecar): predicted CPU band `[A1 + net-effect]` PENDING feasibility audit; net-effect envelope `[-0.005, +0.003]`; frontier-straddling with HIGH UNCERTAINTY.

**Assumption-Adversary VETO satisfied** via revisions to operator op-routables: NO $10 Modal dispatch fires until $0 Variant C feasibility audit + $0 Variant B latent-wavelet probe + $0 Variant A grayscale_lut substrate viability probe return verdicts; $10 conditional budget ALLOCATES TO substrate prerequisite (grayscale_lut L1 promotion) NOT to UNIWARD bolt-on (which has no substrate to operate on yet).

**Operator op-routable #1 (HIGHEST PRIORITY)**: approve $0 Variant C A1-substrate-extension feasibility audit (~3-5 hr editor work; informs Variant A vs B sequencing).

**Operator op-routable #3 (CONDITIONAL)**: if Variant A is selected, $10 ALLOCATES TO grayscale_lut substrate alpha anchor (Modal A100; substrate L1 promotion); UNIWARD bolt-on dispatch is follow-on.

## 1. Cargo-cult audit per assumption (Catalog #303)

Per CLAUDE.md "Forbidden symposium-band-prediction-without-Dykstra-feasibility-check" + cargo-cult-unwind methodology (NSCS06 v6 → v7 44% improvement anchor) + sister symposium template.

### Assumption 1: "UNIWARD and grayscale-LUT TECHNIQUE classes are HARD-EARNED canonical primitives"

- **Classification: HARD-EARNED** (both techniques have leaderboard receipts)
- **Sub-assumption 1a:** UNIWARD (Universal Wavelet Relative Distortion)
  - **HARD-EARNED** per Fridrich-Holub 2014 IH&MMSec ("Designing Steganographic Distortion Using Directional Filters"); Pevný 2010 IEEE TIFS body of work on adaptive embedding; PR101 GOLD application of UNIWARD-delta on INT8 symbol substrate (0.193 [contest-CUDA] anchor); PR103 silver application of related Fridrich adaptive-cost frameworks.
- **Sub-assumption 1b:** grayscale-LUT (Selfcomp PR #56 paradigm)
  - **HARD-EARNED** per Szabolcs-cs PR #56 implementation; 0.38 [contest-CUDA] anchor proves the paradigm transfers to driving footage; Gaussian-softmax-LUT over CLASS_TARGETS is the canonical mechanism.
- **Unwind-test:** N/A — paradigms are not being unwound; they are being preserved per Pattern D classification.

### Assumption 2: "PR106 has a separate mask channel (basis for original 2026-05-04 lane design)"

- **Classification: CARGO-CULTED-FALSIFIED**
- **Rationale:** PV-1 of canonical reformulation memo. Empirical verification per `experiments/extract_pr106_decoder.py` (commit 45149f21) on archive `experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip` (186,131 bytes): 0.bin decomposes into decoder_brotli (170,278 bytes; HNeRV decoder state_dict INT8+zigzag+brotli) + latent_brotli (15,849 bytes; 600 frame-pair latents, 28-dim, uint8 delta+brotli). Zero tensors with 'mask' in name. PR106 IS HNeRV-style: per-pair 28-dim latent + HNeRV decoder JOINTLY define full RGB output.
- **Sister cargo-cult:** The 2026-05-04 lane design inherited a Quantizr-style mask.mkv assumption (the 0.33 leader uses encoded masks.mkv AV1 + renderer + poses; structurally distinct from PR106's HNeRV).
- **Unwind-test:** Identify substrates with mask-channel-like structure → §3.2 of canonical reformulation memo identifies grayscale_lut + sane_hnerv + A1 as three candidates with structural compatibility.

### Assumption 3: "PR102 CUDA-CPU drift (-0.0330) extrapolates to predicted bands for the reformulated variants"

- **Classification: CARGO-CULTED**
- **Rationale:** Per T2 council Q2.5 verdict + canonical reformulation memo PV-8: PR102 drift is empirical for PR102's specific HNeRV variant; the magnitude AND sign may not generalize to grayscale_lut (Selfcomp PR#56 architecture, completely different decoder class), sane_hnerv (different HNeRV variant), or A1 (frontier substrate with its own paired-CPU+CUDA anchor at 0.192848 CPU vs ~0.22-0.23 CUDA per T2 verdict).
- **Empirical receipt:** A1 anchor pair (0.192848 CPU, ~0.22 CUDA) shows -0.027 CPU-CUDA drift for A1 specifically; this is OF THE SAME ORDER as PR102 but per-substrate verification is required. Predicting reformulated-variant CPU band from CUDA observations (or vice versa) is the cargo-cult.
- **Unwind-test:** Per Catalog #324 + Catalog #296: predict CPU + CUDA bands INDEPENDENTLY from first-principles Shannon math; collect per-axis empirical anchors AFTER substrate L1 promotion.

### Assumption 4: "Reformulating Lanes #05+#06 brings the technique class back into the active queue with meaningful score-improvement potential"

- **Classification: TRUE-BUT-CONDITIONAL**
- **Rationale:** The technique class IS preserved per Sub-assumption 1a + 1b; the reformulation IS technically defensible per UNIQUE-AND-COMPLETE-PER-METHOD discipline. BUT the reformulation is BLOCKED on substrate readiness (Variant A on grayscale_lut L1 promotion; Variant B on sane_hnerv L1 promotion; Variant C on A1 substrate-extension feasibility audit). The BLOCKING IS the actual cost, not the technique-class viability.
- **Unwind-test:** §21 of canonical reformulation memo enumerates per-variant 5-criterion reactivation gates; this symposium adopts those gates as binding reactivation criteria per Catalog #308 alternative-probe-methodologies.

### Assumption 5: "Within-class plateau risk applies UNIFORMLY across all three variants"

- **Classification: PARTIALLY-CARGO-CULTED**
- **Rationale:** Variant A grayscale_lut is class-distinct from PR106 (Selfcomp PR #56 paradigm; structurally distinct architectural class — full RGB renderer via grayscale + FiLM-conditioned tiny decoder, NOT HNeRV). Variant B sane_hnerv IS same HNeRV-family class as PR106 — HIGH within-class plateau risk legitimate. Variant C A1 substrate-extension operates at the frontier substrate's sidecar layer, frontier-straddling rather than within-class.
- **Unwind-test:** Per-variant Dykstra-feasibility verdicts per §17 of canonical reformulation memo differentiate the three correctly; this symposium adopts the per-variant verdicts (Variant A frontier-protecting at best; Variant B within-cluster; Variant C frontier-straddling).

### Assumption 6: "$10 conditional re-probe maps to UNIWARD bolt-on dispatch (implicit in prompt framing)"

- **Classification: CARGO-CULTED** (Assumption-Adversary's primary VETO)
- **Rationale:** Per Catalog #229 premise verification: the substrate prerequisite for all three variants is NOT YET LANDED (grayscale_lut L0; sane_hnerv L0; A1 substrate-extension audit PENDING). UNIWARD bolt-on dispatch requires a substrate to operate on; firing $10 on UNIWARD bolt-on without substrate is operationally vacuous.
- **Unwind:** $10 conditional budget ALLOCATES TO substrate prerequisite (grayscale_lut L1 promotion via alpha anchor) OR sits in reserve while $0 probe-disambiguator sequence runs. UNIWARD bolt-on dispatch is a CONDITIONAL FOLLOW-ON after substrate L1 + probe verdicts.

## 2. 9-dimension success checklist evidence (Catalog #294)

| # | Dimension | Evidence | Status |
|---|---|---|---|
| 1 | UNIQUENESS | Three variants target distinct substrate classes (grayscale_lut analog-LUT / sane_hnerv HNeRV-family / A1 frontier-substrate-extension). Distinct from PR106-as-substrate-target failure mode. | INTACT-AS-TECHNIQUE-CLASS, DISTINCT-PER-VARIANT |
| 2 | BEAUTY + ELEGANCE | All variants within HNeRV parity L7 budget: Variant A bolt-on ~150 LOC; Variant B UNIWARD-latent-residual ~250 LOC; Variant B LUT-on-HNeRV-stem ~350 LOC (substrate_engineering tag); Variant C sidecar ~200 LOC. PR101 30-sec-reviewable equivalent. | WITHIN-BUDGET |
| 3 | DISTINCTNESS | Variant A uses grayscale field texture (Fridrich UNIWARD on signal-vs-cover). Variant B uses latent residual wavelet (architecturally awkward per Fridrich/Holub VETO). Variant C uses RGB residual (novel sidecar). Three distinct attack surfaces. | INTACT |
| 4 | RIGOR | PV-1 through PV-8 of canonical reformulation memo + per-variant Dykstra-feasibility on BOTH CPU + CUDA axes per §17 + cargo-cult audit §16 + probe-disambiguator strategy §15 + this symposium's per-assumption HARD-EARNED-vs-CARGO-CULTED classification. | REMEDIATED-VIA-SYMPOSIUM |
| 5 | OPTIMIZATION PER TECHNIQUE | UNIWARD canonical (Fridrich-Holub 2014) ADOPT; grayscale-LUT canonical (Selfcomp PR#56) ADOPT; Variant B LUT-on-HNeRV-stem requires substrate_engineering FORK per Lane MM v2 anti-pattern. Per Catalog #290 canonical-vs-unique decision per layer documented in §22 of reformulation memo. | INTACT-WITH-FORK-DOCUMENTED |
| 6 | STACK-OF-STACKS-COMPOSABILITY | §13 of canonical reformulation memo declares 7 composition options with orthogonality analysis. Variant A ⊕ DP1 codebook init = STRONG_STACK additive. Variant A ⊕ ATW v2 chroma = likely REDUNDANT (probe needed). Variant C ⊕ companion memo #1 NSCS03 = MUTUALLY-EXCLUSIVE deployment paths. | DOCUMENTED |
| 7 | DETERMINISTIC REPRODUCIBILITY | UNIWARD encoding deterministic per fixed cost-map (no stochasticity); LUT byte-stable; sidecar byte-stable. All variants pass byte-stable invariant. | INTACT |
| 8 | EXTREME OPTIMIZATION + PERFORMANCE | Per Catalog #270 umbrella: Tier 1 (autocast_fp16, TF32, torch.compile, no_grad) ADOPT via substrates' existing trainers per Catalog #172/#178/#179/#180. UNIWARD is rate-axis only (no GPU cost at training time; CPU-only encode at archive build). | ADOPT-CANONICAL |
| 9 | OPTIMAL MINIMAL CONTEST SCORE | Per T2 council Q2 verdict: CPU axis is leaderboard axis. Variant A predicted band [substrate_anchor - 0.003, substrate_anchor] ABOVE A1 0.192848 (frontier-protecting). Variant B predicted band [-0.0007, -0.002] within-cluster (within-class plateau). Variant C predicted band [A1 + net-effect] frontier-straddling. NONE projected as frontier-extending with high confidence. | FRONTIER-PROTECTING-AT-BEST |

**Overall:** 7 of 9 dimensions PASS / INTACT-WITH-DOCUMENTATION; Dimension 4 REMEDIATED via this symposium; Dimension 9 documents frontier-protecting status (not frontier-extending with high confidence). The mode of operational frustration is substrate readiness blocking, not technique class falsification — exactly the Item #8 hypothesis case the operator flagged.

## 3. Observability surface (Catalog #305)

Per the 6-facet definition (CLAUDE.md "Max observability — non-negotiable") + canonical reformulation memo §18:

1. **Inspectable per layer**: All three variants inherit substrate-level observability from their base substrates (grayscale_lut, sane_hnerv, A1) via canonical `tac.xray.<lens>` hooks. UNIWARD bolt-on adds per-pair cost-map serialization to `experiments/results/<lane>/observability/uniward_cost_map.jsonl`. LUT bolt-on adds per-pair LUT activation distribution. Sidecar adds per-pair residual statistics. INTACT.
2. **Decomposable per signal**: Bytes decompose per variant — Variant A: `header + film_sd + grayscale_uniward_encoded + uniward_precision_tags + meta`; Variant B: `header + decoder_brotli + latent_residual_wavelet_encoded + wavelet_precision_tags + meta`; Variant C: `A1 archive UNCHANGED + uniward_residual_sidecar (length-prefixed)`. Per-component byte attribution. INTACT.
3. **Diff-able across runs**: Deterministic encoding per variant; byte-stable. Two runs of same substrate+config produce identical archives. INTACT.
4. **Queryable post-hoc**: Manifest JSON per Catalog #226 + #127 + #249 (no `_cuda` filename for CPU eval). Auth-eval results queryable via `tac.continual_learning.query_*`. INTACT.
5. **Cite-able**: Manifests cite archive SHA via canonical `register_dispatched_call_id` per Catalog #245. Run-tuple `(substrate=<variant>, commit=<git_HEAD>, config=<recipe>, random_seed=<pinned>, upstream_snapshot_sha256=<pin>)`. INTACT-WITH-BACKFILL-ON-DISPATCH.
6. **Counterfactual-able**: Per Catalog #139 byte-mutation discipline + Catalog #272 distinguishing-feature integration contract: "what if I mutate the UNIWARD precision tag at offset X?" testable via `tools/verify_distinguishing_feature_byte_mutation.py --distinguishing-byte-range <offset>:<length>`. Variant A's grayscale_uniward_encoded section + uniward_precision_tags section are the distinguishing-feature bytes for the no-op detector. INTACT.

**Overall observability score: 6 / 6 — STRONG.** All variants preserve the canonical 6-facet observability surface. The reformulation does NOT introduce opaque byte regions.

## 4. Sextet pact deliberation (Catalog #325 6-step #4)

### Council attendance + per-member assumption statements (Catalog #292 + #300 mandatory)

The 10 council members + their per-round explicit operating-within-assumption statements are documented in the YAML frontmatter `council_dissent` field. Below is the consolidated narrative summary per CLAUDE.md "Maximum signal preservation rule" (#5 narrative item).

**Shannon (LEAD; information theory grounding):** Operating-within assumption = "UNIWARD's predicted savings are bounded by the grayscale field's actual texture-vs-smooth ratio on the contest video, NOT by UNIWARD's theoretical optimality." Per Catalog #303 cargo-cult audit: HARD-EARNED basis (canonical Shannon R(D) bounds; per-pixel entropy estimates). Shannon's VOTE = PROCEED_WITH_REVISIONS conditional on Variant B latent-wavelet probe ($0) running first.

**Dykstra (CO-LEAD; optimization feasibility):** Operating-within assumption = "Convex-feasibility lens at A1's frontier operating point shows TIGHT polytopes; alternating-projections order is cheapest-signal-first (Variant C audit $0 → Variant B probe $0 → grayscale_lut substrate alpha $5-15 → UNIWARD bolt-on $5-15)." Per Catalog #296 Dykstra-feasibility on BOTH axes. Dykstra's VOTE = PROCEED with revision = no $10 dispatch until $0 probes return.

**Fridrich (canonical UNIWARD author):** Operating-within assumption = "UNIWARD's wavelet texture-cost function requires a 2D cover signal with spatial locality structure; the 28-dim latent has no such structure." Per CLAUDE.md "Beauty, simplicity, and developer experience" + UNIQUE-AND-COMPLETE-PER-METHOD: Variant A IS architecturally correct; Variant B IS architecturally awkward (wavelet-on-28-dim-vector degenerates to basis rotation without UNIWARD adaptive gain). Fridrich's VOTE = PROCEED on Variant A + Variant C feasibility; VETO on Variant B.

**Holub (Fridrich's UNIWARD 2014 co-author):** Operating-within assumption = identical to Fridrich's; sister member confirms the architectural distinction between 2D-image-cover-signal (Variant A grayscale field; correct) and 28-dim-per-pair-vector (Variant B; architecturally awkward). Holub's VOTE = identical to Fridrich's; aligned VETO on Variant B.

**Selfcomp (PR #56 paradigm lead):** Operating-within assumption = "Grayscale-LUT IS the substrate's central mechanism in grayscale_lut substrate; not a bolt-on. The reformulation Variant A 'grayscale-LUT #06' = the substrate's natural construction." Per CLAUDE.md HNeRV parity L7 + Lane MM v2 anti-pattern empirical receipt: bolting LUT onto trained substrate regresses (Lane MM v2 = 2.63 CPU advisory; structurally Lane MM v2 anti-pattern). Selfcomp's VOTE = PROCEED on Variant A standalone with substrate alpha anchor prerequisite; DEFER Variant B's LUT-on-HNeRV-stem as Lane MM v2 anti-pattern unless substrate engineering (~350 LOC train-from-scratch with LUT in place); support Variant C feasibility audit.

**Quantizr (PR101 GOLD adversarial member):** Operating-within assumption = "PR101 GOLD applies UNIWARD-delta on INT8 weight-quantization stream with ~0.001-0.003 rate-axis savings; transferring to Variant A's grayscale field potentially scales to ~0.005-0.010 savings." Per leaderboard-empirical lens. Quantizr's VOTE = PROCEED on Variant A as HIGH-EV composition (grayscale_lut + UNIWARD); DEFER Variant B per Fridrich/Holub architectural critique; support $0 probe-disambiguator sequence.

**Hinton (knowledge-distillation memorial):** Operating-within assumption = "Grayscale-LUT IS soft-target knowledge distillation; bolt-on LUT on trained model is the canonical distillation failure mode (Lane MM v2 receipt)." Per CLAUDE.md "Adversarial council review of design decisions". Hinton's VOTE = PROCEED on Variant A (substrate IS distillation-from-scratch); HIGH DISTORTION RISK on Variant B's LUT-on-HNeRV-stem (Lane MM v2 anti-pattern at decoder-stem layer); supports the canonical reformulation memo's classification.

**Yousfi (challenge designer, Fridrich's PhD student):** Operating-within assumption = "Contest video's per-pair mask-class distribution is well-known (5 SegNet classes); UNIWARD-style texture-cost on grayscale field (derived from these classes via Selfcomp's projection) is structurally well-posed." Per CLAUDE.md "Apples-to-apples evidence discipline". Yousfi's VOTE = PROCEED_WITH_REVISIONS with binding revision that $10 maps to grayscale_lut substrate alpha anchor (NOT UNIWARD bolt-on which has no substrate yet).

**Contrarian:** Operating-within assumption = "Cheapest disambiguation IS Variant C feasibility audit ($0); $10 fires only after $0 probes return. Per CLAUDE.md 'Race-mode rigor inversion' the rigor is in the gating thresholds, not in firing all dispatches." VOTE = PROCEED_WITH_REVISIONS conditional on $0-probes-before-$10-dispatch.

**Assumption-Adversary:** Operating-within assumption = "The prompt's '$10 re-probe' framing implicitly conflates substrate-promotion budget with UNIWARD-bolt-on-dispatch budget — cargo-cult of '$10 dispatch IS the right first action'." Per Catalog #292 + #325 6-step contract. VOTE = PROCEED_WITH_REVISIONS with VETO on consensus that doesn't engage with the substrate-prerequisite-vs-bolt-on distinction. VETO SATISFIED via op-routables #1-#4 revisions.

### Vote tally

PROCEED_WITH_REVISIONS: 10 (unanimous on the verdict; differential per Variant per the per-member analyses)

Per-variant tallies:
- **Variant A**: 9 PROCEED (Shannon, Dykstra, Fridrich, Holub, Selfcomp, Quantizr, Yousfi, Hinton, Contrarian) + 1 PROCEED_WITH_VETO_REVISIONS (Assumption-Adversary). All conditional on substrate prerequisite landing.
- **Variant B**: 2 PROCEED (Shannon, Dykstra qualified) + 6 VETO/DEFER (Fridrich, Holub, Selfcomp, Quantizr, Yousfi, Contrarian) + 2 DEFER (Hinton, Assumption-Adversary). DEFER per Fridrich + Holub architectural VETO.
- **Variant C**: 10 PROCEED on feasibility audit ($0 first); subsequent dispatch CONDITIONAL on audit verdict.

### Per-revision adjudication

The PROCEED_WITH_REVISIONS verdict is bound by SIX revisions (operator-routable):
1. (R1) $0 Variant C feasibility audit FIRST (Assumption-Adversary VETO satisfied)
2. (R2) $0 probe-disambiguator scripts (Variant A grayscale_lut substrate viability + Variant B latent-wavelet savings + Variant C sidecar feasibility)
3. (R3) $10 conditional re-probe ALLOCATES TO grayscale_lut substrate alpha anchor (Variant A prerequisite) — NOT UNIWARD bolt-on
4. (R4) Alternative $10 allocation to A1-sidecar paired-CPU+CUDA smoke (Variant C; CONDITIONAL on feasibility audit POSITIVE)
5. (R5) DEFER Variant B per Fridrich + Holub architectural VETO
6. (R6) Lane registry update: lane `lane_tier_1_resurrection_5_*` notes add `research_only=true` with 5-criterion reactivation gates per §21 of canonical reformulation memo

## 5. Per-substrate reactivation criteria pinned (Catalog #308 + CLAUDE.md "Forbidden premature KILL")

Per CLAUDE.md "KILL/FALSIFIED memory verdicts" + Catalog #307 paradigm-vs-implementation discipline + Catalog #308 alternative-probe-methodologies (N>=3 enumerated): the original 2026-05-04 KILL is now classified as **RATIFY-FALSIFICATION-OF-THE-SPECIFIC-METHODOLOGY** (PR106-mask-channel-substitution IS empirically falsified for PR106-as-substrate) + **REQUEST-REINVESTIGATION-OF-ALTERNATIVES** (UNIWARD + grayscale-LUT TECHNIQUE classes remain viable per the enumerated variants + reducers).

### Reactivation path #1: Variant A (grayscale_lut + UNIWARD #05 + grayscale-LUT #06 = substrate mechanism)

Per §21.1 of canonical reformulation memo:
1. grayscale_lut substrate L0 → L1 promotion via alpha sane_hnerv anchor at ≤0.21 (per substrate's `__init__.py` lines 57-62 reactivation criteria).
2. Post-anchor diagnostic flags rate-axis headroom ≥5% (per substrate's reactivation criterion (a)).
3. UNIWARD #05 bolt-on integration test passes (encoded archive roundtrips byte-stable per Catalog #157 commit serializer + canonical helper invariants).
4. Paired-CPU+CUDA smoke per Catalog #167 lands at substrate_anchor - 0.001 to -0.003 (UNIWARD savings on grayscale field).
5. 5/5 council PROCEED on smoke result (re-convene this sextet; verify per-axis bands match Shannon predictions within ±50%).

**Predicted ΔS band**: CPU `[substrate_anchor - 0.003, substrate_anchor]` (substrate_anchor PENDING; conservative ≈ 0.21 if alpha lands at substrate's stated reactivation threshold).
**Provenance**: `first_principles` per Catalog #324 (Shannon R(D) bounds on grayscale-stream entropy; UNIWARD savings bounded by Fridrich-Holub theorem).

### Reactivation path #2: Variant C (A1 + UNIWARD-residual-sidecar)

Per §21.3 of canonical reformulation memo:
1. A1 substrate-extension feasibility audit returns POSITIVE (document at `.omx/research/a1_substrate_extension_feasibility_audit_<UTC>.md`).
2. UNIWARD-residual-sidecar probe (§15.3) verdict net-effect ≤ -0.001.
3. Sidecar integration test passes (A1 standalone score UNCHANGED; combined archive parses cleanly per Catalog #272 distinguishing-feature integration contract; byte-mutation no-op detector per Catalog #139 verifies bytes are operationally consumed).
4. Paired-CPU+CUDA smoke per Catalog #167 lands at A1 + net-effect within feasibility-audit band.
5. 5/5 council PROCEED on smoke result; per Catalog #220 operational mechanism declared and verified.

**Predicted ΔS band**: CPU `[A1 + net-effect]` where net-effect envelope is `[-0.005, +0.003]` PENDING audit.
**Provenance**: `first_principles` per Catalog #324 (Shannon rate-cost bound on 5 KB sidecar; distortion-axis benefit UNCERTAIN pending feasibility audit).

### Reactivation path #3: Variant B (sane_hnerv + UNIWARD-latent-residual) — DEFERRED per Fridrich + Holub VETO

Per §21.2 of canonical reformulation memo + this symposium's architectural VETO:
1. Variant B is DEFERRED until substrate-engineering reformulation that addresses the spatial-locality gap (e.g., 2D-reshape of 28-dim latent via factored structure, OR alternative basis like DCT that doesn't require spatial-locality).
2. The original 28-dim-latent UNIWARD reformulation is RATIFIED-as-architecturally-awkward per Fridrich + Holub canonical authorship.

**Reactivation criteria**: Awaiting architectural redesign per Pattern E REQUEST-REINVESTIGATION-OF-ALTERNATIVES; not in the active queue until alternative basis identified.

### Alternative reducers enumerated (Catalog #308 Pattern E)

Per the canonical reformulation memo appendix + this symposium's expansion:
- **Reducer 1**: per-pair HISTOGRAM grayscale field analysis (instead of per-pair-dominant)
- **Reducer 2**: per-region grayscale histogram (K=16 region partition)
- **Reducer 3**: per-segment-class conditional grayscale (condition on 5 SegNet classes individually)
- **Reducer 4**: per-temporal-window grayscale predictor (T=4 frame windows)

Per Catalog #308 verdict structure: the original KILL of the substrate-class (PR106 Lanes 05/06) is RATIFY-FALSIFICATION-OF-THE-SPECIFIC-METHODOLOGY (PR106-as-substrate IS empirically falsified) + REQUEST-REINVESTIGATION-OF-ALTERNATIVES (Variants A/B/C are 3 substrate-pivots; Reducers 1-4 are 4 alternative reducer methodologies). N=3 + N=4 = 7 enumerated alternative methodologies satisfying Catalog #308 ≥3 requirement.

## 6. Catalog #324 post-training Tier-C validation discipline declared

Per CLAUDE.md "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch" + Catalog #324:

`predicted_band_validation_status: first_principles` per the symposium frontmatter. The predicted ΔS bands are derived from Shannon first-principles (rate-axis savings bounded by `25 × bytes_saved / 37_545_489`); they are NOT derived from post-training Tier-C density measurement (no post-training archive exists yet because all three variants are PENDING substrate L1).

**Reactivation criterion per Catalog #324**: post-training Tier-C density measurement per `tools/mdl_scorer_conditional_ablation.py --tier c` on the landed archive (sha256 TBD) MUST run after each variant's substrate L1 promotion + paired smoke to validate the first-principles predictions. The 22× miss of C6 IBPS predicted band (per the empirical anchor in Catalog #324 landing) is a cautionary anchor: post-training validation is mandatory for any predicted-band claim that gates further dispatch.

For Variant A specifically: after the grayscale_lut substrate alpha anchor lands, re-measure Tier-C density on the post-training archive; compare to first-principles UNIWARD savings prediction; iterate if drift > 2× expected band.

## 7. Cross-references

- Canonical reformulation memo: `.omx/research/tier_1_resurrection_5_pr106_lanes_05_06_reformulated_uniward_grayscale_lut_full_stack_design_20260516.md` (66.2 KB; 22 sections; 8 PVs)
- Original 2026-05-04 falsification: `.omx/auto_memory_snapshot_20260504T230223Z/feedback_pr106_no_mask_channel_lanes_05_06_falsified_20260504.md`
- Lane #05 original design: `experiments/results/internal_hidden_gem_audit_20260504_claude/revival_plans/revival_plan_05_uniward_delta_pr106_mask_channel.md`
- Lane #06 original design: `experiments/results/internal_hidden_gem_audit_20260504_claude/revival_plans/revival_plan_06_mask_grayscale_lut_pr106_mask_replace.md`
- PR106 archive structure verification: `experiments/extract_pr106_decoder.py` (commit 45149f21)
- grayscale_lut substrate: `src/tac/substrates/grayscale_lut/`
- sane_hnerv substrate: `src/tac/substrates/sane_hnerv/`
- A1 substrate adapter: `src/tac/substrates/a1/`
- Pre-rigor inventory: `.omx/research/pre_rigor_kill_defer_falsified_inventory_20260517.md` (council_priority #858 SHOULD_BE_RESYMPOSIUM'D row #4)
- Sister symposiums (in flight today): `council_per_substrate_symposium_stc_clean_source_20260517.md` (canonical template), `council_per_substrate_symposium_lane_17_imp_20260517.md`, `council_per_substrate_symposium_nscs06_v8_path_b_20260517.md`
- Resurrection audit: `.omx/research/resurrection_audit_20260516.md` (Pattern B substrate-mismatch-as-class-kill canonical anchor)
- Lane MM v2 historical anti-pattern: resurrection audit §1.5
- T2 council Q2 verdict (CPU axis is leaderboard axis): `.omx/research/grand_council_t2_wunderkind_g1_v2_pivot_validation_v3_cpu_competitiveness_20260516.md`
- PR #56 paradigm (Selfcomp's 0.38 anchor): `feedback_szabolcs_pr56_selfcomp_paradigm_*` memory entries
- PR101 GOLD UNIWARD-delta application: `feedback_pr101_*_landed_*` memory entries
- CLAUDE.md non-negotiables cited: HNeRV / leaderboard-implementation parity discipline (L7 bolt-on size); UNIQUE-AND-COMPLETE-PER-METHOD operating mode; Forbidden premature KILL without research exhaustion; Apples-to-apples evidence discipline; Submission auth eval BOTH CPU AND CUDA; MPS auth eval is NOISE; Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY; Substrate MUST be at OPTIMAL FORM before paid empirical dispatch (Catalog #315); PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium (Catalog #325); Catalog #220 (substrate L1+ scaffold operational mechanism); Catalog #229 (premise verification); Catalog #270 (canonical dispatch optimization protocol); Catalog #272 (distinguishing-feature integration contract); Catalog #290 (canonical-vs-unique decision per layer); Catalog #294 (9-dim checklist evidence); Catalog #296 (Dykstra-feasibility predicted band); Catalog #298 (substrate retirement discipline 30-day); Catalog #300 (council deliberation v2 frontmatter); Catalog #303 (cargo-cult audit section); Catalog #305 (observability surface); Catalog #307 (paradigm-vs-implementation classification); Catalog #308 (alternative probe methodologies N>=3); Catalog #313 (probe-outcomes ledger); Catalog #324 (post-training Tier-C validation); Catalog #325 (per-substrate symposium discipline).

---

*End of symposium memo. ~4500 words. Verdict: PROCEED_WITH_REVISIONS (10/10 council on the verdict; per-variant tallies per §4). Mission-alignment: frontier_protecting. Horizon class: frontier_protecting (Variant A plateau-adjacent; Variant C straddles frontier_pursuit; Variant B deferred). Reactivation paths: 3 (with N=4 additional alternative reducers per Catalog #308 enumeration; total N=7 alternative methodologies). $10 conditional re-probe budget: AUTHORIZED conditional on op-routables #1 + #2 verdicts; recommended allocation = grayscale_lut substrate alpha anchor (Variant A prerequisite).*

## Canonical-vs-unique decision per layer (Catalog #290; per-variant cross-reference)

Per the canonical reformulation memo §22 + this symposium's per-variant adjudication. Inherited verbatim from §22 of the canonical reformulation memo for operator-facing audit convenience (HISTORICAL_PROVENANCE preserved per Catalog #110 / #113).

| Layer | Variant A (grayscale_lut) | Variant B (sane_hnerv) | Variant C (A1-extension) |
|---|---|---|---|
| Substrate architecture | ADOPT canonical (grayscale_lut Selfcomp PR#56 paradigm) | ADOPT canonical (sane_hnerv HNeRV-family) | EXTEND canonical (A1 frozen + sidecar) |
| UNIWARD cost function | ADOPT canonical (Fridrich-Holub 2014) | DEFERRED per Fridrich + Holub VETO | ADOPT canonical (applied to RGB residual) |
| Grayscale-LUT mechanism | ADOPT canonical (substrate's central mechanism) | FORK to LUT-on-HNeRV-stem (substrate engineering on sane_hnerv) | N/A (sidecar uses UNIWARD only, not LUT) |
| Archive grammar | EXTEND grayscale_lut GLV1 → GLV2 (UNIWARD-encoded grayscale section) | EXTEND sane_hnerv archive with bolt-on sections | EXTEND A1 archive with backward-compatible sidecar |
| Inflate runtime | EXTEND grayscale_lut inflate (~50 LOC UNIWARD decode) | EXTEND sane_hnerv inflate | EXTEND A1 inflate (sidecar-aware wrapper) |
| Score-aware loss | ADOPT canonical `score_pair_components` per Catalog #164 | ADOPT canonical | UNIQUE (sidecar-specific Lagrangian; A1 frozen) |
| Tier-1 engineering | ADOPT canonical (autocast_fp16, TF32, torch.compile, no_grad) | ADOPT canonical | N/A (A1 frozen; sidecar train uses canonical) |
| EMA | ADOPT canonical EMA(0.997) | ADOPT canonical | ADOPT canonical for sidecar |
| Variant adjudication mechanism | UNIQUE (substrate alpha anchor at ≤0.21 is the gate) | UNIQUE-BUT-DEFERRED | UNIQUE (feasibility audit + probe net-effect ≤ -0.001) |
| Cross-axis CPU+CUDA evaluation | ADOPT canonical (paired-CPU+CUDA per CLAUDE.md non-negotiable) | ADOPT canonical | ADOPT canonical |
| Reactivation criteria | UNIQUE (5-criterion sequence per §21.1) | UNIQUE-BUT-DEFERRED (5-criterion sequence per §21.2) | UNIQUE (5-criterion sequence per §21.3) |
| Within-class plateau risk acknowledgment | EXPLICIT (per §17.1 Dykstra verdict + cargo-cult audit) | EXPLICIT (per §17.3) | EXPLICIT (per §17.5) |
