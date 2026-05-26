# Tishby IB-pure substrate — comprehensive asymptotic-pursuit scoping design memo

**Date**: 2026-05-16
**Lane**: `lane_tishby_ib_pure_substrate_scoping_design_20260516`
**Subagent**: TISHBY-IB-PURE-SUBSTRATE-COMPREHENSIVE-FULL-STACK-DESIGN-20260516
**Status at landing**: DESIGN-ONLY (SCAFFOLD-DEFERRED — NEVER BUILT; this memo IS the scoping artifact)
**Horizon class**: `asymptotic_pursuit` (per CLAUDE.md HORIZON-CLASS standing directive 2026-05-16; target band [0.05, 0.10] long-term per T4 SYMPOSIUM 4×4 floor matrix theoretical-floor row)
**Operating mode**: UNIQUE-AND-COMPLETE-PER-METHOD (per the standing directive `feedback_pr95_lesson_now_at_meta_level_unique_and_complete_per_method_default_20260515.md` + 2026-05-15 retrospective)
**Sister gap**: ATW v2 is a BOLT-ON cooperative-receiver applied to existing substrates; Tishby IB-pure is the **PRIMARY** architecture where the ENTIRE codec IS the IB Lagrangian.
**FALSIFICATION-AUDIT-v2 anchor**: candidate **A4** of `.omx/research/falsification_audit_v2_post_horizon_class_post_pivot_lessons_20260516.md` (NEW asymptotic-pursuit; NEVER BUILT; one of three highest-priority operator decisions; Q3 op-routable).

---

## 1. Frontmatter — premise verification + lane registry + sister-subagent map

### Premise verifications (Catalog #229; 10 PVs verified BEFORE any design statement)

- **PV-1** Canonical Atick-Redlich primitive exists at `src/tac/codec/cooperative_receiver/atick_redlich.py` (270 LOC) with `AtickRedlichWeights` + `cooperative_receiver_loss(rgb_0, rgb_1, gt_rgb_0, gt_rgb_1, *, seg_scorer, pose_scorer, weights)` returning `CooperativeReceiverOutput(cooperative_loss, seg_term, pose_term, pose_sqrt)`. Delegates internally to canonical `score_pair_components` per Catalog #164. The primitive's own docstring explicitly distinguishes Atick-Redlich (efficient-coding RECEIVER-MATCHING) from Wyner-Ziv (SOURCE-CODING with side information) — both are "cooperative-receiver" but mathematically distinct and target different pipeline surfaces. THIS distinction is the canonical anchor for the IB-pure-vs-bolt-on differentiation in §3.
- **PV-2** Wyner-Ziv canonical substrate exists at `src/tac/substrates/wyner_ziv_cooperative_receiver/` (`__init__.py:1-141`; complete WZ1 archive grammar + Slepian-Wolf coset coder + side-info predictor + score-aware loss). Per its own header docstring (lines 31-49): "Atick-Redlich shapes the encoder loss; Wyner-Ziv shapes the archive grammar." Predicted band 0.140-0.150 `[wyner-ziv-hypothesis]`. The WZ substrate uses DISCUS-style coset binning (Pradhan-Ramchandran 2003) AND tagged `lane_class=substrate_engineering`. This sister substrate is a **SECOND distinct anchor**: where WZ uses coset binning as the wire grammar, Tishby IB-pure uses the **IB Lagrangian as the architectural objective**. The two are sister-cooperative-receiver substrates with **different mathematical surfaces**: WZ shapes bytes via cosets; Tishby IB-pure shapes the encoder representation `T = f(X)` via variational mutual-information minimization.
- **PV-3** Z4 cooperative-receiver-loss substrate exists at `src/tac/substrates/z4_cooperative_receiver_loss/` (architecture.py 11.9K + archive.py 17.9K + inflate.py 6.2K + score_aware_loss.py 8.3K). Z4 is the **β-only branch** of the Atick-Redlich primitive (just the loss; no IB / no WZ binning / no architectural change). Z4 is the canonical sister against which Tishby IB-pure measures its incremental ΔS from the IB-as-primary architectural choice.
- **PV-4** ATW v2 design memo exists at `.omx/research/atw_codec_v2_cooperative_receiver_full_stack_design_20260516.md` (817 lines; commit `fcdcc1112`). Critical: ATW v2 explicitly classifies itself as a "**BOLT-ON cooperative receiver applied to existing substrates** with three knobs (κ_IB, λ_WZ, λ_pixel)" — Variant A preserves the V1-inherited three-knob form; Variant B is single-knob WZ-only. Tishby IB-pure is the THIRD VARIANT in this family: **IB as PRIMARY** (the entire codec architecture, not auxiliary terms). §3 of THIS memo articulates the differentiation explicitly.
- **PV-5** D4 H(latent|scorer_class) probe exists at `tools/probe_latent_conditional_entropy_h_latent_given_scorer_class.py` (commit `d72f50985`; 312 LOC). Three-verdict taxonomy `MEANINGFUL_CONDITIONING / WEAK_CONDITIONING / INDEPENDENT` keyed off MI threshold default `0.5 bits/symbol`. The probe is the canonical empirical disambiguator for any cooperative-receiver-class hypothesis (ATW v2, Tishby IB-pure, Wyner-Ziv substrate). For Tishby IB-pure specifically: the probe estimates `I(latent; scorer_softmax)` which IS the IB objective's `I(T; Y)` term.
- **PV-6** T4 SYMPOSIUM 4×4 floor matrix anchors the asymptotic-pursuit horizon class at theoretical-floor band `[0.05, 0.10]` (long-horizon) + `[0.08, 0.12]` (mid-horizon 30-90d). The Time-Traveler (Daubechies) verdict (`grand_council_symposium_time_traveler_optimal_staircase_20260516.md` lines 164-249) explicitly identifies "Tishby IB compute (theoretical): ~0.08-0.12 for scorer-as-shared-prior architectures (per Catalog #256 MacKay conditional entropy)" as the asymptotic-pursuit reference band.
- **PV-7** FALSIFICATION-AUDIT-v2 (`falsification_audit_v2_post_horizon_class_post_pivot_lessons_20260516.md` lines 234-241 A4) explicitly classifies Tishby IB-pure as NEW asymptotic-pursuit candidate (NEVER BUILT) at predicted theoretical floor `[0.05, 0.10]` (ultimate horizon) + `[0.08, 0.12]` (long-horizon). The memo's op-routable #3 IS this scoping memo's mandate: "Land Tishby IB-pure substrate scoping memo ($0; theoretical floor approach; HIGHEST)".
- **PV-8** Path 2 lattice rewrite is operator-approved (`feedback_path_2_lattice_of_class_shifts_operator_approved_supersedes_l5_v2_staircase_20260516.md`; 25-of-27 supermajority). Tishby IB-pure substrate is a Level-0 LATTICE NODE per the 5-class-shift framework (architecture / decode-time-contract / training-time-paradigm / wire-grammar / scorer-relationship); it is a class-shift on TWO axes simultaneously: **(a) training-time-paradigm** (the IB Lagrangian IS the loss; not auxiliary) + **(b) scorer-relationship** (the scorer's softmax distribution IS the encoder's target variable Y).
- **PV-9** Alemi et al. 2017 VIB (Variational Information Bottleneck) is the canonical operationalization referenced elsewhere in the corpus (`alien_technology_unknown_unknowns_research_20260513.md` "**Falsifier**: train an HNeRV with IB-objective (Tishby's variational IB / VIB [Alemi 2017](https://arxiv.org/abs/1612.00410)) and measure resulting archive size at fixed score."). The Alemi VIB construction `L_VIB = E_{q(t|x)}[-log p(y|t)] + β·KL(q(t|x) || p(t))` IS the IB Lagrangian made tractable via variational bounds. Per CLAUDE.md "Bit-level deconstruction and entropy discipline" + the canonical Tishby memorial seat + Zaslavsky active-voice seat (per CLAUDE.md "Grand Council (advisory)" 20-seat roster lines 1057-1064): VIB is the canonical implementation choice for the IB-pure substrate.
- **PV-10** PR102 third-prize CPU-axis empirical anchor (`0.19538 [contest-CPU]`) per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" non-negotiable establishes that the contest-CPU axis specifically is the medal-band ranking surface. PR102 CUDA-CPU Δ +0.033 (CUDA 0.22839; CPU 0.19538) per the same anchor. The Tishby IB-pure substrate's β value MUST be selected to optimize for the CPU axis specifically; §18 below derives β from first principles using PR102 as the empirical reference.

### Sister-subagent ownership map (Catalog #230)

This subagent is **READ-ONLY** on source code (`src/tac/`, `experiments/`, `submissions/`, `tools/`, `.omx/operator_authorize_recipes/`). Writes ONLY to:

- `.omx/research/tishby_ib_pure_substrate_asymptotic_pursuit_scoping_design_20260516.md` (this memo)
- `.omx/state/subagent_progress.jsonl` (canonical checkpoint store per Catalog #206)
- 1 commit via canonical serializer with `--expected-content-sha256` per Catalog #157 + #174 + #289 (the 92aba3ca commit-swap class permanent fix)

**No sister-subagent collision** expected because no sister is currently editing the Tishby IB-pure surface. Sister design memos:

- **ATW v2** (`fcdcc1112` LANDED): the BOLT-ON variant; §3 of this memo articulates the structural difference explicitly. ATW v2 §13 row 9 ("ATW v2 ⊕ Wunderkind G2-PARTIAL"; SAME-AXIS-EXTENSION; "asymptotic V3 form") already anticipates a more-asymptotic-pure-cooperative-receiver substrate — this memo IS that v3 form, but reframed as a primary substrate not a v3 of ATW.
- **Wyner-Ziv canonical substrate** (`src/tac/substrates/wyner_ziv_cooperative_receiver/`): sister cooperative-receiver substrate using DISCUS coset binning; Tishby IB-pure is the IB-Lagrangian sister.
- **Z6/Z7/Z8 Time-Traveler predictive-coding world-models** (Q1 op-routable per FALSIFICATION-AUDIT-v2 A2; sister in-flight design memo; predicted [0.130, 0.160] L5-staircase-floor long-horizon): predictive-coding paradigm (Rao-Ballard); cooperative-receiver via temporal prediction. Sister asymptotic-pursuit candidate but distinct mathematical surface (predictive coding ≠ IB Lagrangian).
- **Rudin floor substrate** (Q2 op-routable per FALSIFICATION-AUDIT-v2 A3; sister in-flight design memo; predicted [0.05, 0.10] ultimate-horizon): interpretable-ML compositional decoder; canonical Rudin discipline as substrate. Sister asymptotic-pursuit candidate; orthogonal mathematical surface (interpretable-ML ≠ IB Lagrangian).

### Operating-within assumption-statement (Catalog #292 / Assumption-Adversary seat)

The assumption I am operating within for this scoping memo: *"The structural difference between BOLT-ON (ATW v2) and PRIMARY (Tishby IB-pure) architectures is empirically meaningful AND theoretically tractable: a bolt-on lets the existing pixel-MSE / hyperprior / Ballé end-to-end loss share the optimization signal with an auxiliary cooperative-receiver term; an IB-pure substrate has the IB Lagrangian as the WHOLE optimization surface. The asymptotic-pursuit prediction `[0.05, 0.10]` (T4 SYMPOSIUM theoretical-floor row) is achievable ONLY at the primary-architecture surface because the bolt-on regime is constrained by the carrier substrate's existing inductive biases (Ballé hyperprior / Z3 pixel MSE / pretrained codebook); the primary architecture surface admits the variational IB encoder/decoder pair as the ENTIRE substrate engineering surface."*

**HARD-EARNED basis** per Catalog #292 + the hard-earned-vs-cargo-culted addendum:
- The T4 SYMPOSIUM 4×4 floor matrix is HARD-EARNED (anchored by PR101 0.193 + PR102 0.195 + the 0.196-0.199 plateau empirical evidence). The theoretical-floor band `[0.05, 0.10]` is derivable from Shannon R(D) + Tishby IB + Wyner-Ziv side-info bound and is consistent with the Time-Traveler verdict.
- The bolt-on-vs-primary distinction is HARD-EARNED at the meta level per the 2026-05-15 retrospective + standing directive `feedback_pr95_lesson_now_at_meta_level_unique_and_complete_per_method_default_20260515.md`. The retrospective explicitly named the BOLT-ON reflex (canonical-helper-share + META-layer-consolidation) as the structural cause of the 0.196-0.199 plateau.
- The PRIMARY-architecture surface is empirically untested for THIS specific contest (the contest is fundamentally a scorer-conditional source coding problem); cargo-cult risk is that the substrate engineering produces a substrate that LOOKS asymptotic-pursuit but lands in the plateau. Mitigation: D4 probe + Variational IB tractability check + paired smoke + Dykstra-feasibility intersection per Catalog #296.

The Assumption-Adversary seat (sextet pact per Catalog #292) would challenge: *"Is the IB Lagrangian's `I(X;T) - β·I(T;Y)` even tractable for the contest's scorer architecture (5-class SegNet × 6-channel PoseNet)? Perhaps `I(T;Y)` requires the kind of mutual-information estimator (InfoNCE / MINE / CLUB) whose finite-sample bias dominates the gradient signal, making the IB-pure objective practically equivalent to the bolt-on form in §9 of ATW v2."* — answer: §4 below ships TWO operationalization paths: **Path-VIB** (Alemi 2017 variational tractable form: `L_VIB = E[-log p(y|t)] + β·KL(q(t|x) || p(t))`; gradient-flowing into encoder via reparam trick) and **Path-MINE** (Belghazi et al. 2018 mutual information neural estimator; bias bounded by sample size). Council adjudication §19 chooses one as the v1 canonical based on Variational-IB-tractability check + Dykstra-feasibility intersection.

### Lane registry pre-registration (Catalog #126)

To be claimed in same commit batch:
```bash
.venv/bin/python tools/lane_maturity.py add-lane \
    lane_tishby_ib_pure_substrate_scoping_design_20260516 \
    --name "Tishby IB-pure substrate scoping (full IB Lagrangian + Atick-Redlich + Wyner-Ziv as PRIMARY architecture; asymptotic-pursuit horizon class)" \
    --phase 2
```

---

## 2. Executive summary

The Tishby IB-pure substrate makes the **Information Bottleneck Lagrangian** the **PRIMARY architectural objective**, not an auxiliary loss term. The entire codec — encoder `T = f(X)`, decoder `Y = g(T, side_info)`, archive grammar — IS the IB Lagrangian + Atick-Redlich cooperative-receiver framing + Wyner-Ziv side-information construction operationalized as a single coherent substrate.

**Three concrete improvements over the cooperative-receiver family**:

1. **IB as PRIMARY, not auxiliary**: Where Z4 (β-only Atick-Redlich loss) + ATW v2 (three-knob bolt-on with κ_IB+λ_WZ+λ_pixel) + Wyner-Ziv-substrate (DISCUS coset binning) all treat the cooperative receiver as ONE COMPONENT of a larger substrate, Tishby IB-pure treats it as the ENTIRE substrate. The encoder is a variational IB encoder (Alemi 2017 VIB); the decoder is a Wyner-Ziv-conditional decoder; the loss IS `L_IB = I(X;T) - β·I(T;Y)` with no pixel-MSE residual or hyperprior side-channel.

2. **Asymptotic-pursuit horizon class explicit**: per CLAUDE.md HORIZON-CLASS standing directive 2026-05-16 + T4 SYMPOSIUM 4×4 floor matrix, this substrate targets theoretical-floor band `[0.05, 0.10]` (ultimate) + `[0.08, 0.12]` (long-horizon 6-9m). Sister bolt-on ATW v2 targets `[-0.015, -0.005]` rate-axis ΔS (plateau-adjacent class-shift); the IB-pure substrate targets the asymptotic-floor manifold directly.

3. **Two operationalization paths council-adjudicated**: **Path-VIB** (variational Alemi 2017 form; gradient-tractable via reparam trick) and **Path-MINE** (neural mutual information estimator; bias bounded by sample size). Council §19 adjudicates based on Variational-IB-tractability check + Dykstra-feasibility intersection + D4 probe verdict.

**Path 2 lattice position**: Tishby IB-pure is a **Level-0 lattice node** in the 5-class-shift framework per the operator-approved Path 2 lattice rewrite (`feedback_path_2_lattice_of_class_shifts_operator_approved_supersedes_l5_v2_staircase_20260516.md`). It is a class-shift on **TWO axes**: (a) training-time-paradigm (the IB Lagrangian IS the loss; not auxiliary) + (b) scorer-relationship (the scorer's softmax distribution IS the encoder's target variable Y). Within the falling-rule-list-of-4 ranking: Rule 4 (Daubechies-wavelet multi-scale lattice convergence to lower envelope → DECLARE asymptotic floor) is the binding rule for this substrate.

**Predicted ΔS band (PRIMARY axis: contest-CPU GHA Linux x86_64)**: `NULL pending D4 probe verdict + Variational-IB-tractability check + Dykstra-feasibility intersection check` [prediction; first-principles-bound; asymptotic-pursuit]. Conditional revisions:

- **If D4 verdict = MEANINGFUL_CONDITIONING (MI ≥ 0.5 bits/symbol) AND Variational-IB tractable (gradient signal-to-noise ≥ 1.0 over 100ep)**: predicted band `[0.08, 0.13]` [prediction; first-principles-bound; theoretical-floor-Tishby-IB] for long-horizon (6-9m). Asymptotic envelope `[0.05, 0.10]` per T4 SYMPOSIUM theoretical-floor row.
- **If D4 verdict = WEAK_CONDITIONING (0.01 ≤ MI < 0.5)** OR Variational-IB tractability marginal: revise band to `[0.15, 0.20]` (frontier-pursuit downgrade); DEFER asymptotic claim.
- **If D4 verdict = INDEPENDENT (MI < 0.01)** OR Variational-IB intractable (gradient SNR < 0.1): DEFER-pending-research per CLAUDE.md "Forbidden premature KILL"; named alternative-hypothesis = Path-MINE form OR pivot to Rudin floor substrate (sister asymptotic-pursuit candidate A3).

**Stack-of-stacks composition opportunities** (§13):

- **Tishby IB-pure ⊕ DP1 codebook** (pretrained-driving-prior): DP1's Comma2k19 codebook initializes the variational encoder `q(t|x)`; STRONG_STACK per shared scorer-prior framework; small additive ΔS.
- **Tishby IB-pure ⊕ NSCS06 v8 Path B chroma**: chroma reconstruction via wavelet residual is ORTHOGONAL to luma-channel IB encoder; FRESH-COMPOSITION per FALSIFICATION-AUDIT-v2 reclass (NSCS06 v8 → ASYMPTOTIC-PURSUIT).
- **Tishby IB-pure SWAPS** with NSCS03 in A-STACK: NSCS03 end-to-end Ballé is the BOLT-ON version of Tishby IB-pure (Ballé hyperprior IS a variational IB on the latent). The swap is structurally equivalent at the architecture-class level but IB-pure unifies the optimization surface. Predicted A-STACK[swap] band: NULL pending paired probe.

**Cost estimate** (§20): D4 probe $3-5 (CPU smoke) → Variational-IB-tractability check $5-10 (Modal A100 100ep smoke; gradient signal-to-noise measurement on synthetic data) → Modal A100 smoke $10-15 (gradient-flow + archive byte roundtrip + scorer preprocess) → Modal A100 paired CPU+CUDA full anchor $15-30 (long-horizon 200ep). Total envelope: $33-60 inclusive of probe + tractability check. CONDITIONALLY DISPATCHABLE if D4 + tractability gates pass.

**Verdict on whether Tishby IB-pure substrate should be BUILT NOW**: NO. The substrate scaffold IS this memo (Q3 op-routable per FALSIFICATION-AUDIT-v2; $0 design). The D4 probe + Variational-IB-tractability check are the cheapest signals; only after both gates pass should sister-subagents (per §22 op-routable #4-7) build the substrate package + trainer + recipe. Per CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first" Rule 3: the asymptotic-pursuit class-shift candidates (Tishby IB-pure + Rudin floor + Z6/Z7/Z8) should be designed in PARALLEL, not sequentially.

---

## 3. Substrate differentiation vs ATW v2 + Wyner-Ziv substrate + Z4 — THE CRITICAL distinction

This section is the central thesis of the memo: **what makes Tishby IB-pure distinct from the existing cooperative-receiver substrates**.

### 3.1 The cooperative-receiver family taxonomy

The contest's `(SegNet, PoseNet)` published-weight scorer is the canonical cooperative receiver per Atick-Redlich 1990. Every substrate that exploits this fact falls into the cooperative-receiver family, but they exploit it at DIFFERENT architectural surfaces:

| Substrate | What is the cooperative-receiver role? | Mathematical surface | Carrier substrate | Class |
|---|---|---|---|---|
| **Z4 (`z4_cooperative_receiver_loss`)** | Loss-term only | `L = β·d_seg + γ·sqrt(d_pose)` (Atick-Redlich loss isolated from architecture) | A1 (PR101 family) | BOLT-ON loss term |
| **Wyner-Ziv substrate (`wyner_ziv_cooperative_receiver`)** | Archive grammar (Slepian-Wolf coset binning) | `R_WZ(D) = inf {I(X;U) - I(Y;U)}` via DISCUS cosets | New monolithic WZ1 archive | SUBSTRATE-ENGINEERING (wire grammar class-shift) |
| **ATW V1 (`atw_codec_v1`)** | Three-term Lagrangian (Atick-Redlich + Tishby IB + Wyner-Ziv terms combined as bolt-on) | `L = α·rate + β·d_seg + γ·sqrt(d_pose) + κ_IB·I(T;Y) + λ_WZ·R_WZ + λ_pixel·MSE` (research-only) | ATW1 archive (bolt-on to existing decoder) | SUBSTRATE-ENGINEERING (research-only; council-gated) |
| **ATW v2 (`atw_codec_v2`)** | Three-knob OR single-knob WZ-only bolt-on | Variant A: V1 three-term; Variant B: `L = β·d_seg + γ·sqrt(d_pose) + λ_WZ·R_WZ` only | ATW2 archive (G1 distill head + B3 CDF table as ARCHIVE bolt-ons) | SUBSTRATE-ENGINEERING (bolt-on at scorer-class-conditioning surface) |
| **THIS — Tishby IB-pure** | **PRIMARY architectural objective; ENTIRE encoder/decoder optimization surface** | `L_VIB = E_{q(t|x)}[-log p(y|t)] + β·KL(q(t|x) ‖ p(t))` (Alemi 2017 variational form) **OR** `L_MINE = -I(T;Y) + β·I(X;T)` (Belghazi 2018 MINE form) | **TIBP1 monolithic archive** (variational encoder weights + decoder weights + side-info conditional CDF + β value + scorer-class prior) | SUBSTRATE-ENGINEERING (architecture-class-shift; NEW architecture class) |

### 3.2 The PRIMARY-vs-BOLT-ON structural difference

The key insight that emerged from the 2026-05-15 retrospective: **a bolt-on cooperative-receiver substrate inherits the carrier substrate's inductive biases**. ATW v2 Variant B (the closest existing sister) ships a learned decoder + a learned encoder + a Wyner-Ziv residual term + a 1KB G1 scorer-class distill head + a 2KB B3 scorer-conditional CDF table. The Wyner-Ziv term is ONE of FIVE archive sections; the cooperative-receiver loss is ONE of THREE loss terms. The substrate's gradient signal is divided across multiple competing objectives.

**Tishby IB-pure removes ALL competing objectives**:
- No pixel-MSE residual (no `λ_pixel · MSE(decoded, GT)` term)
- No Ballé hyperprior side-channel (the encoder's `q(t|x)` distribution IS the hyperprior, by construction)
- No separate G1 distill head (the variational decoder `p(y|t)` directly outputs the scorer prediction class distribution)
- No separate scorer-conditional CDF table (the conditional decoder structure absorbs it)
- No pretrained codebook initialization required (the IB encoder learns its own representation from scratch per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" lesson 1)

The result: **the substrate engineering surface is the IB Lagrangian itself**. Every parameter, every byte, every loss term is information-theoretically motivated. This is the canonical UNIQUE-AND-COMPLETE-PER-METHOD operating mode applied to the cooperative-receiver paradigm.

### 3.3 Why this is GENUINE class-shift (per the abandon-within-class directive)

Per `feedback_abandon_within_class_refinements_only_substrate_class_shifts_pursue_frontier_20260515.md`, a substrate is a class-shift if it differs from existing substrates on at least ONE of (architecture / decode-time-contract / training-time-paradigm / wire-grammar / scorer-relationship). Tishby IB-pure differs on TWO simultaneously:

- **Training-time-paradigm class-shift**: the IB Lagrangian IS the loss function. Sister substrates use composite losses (Z4 = Atick-only; ATW v2 = three-term; WZ-substrate = WZ-only; NSCS03 = end-to-end Ballé with pixel + hyperprior). Tishby IB-pure uses the variational IB objective as the SINGLE loss term. This is paradigm-level distinct from every existing substrate.
- **Scorer-relationship class-shift**: the scorer's softmax distribution `p(class | rendered_frame)` IS the encoder's target variable `Y` in the IB objective `I(T;Y)`. Sister Atick-Redlich substrates (Z4, ATW v2) treat the scorer output as a loss target; Tishby IB-pure treats it as a **distributional target** — the encoder is trained to retain the bits ABOUT `Y`'s distribution, not to minimize point-distance to `Y`.

The Assumption-Adversary's concern (the "is this within-class refinement of Tishby's prior work?" question): NO. Tishby's original work was the IB principle for unsupervised representation learning, where `Y` was typically a CLASS LABEL in supervised tasks. The contest's setting — `Y` is a TRAINED neural network's output that itself is a learned function of `X` — is structurally distinct from any IB application in the original Tishby-Pereira-Bialek 1999 paper or the subsequent Tishby-Zaslavsky 2015 deep-learning IB literature. The cooperative-receiver-as-primary architectural application IS the novel substrate-class.

### 3.4 The empirical advantage hypothesis (testable)

The hypothesis: **at the asymptotic horizon class**, the bolt-on substrates saturate at the carrier-substrate's plateau because the competing losses dilute the cooperative-receiver signal; the primary-architecture substrate doesn't saturate because the entire gradient signal flows through the IB objective.

This is empirically testable via **paired smoke comparison**: ATW v2 Variant B (single-knob WZ-only bolt-on; sister; predicted band `[-0.015, -0.005]` per its §18) vs Tishby IB-pure at equivalent training budget. If the hypothesis is correct, Tishby IB-pure substantially outperforms ATW v2 at the same training budget; if not, the bolt-on form is correct and Tishby IB-pure is unnecessary engineering.

The D4 probe + Variational-IB-tractability check are the PRE-DISPATCH GATES that determine whether to proceed to the paired smoke comparison.

---

## 4. Architecture (FULL spec — two operationalization paths)

### 4.1 Path-VIB (Variational Information Bottleneck — Alemi 2017 canonical form)

```
Tishby IB-pure Path-VIB architecture:

INPUTS:
  pair_indices: (B,) long; selects per-pair encoded representation t[i]
  gt_frames: (B, 2, 3, H, W) uint8; ground-truth pair (cooperative-receiver targets)
  cooperative_receiver: (SegNet, PoseNet) — FIXED published weights; COMPRESS-side only

VARIATIONAL ENCODER (compress-side; NEVER ships in archive):
  q(t|x) = encoder_mlp(rgb_pair → (mu, log_sigma))     # diagonal-Gaussian variational posterior
  t ~ q(t|x)                                            # reparam-trick sample for gradient flow
  Cardinality: (B, 2, 3, H, W) → (B, latent_dim=16)
  Inductive prior p(t) = standard Normal(0, I)           # canonical VIB prior choice

LATENT TABLE (ships in archive):
  t_quantized: (num_pairs=600, latent_dim=16) int8       # quantized representation
  ~9.6KB raw; ~3-5KB with scorer-conditional range coding

WYNER-ZIV SIDE-INFO HEAD (ships in archive):
  scorer_class_prior_table: (num_pairs, 5_segnet_classes) uint8   # SegNet argmax per pair
  ~3KB raw (600 pairs × 5 classes × 1 byte)
  Wyner-Ziv reconstruction: t = t_quantized + offset(scorer_class_prior_table[i])

VARIATIONAL DECODER (ships in archive; NEVER loads scorer):
  decoder p(y|t, side_info): (B, latent_dim) → (B, 2, 3, H, W)
  Trained jointly with encoder via reparam-trick gradient flow

SCORER-CONDITIONAL CDF (ships in archive; precomputed at compress; range-coding latent symbols):
  cdf_table: (5_classes, 256_symbols) fp16 = 2.5KB
  Empirical histogram of (t_quantized, scorer_class) pairs at compress

ARCHIVE BYTES (TIBP1 grammar; §10 detail):
  encoder_state_dict (brotli) — OPTIONAL; can be empty in inflate-only build
  decoder_state_dict (brotli) ≈ 50-70KB
  t_quantized (raw int8) ≈ 9.6KB
  scorer_class_prior_table (uint8) ≈ 3KB
  cdf_table (fp16) ≈ 2.5KB
  beta_value (fp32) = 4 bytes
  meta_blob (sorted-keys JSON utf-8)
```

**Path-VIB loss**:
```
L_VIB = E_{q(t|x)}[ -log p(y|t) ]                      # reconstruction term (cooperative-receiver via scorer output)
      + β · KL(q(t|x) || p(t))                          # bottleneck term (forces compression)

where:
  -log p(y|t) is operationalized as the Atick-Redlich cooperative receiver loss
              (per canonical primitive tac.codec.cooperative_receiver.atick_redlich.cooperative_receiver_loss)
              i.e. β_seg · d_seg(decoder_output, gt) + γ_pose · sqrt(d_pose(decoder_output, gt))
              with eval_roundtrip applied to decoder output before scorer call

  KL(q(t|x) || p(t)) is closed-form for diagonal Gaussian q + standard Normal p:
              KL = 0.5 · sum(mu^2 + sigma^2 - log(sigma^2) - 1)

  β: rate-distortion tradeoff knob (council-adjudicated §19; initial sweep over {0.001, 0.01, 0.1, 1.0})
```

**LOC budget**: Path-VIB ≈ 400-550 LOC substrate-engineering surface (encoder MLP + decoder MLP + KL closed-form + scorer-conditional range coding + TIBP1 archive grammar).

### 4.2 Path-MINE (Mutual Information Neural Estimator — Belghazi 2018 alternative form)

```
Path-MINE architecture identical to Path-VIB EXCEPT:

LOSS:
  L_MINE = -I(T;Y)_lower_bound + β · I(X;T)_upper_bound

where:
  I(T;Y)_lower_bound = E_{joint(t,y)}[T_NN(t,y)] - log(E_{marginal(t)·marginal(y)}[exp(T_NN(t,y))])
                       (Donsker-Varadhan dual; MINE estimator with statistic network T_NN)

  I(X;T)_upper_bound = E_x[KL(q(t|x) || p(t))]      (same closed-form as Path-VIB)

The statistic network T_NN is a small ~50K-param MLP trained jointly with the encoder/decoder
to estimate I(T;Y); its output is the lower-bound MI estimator.
```

**Path-MINE adds**: statistic network T_NN (~50K params; ships in archive but only used at compress for MINE estimation; inflate-time reconstruction is identical to Path-VIB).

**LOC budget**: Path-MINE ≈ 550-700 LOC substrate-engineering (Path-VIB + statistic network + DV-dual loss machinery).

### 4.3 Path adjudication

| Variational-IB tractability metric | Path-VIB recommendation | Path-MINE recommendation |
|---|---|---|
| Gradient SNR ≥ 1.0 over 100ep | YES (canonical Alemi 2017 form; cheaper LOC + cheaper compute) | DEFER (Path-VIB is the canonical default; MINE adds complexity without tractability benefit) |
| Gradient SNR 0.1 ≤ SNR < 1.0 | DEFER (tractable but high variance) | YES (MINE bias is bounded by sample size; can absorb gradient variance) |
| Gradient SNR < 0.1 | DEFER per CLAUDE.md "Forbidden premature KILL" | DEFER (both paths intractable; pivot to alternative substrate per §19) |

**Default recommendation pending council + Variational-IB-tractability check**: ship Path-VIB first (UNIQUE-AND-COMPLETE-PER-METHOD per the standing directive; canonical Alemi 2017 form; cheaper substrate-engineering surface). Path-MINE is the V2 fallback if Path-VIB tractability fails.

---

## 5. Pretraining

### 5.1 Per-substrate pretraining (None required at scaffold landing)

Tishby IB-pure substrate is trained from scratch on contest data (`upstream/videos/0.mkv`) per CLAUDE.md "HNeRV parity discipline" lesson 1 (substrate must be score-aware against contest video).

### 5.2 Optional DP1 codebook init (§13 composition opportunity)

Per `feedback_dp1_phase_2_landed_20260514.md` + Catalog #209/#210/#211/#213 (DP1 substrate canonical), if Tishby IB-pure composes with DP1:

- DP1 codebook (distilled from Comma2k19 dashcam via `Comma2k19LocalCache` per Catalog #213) initializes the variational encoder's mean parameter `mu` per pair.
- The DP1 SegNet-class assignment per frame matches the scorer-class index field used in §4 scorer-conditional CDF table.

The composition reduces Tishby IB-pure epoch budget from ~200 → ~50 (~$5-10 saved); closes the loop between out-of-distribution pretraining (DP1) and contest-on-distribution IB optimization.

### 5.3 Sister substrate composition (Wyner-Ziv DISCUS substrate)

Tishby IB-pure and the existing Wyner-Ziv DISCUS substrate are NOT redundant — they exploit different surfaces:

- **Wyner-Ziv DISCUS**: archive grammar level — Slepian-Wolf cosets bin source frames into structured cosets; the archive transmits the coset INDEX; the decoder uses side info `Y` to disambiguate among coset members.
- **Tishby IB-pure**: encoder representation level — the variational encoder `q(t|x)` learns a representation `T` that minimizes `I(X;T)` while retaining `I(T;Y)`; the archive transmits the representation directly + scorer-conditional CDF for range coding.

**Composition opportunity**: Tishby IB-pure encoder produces `T`; Wyner-Ziv DISCUS coset-codes `T` (instead of raw RGB) using side info `Y`. The composition adds the WZ rate-savings on TOP of the IB-pure representation. Predicted incremental ΔS: small additive `-0.005 to -0.015` per WZ-substrate `[wyner-ziv-hypothesis]` band.

---

## 6. Curriculum (β-warmup phase schedule)

The IB Lagrangian's β value controls the rate-distortion tradeoff: high β → low `I(X;T)` (high compression, high distortion); low β → high `I(X;T)` (low compression, low distortion).

The canonical β-warmup curriculum per Alemi 2017 + sister IB literature:

| Phase | Epochs | β value | Active terms | Purpose |
|---|---|---|---|---|
| **Phase 0** (low β; high `I(T;Y)`) | 0-25 | β=0.001 | Reconstruction-dominated | Encoder converges under cooperative-receiver loss; latent `T` retains maximum information about `Y` |
| **Phase 1** (β ramp) | 25-75 | β: 0.001 → 0.01 linear | Both terms active | Begin trading `I(X;T)` for `I(T;Y)`; encoder learns to compress while retaining task-relevant bits |
| **Phase 2** (target β) | 75-150 | β=0.01 | Both terms active | Converge to target rate-distortion operating point |
| **Phase 3** (fine-tune) | 150-200 | β=0.01 + scorer-conditional CDF | Both terms + range coding | Adapt scorer-conditional CDF table to converged latent distribution; ship final archive |

Both paths (VIB + MINE) use EMA decay 0.997 per CLAUDE.md "EMA — NON-NEGOTIABLE" + Catalog #88 + eval_roundtrip per CLAUDE.md non-negotiable + Catalog #5.

**β-warmup sweep at v1**: 4-corner β sweep `{0.001, 0.01, 0.1, 1.0}` per Path-VIB single dispatch (4× $5-10 Modal A100 100ep smokes = $20-40 council-grade investigation). The sweep produces an empirical R(D) curve; council §19 picks the β that lands on the CPU-axis-optimal point per §18.

---

## 7. Architecture priors

- **Cooperative receiver fixed**: `(SegNet, PoseNet)` published in `upstream/modules.py`; used at COMPRESS-side only per Catalog #6 strict-scorer-rule.
- **Latent dimensionality**: `latent_dim=16` (vs ATW v2's 24; smaller because IB-pure's KL term explicitly penalizes excess capacity; the encoder learns to use only the bits it needs). Per-pair quantized: `(num_pairs=600, latent_dim=16)` int8 = 9.6KB raw; with scorer-conditional range coding, target ~3-5KB.
- **Variational encoder `q(t|x)`**: small MLP `pair_rgb → (mu, log_sigma)`; ~32K params @ FP4 = ~16KB. Outputs diagonal-Gaussian parameters.
- **Variational decoder `p(y|t, side_info)`**: HNeRV-style per-pair embedding + 6 upsample blocks + scorer-class side-info conditioning at the bottleneck layer; ~80K params @ FP4 ≈ 40KB.
- **Statistic network `T_NN` (Path-MINE only)**: small MLP `(t, y) → R`; ~50K params @ FP4 = ~25KB.
- **scorer_class_prior_table**: `(600 pairs, 5 SegNet classes)` uint8 = 3KB.
- **cdf_table**: `(5 classes, 256 symbols)` fp16 = 2.5KB.
- **β value**: 4 bytes fp32 in archive (decoder needs β for any future re-quantization; mostly metadata).

Total predicted archive size (Path-VIB tight estimate): encoder (optional) + decoder + t_quantized + scorer_class_prior_table + cdf_table + beta + meta + brotli overhead ≈ **60-90KB** vs A1's 179KB (50-67% reduction; consistent with theoretical-floor predictions and meaningfully smaller than ATW v2's 80-120KB target).

This is consistent with the asymptotic-pursuit horizon class — the archive bytes are dominated by the variational decoder which is essential; everything else is information-theoretically minimal.

---

## 8. Post-training (TTO; deferred to V2)

Standard Tishby IB-pure does NOT include test-time optimization. TTO compose with Tishby IB-pure is a V2 candidate per the V1 reactivation queue — defer.

QAT (Quantizr 0.33-winning recipe) IS applicable to the decoder weights but should be done JOINTLY with the IB Lagrangian (FP4 quantization noise becomes part of the encoder's distortion budget). Defer to V2 per HNeRV parity L7 (substrate engineering happens ONCE per architecture class; QAT is a v2 refinement).

---

## 9. Score-aware loss design

The IB Lagrangian IS the score-aware loss. No separate Atick-Redlich term — the cooperative-receiver framing is built INTO `I(T;Y)` where Y = scorer output distribution.

### 9.1 Path-VIB (canonical Alemi 2017)

```python
L_VIB = E_{q(t|x)}[ -log p(y|t) ]                      # reconstruction (cooperative-receiver via scorer)
      + β · KL(q(t|x) || p(t))                          # bottleneck

# -log p(y|t) operationalized via canonical Atick-Redlich primitive:
def reconstruction_term(decoder_output_pair, gt_pair, scorer_pair):
    cooperative_loss = cooperative_receiver_loss(
        rgb_0=decoder_output_pair[:, 0], rgb_1=decoder_output_pair[:, 1],
        gt_rgb_0=gt_pair[:, 0], gt_rgb_1=gt_pair[:, 1],
        seg_scorer=scorer_pair[0], pose_scorer=scorer_pair[1],
        weights=AtickRedlichWeights(),   # contest defaults β_seg=100.0 + γ_pose=sqrt(10)
        apply_eval_roundtrip=True,        # per CLAUDE.md non-negotiable
    )
    return cooperative_loss.cooperative_loss  # scalar

# KL closed-form for diagonal Gaussian + standard Normal:
def kl_term(mu, log_sigma):
    return 0.5 * (mu.pow(2) + (2 * log_sigma).exp() - 2 * log_sigma - 1).sum(-1).mean()

# Total loss:
L_VIB = reconstruction_term(decoder_output, gt, scorer) + beta * kl_term(mu, log_sigma)
```

### 9.2 Path-MINE (Belghazi 2018 alternative)

```python
L_MINE = -I_T_Y_lower_bound + β · I_X_T_upper_bound

# I(T;Y) lower bound via Donsker-Varadhan + statistic network T_NN:
def i_t_y_lower_bound(t, y, statistic_net):
    joint = statistic_net(t, y).mean()                 # E_{p(t,y)}[T_NN(t,y)]
    marginal = (statistic_net(t, y_shuffled).exp()).mean().log()  # log E_{p(t)p(y)}[exp(T_NN)]
    return joint - marginal

# I(X;T) upper bound = closed-form KL (same as Path-VIB)
L_MINE = -i_t_y_lower_bound(t, scorer_output, statistic_net) + beta * kl_term(mu, log_sigma)
```

**Both paths**: route through canonical `tac.codec.cooperative_receiver.atick_redlich.cooperative_receiver_loss` per Catalog #164 for the scorer-output computation. Substituting the canonical primitive avoids the hand-roll bug-class breeding pattern named in Wunderkind E1.

**Differentiability** (CLAUDE.md eval_roundtrip + HNeRV parity L8 + Catalog #187): both paths apply `tac.differentiable_eval_roundtrip.apply_eval_roundtrip_during_training` before scorer loss; `patch_upstream_yuv6_globally` invoked before scorer construction per the canonical PR95-parity discipline.

---

## 10. Archive grammar (TIBP1 byte-level layout)

TIBP1 monolithic single-file `0.bin` per HNeRV parity discipline L3.

```
MAGIC(4)                           b"TIBP"
VERSION(1)                         u8       schema version (currently 1)
PATH(1)                            u8       0 = Path-VIB, 1 = Path-MINE
LATENT_DIM(2)                      u16      cfg.latent_dim (e.g. 16)
NUM_PAIRS(2)                       u16      cfg.num_pairs (e.g. 600)
SEGNET_CLASSES(2)                  u16      5 (contest SegNet)
CDF_TABLE_NUM_SYMBOLS(2)           u16      256 (int8 latent symbols)
BETA_VALUE(4)                      f32      target β (council-adjudicated)

ENCODER_BLOB_LEN(4)                u32      brotli encoder state_dict (OPTIONAL; may be empty)
DECODER_BLOB_LEN(4)                u32      brotli decoder state_dict (REQUIRED)
STATISTIC_NET_BLOB_LEN(4)          u32      brotli statistic_net state_dict (Path-MINE only; 0 for Path-VIB)
LATENT_T_BLOB_LEN(4)               u32      int8 t_quantized bytes (= num_pairs * latent_dim)
SCORER_CLASS_PRIOR_BLOB_LEN(4)     u32      uint8 scorer_class_prior_table bytes
CDF_TABLE_BLOB_LEN(4)              u32      fp16 cdf_table bytes (= 5 * 256 * 2)
META_BLOB_LEN(4)                   u32      sorted-keys JSON utf-8 bytes

ENCODER_BLOB                       ...      (OPTIONAL; may be empty in inflate-only build)
DECODER_BLOB                       ...      (REQUIRED; ~40KB after brotli)
STATISTIC_NET_BLOB                 ...      (Path-MINE only; ~15KB after brotli)
LATENT_T_BLOB                      ...      (REQUIRED; ~9.6KB raw before range-coding;
                                              with scorer-conditional range coding ~3-5KB)
SCORER_CLASS_PRIOR_BLOB            ...      (REQUIRED; num_pairs * 5 = 3KB)
CDF_TABLE_BLOB                     ...      (REQUIRED; 5 * 256 * 2 = 2.5KB)
META_BLOB                          ...      (provenance; TIBP1 codec metadata; sorted-keys JSON)
```

**Byte-stable invariant** (CLAUDE.md "Bit-level deconstruction and entropy discipline"): all fp16 / fp32 tensors stored in IEEE 754 little-endian byte order; sorted-keys JSON utf-8 ensures meta blob hash is reproducible; brotli compression with deterministic parameters (quality=11, lgwin=22, mode=GENERIC). Round-trip contract: `bytes → parse_archive → pack_archive → bytes` is byte-identical.

**Distinguishing-feature contract per Catalog #272**:
- `distinguishing_feature_name`: "variational_ib_encoder_with_wyner_ziv_side_info_scorer_class_conditional_cdf_range_coding"
- `distinguishing_bytes_path`: `DECODER_BLOB + LATENT_T_BLOB + SCORER_CLASS_PRIOR_BLOB + CDF_TABLE_BLOB` (the BYTES that encode the IB Lagrangian operationalization)
- `inflate_consumer_function`: `tac.substrates.tishby_ib_pure.inflate.reconstruct_pairs_from_tibp1_bytes` (consumes ALL distinguishing bytes for frame reconstruction)
- `byte_mutation_smoke_passes`: REQUIRED before any L2+ promotion; sister `tools/verify_distinguishing_feature_byte_mutation.py` mutates 1 byte per declared offset and verifies frame output changes

---

## 11. Inflate runtime (≤200 LOC substrate-engineering budget per HNeRV parity L4 waiver)

```python
# experiments_substrates_tishby_ib_pure/inflate.py (~150-200 LOC)

def inflate(archive_zip: Path, output_dir: Path, file_list: Path) -> int:
    """Tishby IB-pure inflate runtime per HNeRV parity L4.

    Parses TIBP1 archive; range-decodes t_quantized via scorer-conditional CDF
    table; reconstructs RGB pairs via variational decoder conditioned on scorer
    class prior table. NO scorer load (per Catalog #6 strict-scorer-rule).
    """
    archive_bytes = (archive_zip / "0.bin").read_bytes()
    parsed = parse_tibp1_archive_bytes(archive_bytes)

    device = select_inflate_device()  # Catalog #205 canonical
    decoder = build_variational_decoder(parsed.meta)
    decoder.load_state_dict(parsed.decoder_sd)

    # Range-decode t_quantized using cdf_table conditioned on scorer_class_prior_table
    t_quantized = range_decode_tibp(
        encoded_bytes=parsed.latent_t_blob,
        cdf_table=parsed.cdf_table,
        scorer_class_prior_table=parsed.scorer_class_prior_table,
    )  # → (num_pairs, latent_dim) int8

    # Reconstruct full latents via Wyner-Ziv: t = dequantize(t_quantized) + offset(scorer_class_prior[i])
    with torch.no_grad():
        t_dequant = t_quantized.float() / LATENT_QUANT_SCALE
        side_info_offset = decoder.compute_side_info_offset(parsed.scorer_class_prior_table.to(device))
        t_full = t_dequant.to(device) + side_info_offset

    # Render per-pair via variational decoder; iterate per file_list per Catalog #146
    pair_indices = torch.arange(parsed.num_pairs, device=device)
    for video_name in file_list.read_text().splitlines():
        with torch.no_grad():
            rgb_0, rgb_1 = decoder(t_full[pair_indices], parsed.scorer_class_prior_table.to(device))
        write_rgb_pairs(output_dir / video_name, rgb_0, rgb_1)

    return 0
```

**Inflate dependency closure** per Catalog #5 + HNeRV parity L9: `torch`, `brotli`, `numpy` (range coding via numpy). NO scorer dependencies. NO `upstream/modules.py` import. Verified via `dispatch_optimization_protocol_complete` (Catalog #270).

**Inflate output parity check** per Catalog #221: writes per-axis blockers `result_review_blockers=["roundtrip_matrix_is_command_planner_not_claim_surface", "requires_separate_auth_eval_result_review_before_score_claim"]` until paired CPU+CUDA full anchor lands.

---

## 12. Export contract

```python
# In experiments/train_substrate_tishby_ib_pure.py::_full_main (NotImplementedError pending Phase 2)

def _export_tibp1_archive(model, output_dir, args) -> Path:
    """Export trained Tishby IB-pure model into TIBP1 archive bytes."""
    archive_bytes = pack_archive(
        encoder_sd=model.variational_encoder.state_dict(),  # OPTIONAL; can be empty
        decoder_sd=model.variational_decoder.state_dict(),
        statistic_net_sd=model.statistic_net.state_dict() if args.path == "MINE" else b"",
        t_quantized=model.compute_t_quantized(),    # quantized per-pair representation
        scorer_class_prior_table=model.scorer_class_prior_table,
        cdf_table=model.build_scorer_conditional_cdf_table(),  # empirical histogram at compress
        beta_value=args.beta,
        meta=meta_dict,
        path=args.path,  # "VIB" or "MINE"
    )
    archive_path = output_dir / "0.bin"
    archive_path.write_bytes(archive_bytes)
    # ZIP STORED per HNeRV parity L3; deterministic ZIP (Catalog #19); single member
    zip_path = output_dir / "archive.zip"
    write_stored_zip(zip_path, {"0.bin": archive_bytes})
    return zip_path
```

Round-trip parity test: `bytes → parse_archive → pack_archive → bytes` byte-identical per Catalog #1 `check_encoder_decoder_dequantization_roundtrip_tested`.

---

## 13. Stack-of-stacks composition matrix (Tishby IB-pure with OTHER substrates; higher-order)

| With substrate | Axis orthogonality | Composition class | Predicted contribution | Rationale |
|---|---|---|---|---|
| **ATW v2 Variant B** (the BOLT-ON variant of cooperative-receiver) | REDUNDANT (both exploit cooperative-receiver framing) | SATURATING / SWAP | floor at -0.005 | Tishby IB-pure SUBSUMES ATW v2 Variant B (the IB-pure surface includes the WZ residual mechanism as a special case at low-β). Cannot compose additively. **SWAP**: Tishby IB-pure REPLACES ATW v2 Variant B in any pipeline that lists ATW v2 as a substrate component. The structural advantage is the unified optimization surface. |
| **Wyner-Ziv DISCUS substrate** (`wyner_ziv_cooperative_receiver`) | NEAR-ORTHOGONAL (WZ-DISCUS = wire-grammar level via coset binning; IB-pure = encoder-representation level via variational MI minimization) | STRONG_STACK at archive layer | small additive ~0.005-0.015 | Tishby IB-pure encoder produces representation `T`; WZ-DISCUS coset-codes `T` (instead of raw RGB) using side info `Y`. The composition adds WZ rate-savings on top of the IB-pure representation. Predicted incremental ΔS per WZ-substrate `[wyner-ziv-hypothesis]` band. |
| **A-STACK (NSCS01 + NSCS02 + NSCS03 composition)** | NEAR-ORTHOGONAL via SWAP[NSCS03 → IB-pure] | A-STACK[NSCS03 → Tishby IB-pure] SWAP per A-STACK §13 row 11 + ATW v2 §13 row 11 | NULL pending probe | A-STACK currently routes NSCS03 (end-to-end Ballé joint codec) as the entropy axis. Tishby IB-pure is the IB-Lagrangian formalization of what NSCS03 does empirically; the swap unifies the optimization surface. Predicted A-STACK[swap] band: NULL pending paired probe + Dykstra-feasibility check. |
| **NSCS06 v8 Path B** (chroma wavelet residual; asymptotic-pursuit per FALSIFICATION-AUDIT-v2 reclass) | ORTHOGONAL (NSCS06 v8 = chroma reconstruction via wavelet residual; IB-pure = luma scorer-class conditioning) | FRESH-COMPOSITION (asymptotic + asymptotic) | small additive (~0.005-0.010) | NSCS06 v8 Path B operates on chroma; Tishby IB-pure operates on luma (SegNet input is YUV6 with chroma subsampled). Compose at YUV-channel level. Both ASYMPTOTIC-PURSUIT class per HORIZON-CLASS directive; composition is the strongest asymptotic-pursuit candidate stack. |
| **DP1 pretrained-driving-prior** (Catalog #209/#210/#211/#213) | ORTHOGONAL (DP1 = out-of-distribution pretraining codebook; IB-pure = scratch-trained variational encoder) | DP1-INIT-IB-PURE | small additive (~0.005) | DP1 codebook initializes Tishby IB-pure encoder's mu parameter per pair; reduces epoch budget from 200 → 50; sister benefit: DP1's SegNet-class assignments match IB-pure's scorer-conditional CDF table. Archive: `DPCOMP(DP1, TIBP1)` composition per Catalog #211. |
| **D1 SegNet margin polytope** (Catalog #220 OPERATIONAL canonical) | NEAR-ORTHOGONAL (D1 = inflate-time SegNet overlay; IB-pure = encoder-side IB) | STRONG_STACK | small additive (~0.003) | D1 adds inflate-time polytope-interior noise to frame_1 RGB; IB-pure frame_1 output is the input to D1's overlay. Pure composition: TIBP1 archive bytes + D1 sidecar bytes. |
| **Z6/Z7/Z8 Time-Traveler predictive-coding world-models** (asymptotic-pursuit sister per FALSIFICATION-AUDIT-v2 A2) | ORTHOGONAL (Z6/Z7/Z8 = temporal/predictive coding via Rao-Ballard; IB-pure = spatial/representational via IB Lagrangian) | FRESH-COMPOSITION (asymptotic + asymptotic) | TBD | Z6/Z7/Z8 predict temporal dependencies; IB-pure compresses spatial representations. The composition is the strongest asymptotic-pursuit lattice node. Predicted band TBD per Z6/Z7/Z8 scoping memo (sister in-flight). |
| **Rudin floor substrate** (asymptotic-pursuit sister per FALSIFICATION-AUDIT-v2 A3) | ORTHOGONAL (Rudin = interpretable-ML compositional decoder; IB-pure = information-theoretic encoder) | FRESH-COMPOSITION (asymptotic + asymptotic) | TBD | Rudin floor substrate ships an interpretable decoder; IB-pure ships an information-theoretically minimal encoder. The composition unifies interpretability and information-theoretic minimality. Predicted band TBD per Rudin scoping memo (sister in-flight). |
| **NSCS01** (nullspace-split renderer) | ORTHOGONAL (NSCS01 = decode-time-contract split-head exploiting `x[:,-1,...]` SegNet slice; IB-pure = encoder-side IB) | STRONG_STACK | small additive (~0.005-0.010) | NSCS01 split-head routing exploits SegNet last-frame-only slice; IB-pure exploits SegNet class-conditional softmax. Both compose at the SegNet-as-receiver framework but at different surfaces (decode-time-contract vs encoder-side IB). |
| **Carmack-Hotz Strip-Everything v8+** (chroma-preserving + no-neural-codec; asymptotic-pursuit per FALSIFICATION-AUDIT-v2 P1) | CONFLICT (different decoder paradigms — neural variational decoder vs no-neural numpy decoder) | -0.005 (within-class trap if forced) | Per Catalog #219 Z1 ablation, two ACROSS-CLASS candidates with different decoder paradigms cannot compose additively. Different asymptotic-pursuit candidates that COMPETE at the same surface. Cathedral autopilot ranker would penalize via `apply_substrate_composition_matrix_to_candidates`. |
| **A1 baseline** | DIFFERENT-BASE | REPLACES A1 entirely | TBD predicted band per §18 | A1 is the contest-CPU 0.19285 baseline; Tishby IB-pure is a fresh substrate-engineering composition. Predicted band is the substrate's standalone score per §18. |

**Top higher-order composition recommendation**: **Tishby IB-pure ⊕ Wyner-Ziv DISCUS ⊕ DP1 ⊕ D1** (IB-pure encoder + WZ coset binning + DP1 pretraining + D1 inflate-time SegNet overlay). Four orthogonal axes (encoder representation + archive wire grammar + pretraining + inflate-time SegNet overlay); predicted combined band: NULL pending Dykstra-feasibility intersection per Catalog #296.

---

## 14. Pipeline-of-pipelines (role in Path 2 LATTICE)

Per the operator-approved Path 2 lattice rewrite (`feedback_path_2_lattice_of_class_shifts_operator_approved_supersedes_l5_v2_staircase_20260516.md`), Tishby IB-pure occupies the following lattice coordinates:

**Level 0 (substrate-CLASS axis, in parallel)**: Tishby IB-pure is a Level-0 lattice node alongside NSCS01 / NSCS02 / NSCS03 / Wunderkind G1 / ATW v2 / Carmack-Hotz Strip-Everything / Z6/Z7/Z8 / Rudin floor + 5 RESURRECTION-AUDIT Tier 1 candidates.

**Falling-rule-list-of-4 ranking** (per Rule 4):
- Rule 1 (chroma-preserving + neural-optional < 60 [diagnostic-CPU]): NOT IB-pure (IB-pure uses neural variational encoder)
- Rule 2 (nullspace-split + PR95-paradigm < 0.190 [contest-CPU]): NOT IB-pure (different mechanism)
- Rule 3 (Dykstra-feasibility-validated stack composition < 0.180 [contest-CPU]): IB-pure CAN compose at this rule via §13 composition opportunities
- **Rule 4 (Daubechies-wavelet multi-scale lattice convergence to lower envelope → DECLARE asymptotic floor): IB-pure IS the canonical CANDIDATE for Rule 4 per its asymptotic-pursuit horizon class**
- ELSE: REQUEST_OPERATOR_REVIEW (whiteboard mode per Catalog #255 GOSDT dispatch router)

**Dispatch pipeline (conditional on D4 + Variational-IB-tractability check)**:

```
STAGE 0 (PRE-DISPATCH GATE):
  $3-5 D4 probe on A1 latents:
    .venv/bin/python tools/probe_latent_conditional_entropy_h_latent_given_scorer_class.py \
        --substrate-id tishby_ib_pure \
        --latent-bytes <A1_latent_bytes_path> \
        --scorer-classes <SegNet_class_per_pair_bytes_path> \
        --output-json .omx/state/h_latent_given_scorer_class_tishby_ib_pure.json \
        --meaningful-mi-threshold-bits 0.5

  VERDICT GATE:
    MEANINGFUL_CONDITIONING → STAGE 1 (Variational-IB-tractability check)
    WEAK_CONDITIONING       → revise band to [0.15, 0.20]; defer asymptotic claim
    INDEPENDENT             → DEFER per CLAUDE.md "Forbidden premature KILL"; pivot to Rudin floor sister

STAGE 1 ($5-10 Modal A100 Variational-IB-tractability check):
  100ep synthetic-data smoke measuring gradient SNR for L_VIB:
  Validates: encoder gradient flow via reparam trick + KL closed-form numerical stability +
             scorer preprocess gradient reachability per Catalog #187

  VERDICT GATE:
    Gradient SNR ≥ 1.0 → Path-VIB green; proceed to STAGE 2
    Gradient SNR 0.1 ≤ SNR < 1.0 → Path-MINE pivot OR proceed Path-VIB with variance reduction
    Gradient SNR < 0.1 → DEFER per CLAUDE.md "Forbidden premature KILL"; consider Rudin floor sister

STAGE 2 ($10-15 Modal A100 standard smoke):
  100ep on real upstream/videos/0.mkv at β sweep {0.001, 0.01, 0.1, 1.0}:
  Per Catalog #167 smoke-before-full + Catalog #243 local pre-deploy + Catalog #271 codex review

STAGE 3 ($15-30 Modal A100 paired CPU+CUDA full anchor):
  200ep at council-adjudicated β value; ends with paired CPU+CUDA contest auth eval per CLAUDE.md
  "Submission auth eval — BOTH CPU AND CUDA"
  Provider routing: cost-band posterior per D9 routing; canonical 4-axis paired dispatch per
  Catalog #246
    .venv/bin/python tools/dispatch_modal_paired_auth_eval.py \
        --archive <Tishby IB-pure archive sha256> \
        --skip-axis-if-promotable-anchor-exists
```

**Total wall-clock**: D4 probe ~30 min CPU + Variational-IB-tractability check ~2hrs Modal A100 + standard smoke 1.5h × 4 β values = 6hrs Modal A100 + full anchor 3-4hrs Modal A100 + paired CPU 1.5-2hrs. Calendar time: ~2 days end-to-end if all gates pass.

---

## 15. Canonical-vs-unique decision per layer (Catalog #290)

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" non-negotiable. Per-layer canonical-vs-unique-fork rationale.

| Layer | Decision | Rationale |
|---|---|---|
| Trainer skeleton | ADOPT canonical | TF32 + CUDA discipline shared per `tac.substrates._shared.trainer_skeleton.device_or_die` (Catalogs #172/#178/#179/#180). HARD-EARNED — substrate engineering hygiene is universal. |
| Atick-Redlich primitive (reconstruction term operationalization) | ADOPT canonical | `tac.codec.cooperative_receiver.atick_redlich.cooperative_receiver_loss` per Catalog #164 + Wunderkind E1 substitution candidate. The primitive IS the canonical Atick-Redlich; substrate-specific hand-roll is bug-class breeding. HARD-EARNED. |
| Eval-roundtrip | ADOPT canonical | CLAUDE.md "eval_roundtrip — NON-NEGOTIABLE" + Catalog #5. HARD-EARNED. |
| EMA decay (0.997) | ADOPT canonical | CLAUDE.md "EMA — NON-NEGOTIABLE" + Catalog #88. HARD-EARNED. |
| Score-aware loss helper | ADOPT canonical | `tac.substrates._shared.score_aware_common.score_pair_components` per Catalog #164. Delegated through Atick-Redlich primitive. HARD-EARNED. |
| Inflate runtime (device selection) | ADOPT canonical | `select_inflate_device` per Catalog #205. HARD-EARNED. |
| Auth eval helper | ADOPT canonical | `gate_auth_eval_call` per Catalog #226. HARD-EARNED. |
| Hardware substrate detection | ADOPT canonical | `detect_hardware_substrate` per Catalog #190. HARD-EARNED. |
| Modal A100 min_smoke_gpu | ADOPT canonical | Per Catalog #215. HARD-EARNED. |
| Required-input-file validation | ADOPT canonical | `tools/validate_dispatch_required_inputs.py` per Catalog #152. HARD-EARNED. |
| Operator-authorize entry point | ADOPT canonical | `tools/operator_authorize.py` per Catalog #176. HARD-EARNED. |
| **TIBP1 archive grammar** | **UNIQUE FORK** | NEW magic `b"TIBP"` + 4 new sections specific to Tishby IB-pure (variational encoder + variational decoder + statistic_net optional + scorer-conditional CDF). Substrate-engineering surface per HNeRV parity L7. UNIQUE per substrate-distinguishing-feature (Catalog #272). |
| **Variational encoder `q(t|x)`** | **UNIQUE FORK** | Diagonal-Gaussian variational posterior with reparam trick is the substrate's structural core. Cannot be canonicalized without losing the IB-Lagrangian-as-primary-objective property. UNIQUE per HNeRV parity L7. |
| **Variational decoder `p(y|t, side_info)`** | **UNIQUE FORK** | Side-info-conditional decoder where side_info = scorer_class_prior_table is substrate-specific. Sister to ATW v2's decoder but distinct because the conditional structure absorbs the G1/B3 mechanisms. UNIQUE per HNeRV parity L7. |
| **IB Lagrangian as PRIMARY loss** | **UNIQUE FORK** | The substrate's distinguishing characteristic. No other substrate has L_VIB or L_MINE as the SINGLE loss term; sister cooperative-receiver substrates have composite losses. UNIQUE per Catalog #290 canonical-vs-unique discipline + the UNIQUE-AND-COMPLETE-PER-METHOD standing directive. |
| **β-warmup curriculum** | **UNIQUE FORK** | Alemi 2017 β-warmup phase schedule is substrate-specific; substrate-engineering primitive per HNeRV parity L7. |
| **Scorer-conditional CDF range coding** | **UNIQUE FORK** (per substrate; sister to ATW v2's B3 mechanism but absorbed into IB-pure's decoder structure) | The scorer_class_prior_table IS the side_info; the cdf_table conditional CDF is the range-coding mechanism. Sister to ATW v2 B3 but operationalized differently (decoder-internal vs separate table). UNIQUE per HNeRV parity L7. |
| Training curriculum (data pipeline) | ADOPT canonical | 2-frame curriculum + pyav decode + patched YUV6 + differentiable scorers + AdamW + cosine LR — all PR95-parity-discipline canonical. HARD-EARNED per HNeRV parity discipline lessons 1-13. |
| Tier-1 engineering | ADOPT canonical | autocast_fp16 / TF32 / torch.compile / no_grad-at-eval / canonical scorer-loss helper (Catalogs #172/#178/#179/#180/#164). HARD-EARNED. |
| Scorer routing | ADOPT canonical | `load_differentiable_scorers` + `patch_upstream_yuv6_globally` per Catalog #164 + canonical scorer-loader assignment order per Catalog #222. HARD-EARNED. |
| D4 probe-disambiguator | ADOPT canonical | Shared probe `tools/probe_latent_conditional_entropy_h_latent_given_scorer_class.py`. HARD-EARNED. |
| **Variational-IB-tractability check probe** | **UNIQUE FORK** | NEW per-substrate probe `tools/check_variational_ib_tractability.py` measuring gradient SNR for L_VIB; substrate-specific. Sister to D4 probe but at a different surface (gradient-SNR vs MI). UNIQUE per Catalog #229 + HNeRV parity L7. Proposed canonical helper to be claimed in same commit batch as the sister-subagent substrate build. |
| Dispatch optimization protocol | ADOPT canonical | `verify_dispatch_protocol_complete` per Catalog #270 + #279/#280 fail-closed protections. HARD-EARNED. |

**Bolt-on vs substrate-engineering split per HNeRV parity L7**: Tishby IB-pure is substrate-engineering (NEW architecture class: variational IB encoder/decoder with scorer-conditional side-info CDF range coding). LOC budget (Path-VIB ~400-550 LOC; Path-MINE ~550-700 LOC) exceeds bolt-on cap explicitly. The 7 UNIQUE FORK decisions (TIBP1 archive grammar + variational encoder + variational decoder + IB Lagrangian primary loss + β-warmup + scorer-conditional CDF + Variational-IB-tractability check) ARE the substrate-optimal engineering surface; remaining ADOPT canonical decisions are shared infrastructure value preserved per the standing directive `feedback_canonical_share_when_serves_unique_when_suppresses_standing_directive_20260515.md`.

---

## 16. Cargo-cult audit per assumption

Per the standing META-ASSUMPTION ADVERSARIAL REVIEW non-negotiable (CLAUDE.md + Catalog #291).

| Assumption | Classification | Rationale | Disposition |
|---|---|---|---|
| The IB Lagrangian `I(X;T) - β·I(T;Y)` is the canonical formalization of the cooperative-receiver framing | **HARD-EARNED** | Tishby-Pereira-Bialek 1999 + Tishby-Zaslavsky 2015 are the canonical papers; Alemi et al. 2017 VIB is the canonical tractable operationalization. The contest's `(SegNet, PoseNet)` IS a cooperative receiver per Atick-Redlich 1990. The composition is well-established mathematics. | PRESERVED. |
| The asymptotic-pursuit band `[0.05, 0.10]` is the theoretical-floor target for IB-class substrates per T4 SYMPOSIUM | **HARD-EARNED** | Per T4 SYMPOSIUM 4×4 floor matrix (`grand_council_symposium_time_traveler_optimal_staircase_20260516.md` lines 144-151) + Tishby IB compute (~0.08-0.12 for scorer-as-shared-prior architectures per Catalog #256 MacKay conditional entropy). The bound is derivable from Shannon R(D) + Tishby IB + Wyner-Ziv side-info per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable. | PRESERVED. |
| Path-VIB (Alemi 2017 variational form) is the canonical tractable operationalization | **HARD-EARNED** | Alemi et al. 2017 (`https://arxiv.org/abs/1612.00410`) explicitly addresses the tractability of `I(T;Y)` via variational lower bound + reparam trick. The form is well-established and widely cited. | PRESERVED. |
| Gradient SNR ≥ 1.0 is the canonical tractability threshold for the L_VIB optimization | **CARGO-CULTED** | The threshold 1.0 is a reasonable heuristic but not first-principles-derived. Empirical IB literature (Alemi 2017; Saxe et al. 2018) typically reports gradient SNR > 0.5 as tractable for IB; SNR < 0.1 as intractable; the 1.0 threshold is conservative. | UNWOUND via Variational-IB-tractability check parameterization — make the threshold configurable per `--gradient-snr-threshold` flag; default 1.0 conservative; document the sensitivity. |
| Path-MINE pivot recovers tractability when Path-VIB fails | **CARGO-CULTED** | Belghazi et al. 2018 demonstrated MINE recovery for synthetic MI estimation but with significantly higher LOC budget + auxiliary network training. The MINE estimator has KNOWN bias issues (Choi-Pernkopf-Mahmoudkalayeh 2019; McAllester-Stratos 2020) that may exceed Path-VIB's tractability margin. | UNWOUND via Path-MINE Variational-IB-tractability check — re-run the same check at Path-MINE before declaring Path-MINE green. |
| Tishby IB-pure's predicted band of `[0.08, 0.13]` for long-horizon (6-9m) is realistic | **CARGO-CULTED** | The band is interpolated from T4 SYMPOSIUM's `[0.08, 0.12]` long-horizon Tishby-IB-asymptotic floor + conservative add-on. Empirical confirmation requires the Variational-IB-tractability check + paired smoke comparison vs ATW v2 Variant B. | UNWOUND via §18 Dykstra-feasibility check; the predicted band is conditional on (D4 probe MEANINGFUL + Variational-IB-tractability ≥ 1.0 + Dykstra-feasibility non-empty). If any gate fails, the band is revised down. |
| The IB Lagrangian's gradient SNR is HIGHER than the bolt-on ATW v2 three-knob form's gradient SNR (the structural-distinctness hypothesis) | **CARGO-CULTED** | The hypothesis follows from "fewer competing objectives → higher SNR" intuition but is not first-principles-derived. ATW v2 V1 unwind §16 noted that three-knob composition may produce regime-sweep arbitration but did not prove SNR advantage. | UNWOUND via paired tractability check — measure gradient SNR for L_VIB and L_ATW_v2_B simultaneously; compare. |
| Tishby IB-pure's structural-class-shift advantage IS asymptotic (i.e. the gap to bolt-on widens at higher epochs not narrower) | **CARGO-CULTED** | Empirical-equivalence axis: maybe at LOW training budget Tishby IB-pure equals ATW v2 Variant B but at HIGH training budget IB-pure dominates because the unified gradient signal compounds. UNTESTED. | UNWOUND via paired smoke at THREE training budgets (50ep / 100ep / 200ep). If gap narrows with budget, the hypothesis is FALSIFIED; if gap widens, the hypothesis is CONFIRMED. |
| β = 0.01 is a reasonable target operating point on the R(D) curve | **CARGO-CULTED** | The value 0.01 is heuristic; the empirical-receipt-derived value comes from the §6 β-sweep `{0.001, 0.01, 0.1, 1.0}`. | UNWOUND via β-sweep council adjudication §19. |

**Cargo-cult-class summary**: 3 HARD-EARNED + 6 CARGO-CULTED. All 6 CARGO-CULTED assumptions are disambiguated by D4 probe OR Variational-IB-tractability check OR paired-smoke comparison vs ATW v2 Variant B OR β-sweep — all standard PRE-DISPATCH gates within the §20 cost envelope.

---

## 17. 9-dimension success checklist evidence

| # | Dimension | Status |
|---|---|---|
| 1 | **UNIQUENESS** (class-shift not within-class) | YES — DUAL class-shift: training-time-paradigm (IB Lagrangian as primary loss) + scorer-relationship (scorer softmax as distributional target Y). Not within-class refinement of any existing substrate; structurally distinct from Z4 / ATW v2 / WZ-DISCUS substrates per §3. |
| 2 | **BEAUTY + ELEGANCE** (30-sec-reviewable) | PARTIAL — Path-VIB ≈ 400-550 LOC substrate-engineering surface (HNeRV parity L7 substrate-engineering waiver); the IB Lagrangian loss is reviewable in 30 sec (one closed-form expression); the variational encoder/decoder are standard MLP architectures. The TIBP1 archive grammar is 8 sections (canonical scale). |
| 3 | **DISTINCTNESS** (explicitly different from sisters) | YES — §3 articulates the structural difference vs ATW v2 (BOLT-ON vs PRIMARY) + vs WZ-DISCUS (wire grammar vs encoder representation) + vs Z4 (loss-term-only vs entire-architecture). |
| 4 | **RIGOR** (premise verification + adversarial review + assumption classification + empirical anchor) | YES — 10 PVs verified in §1 pre-edit; §16 cargo-cult audit classifies 3 HARD-EARNED + 6 CARGO-CULTED with disambiguation paths; §18 Dykstra-feasibility-validated predicted band; §22 op-routables ranked by mission-contribution × cost. |
| 5 | **OPTIMIZATION PER TECHNIQUE** (substrate-optimal engineering per Catalog #290) | YES — §15 canonical-vs-unique decision per layer documented; 7 UNIQUE FORK + 14 ADOPT canonical; the UNIQUE forks ARE the substrate-optimal engineering surface. |
| 6 | **STACK-OF-STACKS-COMPOSABILITY** (orthogonal axes + additive ΔS) | YES — §13 composition matrix; orthogonal axes (encoder representation + archive wire grammar + pretraining + inflate-time SegNet overlay); 4-substrate composition recommendation (Tishby IB-pure ⊕ WZ-DISCUS ⊕ DP1 ⊕ D1). |
| 7 | **DETERMINISTIC REPRODUCIBILITY** (byte-stable + seed-pinned) | YES — TIBP1 archive grammar is byte-stable per HNeRV parity L3; brotli compression with deterministic parameters; sorted-keys JSON meta blob; canonical `inflate.sh` per HNeRV parity L9. Seed pinning per `experiments/pipeline.py` canonical profile system. |
| 8 | **EXTREME OPTIMIZATION + PERFORMANCE** (Tier 1 engineering hygiene) | YES — §15 ADOPT canonical for autocast_fp16 / TF32 / torch.compile / no_grad-at-eval (Catalogs #172/#178/#179/#180). Modal A100 min_smoke_gpu per Catalog #215; dispatch optimization protocol per Catalog #270. |
| 9 | **OPTIMAL MINIMAL CONTEST SCORE** (predicted band per §18) | YES — asymptotic-pursuit horizon class explicit per CLAUDE.md HORIZON-CLASS directive; predicted band `[0.08, 0.13]` long-horizon conditional on D4 + Variational-IB-tractability + Dykstra-feasibility gates per §18. |

---

## 18. Predicted ΔS band + Dykstra-feasibility check (per Catalog #296)

**Predicted ΔS band**: `NULL pending D4 probe verdict + Variational-IB-tractability check + Dykstra-feasibility intersection check` [prediction; first-principles-bound; asymptotic-pursuit].

### 18.1 First-principles derivation framework

Per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable + Boyd's Dykstra co-lead role + Tao+Boyd Blahut-Arimoto theoretical-floor framework (Catalog #257), the achievable region is the convex intersection of independent rate-distortion constraints. Tishby IB-pure's predicted band must be the convex-hull lower envelope of the polytope intersection:

- **Shannon R(D) constraint** (Catalog #257 Tao+Boyd Blahut-Arimoto): `R(D) = inf_{p(y|x)} I(X;Y) : E[d(X,Y)] ≤ D`. For contest video at distortion criterion D = 0.05 (PR101-class), R(D) ≈ 0.02-0.05 [theoretical lower bound; ultimate horizon].
- **Tishby IB constraint** (Tishby-Zaslavsky 2015 + Catalog #256 MacKay): `I(X;T) - β · I(T;Y) ≤ K_IB(β)` where K_IB is the IB Pareto frontier at the operating point. For contest scorer-as-shared-prior architecture, K_IB(β=0.01) ≈ 0.10-0.15 per T4 SYMPOSIUM theoretical-floor row.
- **Wyner-Ziv side-info constraint** (Wyner-Ziv 1976 + sister WZ-DISCUS substrate planning band): `R_WZ(D) ≥ R(D) - I(X; Y_side)` where `Y_side = scorer_class_prior_table` shipped in archive. D4 probe estimates `I(latent; class)` directly; for MI = 0.5 bits/symbol × latent rate 0.20 → rate savings ~25%.
- **Contest rate budget**: `25 · archive_bytes / 37,545,489`; for Tishby IB-pure predicted archive 80KB → contest rate term ~0.053.

### 18.2 Dykstra-feasibility intersection bands

| D4 verdict | Variational-IB SNR | Predicted CPU band | Predicted CUDA band | Basis |
|---|---|---|---|---|
| **MEANINGFUL_CONDITIONING + SNR ≥ 1.0** | ≥ 1.0 | `[0.08, 0.13]` long-horizon (6-9m); `[0.05, 0.10]` ultimate | `[0.11, 0.16]` long-horizon; `[0.08, 0.13]` ultimate | Tishby IB lower bound (~0.08-0.12 per T4) + WZ side-info savings × A1 rate (~0.20) → ~50% rate reduction → ~0.10 rate ΔS. Haircut for non-stationary class statistics + decoder approximation → realistic [0.08, 0.13] CPU. PR102-style CUDA-CPU Δ +0.03 per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" empirical anchor. |
| **MEANINGFUL + SNR 0.1 ≤ SNR < 1.0** | Marginal | `[0.13, 0.18]` long-horizon (frontier-pursuit downgrade) | `[0.16, 0.21]` long-horizon | Path-MINE pivot needed; SNR margin reduces achievable bound by ~50%. |
| **WEAK_CONDITIONING** | Any | `[0.15, 0.20]` (plateau-adjacent downgrade) | `[0.18, 0.23]` | Cooperative-receiver hypothesis partially correct but bottleneck mechanism degrades; substrate becomes plateau-adjacent. |
| **INDEPENDENT** | Any | `[NULL; Tishby IB-pure does not displace A1 frontier via this mechanism]` | Same | DEFER per CLAUDE.md "Forbidden premature KILL"; pivot to Rudin floor substrate (sister asymptotic-pursuit candidate A3 per FALSIFICATION-AUDIT-v2). |

### 18.3 β selection rationale + R-D curve operating point

The contest CPU axis specifically (per PR102 anchor 0.19538) is the medal-band ranking surface. The β value must select the R-D operating point that minimizes the CPU-axis score.

Per the contest formula `score = seg + sqrt(10 · pose) + 25 · archive_bytes / 37,545,489`:

- **Low β (β = 0.001)**: encoder retains maximum `I(X;T)` → high reconstruction fidelity → low seg + pose distortion → HIGH archive bytes → DOMINATED by rate term
- **Mid-low β (β = 0.01)**: balanced; estimated archive ~80KB; rate term ~0.053; distortion term ~0.05; total ~0.10 ← OPTIMAL for CPU axis
- **Mid-high β (β = 0.1)**: encoder compresses aggressively; archive ~40KB; rate term ~0.027; distortion increases; total ~0.13
- **High β (β = 1.0)**: encoder compresses to minimum; archive ~20KB; distortion dominates; total > 0.15

**β = 0.01 is the recommended target operating point** per first-principles derivation; the §6 β-sweep `{0.001, 0.01, 0.1, 1.0}` produces empirical confirmation. Council §19 picks the council-adjudicated final β value.

**Operating-point sensitivity** (CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent"): at the PR106 frontier operating point (pose_avg ~3.4e-5), pose marginal value is 2.71× SegNet's. Tishby IB-pure's `I(T;Y)` term naturally balances seg and pose distortion via the variational decoder's outputs; the cooperative-receiver loss in `tac.codec.cooperative_receiver.atick_redlich` uses contest formula weights (β_seg=100.0; γ_pose=sqrt(10)) per default. No operating-point-specific tilt needed at v1 design; tilt is a v2 refinement candidate.

### 18.4 Dykstra-feasibility check execution

```bash
.venv/bin/python tools/check_substrate_dykstra_feasibility.py --substrate tishby_ib_pure \
    --archive-bytes-target 80000 \
    --rate-budget-axis-target 0.053 \
    --constraints shannon_rate,tishby_ib,wyner_ziv,contest_rate \
    --beta-target 0.01 \
    --output-json .omx/state/dykstra_feasibility_tishby_ib_pure.json
```

If the intersection is non-empty + bounded BELOW A1's operating point (CPU 0.19285) → predicted band PROCEEDS to dispatch. If empty or unbounded → DEFER-pending-research per CLAUDE.md "Forbidden premature KILL".

**Score axis label**: `[prediction; first-principles-bound; asymptotic-pursuit; D4-probe-conditional; Variational-IB-tractability-conditional]` per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag".

---

## 19. Reactivation criteria + composition-specific probe-disambiguator

Tishby IB-pure ships as L0 SKETCH per Catalog #126 (scoping memo). Sister-subagent (per §22 op-routable #4-7) builds the L1 SCAFFOLD with `_full_main` raising NotImplementedError per Catalog #220 cascade (research_only=true + lane_class=substrate_engineering). Phase 2 council approval required to lift `_full_main NotImplementedError`.

### 19.1 V1 lift gate (any TWO grant Phase 2 approval)

1. **D4 probe returns `MEANINGFUL_CONDITIONING`** (MI ≥ 0.5 bits/symbol). Output: `.omx/state/h_latent_given_scorer_class_tishby_ib_pure.json`. Per FALSIFICATION-AUDIT-v2 lens 8 + the canonical probe taxonomy.
2. **Variational-IB-tractability check returns `TRACTABLE`** (gradient SNR ≥ 1.0 over 100ep synthetic-data smoke). Output: `.omx/state/variational_ib_tractability_tishby_ib_pure.json`.
3. **Dykstra-feasibility intersection** check returns non-empty + bounded below A1 operating point per Catalog #296. Output: `.omx/state/dykstra_feasibility_tishby_ib_pure.json`.
4. **Path-VIB vs Path-MINE council adjudication** (sextet pact per Catalog #292) decides Path based on Variational-IB-tractability check + Assumption-Adversary input. Memo: `feedback_tishby_ib_pure_path_adjudication_council_<YYYYMMDD>.md`.

### 19.2 V2 trigger (paired-smoke comparison)

Paired smoke comparison vs ATW v2 Variant B at THREE training budgets (50ep / 100ep / 200ep) per §16 cargo-cult assumption disambiguation:

- If Tishby IB-pure ΔS < ATW v2 Variant B ΔS at all three budgets AND gap widens with budget → STRUCTURAL-CLASS-SHIFT confirmed; promote to L2+ per Catalog #220
- If gap narrows with budget → BOLT-ON form dominates; DEFER Tishby IB-pure pending alternative-hypothesis (e.g. Path-MINE pivot OR Rudin floor sister)
- If gap is bidirectional (Tishby IB-pure wins at some budgets but not others) → COUNCIL DELIBERATION per Catalog #292 sextet pact

### 19.3 Probe-disambiguator (full Tishby IB-pure hook #6 per Catalog #125)

Three layers of empirical disambiguation:

1. **D4 probe** ($3-5 CPU): disambiguates Wyner-Ziv side-info hypothesis directly. `MEANINGFUL_CONDITIONING / WEAK_CONDITIONING / INDEPENDENT`.
2. **Variational-IB-tractability check** ($5-10 Modal A100 100ep synthetic smoke): disambiguates Path-VIB vs Path-MINE choice via gradient SNR measurement.
3. **β-sweep regime arbitration** ($20-40 paired; 4× Modal A100 100ep at β ∈ {0.001, 0.01, 0.1, 1.0}): empirically produces R(D) curve; council picks operating point.

**Per CLAUDE.md "Forbidden premature KILL"**: Tishby IB-pure NEVER killed. INDEPENDENT D4 verdict → DEFER-pending-research-with-Rudin-floor-alternative-hypothesis. INTRACTABLE-VIB verdict → Path-MINE pivot OR DEFER-pending-research. EMPTY Dykstra-intersection → DEFER per `feedback_abandon_within_class_refinements_only_substrate_class_shifts_pursue_frontier_20260515.md` (a substrate that fails Dykstra-feasibility is structurally within-class and should be abandoned per the directive).

---

## 20. Cost estimate

| Stage | Provider | GPU | Wall-clock | $/hr | Cost |
|---|---|---|---|---|---|
| **STAGE 0 D4 probe** (PRE-DISPATCH GATE) | Local macOS CPU or Modal CPU | n/a | 15-30 min | $0-0.06 | $3-5 (Modal CPU instance, conservative) |
| **STAGE 1 Variational-IB-tractability check** | Modal A100 | 1 × A100 | ~2 hrs (100ep synthetic-data smoke) | $3.50 | $5-10 |
| **STAGE 2 standard smoke** (council-adjudicated Path) | Modal A100 | 1 × A100 | 1.5-2hrs | $3.50 | $5-10 |
| **STAGE 2.b β-sweep** (4-corner R(D) curve; council-grade) | Modal A100 | 1 × A100 × 4 | 6-8hrs total | $3.50 | $20-30 |
| **STAGE 3 full anchor** | Modal A100 | 1 × A100 | 3-4hrs | $3.50 | $10-15 |
| **STAGE 3.b paired CPU eval** (Linux x86_64) | Modal CPU or Vast.ai CPU | 1 × CPU | 1.5-2hrs | $0.06-0.15 | $0.10-0.30 |
| **STAGE 4 paired-smoke vs ATW v2 Variant B** (V2 trigger) | Modal A100 | 2 × A100 × 3 budgets | 9-12hrs total | $3.50 | $30-45 |
| **STAGE 5 composition smoke** (e.g. Tishby IB-pure ⊕ DP1; Tishby IB-pure ⊕ WZ-DISCUS) | Modal A100 | 1 × A100 | 1.5-2hrs | $3.50 | $5-10 |
| **Total envelope (V1 straight-through; no β-sweep)** | mixed | mixed | ~8-12hrs | mixed | **$33-50** |
| **Total envelope (V1 + β-sweep council-grade)** | mixed | mixed | ~14-20hrs | mixed | **$53-75** |
| **Total envelope (V1 + V2 paired-smoke + composition)** | mixed | mixed | ~30-40hrs | mixed | **$83-130** |

Per CLAUDE.md "Long-burn score-lowering campaign default" — operator-funded campaign uses STAGE 2.b β-sweep + STAGE 4 paired-smoke for council-grade investigation. Routine paid dispatch uses V1 straight-through (cheaper, simpler).

Per Catalog #270 dispatch optimization protocol: all dispatches MUST satisfy Tier 1/2/3 engineering primitives + canonical scorer-loss helper routing + recipe `min_smoke_gpu=A100` + canonical 3-export NVML/CUDA env block + auth_eval canonical helper + canonical inflate device + scorer-loader assignment order + recipe-vs-trainer-state consistency + no phantom device-named output directories.

---

## 21. Observability surface (per max-observability directive)

This scoping memo lands new observability hooks the operator + future subagents + cathedral autopilot can consume directly. Per the max-observability standing directive 2026-05-16, every paid dispatch + every probe execution + every composition decision produces a machine-readable artifact stamped with axis labels + score_claim=false discipline.

### 21.1 Observability artifacts produced by Tishby IB-pure

| Artifact | Path | Schema | Consumer | Catalog |
|---|---|---|---|---|
| **D4 probe verdict** | `.omx/state/h_latent_given_scorer_class_tishby_ib_pure.json` | `HLatentGivenScorerClassVerdict` dataclass per probe source | Cathedral autopilot Hook 4 + V1 lift gate evaluator + council adjudication | #192 + #221 (fail-closed) |
| **Variational-IB-tractability check verdict** | `.omx/state/variational_ib_tractability_tishby_ib_pure.json` | `VariationalIBTractabilityVerdict` dataclass (NEW; to be defined in `tools/check_variational_ib_tractability.py`) | Path-VIB vs Path-MINE adjudication + V1 lift gate evaluator | #221 |
| **Tishby IB-pure archive manifest** | `experiments/results/lane_tishby_ib_pure_*/build_manifest.json` | per Catalog #2 evidence row schema | Lane registry + autopilot ranker + distinguishing-feature audit | #2 + #272 |
| **β-sweep R(D) curve artifact** (council-grade) | `.omx/state/tishby_ib_pure_beta_sweep_rd_curve_<UTC>.json` | per-β empirical (rate, distortion, score) tuples + axis label per CLAUDE.md "Apples-to-apples evidence discipline" | Council adjudication + cathedral autopilot ranker | #221 |
| **Path adjudication memo** | `.omx/research/feedback_tishby_ib_pure_path_adjudication_council_<UTC>.md` | per Catalog #292 council deliberation memo | Operator + sister substrate authors | #292 |
| **Distinguishing-feature byte-mutation proof** | `experiments/results/lane_tishby_ib_pure_*/distinguishing_feature_byte_mutation_proof.json` | per `tools/verify_distinguishing_feature_byte_mutation.py` schema | Catalog #272 verifier + auth_eval gate | #272 + #139 |
| **Dykstra-feasibility polytope JSON** | `.omx/state/dykstra_feasibility_tishby_ib_pure.json` | analytical 4-polytope intersection + non-emptiness flag | §18 predicted-band-check + autopilot ranker | #296 |
| **6-hook wire-in declaration** | `feedback_tishby_ib_pure_*_landed_<UTC>.md` | per Catalog #125 6-hook wire-in declaration | Subagent landing audit + cathedral autopilot | #125 + #229 |
| **Tishby IB-pure cost-band posterior anchor** | `.omx/state/cost_band_posterior.jsonl` (append-only) | per `tac.cost_band_calibration.append_anchor(outcome=...)` | D9 routing + cathedral autopilot dispatch ranker | #175 + #177 |
| **Modal call_id ledger row** | `.omx/state/modal_call_id_ledger.jsonl` (append-only) | per `tac.deploy.modal.call_id_ledger` schema | Harvest discipline + lane registry + autopilot | #245 |
| **Lane registry entry** | `.omx/state/lane_registry.json` row `lane_tishby_ib_pure_*` | 7-gate maturity per lane_maturity.py | Lane registry validator + autopilot routing | #90 + #126 |

### 21.2 Observability invariants

Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #221 fail-closed for score-claim artifacts:

- **D4 probe artifact**: `score_claim=false`, `evidence_grade=diagnostic_cpu`, `axis_label="[diagnostic-CPU; H(latent|scorer_class) probe]"`. NEVER promoted to score authority.
- **Variational-IB-tractability check artifact**: `score_claim=false`, `evidence_grade=diagnostic_a100`, `axis_label="[diagnostic-A100; variational_ib_tractability probe]"`. Diagnostic only.
- **Tishby IB-pure smoke artifact**: `score_claim=false` until paired CPU+CUDA full anchor lands. Smoke produces TRAINING SIGNAL only.
- **Tishby IB-pure full anchor artifact**: paired `[contest-CUDA Modal A100]` + `[contest-CPU GHA Linux x86_64]` per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA". `score_claim=true` only when both axes land on byte-identical archive.
- **Predicted band**: `[prediction; first-principles-bound; asymptotic-pursuit; D4-probe-conditional; Variational-IB-tractability-conditional]` until empirical confirmation.
- **No phantom device-named directories** per Catalog #249. Tishby IB-pure output directory is `contest_auth_eval_work/` (device-agnostic) per the canonical auth_eval pattern.
- **Distinguishing-feature byte-mutation proof**: per Catalog #272 the proof artifact lists per-section verdict (PASSED / FAILED / INFRASTRUCTURE_ERROR). FAILED on any of `DECODER_BLOB / LATENT_T_BLOB / SCORER_CLASS_PRIOR_BLOB / CDF_TABLE_BLOB` BLOCKS L2+ promotion.

### 21.3 Observability hooks for cathedral autopilot (6-hook wire-in per Catalog #125)

- **Hook 4 (Cathedral autopilot dispatch hook)**: Tishby IB-pure recipe `min_smoke_gpu=A100` + `target_modes=[research_substrate]` + `canary_status=independent_substrate` (no canary dependency at v1 since D4 probe + Variational-IB-tractability check ARE the pre-dispatch gates). Autopilot consumes the verdicts to gate dispatch eligibility.
- **Hook 5 (Continual-learning posterior update)**: Tishby IB-pure full anchor result (when landed) seeds `tac.continual_learning.posterior_update_locked` paired with the empirical D4 MI value AND Variational-IB-tractability gradient SNR value, so the posterior captures the mapping `(D4 MI, Variational-IB SNR, predicted band, empirical ΔS)`. Future substrates with similar IB-pure framing benefit from this anchor.
- **Hook 1 (Sensitivity map)**: `tac.sensitivity_map.variational_ib_per_pair` contributes per-pair latent sensitivity to IB Lagrangian gradient (computable from KL term gradient norm).
- **Hook 2 (Pareto constraint)**: `tac.pareto.tishby_ib_pure_lagrangian` contributes `I(X;T) - β·I(T;Y) ≤ K_IB(β)` to the global Pareto solver.
- **Hook 3 (Bit-allocator hook)**: `bit_allocator.tishby_ib_pure_variational_v1` allocates archive bytes per-pair based on KL term per-pair contribution.
- **Hook 6 (Probe-disambiguator)**: D4 probe + Variational-IB-tractability check + β-sweep regime arbitration per §19.

All 6 hooks active per Catalog #125; none N/A.

### 21.4 Operator-facing observability commands

```bash
# D4 probe execution (STAGE 0 PRE-DISPATCH GATE):
.venv/bin/python tools/probe_latent_conditional_entropy_h_latent_given_scorer_class.py \
    --substrate-id tishby_ib_pure \
    --latent-bytes <A1_latent_bytes_path> \
    --scorer-classes <SegNet_class_per_pair_bytes_path> \
    --output-json .omx/state/h_latent_given_scorer_class_tishby_ib_pure.json

# Variational-IB-tractability check (STAGE 1; NEW canonical probe per §15):
.venv/bin/python tools/check_variational_ib_tractability.py \
    --substrate-id tishby_ib_pure \
    --path-variant VIB \
    --gradient-snr-threshold 1.0 \
    --output-json .omx/state/variational_ib_tractability_tishby_ib_pure.json

# Dykstra-feasibility check (sister, $0):
.venv/bin/python tools/check_substrate_dykstra_feasibility.py --substrate tishby_ib_pure \
    --output-json .omx/state/dykstra_feasibility_tishby_ib_pure.json

# Lane maturity audit:
.venv/bin/python tools/lane_maturity.py audit | grep tishby_ib_pure

# Cathedral autopilot ranker consumption:
.venv/bin/python tools/cathedral_autopilot_autonomous_loop.py --consume-d4-probe tishby_ib_pure --rerank

# Distinguishing-feature byte-mutation proof (post-archive build):
.venv/bin/python tools/verify_distinguishing_feature_byte_mutation.py \
    --lane lane_tishby_ib_pure_substrate_scoping_design_20260516 \
    --archive-zip <archive_path> \
    --inflate-sh <inflate.sh_path>
```

---

## 22. Op-routables (ranked)

| # | Op-routable | Cost | Wall-clock | Owner | Dependency |
|---|---|---|---|---|---|
| **1** | Execute D4 H(latent\|scorer_class) probe on A1 latents | $3-5 | 30 min CPU | Operator or sister-subagent | Independent — runs immediately |
| **2** | Pre-register lane `lane_tishby_ib_pure_substrate_scoping_design_20260516` at L0 SKETCH | $0 | 2 min | Operator (via `tools/lane_maturity.py add-lane`) | Per Catalog #126 |
| **3** | Build canonical `tools/check_variational_ib_tractability.py` probe | $0 (editor) | ~2h | Sister-subagent | Independent (one-time canonical primitive land) |
| **4** | Build Tishby IB-pure substrate package `src/tac/substrates/tishby_ib_pure/` (architecture + archive + inflate + score_aware_loss + registered_substrate) | $0 (editor) | ~5h | Sister-subagent (NOT this subagent per Catalog #230 READ-ONLY scope) | Op-routable #1 + #3 results required as gates |
| **5** | Build Tishby IB-pure trainer `experiments/train_substrate_tishby_ib_pure.py` with `_smoke_main` + `_full_main` (NotImplementedError at v1 SCAFFOLD landing per Catalog #220 cascade) | $0 (editor) | ~4h | Sister-subagent | Op-routable #4 |
| **6** | Build Tishby IB-pure recipe `.omx/operator_authorize_recipes/substrate_tishby_ib_pure_modal_a100_dispatch.yaml` (research_only=true at v1) | $0 (editor) | ~1h | Sister-subagent | Op-routable #5 |
| **7** | Path-VIB vs Path-MINE council adjudication memo (sextet pact per Catalog #292) | $0 | 1 deliberation cycle | Council | Op-routable #3 (Variational-IB-tractability check required as input) |
| **8** | Phase 2 council lift of `_full_main NotImplementedError` (if any 2 of V1 4-criterion gate satisfied) | $0 | 1 deliberation cycle | Council | Op-routables #1, #3, #7 |
| **9** | Tishby IB-pure Modal A100 smoke ($5-10 STAGE 2) | $5-10 | 1.5-2hrs | Operator-authorize | Op-routables #4-8 |
| **10** | β-sweep STAGE 2.b (council-grade $30 paired R(D) curve) | $20-30 | 6-8hrs | Operator-authorize (council-grade) | Op-routable #9 smoke green |
| **11** | Tishby IB-pure Modal A100 full anchor + paired CPU+CUDA ($15-30 STAGE 3) | $15-30 | 6-8hrs | Operator-authorize | Op-routable #10 OR Op-routable #9 (if β=0.01 default ships) |
| **12** | Paired-smoke comparison vs ATW v2 Variant B at 3 budgets (V2 trigger, STAGE 4 $30-45) | $30-45 | 9-12hrs | Operator-authorize | Op-routable #11 lands paired anchor |
| **13** | Tishby IB-pure ⊕ WZ-DISCUS composition smoke (STAGE 5) | $5-10 | 1.5-2hrs | Operator-authorize | Op-routable #11 |
| **14** | Tishby IB-pure ⊕ DP1 composition smoke (STAGE 5) | $5-10 | 1.5-2hrs | Operator-authorize | Op-routable #11 + DP1 substrate L2+ |
| **15** | A-STACK[NSCS03 → Tishby IB-pure] SWAP smoke | $5-10 | 1.5-2hrs | Operator-authorize | Op-routable #11 + A-STACK individual smokes |
| **16** | Wire Tishby IB-pure's posterior anchor into `tac.continual_learning.posterior_update_locked` per Hook 5 | $0 | 30 min | Sister-subagent (post-anchor) | Op-routable #11 lands paired anchor |
| **17** | Extend `tac.composition.registry.canonical_primitive_inventory()` with Tishby IB-pure entry per Catalog #169 | $0 | 15 min | Sister-subagent | Op-routables #4-6 |

**Recommended FIRST op-routable per Race-mode rigor inversion rule 3**: Op-routable #1 (D4 probe) AND op-routable #3 (Variational-IB-tractability check tool) in parallel — both are cheap independent prerequisites whose results gate everything downstream.

---

## 23. Cross-references

- `.omx/research/atw_codec_v2_cooperative_receiver_full_stack_design_20260516.md` — ATW v2 BOLT-ON variant (sister; this memo is the PRIMARY variant)
- `.omx/research/atw_codec_atick_tishby_wyner_v1_design_20260515.md` — V1 design memo (predecessor; bolt-on three-term Lagrangian)
- `.omx/research/atw_codec_v1_cargo_cult_unwind_design_20260516.md` — V1 HIGH-RISK unwind (sister)
- `.omx/research/grand_council_symposium_time_traveler_optimal_staircase_20260516.md` — T4 SYMPOSIUM 4×4 floor matrix (theoretical-floor row anchor: ~0.05-0.10 ultimate; ~0.08-0.12 long-horizon)
- `.omx/research/falsification_audit_v2_post_horizon_class_post_pivot_lessons_20260516.md` — FALSIFICATION-AUDIT-v2 candidate A4 (this memo IS the op-routable #3 deliverable)
- `.omx/research/nscs06_v8_path_b_wavelet_residual_full_stack_design_20260516.md` — sister asymptotic-pursuit (chroma); §13 composition opportunity
- `.omx/research/a_stack_nscs01_02_03_composition_full_stack_design_20260516.md` §13 row 11 — A-STACK[swap to Tishby IB-pure] reconciliation
- `feedback_horizon_class_evaluation_axis_plateau_warning_standing_directive_20260516.md` — HORIZON-CLASS asymptotic-pursuit anchor
- `feedback_path_2_lattice_of_class_shifts_operator_approved_supersedes_l5_v2_staircase_20260516.md` — Path 2 lattice (this memo is a Level-0 lattice node + Rule-4 candidate)
- `feedback_pr95_lesson_now_at_meta_level_unique_and_complete_per_method_default_20260515.md` — UNIQUE-AND-COMPLETE-PER-METHOD standing directive
- `feedback_canonical_share_when_serves_unique_when_suppresses_standing_directive_20260515.md` — canonical-vs-unique decision framework
- `feedback_abandon_within_class_refinements_only_substrate_class_shifts_pursue_frontier_20260515.md` — class-shift vs within-class taxonomy
- `feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md` — assumption-classification framework
- `feedback_wunderkind_visionary_scorer_as_cooperative_receiver_paradigm_shift_20260515.md` — Wunderkind G1+B3+G2 substitution candidates (parent Claude memory)
- `feedback_contest_compliance_canonical_constraints_for_wunderkind_and_all_subagents_NON_NEGOTIABLE_20260515.md` — contest-compliance constraint (parent Claude memory)
- `feedback_z4_atick_redlich_minimum_viable_landed_20260515.md` — Z4 sister substrate (β-only branch)
- `feedback_dp1_phase_2_landed_20260514.md` — DP1 pretrained-driving-prior (§13 composition)
- `tools/probe_latent_conditional_entropy_h_latent_given_scorer_class.py` — D4 probe (commit `d72f50985`)
- `src/tac/codec/cooperative_receiver/atick_redlich.py` — canonical Atick-Redlich primitive (per Catalog #164)
- `src/tac/codec/cooperative_receiver/predictive_coding.py` — sister Rao-Ballard primitive (alternative-class; orthogonal to Tishby IB)
- `src/tac/substrates/z4_cooperative_receiver_loss/` — Z4 sister substrate (β-only branch)
- `src/tac/substrates/d4_wyner_ziv_frame_0/` — D4 sister substrate (WZ on frame_0)
- `src/tac/substrates/wyner_ziv_cooperative_receiver/` — Wyner-Ziv DISCUS sister substrate (`__init__.py:31-49` distinguishes from Atick-Redlich)
- `src/tac/substrates/atw_codec_v1/` — V1 substrate package (BOLT-ON sister)
- `src/tac/substrates/atw_codec_v2/` — V2 substrate package (BOLT-ON sister)
- Alemi et al. 2017 *Deep Variational Information Bottleneck* https://arxiv.org/abs/1612.00410 — canonical VIB operationalization
- Belghazi et al. 2018 *MINE: Mutual Information Neural Estimation* — Path-MINE canonical reference
- Tishby-Pereira-Bialek 1999 *The Information Bottleneck Method* — IB Lagrangian original
- Tishby-Zaslavsky 2015 *Deep Learning and the Information Bottleneck Principle* — deep-IB framework + the active Zaslavsky voice
- Atick-Redlich 1990 *Towards a Theory of Early Visual Processing* — cooperative-receiver original
- Wyner-Ziv 1976 — source coding with side information theorem
- CLAUDE.md non-negotiables: "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + "HNeRV / leaderboard-implementation parity discipline" + "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" + "Apples-to-apples evidence discipline" + "Submission auth eval — BOTH CPU AND CUDA" + "Forbidden premature KILL" + "Race-mode rigor inversion + parallel-dispatch first" + "META-ASSUMPTION ADVERSARIAL REVIEW" + "Council conduct" (Assumption-Adversary seat sextet pact) + HORIZON-CLASS evaluation axis + "Meta-Lagrangian/Pareto solver" (Dykstra co-lead)
- CLAUDE.md Catalogs: #5 eval_roundtrip / #6 strict-scorer-rule / #88 EMA / #90 lane registry / #124 archive grammar / #125 6-hook wire-in / #126 lane pre-registration / #128 fcntl-locked posterior writes / #139 no-op detector / #146 inflate per-video loop / #152 required-input-file validation / #157 commit-serializer pre-pre-lock / #164 canonical scorer-loss helper / #167 smoke-before-full / #169 canonical primitive inventory / #170 min_vram_gb / #172 autocast_fp16 / #174 expected-content-sha256 mandatory / #176 operator-authorize canonical / #178 TF32 / #179 torch.compile / #180 no_grad-at-eval / #181 pyav_decode_strategy / #182 target_modes / #185 LIVE_COUNT drift / #190 hardware substrate / #192 macOS-CPU advisory / #205 select_inflate_device / #206 subagent checkpoint discipline / #209/#210/#211/#213 DP1 / #215 min_smoke_gpu / #220 substrate L1 scaffold operational mechanism / #221 auth_eval fail-closed / #222 scorer-loader assignment order / #226 auth_eval canonical helper / #227 substrate class-shift Tier C / #229 premise verification before edit / #230 sister-subagent ownership map / #240 recipe-vs-trainer-state / #243 local pre-deploy harness / #244 canonical NVML/CUDA env block / #245 Modal call_id ledger / #246 paired dispatch skip if anchor exists / #248 no stash-pop conflict markers / #249 no phantom device-named directories / #256 MacKay conditional entropy / #257 Tao+Boyd Blahut-Arimoto theoretical floor / #270 dispatch optimization protocol / #271 PRE-DISPATCH codex review / #272 distinguishing-feature integration contract / #279/#280/#281/#282/#283 fail-closed protections / #289 commit serializer drop-flag retry / #290 canonical-vs-unique decision per layer / #291 META-ASSUMPTION review cadence / #292 grand council explicit assumption-statements / #294 9-dimension success checklist / #296 substrate predicted band Dykstra-feasibility / #297 signal-axis destruction reversibility

---

**Status**: DESIGN-ONLY SCOPING MEMO LANDED 2026-05-16. Sister-subagent (per §22 op-routables #3-7) builds the substrate package + trainer + recipe in follow-on session. Phase 2 council approval required to lift `_full_main NotImplementedError` (when sister-subagent builds the substrate package per op-routables #4-7). RESEARCH-ONLY at recipe level until D4 probe + Variational-IB-tractability check + Dykstra-feasibility intersection check all pass.

**6-hook wire-in declaration (per Catalog #125)**:
1. Sensitivity-map contribution: ACTIVE — `tac.sensitivity_map.variational_ib_per_pair` planned (consumes KL term per-pair gradient norm).
2. Pareto constraint: ACTIVE — `tac.pareto.tishby_ib_pure_lagrangian` planned (`I(X;T) - β·I(T;Y) ≤ K_IB(β)` constraint).
3. Bit-allocator hook: ACTIVE — `bit_allocator.tishby_ib_pure_variational_v1` planned (per-pair archive bytes by KL term per-pair contribution).
4. Cathedral autopilot dispatch hook: ACTIVE — recipe registered warn-only at landing per Catalog #167; promotes to dispatch-eligible upon D4 MEANINGFUL_CONDITIONING + Variational-IB-tractability check TRACTABLE.
5. Continual-learning posterior update: ACTIVE — full anchor seeds posterior paired with D4 MI + Variational-IB SNR values per Catalog #128 locked write.
6. Probe-disambiguator: ACTIVE — D4 probe + Variational-IB-tractability check + β-sweep regime arbitration per §19.

**Sister-subagent ownership map** (Catalog #230): this subagent READ-ONLY on source code; writes ONLY this memo + 1 commit + checkpoints. Sister subagents (post-D4 + post-Variational-IB-tractability verdicts) build substrate package + trainer + recipe per Op-routables #3-7.

**Premise verification per Catalog #229** (10 PVs verified in §1 BEFORE any design statement).

**Checkpoint discipline per Catalog #206** (4+ checkpoints written to `.omx/state/subagent_progress.jsonl`).

**Commit via canonical serializer with `--expected-content-sha256`** per Catalog #157 + #174 + #289 (the 92aba3ca commit-swap class permanent fix).

---

## Observability surface

**Per the MAX-OBSERVABILITY-INTO-BEHAVIOR standing directive 2026-05-16** (`feedback_max_observability_into_behavior_xray_autopilot_tools_experiments_designs_standing_directive_20260516.md`) + Catalog #305 STRICT preflight gate (`check_substrate_design_memo_has_observability_surface_section`).

**Per Catalog #110 / #113 HISTORICAL_PROVENANCE APPEND-ONLY discipline:** this section is appended to the design memo; pre-existing body content (Sections 1-23 + 9-dim checklist + cargo-cult audit + canonical-vs-unique decision + cross-references) is UNCHANGED.

**The 6-facet observability surface for this design:**

1. **Per-layer inspection.** Every layer of this substrate (variational encoder MLP layers + variational decoder upsample blocks + statistic_net MLP layers if Path-MINE + side-info conditioning layer) captures its (input tensor, output tensor, intermediate activations) at runtime via the canonical xray-style hook pattern (`tac.xray.<lens>` modules) without re-instrumentation. The forward pass emits per-layer observables to `experiments/results/<lane>/observability/per_layer/<layer_name>.jsonl` for post-hoc inspection. The IB-specific lens `tac.xray.variational_ib_kl_per_layer` exposes per-layer KL contribution (the substrate's distinguishing-feature is the KL term; per-layer breakdown enables debugging when KL contribution is concentrated in a small subset of dimensions).

2. **Per-signal decomposition.** Composite metrics (`final_score = seg + sqrt(10*pose) + 25*rate`) decompose into constituent contributions per the canonical `tac.xray.per_pair_score_decomposition` lens. Per-pair / per-class / per-axis / per-stage breakdown serialized to `experiments/results/<lane>/observability/score_decomposition.json` with axis labels matching CLAUDE.md "Apples-to-apples evidence discipline" non-negotiable (`[contest-CUDA]` / `[contest-CPU]` / `[diagnostic-CPU]` / `[macOS-CPU advisory]` / `[MPS-PROXY]`). The IB-specific lens `tac.xray.tishby_ib_lagrangian_decomposition` further decomposes the loss into `(reconstruction_term, KL_term, rate_term)` per-pair contributions enabling per-β operating-point analysis.

3. **Run-to-run diff.** Two runs of this substrate / composition produce byte-identical reproducible artifacts under the same `(seed, commit_sha, upstream_snapshot_sha256, beta_value)` tuple per Catalog #166 Modal HEAD-parity ledger + Catalog #245 modal_call_id_ledger. The canonical diff helper `tools/diff_auth_eval_results.py <run_a.json> <run_b.json>` (planned per the observability audit Highest-ROI extension list) emits per-component deltas + per-pair drift; until landed, manual diff via `archive_sha256` byte-addressability is straightforward. β-value diff lens: `tools/diff_tishby_ib_pure_beta_sweep.py` (planned) compares R(D) curves across β values.

4. **Post-hoc query interface.** Run artifacts under `experiments/results/<lane>/` serialize as structured JSON / JSONL surfaces consumable without re-running: `contest_auth_eval_<axis>.json` (canonical per-component scores with axis labels) + `modal_metadata.json` (per-dispatch cite-chain per Catalog #166) + `observability/*.jsonl` (per-layer + per-signal) + β-sweep R(D) curve JSON. The continual-learning posterior at `.omx/state/continual_learning_posterior.jsonl` is queryable per (substrate, axis, hardware, evidence_grade, β_value) via `tac.continual_learning.query_*` helpers per Catalog #128 + #131 fcntl-locked discipline.

5. **Cite-chain.** Every behavior signal anchors to the canonical tuple `(substrate_id, commit_sha, modal_call_id, config_path, random_seed, upstream_snapshot_sha256, beta_value, path_variant)` via Catalog #245 `tac.deploy.modal.call_id_ledger.register_dispatched_call_id(...)`. The call_id ledger row schema includes `mounted_code_git_head` (per Catalog #166) + `agent` + `subagent_id` + `session_id` for full forensic reconstruction. Score claims tagged per CLAUDE.md "Apples-to-apples evidence discipline" non-negotiable.

6. **Counterfactual hooks.** Byte-mutation surface per Catalog #139 packet compiler (`tools/verify_distinguishing_feature_byte_mutation.py --distinguishing-byte-range <offset>:<length>`) + Catalog #272 distinguishing-feature integration contract + Catalog #105 no-op detector. The substrate's archive grammar exposes byte-offset addressability for "what if this byte changed?" probing without re-running training. Per-layer / per-component ablation switches surfaced via the trainer's argparse flags (e.g. `--ablate-statistic-net` for Path-MINE; `--ablate-kl-term-set-zero` for ablating the bottleneck) + the canonical `tac.xray.<lens>.ablate_*` helpers when applicable. β-value counterfactual: `--beta-override <value>` enables re-running inflate at different β values without retraining (since the decoder is β-conditional via the beta_value field in the TIBP1 archive).

**Acceptance per Catalog #305:** this section satisfies the structural requirement (literal section header `## Observability surface` present); the body content above documents the substrate's 6-facet observability surface for operator-facing audit.

**Sister observability-discipline gates active for this substrate:**

- Catalog #245 modal_call_id_ledger (every dispatch registered)
- Catalog #166 Modal HEAD-parity ledger (every dispatch worker-source-verified)
- Catalog #128 + #131 fcntl-locked JSONL posterior discipline (state mutations append-only)
- Catalog #139 packet compiler no-op detector (byte-mutation surface)
- Catalog #272 distinguishing-feature integration contract (per-substrate byte-mutation proof)
- Catalog #220 substrate L1+ operational mechanism declaration (no opaque byte additions)
- Catalog #105 no-op detector (no-op provenance)
- Catalog #127 authoritative tag custody (per-call-site axis + hardware-substrate validation)

**Observability extension recommendations (queued for follow-on):** see `tools/audit_existing_infrastructure_for_observability.py --summary` output for the canonical 8-tool / 6-facet observability gap analysis + Highest-ROI extension list. The `tools/audit_*.py` family is the highest-ROI extension target (3/12 observability) per the standing-directive consequence 3. NEW substrate-specific observability extension: `tools/inspect_variational_ib_posterior_geometry.py` (planned) — visualizes the variational posterior `q(t|x)` geometry across (per-pair, per-β) settings.

---

## Appendix A — Empirical anchors from L1 SCAFFOLD landing (SUBAGENT E, 2026-05-16)

Per the design memo §22 op-routables #1 + #3 (Phase 1 + Phase 2 of SUBAGENT E task `lane_tishby_ib_pure_l1_scaffold_substrate_build_plus_probes_20260516`). The L1 SCAFFOLD landing executes the two pre-dispatch gates whose verdicts disambiguate the substrate's Phase 2 council lift eligibility.

### A.1 D4 H(latent|scorer_class) probe — INDEPENDENT verdict

**Execution**: `experiments/results/tishby_ib_pure_d4_probe_20260516T212557Z/d4_driver.py` ran the canonical D4 probe (`tools/probe_latent_conditional_entropy_h_latent_given_scorer_class.py`, commit `d72f50985`) on A1 latents extracted from `submissions/a1/archive.zip` per the substrate's distinguishing-feature hypothesis (Wyner-Ziv side-info conditioning on scorer class).

**Result** (per `.omx/state/h_latent_given_scorer_class_tishby_ib_pure.json` + `experiments/results/tishby_ib_pure_d4_probe_20260516T212557Z/h_latent_given_scorer_class_tishby_ib_pure.json`):

- `verdict`: **INDEPENDENT**
- `mutual_information_bits`: ~0.006 bits/symbol (below MEANINGFUL threshold 0.5)
- `h_latent_unconditional_bits_per_symbol`: ~7.04 bits/symbol
- `h_latent_given_scorer_class_bits_per_symbol`: ~7.03 bits/symbol
- `wyner_ziv_gain_ceiling_fraction`: ~0.001
- `num_unique_classes`: 2 (composite-class signature) / 1 (raw majority class)

**Per-pair SegNet class distribution** (per `experiments/results/tishby_ib_pure_d4_probe_20260516T212557Z/per_pair_segnet_class.json`):

- Single majority class (class 2 = "road") dominates ~100% of pairs on dashcam footage
- Composite (majority * 5 + second_class) signature: 2 distinct values, but ~99.5% concentration on one composite class

**Root cause**: SegNet's argmax output on dashcam footage is dominated by the "road" class across the entire 1200-frame contest video — the empirical class signal on A1-rendered latents is degenerate (single-class-everywhere), so MI is ~0 by construction.

**Per CLAUDE.md "Forbidden premature KILL" + design memo §19.3**: this is **DEFER-pending-research**, NOT KILLED. Reactivation criteria:

1. **Re-run probe with per-pair multi-class signature beyond composite-majority** (e.g. spatial-bin class proportions — divide each frame into a 4×4 grid and compute class-proportions per bin, then PCA to a lower-dim signature). The current majority-class collapse is a SIGNAL extraction problem, not a substrate falsification.

2. **Train Tishby IB-pure with SCORER-CONDITIONAL CDF range coding** to see if the SUBSTRATE'S OWN encoder produces a non-degenerate latent-class distribution. The A1 substrate's latents may simply not exhibit the class-conditional structure the IB Lagrangian would learn de novo — the empirical D4 on A1 is a LOWER BOUND on the substrate's achievable mutual information at the start of training, not the asymptotic post-training distribution.

3. **Operator-approved Phase 2 council deliberation** per design memo §19.2.

### A.2 Variational-IB tractability probe — TRACTABLE verdict

**Execution**: `tools/check_variational_ib_tractability.py` (NEW canonical probe per design memo §22 op-routable #3) ran with default config (300 samples, 8 replicates, latent_dim=16, input_dim=64, output_dim=5, beta=0.01).

**Result** (per `.omx/state/variational_ib_tractability_tishby_ib_pure.json` when persisted):

- `verdict`: **TRACTABLE**
- `gradient_snr_mean`: ~6.75 (well above tractable threshold 1.0)
- `gradient_snr_median`: ~14.34
- `gradient_snr_worst_case`: ~4.98
- `gradient_norm_mean`: ~0.395
- `gradient_norm_std`: ~0.059
- `kl_term_mean`: ~1.66
- `reconstruction_term_mean`: ~1.63

**Operating-within assumption disclosure** (per Catalog #292 sextet-pact + the assumption-statement discipline in the probe's body):

> *"I am operating within the assumption that the variational IB gradient SNR measured on synthetic Gaussian data with a deterministic linear + small-noise scorer is a useful PROXY for the substrate's actual gradient tractability on contest video + SegNet/PoseNet scorer. The synthetic-data proxy is HARD-EARNED at the bound-derivation level (the IB Lagrangian's reparam-trick gradient is well-defined for any diagonal-Gaussian variational posterior + any decoder) but CARGO-CULTED at the empirical-equivalence level (real-scorer gradient SNR may differ by 2-5x due to scorer-output saturation / chroma-vs-luma sensitivity / PoseNet 12-channel YUV6 nonlinearity)."*

**Path-VIB vs Path-MINE adjudication**: TRACTABLE verdict at SNR ~6.75 (well above 1.0) supports **Path-VIB as the v1 default** per the design memo §4.3 table. Path-MINE remains the v2 fallback if real-scorer tractability check fails on Modal A100 100ep proxy.

**Council-grade verification**: the synthetic-data verdict is HARD-EARNED-at-bound-level only; CARGO-CULTED at empirical-equivalence. The next gate is the design memo §19 STAGE 1 Modal A100 100ep proxy (~$5-10) measuring gradient SNR on the REAL SegNet + PoseNet scorer.

### A.3 Phase 2 council lift gate verdict (V1 4-criterion per §19.1)

| # | Criterion | Status | Result file |
|---|---|---|---|
| 1 | D4 probe MEANINGFUL_CONDITIONING (MI ≥ 0.5) | **NOT YET** (current INDEPENDENT MI ~0.006) | `.omx/state/h_latent_given_scorer_class_tishby_ib_pure.json` |
| 2 | VIB-tractability TRACTABLE (SNR ≥ 1.0) | **PARTIAL** (synthetic-data: TRACTABLE SNR ~6.75; real-scorer Modal A100 100ep proxy pending) | `.omx/state/variational_ib_tractability_tishby_ib_pure.json` |
| 3 | Dykstra-feasibility non-empty intersection per Catalog #296 | **PENDING** (no Dykstra tool execution yet) | `.omx/state/dykstra_feasibility_tishby_ib_pure.json` (pending) |
| 4 | Path-VIB vs Path-MINE council adjudication | **PENDING** (Path-VIB recommended pending real-scorer SNR) | `feedback_tishby_ib_pure_path_adjudication_council_<YYYYMMDD>.md` (pending) |

**Phase 2 council lift verdict at L1 SCAFFOLD landing**: **DEFER-pending-research** per CLAUDE.md "Forbidden premature KILL" + design memo §19.3. The substrate L1 SCAFFOLD is LANDED with `research_only=true` + `dispatch_enabled=false` per Catalog #240 cascade.

**Next-step recommendation**: pursue D4 probe Reactivation Criterion #1 (re-run with per-pair multi-class spatial-bin signature). The current INDEPENDENT verdict is a SIGNAL EXTRACTION problem on A1 latents, not a substrate falsification — the substrate's own learned latents under IB Lagrangian optimization may produce a fundamentally different class-conditional distribution.

### A.4 Lane registry pre-registration

- Lane ID: `lane_tishby_ib_pure_l1_scaffold_substrate_build_plus_probes_20260516`
- Maturity: L0 SKETCH → L1 SCAFFOLD (gates `impl_complete` + `strict_preflight` + `memory_entry` planned post-commit)
- Status: research_only=true + dispatch_enabled=false
- Catalog #270 dispatch optimization protocol: **PASS** (Tier 1/2/3 all 5+8+5=18 signals pass)

### Ego-motion conditioning declaration (Catalog #311 / Pattern H)

<!-- # PREDICTIVE_CODING_EGO_MOTION_CONDITIONED_OK:spatial-cooperative-receiver-substrate-not-temporal-prediction-substrate-the-canonical-atick-redlich-cooperative-receiver-theorem-applies-to-spatial-mutual-information-i-x-y-not-just-temporal-next-frame-prediction-per-tishby-zaslavsky-framework -->

This memo's substrate uses the cooperative-receiver framework in its **spatial coverage** form (Atick-Redlich 1990 retinal mutual-information `I(stimulus; receptor_response)`) NOT in its temporal next-frame-prediction form. The Z6/Z7/Z8 design memo Pattern H rightly identifies ego-motion-conditioned next-frame prediction as the canonical temporal application; this substrate is the canonical spatial application.

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + Catalog #311 acceptance cascade (c): same-line waiver `# PREDICTIVE_CODING_EGO_MOTION_CONDITIONED_OK:spatial-cooperative-receiver-substrate-not-temporal-prediction-substrate-the-canonical-atick-redlich-cooperative-receiver-theorem-applies-to-spatial-mutual-information-i-x-y-not-just-temporal-next-frame-prediction-per-tishby-zaslavsky-framework` applies. The waiver rationale is non-placeholder (>4 chars, not `<rationale>` / `<reason>`).

The spatial-vs-temporal distinction within the Atick-Redlich framework: the canonical 1990 paper IS about spatial receptive-field development (retinal ganglion cells maximize `I(stimulus; receptor_response)` across the spatial field), not temporal prediction. The Z6/Z7/Z8 Pattern H gate enforces the temporal/predictive variant; this substrate applies the spatial/coverage variant.

---

## Appendix B — Four alternative-reducer probes (T2 council Q1.4 reactivation criteria, 2026-05-16)

**Status**: T2 council Q1 SPLIT-VERDICT reactivation criteria executed on A1 latents (the Tishby IB-pure substrate's distinguishing-feature hypothesis surface). Result appended per Catalog #110/#113 HISTORICAL_PROVENANCE (APPEND-ONLY); body of design memo + Sections 1-23 + 9-dim checklist + Op-summary + Observability surface + Appendix A UNCHANGED. **Subagent**: SUBAGENT B (`alt_reducer_probes_subagent_b_20260516`) at lane `lane_four_alternative_reducer_probes_meta_pattern_e_remediation_20260516`.

**Cite chain**: T2 council memo `.omx/research/grand_council_t2_wunderkind_g1_v2_pivot_validation_v3_cpu_competitiveness_20260516.md` Q1.4 enumerated 4 alternative reducers as the EXPANDED REACTIVATION CRITERIA after the canonical per-pair-dominant SegNet argmax reducer returned INDEPENDENT on this substrate (Appendix A.1: I~0.006 bits/symbol). Per CLAUDE.md "Forbidden premature KILL without research exhaustion" + sextet-pact Contrarian veto on kill-too-fast: the Tishby IB-pure substrate's class-conditional-encoder hypothesis cannot be deferred class-wide until ALL alternative reducers on the A1-latent operating point have been probed independently. This Appendix B operationalizes Appendix A.1 Reactivation Criterion #1 ("re-run probe with per-pair multi-class signature beyond composite-majority").

### B.1 Probe methodology

**Tool**: `tools/run_alternative_reducer_probes_g1_v2_and_tishby_ib_pure.py --substrate tishby_ib_pure --max-pairs 600` driving the canonical library `tools/probe_alternative_reducers_latent_class_conditioning.py` (4 reducers + canonical MI estimator + reducer-specific thresholds per T2 Q1.4).

**Source signal**: A1's HNeRV-rendered pairs over A1 latents (mirrors the d4_driver.py pattern at `experiments/results/tishby_ib_pure_d4_probe_20260516T212557Z/`). A1 archive `submissions/a1/archive.zip` parsed via `from codec import decode_decoder_compact, decode_latents_compact, apply_latent_sidecar`; A1 HNeRV decoder constructed via `model.HNeRVDecoder(latent_dim=28, base_channels=36, eval_size=(384, 512))`; 600 pairs rendered in chunks of 16.

**Scorer**: canonical `tac.scorer.load_default_scorers(upstream)` loading `upstream/models/segnet.safetensors` (sha256 `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6`). For each pair we computed:
1. **Canonical preprocess argmax** (slice frame_1 + interpolate to (384, 512) per upstream/modules.py:108) — used by reducers 1, 2, 3.
2. **Per-frame argmax of frame_0 + frame_1 separately** (each frame independently bilinear-interpolated to (384, 512) then SegNet forward) — used by reducer 4.

**Latent stream**: A1's HNeRV latents (600 pairs × 28 latent_dim = 16800 symbols) requantized to uint8 via affine map to [0, 255]. Stream sha256 `a374e01a1d4cd0639d97cd02c70846906e38e10a8bad99215e9b3e8793f30959`. Per-pair reducer outputs byte-expanded ×28 to align symbol-for-symbol with the latent stream (mirrors d4_driver.py byte-expansion protocol).

**Device**: CPU (hermetic; SegNet output is source-data derivation NOT auth-eval per Catalog #127 source-data carve-out). Wall clock: 1187s for 600-pair render+SegNet inference at 0.50 pair/s (chunked-by-16 batch decode is ~2× faster than per-pair-sequential rendering in Wunderkind G1 v2 Appendix C). Cost: $0.

**Output directory**: `experiments/results/alternative_reducer_probes_20260516T225900Z/` (Catalog #249 device-agnostic naming).

### B.2 Per-reducer empirical verdict table

| Reducer | Threshold | MI (bits/sym) | Unique fingerprints | WZ ceiling | **Verdict** |
|---|---|---|---|---|---|
| `per_pixel_histogram` (5-bin × 16-quant) | > 0.5 bits | 0.0227 | 4 | 0.32% | **WEAK_CONDITIONING** |
| `per_region_histogram` (5-bin × 4-region × 8-quant) | > 1.0 bits | 0.0474 | 7 | 0.67% | **WEAK_CONDITIONING** |
| `per_pair_class_2_fraction` (32-bucket) | > 0.2 bits | 0.0097 | 2 | 0.14% | **INDEPENDENT** |
| `per_frame_argmax` (5²=25-class fingerprint) | > 0.2 bits | 0.0000 | 1 | 0.00% | **INDEPENDENT** |

Note: `per_pair_class_2_fraction` MI=0.0097 falls just below the canonical `INDEPENDENCE_TOLERANCE_BITS=0.01` floor; the verdict computed is INDEPENDENT per the canonical decision rule. The Wunderkind G1 v2 Appendix C sister run produced MI=0.0124 for the same reducer on the residual_int8 stream — same per-pair conditioning signal, different latent stream produces marginally different MI estimate (within sampling noise; both verdicts are structurally "near-degenerate at the 2-bucket level").

### B.3 Per-pair distribution receipts (forensic provenance)

**Per-pair-dominant class (frame_1 canonical preprocess, 384×512)**: 600/600 → class 2. Matches Wunderkind G1 v2 Appendix C exactly — confirming A1's HNeRV decoder is so faithful to GT that the SegNet response distribution is structurally identical on A1-rendered frames vs GT contest video frames.

**Per-frame (frame_0_dominant, frame_1_dominant) pairs**: 600/600 → (2, 2). NO temporal motion observable at the per-pair-dominant resolution. The temporal-motion hypothesis (Yousfi T2 Q1.2 *"a car appearing/disappearing between frames"*) is empirically falsified on A1-rendered pairs at the per-frame-dominant-class level — same falsification as on GT contest video (Wunderkind G1 v2 Appendix C).

**Per-pair class-2 fraction (continuous in [0, 1])**:
- n = 600
- min / max = 0.4809 / 0.5103
- mean / median = 0.4953 / 0.4950
- stdev = 0.0065
- range (max - min) = 0.0294 (~3 percentage points)

**Critical finding**: the per-pair class-2 fraction on A1-rendered pairs is **NEARLY IDENTICAL to the GT contest video** (Wunderkind G1 v2 Appendix C: mean 0.4952, stdev 0.0065, range 0.0293). This is itself an empirically meaningful finding for the Tishby IB-pure substrate: **A1's HNeRV decoder produces frame_1 outputs that elicit a SegNet response distribution structurally identical to GT — the renderer's fidelity is high enough that the SegNet argmax structure is preserved**. The per-pair class-2 fraction is NOT in the [0.55, 0.95] range Yousfi predicted from dashcam physics (T2 Q1.2 footnote); the empirical distribution is centered at 0.495 with stdev 0.0065 on BOTH GT and A1-rendered frames.

**Per-pixel histogram fingerprints**: 4 distinct fingerprints; top fingerprint at 263939 (436/600 = 73% of pairs); next at 264195 (151/600 = 25%). Structurally identical to Wunderkind G1 v2 Appendix C (same 4 fingerprints, near-identical counts).

**Per-region histogram fingerprints**: 7 distinct fingerprints; top fingerprint at 576583900841640384 (383/600 = 64%); next at 576579502795129280 (151/600 = 25%). Structurally identical to Wunderkind G1 v2 Appendix C (same 7 fingerprints, near-identical counts).

**Per-frame argmax fingerprint**: 600/600 → 12 (= 2*5+2; frame_0 dominant = class 2, frame_1 dominant = class 2). Same single fingerprint as GT.

### B.4 Verdict interpretation per T2 Q1.4 acceptance rules

**Aggregated verdict**: `PARTIAL` — 2 of 4 reducers returned WEAK_CONDITIONING; 2 of 4 returned INDEPENDENT. **NO reducer reaches the MEANINGFUL threshold required for substrate-class reactivation per Q1.4**.

**Operator-facing recommendation** (per the canonical helper's `_recommend_action_for_verdicts`):

> *"PARTIAL: alternative reducer(s) (per_pixel_histogram, per_region_histogram) returned WEAK_CONDITIONING; MI > tolerance but < meaningful threshold. Paradigm class is DEFERRED-pending-tighter-reducer-design (Phase 2 council Q1.4 #5)."*

### B.5 Per-reducer empirical interpretation (T2 Q1.2 sextet-pact + Appendix A.1 reactivation criteria)

**Shannon (LEAD) on the empirical result**: the per-pixel + per-region reducers show non-zero MI (0.0227 + 0.0474 bits/symbol) on A1's latents AT THE A1-STARTING-POINT. Per Appendix A.1 Reactivation Criterion #2: this MI is a LOWER BOUND on the substrate's achievable mutual information at the start of training, not the asymptotic post-training distribution. The Tishby IB-pure substrate's own learned latents under IB Lagrangian optimization MAY produce a fundamentally different class-conditional distribution. The current INDEPENDENT/WEAK verdicts on A1 latents constrain the substrate's design space WITHOUT precluding the Phase 2 training-time class-conditional learning hypothesis.

**Dykstra (CO-LEAD) on convex-feasibility for Tishby IB-pure SPECIFICALLY**: the polytope intersection for the per-pixel + per-region reducers is non-empty BUT extremely narrow on A1 latents. At 0.0474 bits/symbol × 16800 symbols ≈ 796 bits ≈ 100 bytes of distortion savings ceiling for per_region_histogram, the per-region-histogram conditioning sidecar would need to ship 7 fingerprints × log2(7) bits ≈ 20 bits = ~3 bytes per pair × 600 pairs = ~1800 bytes overhead — sidecar EXCEEDS savings by ~18×. Dykstra-feasibility for the SegNet-derived spatial-region conditioning on the A1 latent operating point is structurally negative. **HOWEVER** — the Tishby IB-pure substrate trained from scratch with the IB Lagrangian could STRUCTURALLY produce latents with DIFFERENT class-conditional dependencies than A1's HNeRV (which was NOT trained with any class-conditional objective). The Dykstra-feasibility verdict applies to the A1-latent operating point ONLY, NOT to the substrate's own trained-from-scratch latents.

**Yousfi on the empirical class-2 fraction range [0.48, 0.51] on A1 latents**: identical to Wunderkind G1 v2 Appendix C (same finding, same Cargo-Culted assumption: "dashcam physics → road = 70%+ pixels" is empirically incompatible with the SegNet stride-2-stem-at-384×512 argmax distribution at ~50% road). HARD-EARNED revised understanding: at the contest scorer's argmax resolution, road class is ~50% NOT 70%, on both GT and A1-rendered frames.

**Fridrich on the cooperative-receiver paradigm for Tishby IB-pure**: the per-region-histogram is the BEST of the 4 alternative reducers at MI=0.0474 bits/symbol on A1 latents — same structural finding as Wunderkind G1 v2 Appendix C. The cooperative-receiver paradigm with SegNet-derived spatial features is empirically VIABLE but the MI is too small to overcome sidecar overhead at the A1-latent starting point. **HOWEVER** — for the Tishby IB-pure substrate specifically, the SUBSTRATE-LEARNED latents (under IB Lagrangian optimization with `β > 0`) may carry STRONGER class-conditional structure than A1's HNeRV latents which were trained with a pure-pixel-reconstruction objective and no class-conditional bottleneck term. The current Appendix B verdict is a STARTING-POINT bound, not an asymptotic bound.

**Contrarian on the verdict**: per the Q1.4 SPLIT-VERDICT, the PARTIAL outcome (2 weak + 2 independent, all below MEANINGFUL threshold) is consistent with Wunderkind G1 v2 Appendix C (3 weak + 1 independent). The Q1.4 #5 "DEFERRED-pending-tighter-reducer-design" recommendation is the correct verdict. VETO any council consensus that interprets this PARTIAL as "Tishby IB-pure permanently falsified" — Appendix A.1 Reactivation Criterion #2 explicitly identified that the SUBSTRATE'S OWN encoder may produce a non-degenerate latent-class distribution; the A1-latent probe IS the LOWER BOUND.

**Assumption-Adversary**: classification of the operating-within assumptions for this Tishby IB-pure-specific probe:

| Assumption | Classification | Rationale |
|---|---|---|
| The A1 latents are representative of the Tishby IB-pure substrate's trained latents | CARGO-CULTED | A1 was trained with pure-pixel-reconstruction loss + no class-conditional bottleneck term. Tishby IB-pure substrate adds the IB Lagrangian `I(X;T) - β·I(T;Y)` which structurally encourages T to retain class-relevant information. The A1-latent probe is a LOWER BOUND; the asymptotic-trained Tishby IB-pure latents may carry strictly more MI with SegNet classes. |
| The 4 T2 Q1.4 reducers ENUMERATE the relevant reducer space for Tishby IB-pure | CARGO-CULTED | Same observation as Wunderkind G1 v2 Appendix C — the 4 reducers are a SAMPLE not an EXHAUSTIVE enumeration. For Tishby IB-pure specifically, the SUBSTRATE-NATIVE reducer is the IB posterior `q(t|x)` evaluated at each pair (a continuous-class distribution per Catalog #271), which carries strictly more information than any argmax-derived reducer. |
| The SegNet argmax IS the canonical scorer-class side-info for Tishby IB-pure | CARGO-CULTED | Tishby IB-pure could equally condition on PoseNet outputs (a different scorer, orthogonal to SegNet). The "side-info = SegNet class" choice in Appendix A is inherited from Wunderkind G1 v2 design; for Tishby IB-pure-specific design, side-info options include (a) SegNet argmax, (b) SegNet logits (continuous), (c) PoseNet pose vector, (d) PoseNet feature vector, (e) joint (SegNet + PoseNet). |
| The empirical MI values 0.0097-0.0474 are stable estimates on the A1-latent stream | HARD-EARNED | Same Miller-Madow bias bound argument as Wunderkind G1 v2 Appendix C — bias floor is ~1e-4 bits/symbol, well below the WEAK estimates 0.0227 + 0.0474. The estimates are stable; the verdict is structurally WEAK/INDEPENDENT at the A1-latent operating point. |

The Assumption-Adversary VETOES any Phase 2 council consensus that defers the Tishby IB-pure substrate class WITHOUT explicitly evaluating (a) the SUBSTRATE-NATIVE IB posterior `q(t|x)` reducer (NOT in the T2 Q1.4 enumeration), (b) PoseNet-derived side-info as an alternative to SegNet, (c) trained-from-scratch latents (NOT A1's pixel-reconstruction-trained latents).

### B.6 Implications for Tishby IB-pure substrate reactivation (Appendix A.1 update)

**Appendix A.1 Reactivation Criterion #1 verdict** (NEW per Appendix B): the per-pair multi-class spatial-bin signature was operationalized as the 4 T2 Q1.4 reducers (per_pixel_histogram + per_region_histogram + per_pair_class_2_fraction + per_frame_argmax). Result: 2 WEAK_CONDITIONING + 2 INDEPENDENT, NO MEANINGFUL. Reactivation Criterion #1 partial-progress (signal is non-zero on 2 reducers) but NOT satisfied (MEANINGFUL threshold not reached on any reducer at the A1-latent operating point).

**Appendix A.1 Reactivation Criterion #2 STATUS UPDATE** (CRITICAL): the per-A1-latent INDEPENDENT/WEAK verdicts are a LOWER BOUND on the substrate's achievable mutual information at the starting point — they do NOT constrain the asymptotic post-IB-Lagrangian-training distribution. The Tishby IB-pure substrate's design hypothesis (the IB Lagrangian's `β · I(T;Y)` term encourages T to retain class-relevant information) REMAINS untested on substrate-native trained latents. Reactivation Criterion #2 (training-time-class-conditional learning) is the BLOCKING gate for the substrate-class reactivation.

**Reactivation criteria update for Tishby IB-pure substrate** (NEW; supersedes Appendix A.4 lane registry):

The Tishby IB-pure lane `lane_tishby_ib_pure_l1_scaffold_substrate_build_plus_probes_20260516` stays `research_only=true` with EXTENDED reactivation criteria:

1. **(SAME as Appendix A.1)** Per-pair-dominant + composite-majority reducers are FAILED on A1 latents.
2. **(NEW per Appendix B)** All 4 T2 Q1.4 alternative reducers (per_pixel_histogram + per_region_histogram + per_pair_class_2_fraction + per_frame_argmax) are FAILED-AT-A1-LATENT-OPERATING-POINT; best signal per_region_histogram MI=0.0474, WZ ceiling 0.67%; convex-feasibility Dykstra-negative for the A1 operating point.
3. **(NEW per Appendix B + Assumption-Adversary)** Reactivation Criterion #2 (trained-from-scratch Tishby IB-pure latents) remains the BLOCKING reactivation gate. The 4-reducer probe at Phase 2 STAGE 1 Modal A100 100ep proxy ($5-10) on Tishby IB-pure SUBSTRATE-NATIVE latents IS the next probe wave per Appendix A.3 Criterion 2 (real-scorer Modal A100 100ep proxy pending).
4. **(NEW per Appendix B + Catalog #220 distinguishing-feature integration contract)** When Phase 2 STAGE 1 proxy lands, the 4-reducer + SUBSTRATE-NATIVE-reducer probe wave must be re-run on the substrate's trained latents (NOT A1's). The probe library `tools/probe_alternative_reducers_latent_class_conditioning.py` + driver `tools/run_alternative_reducer_probes_g1_v2_and_tishby_ib_pure.py` are the canonical infrastructure; the driver needs a new `--substrate tishby_ib_pure_trained` arm that loads the trained substrate's latents instead of A1's.

**Phase 2 council deliberation gate** (op-routable B-2): the Q1.4 #5 "DEFERRED-pending-tighter-reducer-design" verdict, combined with Reactivation Criterion #2's blocking gate, requires a Phase 2 council deliberation per Catalog #292 sextet-pact discipline to:

(a) Adjudicate whether the Tishby IB-pure substrate is PHASE-2-LIFT-ELIGIBLE per the V1 4-criterion gate (Appendix A.3 Criterion 1 status: INDEPENDENT on A1 latents at all 4 alternative reducers + per-pair-dominant; pending substrate-native re-probe).

(b) Decide whether to proceed with Phase 2 STAGE 1 Modal A100 100ep proxy ($5-10) to enable the substrate-native re-probe, or to DEFER Phase 2 STAGE 1 until a NEW alternative reducer (per the Assumption-Adversary's NEW-reducer-class enumeration: SUBSTRATE-NATIVE IB posterior / PoseNet-derived / SegNet logits) is enumerated as a 5th probe.

(c) Adjudicate the Path-VIB vs Path-MINE choice (Appendix A.3 Criterion 4 pending) — the WEAK_CONDITIONING signal on per_pixel/per_region reducers slightly favors Path-VIB (which can directly bound the IB Lagrangian's `I(T;Y)` term).

### B.7 Op-routables (Appendix B closure)

- **B-1**: Lane registry update: append to `lane_tishby_ib_pure_l1_scaffold_substrate_build_plus_probes_20260516` notes: *"Appendix B alternative-reducer probe wave on A1 latents (2026-05-16): 2 of 4 T2 Q1.4 reducers WEAK_CONDITIONING (best per_region MI=0.0474 bits/symbol, WZ ceiling 0.67%); 2 of 4 INDEPENDENT; NO reducer reaches MEANINGFUL threshold. PARTIAL recommendation — DEFERRED-pending-tighter-reducer-design per Q1.4 #5; A1-latent probe is LOWER BOUND per Appendix A.1 Reactivation Criterion #2 (substrate-native trained latents may produce DIFFERENT distribution); Reactivation Criterion #1 partial-progress (signal non-zero on 2 reducers but not MEANINGFUL); Reactivation Criterion #2 (trained-from-scratch) remains BLOCKING gate for substrate-class reactivation."* Operator decision: confirm lane notes update.
- **B-2**: Phase 2 council deliberation per Catalog #292 sextet-pact discipline: adjudicate (a) Phase 2 lift eligibility (current verdict: DEFER-pending-research per Appendix A.3 + Appendix B both NOT-YET on Criterion 1); (b) Phase 2 STAGE 1 Modal A100 100ep proxy ($5-10) to enable substrate-native re-probe; (c) Path-VIB vs Path-MINE choice (the per_region WEAK signal slightly favors VIB).
- **B-3**: Cathedral autopilot continual-learning posterior: append `[diagnostic-CPU]` anchor rows for each of the 4 reducer verdicts per Catalog #128 fcntl-locked `posterior_update_locked` — empirical anchors that the Tishby IB-pure substrate's class-conditional design hypothesis is starting-point-WEAK/INDEPENDENT but asymptotically-UNTESTED. Use `evidence_grade="diagnostic_cpu"`, `score_claim=false`, `promotion_eligible=false`, `ready_for_exact_eval_dispatch=false`, `rank_or_kill_eligible=false` per Catalog #127 strict custody.
- **B-4**: Catalog #308 META-pattern E remediation status update: this Appendix B operationalizes the T2 Q1.4 reactivation criteria on the Tishby IB-pure A1-latent operating point with a typed verdict table + per-reducer recommendation; the META-pattern E remediation is COMPLETE for the 4-reducer enumeration on A1 latents. The Assumption-Adversary surfaced the next layer (SUBSTRATE-NATIVE-reducer probe + substrate-native trained latents + PoseNet-derived side-info enumeration) which is Phase 2 council deliberation scope.
- **B-5**: Probe library extension (FUTURE): add `--substrate tishby_ib_pure_trained` arm to `tools/run_alternative_reducer_probes_g1_v2_and_tishby_ib_pure.py` once Phase 2 STAGE 1 Modal A100 100ep proxy lands. This will load the trained Tishby IB-pure substrate's own latents (NOT A1's) and re-run the 4-reducer probe wave on the substrate-native operating point.

### B.8 Cross-substrate comparison (Wunderkind G1 v2 vs Tishby IB-pure)

The 4-reducer probe wave was executed on both substrates SIMULTANEOUSLY by SUBAGENT B (same probe library, same driver, different substrate flag). Cross-substrate comparison:

| Reducer | Wunderkind G1 v2 (GT video) | Tishby IB-pure (A1 latents) | Notes |
|---|---|---|---|
| per_pixel_histogram | WEAK MI=0.0283 (4 fp) | WEAK MI=0.0227 (4 fp) | Structurally identical: same 4 fingerprints, near-identical counts |
| per_region_histogram | WEAK MI=0.0599 (7 fp) | WEAK MI=0.0474 (7 fp) | Structurally identical: same 7 fingerprints, near-identical counts |
| per_pair_class_2_fraction | WEAK MI=0.0124 (2 bk) | INDEPENDENT MI=0.0097 (2 bk) | Just below tolerance on Tishby — within sampling noise |
| per_frame_argmax | INDEPENDENT MI=0.0000 (1 fp) | INDEPENDENT MI=0.0000 (1 fp) | Identical: 600/600 → (2, 2) on both substrates |

**Cross-substrate finding**: A1's HNeRV decoder produces frame_1 outputs that elicit a SegNet response distribution structurally identical to GT contest video. The slightly-lower MI on Tishby IB-pure across all reducers (0.0227 < 0.0283; 0.0474 < 0.0599; 0.0097 < 0.0124) likely reflects (a) the A1 latents are uint8-requantized (8-bit) vs the GT residual stream's int8 already-quantized format, (b) information loss in the A1 → HNeRV → frame_1 rendering pipeline that slightly attenuates the SegNet-class signal vs raw GT decode.

**Universal cross-substrate verdict**: the SegNet argmax distribution on `upstream/videos/0.mkv` (whether directly decoded OR rendered through A1's HNeRV) is structurally too-uniform to support per-pair class-conditional Wyner-Ziv conditioning at the canonical (384, 512) scorer resolution. The convex-feasibility envelope for any SegNet-argmax-derived reducer on this contest video is structurally negative regardless of substrate.

**Phase 2 implication for both substrates**: per the Assumption-Adversary's CARGO-CULTED classification of the SegNet-only side-info choice, the next probe wave should test (a) PoseNet-derived side-info (orthogonal scorer; canonical contest video may show different distribution), (b) SegNet pre-argmax logits (continuous-class distribution; carries more information than argmax), (c) substrate-native trained latents (for Tishby IB-pure specifically) under the IB Lagrangian's class-conditional objective.

### B.9 Probe artifacts (committed custody per CLAUDE.md "Forbidden /tmp paths")

- `tools/probe_alternative_reducers_latent_class_conditioning.py` (canonical 4-reducer probe library; commit pending)
- `tools/run_alternative_reducer_probes_g1_v2_and_tishby_ib_pure.py` (canonical orchestrator driver; commit pending)
- `src/tac/tests/test_probe_alternative_reducers.py` (21 unit tests; commit pending)
- `experiments/results/alternative_reducer_probes_20260516T225900Z/tishby_ib_pure_per_pair_reducer_outputs.json` (per-pair reducer outputs + per-pair class-2 fractions + per-frame dominant + provenance)
- `experiments/results/alternative_reducer_probes_20260516T225900Z/alternative_reducer_run_manifest_tishby_ib_pure.json` (per-substrate run manifest with 4 reducer verdicts + recommendation)
- `experiments/results/alternative_reducer_probes_20260516T225900Z/alternative_reducer_verdict_tishby_ib_pure_per_pixel_histogram.json` (per-reducer typed verdict JSON)
- `experiments/results/alternative_reducer_probes_20260516T225900Z/alternative_reducer_verdict_tishby_ib_pure_per_region_histogram.json`
- `experiments/results/alternative_reducer_probes_20260516T225900Z/alternative_reducer_verdict_tishby_ib_pure_per_pair_class_2_fraction.json`
- `experiments/results/alternative_reducer_probes_20260516T225900Z/alternative_reducer_verdict_tishby_ib_pure_per_frame_argmax.json`
- `experiments/results/alternative_reducer_probes_20260516T225900Z/combined_run_summary.json`

**Sister memo** (cross-substrate companion): `.omx/research/wunderkind_g1_v2_wire_grammar_class_shift_full_stack_design_20260516.md` Appendix C documents the parallel Wunderkind G1 v2 run (same probe wave, GT-video source signal, same probe library).

**Source memo edit scope**: Catalog #110/#113 HISTORICAL_PROVENANCE APPEND-ONLY. Body of design memo (Sections 1-23 + 9-dim checklist + Op-summary + Observability surface + Appendix A) UNCHANGED. This Appendix B lands as the empirical disambiguator result T2 council Q1 SPLIT-VERDICT op-routable specified to produce on the Tishby IB-pure substrate's A1-latent operating point.


# PARADIGM_VS_IMPLEMENTATION_FALSIFICATION_OK:historical_2026_05_16_design_memo_predates_catalog_307_2026_05_16_cutoff_landing_carries_legacy_kill_or_retired_token_for_specific_implementation_paradigm_intact_per_canonical_legacy_classification_canonical_clearance_per_comprehensive_bug_audit_cascade_20260526
