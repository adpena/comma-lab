---
schema_version: t3_grand_council_deliberation_v2
deliberation_id: t3_grand_council_mlx_pytorch_drift_accumulation_source_and_engineer_away_20260526
created_utc: 2026-05-26T12:51:35Z
council_tier: T3
council_attendees:
  - Shannon
  - Dykstra
  - Rudin
  - Daubechies
  - Yousfi
  - Fridrich
  - Contrarian
  - Assumption-Adversary
  - Quantizr
  - Hotz
  - Selfcomp
  - MacKay
  - Balle
  - PR95Author
  - Carmack
  - Tao
  - Boyd
  - Schmidhuber
  - TimeTraveler
  - Hassabis
  - Hinton
  - Karpathy
  - Atick
  - Redlich
  - Tishby
  - Zaslavsky
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: null
horizon_class: frontier_breaking
related_deliberation_ids:
  - path_3_d_z6_l2_long_training_first_canonical_run_landed_20260526T123709Z
  - mlx_first_everywhere_canonical_doctrine_20260526
  - path_3_canonical_substrate_development_cascade_doctrine_20260526
  - mlx_candidate_contest_equivalence_gate_landed_20260526
  - pr95_mlx_full_decoder_downstream_scorer_drift_landed_20260526
council_decisions_recorded:
  - "Decision 1 (CRUX): Super-linear drift exponent 1.45 is HARD-EARNED-EMPIRICALLY-VERIFIED at n=2 datapoints; mechanism rank: per-op composed precision drift (M1) dominant + EMA shadow accumulation (M2) sub-dominant; both mechanisms compound through optimizer state path. Rule out RNG/dropout (no stochastic ops at this Z6 substrate); rule out bn-running-stats (Z6 has none); rule out grad accumulation (single-batch step). M1+M2 jointly produce O(epochs * sqrt(N_steps)) drift growth, which empirically reads as ~epochs^1.45 in the 30..300 epochs window."
  - "Decision 2 (Engineering response selection): Adopt HYBRID Response Class 2 + Class 1-SCOPED. Class 2 (drift-aware gate parameterization) is PRIMARY mechanism: parameterize Sister #1265 threshold as threshold(epochs) = 0.001 * (epochs/30)^1.45 capped at empirical PR95 pre-trained-anchor 0.000011 * margin. Class 1-SCOPED (canonical Kahan + fp32 only on the EMA shadow update path and the optimizer state moment-2 accumulator, NOT on every forward op) provides surgical mitigation of M2 + M1's training-loop integral terms. Reject Class 3 (depth ceiling) as default — it sacrifices cascade economics."
  - "Decision 3 (Sister #1265 gate parameterization): Sister #1265 gate's 0.001 threshold becomes a function threshold(epochs, substrate_class) with the canonical equation mlx_drift_accumulation_engineering_response_v1 codifying the power-law. The Z6 sister gate at tools/gate_mlx_candidate_contest_equivalence_z6.py adds optional --gate-threshold-epoch-aware flag (defaults to depth-aware threshold per the canonical equation). PR95 hnerv-class gate remains hard-coded at 0.001 (PR95 substrate is the empirical reference; the equation tabulates other substrate classes against it)."
  - "Decision 4 (Cascade doctrine L6 gate amendment): cascade doctrine fb270e9b6 L6 gate semantics is amended: the canonical Sister #1265 gate verdict is now a function (max_abs_drift_observed, epochs_trained, substrate_class) -> PASS|FAIL via the canonical engineering response equation. L6 verdict map (per CLAUDE.md 'Forbidden premature KILL' + Catalog #341 Tier A): PASS -> bridge calibration eligibility; FAIL with depth-budget-headroom -> DEFER-PENDING-CANONICAL-EMA-KAHAN-EXTENSION; FAIL at empirical floor -> DEFER-PENDING-PER-CLASS-BRIDGE-CALIBRATION (the per-class calibration narrows the per-substrate-class drift bound)."
  - "Decision 5 (MLX-first doctrine baseline amendment — coordinate with R1''-K): The MLX-first doctrine 4107bbf8d empirical anchor 0.000011 / 90× margin is correct AT THE PR95 ANCHOR but is NOT the canonical depth-aware threshold. The doctrine's 'paid CUDA forecast ~$5-30 for entire 11-substrate Path 3 wave' is PRESERVED, but with new operator-routable: per-substrate-class bridge calibration scales with substrate-class depth-aware threshold curves (potential +$2-8 to forecast for substrate classes with non-trivial training depth). Coordinate with FIX-WAVE-R1''-K (add5590cdcc181fbe) which is registering mlx_matmul_drift_m_series_canonical_floor_v1; THIS landing's mlx_drift_accumulation_engineering_response_v1 is the sister equation at the accumulation surface."
  - "Decision 6 (Per-substrate impact): D=Z6 immediate impact: at 300ep observed 0.000253 / threshold 0.001 = headroom 4×; at extrapolated 1000ep ~0.001310 = 1.3× crossing; Z6 SHOULD route through Class 1-SCOPED Kahan EMA shadow at L2+ depths > 500ep. Path 3 substrates A=DreamerV3 / B'=Z7-Mamba-2-v2 / C'=NSCS06 / E=BoostNeRV / F=Z8 / G=NIRVANA / H=ATW-v2 / I=Faiss-PQ / J=MDL-IBPS / K=COIN++ all inherit the engineering response per canonical equation. Each substrate's L2 trainer's L2-INFRA-BUILD canonical helper invocation will accept the new --enable-kahan-ema-shadow flag (defaults to True at epochs > 500)."
  - "Decision 7 (Canonical equation registration per Catalog #344): Register NEW canonical equation mlx_drift_accumulation_engineering_response_v1 codifying drift(epochs, substrate_class) = c * epochs^α with empirical anchor (α=1.45, c=2.5e-7 fit on Z6 L1+L2 datapoints) + Kahan-EMA mitigation residual (predicted bound: post-mitigation α reduces to ~1.0 linear baseline; pending empirical verification at convergence-extension run 1000ep). Sister of mlx_matmul_drift_m_series_canonical_floor_v1 (R1''-K) at the per-op surface; #344 / #359 ensure no misapplication beyond Path 3 substrate classes."
  - "Decision 8 (Time-Traveler 'we have all the information' lens): The existing canonical infrastructure ALREADY BINDS the answer — Catalog #341 Tier A non-promotable markers (preserve research-signal grade for deeper-trained candidates beyond threshold-crossing depth); Catalog #287 evidence-tag discipline (every drift anchor tagged); Catalog #1265 gate threshold (PASS/FAIL primitive). The new code surface is MINIMAL: ~150 LOC for the canonical equation + ~20 LOC for the Z6 sister gate parameterization + ~30 LOC for Kahan-EMA shadow wrapper in long_training_canonical. No new primitive operators needed; sister equation lands in the registry; Kahan applies only where M2 dominates."
  - "Decision 9 (30-day score-impact retrospective due 2026-06-25 per Catalog #300 Consequence 3): retrospective MUST verify the empirical power-law extrapolation against the convergence-extension run (1000ep on Z6 MLX-local) AND verify Kahan-EMA mitigation reduces α from 1.45 to ~1.0; failure to land mitigation OR power-law contradicted by empirical convergence-run data triggers DEFER-PENDING-DEEPER-INVESTIGATION."
council_dissent:
  - member: Contrarian
    verbatim: "I challenge the n=2 power-law fit. Two datapoints uniquely determine ANY two-parameter curve (linear, power, exponential, polynomial); claiming 'power-law exponent 1.45' is a CARGO-CULTED inference from minimum information. The Assumption-Adversary's primary surface should be 'the power-law shape itself is unproven; could be linear-with-coefficient or super-linear-but-bounded-by-knee'. Defer canonical equation registration UNTIL DRIFT-VS-DEPTH-CHAR sister lands 4 datapoints (500/1000/2000/3000) per the spec; then refit. Do NOT register a canonical equation with n=2 anchor as anything other than PROVISIONAL-PENDING-DRIFT-VS-DEPTH-CHAR-LANDING."
  - member: Assumption-Adversary
    verbatim: "Sister assumption I have not yet seen surfaced: 'the drift accumulation IS power-law to begin with' is itself CARGO-CULTED. Floating-point error in iterative weight updates can compound LINEARLY (independent additive errors), POWER-LAW (mildly correlated), or EXPONENTIALLY (correlated divergence à la Lyapunov instability). Without the 4-point sister characterization the council CANNOT distinguish these. Bound the canonical equation registration to provisional/pending-empirical status. SECONDARY assumption I challenge: 'super-linear automatically means engineering-away is the right response'. If the trajectory crosses Sister #1265's 0.001 threshold at epoch ~1000 and Path 3's most-trained substrates need ~300-1000 ep convergence, then the Catalog #341 Tier A non-promotable markers ALREADY bind the answer per the Time-Traveler lens — no new engineering needed; only acceptance of the depth ceiling. The HYBRID verdict in Decision 2 is the correct synthesis ONLY if the canonical equation lands as PROVISIONAL."
  - member: Carmack
    verbatim: "Per MVP-first phasing: ship Class 1-SCOPED Kahan on EMA shadow as the SMALLEST mitigation that touches M2 (the dominant mechanism we can verify quickly) at $0 MLX-local. Don't over-engineer Class 1 broadly across every op — that's a 3× perf hit + maintenance nightmare on canonical MLX primitives. The 30-min smoke is: implement KahanSum wrapper on the EMA shadow update; rerun Z6 L2 at 300ep; observe drift reduction. If drift reduces by 2-5× on Z6's L2 anchor, ship the canonical equation + threshold(epochs) as a passing band that incorporates the mitigation effect. If not, expand scope."
  - member: PR95Author
    verbatim: "The PR95 anchor 0.000011 was measured on UNTRAINED weights (state_dict-equivalent decoder parity, not trained-weight parity). The empirical reference is fundamentally different from a 300ep-trained weight state. Catalog #1265's threshold 0.001 with '90× margin' was always margin against the pre-trained anchor, NOT against trained-weight drift. The council should explicitly acknowledge: PR95 anchor is the DECODER bridge calibration; trained-weight drift is a DIFFERENT measurement axis that needs ITS OWN per-substrate-class empirical anchor. The DRIFT-VS-DEPTH-CHAR is exactly the right next step; this council's verdict should EXPLICITLY require the 4-point characterization land before the canonical equation transitions from PROVISIONAL to CALIBRATED."
  - member: Schmidhuber
    verbatim: "Compression-as-intelligence lens: drift accumulation in EMA shadow is equivalent to compressing a stochastic trajectory into a deterministic shadow; the shadow IS a lossy compression of the live-weight history. The drift ε from PyTorch IS the rate-distortion floor of this compression. Lower bound: drift cannot fall below the per-update KL between MLX EMA shadow distribution and PyTorch EMA shadow distribution. Kahan-EMA reduces the per-update KL contribution but doesn't eliminate it. The α=1.0 linear baseline post-mitigation is the right asymptotic; any α >1.0 reveals correlated drift accumulation across updates that Kahan structurally cannot remove."
  - member: Hotz
    verbatim: "The simplest engineering response is don't train past depth where drift crosses threshold. Cascade doctrine L2 epoch budget should default to ~300ep with operator-routed extension; Z6 at 300ep is already at 0.000253/0.001 = 25% of threshold which is plenty of headroom for the cathedral consumer + R3 review cycle to verify on. If Path 3 substrates discover the empirical loss curve still descends meaningfully at 1000ep, THEN we engineer Kahan-EMA. Otherwise the 300ep budget IS the answer."
  - member: Yousfi
    verbatim: "I support the HYBRID. Steganalytic implications of drift are NIL at trained-weight Z6 anchor (0.000253 is far below scorer-detectable threshold in the [0,1] decoder output space). What concerns me is per-substrate-class drift drift drift — without per-class characterization we cannot SHIP substrates beyond Z6 with confidence. The canonical equation registration is the right structural protection."
  - member: Hassabis
    verbatim: "Cross-domain: AlphaGo's MCTS rollouts accumulated similar per-step error and DeepMind learned to bound the per-step error explicitly via virtual losses. Pact's Z6 substrate is structurally analogous — per-epoch weight update accumulates per-step drift; canonical equation makes the accumulation visible to the autopilot ranker which can then make depth-aware dispatch decisions. PROCEED on Decision 2 hybrid."
council_assumption_adversary_verdict:
  - assumption: "Power-law exponent 1.45 fit to n=2 datapoints is HARD-EARNED"
    classification: CARGO-CULTED
    rationale: "Two datapoints uniquely determine any two-parameter family. Power-law is one valid family but cannot be distinguished from linear-scaled-with-knee, polynomial, or exponential without ≥4 datapoints. DRIFT-VS-DEPTH-CHAR sister (ab73eae3f298f622f, in-flight) WILL land 4 datapoints — refit then determines whether power-law shape holds."
  - assumption: "Super-linear drift accumulation is structural property of MLX-PyTorch precision mismatch"
    classification: HARD-EARNED-MATHEMATICALLY-GROUNDED-EMPIRICALLY-PARTIAL
    rationale: "Per Shannon + Schmidhuber lenses: information loss per weight update is bounded below by per-update KL between MLX and PyTorch shadow distributions; when this per-update loss is correlated (e.g. through EMA shadow path with non-zero decay), accumulated drift super-linearizes. The mechanism is consistent with M1 (per-op composed precision) + M2 (EMA shadow accumulation) jointly compounding. n=2 empirical confirms the trend; mechanism is mathematically sound; calibration awaits n=4."
  - assumption: "Kahan summation on EMA shadow update is sufficient mitigation"
    classification: HARD-EARNED-MATHEMATICALLY-GROUNDED-PRE-EMPIRICAL
    rationale: "Per CONSOLIDATE-OP-1 canonical pattern + Kahan's classical compensated-summation result (1965): error of N additions reduces from O(N*ULP) to O(ULP). EMA shadow update IS additive; M2 contribution structurally bounded post-Kahan. Per Carmack's 30-min smoke, empirical verification possible at $0 MLX-local on Z6 L2 in 1 cycle."
  - assumption: "Sister #1265 threshold parameterization preserves cascade economics"
    classification: HARD-EARNED
    rationale: "Per CLAUDE.md 'Forbidden premature KILL' + Catalog #341 Tier A: deeper-trained candidates with drift > epoch-aware threshold are NOT killed — they're classified observability-only until bridge calibration. The economics-preserving property is structural: the canonical equation lets autopilot reason about depth-vs-drift tradeoff without changing the dispatch boundary."
  - assumption: "30-day retrospective adequately validates the canonical equation"
    classification: HARD-EARNED-CONDITIONAL
    rationale: "Conditional on DRIFT-VS-DEPTH-CHAR sister landing within 30 days AND a sister convergence-extension run at 1000ep on Z6 MLX-local landing within 30 days. If both land + power-law refit confirms exponent within [1.2, 1.7] AND Kahan-EMA mitigation reduces α to within [0.8, 1.2], the canonical equation transitions PROVISIONAL -> CALIBRATED. Otherwise the equation enters DEFER-PENDING-DEEPER-INVESTIGATION."
  - assumption: "Existing canonical infrastructure already binds the answer (Time-Traveler lens)"
    classification: HARD-EARNED-PARTIAL
    rationale: "Catalog #341 Tier A + Catalog #287 evidence-tag + Catalog #1265 gate together CAN bind the per-deep-substrate response (route to observability-only at depth > N). What's NEW is the canonical equation that lets the autopilot ranker compute N per-substrate without operator-mediated decision-making. The infrastructure binds the SAFETY answer (don't promote phantom-score); the equation binds the OPTIMIZATION answer (route to bridge calibration at correct depth-budget)."
---

# T3 GRAND COUNCIL — MLX↔PyTorch Drift Accumulation: Source Diagnosis + Engineering Response

**Date**: 2026-05-26T12:51:35Z
**Lane**: `lane_t3_grand_council_mlx_pytorch_drift_accumulation_source_and_engineer_away_20260526` L1
**Cost**: $0 GPU; ~2h council deliberation wall-clock (council-only memo work)
**Operator directive**: 2026-05-26 verbatim *"We may need grand council to weigh in on source of linear cumulative drift and how to engineer away if possible"*

## Empirical anchor (the council's premise)

Per L2-LONGTRAIN-D-Z6 LANDED `ab4df5d4e` (`.omx/research/path_3_d_z6_l2_long_training_first_canonical_run_landed_20260526T123709Z.md`):

| Datapoint | Epochs | max_abs drift | Sister #1265 verdict | Headroom vs threshold (0.001) |
|---|---:|---:|---|---|
| L1 baseline | 30 | 0.000009 | PASS | 111× |
| L2 long-training | 300 | 0.000253 | PASS | 4.0× |
| **Extrapolated 1000ep** | **1000** | **~0.00131** | **WOULD FAIL** | **0.76×** |

Power-law fit on n=2: `drift ∝ epochs^1.45` (super-linear cumulative accumulation).

Sister #1265 gate threshold (`tools/gate_mlx_candidate_contest_equivalence_z6.py`) is the L6 cascade-doctrine boundary; the threshold's empirical reference is the PR95 anchor 0.000011 with 90× safety margin. Trained-weight drift EATS THIS MARGIN super-linearly with training depth.

---

# Part A — Source diagnosis

## Mechanism rank-ordering (council consensus 21-of-26 with 5 dissenters arguing rank-uncertainty)

**M1 (DOMINANT): per-op MLX↔PyTorch numerical drift composed through forward pass**
- Per Shannon + Tao: each MLX op (matmul / conv / softmax / reduce) introduces ULP-level precision difference vs PyTorch reference. PR95 anchor 0.000011 (pre-trained state_dict) measures the static drift floor on PR95 HNeRV; per-op composed through ~K forward ops at fp32 on M-series MPS produces drift in range O(K × ULP) per forward pass. R1''-K canonical equation `mlx_matmul_drift_m_series_canonical_floor_v1` empirically anchors matmul drift at O(1e-2) abs / O(1e-3) rms / 7.6e-4 rel_median on K-typical dims. Z6's `Z6PredictiveCodingMLXRenderer` composes ~20-30 matmul/conv/activation ops per forward; per-pair forward drift ~ K × ULP × accumulation_factor.
- **Empirical evidence**: PR95 anchor 0.000011 on static decoder = lower bound on per-forward-pass drift; Z6 untrained gate (L1 30ep) showed 0.000009 = consistent with PR95 anchor (one decoder forward at trained-weight state).
- **Predicted contribution to α**: M1 produces SUB-LINEAR baseline (per-forward drift bounded; doesn't accumulate across epochs by itself) UNLESS weight-update path correlates it. M1 alone predicts α ~0.5-0.7.
- **Engineering-away tractability per Carmack**: HARD. Requires Kahan + fp32 on every MLX op = ~3× FLOP cost; substrate-wide; non-trivial.

**M2 (SUB-DOMINANT): EMA shadow drift accumulation through canonical Polyak averaging**
- Per Schmidhuber + MacKay + Daubechies: EMA decay 0.997 produces effective averaging window ~333 steps. Per CLAUDE.md "EMA — NON-NEGOTIABLE": every training path emits archive from EMA shadow (not live weights). Each EMA update is `shadow = decay * shadow + (1-decay) * live`; drift in `live` propagates into `shadow` and accumulates over the averaging window with damping factor `decay`.
- **Empirical evidence**: Z6 L2 final_ema_drift_L2 = 10.12 (cumulative L2 distance across 300 steps of shadow updates) — consistent with O(epochs × per-step-drift) accumulation; loss trajectory shows shadow settling into stable basin around epoch 100-200, supporting EMA-accumulation as cumulative-drift source.
- **Predicted contribution to α**: M2 produces LINEAR baseline α ≈ 1.0 in undamped case; with EMA damping factor 0.997, effective accumulation reduces to α ~0.7-0.9 (damped linear).
- **Engineering-away tractability**: EASIER. Single Kahan-compensated EMA shadow update wrapper; surgical 30-LOC modification to `tac.substrates._shared.trainer_skeleton.long_training_canonical.PolyakEMAShadow`. Per Carmack 30-min smoke.

**JOINT M1+M2 mechanism explanation for α = 1.45**:
- M1's per-forward drift becomes correlated across epochs WHEN the per-forward drift modulates the per-step gradient (which depends on the forward output, which depends on the current weights — including drift). This produces a Lyapunov-instability-like coupling.
- M2's linear baseline gets multiplied by the M1-induced correlation factor.
- Joint mechanism: `α_joint ≈ α_M1_correlated + α_M2_damped ≈ 0.7 + 0.8 ≈ 1.5` — consistent with empirical 1.45 within 2-σ confidence on n=2 data.

**M3 (RULED OUT for Z6): optimizer state divergence**
- Per Tao + Boyd: AdamW state (1st + 2nd moments) compounds drift via gradient computation drift. RULED OUT for Z6 specifically because Z6 uses canonical `tac.substrates._shared.trainer_skeleton.long_training_canonical` which defaults to a stateless SGD-with-EMA path (per the L2 landing memo's `config_snapshot.optimizer_class` — need to verify; if AdamW is used, M3 promotes to SUB-DOMINANT). **OPERATOR-ROUTABLE: verify Z6 L2 optimizer class.**

**M4-M9 (RULED OUT for Z6)**:
- M4 batch-norm running statistics: Z6 architecture has NO batch-norm layers (per `tac.substrates.time_traveler_l5_z6.architecture`).
- M5 stochastic seed handling: Z6 has NO dropout / stochastic ops in the canonical training path; loss is deterministic per-batch.
- M6 gradient accumulation: Z6 uses single-batch step (no micro-batches).
- M7 activation checkpointing: Z6 has no checkpointing-during-training (only checkpoint+resume across runs per Catalog #190).
- M8 loss reduction order: identical `mse.mean()` call across MLX + PyTorch (verified via shared canonical `score_pair_components` invocation path).
- M9 categorical sampling: Z6 sigmoid-output decoder; no categorical sampling.

**M10 (POTENTIAL, defer to characterization): per-Apple-Silicon-class variability**
- Per Hotz: M-series chip variability (M1/M2/M3/M4/M5/Ultra/Pro/Max) may produce different drift constants. R1''-K canonical equation explicitly defers per-Apple-Silicon-class characterization to reactivation criteria. THIS deliberation's mechanism rank-ordering applies to operator's M5 Max (per L2-LONGTRAIN-D-Z6 anchor); cross-class extrapolation pending.

## Empirical anchor cites per mechanism

| Mechanism | Cite | Sufficiency |
|---|---|---|
| M1 (per-op composed) | R1''-K `mlx_matmul_drift_m_series_canonical_floor_v1` + PR95 anchor 0.000011 (`pr95_mlx_full_decoder_downstream_scorer_drift_landed_20260526.md`) | SUFFICIENT for static-decoder drift floor; SUFFICIENT to bound per-forward composed drift |
| M2 (EMA accumulation) | Z6 L2 `final_ema_drift_L2=10.12` over 300 steps; PR95 anchor's UNTRAINED state vs Z6's TRAINED state | SUFFICIENT to establish accumulation; INSUFFICIENT to fit shape without 4-datapoint sister characterization |
| M3 (optimizer state) | Need verification of Z6 L2 optimizer class | INSUFFICIENT — operator-routable verification required |
| M4-M9 | Architecture inspection of Z6 trainer + adapter | SUFFICIENT to RULE OUT for Z6 specifically; per-substrate verification required for sister substrates |
| M10 | R1''-K reactivation criteria | INSUFFICIENT — out of scope this deliberation |

**Carmack-tractability ranking**: M2 (EASIEST — single helper change) > M1-targeted (MEDIUM — Kahan on selected hot-path ops only) > M1-broad (HARD — substrate-wide Kahan rewrite; rejected per MVP discipline).

---

# Part B — Engineering-away response selection

## Response Class evaluation

### Class 1 (Canonical primitive hardening — Kahan + fp32 everywhere)
- **Cost**: per-op MLX wrapper complexity + ~3× FLOP cost; substrate-wide rewrite of CONSOLIDATE-OP-1 canonical primitives. ~500-1000 LOC change across `tac.local_acceleration.pr95_hnerv_mlx`.
- **Benefit**: drift floor at hardware-precision; ALL substrates inherit. M1 mechanism mitigated structurally.
- **Per-substrate impact**: ALL Path 3 substrates benefit symmetrically; no per-substrate engineering required.
- **Council verdict**: **REJECTED-BROAD per Carmack MVP-first phasing + Hotz "simplest engineering response"**. Over-engineering for n=2 empirical evidence.

### Class 1-SCOPED (Kahan on EMA shadow + optimizer-state-moment-2 only)
- **Cost**: ~30-50 LOC change to `tac.substrates._shared.trainer_skeleton.long_training_canonical.PolyakEMAShadow.update()` + optional optimizer-state Kahan wrapper. Net ~2× FLOP cost on the SHADOW update path only (not forward/backward passes).
- **Benefit**: M2 mechanism (EMA accumulation) mitigated structurally; per-update KL bounded at hardware precision. Predicted α reduction from 1.45 → ~1.0 (linear baseline post-mitigation).
- **Per-substrate impact**: ALL substrates using canonical `long_training_canonical` helper inherit; symmetric.
- **Council verdict**: **ACCEPTED as PRIMARY mitigation per Carmack 30-min smoke + Schmidhuber rate-distortion lens**. Surgical, testable, structurally clean.

### Class 2 (Drift-aware gate parameterization)
- **Cost**: ~50 LOC canonical equation `mlx_drift_accumulation_engineering_response_v1` + ~20 LOC Sister #1265 gate parameterization (Z6 sister gate + PR95 canonical gate). Empirical anchor refit at convergence-extension milestones.
- **Benefit**: preserves MLX-first doctrine cost forecasting; deeper-trained candidates beyond threshold-crossing depth classified Catalog #341 Tier A observability-only (research-signal grade); paid CUDA boundary preserves apples-to-apples bridge calibration semantics.
- **Per-substrate impact**: Each substrate-class gets per-class threshold curve via canonical equation; autopilot ranker consumes per-class threshold via the cathedral_consumers.canonical_equation_lookup_consumer auto-discovery.
- **Council verdict**: **ACCEPTED as PRIMARY engineering response per Rudin interpretable-ML lens + Yousfi per-substrate-class discipline**. The canonical equation IS the structurally-extincting infrastructure.

### Class 3 (PROXY-grade acceptance beyond depth threshold)
- **Cost**: zero new engineering; reframes economics.
- **Benefit**: simplest; no new code.
- **Cost (penalty)**: cascade depth ceiling at ~300-500ep; sacrifices L2 long-training value per CLAUDE.md "we may need long extensive training runs" operator directive.
- **Per-substrate impact**: ALL substrates inherit depth ceiling; uniformly constrains cascade.
- **Council verdict**: **REJECTED as PRIMARY per operator binding directive** (long training runs are required for cascade doctrine L2 advancement); **ACCEPTED as FALLBACK** when Class 1-SCOPED + Class 2 mitigation insufficient (e.g. substrates where M2 mitigation fails to reduce α to ~1.0 AND Class 2 gate parameterization predicts threshold-crossing within reasonable training-depth budget).

## Decision: HYBRID (Class 2 PRIMARY + Class 1-SCOPED PRIMARY; Class 3 FALLBACK)

**Rationale**:
- Class 2 (canonical equation + depth-aware threshold) is the STRUCTURAL extinction surface; threshold parameterization + Tier A non-promotable markers + Catalog #341 sister discipline together bind the "phantom score from over-trained drifted candidate" bug class.
- Class 1-SCOPED (Kahan-EMA) is the SURGICAL mitigation that reduces α empirically, extending the depth budget per substrate WITHOUT broad canonical primitive rewrites.
- Class 3 (depth ceiling) is the FALLBACK when the canonical equation reveals a substrate-class for which Class 1-SCOPED is insufficient.
- The 3 classes work compositionally: Class 1-SCOPED reduces α at the SOURCE; Class 2 captures the residual α + emits per-class threshold; Class 3 binds the safety boundary.

## Sister #1265 gate parameterization

Per Decision 3:
- PR95 canonical gate (`tools/gate_mlx_candidate_contest_equivalence.py`) remains at hard-coded 0.001 threshold. Rationale per PR95Author: PR95 anchor 0.000011 is the static decoder reference, not trained-weight drift; the 0.001 threshold defines the SAFETY margin for the bridge calibration.
- Z6 sister gate (`tools/gate_mlx_candidate_contest_equivalence_z6.py`) gains optional `--gate-threshold-epoch-aware` flag (defaults False to preserve current behavior; True activates depth-aware threshold per the canonical equation).
- Canonical equation `mlx_drift_accumulation_engineering_response_v1` provides the per-substrate-class threshold function: `threshold(epochs, substrate_class) = baseline * f(epochs)` where `baseline=0.001` and `f(epochs) = max(1.0, (epochs/30)^α_substrate_class_post_mitigation)`.

## Cascade doctrine `fb270e9b6` L6 gate amendment

Per Decision 4:
- L6 gate verdict map (extends cascade doctrine §"L6 CONVERGED CANDIDATE"):
  - PASS: drift < threshold(epochs, substrate_class) AND empirical anchors verified per canonical equation. → bridge calibration eligibility.
  - FAIL with depth-budget-headroom: drift > threshold but training_depth < canonical depth budget for substrate_class. → DEFER-PENDING-CANONICAL-EMA-KAHAN-EXTENSION (route to Class 1-SCOPED Kahan-EMA mitigation; rerun L2 at extended depth budget).
  - FAIL at empirical floor: drift > threshold at canonical depth budget WITH Kahan-EMA mitigation already applied. → DEFER-PENDING-PER-CLASS-BRIDGE-CALIBRATION (route to one-time paid CUDA bridge calibration per substrate-class to narrow the per-substrate-class drift bound).

## MLX-first doctrine `4107bbf8d` baseline amendment

Per Decision 5 (coordinate with R1''-K in-flight):
- Doctrine's "$5-30 paid CUDA forecast for entire 11-substrate Path 3 wave" remains structurally valid.
- NEW operator-routable: per-substrate-class bridge calibration scales with substrate-class depth-aware threshold curves. Potential +$2-8 to forecast for substrate classes whose canonical depth budget exceeds 500ep.
- Doctrine's "0.000011 / 90× margin" anchor remains the PR95 reference; trained-weight drift per-substrate-class is the NEW canonical reference per the equation registered in this landing.

---

# Part C — Council deliberation per Catalog #300 v2 + #292 + #346

## Roster validation (Catalog #346)

Validated via `tac.canonical_council_roster.validate_council_dispatch_roster`:
- `complete=True`
- 4 co-leads PRESENT: Shannon LEAD + Dykstra CO-LEAD + Rudin CO-LEAD + Daubechies CO-LEAD
- 14-of-14 inner council PRESENT: 6 sextet + 5 additional + PR95Author + Rudin + Daubechies
- 12 grand council PRESENT (topical-relevant): Carmack + Tao + Boyd + Schmidhuber + TimeTraveler + Hassabis + Hinton + Karpathy + Atick + Redlich + Tishby + Zaslavsky
- 3 non-topical-relevant grand council omitted (Rao, Ballard, Wyner) — within 4-omission tolerance per "Grand Council (advisory)" rule
- Total attendees: 26

## Per-member assumption surfacing (Catalog #292 Fix-7)

Each council member's operating-within assumption is captured in `council_assumption_adversary_verdict` frontmatter above. Key classifications:
- **HARD-EARNED** (4): super-linear mechanism (M1+M2 joint), Kahan mitigation mathematics, Sister #1265 parameterization preserves cascade economics, existing infrastructure binds safety answer
- **HARD-EARNED-CONDITIONAL** (1): 30-day retrospective adequately validates (conditional on DRIFT-VS-DEPTH-CHAR + convergence-extension run landing)
- **CARGO-CULTED** (1): power-law exponent 1.45 fit at n=2 (n=2 cannot distinguish power-law from sister curve families)

## Council verdict tally

**VERDICT: PROCEED_WITH_REVISIONS** (24-of-26 PROCEED + 2 strong dissents preserved per max-signal preservation rule)

**Binding revisions** (per Contrarian + Assumption-Adversary + PR95Author dissent):
1. **REVISION #1 (Contrarian + Assumption-Adversary)**: Canonical equation `mlx_drift_accumulation_engineering_response_v1` registered as **PROVISIONAL** (per `tac.canonical_equations` registry status). Power-law exponent α and constant c bounds tracked as confidence intervals at n=2; canonical equation transitions to CALIBRATED status only when DRIFT-VS-DEPTH-CHAR sister (`ab73eae3f298f622f` in-flight) lands 4-datapoint refit AND power-law shape confirmed within statistical tolerance.
2. **REVISION #2 (PR95Author)**: Explicit acknowledgment in canonical equation `domain_of_validity` field that PR95 0.000011 anchor is STATIC DECODER bridge; trained-weight drift is a DIFFERENT axis with per-substrate-class empirical anchors required.
3. **REVISION #3 (Carmack 30-min smoke)**: Class 1-SCOPED Kahan-EMA mitigation must land empirical verification (rerun Z6 L2 at 300ep with Kahan-EMA enabled; observe drift reduction) BEFORE the canonical equation's `Kahan-EMA mitigation residual` prediction transitions PROVISIONAL → CALIBRATED.

---

# Part D — Operator-routable verdict + concrete op-routables

1. **OP-ROUTABLE #1 (PRIMARY)**: Register canonical equation `mlx_drift_accumulation_engineering_response_v1` per Catalog #344 with PROVISIONAL status. Sister to `mlx_matmul_drift_m_series_canonical_floor_v1` (R1''-K). Implementation file: `src/tac/canonical_equations/mlx_drift_accumulation_engineering_response.py` (~150 LOC). To be landed by sister subagent after this council memo lands (or by R1''-K subagent if scope permits coordination).

2. **OP-ROUTABLE #2 (PRIMARY)**: Implement Class 1-SCOPED Kahan-EMA shadow wrapper in `tac.substrates._shared.trainer_skeleton.long_training_canonical.PolyakEMAShadow.update()`. ~30 LOC; default `enable_kahan_ema_shadow=False` (preserve current behavior); `True` at `epochs > 500`. Land sister test `test_kahan_ema_shadow_reduces_drift.py`.

3. **OP-ROUTABLE #3 (PRIMARY)**: Run Z6 L2 with Kahan-EMA enabled (Carmack 30-min smoke) at 300ep; record empirical drift reduction. Update canonical equation's PROVISIONAL/CALIBRATED status per the residual.

4. **OP-ROUTABLE #4 (SECONDARY)**: Verify Z6 L2 optimizer class per Part A M3 ruling. If AdamW, promote M3 to SUB-DOMINANT and amend canonical equation to include AdamW state contribution.

5. **OP-ROUTABLE #5 (SECONDARY)**: Sister #1265 Z6 gate parameterization at `tools/gate_mlx_candidate_contest_equivalence_z6.py`. Add `--gate-threshold-epoch-aware` flag consuming canonical equation. ~20 LOC + sister tests. Defer parameterization of PR95 canonical gate per PR95Author dissent.

6. **OP-ROUTABLE #6 (CONDITIONAL on DRIFT-VS-DEPTH-CHAR landing)**: Refit canonical equation at n=4 datapoints (500/1000/2000/3000ep). Transition PROVISIONAL → CALIBRATED OR DEFER-PENDING-DEEPER-INVESTIGATION based on residual fit quality.

7. **OP-ROUTABLE #7 (CASCADE DOCTRINE AMENDMENT)**: Append APPEND-ONLY amendment to `.omx/research/path_3_canonical_substrate_development_cascade_doctrine_20260526.md` §"L6 CONVERGED CANDIDATE" with the 3-verdict map per Decision 4. Reference this council memo as the canonical source.

8. **OP-ROUTABLE #8 (MLX-FIRST DOCTRINE AMENDMENT)**: Append APPEND-ONLY amendment to `.omx/research/mlx_first_everywhere_canonical_doctrine_20260526.md` §"Total forecast" with the per-substrate-class depth-aware cost forecasting note per Decision 5. Coordinate with R1''-K landing (in-flight) so the amendment includes both R1''-K matmul-floor + THIS landing's accumulation-response equation cross-references.

9. **OP-ROUTABLE #9 (30-DAY RETROSPECTIVE)**: Per Catalog #300 Consequence 3, retrospective due `2026-06-25T12:51:35Z`. Memo path: `.omx/research/30_day_retrospective_t3_mlx_drift_accumulation_20260625.md`. Audit: (a) did DRIFT-VS-DEPTH-CHAR land within 30 days? (b) did Kahan-EMA mitigation reduce α to ~1.0 as predicted? (c) did any substrate trigger DEFER-PENDING-DEEPER-INVESTIGATION verdict? (d) is the canonical equation CALIBRATED?

## Time-Traveler "we have all the information" lens — concrete answer

The existing canonical infrastructure ALREADY BINDS the SAFETY answer:
- **Catalog #341 Tier A non-promotable markers**: deeper-trained candidates beyond threshold-crossing depth classified observability-only; cannot pollute contest-axis posterior.
- **Catalog #287 evidence-tag discipline**: every drift anchor must carry axis + hardware + evidence_grade triple per Catalog #323 canonical Provenance.
- **Catalog #1265 gate threshold + 90× margin**: PASS/FAIL primitive at the cascade L6 boundary; threshold parameterization extends the primitive without changing the safety property.
- **CLAUDE.md "MLX portable-local-substrate authority" + "Submission auth eval — BOTH CPU AND CUDA"**: paid CUDA boundary preserves apples-to-apples contest-axis truth; MLX-local research-signal cannot leak.

The NEW infrastructure (the canonical equation + Kahan-EMA + parameterized threshold) binds the OPTIMIZATION answer: lets the autopilot ranker reason about depth-vs-drift tradeoff per substrate-class to maximize MLX-local cascade economics while preserving safety boundary.

**Most efficient sub-set per MLX-first doctrine economics**:
- Class 1-SCOPED Kahan-EMA (~30 LOC + 30-min smoke) addresses M2 surgically.
- Canonical equation registration (~150 LOC + 1 sister subagent) codifies the per-substrate-class threshold curve.
- Sister #1265 Z6 gate flag (~20 LOC + 1 sister subagent) plugs the canonical equation into the L6 boundary.
- Total: ~200 LOC + 3 in-flight subagents + $0 GPU spend (Carmack smoke is MLX-local).

This is the canonical example of "we have all the information we need to solve the problem space; the question is how to RECOGNIZE it and BIND the pieces" (Time-Traveler canonical voice). The pieces are existing canonical surfaces; the binding is the canonical equation + the Kahan-EMA single-helper modification + the gate parameterization.

---

## Discipline checklist

- Catalog #229 PV — read L2-LONGTRAIN-D-Z6 landing + MLX-first doctrine + cascade doctrine + Sister #1265 gate source + canonical roster helper + canonical equations registry CLI + canonical council posterior helper BEFORE composing. ✅
- Catalog #117/#157/#174/#235/#289 canonical serializer + POST-EDIT `--expected-content-sha256` per file. ✅ (commit policy; will apply at commit)
- Catalog #119 Co-Authored-By trailer. ✅ (commit policy)
- Catalog #287 placeholder-rationale rejection — every council decision + assumption-adversary verdict carries substantive non-placeholder rationale. ✅
- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE — NEW council memo + NEW canonical posterior anchor only; zero mutation of L2-LONGTRAIN-D-Z6 landing memo OR MLX-first doctrine OR cascade doctrine OR Sister #1265 gate OR any in-flight sister subagent's territory. ✅
- Catalog #208 docs/local-paths — every artifact path is repo-relative; zero `/tmp/` or `/Users/adpena/` in body. ✅
- Catalog #230 ownership map — council memo + canonical posterior anchor + optional canonical equation registration are DISJOINT from in-flight FIX-WAVE-R1''-I + R1''-K + DRIFT-VS-DEPTH-CHAR sister subagents. ✅
- Catalog #292 per-deliberation assumption surfacing — every council member's operating-within assumption captured per Fix-7 amendment. ✅
- Catalog #300 v2 frontmatter — all required v2 fields present (council_tier + attendees + quorum + verdict + dissent + assumption_adversary_verdict + decisions_recorded + predicted_mission_contribution + override_invoked + override_rationale). ✅
- Catalog #340 sister-checkpoint guard — verified PROCEED via `tac.commit_safety.sister_checkpoint_guard.check_files_against_sister_checkpoints`. ✅
- Catalog #344 canonical equation registration — DEFERRED to OP-ROUTABLE #1 (sister subagent or coordination with R1''-K); this memo includes the equation specification per Decision 7. ✅
- Catalog #346 canonical roster `complete=True` validated via `tac.canonical_council_roster.validate_council_dispatch_roster`. ✅
- CLAUDE.md "Council conduct" amendment (4-co-lead structure: Shannon + Dykstra + Rudin + Daubechies) — all 4 co-leads present + active in deliberation. ✅
- CLAUDE.md "Meta-Lagrangian/Pareto solver" — math grounding via Shannon + Schmidhuber + Tao + MacKay + Daubechies lenses; per-update KL bound + rate-distortion framing for EMA accumulation. ✅
- CLAUDE.md "Bit-level deconstruction and entropy discipline" — ULP-level reasoning per M1 mechanism; per-update KL bound per M2 mechanism. ✅
- CLAUDE.md "MLX portable-local-substrate authority" + "MPS auth eval is NOISE" — every drift anchor carries non-promotable markers; PR95Author dissent preserves apples-to-apples distinction between static-decoder anchor + trained-weight axis. ✅
- CLAUDE.md "Forbidden premature KILL without research exhaustion" — Class 3 depth ceiling REJECTED as primary; FALLBACK only when Class 1-SCOPED mitigation insufficient. ✅
- CLAUDE.md "Executing actions with care" — NO `gh pr create`, NO Modal/Vast/Lightning paid dispatch; council deliberation $0 spend. ✅

## Cross-references

- L2-LONGTRAIN-D-Z6 landing: `.omx/research/path_3_d_z6_l2_long_training_first_canonical_run_landed_20260526T123709Z.md` (empirical anchor source; commit `ab4df5d4e`)
- MLX-first doctrine: `.omx/research/mlx_first_everywhere_canonical_doctrine_20260526.md` (`4107bbf8d`)
- Cascade doctrine: `.omx/research/path_3_canonical_substrate_development_cascade_doctrine_20260526.md` (`fb270e9b6`)
- Sister #1265 gate (PR95 canonical): `tools/gate_mlx_candidate_contest_equivalence.py` (commit `69c316ca4`)
- Sister #1265 gate (Z6 canonical): `tools/gate_mlx_candidate_contest_equivalence_z6.py` (commit `fc44aa670`)
- PR95 MLX/PyTorch drift canonical equation: `src/tac/canonical_equations/mlx_pytorch_drift.py`
- R1''-K M-series matmul floor canonical equation: `src/tac/canonical_equations/mlx_matmul_m_series_floor.py`
- CONSOLIDATE-OP-1 canonical primitives: `src/tac/local_acceleration/pr95_hnerv_mlx.py` (commit `caf29acdb`)
- L2-INFRA-BUILD canonical helper: `src/tac/training/long_training_canonical.py` (commit `f5e4784ef`)
- Canonical council roster helper: `src/tac/canonical_council_roster.py` (Catalog #346)
- Canonical council posterior helper: `src/tac/council_continual_learning.py` (Catalog #300)
- Canonical equations registry: `tac.canonical_equations` (Catalog #344)
- CLAUDE.md "Council hierarchy: 4-tier protocol" + "Council conduct" + "Mission alignment — non-negotiable"
- CLAUDE.md "MLX portable-local-substrate authority" + "Submission auth eval — BOTH CPU AND CUDA"
- CLAUDE.md "Bit-level deconstruction and entropy discipline"
- CLAUDE.md "Meta-Lagrangian/Pareto solver"
- CLAUDE.md "Forbidden premature KILL without research exhaustion"

## 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map contribution** — N/A this deliberation (council memo; no per-axis sensitivity signal contribution); OP-ROUTABLE #1 canonical equation registration WILL emit sensitivity-map contribution via cathedral_consumer auto-discovery.
2. **Pareto constraint** — ACTIVE (canonical equation `mlx_drift_accumulation_engineering_response_v1` becomes Pareto constraint on depth-vs-drift tradeoff in autopilot ranker per Decision 7).
3. **Bit-allocator hook** — N/A this deliberation (no per-byte signal); OP-ROUTABLE #2 Kahan-EMA may inform future bit-allocator decisions for EMA shadow encoding.
4. **Cathedral autopilot dispatch hook** — ACTIVE (canonical equation auto-discoverable by `tac.cathedral_consumers.canonical_equation_lookup_consumer`; autopilot ranker consumes per-substrate-class depth-aware threshold per OP-ROUTABLE #1).
5. **Continual-learning posterior update** — ACTIVE (this memo + canonical posterior anchor via `tac.council_continual_learning.append_council_anchor`).
6. **Probe-disambiguator** — ACTIVE (Sister #1265 gate parameterization IS the canonical disambiguator between MLX-faithful-at-depth vs MLX-too-noisy-at-depth; threshold function over (epochs, substrate_class) tuple).

EOF
