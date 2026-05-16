# ATW Codec V2 (Atick-Tishby-Wyner) — comprehensive full-stack design memo

**Date**: 2026-05-16
**Lane**: `lane_atw_codec_v2_cooperative_receiver_full_stack_design_20260516`
**Subagent**: ATW-CODEC-V2-FULL-STACK-DESIGN-20260516
**Predecessor**: `.omx/research/atw_codec_atick_tishby_wyner_v1_design_20260515.md` (V1 design) + `.omx/research/atw_codec_v1_cargo_cult_unwind_design_20260516.md` (V1 HIGH-RISK unwind, commit `ae6986c04`)
**L5 v2 staircase position**: Step B2 (scorer-relationship class-shift via cooperative-receiver framing) — per `.omx/research/grand_council_symposium_time_traveler_optimal_staircase_20260516.md` line 63 ordering `A1 → A2 → A3 → A-STACK → B1 → B2 → D → F-asymptote` (Path 2 lattice rewrite preferred)
**D4 probe enabler**: `tools/probe_latent_conditional_entropy_h_latent_given_scorer_class.py` landed today via commit `d72f50985`
**Operating mode**: UNIQUE-AND-COMPLETE-PER-METHOD (per the standing directive `feedback_pr95_lesson_now_at_meta_level_unique_and_complete_per_method_default_20260515.md` + 2026-05-15 retrospective: *"this has been a huge problem since the beginning of the competition"*)
**Status at landing**: DESIGN-ONLY (RESEARCH-ONLY at the recipe level until D4 probe returns `MEANINGFUL_CONDITIONING` + Dykstra-feasibility check + Phase 2 council approval per existing 4-criterion lift gate)

---

## 1. Frontmatter — premise verification + lane registry + sister-subagent map

### Premise verifications (Catalog #229; 9 PVs verified BEFORE any design statement)

- **PV-1** ATW V1 source exists at `experiments/train_substrate_atw_codec_v1.py` (361 LOC; `_full_main` raises `NotImplementedError` at line 335 per Catalog #220 substrate-engineering pre-build cascade) + `src/tac/substrates/atw_codec_v1/{architecture.py (21.0K), archive.py (20.4K), inflate.py (7.6K), score_aware_loss.py (11.4K), registered_substrate.py (3.0K)}`. Smoke path validated at `_smoke_main` (lines 197-319; synthetic-data forward + ATW1 archive roundtrip; no scorer load).
- **PV-2** V1 recipe at `.omx/operator_authorize_recipes/substrate_atw_codec_v1_modal_a100_dispatch.yaml` declares `research_only: true` + `dispatch_enabled: false` + `min_smoke_gpu: A100` (line 48; sister Catalog #215 — Tishby IB κ_IB > 0 requires A100 memory; cannot smoke on T4). Recipe `predicted_delta` correctly rewrites to `NULL pending H(latent|scorer_class) probe` per V1 unwind Catalog #296.
- **PV-3** Canonical Atick-Redlich primitive exists at `src/tac/codec/cooperative_receiver/atick_redlich.py` (270 LOC) with `AtickRedlichWeights` + `cooperative_receiver_loss(rgb_0, rgb_1, gt_rgb_0, gt_rgb_1, *, seg_scorer, pose_scorer, weights)` returning `CooperativeReceiverOutput(cooperative_loss, seg_term, pose_term, pose_sqrt)`. Delegates internally to canonical `score_pair_components` per Catalog #164. Package `__init__.py` re-exports + ships sister `predictive_coding.py` (Rao-Ballard).
- **PV-4** D4 H(latent|scorer_class) probe exists at `tools/probe_latent_conditional_entropy_h_latent_given_scorer_class.py` (commit `d72f50985`; 312 LOC). Three-verdict taxonomy `MEANINGFUL_CONDITIONING / WEAK_CONDITIONING / INDEPENDENT` keyed off MI threshold default `0.5 bits/symbol` per audit foundation §5. Emits `.omx/state/h_latent_given_scorer_class_<substrate_id>.json` with `evidence_grade=diagnostic_cpu / score_claim=false / axis_label=[diagnostic-CPU; H(latent|scorer_class) probe]` per Catalog #192 + #221 fail-closed discipline.
- **PV-5** Wunderkind Visionary research lives at `/Users/adpena/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_wunderkind_visionary_scorer_as_cooperative_receiver_paradigm_shift_20260515.md` (parent-session memory). Top-3 SUBSTITUTION-1:1 candidates: **G1** (scorer-softmax-as-hyperprior-gating; 1KB distill head replaces 50KB Ballé hyperprior; predicted ΔS −0.005 to −0.015 rate-axis; contest-legal); **E1** (Z4 trainer wires canonical Atick-Redlich primitive instead of hand-roll; ΔS −0.001 to −0.003); **G2-PARTIAL** (posterior-matching codec with ≤50KB precomputed scorer-CDF table shipped as side-info; ΔS −0.10 to +0.05 high-variance council-grade). **B3** (scorer-conditional CDF for entropy coder; precomputed-table form; ΔS −0.003 to −0.010 rate-axis; contest-legal). **G3** (universal stateless decoder via shared scorer-CDF; theoretical floor).
- **PV-6** L5 v2 staircase Step B2 corresponds to "scorer-relationship class-shift via cooperative-receiver framing" per `.omx/research/grand_council_symposium_time_traveler_optimal_staircase_20260516.md` line 63 sequential ordering. The Path 2 lattice rewrite (cargo-cult-classified sequential ordering) reorders into a parallel-dispatch friendly fan-out where ATW v2 is a parallel arm of B-cluster (scorer-relationship class-shifts) NOT a sequential successor of NSCS03.
- **PV-7** Existing `lane_atw_codec_design_v1_20260515` registered in `.omx/state/lane_registry.json` at Level 1 (impl_complete + memory_entry per CARGO-CULT-UNWIND landing). V2 lane MUST pre-register at Level 0 SKETCH per Catalog #126 lifecycle discipline (sister to V1; not a renaming).
- **PV-8** Contest-compliance constraint per parent-session memo `feedback_contest_compliance_canonical_constraints_for_wunderkind_and_all_subagents_NON_NEGOTIABLE_20260515.md`: SegNet + PoseNet weights (~73MB) at inflate-time = rate-DOMINATED catastrophic. Any V2 design MUST be COMPRESS-side use of scorer (FREE) + ship a ≤2KB distillation/CDF/sampler-side-info shipped IN archive that the decoder reads. NO inflate-time scorer load.
- **PV-9** Atick-Redlich 1990 + Tishby-Pereira-Bialek 1999 + Wyner-Ziv 1976 are HARD-EARNED citation anchors. The V1 design memo §1 (lines 14-101) correctly composes the three into one Lagrangian; the math IS tractable per Boyd's convex-feasibility lens. CC-3 (Wyner-Ziv 30-50% conditional-entropy reduction hypothesis) is the empirical-equivalence axis the D4 probe disambiguates.

### Sister-subagent ownership map (Catalog #230)

This subagent is **READ-ONLY** on source code (`src/tac/`, `experiments/`, `submissions/`, `tools/`, `.omx/operator_authorize_recipes/`). Writes ONLY to:

- `.omx/research/atw_codec_v2_cooperative_receiver_full_stack_design_20260516.md` (this memo)
- `.omx/state/subagent_progress.jsonl` (canonical checkpoint store per Catalog #206)
- 1 commit via canonical serializer with `--expected-content-sha256` per Catalog #157 + #174 + #289 (the 92aba3ca commit-swap class permanent fix)

No sister-subagent collision expected because no sister is currently editing the ATW v2 surface. Sister A-STACK design memo (commit `5d0b46085`) lists ATW codec at §13 row 11 as "ORTHOGONAL (ATW = entropy-coding via cooperative receiver; A-STACK = entropy-coding via Ballé hyperprior); REDUNDANT-WITH-NSCS03; floor at -0.005; ATW REPLACES NSCS03 in the A-STACK; Alternative composition: A-STACK[NSCS03 → ATW] swap" — that observation MUST be reconciled in §13 below.

### Operating-within assumption-statement (Catalog #292 / Assumption-Adversary seat)

The assumption I am operating within for this V2 design memo: *"The V1 cargo-cult unwind correctly classified CC-1 LOW + CC-2 HIGH + CC-3 HIGH; V2 inherits the unwound design as the structural base, ADDS the D4 H(latent|scorer_class) probe-disambiguator as a $3-5 PRE-DISPATCH-GATE that empirically disambiguates the cooperative-receiver hypothesis BEFORE paid CUDA dispatch, AND productionizes Wunderkind G1+B3+G2-PARTIAL as the three SUBSTITUTION-1:1 V2-distinguishing primitives. The predicted band is NULL until D4 probe returns MEANINGFUL_CONDITIONING + Dykstra-feasibility check; if it does, the band can be derived from first principles via Tishby IB lower bound and Wyner-Ziv side-info bit-savings, NOT extrapolated from sister substrate empirical anchors."*

HARD-EARNED basis: V1 unwind explicitly named the cargo-cult class (CC-2 = hand-waved predicted band; CC-3 = unmeasured WZ gain estimate). V2's contribution is the EMPIRICAL DISAMBIGUATION mechanism that the V1 design already pointed at (V1 §5 reactivation criterion #3 = "WZ side-info bit-savings empirically measured on A1 latents ≥ 20% per-pair latent rate ($0, 1 hour CPU)") — but never executed because no canonical probe existed. The D4 probe IS that canonical mechanism, landed today.

The Assumption-Adversary seat (sextet pact per Catalog #292) would challenge: *"Is the 'three-knob composition' itself a cargo-cult? Perhaps a single-knob design (just Wyner-Ziv side-info residual; drop κ_IB; drop λ_pixel) would be the substrate-optimal engineering per UNIQUE-AND-COMPLETE-PER-METHOD — i.e., binding ONE clean cooperative-receiver mechanism rather than three composable mechanisms."* — answer: §4 below ships a TWO-VARIANT v2 design where Variant A is the canonical three-knob V1-inherited form (preserves probe-disambiguator regime-sweep arbitration) and Variant B is the single-knob WZ-only form (tighter substrate engineering per HNeRV parity L7 bind-all-ingredients). Council adjudication (§19) decides which variant ships first based on D4 probe verdict empirical evidence.

### Lane registry pre-registration (Catalog #126)

To be claimed in same commit batch:
```bash
.venv/bin/python tools/lane_maturity.py add-lane \
    lane_atw_codec_v2_cooperative_receiver_full_stack_design_20260516 \
    --name "ATW Codec V2 (Atick-Tishby-Wyner cooperative-receiver full-stack)" \
    --phase 2
```

---

## 2. Executive summary

ATW V2 binds Atick-Redlich + Tishby IB + Wyner-Ziv into ONE substrate-engineering scaffold whose distinguishing-feature (per Catalog #272) is the **scorer-class-conditional latent residual codec** — the encoder uses the cooperative receiver's softmax class assignments as side-information that the decoder reconstructs WITHOUT loading the scorer at inflate time.

**Three concrete improvements over V1**:

1. **D4 H(latent|scorer_class) probe as PRE-DISPATCH GATE** (was V1 reactivation criterion #3; now V2 mandatory $3-5 CPU smoke BEFORE any paid Modal dispatch). Verdict taxonomy `MEANINGFUL_CONDITIONING / WEAK_CONDITIONING / INDEPENDENT` directly arbitrates whether to dispatch (MEANINGFUL → smoke; WEAK → revise predicted band down; INDEPENDENT → DEFER-pending-research with named alternative-hypothesis). Closes V1 CC-3 empirically.
2. **Wunderkind G1 + B3 + G2-PARTIAL productionized as v2-distinguishing**: G1 = 1KB SegNet-class-distill head replacing Ballé-style 50KB hyperprior; B3 = scorer-conditional CDF table (precomputed at compress, shipped as ~2KB side-info) for range-coding latent symbols; G2-PARTIAL = posterior-matching-codec sampler with ≤50KB scorer-CDF table for stateless decoder refinement. Each maps 1:1 to ATW Lagrangian terms.
3. **Two-variant ship-decision per Assumption-Adversary**: Variant A (V1-inherited three-knob κ_IB / λ_WZ / λ_pixel) preserves probe-disambiguator regime sweep; Variant B (single-knob WZ-only) is tighter substrate-optimal engineering. Council adjudication binds based on D4 probe verdict.

**L5 v2 staircase position**: ATW V2 is the canonical **Step B2** (scorer-relationship class-shift via cooperative-receiver framing) per the staircase ordering `A1 → A2 → A3 → A-STACK → B1 → B2 → D → F-asymptote`. In the Path 2 lattice rewrite (preferred), B-cluster substrates (B1 = NSCS01 scorer-slice exploitation; B2 = ATW v2; B3 = Wunderkind G1/B3/G2-PARTIAL) fan out IN PARALLEL rather than sequentially.

**Predicted ΔS band**: `NULL pending D4 probe verdict + Dykstra-feasibility check` [prediction]. Conditional revisions:
- **If D4 verdict = MEANINGFUL_CONDITIONING (MI ≥ 0.5 bits/symbol)**: predicted band `[-0.005, -0.015]` [prediction; first-principles-bound via Tishby IB lower bound + Wyner-Ziv side-info savings × A1 rate-axis 0.20]
- **If D4 verdict = WEAK_CONDITIONING (0.01 ≤ MI < 0.5)**: revise to `[-0.002, -0.005]` [prediction; downscaled by MI / 0.5 ratio]
- **If D4 verdict = INDEPENDENT (MI < 0.01)**: DEFER-pending-research per CLAUDE.md "Forbidden premature KILL"; named alternative-hypothesis = G2-PARTIAL posterior-matching (decoupled from scorer-class-conditioning hypothesis)

**Stack-of-stacks composition opportunities** (§13):
- **A-STACK ⊕ ATW v2** (§13 detailed): operator-routable swap `A-STACK[NSCS03 → ATW v2]` since ATW v2's entropy-coding term subsumes NSCS03's Ballé hyperprior per A-STACK §13 row 11. Predicted composition band: NULL pending probe.
- **ATW v2 ⊕ NSCS06 v8 Path B chroma** (FRESH per assumptions-challenge-audit matrix): ATW v2 operates on Y-channel + scorer-conditional latent; NSCS06 v8 Path B reconstructs chroma via wavelet residual. ORTHOGONAL by axis (luma scorer vs chroma reconstruction). Predicted small additive ~0.005.
- **ATW v2 ⊕ DP1** (Catalog #209/#210/#211/#213): DP1 pretrained-driving-prior codebook initializes ATW v2's encoder; the SegNet-class assignments used by D4 probe match DP1's frame-iterator class index. STRONG_STACK per shared scorer-prior framework. Predicted small additive ~0.005.

**Cost estimate** (§20): D4 probe $3-5 (CPU smoke) → Modal A100 smoke $5-15 (conditioned on MEANINGFUL_CONDITIONING) → Modal A100 paired CPU+CUDA full anchor $10-30 (conditioned on smoke green). Total envelope: $18-50 inclusive of probe. CONDITIONALLY DISPATCHABLE if D4 verdict gates pass.

**Verdict on whether ATW V2 should fire NOW**: NO. The D4 probe MUST run first ($3-5; cheapest signal in the stack). Per CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first" Rule 3: the loop tick should be (1) fire D4 probe on A1 latents (cheap); (2) parallel-fan G1 + B3 + G2-PARTIAL design memos per Wunderkind candidate (cheap); (3) ONLY THEN fire ATW V2 smoke (Variant A or B per D4 verdict).

---

## 3. Composition rationale + per-substrate / per-paper orthogonality verification

### Why these three papers compose into ONE Lagrangian

The three foundational information-theoretic frameworks ATW v2 binds are mathematically distinct but operationally complementary on the contest substrate:

- **Atick-Redlich (1990)** — *"Towards a Theory of Early Visual Processing"* — establishes that for a KNOWN downstream receiver `R`, the optimal encoder maximizes `MI(B; R(B))`, NOT generic reconstruction fidelity `MI(B; X)`. On our substrate, `R = SegNet + PoseNet` is published in `upstream/modules.py` — this is the textbook cooperative-receiver setup. The canonical primitive `tac.codec.cooperative_receiver.atick_redlich.cooperative_receiver_loss` already implements this as `β_seg · d_seg + γ_pose · sqrt(d_pose)`.
- **Tishby-Pereira-Bialek (1999)** — Information Bottleneck Lagrangian `L_IB = I(X; T) - β · I(T; Y)` where `T` = codec representation and `Y` = scorer outputs. The β knob is the rate-distortion tradeoff. When the cooperative receiver is KNOWN, `p(y|t)` is computable in closed form modulo numerical tractability — Tao+Boyd Blahut-Arimoto computation runs in ~2hrs CPU per V1 §5 reactivation criterion #2.
- **Wyner-Ziv (1976)** — Source coding with side information at decoder: `R_WZ(D) = R_{X|S}(D) ≤ R(D)`. When the decoder has access to `S = scorer outputs` (computable at compress from SegNet on GT), the encoder need transmit only `R_{X|S}(D)` bits. The gap `H(Y|S) - H(Y|X,S)` is the bit-savings the encoder doesn't need to transmit.

**Orthogonality verification at the operational axis**:

| Axis | Atick-Redlich | Tishby IB | Wyner-Ziv |
|---|---|---|---|
| **Operates at** | Loss landscape (training-time) | Encoder-decoder mutual information (training-time) | Source bytes (compress-time) |
| **Mathematical surface** | `β·d_seg + γ·sqrt(d_pose)` | `I(X;T) - β·I(T;Y)` | `R_{X|S}(D)` |
| **Contest-compliance** | ✓ compress-only (training loss) | ✓ compress-only (encoder objective) | ✓ compress-only IF S is shipped as side-info ≤2KB; ✗ if S loaded at inflate |
| **Empirical disambiguation** | Confirmed by Z4 (β-only branch) | Tao+Boyd Blahut-Arimoto floor ($0, 2h CPU) | D4 probe (this is the new V2 mechanism!) |

The three are **operationally orthogonal** when composed correctly: Atick-Redlich shapes the loss; Tishby IB shapes the encoder distribution; Wyner-Ziv shapes the archive bytes. The V1 design memo §1 (line 73-101) correctly notes that knob-zero ablations recover each paper's isolated form — providing a built-in probe-disambiguator regime sweep.

### Per-V2 distinguishing primitives (Wunderkind G1 + B3 + G2-PARTIAL): orthogonality verification

| Primitive | Wunderkind cluster | Mechanism | Bytes shipped | ΔS prediction | Orthogonal to which V2 component |
|---|---|---|---|---|---|
| **G1** scorer-class distill head | G (WUNDERKIND-NEW) | 1KB MLP `g(decoded_latent_per_pixel) → 5-way SegNet softmax`; replaces Ballé-style 50KB hyperprior | +1KB head; saves ~49KB hyperprior | −0.005 to −0.015 rate-axis | Atick-Redlich (loss-side); Tishby IB (encoder-side); WZ-side complementary |
| **B3** scorer-conditional CDF | B (decoder-side hint) | Range-encode latents conditional on scorer's softmax distribution per pixel; CDF table precomputed at compress, shipped ≤2KB | +2KB CDF table; reduces per-latent rate | −0.003 to −0.010 rate-axis | Atick-Redlich (loss-side); WZ residual term (Lagrangian) |
| **G2-PARTIAL** posterior-matching codec | G (WUNDERKIND-NEW) | Stateless decoder = scorer-conditional Langevin sampler `f(seed, scorer_class_map) → frame`; ≤50KB CDF table | 4-byte seed + 12KB class-map + 50KB residual codebook | −0.10 to +0.05 (high variance) | Replaces ENTIRE learned decoder; orthogonal to Atick-Redlich loss surface but DISPLACES Tishby IB encoder |

The three primitives are **structurally complementary** within ATW v2: G1 handles latent-distribution conditional entropy; B3 handles per-symbol range-coding via scorer-class CDF; G2-PARTIAL is the asymptotic-floor swing (Variant C, deferred to V3 if v2 lands).

### Composition class per Z1 ablation framework (Catalog #219 + #227)

ATW v2 is **across-class** per the Z1 cathedral-autopilot ranker framework:
- Tier A density per current canonical-substrate-cluster ≈ 0.85-0.95 (within-class plateau region — the 0.196-0.199 cluster the operator's `feedback_assumptions_challenge_audit_break_out_local_minima_landed_20260515.md` retrospective identified).
- ATW v2's distinguishing-feature is the cooperative-receiver hypothesis IS the across-class break (per the L5 v2 staircase memo line 6: *"ATW Codec V1 (CLASS-SHIFT: scorer-relationship via Atick-Tishby-Wyner triple)"*).
- D4 probe Tier C density estimation (per Catalog #227): if MI ≥ 0.5 bits/symbol → across-class signature confirmed (latent perturbation at σ=1.0 yields large Δscore); if MI < 0.5 → within-class trap (Tier A density indistinguishable from baseline despite different cite-tokens).

The autopilot ranker thus applies the **class-shift +0.01 reward** to ATW v2's ΔS prediction ONLY if D4 verdict = MEANINGFUL_CONDITIONING. The conditional reward closes the loop between empirical disambiguation (D4) and operational ranking (autopilot Hook 4).

---

## 4. Architecture (FULL V2 substrate spec — two variants)

### 4.1 Module-level architecture (Variant A: three-knob V1-inherited canonical)

Variant A preserves V1's three-knob form (κ_IB, λ_WZ, λ_pixel) so the probe-disambiguator regime sweep (knob-zero ablations recovering Atick-only / ATW-canonical / Tishby-IB / Z3-baseline corners) remains a built-in arbitration mechanism.

```
ATW v2 Variant A architecture:

INPUTS:
  pair_indices: (B,) long; selects per-pair learned latent z[i] from latent table
  gt_frames: (B, 2, 3, H, W) uint8; ground-truth pair for cooperative-receiver loss
  cooperative_receiver: (SegNet, PoseNet) — FIXED published weights; COMPRESS-side only

ENCODER (compress-side; NEVER ships in archive):
  E_AR: (B, 2, 3, H, W) → (B, latent_dim)
        Atick-Redlich-aware encoder; trains under cooperative_receiver_loss; converges to
        latent that maximizes I(latent; scorer_output(gt))
  E_IB: (B, latent_dim) → (B, T_dim)  [Tishby bottleneck projection; only active when κ_IB > 0]
        Variational encoder approximating p(t|x) per IB Lagrangian; q(t|x) trained jointly
  E_WZ: (B, latent_dim) → (B, scorer_class_index)  [Wyner-Ziv class assignment from scorer at compress]
        Runs SegNet on gt_frames; returns per-pair class assignment (argmax over softmax classes)

LATENT TABLE (ships in archive):
  z_residual: (num_pairs, latent_dim) int8; z - g_WZ(class_index_table[i])
       where g_WZ is the Wyner-Ziv side-info head reconstruction

WYNER-ZIV SIDE-INFO HEAD (ships in archive; CLOSED-FORM at compress per Wunderkind B3):
  scorer_class_prior_table: (num_pairs, scorer_class_prior_dim) fp16; SegNet argmax distribution per pair
  wz_side_info_head: small MLP scorer_class_prior_table[i] → predicted z_predicted[i]
       At compress: train wz_side_info_head on (class_prior_table, z_AR) supervised regression
       At inflate: z = z_residual + wz_side_info_head(class_prior_table[i])  — NO scorer load at inflate

G1 SCORER-CLASS DISTILL HEAD (ships in archive; replaces Ballé hyperprior):
  distill_head: small MLP decoded_latent_per_pixel → 5-way softmax
       At compress: distilled from SegNet argmax on rendered output (NOT GT — matches inflate-time signal)
       At inflate: distill_head(decoded_latent) provides hyperprior gating for range-coder

B3 SCORER-CONDITIONAL CDF TABLE (ships in archive; precomputed at compress):
  cdf_table: (5_classes, 256_symbols) fp16; conditional CDF P(latent_symbol | scorer_class)
       At compress: empirical histogram of (z_residual_quantized, class_index) pairs
       At inflate: cdf_table[class_index, :] selects the conditional CDF for range-decoder

DECODER (ships in archive; NEVER loads scorer):
  decoder: D : (B, latent_dim) → (B, 2, 3, H, W) uint8
       Per-pair RGB pair reconstruction; same architecture as V1 (encoder+decoder+latent table)

ARCHIVE BYTES (ATW2 grammar; §10 detail):
  encoder_state_dict (brotli)    — needed at compress reconstruction roundtrip
  decoder_state_dict (brotli)
  wz_side_info_head_state_dict (brotli)
  distill_head_state_dict (brotli)
  z_residual_int8 (raw int8)
  scorer_class_prior_table_fp16 (raw)
  cdf_table_fp16 (raw)
  meta_blob (sorted-keys JSON)
```

### 4.2 Module-level architecture (Variant B: single-knob WZ-only tighter form)

Variant B drops Tishby IB term (κ_IB = 0 always) and Z3 pixel-MSE residual (λ_pixel = 0 always) to bind ONLY the Wyner-Ziv side-info residual mechanism. Tighter substrate-engineering per HNeRV parity L7 (substrate engineering happens ONCE per architecture class; ATW Variant B is the WZ-only class). Smaller LOC budget; faster smoke; cleaner first-principles bound.

```
ATW v2 Variant B architecture (simplifies Variant A):

ENCODER:  E_AR + E_WZ only (drop E_IB)
LATENT TABLE: same (z_residual int8)
WYNER-ZIV SIDE-INFO HEAD: same (closed-form per Wunderkind B3)
G1 SCORER-CLASS DISTILL HEAD: same (replaces Ballé hyperprior)
B3 SCORER-CONDITIONAL CDF: same
DECODER: same

DROPPED FROM VARIANT A:
  - E_IB Tishby bottleneck projection (κ_IB knob always 0)
  - Z3 pixel-MSE residual head (λ_pixel knob always 0)
  - Three-knob probe-disambiguator regime sweep (collapses to single-knob Variant B)
```

**LOC budget**: Variant A ≈ 450-600 LOC bolt-on (substrate-engineering waiver per HNeRV parity L7); Variant B ≈ 250-350 LOC bolt-on (within standard substrate-engineering budget).

### 4.3 Variant adjudication

Council adjudication per §19 chooses Variant A vs B based on D4 probe verdict + Assumption-Adversary input:

| D4 verdict | Variant A recommendation | Variant B recommendation |
|---|---|---|
| MEANINGFUL_CONDITIONING (MI ≥ 0.5) | YES — three-knob regime sweep enables empirical disambiguation of Atick vs IB vs WZ contribution shares | YES — WZ-only is the substrate-optimal binding per UNIQUE-AND-COMPLETE-PER-METHOD; cleaner first-principles ΔS prediction |
| WEAK_CONDITIONING (0.01 ≤ MI < 0.5) | DEFER — three-knob may produce false signal that disguises the weak conditioning | DEFER — even single-knob expected ΔS scales as MI/0.5 → small (< 0.005) |
| INDEPENDENT (MI < 0.01) | DEFER per CLAUDE.md "Forbidden premature KILL"; cite alternative-hypothesis G2-PARTIAL | DEFER per same |

**Default recommendation pending council**: ship Variant B first (UNIQUE-AND-COMPLETE-PER-METHOD per the standing directive). The three-knob form was V1's research-only safety net; V2's distinguishing-feature is the WZ mechanism specifically.

---

## 5. Pretraining

### 5.1 Per-substrate pretraining (None required at scaffold landing)

ATW v2 is trained from scratch on contest data (`upstream/videos/0.mkv`). No external pretraining required for Variant A or B.

### 5.2 Optional DP1 codebook init (§13 composition opportunity)

If ATW v2 composes with DP1 (per Catalog #209/#210/#211/#213), the DP1 pretrained codebook initializes encoder weights. DP1 codebook is distilled from Comma2k19 (per `tools/probe_latent_conditional_entropy_h_latent_given_scorer_class.py` sister probes); per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" + Catalog #209, the canonical `Comma2k19FrameIterator` does NOT use contest video bytes (leakage-refused at iterator construction).

---

## 6. Curriculum (joint vs sequential)

Variant A SEQUENTIAL training schedule (preserves probe-disambiguator regime sweep):

| Phase | Epochs | Active knobs | Purpose |
|---|---|---|---|
| Warmup | 0-25 | κ_IB=0, λ_WZ=0, λ_pixel=0 | Pure Atick-Redlich (= Z4 baseline); converge encoder under cooperative receiver |
| WZ activation | 25-75 | κ_IB=0, λ_WZ=1, λ_pixel=0 | Add Wyner-Ziv residual term; train wz_side_info_head + class_prior_table |
| Full ATW | 75-200 | κ_IB ramp 0→0.05, λ_WZ=1, λ_pixel=0 | Add Tishby IB regularizer; converge full Lagrangian |

Variant B SIMPLER training (UNIQUE-AND-COMPLETE):

| Phase | Epochs | Active terms | Purpose |
|---|---|---|---|
| Warmup | 0-25 | Atick-Redlich only | Converge encoder under cooperative receiver |
| Full WZ | 25-200 | Atick-Redlich + Wyner-Ziv residual | Train wz_side_info_head; distill G1 head; build B3 CDF table |

Both variants use EMA decay 0.997 per CLAUDE.md non-negotiable + Catalog #5 eval_roundtrip per CLAUDE.md non-negotiable.

---

## 7. Architecture priors

- **Cooperative receiver fixed**: `(SegNet, PoseNet)` published in `upstream/modules.py`; used at COMPRESS-side only per Catalog #6 strict-scorer-rule.
- **Latent shape**: per-pair `(num_pairs=600, latent_dim=24)` int8 quantized → ~14.4KB raw; with WZ residual encoding + scorer-class-conditional CDF range-coding, target ~4-7KB latent bytes.
- **Wyner-Ziv side-info head**: small MLP `scorer_class_prior_table[i] → z_predicted[i]`; ~32 hidden units; ~1KB params.
- **G1 distill head**: small MLP `decoded_latent_per_pixel → 5-way softmax`; ~256 params; ~1KB.
- **B3 CDF table**: `(5_classes, 256_symbols)` fp16 = 2.5KB.
- **Decoder**: same as V1; HNeRV-style per-pair embedding + 6 upsample blocks; ~80K params @ FP4 ≈ 40KB.

Total predicted archive size (Variant B; tight estimate): encoder + decoder + WZ head + G1 head + z_residual + scorer_class_prior_table + cdf_table + meta + brotli overhead ≈ 80-120KB vs A1's 179KB (40-55% reduction; consistent with V1 design memo's 30-50% latent savings estimate per CC-3).

---

## 8. Post-training (TTO; deferred to V3)

Standard ATW v2 does NOT include test-time optimization. TTO compose with ATW v2 is a V3 candidate per V1 §6 — defer.

---

## 9. Score-aware loss design

### 9.1 Variant A (V1-inherited three-knob)

```python
L_ATW_v2_A = α · B(θ)/N                                # rate from archive bytes
           + β_seg · d_seg(θ)                          # Atick-Redlich SegNet term
           + γ_pose · sqrt(d_pose(θ))                  # Atick-Redlich PoseNet term
           + κ_IB · I(T; Y_predicted)                  # Tishby IB info-preservation
           + λ_WZ · R_WZ_residual(t | t̂(s))          # Wyner-Ziv side-info residual term
           + λ_pixel · MSE(decoded, GT)                # Z3 pixel-MSE residual (default 0)

where:
   α = 25.0 (contest formula)
   β_seg = 100.0
   γ_pose = sqrt(10)
   κ_IB = 0.0 default → ramp to 0.05 in Phase 3
   λ_WZ = 1.0 default
   λ_pixel = 0.0 default
```

### 9.2 Variant B (UNIQUE-AND-COMPLETE single-knob)

```python
L_ATW_v2_B = α · B(θ)/N                                # rate
           + β_seg · d_seg(θ)                          # Atick-Redlich SegNet
           + γ_pose · sqrt(d_pose(θ))                  # Atick-Redlich PoseNet
           + λ_WZ · R_WZ_residual(t | t̂(s))          # Wyner-Ziv residual
```

Both variants route through canonical `tac.codec.cooperative_receiver.atick_redlich.cooperative_receiver_loss` (Catalog #164 + Wunderkind E1) for the Atick-Redlich terms; substituting the canonical primitive avoids the hand-roll bug-class breeding pattern Wunderkind E1 named.

**Differentiability** (CLAUDE.md eval_roundtrip + HNeRV parity L8): both variants apply `tac.differentiable_eval_roundtrip.apply_eval_roundtrip_during_training` before scorer loss; `patch_upstream_yuv6_globally` invoked before scorer construction per V1 design and parity guard Catalog #187.

---

## 10. Archive grammar (ATW2 byte-level layout)

ATW2 monolithic single-file `0.bin` per HNeRV parity discipline L3. Sister to ATW1 (V1) but distinct magic + 2 new sections (G1 distill head + B3 CDF table).

```
MAGIC(4)                           b"ATW2"
VERSION(1)                         u8       schema version (currently 1)
VARIANT(1)                         u8       0 = Variant A (three-knob), 1 = Variant B (WZ-only)
LATENT_DIM(2)                      u16      cfg.latent_dim (e.g. 24)
NUM_PAIRS(2)                       u16      cfg.num_pairs (e.g. 600)
SCORER_CLASS_PRIOR_DIM(2)          u16      cfg.scorer_class_prior_dim (e.g. 16)
DISTILL_HEAD_HIDDEN_DIM(2)         u16      cfg.distill_head_hidden_dim (e.g. 32)
CDF_TABLE_NUM_CLASSES(2)           u16      5 (SegNet)
CDF_TABLE_NUM_SYMBOLS(2)           u16      256 (int8 latent symbols)

ENCODER_BLOB_LEN(4)                u32      brotli-compressed encoder state_dict (NOT shipped at inflate; included for re-train roundtrip; can be empty in inflate-only build)
DECODER_BLOB_LEN(4)                u32      brotli-compressed decoder state_dict
WZ_HEAD_BLOB_LEN(4)                u32      brotli-compressed wz_side_info_head state_dict
DISTILL_HEAD_BLOB_LEN(4)           u32      brotli-compressed G1 distill_head state_dict
LATENT_RESIDUAL_BLOB_LEN(4)        u32      int8 z_residual bytes (= num_pairs * latent_dim)
CLASS_PRIOR_TABLE_BLOB_LEN(4)      u32      fp16 scorer_class_prior_table bytes
CDF_TABLE_BLOB_LEN(4)              u32      fp16 cdf_table bytes (= 5 * 256 * 2)
META_BLOB_LEN(4)                   u32      sorted-keys JSON utf-8 bytes

ENCODER_BLOB                       ...      (optional; may be empty if inflate-only build)
DECODER_BLOB                       ...      (REQUIRED; ~40KB after brotli)
WZ_HEAD_BLOB                       ...      (REQUIRED; ~1KB)
DISTILL_HEAD_BLOB                  ...      (REQUIRED for G1; ~1KB)
LATENT_RESIDUAL_BLOB               ...      (REQUIRED; ~14.4KB raw before range-coding)
CLASS_PRIOR_TABLE_BLOB             ...      (REQUIRED; num_pairs * scorer_class_prior_dim * 2 bytes)
CDF_TABLE_BLOB                     ...      (REQUIRED for B3; ~2.5KB)
META_BLOB                          ...      (provenance; ATW2 codec metadata; sorted-keys JSON)
```

**Byte-stable invariant** (CLAUDE.md "Bit-level deconstruction and entropy discipline"): all fp16 tensors stored in IEEE 754 little-endian byte order; sorted-keys JSON utf-8 ensures meta blob hash is reproducible; brotli compression with deterministic parameters (quality=11, lgwin=22, mode=GENERIC). Round-trip contract: `bytes → parse_archive → pack_archive → bytes` is byte-identical.

**Distinguishing-feature contract per Catalog #272**:
- `distinguishing_feature_name`: "scorer_class_conditional_latent_residual_with_g1_distill_head_b3_cdf_table"
- `distinguishing_bytes_path`: `WZ_HEAD_BLOB + DISTILL_HEAD_BLOB + LATENT_RESIDUAL_BLOB + CLASS_PRIOR_TABLE_BLOB + CDF_TABLE_BLOB` (the BYTES that encode the cooperative-receiver hypothesis)
- `inflate_consumer_function`: `tac.substrates.atw_codec_v2.inflate.reconstruct_pairs_from_atw2_bytes` (consumes ALL distinguishing bytes for frame reconstruction)
- `byte_mutation_smoke_passes`: REQUIRED before any L2+ promotion; sister `tools/verify_distinguishing_feature_byte_mutation.py` mutates 1 byte per declared offset and verifies frame output changes

---

## 11. Inflate runtime (≤200 LOC substrate-engineering budget per HNeRV parity L4 waiver)

```python
# experiments_substrates_atw_codec_v2/inflate.py (~150-200 LOC)

def inflate(archive_zip: Path, output_dir: Path, file_list: Path) -> int:
    """ATW v2 inflate runtime per HNeRV parity L4.

    Parses ATW2 archive; reconstructs per-pair latents from z_residual + WZ side-info head;
    range-decodes via B3 scorer-conditional CDF table; renders RGB pairs via decoder.
    NO scorer load (per Catalog #6 strict-scorer-rule).
    """
    archive_bytes = (archive_zip / "0.bin").read_bytes()
    parsed = parse_atw2_archive_bytes(archive_bytes)

    device = select_inflate_device()  # Catalog #205 canonical
    decoder = build_decoder(parsed.meta)
    decoder.load_state_dict(parsed.decoder_sd)
    wz_head = build_wz_side_info_head(parsed.meta)
    wz_head.load_state_dict(parsed.wz_head_sd)
    distill_head = build_distill_head(parsed.meta)  # G1
    distill_head.load_state_dict(parsed.distill_head_sd)

    # B3: range-decode z_residual using cdf_table conditioned on scorer_class_prior_table
    z_residual = range_decode_b3(
        encoded_bytes=parsed.latent_residual_blob,
        cdf_table=parsed.cdf_table,
        class_prior_table=parsed.scorer_class_prior_table,
    )  # → (num_pairs, latent_dim) int8

    # Reconstruct full latents via Wyner-Ziv: z = z_residual + wz_head(class_prior[i])
    with torch.no_grad():
        z_predicted = wz_head(parsed.scorer_class_prior_table.to(device))
        z_full = z_residual.float().to(device) + z_predicted

    # Render per-pair via decoder; iterate per file_list per Catalog #146
    pair_indices = torch.arange(parsed.num_pairs, device=device)
    for video_name in file_list.read_text().splitlines():
        with torch.no_grad():
            rgb_0, rgb_1 = decoder(z_full[pair_indices])  # (num_pairs, 3, H, W) per frame
        # Write per-frame uint8 RGB raw to output_dir/<video_name>
        write_rgb_pairs(output_dir / video_name, rgb_0, rgb_1)

    return 0
```

**Inflate dependency closure** per Catalog #5 + HNeRV parity L9: `torch`, `brotli`, `numpy` (range coding via numpy). NO scorer dependencies. NO upstream/modules.py import. Verified via `dispatch_optimization_protocol_complete` (Catalog #270).

**Inflate output parity check** per Catalog #221: writes per-axis blockers `result_review_blockers=["roundtrip_matrix_is_command_planner_not_claim_surface", "requires_separate_auth_eval_result_review_before_score_claim"]` until paired CPU+CUDA full anchor lands.

---

## 12. Export contract

```python
# In experiments/train_substrate_atw_codec_v2.py::_full_main (currently NotImplementedError pending Phase 2)

def _export_atw2_archive(model, output_dir, args) -> Path:
    """Export trained ATW v2 model into ATW2 archive bytes."""
    archive_bytes = pack_archive(
        encoder_sd=model.encoder.state_dict(),  # OPTIONAL; can be empty
        decoder_sd=model.decoder.state_dict(),
        wz_head_sd=model.wz_side_info_head.state_dict(),
        distill_head_sd=model.distill_head.state_dict(),  # G1
        z_residual=model.compute_z_residual(),  # z - wz_head(class_prior[i])
        scorer_class_prior_table=model.scorer_class_prior_table,
        cdf_table=model.build_b3_cdf_table(),  # empirical histogram at compress
        meta=meta_dict,
        variant=args.variant,  # "A" or "B"
    )
    archive_path = output_dir / "0.bin"
    archive_path.write_bytes(archive_bytes)
    # ZIP STORED per HNeRV parity L3; deterministic ZIP (Catalog #19); single member
    zip_path = output_dir / "archive.zip"
    write_stored_zip(zip_path, {"0.bin": archive_bytes})
    return zip_path
```

Round-trip parity test: `bytes → parse_archive → pack_archive → bytes` byte-identical. Per Catalog #1 `check_encoder_decoder_dequantization_roundtrip_tested` (B1).

---

## 13. Stack-of-stacks composition matrix (ATW v2 with OTHER substrates; higher-order)

| With substrate | Axis orthogonality | Composition class | Predicted contribution | Rationale |
|---|---|---|---|---|
| **A-STACK** (NSCS01 + NSCS02 + NSCS03 composition) | NEAR-ORTHOGONAL (A-STACK = renderer-arch + entropy; ATW v2 = entropy-coding via cooperative receiver) | A-STACK[NSCS03 → ATW v2] SWAP — per A-STACK §13 row 11 ATW REPLACES NSCS03 redundancy | NULL pending probe | Per A-STACK design memo §13: NSCS03 entropy bottleneck and ATW v2 Wyner-Ziv residual both optimize `-log2 p(y_hat)` over a learned prior; cannot compose additively. The swap: A-STACK = NSCS02 + NSCS01 + ATW v2 (3 axes preserved; entropy axis switched from Ballé hyperprior to cooperative-receiver). Predicted A-STACK[swap] band: NULL pending paired probe per Catalog #296. |
| **NSCS06 v8 Path B** (chroma wavelet residual; per `nscs06_v8_path_b_wavelet_residual_full_stack_design_20260516.md`) | ORTHOGONAL (NSCS06 v8 = chroma reconstruction via wavelet residual; ATW v2 = luma scorer-class conditioning) | FRESH-COMPOSITION (across-class + across-class) | small additive (~0.005-0.010) | NSCS06 v8 Path B redesigns chroma; ATW v2 operates on luma scorer surface (SegNet input is YUV6 with chroma subsampled per `frame_utils.rgb_to_yuv6`). Compose at YUV-channel level: chroma side-info from NSCS06 v8 + luma latent from ATW v2. Both stay score-aware per HNeRV parity L1. |
| **DP1** (pretrained-driving-prior; Catalog #209/#210/#211/#213) | ORTHOGONAL (DP1 = pretraining codebook; ATW v2 = scratch-trained) | DP1-PRETRAIN-INIT-ATW v2 | small additive (~0.005) | DP1 codebook initializes ATW v2 encoder weights; reduces ATW v2 epoch budget from 200 → 50 (~$5 saved). Sister benefit: DP1's Comma2k19 SegNet-class assignments match D4 probe's class index field, so DP1 codebook provides a PRIOR for the WZ side-info head — closes the loop between pretraining and cooperative-receiver framing. Archive: `DPCOMP(DP1, ATW v2)` composition per Catalog #211. |
| **D1 SegNet margin polytope** (Catalog #220 OPERATIONAL canonical) | NEAR-ORTHOGONAL (D1 = SegNet-side margin overlay; ATW v2 = encoder-side scorer-class conditioning) | STRONG_STACK | small additive (~0.003) | D1 adds inflate-time polytope-interior noise to frame_1 RGB; ATW v2 frame_1 output is the input to D1's overlay. Pure composition: ATW v2 archive bytes + D1 sidecar bytes. |
| **NSCS01** (nullspace-split renderer; per `feedback_nscs01_full_main_implementation_pr95_paradigm_landed_20260515.md`) | ORTHOGONAL (NSCS01 = decode-time-contract split-head; ATW v2 = encoder-side scorer-class conditioning) | STRONG_STACK | small additive (~0.005-0.010) | NSCS01 split-head routing exploits SegNet `x[:,-1,...]` slice; ATW v2 exploits SegNet class-conditional softmax. Both compose at the SegNet-as-receiver framework. |
| **NSCS03** (end-to-end Ballé joint codec; per `feedback_nscs03_full_main_implementation_pr95_balle_2018_paradigm_landed_20260515.md`) | REDUNDANT (both are end-to-end entropy coders) | SATURATING / SWAP-WITH-VARIANT-A | floor at -0.005 | Per A-STACK §13 row 11 + NSCS03 design analysis: NSCS03's entropy bottleneck and ATW v2's WZ residual both optimize entropy of latent. Cannot compose additively. CHOICE: ATW v2 Variant A (three-knob with κ_IB > 0 = IB regime) is closer to NSCS03; ATW v2 Variant B is structurally distinct (WZ-only). |
| **Carmack-Hotz Strip-Everything v7+** (composite #4 staircase memo line 7) | NEAR-ORTHOGONAL (Strip-Everything = no-neural-codec minimalism; ATW v2 = neural scorer-class codec) | CONFLICT (different decoder paradigms) | -0.005 (within-class trap if forced) | Per Catalog #219 Z1 ablation, two ACROSS-CLASS candidates with different decoder paradigms cannot compose additively. Cathedral autopilot ranker would penalize via `apply_substrate_composition_matrix_to_candidates`. |
| **Wunderkind G2-PARTIAL** (posterior-matching codec with bounded CDF table) | SAME-AXIS-EXTENSION (G2-PARTIAL is the asymptotic V3 form of ATW v2's B3 + G1) | V3-PROMOTION-CANDIDATE | TBD via G2-PARTIAL council deliberation | G2-PARTIAL replaces ENTIRE learned decoder with stateless scorer-conditional Langevin sampler. ATW v2 ships the precursor B3 CDF table + G1 distill head; G2-PARTIAL is the V3 promotion where the decoder is stateless. Sequence: ATW v2 lands first; if MEANINGFUL_CONDITIONING confirmed, G2-PARTIAL becomes V3 candidate per council deliberation. |
| **A1 baseline** | DIFFERENT-BASE | REPLACES A1 entirely | small (-0.005 expected because ATW v2 is built FROM-SCRATCH) | A1 is the contest-CPU 0.19285 baseline; ATW v2 is a fresh substrate-engineering composition. The within-class plateau A1 sits on is broken by ATW v2's cooperative-receiver across-class component. |

**Top higher-order composition recommendation**: **A-STACK[swap to ATW v2] ⊕ DP1 ⊕ D1** (DP1 pretraining + NSCS02 + NSCS01 + ATW v2 [replacing NSCS03] + D1 inflate-time SegNet overlay). Five orthogonal axes (pretraining + spatial-frequency + decode-time-contract + cooperative-receiver + inflate-time SegNet overlay); predicted combined band: NULL pending Dykstra-feasibility intersection per Catalog #296.

---

## 14. Pipeline-of-pipelines

ATW v2 dispatch pipeline (conditional on D4 probe verdict):

```
STAGE 0 (PRE-DISPATCH GATE):
  $3-5 D4 probe on A1 latents:
    .venv/bin/python tools/probe_latent_conditional_entropy_h_latent_given_scorer_class.py \
        --substrate-id atw_v2 \
        --latent-bytes <A1_latent_bytes_path> \
        --scorer-classes <SegNet_class_per_pair_bytes_path> \
        --output-json .omx/state/h_latent_given_scorer_class_atw_v2.json \
        --meaningful-mi-threshold-bits 0.5

  VERDICT GATE:
    MEANINGFUL_CONDITIONING → STAGE 1
    WEAK_CONDITIONING       → revise predicted band; defer or proceed with Variant B only
    INDEPENDENT             → DEFER per CLAUDE.md "Forbidden premature KILL"; cite G2-PARTIAL alternative

STAGE 1 ($5-15 Modal A100 smoke):
  100ep synthetic-data smoke per V1 §6 cycle:
    .venv/bin/python tools/operator_authorize.py substrate_atw_codec_v2_modal_a100_smoke_dispatch \
        --operator-approved-utc "<UTC>" --operator-handle adpena
  Validates: ATW2 archive byte roundtrip + inflate parity + scorer preprocess gradient flow
  Per Catalog #167 smoke-before-full pattern + Catalog #243 local pre-deploy check + Catalog #271 codex review

STAGE 2 ($10-30 Modal A100 paired CPU+CUDA full anchor):
  200ep on real upstream/videos/0.mkv per Variant B (or A); ends with paired CPU+CUDA contest auth eval per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"
  Provider routing: cost-band posterior per D9 routing; canonical 4-axis paired dispatch per Catalog #246
    .venv/bin/python tools/dispatch_modal_paired_auth_eval.py \
        --archive <ATW v2 archive sha256> \
        --skip-axis-if-promotable-anchor-exists
```

**Total wall-clock**: D4 probe ~30 min CPU + smoke 1.5h Modal A100 + full anchor 3-4h Modal A100 + paired CPU 1.5-2h. Calendar time: <1 day end-to-end if D4 gates pass.

---

## 15. Canonical-vs-unique decision per layer (Catalog #290)

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" non-negotiable. Per-layer canonical-vs-unique-fork rationale.

| Layer | Decision | Rationale |
|---|---|---|
| Trainer skeleton | ADOPT canonical | TF32 + CUDA discipline shared per `tac.substrates._shared.trainer_skeleton.device_or_die` (Catalogs #172/#178/#179/#180). HARD-EARNED — substrate engineering hygiene is universal. |
| Atick-Redlich primitive | ADOPT canonical | `tac.codec.cooperative_receiver.atick_redlich.cooperative_receiver_loss` per Catalog #164 + Wunderkind E1 substitution candidate. The primitive IS the canonical Atick-Redlich; substrate-specific hand-roll is bug-class breeding (V1 design memo §1 noted Z4 hand-roll). HARD-EARNED. |
| Eval-roundtrip | ADOPT canonical | CLAUDE.md "eval_roundtrip — NON-NEGOTIABLE" + Catalog #5. HARD-EARNED. |
| EMA decay (0.997) | ADOPT canonical | CLAUDE.md "EMA — NON-NEGOTIABLE" + Catalog #88. HARD-EARNED. |
| Score-aware loss helper | ADOPT canonical | `tac.substrates._shared.score_aware_common.score_pair_components` per Catalog #164. Delegated through Atick-Redlich primitive. HARD-EARNED. |
| Inflate runtime (device selection) | ADOPT canonical | `select_inflate_device` per Catalog #205. HARD-EARNED. |
| Auth eval helper | ADOPT canonical | `gate_auth_eval_call` per Catalog #226. HARD-EARNED. |
| Hardware substrate detection | ADOPT canonical | `detect_hardware_substrate` per Catalog #190. HARD-EARNED. |
| Modal A100 min_smoke_gpu | ADOPT canonical | Per Catalog #215. HARD-EARNED. |
| Required-input-file validation | ADOPT canonical | `tools/validate_dispatch_required_inputs.py` per Catalog #152. HARD-EARNED. |
| Operator-authorize entry point | ADOPT canonical | `tools/operator_authorize.py` per Catalog #176. HARD-EARNED. |
| ATW2 archive grammar | **UNIQUE FORK** | Sister-to-ATW1 (V1) but new sections (G1 distill head + B3 CDF table) require new magic `b"ATW2"`; substrate-engineering surface per HNeRV parity L7. UNIQUE per substrate-distinguishing-feature (Catalog #272). |
| Wyner-Ziv side-info head closed-form | **UNIQUE FORK** | Closed-form prediction `z_predicted = wz_head(class_prior_table[i])` is substrate-specific; cannot be canonicalized without losing the cooperative-receiver hypothesis. UNIQUE per HNeRV parity L7. Sister of V1's TINY MLP placeholder, but V2 lands the actual closed-form per V1 unwind reactivation criterion. |
| G1 scorer-class distill head | **UNIQUE FORK** | Sister to Z3 substrate's Ballé hyperprior; ATW v2's distill head is structurally distinct (5-way SegNet softmax conditioning vs scale hyperprior). UNIQUE per Wunderkind G1 substitution candidate. |
| B3 scorer-conditional CDF table | **UNIQUE FORK** | Precomputed empirical histogram of `(z_residual_quantized, class_index)` shipped as 2.5KB side-info; sister to range-coding tables in entropy codecs but specific to ATW v2's scorer-class index field. UNIQUE per Wunderkind B3. |
| Three-knob (κ_IB / λ_WZ / λ_pixel) Variant A composition | **UNIQUE FORK** | The probe-disambiguator regime sweep mechanism; UNIQUE substrate-engineering primitive per HNeRV parity L7. (Variant B drops this in favor of single-knob WZ-only.) |
| Training curriculum | ADOPT canonical | 2-frame curriculum + pyav decode + patched YUV6 + differentiable scorers + AdamW + cosine LR — all PR95-parity-discipline canonical. HARD-EARNED per HNeRV parity discipline lessons 1-13. |
| Tier-1 engineering | ADOPT canonical | autocast_fp16 / TF32 / torch.compile / no_grad-at-eval / canonical scorer-loss helper (Catalogs #172/#178/#179/#180/#164). HARD-EARNED. |
| Scorer routing | ADOPT canonical | `load_differentiable_scorers` + `patch_upstream_yuv6_globally` per Catalog #164 + canonical scorer-loader assignment order per Catalog #222. HARD-EARNED. |
| D4 probe-disambiguator | ADOPT canonical | Shared probe `tools/probe_latent_conditional_entropy_h_latent_given_scorer_class.py` per audit foundation §5 op-routable #5. HARD-EARNED. |
| Dispatch optimization protocol | ADOPT canonical | `verify_dispatch_protocol_complete` per Catalog #270 + #279/#280 fail-closed protections. HARD-EARNED. |

**Bolt-on vs substrate-engineering split per HNeRV parity L7**: ATW v2 is substrate-engineering (NEW architecture class: WZ-residual codec with scorer-class CDF range coding). LOC budget exceeds bolt-on cap explicitly. The 6 UNIQUE FORK decisions (ATW2 archive grammar + WZ closed-form + G1 + B3 + three-knob Variant A + B3 CDF) ARE the substrate-optimal engineering surface; remaining ADOPT canonical decisions are shared infrastructure value preserved per the standing directive `feedback_canonical_share_when_serves_unique_when_suppresses_standing_directive_20260515.md`.

---

## 16. Cargo-cult audit per assumption (V2-specific)

Per the standing META-ASSUMPTION ADVERSARIAL REVIEW non-negotiable (CLAUDE.md + Catalog #291).

| Assumption | Classification | Rationale | Disposition |
|---|---|---|---|
| The D4 probe MI threshold 0.5 bits/symbol distinguishes meaningful conditioning from noise | **HARD-EARNED** | Per probe source `DEFAULT_MEANINGFUL_MI_THRESHOLD_BITS = 0.5` + audit foundation §5 op-routable #5: 0.5 bits/symbol distinguishes a meaningful Wyner-Ziv conditioning channel from noise on typical latent streams of 10k-1M symbols. Operators may tighten via `--meaningful-mi-threshold-bits` for high-entropy latents. | PRESERVED; document the threshold sensitivity in §19 reactivation criteria so council can adjust if D4 verdict is borderline. |
| G1 scorer-class distill head saves 30% of latent bits | **CARGO-CULTED** | Wunderkind candidate cites estimate `−0.005 to −0.015 rate-axis`. Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" + Catalog #287: this is `[prediction; first-principles-bound]` until empirical confirmation via paired smoke. | UNWOUND via §18 NULL pending probe; per-prediction `[prediction]` axis tag mandatory. |
| B3 scorer-conditional CDF table reduces per-symbol entropy by 20-30% | **CARGO-CULTED** | Empirical estimate based on Ballé hyperprior 30% reduction × scorer-class-conditioning fraction (50-70%). Untested for our specific latent distribution. | UNWOUND via D4 probe execution; if MEANINGFUL_CONDITIONING confirmed, the CDF table reduction is realistic; otherwise revise down. |
| Three-knob Variant A regime sweep yields cleanly-separable corner ablations | **CARGO-CULTED** | V1 design memo §1 lines 95-98 enumerates 4 corner regimes (Atick-only / ATW-canonical / Tishby IB pure / Z3 baseline). Empirical separability untested. | UNWOUND via knob-zero ablation smoke (council-grade decision per §19). Variant B drops the assumption entirely. |
| ATW v2 composes orthogonally with NSCS06 v8 Path B (chroma) | **CARGO-CULTED** | Composition matrix §13 cites FRESH-COMPOSITION; orthogonality untested for the YUV-channel interface. | UNWOUND via composition smoke ($5-10 incremental) AFTER ATW v2 individual smoke lands. |
| ATW v2 SWAP with NSCS03 in A-STACK is strictly better than original A-STACK | **CARGO-CULTED** | Per A-STACK §13 row 11: ATW REPLACES NSCS03 in A-STACK; both optimize entropy of latent. The "strictly better" claim assumes ATW v2's cooperative-receiver framing dominates NSCS03's Ballé hyperprior — untested. | UNWOUND via A-STACK[swap] vs A-STACK[NSCS03] paired smoke comparison (council-grade per §19). |
| The Wyner-Ziv side-info head closed-form is implementable as a small MLP | **HARD-EARNED** | Per V1 design memo §1 lines 116-122: predict `z_predicted = wz_head(class_prior_table[i])` is a regression task; small MLP (~1KB) sufficient for the limited dimensionality of `scorer_class_prior_dim = 16`. | PRESERVED. |
| ATW v2 is across-class per Z1 ablation framework | **CONDITIONAL** | True IFF D4 probe = MEANINGFUL_CONDITIONING. If WEAK or INDEPENDENT, ATW v2 collapses to within-class. | DEFERRED to D4 probe outcome; cathedral autopilot ranker applies class-shift +0.01 reward conditionally per §3.3. |

**Cargo-cult-class summary**: 3 HARD-EARNED + 4 CARGO-CULTED + 1 CONDITIONAL. All 4 CARGO-CULTED assumptions are disambiguated by D4 probe OR a $5-10 incremental smoke per §19.

---

## 17. 9-dimension success checklist evidence (per Catalog #294)

| # | Dimension | Status |
|---|---|---|
| 1 | UNIQUE-AND-COMPLETE-PER-METHOD bind | PARTIAL — Variant A = three-knob (preserves probe-disambiguator); Variant B = single-knob WZ-only (UNIQUE-AND-COMPLETE binding). Council adjudication §19 chooses based on D4 probe. |
| 2 | Canonical-vs-unique decision per layer | YES — §15 above; 6 UNIQUE FORK + 14 ADOPT canonical. |
| 3 | HARD-EARNED-vs-CARGO-CULTED classification | YES — §16 above; 3 HARD-EARNED + 4 CARGO-CULTED + 1 CONDITIONAL. |
| 4 | Probe-disambiguator per defensible interpretation | YES — D4 H(latent\|scorer_class) probe ($3-5 CPU) + three-knob Variant A regime sweep + Dykstra-feasibility polytope check ($0 analytical). |
| 5 | Premise verification per Catalog #229 | YES — 9 PVs verified in §1; ATW V1 source + canonical Atick-Redlich primitive + D4 probe source + Wunderkind candidates + Atick-Redlich/Tishby/Wyner-Ziv canonical paper anchors. |
| 6 | 6-hook wire-in or N/A rationale per Catalog #125 | DESIGN-TIME at landing. Sensitivity-map = `tac.sensitivity_map.scorer_class_conditional_v2` (planned); Pareto = `tac.pareto.atw_v2_wz_residual_entropy` (planned); Bit-allocator = `bit_allocator.atw_v2_wz_residual_v1` (planned); Cathedral autopilot dispatch = recipe registered warn-only at landing (post-D4 probe verdict promotes to dispatch-eligible); Continual-learning = ATW v2 anchor seeds posterior paired with D4 MI value (per Catalog #128 locked write); Probe-disambiguator = D4 probe + three-knob regime sweep (when Variant A). |
| 7 | Predicted ΔS band with citation | YES — §18 below. NULL pending D4 probe + Dykstra-feasibility check per Catalog #296. Conditional revisions per D4 verdict band. |
| 8 | Reactivation criteria pinned | YES — §19 below. 4-criterion V1 lift gate inherited + 2-criterion V2-specific gate (D4 MEANINGFUL_CONDITIONING + Variant-A-vs-B council adjudication). |
| 9 | Sister-subagent ownership map per Catalog #230 | YES — §1.2 above. READ-ONLY on source; writes ONLY to this memo + 1 commit. |

---

## 18. Predicted ΔS band + Dykstra-feasibility check (per Catalog #296)

**Predicted ΔS band**: `NULL pending D4 probe verdict + Dykstra-feasibility check` [prediction]

**First-principles derivation framework**:

Per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable + Boyd's Dykstra co-lead role, the achievable region is the convex intersection of independent rate-distortion constraints. ATW v2's distinguishing-feature contributes the following constraints to the intersection:

- **Atick-Redlich rate constraint**: `R_AR ≥ H(X) - I(X; Y_cooperative)` where Y_cooperative = (SegNet_out, PoseNet_out). For A1's latent distribution + 5-class SegNet output, empirical I(X;Y) ≈ 11-12 bits/pixel per Wunderkind G1 analysis line 162.
- **IB constraint** (Tishby-Zaslavsky 2015): `I(X;T) - β·I(T;Y) ≤ K_IB` where K_IB depends on Tishby IB analytical floor (Tao+Boyd Blahut-Arimoto computation; deferred to V1 §5 reactivation criterion #2).
- **Wyner-Ziv side-info constraint** (Wyner-Ziv 1976): `R_WZ ≥ R(D) - I(X; Y_side)` where Y_side = `scorer_class_prior_table` shipped in archive. **D4 probe estimates this directly via `I(latent ; class)` measurement.**
- **Contest rate budget**: `25 · archive_bytes / 37,545,489`

**The Dykstra-feasibility intersection** of these 4 polytopes is a SUBSET of the Z3+A1 polytope per V1 design memo §predicted-band lines 138-144. This does NOT imply dominance absent a measured lower-score point on the same archive/runtime/eval axis.

**D4 probe verdict-conditional revisions**:

| D4 verdict | MI threshold | Predicted band | Basis |
|---|---|---|---|
| **MEANINGFUL_CONDITIONING** | MI ≥ 0.5 bits/symbol | `[-0.015, -0.005]` rate-axis only | Tishby IB lower bound: WZ side-info savings = `I(latent; class) / H(latent)` × A1 rate (~0.20); for MI = 0.5 bits and H(latent) ≈ 2 bits/symbol → ~25% rate reduction → ~0.05 rate ΔS. Haircut for non-stationary class statistics + distill head accuracy → realistic [-0.015, -0.005]. |
| **WEAK_CONDITIONING** | 0.01 ≤ MI < 0.5 | `[-0.005, -0.002]` rate-axis | Scales linearly with MI ratio; e.g. MI = 0.1 → ~5% rate reduction → ~0.01 rate ΔS. |
| **INDEPENDENT** | MI < 0.01 | `[NULL; ATW v2 does not displace A1 frontier via this mechanism]` | Cooperative-receiver hypothesis FALSIFIED for ATW v2; pivot to G2-PARTIAL alternative-hypothesis per CLAUDE.md "Forbidden premature KILL" (DEFER, not KILL). |

**Operating-point sensitivity** (CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent"): at PR106's frontier operating point (pose_avg ~3.4e-5), pose marginal value is 2.71× SegNet's. ATW v2's Wyner-Ziv mechanism targets latent bytes (rate axis) — orthogonal to the seg-vs-pose marginal-value flip. ATW v2's predicted rate-axis ΔS is operating-point INVARIANT (rate is rate). The seg/pose distortion side is bounded by Atick-Redlich loss (already canonical).

**Dykstra-feasibility check execution** (sister probe):
```bash
.venv/bin/python tools/check_substrate_dykstra_feasibility.py --substrate atw_v2 \
    --archive-bytes-target 120000 \
    --rate-budget-axis-target 0.080 \
    --constraints atick_redlich,tishby_ib,wyner_ziv,contest_rate \
    --output-json .omx/state/dykstra_feasibility_atw_v2.json
```

If the intersection is non-empty + bounded BELOW A1's operating point → predicted band PROCEEDS to dispatch. If empty or unbounded → DEFER-pending-research per V1 unwind disposition.

**Score axis label**: `[prediction; first-principles-bound; D4-probe-conditional]` per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag".

---

## 19. Reactivation criteria + composition-specific probe-disambiguator

ATW v2 ships as L1 SCAFFOLD per Catalog #220 cascade (research_only=true + lane_class=substrate_engineering + `_full_main` raises NotImplementedError). Phase 2 council approval required to lift `NotImplementedError`.

**V1 4-criterion lift gate INHERITED** (per V1 design memo §5; any TWO grant Phase 2 approval):
1. Z4-V2 successful contest-CUDA anchor (`fc-01KRPJVEMQ5S7Q8EKGKQWKCS93`) with ΔS in [-0.005, -0.010] vs A1.
2. Tao+Boyd Blahut-Arimoto analytical floor computed (~$0, 2hrs CPU) AND floor for A1's operating point shows ATW theoretical gain ≥ -0.020 vs A1.
3. WZ side-info bit-savings empirically measured on A1 latents ≥ 20% per-pair latent rate ($0, 1hr CPU) — **THIS IS THE D4 PROBE**.
4. Z3-G1 substitution (Wunderkind G1, $10) returns successful CUDA anchor.

**V2-specific reactivation criteria** (BOTH required in addition to any 2 of the V1 criteria):
- **V2-1**: D4 probe returns `MEANINGFUL_CONDITIONING` (MI ≥ 0.5 bits/symbol). Output: `.omx/state/h_latent_given_scorer_class_atw_v2.json`.
- **V2-2**: Variant adjudication council adjudication (sextet pact per Catalog #292) decides Variant A vs B based on D4 probe + Assumption-Adversary input. Memo: `feedback_atw_v2_variant_adjudication_council_<YYYYMMDD>.md`.

**Probe-disambiguator (full ATW v2 hook #6 per Catalog #125)**:

Three layers of empirical disambiguation:

1. **D4 probe** ($3-5 CPU): disambiguates Wyner-Ziv hypothesis directly. `MEANINGFUL_CONDITIONING / WEAK_CONDITIONING / INDEPENDENT`.
2. **Three-knob regime sweep** (Variant A only; $30 paired): 4-corner ablations `(κ_IB=0, λ_WZ=0, λ_pixel=0)=Atick-only / (κ_IB=0, λ_WZ=1, λ_pixel=0)=ATW canonical / (κ_IB=0.1, λ_WZ=0, λ_pixel=0)=Tishby IB pure / (κ_IB=0, λ_WZ=0, λ_pixel=1)=Z3 baseline`. Arbitrates between {Atick / IB / WZ / classical}.
3. **Variant A vs B paired smoke** ($10-20): if D4 verdict is MEANINGFUL and Council adjudication is split, paired smoke arbitrates Variant choice empirically.

**Per CLAUDE.md "Forbidden premature KILL"**: ATW v2 NEVER killed. INDEPENDENT D4 verdict → DEFER-pending-research-with-G2-PARTIAL-alternative-hypothesis. WEAK_CONDITIONING → revise predicted band down + ship Variant B only.

---

## 20. Cost estimate

| Stage | Provider | GPU | Wall-clock | $/hr | Cost |
|---|---|---|---|---|---|
| **STAGE 0 D4 probe** (PRE-DISPATCH GATE) | Local macOS CPU or Modal CPU | n/a | 15-30 min | $0-0.06 | $3-5 (Modal CPU instance, conservative) |
| **STAGE 1 ATW v2 smoke** | Modal A100 | 1 × A100 | 1.5-2hrs | $3.50 | $5-10 |
| **STAGE 1.b** Variant A regime sweep (4-corner ablations, optional council-grade) | Modal A100 | 1 × A100 × 4 corners | 6-8hrs total | $3.50 | $20-30 |
| **STAGE 2 ATW v2 full anchor** | Modal A100 | 1 × A100 | 3-4hrs | $3.50 | $10-15 |
| **STAGE 2.b** paired CPU eval (Linux x86_64) | Modal CPU or Vast.ai CPU | 1 × CPU | 1.5-2hrs | $0.06-0.15 | $0.10-0.30 |
| **STAGE 3** composition smoke (e.g. ATW v2 ⊕ DP1; ATW v2 ⊕ A-STACK[swap]) | Modal A100 | 1 × A100 | 1.5-2hrs | $3.50 | $5-10 |
| **Total envelope (Variant B straight-through)** | mixed | mixed | ~6-8hrs | mixed | **$18-30** |
| **Total envelope (Variant A with regime sweep)** | mixed | mixed | ~12-16hrs | mixed | **$48-65** |

Per CLAUDE.md "Long-burn score-lowering campaign default" — operator-funded campaign uses Variant A regime sweep as the council-grade investigation. Routine paid dispatch uses Variant B (cheaper, simpler, UNIQUE-AND-COMPLETE).

Per Catalog #270 dispatch optimization protocol: all dispatches MUST satisfy Tier 1/2/3 engineering primitives + canonical scorer-loss helper routing + recipe `min_smoke_gpu=A100` + canonical 3-export NVML/CUDA env block + auth_eval canonical helper + canonical inflate device + scorer-loader assignment order + recipe-vs-trainer-state consistency + no phantom device-named output directories.

---

## 21. Observability surface (per max-observability directive)

This V2 design memo lands new observability hooks the operator + future subagents + cathedral autopilot can consume directly. Per the max-observability standing directive, every paid dispatch + every probe execution + every composition decision produces a machine-readable artifact stamped with axis labels + score_claim=false discipline.

### 21.1 Observability artifacts produced by ATW v2

| Artifact | Path | Schema | Consumer | Catalog |
|---|---|---|---|---|
| **D4 probe verdict** | `.omx/state/h_latent_given_scorer_class_atw_v2.json` | `HLatentGivenScorerClassVerdict` dataclass per probe source | Cathedral autopilot Hook 4 + V2 lift gate evaluator + council adjudication | #192 + #221 (fail-closed) |
| **ATW v2 archive manifest** | `experiments/results/lane_atw_codec_v2_*/build_manifest.json` | per Catalog #2 evidence row schema | Lane registry + autopilot ranker + distinguishing-feature audit | #2 + #272 |
| **Three-knob regime sweep verdict** (Variant A only) | `.omx/state/atw_v2_three_knob_regime_sweep_verdict_<UTC>.json` | per-corner ΔS + axis label `[contest-CUDA Modal A100]` paired with `[contest-CPU GHA Linux x86_64]` | Council adjudication + cathedral autopilot ranker | #221 |
| **Variant A vs B adjudication memo** | `.omx/research/feedback_atw_v2_variant_adjudication_council_<UTC>.md` | per Catalog #292 council deliberation memo | Operator + sister substrate authors | #292 |
| **Distinguishing-feature byte-mutation proof** | `experiments/results/lane_atw_codec_v2_*/distinguishing_feature_byte_mutation_proof.json` | per `tools/verify_distinguishing_feature_byte_mutation.py` schema | Catalog #272 verifier + auth_eval gate | #272 + #139 |
| **Dykstra-feasibility polytope JSON** | `.omx/state/dykstra_feasibility_atw_v2.json` | analytical 4-polytope intersection + non-emptiness flag | §18 predicted-band-band check + autopilot ranker | #296 |
| **6-hook wire-in declaration** | `feedback_atw_v2_*_landed_<UTC>.md` | per Catalog #125 6-hook wire-in declaration | Subagent landing audit + cathedral autopilot | #125 + #229 |
| **ATW v2 cost-band posterior anchor** | `.omx/state/cost_band_posterior.jsonl` (append-only) | per `tac.cost_band_calibration.append_anchor(outcome=...)` | D9 routing + cathedral autopilot dispatch ranker | #175 + #177 |
| **Modal call_id ledger row** | `.omx/state/modal_call_id_ledger.jsonl` (append-only) | per `tac.deploy.modal.call_id_ledger` schema | Harvest discipline + lane registry + autopilot | #245 |
| **Lane registry entry** | `.omx/state/lane_registry.json` row `lane_atw_codec_v2_*` | 7-gate maturity per lane_maturity.py | Lane registry validator + autopilot routing | #90 + #126 |

### 21.2 Observability invariants

Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #221 fail-closed for score-claim artifacts:

- **D4 probe artifact**: `score_claim=false`, `evidence_grade=diagnostic_cpu`, `axis_label="[diagnostic-CPU; H(latent|scorer_class) probe]"`. NEVER promoted to score authority.
- **ATW v2 smoke artifact**: `score_claim=false` until paired CPU+CUDA full anchor lands. Smoke produces TRAINING SIGNAL only.
- **ATW v2 full anchor artifact**: paired `[contest-CUDA Modal A100]` + `[contest-CPU GHA Linux x86_64]` per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA". `score_claim=true` only when both axes land on byte-identical archive.
- **Predicted band**: `[prediction; first-principles-bound; D4-probe-conditional]` until empirical confirmation.
- **No phantom device-named directories** per Catalog #249. ATW v2 output directory is `contest_auth_eval_work/` (device-agnostic) per the canonical auth_eval pattern.
- **Distinguishing-feature byte-mutation proof**: per Catalog #272 the proof artifact lists per-section verdict (PASSED / FAILED / INFRASTRUCTURE_ERROR). FAILED on any of `WZ_HEAD_BLOB / DISTILL_HEAD_BLOB / LATENT_RESIDUAL_BLOB / CLASS_PRIOR_TABLE_BLOB / CDF_TABLE_BLOB` BLOCKS L2+ promotion.

### 21.3 Observability hooks for cathedral autopilot

- **Hook 4 (Cathedral autopilot dispatch hook)**: ATW v2 recipe `min_smoke_gpu=A100` + `target_modes=[research_substrate]` + `canary_status=independent_substrate` (no canary dependency at V2 since D4 probe IS the pre-dispatch gate). Autopilot consumes the D4 probe verdict to gate dispatch eligibility.
- **Hook 5 (Continual-learning posterior update)**: ATW v2 full anchor result (when landed) seeds `tac.continual_learning.posterior_update_locked` paired with the empirical D4 MI value, so the posterior captures the mapping `(D4 MI, predicted band, empirical ΔS)`. Future substrates with similar cooperative-receiver framing benefit from this anchor.
- **Hook 1 (Sensitivity map)**: `tac.sensitivity_map.scorer_class_conditional_v2` contributes per-pair latent sensitivity to scorer class index (computable from D4 probe MI gradient).
- **Hook 2 (Pareto constraint)**: `tac.pareto.atw_v2_wz_residual_entropy` contributes `R_WZ ≥ H(latent | scorer_class)` to the global Pareto solver.
- **Hook 3 (Bit-allocator hook)**: `bit_allocator.atw_v2_wz_residual_v1` allocates archive bytes per-pair based on scorer-class entropy contribution.
- **Hook 6 (Probe-disambiguator)**: D4 probe + three-knob Variant A regime sweep + Variant A vs B paired smoke (per §19).

All 6 hooks active per Catalog #125; none N/A.

### 21.4 Operator-facing observability commands

```bash
# D4 probe execution (STAGE 0 PRE-DISPATCH GATE):
.venv/bin/python tools/probe_latent_conditional_entropy_h_latent_given_scorer_class.py \
    --substrate-id atw_v2 \
    --latent-bytes <A1_latent_bytes_path> \
    --scorer-classes <SegNet_class_per_pair_bytes_path> \
    --output-json .omx/state/h_latent_given_scorer_class_atw_v2.json

# Dykstra-feasibility check (sister, $0):
.venv/bin/python tools/check_substrate_dykstra_feasibility.py --substrate atw_v2 \
    --output-json .omx/state/dykstra_feasibility_atw_v2.json

# Lane maturity audit:
.venv/bin/python tools/lane_maturity.py audit | grep atw_v2

# Cathedral autopilot ranker consumption:
.venv/bin/python tools/cathedral_autopilot_autonomous_loop.py --consume-d4-probe atw_v2 --rerank

# Distinguishing-feature byte-mutation proof (post-archive build):
.venv/bin/python tools/verify_distinguishing_feature_byte_mutation.py \
    --lane lane_atw_codec_v2_cooperative_receiver_full_stack_design_20260516 \
    --archive-zip <archive_path> \
    --inflate-sh <inflate.sh_path>
```

---

## 22. Op-routables (ranked)

| # | Op-routable | Cost | Wall-clock | Owner | Dependency |
|---|---|---|---|---|---|
| **1** | Execute D4 H(latent\|scorer_class) probe on A1 latents | $3-5 | 30 min CPU | Operator or sister-subagent | Independent — runs immediately |
| **2** | Pre-register lane `lane_atw_codec_v2_cooperative_receiver_full_stack_design_20260516` at L0 SKETCH | $0 | 2 min | Operator (via `tools/lane_maturity.py add-lane`) | Per Catalog #126 |
| **3** | Variant A vs B council adjudication memo (sextet pact per Catalog #292) | $0 | 1 deliberation cycle | Council | Op-routable #1 (D4 verdict required as input) |
| **4** | Build ATW v2 substrate package `src/tac/substrates/atw_codec_v2/` (architecture + archive + inflate + score_aware_loss + registered_substrate) | $0 (editor) | ~4h | Sister-subagent (NOT this subagent per Catalog #230 READ-ONLY scope) | Op-routable #3 (council decision binds Variant) |
| **5** | Build ATW v2 trainer `experiments/train_substrate_atw_codec_v2.py` with `_smoke_main` + `_full_main` (PROPER implementation, NOT NotImplementedError) | $0 (editor) | ~3h | Sister-subagent | Op-routable #4 |
| **6** | Build ATW v2 recipe `.omx/operator_authorize_recipes/substrate_atw_codec_v2_modal_a100_dispatch.yaml` | $0 (editor) | ~1h | Sister-subagent | Op-routable #5 |
| **7** | Phase 2 council lift of `_full_main NotImplementedError` (if any 2 of V1 4-criterion gate + V2-1 + V2-2 satisfied) | $0 | 1 deliberation cycle | Council | Op-routables #1, #3 |
| **8** | ATW v2 Modal A100 smoke ($5-10 Stage 1) | $5-10 | 1.5-2hrs | Operator-authorize | Op-routables #4-7 + D4 probe MEANINGFUL_CONDITIONING |
| **9** | ATW v2 Modal A100 full anchor + paired CPU+CUDA ($10-30 Stage 2) | $10-30 | 6-8hrs | Operator-authorize | Op-routable #8 smoke green |
| **10** | Three-knob Variant A regime sweep (council-grade $30 paired) | $20-30 | 6-8hrs | Operator-authorize (council-grade) | Op-routable #9 OR Op-routable #8 (independent if council decides Variant A) |
| **11** | ATW v2 ⊕ DP1 composition smoke | $5-10 | 1.5-2hrs | Operator-authorize | Op-routable #9 |
| **12** | A-STACK[NSCS03 → ATW v2] SWAP smoke (per A-STACK §13 row 11) | $5-10 | 1.5-2hrs | Operator-authorize | Op-routable #9 + A-STACK individual smokes |
| **13** | G2-PARTIAL V3 promotion council deliberation (if MEANINGFUL_CONDITIONING confirmed AND ATW v2 full anchor lands ΔS in predicted band) | $0 | 1 deliberation cycle | Council | Op-routable #9 |
| **14** | Wire ATW v2's posterior anchor into `tac.continual_learning.posterior_update_locked` per Hook 5 | $0 | 30 min | Sister-subagent (post-anchor) | Op-routable #9 lands paired anchor |
| **15** | Extend `tac.composition.registry.canonical_primitive_inventory()` with ATW v2 entry per Catalog #169 | $0 | 15 min | Sister-subagent | Op-routables #4-6 |

**Recommended FIRST op-routable per Race-mode rigor inversion rule 3**: Op-routable #1 (D4 probe) — cheap ($3-5), fast (30 min), and the disambiguator-gate for the entire dispatch decision chain.

---

## 23. Cross-references

- `.omx/research/atw_codec_atick_tishby_wyner_v1_design_20260515.md` — V1 design memo (predecessor)
- `.omx/research/atw_codec_v1_cargo_cult_unwind_design_20260516.md` — V1 HIGH-RISK unwind (predecessor unwind; commit `ae6986c04`)
- `.omx/research/a_stack_nscs01_02_03_composition_full_stack_design_20260516.md` §13 row 11 — A-STACK[swap to ATW] reconciliation
- `.omx/research/nscs06_v8_path_b_wavelet_residual_full_stack_design_20260516.md` — NSCS06 v8 Path B composition opportunity
- `.omx/research/grand_council_symposium_time_traveler_optimal_staircase_20260516.md` — L5 v2 staircase Step B2 anchor + Path 2 lattice rewrite
- `feedback_wunderkind_visionary_scorer_as_cooperative_receiver_paradigm_shift_20260515.md` — G1 + B3 + G2-PARTIAL substitution candidates (parent Claude memory)
- `feedback_contest_compliance_canonical_constraints_for_wunderkind_and_all_subagents_NON_NEGOTIABLE_20260515.md` — contest-compliance constraint (parent Claude memory)
- `feedback_z4_atick_redlich_minimum_viable_landed_20260515.md` — Z4 sister substrate (β-only branch)
- `feedback_assumptions_challenge_audit_break_out_local_minima_landed_20260515.md` — 18-assumption matrix + the 0.196-0.199 cluster empirical anchor
- `feedback_pr95_lesson_now_at_meta_level_unique_and_complete_per_method_default_20260515.md` — UNIQUE-AND-COMPLETE-PER-METHOD standing directive
- `feedback_canonical_share_when_serves_unique_when_suppresses_standing_directive_20260515.md` — canonical-vs-unique decision framework
- `tools/probe_latent_conditional_entropy_h_latent_given_scorer_class.py` — D4 probe (commit `d72f50985`)
- `src/tac/codec/cooperative_receiver/atick_redlich.py` — canonical Atick-Redlich primitive (the V2 substrate consumes via Catalog #164)
- `src/tac/codec/cooperative_receiver/predictive_coding.py` — sister Rao-Ballard primitive (not consumed by V2; alternative-class candidate)
- `src/tac/substrates/z4_cooperative_receiver_loss/` — Z4 sister substrate (β-only branch; canary dependency)
- `src/tac/substrates/d4_wyner_ziv_frame_0/` — D4 sister substrate (WZ on frame_0)
- `src/tac/substrates/atw_codec_v1/` — V1 substrate package (sister; V2 is a new package, NOT a rename)
- `experiments/train_substrate_atw_codec_v1.py` — V1 trainer (sister; V2 needs new trainer)
- `.omx/operator_authorize_recipes/substrate_atw_codec_v1_modal_a100_dispatch.yaml` — V1 recipe (sister)
- CLAUDE.md non-negotiables: "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + "HNeRV / leaderboard-implementation parity discipline" + "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" + "Apples-to-apples evidence discipline" + "Submission auth eval — BOTH CPU AND CUDA" + "Forbidden premature KILL" + "Race-mode rigor inversion + parallel-dispatch first" + "META-ASSUMPTION ADVERSARIAL REVIEW" + "Council conduct" (Assumption-Adversary seat sextet pact)
- CLAUDE.md Catalogs: #5 eval_roundtrip / #6 strict-scorer-rule / #88 EMA / #90 lane registry / #124 archive grammar / #125 6-hook wire-in / #126 lane pre-registration / #128 fcntl-locked posterior writes / #139 no-op detector / #146 inflate per-video loop / #152 required-input-file validation / #157 commit-serializer pre-pre-lock / #164 canonical scorer-loss helper / #167 smoke-before-full / #169 canonical primitive inventory / #170 min_vram_gb / #172 autocast_fp16 / #174 expected-content-sha256 mandatory / #176 operator-authorize canonical / #178 TF32 / #179 torch.compile / #180 no_grad-at-eval / #181 pyav_decode_strategy / #182 target_modes / #185 LIVE_COUNT drift / #190 hardware substrate / #192 macOS-CPU advisory / #205 select_inflate_device / #206 subagent checkpoint discipline / #209/#210/#211/#213 DP1 / #215 min_smoke_gpu / #220 substrate L1 scaffold operational mechanism / #221 auth_eval fail-closed / #222 scorer-loader assignment order / #226 auth_eval canonical helper / #227 substrate class-shift Tier C / #229 premise verification before edit / #230 sister-subagent ownership map / #240 recipe-vs-trainer-state / #243 local pre-deploy harness / #244 canonical NVML/CUDA env block / #245 Modal call_id ledger / #246 paired dispatch skip if anchor exists / #248 no stash-pop conflict markers / #249 no phantom device-named directories / #270 dispatch optimization protocol / #271 PRE-DISPATCH codex review / #272 distinguishing-feature integration contract / #279/#280/#281/#282/#283 fail-closed protections / #289 commit serializer drop-flag retry / #290 canonical-vs-unique decision per layer / #291 META-ASSUMPTION review cadence / #292 grand council explicit assumption-statements / #294 9-dimension success checklist / #296 substrate predicted band Dykstra-feasibility / #297 signal-axis destruction reversibility / #303/#304 (most-recent)

---

**Status**: DESIGN-ONLY LANDED 2026-05-16. Phase 2 council approval REQUIRED to lift `_full_main NotImplementedError` (when sister-subagent builds the V2 substrate package per Op-routable #4-7). RESEARCH-ONLY at recipe level until D4 probe returns `MEANINGFUL_CONDITIONING` + Dykstra-feasibility check + Variant A vs B council adjudication.

**6-hook wire-in declaration (per Catalog #125)**:
1. Sensitivity-map contribution: ACTIVE — `tac.sensitivity_map.scorer_class_conditional_v2` planned (consumes D4 MI gradient).
2. Pareto constraint: ACTIVE — `tac.pareto.atw_v2_wz_residual_entropy` planned (R_WZ ≥ H(latent | scorer_class) constraint).
3. Bit-allocator hook: ACTIVE — `bit_allocator.atw_v2_wz_residual_v1` planned (per-pair archive bytes by scorer-class entropy).
4. Cathedral autopilot dispatch hook: ACTIVE — recipe registered warn-only at landing per Catalog #167; promotes to dispatch-eligible upon D4 MEANINGFUL_CONDITIONING.
5. Continual-learning posterior update: ACTIVE — full anchor seeds posterior paired with D4 MI value per Catalog #128 locked write.
6. Probe-disambiguator: ACTIVE — D4 probe + three-knob Variant A regime sweep + Variant A vs B paired smoke per §19.

**Sister-subagent ownership map** (Catalog #230): this subagent READ-ONLY on source code; writes ONLY this memo + 1 commit + checkpoints. Sister subagents (post-D4-verdict) build V2 substrate package + trainer + recipe per Op-routables #4-7.

**Premise verification per Catalog #229** (9 PVs verified in §1 BEFORE any design statement).

**Checkpoint discipline per Catalog #206** (3+ checkpoints written to `.omx/state/subagent_progress.jsonl`).

**Commit via canonical serializer with `--expected-content-sha256`** per Catalog #157 + #174 + #289 (the 92aba3ca commit-swap class permanent fix).
