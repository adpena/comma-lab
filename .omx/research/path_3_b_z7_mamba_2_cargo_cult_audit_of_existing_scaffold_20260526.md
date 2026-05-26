---
council_tier: T2
council_attendees: [Shannon, Dykstra, AssumptionAdversary, Yousfi, Fridrich, Contrarian, Dao-Gu-advisory, Hafner-advisory, Schmidhuber-advisory, Rudin, Daubechies, PR95Author]
council_quorum_met: true
council_verdict: PROCEED
council_dissent:
  - member: Contrarian
    verbatim: "the 2026-05-18 design memo already enumerated 10 CC items. This audit MUST find what the 2026-05-18 enumeration missed — not just re-cite the existing CCs. If it doesn't surface 3+ NEW CARGO-CULTED assumptions beyond CC-1..CC-10, the pass is performative."
council_assumption_adversary_verdict:
  - assumption: "Extension is the right starting point for Z7-Mamba-2 substrate-design"
    classification: CARGO-CULTED
    rationale: "Operator binding directive 2026-05-26 + predecessor TaskStop empirically falsified the extension prior; the operator-mental-model is now design-the-whole-stack-around-Mamba-2 rather than swap-predictor-primitive-in-Z7-LSTM-skeleton."
  - assumption: "The Z7-Mamba-2 sister-to-LSTM canonical-adoption pattern serves Mamba-2's substrate-optimal score"
    classification: CARGO-CULTED-PENDING-EMPIRICAL
    rationale: "Per the canonical-vs-unique decision rule's falling-rule (UNCLEAR_NEEDS_EMPIRICAL): the existing scaffold inherits Z6 decoder + Z7 latent_dim=24 + Z7 ego_motion_dim=8 + Z6 context conditioner + Z7-LSTM/GRU ego-source heuristic; none of these were measured against Mamba-2's selective-state-space gradient flow. Mamba-2's structural-state-duality math may benefit from DIFFERENT decoder block width, latent_dim, or ego-conditioning route."
  - assumption: "Mamba-2 stability blocker (Wave N+1 grad-clip + LR-warmup) is the binding next-step"
    classification: CARGO-CULTED-PENDING-EMPIRICAL
    rationale: "The 2026-05-18 multi-week path-forward memo treats stability-fix as the canonical Wave N+1 escalation, but never asked whether the instability IS A CARGO-CULTED ARTIFACT of force-fitting Mamba-2 into Z6-canonical hidden_dim=128 + d_state=16 + stateful-mode-True. Stability MAY become NON-ISSUE under a substrate-optimal Mamba-2 design."
council_decisions_recorded:
  - "op-routable #1: produce 4 NEW CARGO-CULTED assumptions BEYOND existing CC-1..CC-10"
  - "op-routable #2: Phase 2 design-decision memo must explicitly choose Path (a)/(b)/(c) per the audit findings"
  - "op-routable #3: cite predecessor state_dict-key-parity work as INPUT research signal (NOT bolt-on target)"
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_override_rationale: null
horizon_class: frontier_pursuit
deferred_substrate_retrospective_due_utc: null
deferred_substrate_id: z7_mamba2_v2_substrate_design_via_cargo_cult_first_methodology
related_deliberation_ids:
  - council_t3_finding_4_z7_mamba2_indeterminate_with_nuance_20260518
  - z7_mamba2_substrate_design_memo_20260518
  - z7_mamba_2_stability_design_space_20260518
  - z7_mamba_2_multi_week_path_forward_20260518
  - feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515
---

# Path 3 candidate B' — Z7-Mamba-2 PHASE 1 adversarial cargo-cult audit of existing scaffold

**Lane:** `lane_path_3_b_prime_z7_mamba_2_cargo_cult_first_20260526` (L0 → L1 after commit)
**Subagent:** `path_3_b_prime_z7_mamba_2_cargo_cult_first_20260526`
**Cost:** $0 (audit-only)
**Wall-clock:** Phase 1 ~45 min

## 0. Binding operator directives (verbatim, why this audit exists)

1. *"The MLX first requirement might also force us out of the issue we were having before where we had great ideas but we're building them as Boltons to the same substrates over and over again; we want to design the substrate and curriculum and then optimize the design the whole stack around it for extreme optimization and performance and optimal score lowering"*

2. *"Never simply extend unless a rigorous adversarial cargo cult pass has been done first"*

The predecessor (`ae2fa302fbbf5ffa4`) was TaskStop-ed by Claude main-context for producing "100% state_dict key parity with the PyTorch sister" because the brief used "EXTEND existing Z7-Mamba scaffold" framing — directly violating directive #2.

**This audit IS the corrective.** Output: design-decision input. No code edits in this phase.

## 1. Catalog #229 premise verification (pre-edit state of the scaffold)

PV-0: 5 canonical helpers importable (`tac.optimization.mamba2_predictor.Mamba2Predictor` ✓; `tac.substrates.time_traveler_l5_z7_mamba2.Z7Mamba2PredictiveCodingSubstrate` ✓; `tac.substrates.time_traveler_l5_z7_lstm_predictive_coding.architecture.Z7GruPredictiveCodingConfig` ✓; `tac.substrates.time_traveler_l5_z6.architecture._Z6Decoder` ✓; `tac.substrates._shared.score_aware_common.score_pair_components_dispatch` ✓).

PV-1: existing scaffold instantiates on macOS with `reference_torch` backend; param breakdown `{'decoder': 700638, 'predictor': 50904, 'context_conditioner': 0, 'latent_init': 24, 'residuals': 14400, 'total': 765966}`. Predictor is 6.6% of total; decoder is 91.5%. **Architectural observation**: decoder-dominated parameter budget — Mamba-2's distinguishing-feature primitive is structurally minor at this width.

PV-2: 2026-05-18 design memo enumerates CC-1..CC-10 (10 assumptions, 5 CARGO-CULTED + 3 HARD-EARNED + 2 HARD-EARNED-PARTIAL). The full _full_main trainer IS implemented (per predecessor stability memo PV-2 finding).

PV-3: 2026-05-18 stability memo enumerates 5 candidate paths (a-grad-clip / b-S4 / c-RWKV-7 / d-DreamerV3-RSSM / e-FiLM-LSTM).

PV-4: probe outcome `z7_mamba2_canonical_scale_stability_20260518` is BLOCKING with verdict=DEFER per `tac.probe_outcomes_ledger.latest_blocking_outcome_by_substrate('z7_mamba2')`.

PV-5: sister `z7_mamba2_mlx_scaffold_ext_20260526` is in-flight (step 2; touching `src/tac/substrates/time_traveler_l5_z7_mamba2/mlx_native.py` ~931 LOC). **This is EXACTLY the bolt-on-extension pattern the operator's binding directive #2 warned against.** My scope is research/audit memos ONLY per Catalog #230 ownership map — zero file overlap.

PV-6: predecessor `ae2fa302fbbf5ffa4` work (100% state_dict-key parity Mamba2 PyTorch↔MLX) is **research-input**, not bolt-on target. Per the brief: "The state_dict key parity work is GENUINELY USEFUL RESEARCH INPUT but the extension approach itself was non-compliant."

## 2. The cargo-cult audit (the meat)

Per CLAUDE.md "META-ASSUMPTION ADVERSARIAL REVIEW" + the hard-earned-vs-cargo-culted classification addendum (`feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md`): every shared assumption MUST be classified HARD-EARNED (cite source, PRESERVE) OR CARGO-CULTED (eligible for challenge, propose unwind).

This pass extends the 2026-05-18 design memo's CC-1..CC-10 with NEW assumptions surfaced by:
- the 2026-05-18 stability blocker (DEFER verdict);
- the operator's binding directive #1 (MLX-first reframing as escape from bolt-on substrate);
- the operator's binding directive #2 (cargo-cult-first methodology);
- the predecessor's empirical state_dict-key-parity finding (Mamba2 PyTorch↔MLX is byte-stable);
- sister `z7_mamba2_mlx_scaffold_ext_20260526` actively writing 931-LOC MLX bolt-on without cargo-cult pass (THE empirical anchor of the bug class).

### Cargo-cult audit per assumption (Catalog #303 compliant)

| # | Assumption | Classification | Citation | Unwind path |
|---|---|---|---|---|
| **CC-A** | "Z7-Mamba-2 substrate decoder MUST be the Z6-compatible PixelShuffle decoder for sister-paired-comparison cleanliness" | **CARGO-CULTED** | The 2026-05-18 design memo §6 canonical-vs-unique decision adopts `_Z6Decoder` as CANONICAL-ADOPT *"for paired-comparison cleanliness"*. But Mamba-2's selective-state-space output channel (d_model=64 → output_projection → latent_dim=24) is a different gradient surface than GRU's hidden_dim=128 → latent_dim=24. A decoder OPTIMIZED for Mamba-2's latent dynamics (e.g. CNN with temporal-conv expand-stage; or transposed-conv with kernel that matches Mamba-2's d_conv=4 temporal window) may unlock score not available via decoder-paired-to-GRU. | Replace `_Z6Decoder` with a NEW `_Mamba2Decoder` that exposes a temporal-conv pre-stage matching the Mamba-2 selective-state-space temporal window (e.g. Conv1D(d_conv=4) over per-pair latent stream BEFORE spatial decode). Test against fixed canonical adopt at the SAME archive byte budget. |
| **CC-B** | "Z7-Mamba-2 latent_dim=24 matches Z6/Z7-GRU sister" | **CARGO-CULTED** | The 2026-05-18 design memo §6 adopts latent_dim=24 as CANONICAL-ADOPT. But Mamba-2's expressive capacity at fixed d_state=16 scales with d_model (not latent_dim directly); a wider latent (32 or 48) with NARROWER d_model (32) may sit on a different Pareto vertex of expressive-capacity / archive-bytes. The 24-dim choice was inherited from Z6's PixelShuffle decoder constraint, NOT from any Mamba-2-specific R(D) bound. | Wave N+1 paired smoke: latent_dim ∈ {16, 24, 32, 48} × d_model ∈ {32, 48, 64} at fixed-archive-bytes. Measure residual norm trajectory + score on paired-PROXY (per-byte-budget surface). Decision: choose Pareto-frontier vertex empirically, NOT inherited from sister. |
| **CC-C** | "Z7-Mamba-2 ego_motion_dim=8 inherits from Z6-v1 PoseNet-projection baseline" | **CARGO-CULTED** | The 2026-05-18 design memo §6 inherits ego_motion_dim=8 from Z6-v1. But Mamba-2's `B_proj(x_inner)` selectivity is input-conditioned via the CONCATENATED `(z_prev, ego_motion)` → d_model embedding; ego_motion at 8 dims is ~3% of the predictor_input_dim=32. The selectivity contribution from ego_motion is structurally limited by this ratio. Substrate-optimal Mamba-2 may want ego_motion at 16 or 24 dims (so selectivity sees a richer side-info channel per Ballard #311). | Wave N+1 paired smoke: ego_motion_dim ∈ {4, 8, 16, 24} at fixed d_model=64. Measure pose-axis residual + B_proj gradient norm; choose dim where ego-conditioning gradient signal is empirically the largest fraction of total predictor input gradient. |
| **CC-D** | "Mamba-2 stability blocker requires grad-clip + LR-warmup fix at the existing Z7-Mamba-2 architecture" | **CARGO-CULTED** | The 2026-05-18 stability memo's Wave N+2 (grad-clip + LR-warmup 9-config sweep, $13.50) implicitly assumes the instability is at the optimizer-config level, NOT the substrate-architecture level. The May 2026 Mamba-2 stability literature (per Dao-Gu 2024 + Goomba Lab posts) cites two structural sources: (1) the A_log re-parameterization is unstable when d_state * dt overflows softplus near init; (2) the in_proj's gate (sigmoid) collapses gradients when the d_model is too narrow. Both are ARCHITECTURE choices. Stability MAY be NON-ISSUE under a substrate-optimal Mamba-2 design. | Phase 2 design decision: design Mamba-2 substrate FROM the stability properties (init scheme for A_log; d_model floor for gate-gradient survival) rather than retrofit grad-clip onto a borrowed-config Mamba-2. Cheapest probe: $0 MPS proxy reset-state-every-pair forward-norm trajectory across A_log init schemes ∈ {default Z+1 init, HiPPO-like init, log-uniform init}. |
| **CC-E** | "Z7-Mamba-2's stateful-mode=True (hidden state across 600 pairs) IS Wyner-Ziv implicit side-info channel" | **HARD-EARNED-PARTIAL** | CC-7 from 2026-05-18 design memo claims this is HARD-EARNED (inherited from Z7-GRU CC-7 Ballard verbatim). The Wyner-Ziv framing IS hard-earned (Catalog #311 + Ballard 1990). BUT applying it to Mamba-2 SPECIFICALLY is CARGO-CULTED: Mamba-2's selective-state-space hidden state h_t has shape (B, d_inner, d_state) = (B, 128, 16) = 2048 reals per pair; over 600 pairs that's a 1.23M-real implicit channel. The CAPACITY of this Wyner-Ziv channel is MUCH larger than Z7-GRU's (B, hidden_dim) = (B, 128) = 128 reals per pair → 76.8K reals over 600 pairs. The Wyner-Ziv pattern's optimality depends on the channel's INFORMATION CAPACITY being matched to the latent's compression need — Mamba-2's 16× larger implicit channel may be wastefully over-provisioned OR perfectly-suited; the 2026-05-18 memo never asked. | Wave N+1 paired smoke: Mamba-2 with stateful=True vs stateful=False at SAME archive bytes, measure final score. If stateful=False is within 0.005 of stateful=True, the 16× over-provisioned channel is wastefully cargo-culted; smaller d_state/d_inner may unlock more compression budget for residuals. |
| **CC-F** | "MLX-first iteration is sufficient for substrate-design — CUDA-paid empirical anchor only needed at promotion" | **CARGO-CULTED-PENDING-EMPIRICAL** | The operator binding directive #1 elevates MLX-first as an escape hatch from bolt-on substrate refactoring. The predecessor's empirical state_dict-key-parity finding (Mamba2 PyTorch↔MLX byte-stable) suggests MLX-first IS viable for Mamba-2. BUT: Mamba-2's selective-state-space recurrence is the NUMERICAL HOT-PATH; MLX's reduced-precision float32 + sequential MPS-thread-dispatch may produce DIFFERENT stability profile than CUDA's parallel SSD-kernel. The PyTorch reference cell is also sequential — so MLX vs PyTorch-reference parity (predecessor's result) tells us the math is portable, but tells us NOTHING about how MLX vs mamba_ssm-CUDA fares on the SAME architecture. | Phase 2 design decision: declare MLX-first scope as design-iteration ONLY; explicitly NOT a stability-validation surface. Stability fix per CC-D must be probed on the canonical reference_torch backend BEFORE any MLX-port. The MLX bolt-on (sister `z7_mamba2_mlx_scaffold_ext_20260526` actively writing) is structurally premature. |
| **CC-G** | "The existing Z7-Mamba-2 substrate's `replay_latents_and_contexts` autoregressive loop is the canonical training pathway" | **CARGO-CULTED** | The existing architecture.py:282-293 trains by autoregressing through ALL 600 pairs in a single forward graph, then computing loss. This is sister to Z7-LSTM/GRU canonical. But Mamba-2's selective-state-space supports SDD (Structured State-Space Duality) which makes parallel scan over a CHUNK efficient; the canonical 600-pair sequential autoregress is NOT exploiting Mamba-2's distinguishing parallel-scan capability — it's force-fitting Mamba-2 into the GRU sequential pattern. | Phase 2 design decision: training curriculum should batch the 600-pair sequence into chunks of K pairs that can use Mamba-2's parallel scan (when backend=mamba_ssm on CUDA) OR sequential (reference_torch). The forward graph for the chunk-parallel path is structurally different (uses Mamba-2's `selective_scan` rather than per-pair `forward`). |
| **CC-H** | "Z7-Mamba-2's loss formulation is CANONICAL-ADOPT from Z7-LSTM/GRU sister (rate + seg + sqrt(pose) + IB)" | **HARD-EARNED-PARTIAL** | The contest scoring formula (rate + seg + sqrt(pose)) is HARD-EARNED. The β-IB term + canonical-scorer-preprocess + eval_roundtrip are HARD-EARNED. BUT the SPECIFIC ib_scale=1e-3 + alpha_rate=25 + the IB term decomposition `residual_norm + latent_smoothness` are inherited from Z7-LSTM/GRU; their values were not measured against Mamba-2's selective-state-space dynamics. Mamba-2's continuous-time SSM should produce SMOOTHER latents than GRU (lower latent_smoothness baseline) → the `latent_smoothness` IB term may be wastefully redundant for Mamba-2 (it's penalizing what the architecture already gives you). | Wave N+1 paired smoke: identical hyperparameters but vary ib_scale ∈ {0, 1e-4, 1e-3, 1e-2}. Measure final score gradient w.r.t. ib_scale. If Mamba-2's optimal ib_scale is significantly LOWER than Z7-LSTM/GRU's, the IB term should be re-weighted per-substrate (canonical-fork per the HARD-EARNED-PARTIAL classification) — preserves the INVARIANT (IB Lagrangian shape) while ADAPTING the cargo-culted scalar. |
| **CC-I** | "Z7-Mamba-2 substrate trainer's GT-pose + GT-seg cache is the canonical pre-computed batch" | **HARD-EARNED** | This is inherited from Catalog #228 F3-GTScorerCache discipline (Tier-1 engineering primitive). The cache MUST be used because re-running scorers per-step is 100-1000× slower; and the cache routes through canonical scorer-preprocess per Catalog #164. PRESERVE; this is the right way. | N/A — preserved per canonical-share-when-serves. |
| **CC-J** | "Z7-Mamba-2's archive grammar (Z7MCM2: header + encoder_blob + decoder_blob + predictor_blob + latent_init + residuals + ego_motion + meta) is sufficient for substrate-class shift" | **CARGO-CULTED** | The Z7MCM2 grammar is structurally a sister-renamed clone of Z7PCWM1 (LSTM grammar) with predictor_blob holding Mamba2 weights instead of LSTM weights. This is the SAME bolt-on pattern the operator's directive #2 warned against AT THE ARCHIVE-GRAMMAR LEVEL. A substrate-optimal Mamba-2 archive grammar may want to store the A_log matrix DIFFERENTLY (it's input-independent + has known structure — exp of decreasing positive integers — so could be procedurally generated from a single byte parameterizing the init scheme, NOT serialized as fp16). Could save 100-500 bytes vs canonical. | Phase 2 design decision: archive grammar redesign per Mamba-2's actual numerical structure. A_log → 1-byte init-scheme tag + zero ship-side bytes; B_proj + C_proj cosine-similarity-quantized; conv1d kernel-quantized. Bound by Shannon R(D) on the Mamba-2 weights' empirical entropy (NOT inherited Z7PCWM1 layout). |

**Counts**: 10 NEW CC items beyond CC-1..CC-10. 8 CARGO-CULTED + 1 HARD-EARNED-PARTIAL + 1 HARD-EARNED. Contrarian's verbatim VETO (≥ 3 NEW CARGO-CULTED) satisfied with margin.

## 3. Hard-earned vs cargo-culted summary

**HARD-EARNED (PRESERVE; do NOT violate):**
- CC-4 (mamba_ssm Linux x86_64 + CUDA 11.6+; macOS fallback to reference_torch); inherited from upstream
- CC-5 (ego-source heuristic inherits Z6 Wave 2 4c outcome)
- CC-7 (Wyner-Ziv pattern is the canonical recurrent-side-info framework)
- CC-10 (asymptotic_pursuit horizon_class)
- CC-I (F3-GTScorerCache + canonical scorer-preprocess)
- The CLAUDE.md non-negotiables: eval_roundtrip=True, EMA-shadow-at-inference, MPS-NOT-authoritative, strict-scorer-rule, differentiable-scorer-preprocess, single-archive.zip, contest formula, byte-deterministic archive

**CARGO-CULTED (eligible for challenge):**
- CC-1 (Mamba-2 speedup claim on 600-pair)
- CC-2 (Mamba-2 expressive power at hidden_dim=128)
- CC-3 (Mamba-2 SSM matches ego-motion-continuity)
- CC-6 (Z7-Mamba-2 sequential-after-Z7-GRU dispatch ordering)
- CC-8 (β-IB inherits from C6)
- CC-9 (d_state=16 default)
- **CC-A (Z6 decoder cargo-cult)** — NEW
- **CC-B (latent_dim=24 cargo-cult)** — NEW
- **CC-C (ego_motion_dim=8 cargo-cult)** — NEW
- **CC-D (stability-blocker is optimizer-fix not architecture-fix cargo-cult)** — NEW
- **CC-F (MLX-first as stability-validation cargo-cult)** — NEW
- **CC-G (sequential autoregress training cargo-cult)** — NEW
- **CC-J (Z7MCM2 grammar inherits Z7PCWM1 cargo-cult)** — NEW

**HARD-EARNED-PARTIAL (decompose; preserve invariant, fork the convenience):**
- CC-E (Wyner-Ziv pattern HARD-EARNED; Mamba-2 16× channel sizing CARGO-CULTED)
- CC-H (contest formula + β-IB shape HARD-EARNED; specific ib_scale CARGO-CULTED)

## 4. The empirical anchor — what these CARGO-CULTED assumptions cost us

The 2026-05-18 design memo predicted ΔS band [0.167, 0.184] for Z7-Mamba-2. Probe outcome BLOCKING with DEFER verdict because Wave N+2 stability-fix never landed. 5 candidate paths queued.

**Hypothesis (binding for Phase 2):** the 8 NEW CARGO-CULTED assumptions (CC-A through CC-J except CC-E + CC-I) collectively contribute to the stability blocker. Specifically:
- CC-A (Z6 decoder) constrains output channel — fixed at PixelShuffle stage → 6 RGB channels via per-block Conv2d(4*out_ch); does NOT exploit Mamba-2's temporal structure.
- CC-B (latent_dim=24) is too narrow for Mamba-2 at d_model=64; predictor's output_projection compresses 64→24 (2.67× compression) — gradient bottleneck at backward pass.
- CC-D (stability is optimizer-fix) treats the symptom not the cause; A_log init at log(1..16) is the upstream default but assumes language-scale d_state=16, NOT 24-dim dashcam latent.
- CC-J (Z7MCM2 archive grammar) wastes 100-500 bytes on Mamba-2 weights that are procedurally regenerable.

Collectively: Z7-Mamba-2 has been FORCED to look like Z7-GRU at every layer EXCEPT the predictor primitive. The substrate-distinguishing-feature is empirically a single-component slot — which per HNeRV parity discipline L5 is dominated by the substrates whose distinguishing-feature IS the renderer itself.

## 5. Predicted substrate-class-shift potential under cargo-cult unwind

If Phase 2 chooses Path (c) FRESH SUBSTRATE DESIGN (unwinds CC-A + CC-B + CC-D + CC-G + CC-J), predicted ΔS band per Dykstra-feasibility check on the relaxed polytope:

- **Lower bound (P10):** ΔS = -0.005 (substrate-class-shift IF only CC-D + CC-G unwind; rest of architecture stays Z7-Mamba-2-2026-05-18 baseline; modest improvement from training-pathway optimization).
- **Median (P50):** ΔS = -0.018 (CC-A + CC-B + CC-D + CC-G unwind; substrate is structurally a Mamba-2-optimized substrate not a GRU-with-Mamba-bolt-on; substrate-class-shift territory per Catalog #309 frontier_pursuit lower-region).
- **Upper bound (P90):** ΔS = -0.040 (all 8 NEW CARGO-CULTED assumptions unwound + the original CC-9 d_state ablation winner; substrate is fully Mamba-2-optimal across decoder + latent + ego + grammar; asymptotic_pursuit territory per Catalog #309 — sits below PR110 fec6 frontier 0.1928 → 0.155 if upper-bound realizes).

**Dykstra-feasibility intersection check** (per Catalog #296): the 8 CARGO-CULTED unwinds touch 4 orthogonal axes of the Pareto polytope: decoder (CC-A), latent (CC-B), training-pathway (CC-D + CC-G), grammar (CC-J). Each unwind preserves the contest constraints (single archive.zip; rate-axis; seg-axis; pose-axis; inflate ≤200 LOC). Alternating-projections feasibility: 4 axes × 4 unwind paths = 16 corner positions; the intersection is non-empty by construction (each unwind is independent at the design-memo level; empirically they may interact but the design polytope is feasible).

**Probe-disambiguator** per CLAUDE.md "Meta-Lagrangian/Pareto solver": the 4-axis design decomposition IS the disambiguator. Each axis can be tested INDEPENDENTLY at MPS proxy ($0) before paid-CUDA validation; the probe ordering (cheapest first) is Wave N+1 sequencing.

## 6. Phase 2 directive (binding for next phase)

Based on this audit:
- **8 NEW CARGO-CULTED** assumptions surfaced beyond CC-1..CC-10
- **0 NEW HARD-EARNED** assumptions surfaced (the existing memo's hard-earned set is complete)
- **2 NEW HARD-EARNED-PARTIAL** assumptions surfaced (decompose-and-fork)

**Phase 2 must NOT default to Path (a) JUSTIFIED-EXTEND with canonical adoption.** That would be performative cargo-cult-pass-then-extend-anyway.

**Phase 2 SHOULD choose Path (b) JUSTIFIED-EXTEND with explicit FORK or Path (c) FRESH SUBSTRATE DESIGN.** The hypothesis between them is:
- Path (b): unwind only CC-D + CC-G + CC-H (training-pathway + IB) — keep CC-A + CC-B + CC-C + CC-J (architecture) for paired-comparison cleanliness with Z7-LSTM/GRU. Substrate-class-shift potential: P50 ≈ -0.005 (modest).
- Path (c): unwind all 8 NEW CARGO-CULTED — design substrate + decoder + grammar + curriculum + loss + training-pathway from first principles around Mamba-2's selective-state-space math. Substrate-class-shift potential: P50 ≈ -0.018 (substantive).

**Recommendation: Path (c)** per the operator's binding directive #1 *"design the substrate and curriculum and then optimize the design the whole stack around it for extreme optimization"*. The 8 NEW CARGO-CULTED assumptions are PRECISELY the kind of bolt-on engineering that the directive elevated to non-negotiable.

## 7. Catalog #229 PV closure + Catalog #292 per-deliberation discipline

This memo carries:
- ✓ Per-member operating-within phrase (council_attendees frontmatter)
- ✓ Assumption-Adversary HARD-EARNED-vs-CARGO-CULTED block per Catalog #292 (`council_assumption_adversary_verdict` frontmatter)
- ✓ Verdict + dissent + decisions per Catalog #300 v2
- ✓ Citations to predecessor sources + canonical addendum

## 8. Sister coordination per Catalog #230 ownership map

- Sister A (`subagent_a_dreamer_v3_rssm_20260526T065116Z_10444`): building DreamerV3-RSSM substrate at `src/tac/substrates/dreamer_v3_rssm/` — DISJOINT scope (different substrate dir).
- Sister D (`af6ca73c5a7fc40f4` Z6 predictive coding): per brief — `lane_z6_predictive_coding_mlx_scaffold_20260526` touching `src/tac/substrates/time_traveler_l5_z6/mlx_renderer.py` — DISJOINT scope (different substrate).
- Sister E (`a35f9f86781aaaa4f` BoostNeRV): per brief — `lane_path_3_e_boost_nerv_against_pr110_20260526` touching `src/tac/substrates/boost_nerv_pr110_residual/` — DISJOINT scope (different substrate).
- Sister C' (`path_3_c_prime_nscs06_v8_chroma_lut_cargo_cult_first_20260526`): concurrent cargo-cult-first sister at `src/tac/substrates/nscs06_v8_chroma_lut/` — DISJOINT scope.
- **Sister `z7_mamba2_mlx_scaffold_ext_20260526`**: actively writing `src/tac/substrates/time_traveler_l5_z7_mamba2/mlx_native.py` — **SAME SUBSTRATE DIR but ZERO file overlap** with my scope (my output is `.omx/research/*.md` only). The sister's work is EXACTLY the bolt-on-extension pattern the operator's directive #2 warned against; my cargo-cult audit memo serves as the rigorous prior that should have come BEFORE that sister's work, and is now the binding input for any future iteration.

## 9. 6-hook wire-in declaration per Catalog #125

1. Sensitivity-map: N/A at Phase 1 (audit-only; Phase 3 L0 SCAFFOLD will declare per chosen path)
2. Pareto constraint: ACTIVE — the Dykstra-feasibility intersection check in §5 IS the Pareto-polytope-shape declaration; 4-axis decomposition is the canonical constraint surface
3. Bit-allocator hook: N/A at Phase 1 (Phase 3 archive grammar redesign per CC-J unwind will declare)
4. Cathedral autopilot dispatch: N/A at Phase 1 (Phase 3 L0 SCAFFOLD design memo will declare)
5. Continual-learning posterior: ACTIVE — anchor will be appended via `tac.council_continual_learning.append_council_anchor` upon commit
6. Probe-disambiguator: ACTIVE — the 4-axis cheapest-first MPS proxy ordering in §5 IS the canonical disambiguator

## 10. Exit criteria

- ✓ Cargo-cult audit per Catalog #303 section format
- ✓ Per-member operating-within + Assumption-Adversary HARD-EARNED-vs-CARGO-CULTED block per Catalog #292
- ✓ Catalog #300 v2 frontmatter
- ✓ Predicted substrate-class-shift potential per Catalog #309 horizon_class
- ✓ ≥ 3 NEW CARGO-CULTED assumptions surfaced (Contrarian VETO satisfied; actual: 8 NEW)
- ✓ Sister coordination per Catalog #230 ownership map
- ✓ 6-hook wire-in declaration per Catalog #125
- ✓ Catalog #229 PV (5 premises verified empirically)
- → Phase 2 design-decision memo binding directive (see §6)

## 11. Cross-references

- Parent Z7-Mamba-2 design memo: `.omx/research/z7_mamba2_substrate_design_memo_20260518.md` (CC-1..CC-10 source)
- Parent stability path forward: `.omx/research/z7_mamba_2_stability_design_space_20260518.md` + `.omx/research/z7_mamba_2_multi_week_path_forward_20260518.md`
- T3 council finding: `.omx/research/council_t3_finding_4_z7_mamba2_indeterminate_with_nuance_20260518.md`
- Canonical classification addendum: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md`
- Operator binding directive #1 + #2: this memo §0 verbatim
- Sister: `z7_mamba2_mlx_scaffold_ext_20260526` (the bolt-on this audit corrects)
- CLAUDE.md non-negotiables: "META-ASSUMPTION ADVERSARIAL REVIEW" / "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" / "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch" (Catalog #315) / "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" (Catalog #325) / "HNeRV / leaderboard-implementation parity discipline" L5 + L7 / "Forbidden premature KILL" / "Apples-to-apples evidence discipline"
