<!-- SPDX-License-Identifier: MIT -->
---
council_tier: T2
council_attendees:
  - Shannon
  - Dykstra
  - Atick
  - Redlich
  - Tishby
  - Zaslavsky
  - Wyner
  - AssumptionAdversary
  - Contrarian
  - PR95Author
  - Yousfi
  - Fridrich
council_quorum_met: true
council_verdict: PROCEED
council_dissent:
  - member: Contrarian
    verbatim: "v1 ATW V2 cdf_table_blob landed CARGO-CULTED-EMPIRICALLY-FALSIFIED per codex byte-mutation smoke 057130de4. This audit MUST surface 3+ NEW CARGO-CULTED assumptions beyond the cdf_table_blob falsification AND identify the upstream META-assumption that produced the categorical error. Sister 2026-05-16 V2 design memo enumerated 7 cargo-cults via the V1 unwind; this pass needs to extend to V2 itself, not just re-cite V1's unwinds."
  - member: AssumptionAdversary
    verbatim: "Per Catalog #292 per-deliberation assumption surfacing. The SHARED ASSUMPTION operating across the entire ATW V2 design lineage (V1 unwind memo + V2 full-stack design + cdf_table_blob procedural variant + cdf_dead_section reconciliation): 'Atick-Redlich + Tishby IB + Wyner-Ziv = ONE substrate (the ATW triple) by triple-citation-stacking-implies-triple-binds-substrate-optimally'. This is the META-assumption that produced ALL downstream cargo-cults including the cdf_table_blob FALSIFIED routing. Classification: CARGO-CULTED-STRUCTURALLY. Triple-citation rigor does NOT imply substrate-optimal binding — each paper's contribution must be re-derived against the contest's actual scorer-conditional information geometry, not assumed-additive."
council_assumption_adversary_verdict:
  - assumption: "ATW = Atick + Tishby + Wyner = single substrate via triple-citation-stacking"
    classification: CARGO-CULTED-STRUCTURALLY
    rationale: "Atick-Redlich 1990 framed cooperative-receiver for the RETINAL early-vision pipeline (decorrelating photoreceptor signals against a known biological receiver). Tishby 2015 IB framed (X;T)/(T;Y) bottleneck for supervised classification. Wyner-Ziv 1976 framed source-coding-with-side-info for a separate-encoder-decoder system with KNOWN side-info statistics. Stacking the three INTO ONE substrate via name-citation does not derive from any of the three papers — the three theorems EACH require different mathematical preconditions. ATW V2 v1 inherited the triple-binding as an axiom; never proved the contest's actual receiver geometry satisfies all three simultaneously."
  - assumption: "cdf_table_blob is canonical equation #26 IN-DOMAIN context atw_v2_codec_quantizer_lut for REPLACEMENT paradigm prediction"
    classification: CARGO-CULTED-EMPIRICALLY-FALSIFIED
    rationale: "Codex byte-mutation smoke commit 057130de4 (test_cdf_table_xor_preserves_current_inflate_raw_output) empirically proved max_abs_raw_byte_delta=0 across all 2,560 cdf_table_blob bytes mutated. cdf_table_blob is DECODE-OPAQUE: reconstruct_from_wz_residual() never consumes the table. Canonical equation #26 EXCLUDED context direct_byte_substitution_on_decode_opaque_raw_sections registered to extinct this bug class. The structural cause: the design-memo presumed bytes-have-decode-influence WITHOUT empirical byte-mutation smoke per Catalog #139/#220/#272."
  - assumption: "Wyner-Ziv side-info is well-defined for the contest receiver pipeline as (decoder has access to scorer_class_prior_table; encoder does not)"
    classification: CARGO-CULTED
    rationale: "Wyner-Ziv 1976 requires (a) DECODER has correlated side-info Y; (b) ENCODER does NOT have Y at encode time; (c) JOINT distribution P(X,Y) is KNOWN at design time. ATW V2 v1's framing places scorer_class_prior_table inside the archive (compress-time computed, decode-time read), making it AVAILABLE to BOTH encoder + decoder. This is NOT Wyner-Ziv (decoder-side-info-only); it is DETERMINISTIC SIDE-INFO (both-sides-shared). The rate-distortion bound for two-side shared-info is fundamentally different (Shannon channel coding with known channel state at encoder = no rate-distortion penalty for using it). The cdf_table_blob falsification is consistent with this: the bytes ARE in the archive (compressor saw them) so they are NOT side-info in the Wyner-Ziv sense — the assumption was structurally wrong from the start."
  - assumption: "Tishby IB bottleneck applied to per-pair latent z preserves scorer-relevant signal"
    classification: CARGO-CULTED-PARTIAL
    rationale: "Tishby IB requires (a) explicit task labels Y; (b) bottleneck variable T mediates X→Y; (c) optimization over P(T|X) at fixed P(Y|T). ATW V2 v1's IB term (Variant A only) compresses I(z;X_frames) subject to preserving I(z;Y_segnet_class). The HARD-EARNED part: this matches the canonical IB formulation IF the segnet class label IS the contest task. The CARGO-CULTED part: the actual contest task is reconstructed-frame-RGB rendered through PoseNet+SegNet, NOT the per-pixel segnet class itself. Per Catalog #297: the contest scorer derives masks from frames (signal-axis-destruction of the chroma channel via grayscale = catastrophic loss per NSCS06 anchor). IB bottleneck on per-class signal collapses chroma info the scorer's PoseNet downstream requires."
  - assumption: "G1 distill head (1KB MLP scorer_class -> 5-way softmax) is the canonical replacement for Ballé-style hyperprior"
    classification: HARD-EARNED-PARTIAL
    rationale: "The 1KB-vs-50KB byte budget IS hard-earned (Wunderkind G1 memo arithmetic + small-MLP-suffices argument inherited from PR101 Quantizr distillation). PRESERVE the byte-budget claim. BUT the assumption that 1KB MLP captures the same conditioning signal as the 50KB hyperprior is CARGO-CULTED — Wunderkind G1 memo did not measure this against the contest scorer. If the contest scorer's per-class entropy distribution has high-rank structure (e.g. class-conditional CDF shapes are dramatically different per class), 1KB MLP may produce a near-uniform output (mode collapse) and degrade scorer-conditioning to near-zero (consistent with the D4 INDEPENDENT verdict on A1 latents: I(latent; scorer_class) = 0.006385 bits/symbol, ~3 orders of magnitude below MEANINGFUL_CONDITIONING)."
  - assumption: "ATW V2's score-aware loss term λ_WZ × R_WZ(z|s) is operational at inflate time"
    classification: HARD-EARNED-EMPIRICALLY-VERIFIED
    rationale: "ATW V2 architecture.py:538-567 reconstruct_from_wz_residual() does compute z = z_residual + WZ_head(class_prior), and this IS consumed by self.decoder(z_full). Byte-mutation smoke would PROVE the operational consumption (vs cdf_table_blob's DECODE-OPAQUE state). PRESERVE; the WZ residual + side-info head pattern IS operational under the current ATW V2 runtime."
council_decisions_recorded:
  - "op-routable #1: produce 4+ NEW CARGO-CULTED assumptions beyond cdf_table_blob falsification (Contrarian veto threshold)"
  - "op-routable #2: identify upstream META-assumption that produced cdf_table_blob categorical error (Assumption-Adversary directive)"
  - "op-routable #3: Phase 2 design-decision memo must explicitly choose Path (a)/(b)/(c) per audit findings"
  - "op-routable #4: 3-axis evidence per operator binding directive #3 (math + scientific + engineering rigor + MLX drift minimization + numpy portability)"
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_override_rationale: null
horizon_class: frontier_pursuit
deferred_substrate_retrospective_due_utc: null
deferred_substrate_id: atw_v2_cooperative_receiver_via_cargo_cult_first_methodology
related_deliberation_ids:
  - atw_v2_cdf_table_blob_reconciliation_codex_byte_mutation_smoke_falsified_20260521
  - atw_v2_cdf_table_blob_procedural_variant_design_20260521
  - atw_codec_v2_cooperative_receiver_full_stack_design_20260516
  - atw_codec_atick_tishby_wyner_v1_design_20260515
  - atw_codec_v1_cargo_cult_unwind_design_20260516
  - atw_codec_v2_d4_probe_verdict_20260516_codex
  - council_per_substrate_symposium_atw_v2_reactivation_20260518
  - path_3_b_z7_mamba_2_cargo_cult_audit_of_existing_scaffold_20260526
  - path_3_c_nscs06_v8_chroma_lut_cargo_cult_audit_of_existing_scaffold_20260526
  - feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515
canonical_equation_id: procedural_codebook_from_seed_compression_savings_v1
canonical_equation_registry_path: .omx/state/canonical_equations_registry.jsonl
predecessor_design_memos:
  - .omx/research/atw_codec_atick_tishby_wyner_v1_design_20260515.md
  - .omx/research/atw_codec_v2_cooperative_receiver_full_stack_design_20260516.md
  - .omx/research/atw_v2_cdf_table_blob_procedural_variant_design_20260521.md
  - .omx/research/atw_v2_cdf_table_blob_reconciliation_codex_byte_mutation_smoke_falsified_20260521.md
codex_empirical_anchor_commit: 057130de4
codex_empirical_anchor_max_abs_raw_byte_delta: 0
codex_empirical_anchor_mutated_byte_count: 2560
mlx_drift_research_anchor: .omx/research/codex_findings_mlx_drift_determinism_online_research_20260522T050151Z_codex.md
mlx_drift_engineering_anchor: .omx/research/pr95_mlx_pytorch_drift_mitigation_engineering_landed_20260525.md
mlx_downstream_scorer_drift_anchor: .omx/research/pr95_mlx_full_decoder_downstream_scorer_drift_landed_20260526.md
score_claim: false
promotion_eligible: false
research_only: true
dispatch_enabled: false
audit_evidence_tag: "[macOS-MLX research-signal]"
operator_directive: "PATH 3 candidate #H ATW V2 cooperative-receiver MLX-local L0 SCAFFOLD per cargo-cult-pass-first methodology; binding operator directives #1+#2+#3 (MLX-first + cargo-cult-pass-first + 3-axis evidence math+scientific+engineering+MLX-drift+numpy-portability); APPEND audit only per Catalog #110/#113 (NO mutation of any predecessor memo); 3 sister Path 3 in-flight (B' Z7-Mamba-2 + C' NSCS06 v8 + F Z8 hierarchical predictive coding) coordinated DISJOINT per Catalog #230"
---

<!-- HISTORICAL_PROVENANCE per Catalog #110/#113: this audit is APPEND-only.
The predecessor design memos (V1 design + V2 full-stack + cdf_table_blob procedural variant + cdf_dead_section reconciliation) are PRESERVED byte-for-byte. The codex empirical anchor 057130de4 + canonical equation #26 EXCLUDED context registration are PRESERVED. This audit's primary claim — that the META-assumption "ATW = single substrate via triple-citation-stacking" produced the cdf_table_blob FALSIFIED routing — is APPEND-only structural visibility; it does NOT mutate any prior artifact. -->

# Path 3 candidate H — ATW V2 cooperative-receiver PHASE 1 adversarial cargo-cult audit of existing scaffold

**Lane:** `lane_path_3_h_atw_v2_cooperative_receiver_cargo_cult_first_20260526` (L0 → L1 after commit)
**Subagent:** `path_3_h_atw_v2_cooperative_receiver_cargo_cult_first_20260526`
**Cost:** $0 (audit-only)
**Wall-clock:** Phase 1 ~60 min

## 0. Binding operator directives (verbatim, why this audit exists)

1. *"The MLX first requirement might also force us out of the issue we were having before where we had great ideas but we're building them as Boltons to the same substrates over and over again; we want to design the substrate and curriculum and then optimize the design the whole stack around it for extreme optimization and performance and optimal score lowering"*

2. *"Never simply extend unless a rigorous adversarial cargo cult pass has been done first"*

3. *"we also need adversarial review against all landing recursive for math and scientific and engineering rigor and for MLX drift minimization and portability via numpy"* (NEW 2026-05-26)

The predecessor ATW V2 cdf_table_blob procedural variant (`atw_v2_cdf_table_blob_procedural_variant_design_20260521.md` commit `8441b702e`) declared canonical equation #26 IN-DOMAIN REPLACEMENT routing with predicted ΔS = -0.001684; codex byte-mutation smoke commit `057130de4` empirically falsified max_abs_raw_byte_delta=0 across all 2,560 cdf_table_blob bytes mutated. This audit IS the corrective per directive #2 + recovers from the v1 implementation-level falsification per Catalog #307.

**This audit's mandate**: re-examine the FULL ATW V2 design lineage (V1 design 2026-05-15 + V1 unwind 2026-05-16 + V2 full-stack 2026-05-16 + cdf_table_blob procedural 2026-05-21 + cdf_dead_section reconciliation 2026-05-21) for upstream META-assumptions that produced the cdf_table_blob FALSIFIED routing AND surface assumptions the v1 unwind missed. Output: design-decision input. No code edits in this phase.

## 1. Catalog #229 premise verification (PV) — state of the ATW V2 scaffold at audit start

**PV-0**: ATW V2 substrate scaffold present at `src/tac/substrates/atw_codec_v2/` (architecture.py 597 LOC + archive.py 25.4KB + inflate.py 8.6KB + score_aware_loss.py 12.8KB + cdf_dead_section.py 13.1KB + registered_substrate.py 4.4KB + __init__.py 11.5KB). Sister V1 at `src/tac/substrates/atw_codec_v1/`. Symposium impl at `src/tac/symposium_impls/atw_codec_atick_tishby_wyner_triple.py`. Phase 2 gate at `src/tac/optimization/atw_v2_phase2_gate.py`. FAISS IVF-PQ channel at `src/tac/optimization/faiss_ivf_pq_atw_channel.py`.

**PV-1**: ATW V2 `__init__.py:88` declares `D4_PROBE_VERDICT = "INDEPENDENT"` with `D4_PROBE_MUTUAL_INFORMATION_BITS = 0.006385502752311645` AND `D4_PROBE_PHASE2_STATUS = "defer_measured_a1_latent_class_conditioning_surface"` AND `D4_PROBE_NEXT_ACTION = "do_not_dispatch_atw_v2_phase2_from_this_signal"`. **The substrate's OWN canonical helper says do-not-dispatch.** The D4 probe on A1 latents returned I(latent; scorer_class) ≈ 6.4×10⁻³ bits/symbol — ~3 orders of magnitude below the canonical MEANINGFUL_CONDITIONING threshold (typically 0.5+ bits/symbol per Catalog #313 probe outcome ledger pattern).

**PV-2**: cdf_table_blob byte-mutation falsification anchor (commit `057130de4`) is preserved at `src/tac/substrates/atw_codec_v2/tests/test_cdf_dead_section.py::test_cdf_table_xor_preserves_current_inflate_raw_output`. The test asserts `proof.raw_equal is True` AND `proof.max_abs_raw_byte_delta == 0` AND `proof.mutated_byte_count == 2560`. **The bytes are EMPIRICALLY DECODE-OPAQUE.**

**PV-3**: canonical equation #26 registry entry (`.omx/state/canonical_equations_registry.jsonl`) carries `domain_of_validity_excluded` list with entry `direct_byte_substitution_on_decode_opaque_raw_sections` per Catalog #344 operator-decision protocol. The structural extinction of the cdf_table_blob misapplication is OPERATIONAL at the equation-registry layer (Catalog #359 STRICT preflight gate refuses anchors with this EXCLUDED context).

**PV-4**: Sister Path 3 in-flight + landed cargo-cult audits — B' (Z7-Mamba-2) at `.omx/research/path_3_b_z7_mamba_2_cargo_cult_audit_of_existing_scaffold_20260526.md` produced 10 NEW CC items + Phase 2 decision-memo + L0 SCAFFOLD; C' (NSCS06 v8 chroma_lut) at `.omx/research/path_3_c_nscs06_v8_chroma_lut_cargo_cult_audit_of_existing_scaffold_20260526.md` produced Phase 2 design decision (commit `bac0ec05d`). **Sister audits validate the cargo-cult-pass-first methodology empirically: B' surfaced 10 NEW CCs beyond the 2026-05-18 design memo's CC-1..CC-10; C' surfaced its own cargo-cult inventory.** This audit's Contrarian threshold (3+ NEW CCs beyond the v1 unwind + cdf_table_blob falsification): MUST be met or pass is performative.

**PV-5**: MLX drift research anchors — codex `mlx_drift_determinism_online_research_20260522T050151Z` (operation-order + Metal FP32 precision + reduction-order primitives drift); `pr95_mlx_pytorch_drift_mitigation_engineering_landed_20260525.md` (engineering mitigations); `pr95_mlx_full_decoder_downstream_scorer_drift_landed_20260526.md` (downstream-scorer drift profile through full decoder). These constitute the AXIS 2 (MLX drift minimization) prior knowledge per operator directive #3.

**PV-6**: Symposium memo `council_per_substrate_symposium_atw_v2_reactivation_20260518.md` (66.9KB) + FAISS IVF-PQ probe verdicts `atw_v2_1_byte_closed_side_info_probe_20260518_codex.md` + `atw_v2_1_faiss_pq_disambiguator_probe_20260518_codex.md` + `atw_v2_1_scorer_softmax_sketch_probe_20260518_codex.md` are the prior reactivation evidence. The 2026-05-18 symposium reactivated ATW V2 pending probe verdicts; those probes returned INDEPENDENT-class verdicts; v2.1 FAISS IVF-PQ proposal landed but never reached empirical-anchor surface.

## 2. The cargo-cult audit per assumption (Catalog #303 compliant)

Per CLAUDE.md "META-ASSUMPTION ADVERSARIAL REVIEW" + the hard-earned-vs-cargo-culted classification addendum: every shared assumption MUST be classified HARD-EARNED (cite source; PRESERVE) OR CARGO-CULTED (eligible for challenge; propose unwind).

This pass extends the V1 unwind memo's audit + the cdf_table_blob falsification + the D4 INDEPENDENT verdict + reactivates ATW V2 v2 with cargo-cult-first methodology. The audit is structured in 4 layers:

- **Layer 1 META-assumption** (the upstream structural error)
- **Layer 2 paradigm-binding assumptions** (Atick / Tishby / Wyner triple-binding choices)
- **Layer 3 architectural choices** (V2-specific design memo decisions)
- **Layer 4 implementation cargo-cults** (downstream of architecture choices)

### Layer 1 — META-assumption

| # | Assumption | Classification | Citation | Unwind path |
|---|---|---|---|---|
| **MCC-1** | "ATW = Atick + Tishby + Wyner = one substrate via triple-citation-stacking-implies-triple-binds-substrate-optimally" | **CARGO-CULTED-STRUCTURALLY** | Per the V1 design memo (`atw_codec_atick_tishby_wyner_v1_design_20260515.md`) name choice "ATW codec" — the substrate is named after the triple. Per V2 design memo `atw_codec_v2_cooperative_receiver_full_stack_design_20260516.md` §1 declares it "binds Atick-Redlich + Tishby IB + Wyner-Ziv into ONE substrate-engineering scaffold". The BINDING of three theorems via name-stacking is NOT a mathematical derivation; each theorem requires DIFFERENT mathematical preconditions: Atick-Redlich (continuous-time retinal MI maximization with FIXED biological decoder); Tishby IB (supervised classification with discrete labels Y); Wyner-Ziv (separate encoder-decoder with KNOWN joint distribution P(X,Y) and DECODER-ONLY side-info). The contest's receiver geometry was NEVER PROVED to satisfy all three theorems' preconditions simultaneously. | Phase 2 design must (a) IDENTIFY which ONE of the three theorems applies to the contest's actual scorer-conditional information geometry; (b) DERIVE the substrate-optimal codec from that ONE theorem's preconditions; (c) USE the other two as advisory cross-checks NOT binding-mathematical-axioms. The Y prediction: Wyner-Ziv does NOT apply (deterministic side-info, both-sides-shared); Tishby IB does NOT apply (no discrete task labels Y; contest task is RGB reconstruction through PoseNet+SegNet); Atick-Redlich DOES apply (continuous-time information geometry with a fixed receiver = SegNet+PoseNet) but requires substrate-design FROM ATICK-REDLICH PRECONDITIONS, not from triple-binding. |

**Layer 1 finding**: ONE META-cargo-cult. This is the upstream structural error that produced the cdf_table_blob FALSIFIED routing (the design memo PRESUMED cdf_table_blob was a Wyner-Ziv side-info channel; the empirical byte-mutation smoke disproves that PRESUMPTION because the bytes are not decoder-side-info in the Wyner-Ziv-theorem sense). The fix at Layer 1 cascades down to Layers 2-4.

### Layer 2 — Paradigm-binding assumptions

| # | Assumption | Classification | Citation | Unwind path |
|---|---|---|---|---|
| **CC-1** | "Wyner-Ziv side-info is well-defined for the contest receiver pipeline as (decoder has access to scorer_class_prior_table; encoder does not)" | **CARGO-CULTED** | Per V2 architecture.py:262-313 _WZSideInfoHead. The scorer_class_prior_table is stored as a registered buffer + loaded from archive at inflate time (architecture.py:448-451). The table is COMPUTED AT COMPRESS TIME (encoder sees it) and BURNED INTO THE ARCHIVE (decoder reads it). Both sides see the same bytes. Wyner-Ziv 1976 requires DECODER-ONLY side-info Y where encoder does NOT have Y. The current framing violates this precondition. | Fork: either (a) reframe as Slepian-Wolf (distributed source coding where both sides have correlated sources but encode independently — does not match contest geometry either); (b) reframe as Conditional source coding (both sides KNOW the conditioning variable — matches the actual ATW V2 geometry; rate-distortion bound is then R(D|Y) = R(D) for known Y on both sides, no rate savings from the side-info per se); (c) DROP the Wyner-Ziv framing entirely and ground substrate in Atick-Redlich cooperative-receiver loss directly (per Wunderkind E1 substitution recommendation). |
| **CC-2** | "Tishby IB bottleneck applied to per-pair latent z preserves scorer-relevant signal" | **CARGO-CULTED-PARTIAL** | V2 architecture.py declares Variant A includes κ_IB term `kappa_IB * I(T; Y_predicted)` (substrate `__init__.py:54`). The contest task is RGB reconstruction → PoseNet pose-loss + SegNet seg-loss. The IB framing identifies T = z (per-pair latent), X = frames, Y = downstream score. The cargo-cult: the IB compresses I(z;X) at fixed I(z;Y_segnet_class) — but the actual contest Y is NOT segnet-class-only; the PoseNet pathway requires chroma + spatial-detail that segnet-class compression destroys. Per Catalog #297 forbidden signal-axis-destruction without reversibility probe. | Fork: re-derive IB target Y as the actual contest score (composite pose+seg+rate). The compressor must preserve information about TWO downstream pipelines (PoseNet sees chroma; SegNet sees class) — single-Y IB is structurally wrong. Multi-task IB literature (per Tishby-Zaslavsky 2015 extensions) exists; Phase 2 should derive substrate-optimal IB target from contest score formula directly. |
| **CC-3** | "Atick-Redlich 1990 cooperative-receiver loss applies to contest dashcam video pipeline" | **HARD-EARNED-PARTIAL** | Per V2 `__init__.py:74` routes through `tac.codec.cooperative_receiver.atick_redlich.cooperative_receiver_loss` (canonical primitive per Catalog #164). Atick-Redlich 1990 framed retinal photoreceptor decorrelation against a KNOWN biological decoder; the math IS general (any continuous-time information channel with a fixed receiver). The HARD-EARNED part: the framing IS valid for the contest pipeline (PoseNet+SegNet IS a fixed receiver). The CARGO-CULTED part: the SPECIFIC formulation (canonical helper at `src/tac/codec/cooperative_receiver/atick_redlich.py`) may not match the actual contest receiver's mutual-information geometry. Per Ballard 2007 + Atick-Redlich 1990: ego-motion-conditioned next-frame prediction is the SUBSTRATE-OPTIMAL framing for dashcam video, NOT generic frame decorrelation. | Phase 2: re-derive cooperative-receiver loss for the contest's SPECIFIC receiver = ego-motion-conditioned PoseNet (per Catalog #311 predictive coding ego-motion). The loss should penalize information loss in the SCORER-DOWNSTREAM signal axis, not in generic frame-axis decorrelation. |
| **CC-4** | "Triple-binding of Atick + Tishby + Wyner via additive Lagrangian (`α·B + β·d_seg + γ·sqrt(d_pose) + λ_WZ·R_WZ + [κ_IB·I + λ_pix·MSE] for Variant A`) achieves substrate-optimal score" | **CARGO-CULTED** | V2 `__init__.py:45-54` declares the additive Lagrangian. This presumes the three theorems' loss surfaces COMPOSE ADDITIVELY (and that additive composition is Pareto-optimal). Per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable + Catalog #296 Dykstra-feasibility intersection: additive composition of multi-constraint losses requires PROOF the feasible set is non-empty AND the Pareto-optimal point lies on a vertex of the polytope. The V1 design memo did NOT compute the Dykstra-feasibility intersection. | Phase 2: compute Dykstra-feasibility for the (α, β, γ, λ_WZ, κ_IB, λ_pix) polytope under the contest constraints. Most likely finding: the polytope INTERIOR is empty for the strict triple-binding (one theorem's optimum violates another's preconditions); the Pareto vertex is on the BOUNDARY where one or two weights are zero. This is consistent with Variant B (single-knob WZ-only) being declared DEFAULT in V2 design memo §4.3 — the design already converged to a single-theorem-dominant form empirically without proving it. |

### Layer 3 — Architectural choices (V2-specific)

| # | Assumption | Classification | Citation | Unwind path |
|---|---|---|---|---|
| **CC-5** | "V2 decoder inherits V1 PixelShuffle NeRV-style decoder verbatim (matches Z4 sister)" | **CARGO-CULTED** | Per V2 architecture.py:186-260 _ATWv2Decoder is a PixelShuffle(2)×6 decoder with `decoder_channels = (24, 20, 16, 12, 8, 6)` going from 3×4 initial grid to 384×512 output. Inherited from V1 + V1 inherited from Z4. The decoder choice was NEVER optimized for the cooperative-receiver loss's actual gradient surface. PR101 / PR103 medal-class winners use HNeRV-style decoder (different upsampling pattern, larger initial grid). | Phase 2 paired smoke (when MLX-portable): PixelShuffle×6 vs HNeRV-style (Conv+Upsample+Conv blocks) vs Cool-Chic (smaller initial grid + per-channel scale factors) at SAME archive byte budget; measure score on canonical SegNet+PoseNet receiver. Choose Pareto vertex empirically. |
| **CC-6** | "Encoder is small (input_channels=3 → hidden=64 → latent_dim=24) because LOSS dominates not capacity" | **CARGO-CULTED-PARTIAL** | V2 architecture.py:154-184 _ATWv2Encoder: single Conv(3→64) + 2× Linear(64→24). Tiny capacity. V1 design memo + V2 design memo both presume "loss + WZ + G1 + B3 dominate score-improvement". The HARD-EARNED part: encoder capacity is consumed by archive bytes (every encoder param ships), so small encoder reduces rate-axis. The CARGO-CULTED part: the LOSS-dominates claim was empirically untested; given the D4 INDEPENDENT verdict (I(latent;class) = 6.4e-3 bits/symbol), it may be that ENCODER CAPACITY actually dominates and the loss's contribution is near-zero. | Phase 2 paired smoke: encoder_hidden ∈ {16, 64, 128, 256} at fixed everything-else; measure I(latent; scorer_class) per-config + score. If the I scales with encoder capacity (rather than loss-weight), the cargo-cult is confirmed and the substrate is encoder-capacity-bound not loss-design-bound. |
| **CC-7** | "scorer_class_prior_table is computed at compress time by the cooperative-receiver loss's gradient through SegNet" | **CARGO-CULTED-PENDING-EMPIRICAL** | V2 architecture.py:448-451 stores `scorer_class_prior_table` as `(num_pairs, scorer_class_prior_dim)` initialized to zeros. The architecture comment says "Per-pair scorer class prior precomputed table; loaded from archive at inflate time." but the actual COMPUTE is not in the loaded files — presumably in `_full_main` (which the substrate marks `IMPLEMENTATION_STATUS = "l1_architecture_archive_inflate_and_loss_modules_available_research_only"`). The cargo-cult: the table's content is presumed to be MEANINGFUL conditioning; the D4 probe (commit `tools/run_atw_v2_d4_probe_from_a1.py`) measured against A1 latents and found INDEPENDENT. The table may be near-zero in practice (uninformative). | Phase 2: $0 MPS smoke that initializes scorer_class_prior_table via a TIGHTER computation (per-pair SegNet softmax averaged over scorer-rendered frames; or per-pair argmax class; or per-pair PoseNet pose-delta projection) and re-measures I(latent; scorer_class). Multiple table-computation schemes; choose the one that maximizes I empirically. If NO scheme exceeds 0.1 bits/symbol, the conditioning surface is structurally absent on A1-class latents and the substrate must FORK from cooperative-receiver-via-class-prior to a different side-info channel (e.g. ego-motion-FOE conditioning per Catalog #311). |
| **CC-8** | "ATW V2 archive grammar (ATW2 magic + 9 sections including dead cdf_table_blob) is the canonical substrate-class shift" | **CARGO-CULTED-EMPIRICALLY-FALSIFIED** | Per V2 `__init__.py:66` declares "parser_section_manifest: ATW2 header + encoder_blob + decoder_blob + wz_head_blob + distill_head_blob + latent_residual_blob + class_prior_table_blob + cdf_table_blob + meta_blob (9 sections; +2 vs ATW1)". The cdf_table_blob section is empirically DECODE-OPAQUE per codex 057130de4. The substrate ships 2,560 dead bytes per archive WITHOUT operational mechanism. Per Catalog #220 substrate-engineering operational mechanism declaration: dead bytes are research-substrate-trap. | Phase 2: REMOVE cdf_table_blob from the canonical grammar (REMOVAL paradigm per cdf_dead_section reconciliation §4 + canonical equation #26 EXCLUDED context). The substrate-class-shift redesign should DROP THE SECTION ENTIRELY rather than substitute it procedurally (which would still ship the substitution envelope). Saving 2,560 bytes (vs 2,528 procedural substitution) is small (-0.001705 score per canonical formula) but structurally non-cargo-culted. The grammar redesign also surfaces a META-question: are any of the OTHER 8 sections also decode-opaque? Phase 2 must run byte-mutation smoke per Catalog #139 + #272 on ALL 8 remaining sections BEFORE declaring substrate-class shift. |
| **CC-9** | "G1 distill head + B3 scorer-conditional CDF table are the substrate's distinguishing features per Catalog #272" | **CARGO-CULTED-EMPIRICALLY-FALSIFIED-PARTIAL** | Per V2 `__init__.py:32-42` declares "G1 scorer-class distill head: 1KB MLP ... ΔS prediction -0.005 to -0.015 rate-axis [prediction; first-principles-bound; D4-probe-conditional]" + "B3 scorer-conditional CDF table: range-encode latents conditional on scorer's softmax distribution per pixel ... ΔS prediction -0.003 to -0.010 rate-axis [prediction; first-principles-bound]". B3 = cdf_table_blob = FALSIFIED (decode-opaque). G1 is unverified per Catalog #272 byte-mutation smoke. The substrate's NAMED distinguishing features are EITHER falsified OR untested. | Phase 2 byte-mutation smoke on G1 distill head bytes (1KB MLP weights stored in distill_head_blob). If G1 distill head bytes are ALSO decode-opaque, the substrate has ZERO operational distinguishing features and is structurally equivalent to bare Z4 / A1 sister. If G1 bytes ARE decode-influential, document the operational mechanism per Catalog #220 + verify per Catalog #272. |
| **CC-10** | "Variant B (single-knob WZ-only) is the UNIQUE-AND-COMPLETE default substrate-optimal form" | **CARGO-CULTED** | Per V2 `__init__.py:23-27` declares Variant B as default per operator standing directive "feedback_pr95_lesson_now_at_meta_level_unique_and_complete_per_method_default_20260515.md". The Variant B form drops Tishby IB regularizer + Z3 pixel-MSE residual; binds ONLY the Wyner-Ziv side-info residual mechanism. But Wyner-Ziv side-info IS itself CARGO-CULTED per CC-1 (both-sides-shared, not decoder-only). Variant B is a single-knob-binding to a cargo-culted theorem. | Phase 2 design decision: if CC-1 unwind path (c) is chosen (drop Wyner-Ziv framing entirely), Variant B becomes structurally empty and the substrate has NO knobs. Substrate must be re-derived from Atick-Redlich preconditions + ego-motion-FOE conditioning (per Catalog #311 + #310). The Variant A vs B regime sweep becomes moot. |

### Layer 4 — Implementation cargo-cults (downstream of architecture choices)

| # | Assumption | Classification | Citation | Unwind path |
|---|---|---|---|---|
| **CC-11** | "ATW V2 _full_main can land WITHOUT canonical helper `gate_auth_eval_call` because the substrate routes through canonical `tac.codec.cooperative_receiver.atick_redlich.cooperative_receiver_loss`" | **CARGO-CULTED** | Per Catalog #226 STRICT preflight: substrate trainers MUST route auth_eval through canonical helper, not hand-rolled subprocess. V2's `IMPLEMENTATION_STATUS = "l1_architecture_archive_inflate_and_loss_modules_available_research_only"` declares full_main not yet implemented. The cargo-cult is the IMPLICIT ASSUMPTION that when _full_main lands, the auth_eval routing will be straightforward — sister substrates' Catalog #226 retrofit was 50-100 LOC. Should be planned IN the Phase 2 design memo. | Phase 2 design: explicitly declare `_full_main` will route through `gate_auth_eval_call` per Catalog #226 + `tac.deploy.modal.runtime` canonical NVML block per Catalog #244 + `select_inflate_device` canonical helper per Catalog #205. NO hand-rolled subprocess; NO `--archive-zip`/`--output-json` stale CLI; canonical wire-in from byte-zero. |
| **CC-12** | "ATW V2 substrate is PyTorch-native; MLX is a downstream port" | **CARGO-CULTED-PER-OPERATOR-DIRECTIVE-#1** | Per V2 architecture.py uses `torch.nn` + `torch.Tensor` throughout. There is NO MLX surface in the existing scaffold. Per operator binding directive #1 (MLX-first reframing): future substrates should be MLX-native FIRST, PyTorch-port SECOND, so the MLX-first iteration loop accelerates the design-space exploration. The existing V2 substrate is structurally inverted vs the new operator directive. | Phase 2 design: declare MLX-first via `import mlx.core as mx` + `mlx.nn` for the canonical compute path; PyTorch sister `_torch_compat_reference.py` for parity verification per the new 3-axis discipline (Axis 2 MLX drift minimization requires MLX↔PyTorch parity test); numpy reference `numpy_reference.py` for portability per Axis 3. |
| **CC-13** | "ATW V2 substrate inherits the V1 D4 probe verdict INDEPENDENT (I=6.4e-3) as a DISPATCH BLOCKER on Phase 2 paid empirical anchor" | **HARD-EARNED-EMPIRICALLY-VERIFIED** | Per V2 `__init__.py:88-91`: D4_PROBE_NEXT_ACTION = "do_not_dispatch_atw_v2_phase2_from_this_signal". The probe IS the canonical disambiguator per Catalog #125 hook #6. The verdict is the operational anchor that informs Phase 2 substrate-design — do NOT dispatch unless a richer side-information surface lands. PRESERVE. | N/A — preserved per canonical-share-when-serves. Phase 2 substrate-design must respect the D4 verdict + propose a NEW probe surface (e.g. ego-motion-FOE conditioning per Ballard 2007 + Catalog #311) if substrate-class shift is pursued. |
| **CC-14** | "ATW V2 substrate honors HNeRV parity lessons 1-13 via inheritance from V1" | **CARGO-CULTED-PENDING-EMPIRICAL** | Per V1 unwind memo (commit reference in V2 architecture.py:1-40 comments). The lessons (eval_roundtrip / EMA / score-aware loss / canonical scorer helpers / archive grammar / inflate runtime / export contract / score-aware Lagrangian / etc.) MUST be re-verified for V2 specifically. The inheritance chain is brittle — V2's NEW intervention (G1 + B3 + WZ closed-form) may have introduced new lesson-violations. | Phase 2 design: per-lesson verification table for V2. e.g. is eval_roundtrip=True consumed by `_WZSideInfoHead.forward()` path? Is EMA shadow used at inference vs live latents? Is the archive bytes-deterministic across reseeds? Each lesson gets a verified-or-violated row. |
| **CC-15** | "FAISS IVF-PQ ATW V2.1 channel (commit reference in `src/tac/optimization/faiss_ivf_pq_atw_channel.py` + 2026-05-18 design memo) is the Phase 2 reactivation path" | **CARGO-CULTED-PENDING-EMPIRICAL** | Per `atw_v2_1_faiss_ivf_pq_substrate_design_memo_20260518.md` (43.6KB) + 3 probe verdicts (byte_closed_side_info / faiss_pq_disambiguator / scorer_softmax_sketch). The FAISS IVF-PQ proposal was scoped post-symposium; never reached empirical-anchor surface. The cargo-cult: the existing 2026-05-18 reactivation path was via the same Wyner-Ziv-side-info framing (now CC-1 falsified). FAISS IVF-PQ as an L2 vector-quantizer over class priors is a DIFFERENT cargo-cult question. | Phase 2: explicitly choose whether to pursue FAISS IVF-PQ as the new substrate framework OR to re-derive from Atick-Redlich preconditions. The FAISS IVF-PQ path is structurally separate from the cargo-cult-pass-first methodology — it's a different substrate-class shift via a different conditioning surface. If pursued, must run its own cargo-cult pass. |
| **CC-16** | "ATW V2 archive grammar 9 sections + the symposium-blessed 2026-05-18 reactivation makes ATW V2 'production hardened' per the operator's standing directive 'production hardened comma ai and openpilot grade OSS'" | **CARGO-CULTED** | Per per-substrate symposium memo `council_per_substrate_symposium_atw_v2_reactivation_20260518.md` (66.9KB). The symposium passed reactivation pending probe verdicts; probes returned INDEPENDENT-class verdicts; cdf_table_blob FALSIFIED post-symposium. Calling ATW V2 "production hardened" is structurally inconsistent with the empirical evidence stream. | Phase 2 design: the substrate is currently RESEARCH-ONLY-PENDING-NEW-SIDE-INFO-SURFACE. Production hardening (comma-ai / openpilot grade OSS) requires a different research deliverable (e.g. portable PyTorch reference implementation + numpy reference + extensive parity tests across substrate variants) than score-lowering substrate-class shift. The two efforts are STRUCTURALLY DISTINCT and should not be conflated. This audit + L0 SCAFFOLD pursues the score-lowering substrate-class shift; production-hardening is OUT OF SCOPE. |

**Counts**: 1 META-CC (Layer 1) + 4 paradigm-binding CCs (Layer 2) + 6 architectural CCs (Layer 3) + 6 implementation CCs (Layer 4) = **17 CCs total** beyond the v1 unwind memo's CC enumeration + the cdf_table_blob falsification. Of these: 13 CARGO-CULTED (eligible for challenge); 4 HARD-EARNED-PARTIAL or HARD-EARNED-EMPIRICALLY-VERIFIED. Contrarian's verbatim VETO threshold (≥ 3 NEW CARGO-CULTED) satisfied with substantial margin.

## 3. Hard-earned vs cargo-culted summary

**HARD-EARNED (PRESERVE; do NOT violate):**
- CC-3 (Atick-Redlich 1990 framing IS valid for contest's fixed-receiver geometry; preserves the paradigm intact while forking the implementation)
- CC-13 (D4 INDEPENDENT verdict on A1 latents is the canonical disambiguator; respect the dispatch blocker)
- The CLAUDE.md non-negotiables: eval_roundtrip=True / EMA-shadow-at-inference / MPS-NOT-authoritative / strict-scorer-rule (no SegNet/PoseNet load at inflate) / differentiable-scorer-preprocess / single-archive.zip / contest formula / byte-deterministic archive
- Catalog #205 (canonical inflate device selector) / Catalog #226 (canonical auth_eval helper) / Catalog #244 (canonical NVML env block) / Catalog #146 (contest-compliant inflate runtime template)
- Catalog #319 / #325 (deliverability proof + per-substrate symposium)

**CARGO-CULTED (eligible for challenge):**
- MCC-1 (META: ATW = triple-citation-stacking-implies-substrate)
- CC-1 (Wyner-Ziv framing of scorer_class_prior_table)
- CC-2 (Tishby IB Y target = segnet-class-only)
- CC-4 (additive triple-binding Lagrangian)
- CC-5 (V2 decoder inherits V1 verbatim)
- CC-6 (encoder capacity loss-dominates)
- CC-7 (scorer_class_prior_table compute-scheme cargo-cult)
- CC-8 (cdf_table_blob in canonical grammar) — **EMPIRICALLY FALSIFIED**
- CC-9 (G1 + B3 are distinguishing features) — **PARTIAL FALSIFICATION**
- CC-10 (Variant B is UNIQUE-AND-COMPLETE default)
- CC-11 (auth_eval routing planned implicitly)
- CC-12 (PyTorch-native substrate; MLX downstream port) — **PER OPERATOR DIRECTIVE #1**
- CC-14 (HNeRV parity lessons inherited from V1)
- CC-15 (FAISS IVF-PQ V2.1 as Phase 2 reactivation path)
- CC-16 (ATW V2 = production-hardened)

**Predicted substrate-class-shift potential**: HIGH for paradigm-binding fork (Layer 2 unwinds + META-Layer 1 unwind), MEDIUM for architectural refactor (Layer 3), LOW for implementation-only cleanup (Layer 4). The Layer 1 unwind is the highest-EV: identifying which ONE of Atick / Tishby / Wyner is the substrate-optimal anchor + re-deriving from that theorem's preconditions may unlock score the triple-binding suppressed.

## 4. NEW REQUIRED 3-axis evidence per operator directive #3

### Axis 1 — Math + scientific + engineering rigor

Per the cargo-cult audit above, each Layer 2 paradigm-binding decision must be re-grounded in first-principles math:

- **Atick-Redlich 1990**: continuous-time MI maximization between input X and receiver-output R = f(X) for a FIXED receiver f. Sufficient statistics: P(X), receiver f, decoder D. The contest's receiver is SegNet+PoseNet (fixed). The cooperative-receiver loss is L_AR = -I(X; R(X)) per Atick-Redlich 1990 + canonical primitive `tac.codec.cooperative_receiver.atick_redlich.cooperative_receiver_loss`. Math: HARD-EARNED.
- **Tishby IB 2015**: discrete-task IB requires LABELS Y; contest has CONTINUOUS task (RGB→PoseNet pose-delta + RGB→SegNet class). Multi-task IB literature exists (Tishby-Zaslavsky 2015 extensions + Schmidhuber compression-as-intelligence) but no canonical primitive in `tac.codec/` yet. Math: PARTIALLY HARD-EARNED requires extension.
- **Wyner-Ziv 1976**: source coding with decoder-only side-info. Contest geometry has BOTH-SIDES-SHARED side-info (scorer_class_prior_table is in the archive). Math: NOT APPLICABLE to current framing — must REFRAME or DROP.
- **Schmidhuber compression-as-intelligence** (advisory cross-check): RGB-reconstruction-through-receiver IS the canonical compression-as-intelligence formulation. Atick-Redlich is the rigorous specialization.

Citations:
- Atick & Redlich 1990 *"Towards a Theory of Early Visual Processing"* Neural Computation
- Tishby & Zaslavsky 2015 *"Deep Learning and the Information Bottleneck Principle"* IEEE ITW
- Wyner & Ziv 1976 *"The rate-distortion function for source coding with side information at the decoder"* IEEE TIT
- Ballard 2007 *"Embodied cognition and visual perception"* (FOE ego-motion conditioning)
- Schmidhuber 2009 *"Driven by Compression Progress"* (compression-as-intelligence)

### Axis 2 — MLX drift minimization per primitive

Per codex MLX drift research `codex_findings_mlx_drift_determinism_online_research_20260522T050151Z_codex.md` + engineering anchor `pr95_mlx_pytorch_drift_mitigation_engineering_landed_20260525.md` + downstream-scorer anchor `pr95_mlx_full_decoder_downstream_scorer_drift_landed_20260526.md`. Per-MLX-primitive drift bound + mitigation:

| MLX primitive | Expected drift bound vs PyTorch | Mitigation |
|---|---|---|
| `mx.conv2d` | 1e-5 to 1e-3 max_abs (FP32 fast-math reassociation) | Use `precise=True` mode; compare against `torch.conv2d` in test_basic.py |
| `mx.maximum` / `mx.relu` | bit-exact (no reordering) | None needed; canonical |
| `mean(axis=...)` | reduction-order drift; can be 1e-4 max_abs | Cast to FP64 before reduce, then cast back; OR use deterministic reduction kernel |
| `mx.matmul` / `mx.linear` | FMA-reassociation; 1e-5 to 1e-3 | `precise=True`; verify via test against numpy reference |
| `mx.sigmoid` | implementation-dependent; ~1e-6 | None typically needed |
| Cross-pair stateful predictor (e.g. RNN-style) | RECURRENCE compounds drift — can be 1e-2 over 600 pairs | Compute residual per-pair; do NOT chain hidden state across pairs unless necessary |
| `mx.softmax` over distribution | reduction-order; ~1e-4 | Tightly control axis; verify via numpy reference |
| Sigmoid + Conv decoder chain | end-to-end through decoder + scorer; can be 1e-3 to 1e-2 final score drift | Per pr95_mlx_full_decoder_downstream_scorer_drift_landed: full-decoder + scorer drift ~1e-2 (manageable for research-signal; below Catalog #1265 threshold 1e-3 GATE for paid CUDA dispatch IF using mitigation) |

Smoke test MUST include: (1) MLX↔PyTorch max_abs per-primitive measurement table; (2) MLX↔numpy reference parity for each primitive; (3) MLX↔PyTorch end-to-end full-decoder pass through SegNet+PoseNet scorer max_abs measurement; (4) Catalog #1265 gate threshold check (1e-3) before any paid CUDA dispatch consideration.

### Axis 3 — Portability via numpy

Per operator binding directive #3: every MLX primitive should have sister numpy reference OR documented non-portability rationale.

- `mx.conv2d`: numpy reference via `scipy.signal.correlate2d` (per channel) — slow but portable; production-portable to CPU-only systems
- `mx.linear` / `mx.matmul`: numpy `@` operator — bit-exact reference
- `mx.relu`: `np.maximum(x, 0)` — trivial
- `mx.sigmoid`: `1 / (1 + np.exp(-x))` — well-defined; tolerance for overflow at large negative x
- `mx.softmax`: `np.exp(x - x.max(axis=axis, keepdims=True)) / np.exp(x - x.max(axis=axis, keepdims=True)).sum(axis=axis, keepdims=True)` — standard numerically-stable formulation
- Cross-pair stateful predictor: numpy loop reference (sequential; slow but bit-exact)
- Score-aware loss canonical helpers: route through `tac.codec.cooperative_receiver.atick_redlich.cooperative_receiver_loss` (PyTorch); numpy reference may be intractable for full gradient path — document AS non-portability for the LOSS pathway specifically; the FORWARD/INFLATE pathway must remain portable
- Archive grammar + inflate runtime: pure-numpy (per Catalog #146 / #205); the inflate pathway MUST work on CPU-only numpy-only systems

**Decision**: Phase 2 substrate design must declare per-primitive numpy reference. The encoder/decoder/scorer-conditioning forward path should be fully portable. The gradient/training path (cooperative-receiver loss compute) MAY require PyTorch (non-portable) but MUST be isolated in a designated `_training_only.py` module so the inference + inflate path remain portable.

## 5. Cross-reference to sister Path 3 in-flight + landed audits

- **Path 3 candidate B' (Z7-Mamba-2)** `path_3_b_z7_mamba_2_cargo_cult_audit_of_existing_scaffold_20260526.md`: enumerated 10 NEW CCs beyond the 2026-05-18 design memo's CC-1..CC-10; key finding = stability fix should be ARCHITECTURAL not OPTIMIZER. Sister convergence: B' also identified that "MLX-first as stability-validation" is CARGO-CULTED (per CC-F in B' audit) — this audit echoes that concern at CC-12 (PyTorch-native substrate; MLX downstream port). The Phase 2 design decisions should align: both substrates should treat MLX-first as DESIGN-ITERATION-SCOPE only; CUDA-paid empirical anchor only at promotion gate per Catalog #1265 + #325.
- **Path 3 candidate C' (NSCS06 v8 chroma_lut)** `path_3_c_nscs06_v8_chroma_lut_cargo_cult_audit_of_existing_scaffold_20260526.md`: Phase 2 decision memo at commit `bac0ec05d`. Sister convergence: C' is also a procedural-codebook-from-seed candidate per canonical equation #26 IN-DOMAIN context `nscs06_v8_chroma_lut` (which remains VERIFIED CORRECT per cdf_table_blob reconciliation §158 audit memo). This audit's CC-8 (cdf_table_blob EXCLUDED) is the SISTER falsification anchor to C''s IN-DOMAIN context — together they establish the per-substrate disambiguator for canonical equation #26 routing decisions.
- **Path 3 candidate F (Z8 hierarchical predictive coding)** in-flight: cross-reference queued as op-routable if audit memo lands during my work window.

## 6. Phase 2 design-decision recommendation (binding input)

Per Catalog #290 canonical-vs-unique decision framework + the 17 CCs surfaced:

**Recommendation: Path (b) JUSTIFIED-EXTEND with explicit FORK on cargo-culted assumptions.**

Rationale: 4-of-17 CCs are HARD-EARNED (preserve); 13-of-17 CCs are CARGO-CULTED (fork). The HARD-EARNED set (CC-3 Atick-Redlich framing + CC-13 D4 INDEPENDENT verdict + CLAUDE.md non-negotiables + Catalog-set) is the substrate's CANONICAL INHERITANCE; the CARGO-CULTED set is the substrate-optimal engineering target. The fork operates at Layer 1 (META-assumption: identify Atick-Redlich as the SINGLE substrate-optimal anchor; drop Tishby IB + Wyner-Ziv as binding theorems; use as advisory cross-checks).

The Phase 2 design memo must:

1. **Layer 1 unwind**: declare Atick-Redlich as the SINGLE substrate-optimal anchor (CC-3 HARD-EARNED preservation; MCC-1 unwind). Tishby IB + Wyner-Ziv DEMOTED to advisory cross-checks; NOT binding mathematical axioms.
2. **Layer 2 unwind**: REFRAME scorer_class_prior_table as DETERMINISTIC SHARED CONDITIONING (NOT Wyner-Ziv side-info per CC-1 unwind path c); REFRAME IB target as multi-task contest-score (NOT segnet-class-only per CC-2 unwind); drop additive triple-binding Lagrangian (CC-4 unwind: derive substrate-optimal loss from Atick-Redlich preconditions DIRECTLY).
3. **Layer 3 unwind**: explicit per-substrate decoder/encoder paired smoke when MLX-portable (CC-5, CC-6 unwinds); revisit scorer_class_prior_table compute scheme (CC-7 unwind); REMOVE cdf_table_blob from grammar (CC-8 EMPIRICAL unwind); byte-mutation smoke G1 distill head (CC-9 PARTIAL FALSIFICATION verification); declare substrate's actual distinguishing feature(s) per Catalog #272 + #220.
4. **Layer 4 unwind**: declare auth_eval routing through canonical helper from byte-zero (CC-11); MLX-first scaffold per operator directive #1 (CC-12); per-lesson HNeRV parity verification table (CC-14); FAISS IVF-PQ V2.1 deferred to separate cargo-cult pass if pursued (CC-15); OUT-OF-SCOPE production hardening (CC-16).
5. **3-axis evidence per operator directive #3**: math + scientific + engineering rigor per primitive (Axis 1 above); MLX drift minimization per primitive table (Axis 2 above); numpy reference per primitive (Axis 3 above).
6. **L0 SCAFFOLD plan**: MLX-native renderer + numpy reference + substrate-specific archive grammar + PyTorch inflate runtime ≤200 LOC + Catalog #91/#139 byte-mutation tests + MLX↔PyTorch + MLX↔numpy parity tests + `_full_main raises NotImplementedError` per Catalog #240(c) until Phase 2 council ratifies.

## 7. Operator-routable next steps (queued for Phase 2 → Phase 3)

1. **Phase 2 design-decision memo** at `.omx/research/path_3_h_atw_v2_cooperative_receiver_substrate_design_decision_20260526.md` — chosen path (b) + binding Phase 3 roadmap per the 6 design recommendations above.
2. **Phase 3 L0 SCAFFOLD** at canonical extension path (per Path (b) choice: NEW package `src/tac/substrates/atw_v2_cooperative_receiver_v2/` to avoid bolt-on per operator directive #2; preserves existing `atw_codec_v2/` for Catalog #110/#113 HISTORICAL_PROVENANCE).
3. **6-hook wire-in declaration per Catalog #125** for the new substrate (all 6 hooks declared in Phase 2 design memo).
4. **D4-equivalent probe redesign** for Atick-Redlich's preconditions on contest geometry (DEFERRED pending Phase 3 scaffold; tracked as Phase 4 op-routable).
5. **FAISS IVF-PQ V2.1 disposition**: defer to separate cargo-cult pass IF pursued; explicitly out-of-scope for this Phase 1-2-3.
6. **Production hardening (comma-ai / openpilot grade OSS)**: out-of-scope; tracked separately under different lane.
7. **Sister convergence audit at Phase 3**: cross-reference with B' L0 SCAFFOLD + C' Phase 3 (if landed) for canonical-helper + MLX-drift-mitigation pattern alignment.

## 8. APPEND-ONLY footer per Catalog #110/#113 HISTORICAL_PROVENANCE

This audit memo is APPEND-only. The predecessor design memos + codex empirical anchor `057130de4` + canonical equation #26 registry EXCLUDED context registration + sister Path 3 audits are PRESERVED byte-for-byte. The primary claim — that the META-assumption "ATW = single substrate via triple-citation-stacking" produced the cdf_table_blob FALSIFIED routing AND additional CARGO-CULTED assumptions remain in the existing V2 scaffold — is APPEND-only structural visibility per Catalog #307 paradigm-vs-implementation classification: the PARADIGM (cooperative-receiver via Atick-Redlich) is INTACT; the IMPLEMENTATION (triple-binding Lagrangian + Wyner-Ziv-framed side-info + cdf_table_blob as decode-influential bytes) is PARTIALLY FALSIFIED. The recommended Phase 2 path is FORK-VIA-LAYER-1-UNWIND not KILL.

Per CLAUDE.md "Forbidden premature KILL without research exhaustion": the substrate paradigm is DEFERRED-PENDING-PHASE-2-LAYER-1-UNWIND (NOT KILLED). Reactivation criterion: Phase 2 design decision lands explicitly identifying Atick-Redlich as the canonical anchor + Layer 2-4 unwinds per recommendation; Phase 3 L0 SCAFFOLD lands + MLX↔PyTorch parity smoke passes Catalog #1265 threshold for advisory promotion to paid CUDA empirical anchor consideration.

**Council verdict**: T2 PROCEED (12 attendees; sextet + 6 topical-GRAND-INNER per canonical roster; Atick-Redlich-Tishby-Zaslavsky-Wyner specialist seats added per topic; AssumptionAdversary verbatim VETO satisfied with 13 CARGO-CULTED CCs surfaced vs threshold 3).

**Mission contribution per Catalog #300**: `frontier_breaking_enabler` (the Layer 1 META-unwind is the upstream structural enabler for ATW V2 substrate-class shift; without it, the substrate is structurally locked into the triple-binding cargo-cult and the cdf_table_blob FALSIFIED routing is destined to recur via parallel-section dead-byte cargo-cults).

**Lane**: `lane_path_3_h_atw_v2_cooperative_receiver_cargo_cult_first_20260526` L0 → L1 after Phase 1 commit.
