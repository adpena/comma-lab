---
council_tier: T1
council_attendees: [Claude]
council_quorum_met: false
council_verdict: PROCEED_WITH_REVISIONS
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "PR106 #05 + #06 'FALSIFICATION' was substrate-class kill"
    classification: CARGO-CULTED
    rationale: "Per Cable C6 synthesis META-bug attribution: substrate-mismatch-as-class-kill (Catalog #185 META-class). The original UNIWARD-delta on PR106 substrate killed the IMPLEMENTATION at the wrong substrate, NOT the technique. PR101 gold (0.193) uses grayscale-LUT analog mask paradigm + UNIWARD-style adaptive embedding distortion. PR103 silver (0.195) uses sister UNIWARD-derived steganalytic distortion measure. Per CLAUDE.md FORBIDDEN PATTERNS: 'Forbidden premature KILL': KILL is LAST RESORT after exhausting all research; original verdict was IMPLEMENTATION-LEVEL not PARADIGM-LEVEL falsification per Catalog #307."
  - assumption: "Lane 05 (UNIWARD-derived embedding distortion) requires direct mask channel in substrate"
    classification: CARGO-CULTED
    rationale: "PR106 has no mask channel. UNIWARD-derived embedding distortion was applied to LATENT channels via direct port, NOT reformulated for PR106's latent-bias correction surface. The reformulation: UNIWARD computes Daubechies wavelet-domain 'undetectable' distortion at PIXEL level; the PR106 reformulation translates this to LATENT-BIAS correction at the bit-allocation layer (where PR106's latent-bias is the canonical perturbation site)."
  - assumption: "Lane 06 (grayscale-LUT analog mask) requires substrate-native mask emission"
    classification: CARGO-CULTED
    rationale: "PR101 gold demonstrates grayscale-LUT on REGISTERED mask channel; PR106 lacks this. The reformulation: PR106 has 23 bytes per pair of latent-bias side-information; this IS a discrete-class-derived sidecar in the Quantizr paradigm sense. Grayscale-LUT analog mask paradigm applied to LATENT-BIAS class assignments (the actual sidecar bytes PR106 charges)."
council_decisions_recorded:
  - "Wave 1: $0 design memo (THIS) — reformulate UNIWARD-delta + grayscale-LUT for PR106's actual surface"
  - "Wave 2 (CONTINGENT on operator approval): $10 paired CPU+CUDA smoke testing reformulated UNIWARD-bit-allocation + grayscale-LUT-latent-bias variants"
  - "Wave 3 (CONTINGENT on Wave 2): $5-10 composition with Z6/Z7 sister substrates"
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: true
council_override_rationale: |
  Operator quote 2026-05-19 verbatim: "All operator fates and decisions approved" + Cable C6 synthesis cheap-signal-first sequencing
  (commit `4c056724c`) designates this $0 design memo as Tier 1 (no spend; PROCEED IMMEDIATELY).
horizon_class: frontier_pursuit
council_assumption_classification_addendum: |
  Per Catalog #307 paradigm-vs-implementation-level falsification discipline + CLAUDE.md "Forbidden
  premature KILL": the 4 alternative probe methodologies enumerated below (per Catalog #308) are the
  canonical research path forward; standalone PR106 #05/#06 FALSIFICATION was IMPLEMENTATION-LEVEL +
  SUBSTRATE-MISMATCH-LEVEL, NOT PARADIGM-LEVEL.
related_deliberation_ids:
  - cable_c6_re_eval_high_symposium_drafts_synthesis_20260519T060557Z
  - tier_1_resurrection_5_pr106_lanes_05_06_reformulated_uniward_grayscale_lut_full_stack_design_20260516
---

# PR106 #05 + #06 REFORMULATED — Design memo

## Authority

Per Cable C6 synthesis 2026-05-19 (commit `4c056724c`) "cheap-signal-first sequencing" + operator-frontier-override 2026-05-19 "All operator fates and decisions approved" + Cable C6.3 DRAFT verdict (`DRAFT_PENDING_CONVOCATION` + `frontier-EV` priority).

## META-bug attribution

Per Cable C6 synthesis: Cable C6.3 META-bug = **substrate-mismatch-as-class-kill** (Catalog #185 META-class) + Catalog #324 predicted-band cargo-cult.

The 2026-04-29 PR106 #05 + #06 FALSIFICATION at scores 1.78 + 1.84 [contest-CUDA] was attributed to "TECHNIQUE FALSIFIED" but the actual root cause was **SUBSTRATE MISMATCH**:

| Lane | Original | Falsification cause | Reformulated for PR106 |
|---|---|---|---|
| #05 | UNIWARD-derived embedding distortion port to PR106 LATENT channels | UNIWARD assumes pixel-level wavelet distortion (Daubechies basis); direct port to latents disregards substrate's latent-bias dynamics | Translate UNIWARD wavelet-domain cost function to LATENT BIT-ALLOCATION cost; cost = wavelet-coefficient detectability on the substrate's latent grid |
| #06 | Grayscale-LUT analog mask port to PR106 | PR106 has no mask channel; the port silently became no-op | Translate grayscale-LUT to LATENT-BIAS class assignments (PR106's 23-byte sidecar IS a discrete-class-derived sidecar in the Quantizr paradigm sense) |

## Canonical-vs-unique decision per layer

Per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD standing directive 2026-05-15:

| Layer | Decision | Rationale |
|---|---|---|
| Substrate base (PR106 r2) | ADOPT canonical (lane_pr106_latent_sidecar_r2_pr101_grammar_contest_cpu) | Verified anchor 0.20533 [contest-CUDA] / archive `9cb989cef519...` |
| UNIWARD distortion math | FORK (substrate-mismatch fix) | Reformulate from pixel-wavelet to latent-bit-allocation cost; canonical UNIWARD source-fidelity preserved at MATH level (Daubechies wavelet) |
| Grayscale-LUT mask paradigm | FORK (substrate-mismatch fix) | Apply to LATENT-BIAS class assignments (PR106's actual discrete-class sidecar surface); canonical Quantizr paradigm preserved at PARADIGM level |
| Archive grammar | ADOPT PR106 r2 grammar | Single-file `0.bin`; latent bytes = 23 per pair; sidecar preserved |
| Inflate runtime | ADOPT canonical (`submissions/pr106_latent_sidecar_r2_pr101_grammar`) | Already <=100 LOC per HNeRV L4; CUDA-or-CPU agnostic |
| Score-aware training | ADOPT canonical pyav+SegNet+PoseNet+EMA+eval_roundtrip per CLAUDE.md non-negotiables | PR101 paradigm; differentiable scorer routing per Catalog #164 |
| Export contract | ADOPT canonical | Existing PR106 export already verified |
| Tier-1 engineering | ADOPT canonical (autocast / TF32 / torch.compile / no_grad / canonical scorer-loss helper) | Catalogs #172/#178/#179/#180/#164 |

**Bolt-on size**: ≤350 LOC per HNeRV parity L7 (UNIWARD reformulation + grayscale-LUT-latent-bias mapper).

## 9-dimension success checklist evidence

1. **UNIQUENESS** — reformulated UNIWARD-bit-allocation + grayscale-LUT-latent-bias are DISTINCT from standalone PR106 (no embedding distortion / no class-aware bit allocation in r2 baseline).
2. **BEAUTY+ELEGANCE** — ≤350 LOC bolt-on per L7; reviewable in 30 seconds per PR101 paradigm.
3. **DISTINCTNESS** — explicitly different from PR106 r2 baseline (`9cb989cef519`); from PR101 gold (different substrate); from PR103 silver (different distortion measure target).
4. **RIGOR** — premise verification (Catalog #229) + adversarial review (Catalog #292) + assumption classification (HARD-EARNED/CARGO-CULTED per addendum) + empirical anchor (PR106 r2 0.20533).
5. **OPTIMIZATION PER TECHNIQUE** — Dimension 5 covered by Catalog #290 canonical-vs-unique section above.
6. **STACK-OF-STACKS-COMPOSABILITY** — orthogonal to Z6 cooperative-receiver (Wave 2 4c if revives) + DP1 pretrained driving prior + sister Cable C substrates; additive ΔS per Catalog #322 composition_alpha v2 cascade.
7. **DETERMINISTIC REPRODUCIBILITY** — byte-stable archive (PR106 r2 grammar deterministic) + seed-pinned training.
8. **EXTREME OPTIMIZATION + PERFORMANCE** — Tier-1 engineering primitives ADOPT canonical per layer-decision above.
9. **OPTIMAL MINIMAL CONTEST SCORE** — predicted band [0.180, 0.200] per Cable C6.3 DRAFT; specific target = beat current PR106 r2 0.20533 by ≥0.005 (-0.025 to -0.005 ΔS).

## Observability surface

1. **Inspectable per layer** — UNIWARD wavelet decomposition coefficients capturable per training step; grayscale-LUT class-assignment histogram capturable per archive build.
2. **Decomposable per signal** — `final_score` decomposable into PR106 r2 baseline contribution + UNIWARD-reformulation delta + grayscale-LUT-latent-bias delta (separable A/B testing).
3. **Diff-able across runs** — byte-stable archive enables exact byte-level diff vs PR106 r2 baseline.
4. **Queryable post-hoc** — archive JSON manifest + per-pair distortion log + class-assignment histogram all queryable.
5. **Cite-able** — every byte tagged to (substrate / commit / call_id / config / random_seed / upstream_snapshot_sha256) per Catalog #245.
6. **Counterfactual-able** — Catalog #139 packet compiler no-op detector + Catalog #272 byte-mutation smoke verify reformulated bytes affect score.

## Cargo-cult audit per assumption

Per Catalog #303:

| Assumption | HARD-EARNED / CARGO-CULTED | Unwind path |
|---|---|---|
| UNIWARD wavelet distortion is universal across substrates | CARGO-CULTED | Reformulate per substrate; PR106's latent-bias surface is NOT pixel-wavelet |
| Grayscale-LUT requires native mask channel | CARGO-CULTED | Apply to PR106's discrete-class-derived sidecar (the 23-byte latent-bias) |
| Standalone PR106 #05/#06 falsification = paradigm-level kill | CARGO-CULTED | Reformulate per substrate per Catalog #307; original verdict was IMPLEMENTATION-LEVEL substrate-mismatch |
| Composition with sister substrates is additive | HARD-EARNED | Per Catalog #322 composition_alpha v2 cascade + per Z1 ablation |
| Predicted band [0.180, 0.200] is calibrated | CARGO-CULTED-PENDING-EMPIRICAL | Per Catalog #324: predicted_band_validation_status = pending_post_training; reactivation criterion = Wave 2 paired CPU+CUDA smoke |

## Predicted ΔS band (per Catalog #296 + #324 discipline)

**Predicted band**: [0.180, 0.200] contest-CUDA (per Cable C6.3 DRAFT REFORMULATED estimate).
**Predicted_band_validation_status**: pending_post_training.
**Dykstra-feasibility check**: REFORMULATED variants are orthogonal axes (UNIWARD-bit-allocation operates on latent distortion; grayscale-LUT-latent-bias operates on class assignment); convex feasibility = intersection of (rate-constraint preserved at 23 bytes/pair) ∩ (distortion-constraint relaxed to allow UNIWARD-cost embedding) ∩ (class-shift-constraint preserved at PR106's discrete-class sidecar surface).
<!-- PREDICTED_BAND_VIBES_OK:Cable C6.3 DRAFT REFORMULATED predicted band [0.180, 0.200] is HIGH-VARIANCE planning prior pending Wave 2 paired CPU+CUDA empirical anchor. Per Catalog #296 + #324 + #325: REACTIVATION via Wave 2 $10 paired smoke ($5 CPU + $5 CUDA) decides Wave 3 composition path. -->

## 4 alternative probe methodologies (Catalog #308)

1. **UNIWARD-bit-allocation reformulation alone** ($5 CUDA smoke): test UNIWARD-cost-based latent-bit-allocation in isolation on PR106 r2 substrate.
2. **Grayscale-LUT-latent-bias reformulation alone** ($5 CUDA smoke): test grayscale-LUT class-assignment on PR106's 23-byte sidecar in isolation.
3. **Composition #1 + #2 over PR106** ($5 CUDA smoke): test stacked variants.
4. **Composition over alternate substrate (PR101 gold)** ($5 CUDA smoke): test reformulated variants on PR101 gold to compare substrate-matched performance.

## Reactivation criteria per CLAUDE.md "Forbidden premature KILL"

- (a) Wave 2 paired CPU+CUDA smoke ($10) → predicted ΔS within band → Wave 3 composition lands
- (b) Wave 2 within-band confirms reformulation paradigm → Cable B substrate cascade routing receives composition matrix update
- (c) Wave 2 OUTSIDE band → 30-day deferred-substrate retrospective (Catalog #300 Consequence 3) decides re-iteration vs DEFER

## 6-hook wire-in declaration (Catalog #125)

1. **Sensitivity-map** = ACTIVE (UNIWARD-cost-derived per-latent-bit sensitivity contributes to `tac.sensitivity_map.*`)
2. **Pareto constraint** = ACTIVE (reformulated variants are Pareto-rank candidates)
3. **Bit-allocator hook** = ACTIVE (reformulated UNIWARD operates AT the bit-allocator layer)
4. **Cathedral autopilot dispatch hook** = ACTIVE (Wave 2 smoke verdict feeds substrate composition matrix per Catalog #322 v2 cascade)
5. **Continual-learning posterior update** = ACTIVE (Wave 2 smoke verdict registered via `tac.council_continual_learning.append_council_anchor`)
6. **Probe-disambiguator** = N/A (no 2+ defensible interpretations; reformulation paradigm is canonical)

## Cross-references

- Cable C6 synthesis 2026-05-19: `.omx/research/cable_c6_re_eval_high_symposium_drafts_synthesis_20260519T060557Z.md`
- Tier 1 resurrection design memo 2026-05-16: `.omx/research/tier_1_resurrection_5_pr106_lanes_05_06_reformulated_uniward_grayscale_lut_full_stack_design_20260516.md`
- CLAUDE.md "Forbidden premature KILL" + "Forbidden artifact-lifecycle violations" non-negotiables
- Catalog #185 (META-meta-meta drift detection); Catalog #307 (paradigm-vs-implementation classification); Catalog #308 (alternative probe methodologies); Catalog #324 (predicted-band post-training validation)


<!-- # FORMALIZATION_PENDING:pre_framework_memo_dated_2026-05-19_predates_canonical_equations_birthday_registry_population_in_progress_appended_by_strict_flip_enablers_per_operator_blanket_approval_per_claude_md_forbidden_premature_kill_without_research_exhaustion_this_is_DEFER_pending_canonical_equation_backfill_NOT_kill -->
