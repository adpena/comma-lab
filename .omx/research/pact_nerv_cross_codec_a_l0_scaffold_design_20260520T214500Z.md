<!-- # FORMALIZATION_PENDING:cross_codec_super_additive_alpha_anchor_pending_stage_1_dispatch_per_catalog_344 -->
<!-- # COUNCIL_ASSUMPTION_STATEMENT_WAIVED:l0_scaffold_design_memo_predates_council_per_catalog_292 -->

# Pact-NeRV-CROSS-CODEC-A L0 SCAFFOLD design memo

**Date:** 2026-05-20
**Lane:** `lane_pact_nerv_cross_codec_a_l0_scaffold_20260520`
**Variant:** PACT-NERV-ULTIMATE Variant #16 — fec6 base codec + Pact-NeRV-A1 side-info bolt-on
**Literature anchor:** Atick-Redlich 1990 cooperative-receiver + CROSS-CANDIDATE finding #3 (PR101/A1/fec6 ↔ PR106 per-axis Pearson [-0.094, -0.078] SUPER_ADDITIVE signature per Catalog #322)
**horizon-class:** `plateau_adjacent` per Catalog #309
**research_only:** true per HNeRV parity L2 + Catalog #220

---

## Executive summary

Pact-NeRV-CROSS-CODEC-A composites the fec6 frontier base codec (Huffman k=16
selector + frame-exploit selector per PR101 GOLD lineage) with a Pact-NeRV-A1
class HNeRV decoder as a side-info **residual** bolt-on. The hypothesis per
CROSS-CANDIDATE finding #3: SUPER_ADDITIVE composition because the two codecs
operate on DIFFERENT receptive fields of the contest scorer (top-K Jaccard < 0.05).

L0 SCAFFOLD lands substrate package + trainer + recipe + driver + 15 tests but
does NOT fire paid dispatch. Stage 1 dispatch is operator-gated per Catalog
#325/#240/#315.

---

## Canonical-vs-unique decision per layer

| Layer | Decision | Rationale |
|---|---|---|
| Trainer skeleton (device_or_die / pin_seeds / git_head_sha) | ADOPT_CANONICAL | Cross-substrate hygiene per Catalog #190/#178 |
| SubstrateContract (Catalog #241/#242) | ADOPT_CANONICAL | META-layer registration enforces all hooks |
| Score-aware loss helper (`score_pair_components_dispatch`) | ADOPT_CANONICAL | Catalog #164/#222 canonical scorer-preprocess routing |
| Differentiable eval_roundtrip (`patch_upstream_yuv6_globally`) | ADOPT_CANONICAL | Catalog #6 MANDATORY DEFAULT non-negotiable |
| Archive grammar (`CCAA` magic + 30-byte header + fec6_base + decoder + latents + selectors + meta) | FORK_BECAUSE_PRINCIPLED_MISMATCH | Cross-codec composition requires CC_A-specific magic + per-pair fec6 selector bytes section |
| Inflate runtime (per-substrate ≤200 LOC) | FORK_BECAUSE_PRINCIPLED_MISMATCH | HNeRV parity L4: cross-codec composition has its own inflate path |
| Fec6BaseCodecPlaceholder (L0) | FORK_BECAUSE_PRINCIPLED_MISMATCH | At L0 the placeholder is a deterministic palette-color render; L1 will swap for actual fec6 runtime from `submissions/pr101_*/inflate.py` |
| HNeRV-class side-info decoder | ADOPT_CANONICAL | Sister of pact_nerv_selector_v3 architecture; depthwise-sep + SIREN + PixelShuffle |
| Composition (static alpha residual additive) | FORK_PROVISIONAL_AT_L0 | CARGO-CULTED at L0; L1 needs learned composition gate per Atick-Redlich 1990 cooperative-receiver |

---

## 9-dimension success checklist evidence

1. **UNIQUENESS** — first variant in the canonical 3-variant G4 CROSS-CODEC series; the fec6 base + Pact-NeRV residual pattern is structurally distinct from sister variants B (PR106 base + IA3) and C (E2E neural).
2. **BEAUTY + ELEGANCE** — substrate ~230 LOC (architecture only) reviewable in 30 sec; clean canonical-pattern reuse for trainer + score-aware loss.
3. **DISTINCTNESS** — vs CC_B: uses fec6 base + Pact-NeRV-A1 (NOT IA3); vs CC_C: uses fixed base + Pact-NeRV residual (NOT end-to-end neural composition).
4. **RIGOR** — 15 tests covering pack/parse roundtrip + gradient flow + canonical scorer routing + recipe research_only + driver NVML block; premise-verifier-style architecture-import-instantiate smoke (`num_parameters=69014`).
5. **OPTIMIZATION PER TECHNIQUE** — canonical scorer-loss helper routing per #164; canonical eval_roundtrip per #6; canonical hardware substrate detection per #190.
6. **STACK-OF-STACKS-COMPOSABILITY** — designed as a cross-codec composition primitive; predicted SUPER_ADDITIVE per Catalog #322 cross_codec_orthogonality_predictor_consumer (commit 80484241f).
7. **DETERMINISTIC REPRODUCIBILITY** — seed pinned (default 20260520); pack-then-parse roundtrip invariant tested; SIREN init bounds explicit per architecture.
8. **EXTREME OPTIMIZATION + PERFORMANCE** — autocast_fp16 + torch_compile flags declared with `# AUTOCAST_FP16_WAIVED:l0-scaffold` waivers; full training path operator-gated.
9. **OPTIMAL MINIMAL CONTEST SCORE** — predicted ΔS pending Stage 1 dispatch + Catalog #322 SUPER_ADDITIVE proof; absolute magnitude undisclosed per Catalog #321 non-promotable + Catalog #324 post-training Tier-C validation requirement.

---

## Cargo-cult audit per assumption

| Assumption | Classification | Unwind path |
|---|---|---|
| Cross-codec orthogonal composition yields SUPER_ADDITIVE | HARD-EARNED-EMPIRICAL (CROSS-CANDIDATE finding #3 anchor) | N/A — empirically supported |
| Static residual additive composition (alpha=0.1) | CARGO-CULTED at L0 | L1: learned composition gate per Atick-Redlich 1990 cooperative-receiver |
| fec6 base codec inherited as-is at L0 | HARD-EARNED-EMPIRICAL (PR101 GOLD frontier anchor) | N/A |
| HNeRV-class Pact-NeRV-A1 backbone | HARD-EARNED-LITERATURE (sister pact_nerv_ia3 architecture) | N/A |
| Per-pair selector u8 (16 values) | CARGO-CULTED at L0 (Fec6BaseCodecPlaceholder) | L1: actual fec6 Huffman k=16 selector + frame-exploit selector from `submissions/pr101_*` |
| SIREN init for HNeRV decoder | HARD-EARNED-LITERATURE (Sitzmann 2020) | N/A |

---

## Observability surface

1. **Inspectable per layer** — per-pair selector + per-pair latent inspectable via `model.selectors` + `model.latents` (Parameter access).
2. **Decomposable per signal** — score-aware loss returns `(loss, parts)` dict with `rate_term` / `seg_term` / `pose_term` / `loss_total` per Catalog #305.
3. **Diff-able across runs** — pack-then-parse roundtrip + state_dict comparison; deterministic seeded init.
4. **Queryable post-hoc** — provenance JSON written per dispatch with `n_params` / `composition_alpha` / `fec6_palette_size` / `hardware_substrate_detected` / `evidence_grade=scaffold-smoke-no-score-axis`.
5. **Cite-able** — provenance carries `git_head` + `trainer` + `lane_id` + `substrate_tag` + `started_at` UTC ISO.
6. **Counterfactual-able** — Catalog #139 byte-mutation smoke planned for archive blobs (fec6_base + decoder + latents); pack-then-parse roundtrip is the L0 surface.

---

## Predicted ΔS band

**Status:** UNKNOWN at L0 SCAFFOLD per Catalog #324 (no empirical anchor; predicted_band_validation_status=pending_post_training).

**Dykstra feasibility check:** the cross-codec composition is a sum-of-two-rate-and-distortion problem; predicted SUPER_ADDITIVE per cross_codec_orthogonality_predictor_consumer (commit 80484241f) on the basis of CROSS-CANDIDATE finding #3 per-axis Pearson [-0.094, -0.078] (top-K Jaccard <0.05) — the codecs operate on DIFFERENT receptive fields of the contest scorer so their composition lies on the interior of the rate-distortion polytope intersection per Boyd convex feasibility.

**First-principles citation:** Shannon 1948 R(D) bound applied separately to each codec class + Atick-Redlich 1990 cooperative-receiver bound on the sum.

**Probe disambiguator:** `tools/probe_pact_nerv_cross_codec_a_super_additive_alpha_sweep.py` (planned at Stage 1 dispatch) sweeps composition_alpha ∈ {0.05, 0.1, 0.2, 0.5, 1.0} and measures contest-CUDA score delta.

---

## Reactivation criteria for paid dispatch

1. STAIRCASE Step 16 dispatch operator-gated per Catalog #325.
2. Per-substrate symposium proceed-unconditional per Catalog #315.
3. L1 learned composition gate landed per Atick-Redlich 1990.
4. Real fec6 base codec runtime swapped in (placeholder retired).
5. Score-aware Lagrangian + canonical auth-eval helper wired per #226.
6. `research_only` flips to false; `predicted_band` declared per #324.
