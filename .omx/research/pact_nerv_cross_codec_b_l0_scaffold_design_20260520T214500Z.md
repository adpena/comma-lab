<!-- # FORMALIZATION_PENDING:cross_codec_super_additive_alpha_anchor_pending_stage_1_dispatch_per_catalog_344 -->
<!-- # COUNCIL_ASSUMPTION_STATEMENT_WAIVED:l0_scaffold_design_memo_predates_council_per_catalog_292 -->

# Pact-NeRV-CROSS-CODEC-B L0 SCAFFOLD design memo

**Date:** 2026-05-20
**Lane:** `lane_pact_nerv_cross_codec_b_l0_scaffold_20260520`
**Variant:** PACT-NERV-ULTIMATE Variant #17 — PR106 latent-score-table base + Pact-NeRV-IA3 side-info
**Literature anchor:** Atick-Redlich 1990 cooperative-receiver + CROSS-CANDIDATE finding #3 + Liu et al. 2022 *"IA3"* arXiv:2205.05638
**horizon-class:** `plateau_adjacent` per Catalog #309
**research_only:** true per HNeRV parity L2 + Catalog #220

---

## Executive summary

Pact-NeRV-CROSS-CODEC-B composites the PR106 latent-score-table format0d base
codec with a Pact-NeRV-IA3 side-info **residual** bolt-on (HNeRV decoder +
ego-pose-conditioned γ-only per-block modulation per Liu 2022). The hypothesis
per CROSS-CANDIDATE finding #3: SUPER_ADDITIVE composition because PR106 and
HNeRV-IA3 operate on DIFFERENT receptive fields. Sister of CC_A but with PR106
base (NOT fec6) and IA3 γ-only modulation (NOT plain HNeRV).

L0 SCAFFOLD lands substrate package + trainer + recipe + driver + 16 tests but
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
| Archive grammar (`CCBA` magic + 32-byte header + pr106_base + ia3_decoder + latents + score_indices + ego_pose + meta) | FORK_BECAUSE_PRINCIPLED_MISMATCH | CC_B-specific magic + per-pair PR106 score-table indices + per-pair ego-pose bytes (IA3 requires) |
| Inflate runtime (per-substrate ≤200 LOC) | FORK_BECAUSE_PRINCIPLED_MISMATCH | HNeRV parity L4 |
| Pr106BaseCodecPlaceholder (L0) | FORK_BECAUSE_PRINCIPLED_MISMATCH | At L0 placeholder is a 64-value position-encoded color table; L1 will swap for actual PR106 runtime from `submissions/pr106_*/inflate.py` |
| IA3 γ-only modulation (`_Ia3GammaBlock`) | ADOPT_CANONICAL | Sister of pact_nerv_ia3 architecture; Liu 2022 §3.2 reference impl |
| Composition (static alpha residual additive) | FORK_PROVISIONAL_AT_L0 | CARGO-CULTED at L0; L1 needs learned composition gate per Atick-Redlich 1990 |

---

## 9-dimension success checklist evidence

1. **UNIQUENESS** — second variant in the canonical 3-variant G4 CROSS-CODEC series; the PR106 base + Pact-NeRV-IA3 pattern is structurally distinct from sister variants A (fec6 + A1) and C (E2E neural composition).
2. **BEAUTY + ELEGANCE** — substrate ~260 LOC (architecture only; +30 vs CC_A for IA3 γ blocks) reviewable in 30 sec.
3. **DISTINCTNESS** — vs CC_A: PR106 base codec (NOT fec6) + IA3 γ-only ego-pose modulation (NOT plain HNeRV); vs CC_C: fixed base + Pact-NeRV residual (NOT end-to-end neural).
4. **RIGOR** — 16 tests covering pack/parse roundtrip + IA3 γ-zero-init invariant + gradient flow + canonical scorer routing + recipe research_only + driver NVML block (`num_parameters=70358`).
5. **OPTIMIZATION PER TECHNIQUE** — canonical scorer-loss helper routing per #164; canonical eval_roundtrip per #6; IA3 γ-zero init preserved through SIREN re-init logic.
6. **STACK-OF-STACKS-COMPOSABILITY** — designed as cross-codec composition primitive; predicted SUPER_ADDITIVE per Catalog #322 cross_codec_orthogonality_predictor_consumer.
7. **DETERMINISTIC REPRODUCIBILITY** — seed pinned; pack-then-parse roundtrip invariant; ego_pose buffer registered + loaded explicitly from archive.
8. **EXTREME OPTIMIZATION + PERFORMANCE** — autocast_fp16 + torch_compile flags declared with waivers; full training path operator-gated.
9. **OPTIMAL MINIMAL CONTEST SCORE** — predicted ΔS pending Stage 1 dispatch + Catalog #322 SUPER_ADDITIVE proof; absolute magnitude undisclosed per Catalog #321 non-promotable + Catalog #324.

---

## Cargo-cult audit per assumption

| Assumption | Classification | Unwind path |
|---|---|---|
| PR106 ↔ HNeRV-IA3 cross-codec composition yields SUPER_ADDITIVE | HARD-EARNED-EMPIRICAL (CROSS-CANDIDATE finding #3) | N/A |
| IA3 γ-only ego-pose modulation outperforms FiLM γ+β on driving video | HARD-EARNED-LITERATURE (Liu 2022 §3.2 + 4) BUT contest-domain-pending | Stage 1 dispatch tests on contest video |
| Static residual additive composition (alpha=0.1) | CARGO-CULTED at L0 | L1: learned composition gate per Atick-Redlich 1990 |
| PR106 base codec inherited as-is at L0 | HARD-EARNED-EMPIRICAL (PR106 frontier anchor) | N/A |
| Per-pair score-table u8 (64 values) | CARGO-CULTED at L0 (Pr106BaseCodecPlaceholder) | L1: actual PR106 format0d runtime |
| IA3 γ-zero init (`ia3_init_delta_std=0.01`) | HARD-EARNED-LITERATURE (Liu 2022 §3.2 zero-init) | N/A |
| Per-pair ego-pose ∈ R^6 | HARD-EARNED-EMPIRICAL (upstream PoseNet first 6 dims; canonical) | N/A |

---

## Observability surface

1. **Inspectable per layer** — per-pair score_indices + per-pair latent + per-pair ego_poses inspectable via Parameter/buffer access.
2. **Decomposable per signal** — score-aware loss returns `(loss, parts)` dict per Catalog #305.
3. **Diff-able across runs** — pack-then-parse roundtrip + state_dict comparison; seeded init.
4. **Queryable post-hoc** — provenance JSON with `n_params` / `composition_alpha` / `ia3_init_delta_std` / `pr106_score_table_size` / `hardware_substrate_detected`.
5. **Cite-able** — provenance carries `git_head` + `trainer` + `lane_id` + `substrate_tag` + `started_at` UTC ISO.
6. **Counterfactual-able** — Catalog #139 byte-mutation smoke planned for pr106_base + ia3_decoder + latents + ego_pose blobs.

---

## Predicted ΔS band

**Status:** UNKNOWN at L0 SCAFFOLD per Catalog #324.

**Dykstra feasibility check:** PR106 + HNeRV-IA3 composition is a sum-of-two-rate-and-distortion problem; predicted SUPER_ADDITIVE per cross_codec_orthogonality_predictor_consumer on the basis of CROSS-CANDIDATE finding #3 + IA3 γ-only adapter literature (Liu 2022).

**First-principles citation:** Shannon 1948 R(D) bound + Atick-Redlich 1990 cooperative-receiver + Liu 2022 IA3 parameter-efficiency adapter bound.

**Probe disambiguator:** `tools/probe_pact_nerv_cross_codec_b_super_additive_alpha_ia3_sweep.py` (planned at Stage 1) sweeps composition_alpha × ia3_init_delta_std.

---

## Reactivation criteria for paid dispatch

1. STAIRCASE Step 17 dispatch operator-gated per Catalog #325.
2. Per-substrate symposium proceed-unconditional per Catalog #315.
3. L1 learned composition gate landed per Atick-Redlich 1990.
4. Real PR106 base codec runtime swapped in (placeholder retired).
5. Score-aware Lagrangian + canonical auth-eval helper wired per #226.
6. `research_only` flips to false; `predicted_band` declared per #324.
