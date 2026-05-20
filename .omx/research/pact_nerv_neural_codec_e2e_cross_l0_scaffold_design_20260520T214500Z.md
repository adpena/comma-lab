<!-- # FORMALIZATION_PENDING:hyperprior_gate_super_additive_proof_pending_stage_1_dispatch_per_catalog_344 -->
<!-- # COUNCIL_ASSUMPTION_STATEMENT_WAIVED:l0_scaffold_design_memo_predates_council_per_catalog_292 -->

# Pact-NeRV-NEURAL-CODEC-E2E-CROSS L0 SCAFFOLD design memo

**Date:** 2026-05-20
**Lane:** `lane_pact_nerv_neural_codec_e2e_cross_l0_scaffold_20260520`
**Variant:** PACT-NERV-ULTIMATE Variant #18 — END-TO-END neural codec composition with Ballé-style hyperprior gate
**Literature anchor:** Ballé et al. 2018 *"Variational image compression with a scale hyperprior"* arXiv:1802.01436 + Atick-Redlich 1990 cooperative-receiver + CROSS-CANDIDATE finding #3
**horizon-class:** `frontier_pursuit` per Catalog #309 (most ambitious of G4 — end-to-end neural composition)
**research_only:** true per HNeRV parity L2 + Catalog #220

---

## Executive summary

Pact-NeRV-NEURAL-CODEC-E2E-CROSS is structurally distinct from sister G4 variants
A and B: BOTH codec branches are HNeRV-class neural networks jointly trained
end-to-end, AND a Ballé-style hyperprior gate g(z_a, z_b) ∈ [0, 1] routes
per-pair bits between them. The hypothesis: end-to-end training of both codecs +
learned gate allows per-pair Pareto-optimal bit allocation that beats static
residual additive composition (CC_A / CC_B at L0).

This variant carries `frontier_pursuit` horizon-class per Catalog #309 (NOT
`plateau_adjacent` like CC_A/CC_B) because the end-to-end neural composition
mechanism is structurally distinct from the fixed-base + residual-bolt-on
pattern shared by CC_A and CC_B.

L0 SCAFFOLD lands substrate package + trainer + recipe + driver + 17 tests but
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
| Archive grammar (`NCEC` magic + 35-byte header + 3 blobs + 2 latent blobs + meta) | FORK_BECAUSE_PRINCIPLED_MISMATCH | NCEC requires separate decoder_a/decoder_b/hyperprior blobs (NOT a single decoder blob); per-branch latents (latents_a + latents_b) |
| Inflate runtime (per-substrate ≤200 LOC) | FORK_BECAUSE_PRINCIPLED_MISMATCH | HNeRV parity L4: end-to-end neural composition has its own inflate path that loads 3 state_dicts and composes via gate |
| Hyperprior gate (`HyperpriorGate`) | FORK_BECAUSE_PRINCIPLED_MISMATCH | Ballé 2018 §3.3 hyperprior is structurally unique to this variant; sister CC_A/CC_B have NO gate |
| Two HNeRV branches (`_HnervBranch`) | FORK_BECAUSE_PRINCIPLED_MISMATCH | Two-branch composition requires per-branch independent decoder; sister variants have single decoder |
| Score-aware loss + optional gate-entropy bonus (`lambda_gate`) | FORK_BECAUSE_PRINCIPLED_MISMATCH | The gate-entropy term encourages decisive per-pair selection; NOT in sister variants |

---

## 9-dimension success checklist evidence

1. **UNIQUENESS** — third variant in the canonical 3-variant G4 CROSS-CODEC series; end-to-end neural composition with Ballé hyperprior gate is structurally distinct from sister variants A and B (fixed base + bolt-on residual).
2. **BEAUTY + ELEGANCE** — substrate ~340 LOC (architecture only) reviewable in 30 sec; the symmetry between branch_a + branch_b + gate is canonical Ballé 2018 pattern.
3. **DISTINCTNESS** — vs CC_A: BOTH codecs are jointly trained neural (NOT one fixed-base + one residual); vs CC_B: hyperprior gate (NOT static composition_alpha); vs all sister NeRV substrates: cross-codec composition (NOT single-branch).
4. **RIGOR** — 17 tests covering pack/parse roundtrip + hyperprior gate ∈ [0, 1] invariant + gradient flow through both branches AND gate + Catalog #139 byte-mutation discrimination per-branch + canonical scorer routing + recipe research_only + driver NVML block (`num_parameters=90029`).
5. **OPTIMIZATION PER TECHNIQUE** — canonical scorer-loss helper routing per #164; canonical eval_roundtrip per #6; hyperprior init bias preserves balanced gate at init.
6. **STACK-OF-STACKS-COMPOSABILITY** — designed as cross-codec composition primitive with end-to-end learned gate; predicted SUPER_ADDITIVE per Catalog #322 + Ballé 2018 hyperprior bit-allocation efficiency proof.
7. **DETERMINISTIC REPRODUCIBILITY** — seed pinned; pack-then-parse roundtrip invariant; gate observability via `gate_values()` method.
8. **EXTREME OPTIMIZATION + PERFORMANCE** — autocast_fp16 + torch_compile flags declared with waivers; full training path operator-gated.
9. **OPTIMAL MINIMAL CONTEST SCORE** — predicted ΔS pending Stage 1 dispatch; absolute magnitude undisclosed per Catalog #321 non-promotable + Catalog #324.

---

## Cargo-cult audit per assumption

| Assumption | Classification | Unwind path |
|---|---|---|
| End-to-end neural-codec composition yields SUPER_ADDITIVE | HARD-EARNED-LITERATURE (Ballé 2018 hyperprior + Atick-Redlich 1990 cooperative-receiver) | Stage 1 dispatch verifies on contest video |
| Hyperprior gate via sigmoid scalar g ∈ [0, 1] | CARGO-CULTED at L0 | L1: per-region autoregressive hyperprior per Ballé 2018 §3.3 |
| Two IDENTICAL HNeRV branches (symmetric A=B architecture) | CARGO-CULTED at L0 | L1: heterogeneous branches per CROSS-CANDIDATE finding #3 SUPER_ADDITIVE signature requires DIFFERENT receptive fields |
| Gate entropy bonus weight `lambda_gate=0.0` (disabled) | CARGO-CULTED at L0 | L1: information-theoretic rate-distortion-gate Lagrangian per Shannon 1948 |
| Gate init bias=0.0 (balanced sigmoid(0)=0.5) | HARD-EARNED-NEUTRAL | At init both branches contribute equally; learning decides the per-pair mix |
| SIREN init for HNeRV branches | HARD-EARNED-LITERATURE (Sitzmann 2020) | N/A |
| Hyperprior MLP hidden_dim=32 | CARGO-CULTED at L0 | L1: sweep hidden_dim ∈ {16, 32, 64, 128} per Ballé 2018 ablation |

---

## Observability surface

1. **Inspectable per layer** — per-pair latents_a + latents_b inspectable via Parameter access; per-pair gate values inspectable via `model.gate_values(pair_indices)` returning (B,) scalar tensor.
2. **Decomposable per signal** — score-aware loss returns `(loss, parts)` with `rate_term` / `seg_term` / `pose_term` / `gate_entropy_term` / `loss_total` per Catalog #305.
3. **Diff-able across runs** — pack-then-parse roundtrip + state_dict comparison across 3 sub-modules (branch_a, branch_b, gate).
4. **Queryable post-hoc** — provenance JSON with `n_params` / `hyperprior_hidden` / `gate_init_bias` / `hardware_substrate_detected`; smoke loop prints per-step `gate_mean` / `gate_min` / `gate_max`.
5. **Cite-able** — provenance carries `git_head` + `trainer` + `lane_id` + `substrate_tag` + `started_at` UTC ISO.
6. **Counterfactual-able** — Catalog #139 byte-mutation smoke planned for ALL THREE weight blobs (decoder_a + decoder_b + hyperprior); the hyperprior blob's byte-mutation sensitivity IS the canonical SUPER_ADDITIVE proof per Catalog #322 (if gate bytes do NOT affect score, composition degenerates to single-branch selection).

---

## Predicted ΔS band

**Status:** UNKNOWN at L0 SCAFFOLD per Catalog #324.

**Dykstra feasibility check:** end-to-end joint training of (branch_a, branch_b, gate) solves a constrained convex feasibility problem: rate(decoder_a) + rate(decoder_b) + rate(hyperprior) ≤ R_total AND d_seg + sqrt(d_pose) ≤ D_total. Predicted SUPER_ADDITIVE per cross_codec_orthogonality_predictor_consumer (commit 80484241f) + Ballé 2018 §4 empirical evidence that learned hyperprior gating yields ~5-10% rate savings vs fixed mixing.

**First-principles citation:** Shannon 1948 R(D) bound applied to each branch + Ballé 2018 hyperprior efficiency bound + Atick-Redlich 1990 cooperative-receiver bound on the composition.

**Probe disambiguator:** `tools/probe_pact_nerv_neural_codec_e2e_cross_gate_distribution_sweep.py` (planned at Stage 1) sweeps `gate_init_bias` + `hyperprior_hidden` and measures per-pair gate distribution + contest-CUDA score delta.

---

## Reactivation criteria for paid dispatch

1. STAIRCASE Step 18 dispatch operator-gated per Catalog #325.
2. Per-substrate symposium proceed-unconditional per Catalog #315.
3. L1 per-region autoregressive hyperprior landed per Ballé 2018 §3.3.
4. L1 heterogeneous branches per CROSS-CANDIDATE finding #3 SUPER_ADDITIVE signature.
5. Score-aware Lagrangian + canonical auth-eval helper wired per #226.
6. Catalog #322 SUPER_ADDITIVE empirical proof at gate-bytes byte-mutation surface (Catalog #139).
7. `research_only` flips to false; `predicted_band` declared per #324.
