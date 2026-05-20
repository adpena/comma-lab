# Pact-NeRV-VQ L0 SCAFFOLD design memo (WAVE-3-PACT-NERV-G2-MID-LOC-L0-BUILD)

**Date**: 2026-05-20T21:15Z
**Lane**: `lane_pact_nerv_vq_l0_scaffold_20260520`
**Status**: L0 SCAFFOLD; `research_only=true`; `dispatch_enabled=false`
**Variant**: #7 of PACT-NERV-ULTIMATE Group 2
**Literature anchor**: van den Oord 2017 VQ-VAE (arXiv:1711.00937)
**Predicted band**: `[-0.005, +0.003]` per PACT-NERV-ULTIMATE memo

## Distinguishing primitive

Per-pair vector-quantization of latents via learnable codebook + EMA update +
commitment loss (van den Oord 1711.00937 §3.1-3.2). Aaron van den Oord inner
council seat alignment. Discrete latents shrink per-pair latent bytes to
log2(codebook_size) bits each — the rate-axis lever IA3 / DistilledScorer
variants do not have.

## 9-dimension success checklist evidence

1. UNIQUENESS: VQ-VAE codebook + STE; no sister variant ships this primitive.
2. BEAUTY+ELEGANCE: ~300 LOC core; reviewable in 30s.
3. DISTINCTNESS: vs IA3 (γ-only conditioning, no quantization); vs Distilled (no codebook); vs Bayesian (no posterior); vs multi-modal (single-modal); vs diffusion (no per-step refinement).
4. RIGOR: Catalog #229 PV + #292 council assumption surfacing planned for Stage 1.
5. OPTIMIZATION-PER-TECHNIQUE: van den Oord canonical EMA codebook + commitment loss (HARD-EARNED §3.1); unique-and-complete-per-method per CLAUDE.md.
6. STACK-OF-STACKS-COMPOSABILITY: SUB-ADD with eval_roundtrip (both apply quantization); ORTH with PR110 fec6.
7. DETERMINISTIC-REPRODUCIBILITY: byte-stable archive (deterministic ZIP per Catalog #5).
8. EXTREME-OPTIMIZATION-PERFORMANCE: T1 engineering hooks declared.
9. OPTIMAL-MINIMAL-CONTEST-SCORE: predicted [-0.005, +0.003] per PACT-NERV-ULTIMATE; LITERATURE-PREDICTION (VQ-VAE excels at discrete representation, contest rate term may or may not benefit).

## ## Canonical-vs-unique decision per layer

- **Scorer-preprocess routing**: ADOPT_CANONICAL (`score_pair_components_dispatch`).
- **eval_roundtrip patching**: ADOPT_CANONICAL (`patch_upstream_yuv6_globally`).
- **HNeRV decoder backbone**: ADOPT_CANONICAL (DepthSep + SIREN + PixelShuffle).
- **Archive grammar / inflate**: FORK_BECAUSE_PRINCIPLED_MISMATCH (PVQ magic; CODEBOOK_BLOB + INDICES_BLOB are unique sections).
- **VectorQuantizerEMA class**: FORK_BECAUSE_SUPPRESSES (unique per-variant; van den Oord canonical pattern).
- **Score-aware Lagrangian**: ADOPT_CANONICAL + EXTEND with `commitment_weight * commitment_loss` term (unique).

## ## Cargo-cult audit per assumption

| Assumption | Classification | Unwind path (Stage 1 sweep) |
|---|---|---|
| VQ codebook + EMA update | HARD-EARNED | van den Oord 1711.00937 + inner council seat |
| Commitment loss weight 0.25 | HARD-EARNED | van den Oord §3.1 canonical |
| Codebook size 512 + codebook dim 24 | CARGO-CULTED | RVQ residual / FSQ Mentzer 2309.15505 / canonical VQ sweep |
| Per-pair single-token quantization | CARGO-CULTED | Per-pair sequence of tokens (multi-token) |
| Codebook decay 0.99 | HARD-EARNED | van den Oord canonical EMA decay |

## ## Observability surface

1. **Inspectable per layer**: `quantizer.last_indices` + `quantizer.last_commitment_loss` accessible per forward.
2. **Decomposable per signal**: `seg_term` / `pose_term` / `rate_term` / `commitment_term` separately tracked.
3. **Diff-able across runs**: deterministic seeds + canonical Provenance.
4. **Queryable post-hoc**: archive parses to (decoder_state_dict, codebook, indices, meta); codebook + indices separately inspectable.
5. **Cite-able**: provenance.json + canonical Provenance per Catalog #323.
6. **Counterfactual-able**: byte-mutation smoke test (Catalog #139).

## ## horizon-class

**horizon_class: plateau_adjacent** — VQ-VAE rate gain is bounded by log2(codebook_size) compression; not a class-shift architecturally.

## ## Predicted ΔS band

`[-0.005, +0.003]` per PACT-NERV-ULTIMATE memo. Dykstra-feasibility: codebook 512×24 ≈ 12K params + indices uint16 × 600 pairs = 1200 bytes. Brotli ~4-6 KB. Rate cost ~0.000005. VQ may improve rate by replacing fp16 latents with uint16 indices (savings ~600×8 bytes vs 600×24×2 = 14.4 KB → 1.2 KB; rate-axis win ~0.0085 if codebook is rate-justified). Distortion impact bounded by codebook size.

## Reactivation criteria

1. PACT-NERV symposium Stage 1 dispatch operator-gated per Catalog #325.
2. Cargo-cult audit unwinds CARGO-CULTED assumptions (codebook_size + per-pair-single-token + per-block modulation).
3. 9-dim checklist + observability + Dykstra feasibility land.
4. `_full_main` replaces NotImplementedError with real score-aware loop.
5. Recipe `research_only` flips false; `dispatch_enabled` flips true.

## Cross-references

- Substrate package: `src/tac/substrates/pact_nerv_vq/`
- Trainer: `experiments/train_substrate_pact_nerv_vq.py`
- Recipe: `.omx/operator_authorize_recipes/substrate_pact_nerv_vq_modal_t4_dispatch.yaml`
- Driver: `scripts/remote_lane_substrate_pact_nerv_vq.sh`
- Tests: `src/tac/substrates/pact_nerv_vq/tests/`
- Sister memo: `.omx/research/pact_nerv_ultimate_research_and_design_20260520T193443Z.md`
- Landing memo: `feedback_wave_3_pact_nerv_g2_mid_loc_l0_build_landed_20260520.md`
