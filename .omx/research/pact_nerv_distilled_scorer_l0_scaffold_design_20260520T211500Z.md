# Pact-NeRV-DistilledScorer L0 SCAFFOLD design memo (WAVE-3-PACT-NERV-G2-MID-LOC-L0-BUILD)

**Date**: 2026-05-20T21:15Z
**Lane**: `lane_pact_nerv_distilled_scorer_l0_scaffold_20260520`
**Status**: L0 SCAFFOLD; `research_only=true`; `dispatch_enabled=false`
**Variant**: #6 of PACT-NERV-ULTIMATE Group 2 (mid-LOC apparatus-aligned)
**Literature anchor**: Hinton-Vinyals-Dean 2015 (arXiv:1503.02531; KL-T=2.0 distillation)
**Predicted band**: `[-0.003, +0.001]` per PACT-NERV-ULTIMATE memo (HARD-EARNED-via-Quantizr-anchor)

## Distinguishing primitive

A small Conv2d-based `DistilledScorerSurrogate` is co-trained via Hinton-Vinyals-Dean
KL-T=2.0 distillation to mimic frozen upstream SegNet + PoseNet logits. Its
globally-pooled feature vector then conditions the HNeRV decoder via per-block
channel-bias projection.

The surrogate IS the structural distinguishing element vs sister IA3 / VQ /
Bayesian variants. Hinton inner council seat alignment per CLAUDE.md.

## 9-dimension success checklist evidence

1. UNIQUENESS: distilled-scorer-surrogate-as-conditioner; no sister variant ships this primitive.
2. BEAUTY+ELEGANCE: ~250 LOC core; reviewable in 30s per HNeRV parity L12.
3. DISTINCTNESS: vs IA3 (no γ-only modulation); vs VQ (no codebook); vs Bayesian (no posterior); vs multi-modal (single-tower distill not 3-tower fusion); vs diffusion (no per-step refinement).
4. RIGOR: Catalog #229 premise verification + Catalog #292 council assumption surfacing planned for Stage 1 symposium.
5. OPTIMIZATION-PER-TECHNIQUE: per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD — canonical helpers used WHERE THEY SERVE (scorer-preprocess routing per Catalog #164, fcntl-locked state, canonical Provenance); the distillation surrogate is unique to this variant.
6. STACK-OF-STACKS-COMPOSABILITY: ADDITIVE with Pact-NeRV-A1 per PACT-NERV-ULTIMATE composability matrix (cross-attention provides additional conditioning surface); ORTH with PR110 fec6 selector.
7. DETERMINISTIC-REPRODUCIBILITY: byte-stable archive (deterministic ZIP per Catalog #5); seed-pinned trainer via `_canonical_pin_seeds`.
8. EXTREME-OPTIMIZATION-PERFORMANCE: T1 engineering hooks declared in `TIER_1_OPERATOR_REQUIRED_FLAGS` for Stage 1 (autocast_fp16 / torch.compile / gt_scorer_cache).
9. OPTIMAL-MINIMAL-CONTEST-SCORE: predicted [-0.003, +0.001] per PACT-NERV-ULTIMATE; HARD-EARNED-via-Quantizr-0.33-anchor. <!-- HISTORICAL_SCORE_LITERAL_OK:quantizr_0_33_anchor_in_dim_9_evidence -->

## ## Canonical-vs-unique decision per layer

- **Scorer-preprocess routing**: ADOPT_CANONICAL_BECAUSE_SERVES (`tac.substrates.score_aware_common.score_pair_components_dispatch`; identical scorer protocol).
- **eval_roundtrip patching**: ADOPT_CANONICAL_BECAUSE_SERVES (`patch_upstream_yuv6_globally` non-negotiable per Catalog #6).
- **HNeRV decoder backbone**: ADOPT_CANONICAL_BECAUSE_SERVES (DepthSep + SIREN + PixelShuffle; HNeRV-class).
- **Archive grammar / inflate**: FORK_BECAUSE_PRINCIPLED_MISMATCH (PDS magic + distinct header schema; surrogate weights logical-grouped in DECODER_BLOB).
- **DistilledScorerSurrogate class**: FORK_BECAUSE_SUPPRESSES (unique per-variant distinguishing primitive; canonical IA3 / VQ wrappers would suppress).
- **Score-aware Lagrangian terms**: ADOPT_CANONICAL_BECAUSE_SERVES (`CONTEST_POSE_SQRT_WEIGHT` + score-domain canonical formula) + EXTEND with `delta_distill * KL_distill_T2` term (unique).

## ## Cargo-cult audit per assumption

| Assumption | Classification | Unwind path (Stage 1 sweep) |
|---|---|---|
| KL-T=2.0 distill from frozen teacher | HARD-EARNED | Hinton 1503.02531 §3 + Quantizr 0.33 anchor + Hinton inner council seat |
| Internal scorer surrogate REPLACES direct scorer routing at training | CARGO-CULTED-MAY-BE-PROMISING | Hybrid: distill early epochs + direct scorer late epochs |
| Cross-attention-to-frozen-scorer-features (per-block channel bias only) | CARGO-CULTED | KL-on-logits-only per pure Hinton; or full cross-attention |
| Surrogate hidden = 32 / feature_dim = 16 | CARGO-CULTED | Stage 1 sweep over {8,16,32,64} |

## ## Observability surface

1. **Inspectable per layer**: surrogate `forward` returns (B, feature_dim) — operator can hook + log per-pair.
2. **Decomposable per signal**: `seg_term` / `pose_term` / `rate_term` / `distill_kl_term` all separately tracked in `score_aware_loss.parts`.
3. **Diff-able across runs**: deterministic seeds + canonical Provenance.
4. **Queryable post-hoc**: archive parses to (decoder_state_dict, latents, meta) with surrogate weights logical-grouped.
5. **Cite-able**: provenance.json carries git_head + lane_id + canonical_provenance per Catalog #323.
6. **Counterfactual-able**: byte-mutation smoke test in test suite (Catalog #139).

## ## horizon-class

**horizon_class: plateau_adjacent** — predicted band straddles the 0.196-0.199 cluster per PACT-NERV-ULTIMATE. PR101-class refinement, not class-shift.

## ## Predicted ΔS band

`[-0.003, +0.001]` per PACT-NERV-ULTIMATE memo (HARD-EARNED-via-Quantizr-0.33-anchor; KL-T=2.0 IS the canonical Quantizr technique). Dykstra-feasibility check: surrogate adds ~10K params; brotli compresses to ~5-8 KB; rate cost ~0.000007 — well within polytope. Distill cross-attention may add 0.001-0.003 score reduction if surrogate features carry pose+seg signal the decoder otherwise lacks. <!-- HISTORICAL_SCORE_LITERAL_OK:quantizr_0_33_in_dykstra_band_check -->

## Reactivation criteria

1. PACT-NERV symposium Stage 1 dispatch operator-gated per Catalog #325.
2. Cargo-cult audit unwinds the 3 CARGO-CULTED assumptions above.
3. 9-dim checklist + observability + Dykstra feasibility land.
4. `_full_main` replaces NotImplementedError with real KL-T=2.0 distillation loop.
5. Recipe `research_only` flips false; `dispatch_enabled` flips true.

## Cross-references

- Substrate package: `src/tac/substrates/pact_nerv_distilled_scorer/`
- Trainer: `experiments/train_substrate_pact_nerv_distilled_scorer.py`
- Recipe: `.omx/operator_authorize_recipes/substrate_pact_nerv_distilled_scorer_modal_t4_dispatch.yaml`
- Driver: `scripts/remote_lane_substrate_pact_nerv_distilled_scorer.sh`
- Tests: `src/tac/substrates/pact_nerv_distilled_scorer/tests/`
- Sister memo: `.omx/research/pact_nerv_ultimate_research_and_design_20260520T193443Z.md`
- Landing memo: `feedback_wave_3_pact_nerv_g2_mid_loc_l0_build_landed_20260520.md`
