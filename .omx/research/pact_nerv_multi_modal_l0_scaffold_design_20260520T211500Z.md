# Pact-NeRV-MultiModal L0 SCAFFOLD design memo (WAVE-3-PACT-NERV-G2-MID-LOC-L0-BUILD)

**Date**: 2026-05-20T21:15Z
**Lane**: `lane_pact_nerv_multi_modal_l0_scaffold_20260520`
**Status**: L0 SCAFFOLD; `research_only=true`; `dispatch_enabled=false`
**Variant**: #9 of PACT-NERV-ULTIMATE Group 2
**Literature anchor**: Baltrušaitis 2019 multimodal-fusion taxonomy + Radford 2021 CLIP (arXiv:2103.00020)
**Predicted band**: `[-0.004, +0.002]` per PACT-NERV-ULTIMATE memo

## Distinguishing primitive

3-tower conditioning fusion (ego-pose + SegNet-class-prior + odometry) via
concatenated late-fusion projection per Baltrušaitis 2019 multimodal-fusion
taxonomy. The fused vector conditions HNeRV decoder via per-block channel-bias.

Per Catalog #311 sister discipline: ego-motion-conditioned conditioning is the
canonical Rao-Ballard + Atick-Redlich predictive-coding framing; this variant
extends it with class-aware + odometry side information.

## 9-dimension success checklist evidence

1. UNIQUENESS: 3-tower fusion; no sister variant ships multi-modal conditioning.
2. BEAUTY+ELEGANCE: ~250 LOC core; reviewable in 30s.
3. DISTINCTNESS: vs IA3 (single-modal pose); vs VQ (no conditioning); vs Distilled (single-tower scorer surrogate); vs Bayesian (no posterior); vs diffusion (no per-step refinement).
4. RIGOR: Catalog #229 PV + #292 council planned.
5. OPTIMIZATION-PER-TECHNIQUE: per-tower projection (Baltrušaitis canonical).
6. STACK-OF-STACKS-COMPOSABILITY: ADDITIVE with Pact-NeRV-A1 per PACT-NERV-ULTIMATE (subsumes single-modality conditioning); ORTH with PR110 fec6.
7. DETERMINISTIC-REPRODUCIBILITY: byte-stable archive.
8. EXTREME-OPTIMIZATION-PERFORMANCE: T1 hooks declared.
9. OPTIMAL-MINIMAL-CONTEST-SCORE: predicted [-0.004, +0.002]; multi-modal fusion improves downstream tasks at LOC cost.

## ## Canonical-vs-unique decision per layer

- **Scorer-preprocess routing**: ADOPT_CANONICAL.
- **eval_roundtrip patching**: ADOPT_CANONICAL.
- **HNeRV decoder backbone**: ADOPT_CANONICAL.
- **Archive grammar**: FORK_BECAUSE_PRINCIPLED_MISMATCH (PMM magic; 3 per-tower data blobs).
- **MultiModalConditioningFusion class**: FORK_BECAUSE_SUPPRESSES (unique per-variant; Baltrušaitis canonical concat-fusion).
- **Score-aware Lagrangian**: ADOPT_CANONICAL (no fusion-specific term; the fusion conditions decoder features, not the loss).

## ## Cargo-cult audit per assumption

| Assumption | Classification | Unwind path |
|---|---|---|
| 3-tower concat late-fusion | CARGO-CULTED | Cross-attention per Baltrušaitis 2019 taxonomy / gated fusion (Highway) |
| Tower weights shared across (frame_0, frame_1) | CARGO-CULTED | Per-frame tower projections (doubles head count) |
| 5-class SegNet prior dim | HARD-EARNED | Matches upstream SegNet classes; aligns with Catalog #311 |
| 4-D odometry vector | CARGO-CULTED | 9-DoF IMU per Comma2k19 schema |
| fusion_dim 16 | CARGO-CULTED | Sweep {8, 16, 32, 64} |

## ## Observability surface

1. **Inspectable per layer**: `fusion(pose, class_prior, odometry)` returns (B, fusion_dim) — operator-hookable.
2. **Decomposable per signal**: per-tower projections (h_p / h_c / h_o) separately inspectable in `fusion.forward`.
3. **Diff-able across runs**: deterministic seeds.
4. **Queryable post-hoc**: archive parses to (decoder_state_dict, latents, pose, class_prior, odometry, meta); per-tower data separately queryable.
5. **Cite-able**: provenance.json.
6. **Counterfactual-able**: byte-mutation smoke test.

## ## horizon-class

**horizon_class: plateau_adjacent** — multi-modal conditioning is bounded gain unless ego-motion side info is structurally novel. Pure concat fusion is plateau-class; full cross-attention (Stage 1 ablation) might cross into frontier-pursuit.

## ## Predicted ΔS band

`[-0.004, +0.002]` per PACT-NERV-ULTIMATE. Dykstra-feasibility: 3-tower fusion adds ~500 weights at smoke config. Brotli ~3-5 KB. Per-pair conditioning data (pose 6 + class 5 + odometry 4) × 600 pairs × int16 = 18 KB. Significant rate cost but conditioning may improve distortion by ~0.002-0.004 if multimodal signal is informative on this video.

## Reactivation criteria

1. PACT-NERV Stage 1 dispatch operator-gated per Catalog #325.
2. Cargo-cult audit unwinds CARGO-CULTED assumptions.
3. 9-dim checklist + observability + Dykstra feasibility.
4. `_full_main` replaces NotImplementedError.
5. Recipe `research_only` flips false.

## Cross-references

- Substrate package: `src/tac/substrates/pact_nerv_multi_modal/`
- Trainer: `experiments/train_substrate_pact_nerv_multi_modal.py`
- Recipe: `.omx/operator_authorize_recipes/substrate_pact_nerv_multi_modal_modal_t4_dispatch.yaml`
- Driver: `scripts/remote_lane_substrate_pact_nerv_multi_modal.sh`
- Tests: `src/tac/substrates/pact_nerv_multi_modal/tests/`
- Sister memo: `.omx/research/pact_nerv_ultimate_research_and_design_20260520T193443Z.md`
- Landing memo: `feedback_wave_3_pact_nerv_g2_mid_loc_l0_build_landed_20260520.md`
