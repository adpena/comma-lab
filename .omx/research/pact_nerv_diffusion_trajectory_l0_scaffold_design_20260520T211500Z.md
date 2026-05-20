# Pact-NeRV-DiffusionTrajectory L0 SCAFFOLD design memo (WAVE-3-PACT-NERV-G2-MID-LOC-L0-BUILD)

**Date**: 2026-05-20T21:15Z
**Lane**: `lane_pact_nerv_diffusion_trajectory_l0_scaffold_20260520`
**Status**: L0 SCAFFOLD; `research_only=true`; `dispatch_enabled=false`; PAPER target only.
**Variant**: #10 of PACT-NERV-ULTIMATE Group 2 (paper-worthy bleeding-edge)
**Literature anchor**: Rombach 2022 LDM (arXiv:2112.10752) + Blattmann 2023 video latent diffusion (arXiv:2304.08818)
**Predicted band**: `[+0.001, +0.020]` per PACT-NERV-ULTIMATE memo (predicted regression on contest rate term)

## Distinguishing primitive

Per-pair latent diffusion trajectory: store per-pair Gaussian noise SEEDS +
a learnable LIGHTWEIGHT diffusion-trajectory predictor (5-step depth-2 MLP).
At inflate time the predictor refines noise → latent in 5 steps per pair.

Per the PACT-NERV-ULTIMATE memo: CARGO-CULTED-MAY-BE-PROMISING + RISK-LOC-EXCESS.
At L0 task-cap (~400 LOC) we ship the LIGHTWEIGHT variant; full UNet latent
diffusion is LOC-prohibitive at this budget. PAPER target only at this LOC.

## 9-dimension success checklist evidence

1. UNIQUENESS: per-pair latent diffusion trajectory; no sister variant ships this primitive.
2. BEAUTY+ELEGANCE: ~400 LOC core; reviewable in 30s.
3. DISTINCTNESS: vs IA3 / VQ / Distilled / Bayesian / multi-modal (all single-step latent paths); this variant is iterative refinement.
4. RIGOR: Catalog #229 PV + #292 council planned. PAPER scope acknowledged honestly.
5. OPTIMIZATION-PER-TECHNIQUE: lightweight per-step MLP (CARGO-CULTED at L0; full UNet at L1+).
6. STACK-OF-STACKS-COMPOSABILITY: SUB-ADD with Pact-NeRV-DiffusionDistilled (sister variant #3); ORTH with PR110 fec6.
7. DETERMINISTIC-REPRODUCIBILITY: byte-stable archive; deterministic trajectory at inflate.
8. EXTREME-OPTIMIZATION-PERFORMANCE: T1 hooks declared.
9. OPTIMAL-MINIMAL-CONTEST-SCORE: predicted [+0.001, +0.020]; PAPER-only target per PACT-NERV-ULTIMATE; predicted regression on contest rate term.

## ## Canonical-vs-unique decision per layer

- **Scorer-preprocess routing**: ADOPT_CANONICAL.
- **eval_roundtrip patching**: ADOPT_CANONICAL.
- **HNeRV decoder backbone**: ADOPT_CANONICAL.
- **Archive grammar**: FORK_BECAUSE_PRINCIPLED_MISMATCH (PDT magic; NUM_TIMESTEPS u8 header field).
- **DiffusionTrajectoryPredictor class**: FORK_BECAUSE_SUPPRESSES (unique per-variant; Rombach canonical pattern).
- **Score-aware Lagrangian**: ADOPT_CANONICAL (no diffusion-specific term; trajectory operates on latents pre-decoder).

## ## Cargo-cult audit per assumption

| Assumption | Classification | Unwind path |
|---|---|---|
| 5 diffusion timesteps | CARGO-CULTED | Sweep {2, 5, 10, 20} |
| Per-step depth-2 MLP | CARGO-CULTED | Shared MLP / full UNet (LOC-prohibitive at L0) |
| Gaussian noise seed (per-pair int16) | HARD-EARNED | Rombach §3.1 canonical |
| Linear noise schedule | CARGO-CULTED | Cosine schedule per Nichol-Dhariwal 2102.09672 |

## ## Observability surface

1. **Inspectable per layer**: per-step `predictor` MLPs accessible; `alphas` buffer queryable.
2. **Decomposable per signal**: `seg_term` / `pose_term` / `rate_term` separately tracked.
3. **Diff-able across runs**: deterministic seeds + canonical Provenance.
4. **Queryable post-hoc**: archive parses to (decoder_state_dict, seeds, meta, num_timesteps); predictor weights logical-grouped via `predictor.` prefix.
5. **Cite-able**: provenance.json.
6. **Counterfactual-able**: byte-mutation smoke test (Catalog #139).

## ## horizon-class

**horizon_class: plateau_adjacent** — lightweight diffusion at L0 is bounded; full UNet at L1+ might cross into frontier-pursuit. CARGO-CULTED-MAY-BE-PROMISING per PACT-NERV-ULTIMATE.

## ## Predicted ΔS band

`[+0.001, +0.020]` per PACT-NERV-ULTIMATE (regression-leaning band). Dykstra-feasibility: predictor adds ~2K params per timestep × 5 = 10K params. Brotli ~5-8 KB. Seeds stored at int16 per-pair (600 × 24 × 2 = 28.8 KB). Total rate cost ~35-40 KB vs ~14 KB direct latents. Net rate-axis REGRESSION ~+0.018. Distortion improvement bounded by 5-step refinement quality. PAPER target only.

## Reactivation criteria

1. PACT-NERV Stage 1 dispatch operator-gated per Catalog #325. PAPER target; not frontier candidate.
2. Cargo-cult audit unwinds CARGO-CULTED assumptions.
3. 9-dim checklist + observability + Dykstra feasibility.
4. `_full_main` replaces NotImplementedError.
5. Recipe `research_only` flips false (HIGHLY UNLIKELY for this variant at L0 LOC budget).

## Cross-references

- Substrate package: `src/tac/substrates/pact_nerv_diffusion_trajectory/`
- Trainer: `experiments/train_substrate_pact_nerv_diffusion_trajectory.py`
- Recipe: `.omx/operator_authorize_recipes/substrate_pact_nerv_diffusion_trajectory_modal_t4_dispatch.yaml`
- Driver: `scripts/remote_lane_substrate_pact_nerv_diffusion_trajectory.sh`
- Tests: `src/tac/substrates/pact_nerv_diffusion_trajectory/tests/`
- Sister memo: `.omx/research/pact_nerv_ultimate_research_and_design_20260520T193443Z.md`
- Landing memo: `feedback_wave_3_pact_nerv_g2_mid_loc_l0_build_landed_20260520.md`
