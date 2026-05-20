# Pact-NeRV-Bayesian L0 SCAFFOLD design memo (WAVE-3-PACT-NERV-G2-MID-LOC-L0-BUILD)

**Date**: 2026-05-20T21:15Z
**Lane**: `lane_pact_nerv_bayesian_l0_scaffold_20260520`
**Status**: L0 SCAFFOLD; `research_only=true`; `dispatch_enabled=false`
**Variant**: #8 of PACT-NERV-ULTIMATE Group 2
**Literature anchor**: Blundell et al. 2015 Bayes by Backprop (arXiv:1505.05424) + Kingma 2014 VAE (arXiv:1312.6114) + MacKay 1992
**Predicted band**: `[-0.002, +0.003]` per PACT-NERV-ULTIMATE memo

## Distinguishing primitive

Bayesian latent embedding layer per Blundell 1505.05424 Bayes by Backprop:
each weight is a learnable Gaussian (mu, log_sigma). Training samples via
reparameterization trick; inflate uses posterior mean per Blundell §4. KL
divergence vs unit-Gaussian prior regularizes per §3.1. MacKay inner council
(memorial) seat alignment.

Per-pair posterior variance IS per-pair difficulty signal — uncertainty IS
difficulty per the MacKay framing.

## 9-dimension success checklist evidence

1. UNIQUENESS: Bayesian latent embed via Bayes-by-Backprop; no sister variant ships this primitive.
2. BEAUTY+ELEGANCE: ~350 LOC core; reviewable in 30s.
3. DISTINCTNESS: vs IA3 (deterministic γ-only); vs VQ (no posterior); vs Distilled (no surrogate); vs multi-modal (no uncertainty); vs diffusion (no per-step noise).
4. RIGOR: Catalog #229 PV + #292 council assumption surfacing planned.
5. OPTIMIZATION-PER-TECHNIQUE: Blundell §3.2 reparameterization trick (HARD-EARNED).
6. STACK-OF-STACKS-COMPOSABILITY: ADDITIVE with per-pair difficulty conditioning per PACT-NERV-ULTIMATE memo.
7. DETERMINISTIC-REPRODUCIBILITY: byte-stable archive; inflate uses posterior mean (deterministic).
8. EXTREME-OPTIMIZATION-PERFORMANCE: T1 engineering hooks declared.
9. OPTIMAL-MINIMAL-CONTEST-SCORE: predicted [-0.002, +0.003] per PACT-NERV-ULTIMATE; LITERATURE-PREDICTION via MacKay inner council seat.

## ## Canonical-vs-unique decision per layer

- **Scorer-preprocess routing**: ADOPT_CANONICAL.
- **eval_roundtrip patching**: ADOPT_CANONICAL.
- **HNeRV decoder backbone (non-Bayesian)**: ADOPT_CANONICAL.
- **Archive grammar**: FORK_BECAUSE_PRINCIPLED_MISMATCH (PBN magic; posterior mean-only at inflate per Blundell §4).
- **BayesianLinearLayer class**: FORK_BECAUSE_SUPPRESSES (unique per-variant; Blundell canonical pattern).
- **Score-aware Lagrangian**: ADOPT_CANONICAL + EXTEND with `kl_weight * KL_div` term (unique).

## ## Cargo-cult audit per assumption

| Assumption | Classification | Unwind path |
|---|---|---|
| Bayes-by-Backprop reparameterization | HARD-EARNED | Blundell 1505.05424 + Kingma 1312.6114 + MacKay |
| Unit-Gaussian prior N(0,1) | HARD-EARNED | Blundell §3.1 canonical |
| KL weight 1.0 | CARGO-CULTED | KL annealing per epoch / fixed weight sweep |
| Bayesian ONLY on latent embedding layer | CARGO-CULTED | Bayesian decoder + Bayesian latents (full Bayesian) |
| Mean (not sample) at inflate | HARD-EARNED | Blundell §4 canonical |

## ## Observability surface

1. **Inspectable per layer**: `model.last_kl_div` after each forward; per-weight (mu, sigma) accessible via `bayesian_latent_embed.weight_mu` / `weight_rho`.
2. **Decomposable per signal**: `seg_term` / `pose_term` / `rate_term` / `kl_term` separately tracked.
3. **Diff-able across runs**: deterministic seeds + canonical Provenance.
4. **Queryable post-hoc**: archive parses to (decoder_state_dict, latents, meta); Bayesian mean params logical-grouped via `weight_mu` / `bias_mu` prefix.
5. **Cite-able**: provenance.json + canonical Provenance.
6. **Counterfactual-able**: byte-mutation smoke test.

## ## horizon-class

**horizon_class: plateau_adjacent** — KL regularization is bounded improvement; not a class-shift architecturally. PER-PAIR posterior variance MAY surface a frontier-pursuit if Stage 1 ablates the difficulty signal.

## ## Predicted ΔS band

`[-0.002, +0.003]` per PACT-NERV-ULTIMATE. Dykstra-feasibility: Bayesian latent embed doubles parameter count for that layer (mu + rho per weight). At smoke config 8x16=128 weights → 256 params; at full config 24x768=18,432 weights → 36,864 params. Brotli ~10-20 KB rate cost. Regularization may improve generalization marginally; predicted band centered near zero per MacKay's "structure prior" framing.

## Reactivation criteria

1. PACT-NERV Stage 1 dispatch operator-gated per Catalog #325.
2. Cargo-cult audit unwinds KL weight + Bayesian-only-latent-embed assumptions.
3. 9-dim checklist + observability + Dykstra feasibility land.
4. `_full_main` replaces NotImplementedError.
5. Recipe `research_only` flips false.

## Cross-references

- Substrate package: `src/tac/substrates/pact_nerv_bayesian/`
- Trainer: `experiments/train_substrate_pact_nerv_bayesian.py`
- Recipe: `.omx/operator_authorize_recipes/substrate_pact_nerv_bayesian_modal_t4_dispatch.yaml`
- Driver: `scripts/remote_lane_substrate_pact_nerv_bayesian.sh`
- Tests: `src/tac/substrates/pact_nerv_bayesian/tests/`
- Sister memo: `.omx/research/pact_nerv_ultimate_research_and_design_20260520T193443Z.md`
- Landing memo: `feedback_wave_3_pact_nerv_g2_mid_loc_l0_build_landed_20260520.md`
