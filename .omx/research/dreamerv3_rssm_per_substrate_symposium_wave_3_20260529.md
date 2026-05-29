---
council_tier: T2
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Hafner, Schmidhuber, PR95Author]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "The 1% unimix fix is canonical 1:1 with Hafner 2023; revisions remaining are the Path B2 PyTorch port + Modal smoke per symposium op-routable #3 which produces the sister empirical anchor with actual trained logits."
  - member: Assumption-Adversary
    verbatim: "The CARGO-CULTED-vs-HARD-EARNED classification of the 24x256 vs 32x32 G/K config is HARD-EARNED for both substrates (RL world model vs video compression dashcam), but the operator should be told explicitly that the C6 24-dim baseline match is an ARCHITECTURAL inheritance not a Hafner-2023 recommendation."
council_assumption_adversary_verdict:
  - assumption: "Hafner 2023 1% unimix is mandatory for all categorical posteriors regardless of domain"
    classification: HARD-EARNED
    rationale: "Hafner 2023 §3 explicitly states the unimix is the canonical robustness mechanism enabling fixed hyperparameters across diverse domains; the dashcam video compression substrate is structurally analogous (per-pair posterior must avoid one-hot collapse to preserve STE gradient flow during training)"
  - assumption: "Omitting GRU at L0 is HARD-EARNED-PROBLEM-SPACE not CARGO-CULTED"
    classification: HARD-EARNED
    rationale: "Per the contest scorer per-pair structure (600 independent pairs), temporal recurrence via GRU has no architectural target at L0; the per-pair-independent latent matches the scorer's operational structure; full RSSM with GRU is queued for L1+ per symposium decision"
  - assumption: "Omitting symlog observation squashing is HARD-EARNED-PROBLEM-SPACE not CARGO-CULTED"
    classification: HARD-EARNED
    rationale: "Hafner 2023 symlog squashes observations to handle diverse RL domains' value scales (Atari pixels vs proprioceptive floats); video frames are bound to [0, 255] uint8 natively + the sigmoid * 255 RGB head produces values in the same range; symlog is structurally redundant"
  - assumption: "Omitting KL balancing + free bits is HARD-EARNED-PROBLEM-SPACE not CARGO-CULTED"
    classification: HARD-EARNED
    rationale: "Hafner 2023 KL balance + free bits regularize the prior-posterior gap in the RSSM dynamics network; at L0 we have no separate prior network (per-pair learned logits ARE the posterior with no temporal prior); when GRU + dynamics prior is added at L1+ this classification must be re-evaluated"
council_decisions_recorded:
  - "op-routable #1: Land the 1% unimix fix as a HARD-EARNED canonical 1:1 element with the canonical Hafner 2023 source citation."
  - "op-routable #2: Register the first empirical anchor on canonical equation categorical_posterior_capacity_vs_continuous_gaussian_v1 per Catalog #344 (the closed-form mixture identity verifiable at fp32 precision)."
  - "op-routable #3: Queue Path B2 PyTorch port + Modal smoke as the sister empirical anchor source (the trained-logits anchor; this audit's anchor is the closed-form math identity)."
  - "op-routable #4: When L1+ extension adds GRU + dynamics prior, re-evaluate the KL balancing + free bits + symlog observation classifications per Catalog #303 cargo-cult audit."
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
council_override_rationale: null
related_deliberation_ids:
  - council_t3_dreamerv3_rssm_paradigm_bridge_per_substrate_symposium_20260519
  - dreamerv3_rssm_categorical_rd_canonical_equation_derivation_20260520T131815Z
---

# DreamerV3 RSSM per-substrate symposium — Wave 3 math-fidelity audit

Per Catalog #325 per-substrate optimal form symposium discipline + Wave 3
of the 12-wave 15-item math-fidelity audit cascade (operator blanket
approval 2026-05-29).

## 1. Cargo-cult audit per assumption (Catalog #303)

Per Catalog #303 cargo-cult audit section + the HARD-EARNED-vs-CARGO-CULTED
classification framework. Each canonical Hafner 2023 element is audited:

| Element | Implementation | Classification | Rationale |
|---|---|---|---|
| Straight-through gradient estimator | `use_straight_through=True` default in `gumbel_softmax_sample` | HARD-EARNED-CANONICAL-1:1 | Jang 2016 + Maddison 2016 + Hafner 2023 canonical recipe; verified by `test_ste_forward_value_is_one_hot_with_unimix` |
| Gumbel-Softmax categorical sampling | `gumbel_softmax_sample()` | HARD-EARNED-CANONICAL-1:1 | Same source citation chain; canonical reparametrization for discrete latents |
| 1% unimix on all categoricals | Wave 3 fix: `unimix_alpha=0.01` default; `apply_unimix_to_logits` helper | HARD-EARNED-CANONICAL-1:1 (post-Wave-3-fix; PRE-fix was CARGO-CULTED OMISSION) | Hafner 2023 §3 "Robustness" canonical robustness mechanism; prevents posterior collapse to one-hot; verified by `test_unimix_prevents_posterior_collapse_to_hard_one_hot` |
| G x K = 32 x 32 categorical | Parameterizable; default `G=24, K=256` C6 adaptation; Hafner config validated in tests | HARD-EARNED-DOCUMENTED-ADAPTATION (problem space) | The C6 24-dim baseline match is HARD-EARNED architectural inheritance (matches the substrate it replaces); K=256 = 1 byte = canonical int8 packing surface; both configs validated in `test_g_k_parameterization_supports_both_hafner_and_c6_canonical_configs` |
| RSSM = GRU + dense | NO GRU at L0 (per symposium decision); per-pair-independent latent | HARD-EARNED-DOCUMENTED-ADAPTATION (problem space) | Contest scorer is per-pair (600 independent pairs); temporal recurrence has no architectural target at L0; verified by `test_no_gru_at_l0_per_symposium_decision_canonical_unwind` |
| symlog observation squashing | Not present | HARD-EARNED-N/A (problem space) | Video frames are bound to [0, 255] uint8 natively; sigmoid * 255 RGB head produces same range; symlog structurally redundant |
| KL balancing + free bits | Not present | HARD-EARNED-N/A (problem space) | No separate prior network at L0 (per-pair learned logits ARE the posterior); will need re-evaluation when L1+ adds GRU + dynamics prior |
| Percentile return normalization | Not present | HARD-EARNED-N/A (problem space) | No RL reward signal; video compression uses contest scorer loss directly |
| symexp twohot loss for reward/critic | Not present | HARD-EARNED-N/A (problem space) | No value/critic heads; verified by `test_no_grl_critic_heads_at_l0_per_symposium_decision` |

## 2. 9-dimension success checklist evidence (Catalog #294)

1. **UNIQUENESS**: substrate-CLASS shift from HNeRV-family continuous-Gaussian C6 IBPS baseline to discrete categorical posterior per Hafner 2023 + vdOord VQ-VAE 2017 lineage. Class-shift not within-class.
2. **BEAUTY + ELEGANCE**: 27/27 tests pass in <1s; `apply_unimix_to_logits` is a 12-line canonical helper; canonical contract reviewable in 30 seconds.
3. **DISTINCTNESS**: explicitly different from sister substrates (V1 Faiss V8 = side-info channel; NSCS06 v8 = chroma residual). All 3 use discrete-posterior strategy at different surfaces.
4. **RIGOR**: pre-edit premise verification per Catalog #229 + WebSearch on Hafner 2023 + assumption-adversary HARD-EARNED-vs-CARGO-CULTED classification + sister symposium memo cross-references.
5. **OPTIMIZATION PER TECHNIQUE**: 1% unimix is the OPTIMAL ENGINEERING for the categorical posterior per Hafner 2023 §3; the prior omission was suppression by reflex; restored via Wave 3 audit.
6. **STACK-OF-STACKS COMPOSABILITY**: composes with V1 Faiss V8 (categorical-over-SegNet histograms) + NSCS06 v8 (chroma residual entropy bottleneck) per DD aggregate predicted band [0.187, 0.205].
7. **DETERMINISTIC REPRODUCIBILITY**: archive grammar is byte-deterministic per existing `test_archive_round_trip_byte_determinism`; unimix is deterministic for fixed key per `test_unimix_mixture_distribution_first_empirical_anchor`.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: MLX-local at $0 cost per Catalog #192; tests run in 0.69s on M5 Max.
9. **OPTIMAL MINIMAL CONTEST SCORE**: empirical contest score deferred to Path B2 PyTorch port + Modal smoke per symposium op-routable #3; this Wave 3 audit lands the structural foundation (canonical Hafner 2023 fidelity) that enables a credible Modal smoke.

## 3. Observability surface (Catalog #305)

1. **Inspectable per layer**: `architecture_manifest()` surfaces `unimix_alpha` (Wave 3 addition) + every config field.
2. **Decomposable per signal**: per-group categorical logits inspectable; mixture probabilities reconstructable via `apply_unimix_to_logits`.
3. **Diff-able across runs**: archive pack/parse round-trip is byte-deterministic; `unimix_alpha` is a config-time constant (no run-time drift).
4. **Queryable post-hoc**: canonical equation residual queryable via `tac.canonical_equations.query_equations()`.
5. **Cite-able**: every Hafner 2023 element carries explicit source citation in module docstrings; canonical equation #344 anchored to this audit's commit + provenance.
6. **Counterfactual-able**: `unimix_alpha=0.0` ablation surface exists for sensitivity studies per `test_unimix_ablation_alpha_zero_produces_valid_output`.

## 4. Sextet pact deliberation summary

The T2 deliberation reached PROCEED_WITH_REVISIONS with the Wave 3 fix
landed; the revisions are the Path B2 PyTorch port + Modal smoke per
symposium op-routable #3 (sister wave; not blocking this audit's structural
landing). Per CLAUDE.md "Forbidden premature KILL": the prior CARGO-CULTED
unimix omission is IMPLEMENTATION-LEVEL per Catalog #307 (not paradigm-level
falsification); the substrate paradigm remains INTACT.

## 5. Per-substrate reactivation criteria

The substrate is now at L0+canonical-Hafner-2023-fidelity. Reactivation
paths from this state to L1+:

1. **L1 — Path B2 PyTorch port**: queued per symposium op-routable #3; reuses
   MLX module's state_dict via canonical export bridge; lands the sister
   empirical anchor with trained logits (this audit's anchor is the
   closed-form math identity).
2. **L1 — Probe disambiguator**: `tools/probe_dreamer_v3_rssm_g_k_sweep_disambiguator.py`
   per symposium op-routable #1a; sweeps (G, K) ∈ {(8, 16), (16, 32), (24, 256), (32, 32)}
   to triangulate which axis dominates the contest's PoseNet+SegNet feedback.
3. **L1 — GRU + dynamics prior extension**: full RSSM per Hafner 2023 canonical;
   requires re-evaluation of KL balancing + free bits classifications per
   Catalog #303.
4. **L1 — Modal smoke**: paired CPU + CUDA smoke per Catalog #246; first paid
   GPU dispatch gated on Path B2 PyTorch port landing.

## 6. Catalog #324 post-training Tier-C validation discipline

The substrate's `predicted_band_validation_status` remains
`pending_post_training` per the original symposium memo's Catalog #324
classification; this Wave 3 audit's empirical anchor is the closed-form
mixture identity, NOT a post-training Tier-C density measurement. The
sister Path B2 trainer + Modal smoke lands the post-training anchor.
