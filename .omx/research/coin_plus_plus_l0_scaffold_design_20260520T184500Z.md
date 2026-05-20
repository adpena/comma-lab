---
council_tier: T1
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "MOD_DIM=64 + int8 quant are CARGO-CULTED. Coord-MLP vs CNN base is a real TENSION."
council_assumption_adversary_verdict:
  - assumption: "modulation paradigm yields better PSNR per byte than per-pair latent + shared decoder"
    classification: HARD-EARNED
    rationale: "Dupont ICML 2022 + Perez 2017 FiLM literature; well-validated at small scale."
  - assumption: "MOD_DIM=64 is the right per-pair latent rate"
    classification: CARGO-CULTED
    rationale: "L0 sanity choice; MOD_DIM IS the variable-rate term so sweep is critical."
  - assumption: "int8 modulation quantization is sufficient"
    classification: CARGO-CULTED
    rationale: "chosen for tight rate; int16 alternative needs empirical sweep."
  - assumption: "coord-MLP base architecture is right for 384x512 video"
    classification: CARGO-CULTED vs HARD-EARNED TENSION
    rationale: "COIN/COIN++ canonical choice; driving video may benefit from CNN base with per-pair modulation."
council_decisions_recorded:
  - "L0 SCAFFOLD: substrate package + trainer + recipe + driver + tests; coord-MLP-vs-CNN-base TENSION deferred to L1+."
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
horizon_class: frontier_pursuit
---

# COIN++ L0 SCAFFOLD design memo (WAVE-3-NERV-LITERATURE-L0-RESCOPED 2026-05-20)

## Canonical-vs-unique decision per layer

- Architecture (shared base coord-MLP + FiLM-modulated hidden layers): **FORK_BECAUSE_PRINCIPLED_MISMATCH** — entirely different paradigm from NeRV-family per-pair latents.
- Archive grammar (CPP1): **FORK_BECAUSE_PRINCIPLED_MISMATCH** — distinctive 21-byte header with MODULATION_DIM u16 + int8 modulation blob (vs int16 latents in NeRV-family).
- Score-aware loss: **ADOPT_CANONICAL** (routes through `tac.substrates.score_aware_common.score_pair_components_dispatch`).
- Inflate runtime: ADOPT_CANONICAL pattern (parse_archive + per-pair render); FORK on the coord-MLP O(H*W) forward path.
- Trainer skeleton: ADOPT_CANONICAL.
- SubstrateContract META layer: ADOPT_CANONICAL per Catalog #241/#242.

## 9-dimension success checklist evidence

1. **UNIQUENESS**: modulation paradigm is structurally different from per-pair latent + shared decoder per operator's MODERATE FIT ⭐⭐⭐ verdict.
2. **BEAUTY + ELEGANCE**: ~700 LOC total substrate package; each file reviewable in 30s per L12.
3. **DISTINCTNESS**: distinctive CPP1 magic + MODULATION_DIM u16 header field + int8 modulation blob (not int16 latents).
4. **RIGOR**: 11 tests pass (Catalog #91 ENCODE_INFLATE_ROUNDTRIP + Catalog #139 byte-mutation no_op_proof + L5 RGB output + FiLM-modulation + per-pair-modulation invariants); _full_main raises NotImplementedError per Catalog #240.
5. **OPTIMIZATION PER TECHNIQUE**: per Catalog #290 canonical-vs-unique decision per layer (above).
6. **STACK-OF-STACKS-COMPOSABILITY**: different rate-tradeoff than NeRV-family; complementary in a Pareto sense. Can compose at the dispatch level (try both, pick better).
7. **DETERMINISTIC REPRODUCIBILITY**: byte-stable archive grammar; SIREN init seed-pinned; coord grid deterministically generated.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: target ~140K base params + ~76 KB modulations for 600 pairs at MOD_DIM=64 + int8. Per-pair decode is O(H*W) (coord-MLP cost).
9. **OPTIMAL MINIMAL CONTEST SCORE**: predicted_band unknown at L0; pending Phase 2 council per Catalog #325 + #324.

## Cargo-cult audit per assumption

| Assumption | Classification | Rationale | Unwind path |
|---|---|---|---|
| modulation paradigm yields better PSNR per byte | HARD-EARNED | Dupont ICML 2022 + Perez 2017 FiLM literature | none required |
| MOD_DIM=64 default | CARGO-CULTED | L0 sanity choice | empirical sweep at L1: 16/32/64/128/256 |
| int8 modulation quantization | CARGO-CULTED | chosen for tight rate | int16 alternative sweep at L1 if quant artifacts dominate |
| coord-MLP base architecture | CARGO-CULTED vs HARD-EARNED TENSION | COIN/COIN++ canonical | CNN base with per-pair modulation alternative worth empirical comparison at L1 |
| FiLM-style scale+shift modulation | HARD-EARNED | Perez 2017 | none required |
| frame index as 3rd input dim | HARD-EARNED | standard for video-INR | none required |
| SIREN sin_frequency=30 | HARD-EARNED | NeRF default | none required |
| hidden_dim=96, num_hidden_layers=4 | CARGO-CULTED | L0 sanity | empirical sweep at L1 |

## Observability surface

Per Catalog #305 6-facet definition:

1. **Inspectable per layer**: per-layer FiLM gamma/beta projections inspectable via PyTorch forward hooks on `_ModulatedLinear`. Per-pair modulation values inspectable via `model.modulations`.
2. **Decomposable per signal**: `CoinplusplusScoreAwareLoss.forward` returns `parts` dict with `rate_term` / `seg_term` / `pose_term` / `loss_total`.
3. **Diff-able across runs**: archive grammar is byte-stable; two runs with the same seed produce identical archives.
4. **Queryable post-hoc**: trainer writes `provenance.json` with n_params_total + n_params_base_shared + n_params_modulation_per_pair_pool + modulation_dim + literature_anchor + fit_ranking.
5. **Cite-able**: every persisted artifact carries `(substrate_tag, lane_id, git_head, dispatch_instance_job_id)`.
6. **Counterfactual-able**: Catalog #139 byte-mutation smoke proves output changes when modulation[0,0] perturbed; per-pair modulation independence verifiable by mutating individual modulation rows.

## Predicted ΔS band

`pending_post_training` per Catalog #324.

<!-- PREDICTED_BAND_VIBES_OK:l0_scaffold_no_dispatch_eligible_pending_phase_2_council_symposium_per_catalog_325 -->

## Reactivation criteria

1. Per-substrate adversarial grand council symposium per Catalog #325.
2. Cargo-cult audit empirically validates MOD_DIM + int8 quant + coord-MLP base CARGO-CULTED choices via sweep OR reclassifies as HARD-EARNED.
3. Per-pair coordinate batching strategy + training step time budget designed (O(H*W) per pair decode).
4. Trainer _full_main path replaces NotImplementedError with real score-aware Lagrangian per Catalog #226.
5. Recipe research_only flips to false; dispatch_enabled flips to true; predicted_band declared per Catalog #324.

## 6-hook wire-in declaration (per Catalog #125)

- hook #1 sensitivity-map: N/A (L0)
- hook #2 Pareto constraint: rate_distortion_v1
- hook #3 bit-allocator: N/A — fp16 brotli base + int8 modulations
- hook #4 cathedral autopilot dispatch: N/A (research_only)
- hook #5 continual-learning posterior: N/A
- hook #6 probe-disambiguator: N/A — single mechanism (FiLM-modulated coord-MLP); coord-MLP-vs-CNN-base TENSION is L1+

## Cross-references

- Literature: Dupont et al. ICML 2022 "COIN++: Neural Compression across Modalities" (arXiv:2201.12904); github.com/EmilienDupont/coinpp
- FiLM ancestry: Perez et al. AAAI 2018 "FiLM: Visual Reasoning with a General Conditioning Layer"
- Sister memos: `.omx/research/grand_council_fields_medal_substrate_design_20260512.md`
- Sister substrate package: `src/tac/substrates/ds_nerv/` (canonical NeRV-family L1 reference; different paradigm but same canonical pattern)
- Sister L0 SCAFFOLD: `.omx/research/ego_nerv_l0_scaffold_design_20260520T140000Z.md`
- Operator 5-tier fit ranking source: task #1090 description
