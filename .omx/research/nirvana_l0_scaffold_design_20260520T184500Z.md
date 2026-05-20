---
council_tier: T1
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "Adaptive scheduler NOT in L0 scaffold; only the patch grid. Phase 2 must wire scheduler."
council_assumption_adversary_verdict:
  - assumption: "patch-wise modeling improves PSNR on driving video"
    classification: HARD-EARNED
    rationale: "Maiya CVPR 2024 + patch-CNN literature; per-patch specialization well-validated."
  - assumption: "PATCH_GRID 4x4 = 16 patches is the right default"
    classification: CARGO-CULTED
    rationale: "L0 sanity; 2x2 / 8x8 alternatives need empirical sweep."
  - assumption: "shared per-patch decoder weights"
    classification: CARGO-CULTED
    rationale: "rate-saving cheap variant; per-patch independent decoders inflate ~16x."
council_decisions_recorded:
  - "L0 SCAFFOLD: substrate package + trainer + recipe + driver + tests; adaptive scheduler deferred to L1."
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
horizon_class: frontier_pursuit
---

# NIRVANA L0 SCAFFOLD design memo (WAVE-3-NERV-LITERATURE-L0-RESCOPED 2026-05-20)

## Canonical-vs-unique decision per layer

- Architecture (per-patch decoder + spatial assembly): **FORK_BECAUSE_PRINCIPLED_MISMATCH** for the patch-wise decoder + stitching; ADOPT_CANONICAL for DepthSep blocks.
- Archive grammar (NRV1): **FORK_BECAUSE_PRINCIPLED_MISMATCH** — distinctive 25-byte header with PATCH_GRID_H/W + PATCH_EMBED_DIM fields; cannot be shared.
- Score-aware loss: **ADOPT_CANONICAL** (routes through `tac.substrates.score_aware_common.score_pair_components_dispatch`).
- Inflate runtime: ADOPT_CANONICAL pattern + FORK on patch-decode + stitch path.
- Trainer skeleton: ADOPT_CANONICAL.
- SubstrateContract META layer: ADOPT_CANONICAL per Catalog #241/#242.
- Adaptive per-patch scheduler: **DEFERRED to L1+** (NIRVANA distinctive contribution; training-loop mechanism only).

## 9-dimension success checklist evidence

1. **UNIQUENESS**: patch-wise specialization is paradigm-orthogonal to global NeRV substrates per operator's MODERATE-HIGH FIT ⭐⭐⭐⭐ verdict.
2. **BEAUTY + ELEGANCE**: ~750 LOC total substrate package; each file reviewable in 30s per L12.
3. **DISTINCTNESS**: distinctive NRV1 magic + PATCH_GRID_H/W + PATCH_EMBED_DIM header fields; explicit per-patch decode + stitch in architecture.
4. **RIGOR**: 11 tests pass (Catalog #91 ENCODE_INFLATE_ROUNDTRIP + Catalog #139 byte-mutation no_op_proof + L5 RGB output + patch-grid invariants); _full_main raises NotImplementedError per Catalog #240.
5. **OPTIMIZATION PER TECHNIQUE**: per Catalog #290 canonical-vs-unique decision per layer (above).
6. **STACK-OF-STACKS-COMPOSABILITY**: patch-wise paradigm stacks orthogonally with global NeRV substrates per operator's MODERATE-HIGH FIT verdict.
7. **DETERMINISTIC REPRODUCIBILITY**: byte-stable archive grammar; SIREN init seed-pinned.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: target ~180K params with SHARED per-patch decoder; rate term tight via int16 latents + brotli.
9. **OPTIMAL MINIMAL CONTEST SCORE**: predicted_band unknown at L0; pending Phase 2 council per Catalog #325 + #324.

## Cargo-cult audit per assumption

| Assumption | Classification | Rationale | Unwind path |
|---|---|---|---|
| patch-wise modeling improves PSNR | HARD-EARNED | Maiya CVPR 2024 + patch-CNN literature | none required |
| PATCH_GRID_H=PATCH_GRID_W=4 (16 patches) | CARGO-CULTED | L0 sanity | empirical sweep at L1: 2x2 / 4x4 / 8x8 |
| shared per-patch decoder weights | CARGO-CULTED | rate-saving cheap variant | per-patch independent decoders at L1 if needed |
| bilinear spatial assembly | HARD-EARNED | standard for patch-based codecs | none required |
| DepthSep per-patch decoder | HARD-EARNED | mirrors ds_nerv canonical sister | none required |
| adaptive scheduler in training only | HARD-EARNED | inflate-time runtime is unaware of training dynamics | L0 omission is honest; L1+ wires the scheduler |
| 5 upsample blocks per patch decoder | CARGO-CULTED | chosen to hit reasonable patch resolution | empirical sweep at L1 |

## Observability surface

Per Catalog #305 6-facet definition:

1. **Inspectable per layer**: per-patch output shape + magnitude inspectable via PyTorch forward hooks on `_DsUpBlock`. Patch-embedding values inspectable via `model.patch_embeddings`.
2. **Decomposable per signal**: `NirvanaScoreAwareLoss.forward` returns `parts` dict with `rate_term` / `seg_term` / `pose_term` / `loss_total`.
3. **Diff-able across runs**: archive grammar is byte-stable; two runs with the same seed produce identical archives.
4. **Queryable post-hoc**: trainer writes `provenance.json` with n_params + patch_grid_h/w + num_patches + adaptive_scheduler_active + literature_anchor + fit_ranking.
5. **Cite-able**: every persisted artifact carries `(substrate_tag, lane_id, git_head, dispatch_instance_job_id)`.
6. **Counterfactual-able**: Catalog #139 byte-mutation smoke proves output changes when latent[0,0] perturbed; per-patch decode independence verifiable by mutating individual patch_embedding rows.

## Predicted ΔS band

`pending_post_training` per Catalog #324.

<!-- PREDICTED_BAND_VIBES_OK:l0_scaffold_no_dispatch_eligible_pending_phase_2_council_symposium_per_catalog_325 -->

## Reactivation criteria

1. Per-substrate adversarial grand council symposium per Catalog #325.
2. Cargo-cult audit empirically validates the 4 CARGO-CULTED choices via sweep.
3. Adaptive per-patch scheduler (the NIRVANA distinctive contribution) lands in trainer loop with per-patch loss EMA + proportional sampling.
4. Trainer _full_main path replaces NotImplementedError with real score-aware Lagrangian per Catalog #226.
5. Recipe research_only flips to false; dispatch_enabled flips to true; predicted_band declared per Catalog #324.

## 6-hook wire-in declaration (per Catalog #125)

- hook #1 sensitivity-map: N/A (L0)
- hook #2 Pareto constraint: rate_distortion_v1
- hook #3 bit-allocator: N/A — fp16 brotli on shared per-patch decoder + patch embeddings
- hook #4 cathedral autopilot dispatch: N/A (research_only)
- hook #5 continual-learning posterior: N/A
- hook #6 probe-disambiguator: N/A — single mechanism (patch-wise + shared decoder); adaptive scheduler alternative is L1+

## Cross-references

- Literature: Maiya et al. CVPR 2024 "NIRVANA: Neural Implicit Representations of Videos with Adaptive Networks and Autoregressive patch-wise Modeling" (arXiv:2212.14593)
- Sister memos: `.omx/research/grand_council_fields_medal_substrate_design_20260512.md`
- Sister substrate package: `src/tac/substrates/ds_nerv/` (canonical NeRV-family L1 reference)
- Sister L0 SCAFFOLD: `.omx/research/ego_nerv_l0_scaffold_design_20260520T140000Z.md`
- Operator 5-tier fit ranking source: task #1090 description
