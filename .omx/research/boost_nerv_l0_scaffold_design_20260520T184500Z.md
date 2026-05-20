---
council_tier: T1
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "L0 SCAFFOLD only. Full path council-gated until cargo-cult choices empirically validated."
council_assumption_adversary_verdict:
  - assumption: "boosting paradigm reduces worst-case error on driving video"
    classification: HARD-EARNED
    rationale: "Liu ECCV 2024 + general gradient-boosting literature; well-validated."
  - assumption: "num_boosting_rounds=2 is the right default"
    classification: CARGO-CULTED
    rationale: "chosen for L0 sanity; needs empirical sweep at L1+."
  - assumption: "shared latent across rounds is sufficient"
    classification: CARGO-CULTED
    rationale: "cheap variant; per-round latents would inflate rate ~1.5x."
council_decisions_recorded:
  - "L0 SCAFFOLD: substrate package + trainer + recipe + driver + tests; full path council-gated per Catalog #240."
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
horizon_class: frontier_pursuit
---

# BoostNeRV L0 SCAFFOLD design memo (WAVE-3-NERV-LITERATURE-L0-RESCOPED 2026-05-20)

## Canonical-vs-unique decision per layer

- Architecture (DepthSep base + boosting heads): **FORK_BECAUSE_PRINCIPLED_MISMATCH** for the boosting heads (new canonical primitive); ADOPT_CANONICAL for the DepthSep base (mirrors ds_nerv).
- Archive grammar (BSV1): **FORK_BECAUSE_PRINCIPLED_MISMATCH** — distinctive 22-byte header with NUM_BOOSTING_ROUNDS u8 field; cannot be shared with sister DSV1/NRV1/CPP1.
- Score-aware loss: **ADOPT_CANONICAL** (routes through `tac.substrates.score_aware_common.score_pair_components_dispatch`).
- Inflate runtime: ADOPT_CANONICAL pattern (parse_archive + per-pair render); FORK on the boosting-chain forward loop.
- Trainer skeleton: ADOPT_CANONICAL (routes through `tac.substrates._shared.trainer_skeleton.{device_or_die, pin_seeds, sha256_bytes, detect_hardware_substrate, ...}`).
- SubstrateContract META layer: ADOPT_CANONICAL per Catalog #241/#242.

## 9-dimension success checklist evidence

1. **UNIQUENESS**: iterative residual-refinement chain is paradigm-orthogonal to existing NeRV variants per operator's HIGH FIT ⭐⭐⭐⭐⭐ verdict; composes WITH any base substrate at L1+.
2. **BEAUTY + ELEGANCE**: ~700 LOC total substrate package; each file reviewable in 30s per L12.
3. **DISTINCTNESS**: distinctive BSV1 magic + NUM_BOOSTING_ROUNDS u8 header field; explicit boosting-head loop in architecture.
4. **RIGOR**: 11 tests pass (Catalog #91 ENCODE_INFLATE_ROUNDTRIP + Catalog #139 byte-mutation no_op_proof + L5 RGB output + boosting-clamp invariant); _full_main raises NotImplementedError per Catalog #240.
5. **OPTIMIZATION PER TECHNIQUE**: per Catalog #290 canonical-vs-unique decision per layer (above).
6. **STACK-OF-STACKS-COMPOSABILITY**: boosting paradigm composes with any base substrate at L1+ (TCNeRV / DSNeRV / sane_hnerv / etc.) per operator's HIGH FIT verdict.
7. **DETERMINISTIC REPRODUCIBILITY**: byte-stable archive grammar; SIREN init pinned via seed; canonical pin_seeds.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: target ~170K params including boosting; rate term tight via int16 latents + brotli decoder blob.
9. **OPTIMAL MINIMAL CONTEST SCORE**: predicted_band unknown at L0; pending Phase 2 council symposium per Catalog #325 + #324 validation discipline.

## Cargo-cult audit per assumption

| Assumption | Classification | Rationale | Unwind path |
|---|---|---|---|
| boosting paradigm reduces worst-case error on driving video | HARD-EARNED | Liu ECCV 2024 + gradient-boosting literature | none required |
| num_boosting_rounds=2 default | CARGO-CULTED | L0 sanity choice | empirical sweep at L1: 1/2/3/4 rounds |
| boosting_gain_clamp=[-0.1, 0.1] | CARGO-CULTED | chosen to prevent runaway residuals | per-substrate empirical tuning |
| shared latent z across rounds | CARGO-CULTED | cheap variant | per-round latents (rate +50%) at L1 if needed |
| shared boosting head across frame_0/frame_1 | CARGO-CULTED | keeps rate term tight | per-frame heads (head count 2x) at L1 if needed |
| DepthSep base decoder | HARD-EARNED | mirrors ds_nerv canonical sister | none required |
| FP16 brotli weights + int16 latents | HARD-EARNED | canonical NeRV-family export | none required |

## Observability surface

Per Catalog #305 6-facet definition:

1. **Inspectable per layer**: per-boosting-round residual magnitude can be logged via hook on `_BoostingHead.forward` return (clamp-input vs clamp-output). Per-block PixelShuffle output shape inspectable via standard PyTorch forward hooks.
2. **Decomposable per signal**: `BoostnervScoreAwareLoss.forward` returns `parts` dict with `rate_term` / `seg_term` / `pose_term` / `loss_total` as detached tensors per the score-domain Lagrangian.
3. **Diff-able across runs**: archive grammar is byte-stable (Catalog #139 byte-mutation smoke proves this); two runs with the same seed produce identical archives.
4. **Queryable post-hoc**: trainer writes `provenance.json` with n_params + num_boosting_rounds + literature_anchor + fit_ranking.
5. **Cite-able**: every persisted artifact carries `(substrate_tag, lane_id, git_head, dispatch_instance_job_id)` per Catalog #245 modal_call_id_ledger pattern.
6. **Counterfactual-able**: Catalog #139 byte-mutation smoke proves frame output changes when latent[0,0] is perturbed; boosting-round residual clamp bounded by `boosting_gain_clamp` per dedicated test.

## Predicted ΔS band

`pending_post_training` per Catalog #324. No empirical anchor at L0. Pre-training Tier-C density prediction is CARGO-CULTED per the sister C6 IBPS 22x miss anchor 2026-05-17.

<!-- PREDICTED_BAND_VIBES_OK:l0_scaffold_no_dispatch_eligible_pending_phase_2_council_symposium_per_catalog_325 -->

## Reactivation criteria

1. Per-substrate adversarial grand council symposium per CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" (Catalog #325) returns PROCEED or PROCEED_WITH_REVISIONS verdict.
2. Cargo-cult audit (above) empirically validates the 5 CARGO-CULTED choices via sweep OR reclassifies as HARD-EARNED with citation.
3. Trainer _full_main path replaces NotImplementedError with real score-aware Lagrangian + EMA + canonical auth-eval helper invocation per Catalog #226.
4. Recipe research_only flips to false; dispatch_enabled flips to true; predicted_band declared per Catalog #324.

## 6-hook wire-in declaration (per Catalog #125)

- hook #1 sensitivity-map: N/A — L0 SCAFFOLD, no sensitivity signal until Phase 2 + full path
- hook #2 Pareto constraint: rate_distortion_v1 (declared in SubstrateContract)
- hook #3 bit-allocator: N/A — fp16 brotli on combined base+boosting weight blob; per-tensor allocator is L1+
- hook #4 cathedral autopilot dispatch: N/A — research_only; not dispatch-eligible
- hook #5 continual-learning posterior: N/A — no anchor until [contest-CUDA] measurement
- hook #6 probe-disambiguator: N/A — single mechanism (iterative residual chain); per-round-latent vs shared-latent alternative is L1+

## Cross-references

- Literature: Liu et al. ECCV 2024 "BoostNeRV: Iterative Refinement for Implicit Neural Video Representations"
- Sister memos: `.omx/research/grand_council_fields_medal_substrate_design_20260512.md` (ds_nerv design memo)
- Sister substrate package: `src/tac/substrates/ds_nerv/` (canonical NeRV-family L1 reference)
- Sister L0 SCAFFOLD: `experiments/train_substrate_ego_nerv.py` + `.omx/research/ego_nerv_l0_scaffold_design_20260520T140000Z.md`
- Operator 5-tier fit ranking source: task #1090 description
