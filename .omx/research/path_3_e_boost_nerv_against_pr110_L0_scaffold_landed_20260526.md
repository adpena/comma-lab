<!-- SPDX-License-Identifier: MIT -->
<!-- Catalog #110 / #113 APPEND-ONLY HISTORICAL_PROVENANCE — landing memo; do not mutate. -->
<!-- Catalog #229 PV: this landing memo verifies premises empirically: 25/25 tests pass via .venv/bin/python -m pytest src/tac/substrates/boost_nerv_pr110_residual/tests/test_l0_scaffold.py -x -q (run 2026-05-26T07:14Z; output captured below). NO bulk-edit claims. -->
---
council_tier: T1
council_attendees: [Shannon, PR95Author, Time-Traveler]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "L0 SCAFFOLD scope is correct: design memo + MLX scaffold + tests; no MLX trainer; no archive build; no dispatch"
    classification: HARD-EARNED
    rationale: "Operator directive 2026-05-26 'design the substrate and curriculum and then optimize the design the whole stack around it' is precisely the L0 SCAFFOLD posture. Phase 2 council symposium per Catalog #325 + MLX smoke per Stage 3 convergence target are the L1+ work."
council_decisions_recorded:
  - "Lane registered at L0/L1 (impl_complete=true + memory_entry=true)"
  - "Substrate package boost_nerv_pr110_residual/ landed with 25 passing tests"
  - "Design memo path_3_e_boost_nerv_against_pr110_substrate_design_20260526.md carries Catalog #290/#294/#296/#303/#305/#309/#324 sections"
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
horizon_class: frontier_pursuit
related_deliberation_ids:
  - path_3_e_boost_nerv_against_pr110_substrate_design_20260526
  - boost_nerv_l0_scaffold_design_20260520T184500Z
  - mlx_candidate_contest_equivalence_gate_landed_20260526
---

# Path 3 candidate #E: BoostNeRV against PR110 fec6 — L0 SCAFFOLD LANDED

**Lane**: `lane_path_3_e_boost_nerv_against_pr110_20260526` L1 (impl_complete + memory_entry)
**Cost**: $0 (design + MLX scaffold + tests; no paid GPU)
**Wall-clock**: ~3h
**Operator directive**: 2026-05-26 verbatim *"We should add boostnerv to the priority list too, maybe against PR110, because it seems like it could be free gains if done right."* + binding strategic reframing *"design the substrate and curriculum and then optimize the design the whole stack around it for extreme optimization and performance and optimal score lowering"*.

## Premise verification per Catalog #229 (empirical — NOT bulk-edit claim)

Verdict table (run 2026-05-26T07:14Z; reproducer: `.venv/bin/python -m pytest src/tac/substrates/boost_nerv_pr110_residual/tests/test_l0_scaffold.py -x -q`):

| test | verdict | notes |
|---|---|---|
| `test_module_imports_without_mlx` | PASS | top-level import succeeds without MLX |
| `test_module_exposes_canonical_public_api` | PASS | `__all__` narrow + explicit (Catalog #335 contract) |
| `test_config_dataclass_lazy_loads_via_getattr` | PASS | lazy MLX import via `__getattr__` |
| `test_curriculum_canonical_5_plus_1_stages_declared` | PASS | curriculum machine-readable at rest |
| `test_curriculum_total_wallclock_estimate_excludes_optional_by_default` | PASS | L0 wallclock budget = sum without round-2 |
| `test_full_main_stub_raises_per_catalog_240` | PASS | NotImplementedError refuses dispatch |
| `test_bpr1_header_pack_round_trip` | PASS | 29-byte deterministic header |
| `test_bpr1_header_invalid_magic_rejected` | PASS | mis-magic'd sidecar refused |
| `test_bpr1_header_invalid_version_rejected` | PASS | wrong version refused |
| `test_pack_rejects_bad_num_rounds` | PASS | num_rounds ∈ [1, 4] enforced |
| `test_pack_rejects_bad_sha_prefix_length` | PASS | sha_prefix must be 16 bytes |
| `test_compose_archive_binds_to_pr110_sha256` | PASS | sha256 prefix auto-computed from PR110 bytes |
| `test_split_composed_archive_round_trip` | PASS | Catalog #91 ENCODE_INFLATE_ROUNDTRIP |
| `test_split_refuses_non_matching_pr110_base` | PASS | structural-extinction primitive: mutated base refused |
| `test_byte_mutation_no_op_proof_per_catalog_139` | PASS | Catalog #139 byte-mutation discipline |
| `test_write_composed_archive_to_zip_deterministic` | PASS | Catalog #19 deterministic ZIP |
| `test_residual_extraction_plan_refuses_tmp_paths` | PASS | Catalog #113 forbidden-/tmp/-paths |
| `test_residual_extraction_plan_refuses_missing_files` | PASS | Catalog #229 PV at construction time |
| `test_diagnose_residual_target_magnitude_verdicts` | PASS | Stage 1 verdict taxonomy works |
| `test_diagnose_rejects_shape_mismatch` | PASS | shape validation |
| `test_write_diagnostic_manifest_rejects_score_claim` | PASS | Catalog #127 + #341 non-promotable markers |
| `test_num_residual_parameters_matches_paper_calculation` | PASS | per-round = 1971 params @ default config |
| `test_num_residual_parameters_scales_with_rounds` | PASS | 3 rounds = 3 × 1971 params |
| `test_residual_head_mlx_forward_shape` | PASS | MLX NHWC forward (B, H, W, 3) |
| `test_compose_pr110_base_plus_residual_clamps_correctly` | PASS | composition clamps to [0, 1] correctly |

**25/25 PASS.** Empirical anchor: this is the L0 SCAFFOLD verdict, NOT a contest score claim per Catalog #127/#192/#317/#341. All artifacts carry `[macOS-MLX research-signal]` axis tag and `score_claim=false` / `promotion_eligible=false` / `ready_for_exact_eval_dispatch=false` per CLAUDE.md "MLX portable-local-substrate authority".

## (1) Catalog #290 per-layer canonical-vs-unique decisions

Per design memo §"Canonical-vs-unique decision per layer" (10 layers, 6 FORK + 3 ADOPT_CANONICAL + 1 N/A at L0):

- **FORK_BECAUSE_PRINCIPLED_MISMATCH** (6/10): base learner (PR110 frozen + external); residual codec (boosting paradigm); archive grammar (BPR1 with sha256 binding); inflate runtime (2-stage); MLX trainer (Catalog #1265 cascade); tests (2-stage inflate parity is a NEW canonical pattern)
- **ADOPT_CANONICAL** (3/10): score-aware loss (routes through Catalog #164 scorer-loss helper); eval_roundtrip + EMA 0.997 (NON-NEGOTIABLE); canonical Provenance umbrella (Catalog #323)
- **N/A at L0; ADOPT_CANONICAL at L1+** (1/10): Tier-1 PyTorch engineering primitives (used only at MLX → PyTorch export bridge; not at MLX training)

## (2) Which canonical helpers we FORKED and why (substrate-optimal not convenience)

1. **Base learner**: FORKED because the operator's "against PR110" framing structurally REQUIRES inheriting PR110's 0.193 [contest-CPU] frontier, NOT retraining a base. ADOPT_CANONICAL ("train your own DepthSep base") would throw away the contest-grade signal. Substrate-optimal.
2. **Archive grammar**: FORKED because the BPR1 magic + 16-byte PR110_BASE_SHA256_PREFIX binding is the structural-extinction primitive that prevents the residual sidecar from being silently mis-applied to a non-PR110 base archive. No canonical NeRV-family pattern supports this binding (sister `boost_nerv/`'s BSV1 doesn't bind to an external base). Substrate-optimal per CLAUDE.md "Bit-level deconstruction and entropy discipline".
3. **Inflate runtime**: FORKED because the 2-stage inflate path (subprocess-invoke PR110 inflate.sh → load frames → apply residual) is the only way to preserve PR110's archive bytes unchanged. Alternative ADOPT_CANONICAL would vendor PR110's HNeRV decoder weights INTO our archive, defeating the entire "free gains on top of PR110" premise. Substrate-optimal.
4. **MLX trainer**: FORKED because the existing canonical training stack is PyTorch-on-CUDA; per binding 2026-05-26 reframing + Catalog #1265 MLX↔PyTorch parity anchor (0.000011 contest-units, 72× margin), MLX is the dev-velocity-optimal local-training path that closes the cargo-cult-unwind loop without paid GPU. Substrate-optimal per the binding strategic correction.
5. **Tests**: FORKED the 2-stage inflate parity test pattern (NEW canonical pattern this substrate introduces); ADOPT_CANONICAL the Catalog #91 ENCODE_INFLATE_ROUNDTRIP + Catalog #139 byte-mutation no_op_proof patterns. Substrate-optimal.
6. **Residual codec**: FORKED because "iterate over frozen-base residuals" has no existing canonical bolt-on pattern. The MLP architecture mirrors sister `boost_nerv/_BoostingHead` but the conditioning input differs (PR110 latent, not local renderer z). Substrate-optimal.

## (3) Predicted ΔS band with Dykstra-feasibility check / Shannon R(D) bound

**Predicted band**: `pending_post_training` per Catalog #324 (refuses phantom_random_init predictions per sister C6 IBPS 22× miss anchor 2026-05-17).

**First-principles Shannon R(D) framing** (paper calculation, NOT runtime anchor):
- Rate cost: `Δrate = +0.00546 contest-units` (8192 bytes sidecar at 25/37545489 contest formula)
- Predicted ΔS range: `[-0.010, +0.0045]` contest-units (sign-ambiguous at L0)
  - Lower bound (-0.010): residual learner extracts 50%+ of available scorer-conditional entropy
  - Upper bound (+0.0045): residual carries near-random vs scorer signal (REGRESSION case)
- Dykstra-feasibility verdict: feasible at 16 KB sidecar (would clearly admit non-trivial residual signal); MARGINALLY feasible at 8 KB (the L0 budget); infeasible at 4 KB without sparse encoding

**Phase 2 council symposium per Catalog #325 MUST run formal Dykstra-feasibility intersection check + post-training Tier-C density measurement on the actual trained residual learner BEFORE any paid CUDA dispatch.** Sign-ambiguity at L0 is precisely why L0 SCAFFOLD posture is correct.

## (4) MLX-trainable curriculum stages with smoke convergence targets

Per `boosting_curriculum.py::CANONICAL_CURRICULUM` (6 stages: 5 canonical + 1 optional L1+):

| stage | name | MLX | convergence target | wallclock estimate |
|---|---|---|---|---|
| 0 | pr110_base_extraction | NO (subprocess) | 600 pair-RGB frames cached + sha256 prefix matches | 30 s |
| 1 | residual_target_computation | NO (numpy) | p99 ≥ 0.05 → PROCEED to Stage 2; < 0.01 → DEFER | 10 s |
| 2 | mlx_residual_learner_warmup | YES (Adam lr=1e-3) | training_loss_reduction ≥ 50% in 10 epochs | ~2 min |
| 3 | mlx_score_aware_finetune | YES (Lagrangian + EMA + eval_roundtrip) | MLX scorer proxy ≥ -0.001 contest-units improvement vs PR110-alone baseline | ~15 min |
| 4 | archive_build_plus_catalog_1265_gate | NO (subprocess + #1265 gate) | Catalog #1265 verdict=PASS at threshold 0.001 (MANDATORY before paid CUDA) | ~3 min |
| 5 | [L1+] optional_round_2_boosting | YES (deferred) | Same as Stage 3+4 applied to round-2 residual | ~20 min |

**Total L0 wallclock estimate (without optional round 2)**: ~1240 s = ~21 min on M-series MLX. **Total with round 2**: ~2440 s = ~41 min.

## (5) Residual extraction from PR110 — byte / score math

**Byte accounting** (predicted):
- PR110 base archive: 178417 bytes (per `.omx/research/pr110_final_evidence_pack_20260520T141144Z_codex/archive_metadata.json`)
- Per-pair residual at (96×128) downsampled int8: rough estimate ~8800 brotli-q9-compressed bytes (near-zero-distribution residual highly compressible)
- BPR1 header overhead: 29 bytes
- **Composed archive bytes: 178417 + 29 + ~8800 = ~187246 bytes**
- **Δrate contest-units**: `25 × 8829 / 37545489 = +0.00588 contest-units`

**Score accounting** (predicted, paper calculation):
- PR110 baseline score: 0.193 [contest-CPU] (frontier band)
- Three predicted scenarios (Phase 2 council symposium per Catalog #325 must disambiguate empirically):
  - Optimistic (residual extracts 80% of available scorer-conditional entropy): net ΔS ≈ -0.009 → composed score ≈ 0.184
  - Realistic (50%): net ΔS ≈ -0.006 → composed score ≈ 0.187
  - Pessimistic (10% — noisy residual): net ΔS ≈ +0.004 → composed score ≈ 0.197 (REGRESSION)

**The sign-ambiguous predicted band [-0.010, +0.004] is precisely why the operator-routable next step is Stage 3 MLX smoke** — the [macOS-MLX research-signal] verdict reduces the uncertainty band by 10-100× at $0 cost BEFORE any paid CUDA dispatch.

## (6) Operator-routable next steps

1. **OP-1 (Phase 2 council symposium per Catalog #325)**: convene sextet (Shannon LEAD + Dykstra CO-LEAD + Yousfi + Fridrich + Contrarian + Assumption-Adversary) + grand-council topical specialists (Atick-Redlich + Friedman + Liu-ECCV-2024-BoostNeRV-author) on the design memo. Required outputs: Dykstra-feasibility intersection check, cargo-cult audit verdict (8 CARGO-CULTED assumptions classified), 9-dim checklist evidence, Tier-C validation discipline declaration, reactivation criteria pinned.
2. **OP-2 (Shannon R(D) bound derivation)**: formally derive the achievable ΔS lower bound from `H(GT | PR110_base)` measured via `tools/mdl_scorer_conditional_ablation.py --tier c` on the PR110 frontier archive. Replaces the paper-calculation estimate with a first-principles bound.
3. **OP-3 (Empirical 100ep MLX smoke convergence verdict)**: implement Stages 0-3 of the canonical curriculum; run on `upstream/videos/0.mkv` for 50-100 epochs; produce a `[macOS-MLX research-signal]` anchor per Catalog #341 Tier A that the cathedral autopilot ranker can consume per Catalog #336 invocation. Convergence verdict disambiguates the sign-ambiguous predicted band BEFORE any paid CUDA dispatch.
4. **OP-4 (Catalog #1265 gate dry-run)**: once a composed archive exists (post-Stage 4), invoke `tools/gate_mlx_candidate_contest_equivalence.py --archive-zip <composed_archive> --candidate-label boost_nerv_pr110_residual_v0 --gate-threshold-contest-units 0.001`. PASS → operator-routable to paired contest-CPU + contest-CUDA dispatch. FAIL → do NOT dispatch; audit per #1251 + #1257 + #1258.

## 6-hook wire-in declaration (per Catalog #125)

Per design memo §"6-hook wire-in declaration":
- **hook #1 sensitivity-map**: ACTIVE at L1+ (per-pair residual magnitude + per-pair Δscore-decomposition); N/A at L0 (no measurement yet)
- **hook #2 Pareto constraint**: `rate_distortion_v1` (the rate-vs-distortion tradeoff is the substrate's defining axis)
- **hook #3 bit-allocator**: ACTIVE at L1+ (per-pair residual bit-budget allocation); L0 uniform allocation
- **hook #4 cathedral autopilot dispatch**: ACTIVE (Phase 2 onward; ranks against sister Path 3 candidates A/B/C/D per Catalog #335/#336/#341 Tier A)
- **hook #5 continual-learning posterior**: ACTIVE (MLX smoke verdicts append `[macOS-MLX research-signal]` anchors per Catalog #341 + #323)
- **hook #6 probe-disambiguator**: ACTIVE (Catalog #1265 gate IS the canonical disambiguator between MLX-faithful vs MLX-too-noisy routing)

## Sister Path 3 candidate coordination (concurrent 2026-05-26 fanout)

Per Catalog #230 sister-subagent ownership map:
- **Candidate A**: DreamerV3 RSSM (`src/tac/substrates/dreamer_v3_rssm/`) — DISJOINT scope
- **Candidate B**: Z7-Mamba-2 — DISJOINT scope (sister scaffold path TBD)
- **Candidate C**: NSCS06 v8 chroma_lut (`src/tac/substrates/nscs06_v8_chroma_lut/`) — DISJOINT scope
- **Candidate D**: Z6 predictive coding (likely `src/tac/substrates/c1_world_model_foveation/` or sister) — DISJOINT scope
- **Candidate E (THIS LANE)**: BoostNeRV-against-PR110 (`src/tac/substrates/boost_nerv_pr110_residual/`) — DISJOINT scope

No file collision with any sister. All 5 candidates land NEW substrate packages per the binding 2026-05-26 reframing.

## Discipline applied

- Catalog #229 PV (read CLAUDE.md non-negotiables + 5 most-recent MEMORY.md entries + #1265 gate landing + #1258 corrected closure + #1257 inflate parity memo + sister `boost_nerv/` scaffold for shape reference + PR110 final evidence pack at `.omx/research/pr110_final_evidence_pack_20260520T141144Z_codex/`)
- Catalog #117 / #157 / #174 canonical serializer with POST-EDIT `--expected-content-sha256` (commit pending)
- Catalog #206 subagent checkpoint discipline (2 in_progress checkpoints emitted to `.omx/state/subagent_progress.jsonl`)
- Catalog #119 Co-Authored-By Claude trailer (commit pending)
- Catalog #287 placeholder-rationale rejection (every rationale ≥ 4 chars, non-placeholder)
- Catalog #110 / #113 APPEND-ONLY HISTORICAL_PROVENANCE (NEW research artifact; no mutation of existing memos)
- Catalog #208 docs/local-paths (no `/Users/adpena/...` in body text — sample paths use repo-relative form)
- Catalog #230 sister-subagent ownership map (A/B/C/D DISJOINT scope cited above)
- Catalog #340 sister-checkpoint guard (handled structurally by canonical serializer at commit time)
- Catalog #240 L0 SCAFFOLD posture (`_full_main_stub_raises NotImplementedError` per test_full_main_stub_raises_per_catalog_240 verified)
- Catalog #1265 MLX-first contest-equivalence gate (Stage 4 of canonical curriculum MANDATORY before paid CUDA dispatch)
- Catalog #127 / #192 / #317 / #341 non-promotable markers (`[macOS-MLX research-signal]` + `score_claim=false` + `promotion_eligible=false` + `ready_for_exact_eval_dispatch=false`)
- Catalog #341 dual-tier consumer architecture (Tier A observability-only at L0; substrate REGISTERS as Tier B at L1 after Phase 2 council symposium)
- Catalog #290 / #294 / #296 / #303 / #305 / #309 / #324 design-memo discipline (all 7 required sections present in design memo)
- CLAUDE.md "Executing actions with care" (NO `gh pr create` + NO `gh release create` + NO Modal/Vast/Lightning dispatch invocations)
- CLAUDE.md "Forbidden premature KILL without research exhaustion" (L0 SCAFFOLD posture means `research_only=true`/`dispatch_enabled=false` at recipe level; NOT a KILL)
- CLAUDE.md "Strategic Secrecy" (no public/external disclosure of approach)
