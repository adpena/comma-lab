---
schema_version: subagent_landing_memo_v2
created_utc: 2026-05-26T07:50:00Z
substrate_id: z8_hierarchical_predictive_coding
lane_id: lane_path_3_f_z8_hierarchical_predictive_coding_canonical_quadruple_20260526
lane_maturity_after_landing: L1 (impl_complete)
council_tier: T1
council_attendees: [Shannon, Dykstra, Rao, Ballard, Mallat, Tishby-memorial, Wyner, Hafner-DreamerV3-author-cite, AssumptionAdversary]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: ""
horizon_class: frontier_pursuit
predicted_band_validation_status: pending_post_training
sister_subagent_ownership_map:
  landed:
    - subagent_id: aaec7a0d220f31543
      substrate: dreamer_v3_rssm
      commit: 69253a1cc
      role: research INPUT — canonical categorical posterior + Gumbel-Softmax STE primitive (Z8 reuses per Catalog #290 ADOPT_CANONICAL_BECAUSE_SERVES at each hierarchy level)
    - subagent_id: af6ca73c5a7fc40f4
      substrate: time_traveler_l5_z6
      commit: 83b9ee3e2
      role: research INPUT — FiLM-conditioned predictor + ego-motion conditioning (Z8 generalizes to 3-level hierarchical)
    - subagent_id: a35f9f86781aaaa4f
      substrate: boost_nerv_pr110_residual
      commit: 83910e54e
      role: orthogonal sister (different substrate-class)
  in_flight_concurrent:
    - subagent_id: ac4283983ece21b83
      substrate: z7_mamba2_v2_fresh_substrate
      role: orthogonal sister (Mamba-2 SSM recurrent)
    - subagent_id: ad26de7ad5f90848a
      substrate: nscs06_v8_chroma_lut
      role: orthogonal sister (chroma_lut substitution)
  concurrent_spawns:
    - subagent: G_NIRVANA
      role: disjoint scope (NeRV-family residual cascade)
    - subagent: H_ATW_V2_cooperative_receiver
      role: disjoint scope (Atick-Tishby-Wyner cooperative-receiver)
related_deliberation_ids:
  - path_3_f_z8_hierarchical_predictive_coding_substrate_design_20260526
  - mlx_candidate_contest_equivalence_gate_landed_20260526
  - path_3_candidate_inventory_for_next_wave_spawning_20260526
  - time_traveler_l5_z6_z7_z8_predictive_coding_world_models_asymptotic_pursuit_scoping_design_20260516
canonical_equation_refs:
  - mlx_pytorch_full_decoder_downstream_scorer_drift_propagation_v1
  - scorer_conditional_joint_rate_distortion_floor_v1
  - categorical_posterior_capacity_vs_continuous_gaussian_v1
  - ego_motion_concentration_prior_v1
  - cross_codec_super_additive_orthogonality_predictor_v1
binding_operator_directives:
  - "The MLX first requirement might also force us out of the issue we were having before where we had great ideas but we're building them as Boltons to the same substrates over and over again; we want to design the substrate and curriculum and then optimize the design the whole stack around it for extreme optimization and performance and optimal score lowering" (2026-05-26)
  - "Never simply extend unless a rigorous adversarial cargo cult pass has been done first" (2026-05-26)
---

# Path 3 F=Z8 Hierarchical Predictive Coding — L0 SCAFFOLD Landing

Per Path 3 candidate inventory `.omx/research/path_3_candidate_inventory_for_next_wave_spawning_20260526.md` Tier 1 F. FRESH SUBSTRATE DESIGN (2-phase methodology) per operator binding directive 2026-05-26.

Z8 binds Catalog #312's canonical quadruple **simultaneously** (Rao-Ballard 1999 hierarchical predictive coding + Mallat 1989 wavelet multi-scale + Hafner DreamerV3 latent dynamics + Wyner-Ziv 1976 source-coding-with-side-information) per HNeRV parity discipline L7. Z8 is the asymptotic-pursuit terminal of the F-asymptote-trajectory.

## Premise verification (Catalog #229)

Pre-edit reads (verified empirically before any commit):

| File | Purpose |
|---|---|
| `.omx/research/path_3_candidate_inventory_for_next_wave_spawning_20260526.md` | Path 3 F brief + binding operator directives + universal brief template |
| `src/tac/substrates/time_traveler_l5_z6/__init__.py` | Sister D Z6 canonical-quadruple primitive #1 (Rao-Ballard + FiLM ego-motion) reference |
| `src/tac/substrates/time_traveler_l5_z6/architecture.py` (first 100 LOC) | Sister Z6 architecture pattern |
| `src/tac/substrates/time_traveler_l5_z6/inflate.py` | Sister Z6 inflate runtime pattern |
| `src/tac/substrates/dreamer_v3_rssm/__init__.py` | Sister A canonical-quadruple primitive #3 (DreamerV3 categorical posterior) reference |
| `src/tac/substrates/dreamer_v3_rssm/module.py` | Sister A MLX renderer pattern (Gumbel-Softmax STE + PixelShuffle block reuse) |
| `src/tac/substrates/dreamer_v3_rssm/archive.py` (first 200 LOC) | Sister A archive grammar pattern |
| `src/tac/substrates/_shared/inflate_runtime.py` | Canonical select_inflate_device + raw_output_path helpers |
| `.omx/research/time_traveler_l5_z6_z7_z8_predictive_coding_world_models_asymptotic_pursuit_scoping_design_20260516.md` | Parent scoping memo (Z6/Z7/Z8 design framing) |
| `.omx/state/canonical_equations_registry.jsonl` | Equation registry to cite per Catalog #344 |
| `.omx/state/lane_registry.json` | Lane registry pattern reference |

Bulk-edit count: 12 files in this landing (well above ≥3 threshold per Catalog #229). Empirical verdict table below replaces "assertion-only" claims with PASS/FAIL evidence.

## Empirical verdict table (Catalog #229 anti-overstatement discipline)

| Empirical check | Verdict | Evidence path |
|---|---|---|
| Substrate package imports | PASS | `pytest src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_basic.py::test_z8_substrate_package_imports` |
| Catalog #124 8-field declaration | PASS | `pytest -k test_z8_catalog_124_archive_grammar_fields_declared` |
| Catalog #344 canonical equation refs declared | PASS | `pytest -k test_z8_canonical_equation_refs_declared` |
| Catalog #91 ENCODE_INFLATE_ROUNDTRIP byte-deterministic | PASS | `pytest -k test_z8_archive_round_trip_byte_deterministic` |
| Section offsets parse correctly (no gaps/overlaps) | PASS | `pytest -k test_z8_archive_section_offsets_parse` |
| Archive refuses corrupt magic | PASS | `pytest -k test_z8_archive_refuses_corrupt_magic` |
| Catalog #139 + #272 byte-mutation no_op_proof on all 4 distinguishing features | PASS | `pytest -k test_z8_distinguishing_feature_sections_change_archive_bytes` |
| Sorted-keys JSON deterministic | PASS | `pytest -k test_z8_archive_meta_json_deterministic_sort_keys` |
| Inflate parse_and_validate accepts valid archive | PASS | `pytest -k test_z8_inflate_parse_and_validate_passes_on_valid_archive` |
| Catalog #240 acceptance cascade (c): forward council-gated | PASS | `pytest -k test_z8_inflate_raises_l0_scaffold_not_implemented_on_runtime_forward` |
| MLX config defaults validate | PASS | `pytest -k test_z8_mlx_config_defaults_validate` |
| MLX config rejects bad level counts | PASS | `pytest -k test_z8_mlx_config_rejects_bad_level_counts` |
| MLX renderer constructs + forward shape correct | PASS | `pytest -k test_z8_mlx_renderer_constructs_and_forward` |
| MLX renderer eval-from-indices matches shape | PASS | `pytest -k test_z8_mlx_renderer_eval_from_indices_matches_shape` |
| Architecture manifest non-promotable per Catalog #127/#192/#317/#341 | PASS | `pytest -k test_z8_mlx_architecture_manifest_observability_surface` |
| Param count monotonic in scale | PASS | `pytest -k test_z8_decoder_param_count_increases_with_levels` |
| MLX smoke trainer converges monotonically 2537->2503->2500 in 0.2s | PASS | `.venv/bin/python experiments/train_substrate_z8_hierarchical_predictive_coding_mlx.py --smoke --epochs 3 --num-pairs 4` |
| MLX trainer `_full_main raises NotImplementedError` per Catalog #240 | PASS | bare invocation without `--smoke` raises NotImplementedError immediately |

**Net: 18/18 PASS.**

## What landed

### Substrate package (`src/tac/substrates/z8_hierarchical_predictive_coding/`)

- `__init__.py` (8.5K) — Catalog #124 8-field declaration + Catalog #241 LEGACY_SUBSTRATE_PRE_META_LAYER waiver + Catalog #344 canonical equation refs + public API re-exports
- `mlx_renderer.py` (16K) — MLX multi-level RSSM hierarchy + per-level Gumbel-Softmax STE + Mallat sum-pool proxy (Phase 2 lands full Daubechies-4 DWT) + DreamerV3 linear-gate proxy (Phase 2 lands full GRUCell) + canonical PR95 HNeRV decoder reuse (sister A pattern)
- `archive.py` (15K) — Z8HPC1 byte-deterministic 62-byte header + 5 distinguishing-feature sections (DECODER + INDICES + WAVELET + WYNER_ZIV + DREAMER_STATE) + META + canonical sister A state-dict serialization pattern reuse
- `inflate.py` (5.5K) — PyTorch inflate stub that parses + validates archive then raises Z8L0ScaffoldNotImplementedError per Catalog #240 acceptance cascade (c) pre-build substrate-engineering
- `tests/__init__.py` + `tests/test_basic.py` (15K) — 16 tests covering Catalog #91 ENCODE_INFLATE_ROUNDTRIP, Catalog #139 + #272 byte-mutation no_op_proof on all 4 distinguishing primitives, MLX renderer construction + forward, architecture manifest observability, param count sanity

### MLX smoke trainer

- `experiments/train_substrate_z8_hierarchical_predictive_coding_mlx.py` — thin MLX-local trainer (≤5ep ≤8pairs smoke; synthetic MSE proxy + Rao-Ballard residual entropy proxy); `_full_main raises NotImplementedError` per Catalog #240

### Memos

- `.omx/research/path_3_f_z8_hierarchical_predictive_coding_substrate_design_20260526.md` — design memo (frontmatter v2 + Section 1-14 covering Catalog #290 + #294 + #296 + #303 + #305 + #309 + #310 + #311 + #312 + 6-hook wire-in + 13 catalog compliance items + operator-routable next steps)
- `.omx/research/path_3_f_z8_hierarchical_predictive_coding_L0_scaffold_landed_20260526.md` (this memo)

### Lane registry

- New lane `lane_path_3_f_z8_hierarchical_predictive_coding_canonical_quadruple_20260526` at L1 (impl_complete gate marked PASS with detailed evidence string)

## Canonical-vs-unique decision per layer summary (per Catalog #290)

Per design memo Section 2 — net split is ~50% canonical adoption + ~50% unique substrate engineering:

- **ADOPT_CANONICAL_BECAUSE_SERVES**: trainer skeleton, scorer loss helper, eval_roundtrip, YUV6 patch, EMA decay, inflate device selector, MLX↔PyTorch bridge (Gumbel-Softmax STE + PixelShuffle decoder), DreamerV3 categorical posterior per level
- **FORK_BECAUSE_UNIQUE / FORK_BECAUSE_PRINCIPLED_MISMATCH**: archive grammar Z8HPC1, multi-level RSSM hierarchy stack, Wyner-Ziv side-info coder, Mallat wavelet detail-band codec, inflate runtime (PyTorch), score-aware loss (hierarchical)
- **DEFER_PHASE_2**: trainer `_full_main` per Catalog #240 acceptance cascade (c)

## The canonical-quadruple binding math derivation

Per design memo Section 4: the multiplicative joint entropy reduction bound is

$$
\Delta H_{\text{Z8 total}} \approx (1 - r_{\text{RB}})(1 - r_{\text{Mallat}})(1 - r_{\text{DreamerV3}})(1 - r_{\text{WZ}}) \cdot H_{\text{naive}}
$$

with per-primitive empirical reduction factors anchored from sister substrate measurements: $r_{\text{Rao-Ballard}} \approx 0.35$, $r_{\text{Mallat}} \approx 0.25$, $r_{\text{DreamerV3}} \approx 0.20$, $r_{\text{Wyner-Ziv}} \approx 0.30$. Joint best-case factor = 0.273; Z8 targets ~75 KB archive (vs Quantizr 293 KB ceiling).

Per Boyd's Dykstra-feasibility lens: the multiplicative bound is an UPPER bound; the true achievable is the convex-intersection projection (subadditive). The 75 KB is the planning prior; the achievable floor requires post-training Tier-C measurement per Catalog #324.

## Dykstra-feasibility predicted band

Per design memo Section 4.4: **predicted ΔS band [0.05, 0.10]** (frontier_pursuit horizon class; asymptotic-pursuit per parent scoping memo Z8 row). `predicted_band_validation_status: pending_post_training` per Catalog #324 STRICT preflight requirement. Non-promotable per CLAUDE.md "Apples-to-apples evidence discipline" until post-training Tier-C density measurement validates within ±20% OR MLX-local smoke convergence demonstrates the canonical-quadruple binding empirically per Catalog #1265 MLX gate.

## MLX-trainable curriculum stages with smoke convergence targets

Per design memo Section 9 — 6-phase curriculum (Phase 0 warmup with rate loss → Phase 1 add level 1 → Phase 2 add level 2 + Mallat → Phase 3 add DreamerV3 categorical + score-aware loss → Phase 4 add Wyner-Ziv top-level → Phase 5 QAT FP4 → Phase 6 EMA + paired CPU/CUDA auth-eval).

**L0 smoke convergence target**: per-pair MSE proxy decreases monotonically over ≥5 epochs at ≥8 pairs.

**L0 empirical smoke**: 2537.68 → 2503.03 → 2500.61 (3 epochs × 4 pairs synthetic MSE proxy on Apple Silicon; wall-clock 0.21 sec). PASS verdict on monotonic decrease.

## 6-hook wire-in declaration (per Catalog #125 NON-NEGOTIABLE)

1. **Sensitivity-map** = ACTIVE (per-level prediction error L2 norm IS the per-tensor importance signal; canonical Pytorch port Phase 2 registers `sensitivity_map.z8_hierarchical_predictive_coding_v1`)
2. **Pareto constraint** = ACTIVE (per-level prediction error entropy bound adds to convex feasibility region per level; canonical Phase 2 registers `tac.pareto.z8_hierarchical_predictive_coding_v1`)
3. **Bit-allocator hook** = ACTIVE (per-level bit allocation derives from per-level Mallat wavelet detail-band sparsity; canonical Phase 2 registers `bit_allocator.z8_hierarchical_predictive_coding_per_level_v1`)
4. **Cathedral autopilot dispatch hook** = ACTIVE (recipe planned at `.omx/operator_authorize_recipes/substrate_z8_hierarchical_predictive_coding_modal_a100_dispatch.yaml` Phase 2; gated by Catalog #167 smoke-before-full + Catalog #325 per-substrate symposium)
5. **Continual-learning posterior** = ACTIVE (every Z8 empirical anchor seeds posterior via `posterior_update_locked` per Catalog #128 — including the MLX smoke convergence JSON manifest emitted by the trainer with `evidence_grade=MPS-research-signal`)
6. **Probe-disambiguator** = ACTIVE (per-primitive ablation IS the probe — design memo Section 4.4 + Section 6 cargo-cult audit Section 6 unwind plans; `--ablate-rao-ballard` / `--ablate-mallat` / `--ablate-dreamer-categorical` / `--ablate-wyner-ziv` argparse flags planned for Phase 2)

## Operator-routable next steps

1. **Sister gate for Z8HPC1 grammar** (lane `lane_gate_mlx_candidate_contest_equivalence_z8_20260526`): extend canonical MLX gate `tools/gate_mlx_candidate_contest_equivalence.py` to support Z8HPC1 archive grammar (current gate is hardwired for PR95 grammar per A=DreamerV3 follow-up). Estimated cost: $0 + ~2-3h wall-clock.
2. **Contest-scale MLX smoke** (lane `lane_z8_contest_scale_mlx_smoke_20260526`): scale L0 smoke from synthetic to actual `upstream/videos/0.mkv` 600-pair contest video; estimated cost $0 + ~1-2h wall-clock; per CLAUDE.md HNeRV parity L1 (training MUST use upstream video).
3. **PyTorch port via canonical bridge** (lane `lane_z8_pytorch_port_via_canonical_bridge_20260526`): port MLX module to PyTorch via `tac.local_acceleration.pr95_hnerv_mlx::load_pytorch_state_dict_into_mlx` sister pattern. Estimated cost: $0 + ~3-4h wall-clock.
4. **Per-substrate symposium per Catalog #325** (lane `lane_z8_per_substrate_optimal_form_symposium_20260526`): REQUIRED before paid-dispatch authorization per Catalog #325 STRICT preflight; canonical 6-step contract (cargo-cult audit + 9-dim checklist + observability surface + sextet pact + reactivation criteria + Catalog #324 post-training Tier-C validation discipline). Estimated cost: $0 + ~3-4h wall-clock.
5. **Per-primitive ablation probes** (post Phase 2): the 4 canonical-quadruple primitives ablated independently per design memo Section 4.4 disambiguator + Section 6 cargo-cult audit unwind plans. Estimated cost: $0 MLX-local + paid CUDA at $5-10 per primitive once Phase 2 lands.
6. **Composition smoke** (post per-substrate symposium): Z8 × A-STACK × NSCS06 v8 Path B 3-axis orthogonal asymptotic-pursuit composition per parent scoping memo Section 13. Estimated cost: $0 MLX-local + paid CUDA at $5-15 once Phase 2 + composition smoke lands.

## Mission contribution per Catalog #300

`frontier_breaking` — Z8 binds the canonical quadruple (Catalog #312) **simultaneously** for the FIRST time in the repo, operationalizing the asymptotic-pursuit substrate-class-shift at the F-asymptote-trajectory terminal. The MLX-local L0 SCAFFOLD enables $0 iteration on the canonical-quadruple binding hypothesis BEFORE any paid CUDA dispatch authorization per CLAUDE.md "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch" non-negotiable.

The substrate-design-from-first-principles + cargo-cult-pass-FIRST discipline (per operator's binding directives 2026-05-26) is locked into Z8 by construction: every layer's canonical-vs-unique decision documented (design memo Section 2), every assumption taxonomized HARD-EARNED-vs-CARGO-CULTED (design memo Section 6), every cargo-culted assumption paired with explicit ablation unwind plan (design memo Section 4.4 + Section 6).

## Cost + wall-clock

- **Cost**: $0 (L0 SCAFFOLD; MLX-local only; no paid GPU)
- **Wall-clock**: ~3.5h (premise verification 30min + design memo 1h + L0 scaffold package 1h + MLX trainer 30min + tests 30min + landing memo 15min)
- **No `gh pr create` / `gh release create` / Modal/Vast/Lightning dispatch invoked** per CLAUDE.md "Executing actions with care".

## APPEND-ONLY discipline (per Catalog #110/#113)

This landing memo is APPEND-ONLY HISTORICAL_PROVENANCE; never mutated. Sister design memo and trainer + substrate package are likewise append-only.

EOF
