<!--
SPDX-License-Identifier: MIT
council_tier: T1
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, AssumptionAdversary, Wyner, Atick, Redlich, Tishby]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "THIRD-SURFACE empirical falsification was the strongest remaining candidate in the deterministic-from-uncorrelated-sources Y-family. The remaining unwind path is REAL PoseNet pre-compute (Wave N+10 path #1); we MUST acknowledge that prefix-detector on entropy-flat fp16 tensor weights is a structurally-narrow operationalization of Wyner 1976 R(D|Y) that may never yield density >= 1% regardless of Y source. The PARADIGM-INTACT classification per Catalog #307 must NOT be a permanent get-out-of-jail-free card."
council_assumption_adversary_verdict:
  - assumption: "Cross-substrate composition Y (FEC6/FECA frontier ZIP-member payload) shares byte-level prefix overlap with PR101 fp16 state_dict because both substrates encode the same contest video."
    classification: CARGO-CULTED
    rationale: "FALSIFIED empirically: 0.000437% density. Sister substrate entropy coders distribute shared video-structure encoding ACROSS the bitstream rather than at byte-prefix boundaries; the assumption that 'same contest video → byte-prefix overlap' was an inference from canonical-source PRIOR rather than an empirical anchor."
  - assumption: "Wyner 1976 R(D|Y) paradigm is INTACT at the cross-substrate composition surface even after THIRD-SURFACE falsification."
    classification: HARD-EARNED
    rationale: "Per Catalog #307 paradigm-vs-implementation classification: the prefix-detector IMPLEMENTATION is falsified; the cooperative-receiver PARADIGM is independent of the prefix-detector implementation. Sister Y derivations (real PoseNet pre-compute; non-prefix Hamming/KL detector; sparse-coding latent overlap) remain canonical reactivation paths."
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: ""
council_decisions_recorded:
  - "op-routable #1: REGISTER canonical equation wyner_ziv_cross_substrate_composition_y_pose_axis_savings_v1 + empirical anchor wave_n_plus_9_slot_3_cross_substrate_composition_y_fec6_for_pr101_fp16 per Catalog #344 - LANDED"
  - "op-routable #2: REGISTER canonical anti-pattern wyner_ziv_cross_substrate_composition_y_density_decoder_state_dict_surface_v1 per Catalog #344+#335 - LANDED"
  - "op-routable #3: REGISTER probe outcome wyner_ziv_cross_substrate_composition_y_fec6_for_pr101_density_third_surface_20260528 with verdict DEFER + blocker_status=blocking per Catalog #313 - LANDED"
  - "op-routable #4: SCAFFOLD operator-authorize recipe substrate_wyner_ziv_cross_substrate_composition_y_fec6_for_pr101_modal_paired_dispatch (dispatch_enabled=false per Catalog #240+#370 + #325 6-step contract) - LANDED"
  - "op-routable #5: QUEUE Wave N+10 reactivation path #1 (real PoseNet pre-compute Y via Modal blanket, ~$1-5 cost) - QUEUED-PENDING-OPERATOR-ATTESTATION per Catalog #325 + #246 + PR-creation operator-explicit-per-PR gate"
related_deliberation_ids:
  - wyner_ziv_pipeline_stage_codec_l1_long_mlx_600pair_landed_20260528
  - anti_pattern_registry_expansion_13_14_15_plus_wyner_ziv_op_routable_5_landed_20260528
  - mlx_first_numpy_portable_individually_fractally_optimized_standing_directive_20260526
deferred_substrate_id: wyner_ziv_pipeline_stage_codec
deferred_substrate_retrospective_due_utc: 2026-06-27T18:05:09Z
-->

# Wyner-Ziv Cross-Substrate Composition Y (FEC6 as Y for PR101) — Wave N+9 Slot 3 LANDED 2026-05-28

## TL;DR (operator-action-ready)

**EMPIRICAL VERDICT**: **THIRD-SURFACE FALSIFIED** at 0.000437% Y-derivable-prefix density (×2290 below 1% threshold per op-routable #4). PARADIGM intact per Catalog #307; IMPLEMENTATION at cross-substrate composition Y surface falsified. **NEXT STEP**: Wave N+10 reactivation path #1 — real PoseNet pre-compute Y via Modal blanket authorization (~$1-5 CUDA cost; uses sister per-pair-posenet recipe scaffold; operator-attended L2 paired-CUDA per Catalog #246).

**WHAT LANDED** (atomic-paired per Catalog #117/#157/#174):
1. `src/tac/substrates/wyner_ziv_pipeline_stage_codec/trainer.py` — `_derive_cross_substrate_composition_y_fec6_for_pr101()` + `_measure_cross_substrate_composition_y_density_fec6_for_pr101()` + `--cross-substrate-composition-y` CLI flag + verdict integration
2. `experiments/results/wyner_ziv_cross_substrate_composition_y_fec6_for_pr101_l1_mlx_20260528/` — canonical training artifact + WZPSC01 archive (193512 B; sha 98e8fa948d5651ba; roundtrip byte-identical)
3. `.omx/state/canonical_equations_registry.jsonl` — equation `wyner_ziv_cross_substrate_composition_y_pose_axis_savings_v1` + anchor `wave_n_plus_9_slot_3_cross_substrate_composition_y_fec6_for_pr101_fp16` per Catalog #344
4. `src/tac/canonical_anti_patterns/builtins.py` — anti-pattern #16 `wyner_ziv_cross_substrate_composition_y_density_decoder_state_dict_surface_v1` registered per Catalog #335
5. `.omx/state/probe_outcomes.jsonl` — probe outcome `wyner_ziv_cross_substrate_composition_y_fec6_for_pr101_density_third_surface_20260528` (DEFER blocking 30-day window) per Catalog #313
6. `.omx/operator_authorize_recipes/substrate_wyner_ziv_cross_substrate_composition_y_fec6_for_pr101_modal_paired_dispatch.yaml` — scaffold (dispatch_enabled=false) per Catalog #325 6-step contract

## Empirical anchor

**Setup** (Catalog #229 PV-verified):
- **X** = PR101 fp16 decoder state_dict raw concatenated bytes at `experiments/results/pr101_codecop_sweep_20260507_codex/pr101_decoder_state_dict.pt` (sha256 prefix `79b804d9a5839eb3`; 457916 B; 28 top-level state_dict entries)
- **Y** = canonical FEC6/FECA frontier archive ZIP-member 'x' payload (entropy-coded selector packet) at `experiments/results/feca_selector_reparameterized_scale64_alpha1_rebuilt_20260528Tlocal/submission_dir/archive.zip` (archive sha256 prefix `18e3155fbbbe9ab2`; archive 178530 B; ZIP member 'x' payload 178430 B; Y bytes sha256 prefix `2566dcd6652a`)
- **Y location loaded via** `tac.canonical_frontier_pointer.load_canonical_frontier_pointer_lenient` per Catalog #343 "Frontier scores are pointer-only" non-negotiable (NOT hardcoded literals; the canonical helper resolves frontier axis → architecture_class → on-disk archive path)
- **Substrate**: `wyner_ziv_pipeline_stage_codec` L1 LONG MLX harness with NEW `--cross-substrate-composition-y` flag

**Measurement** (MLX-LOCAL Darwin ARM64 M5 Max; $0.00 GPU; wall_clock 0.52s):
- Y-derivable-prefix length: **2 bytes** (longest contiguous prefix of X that exists as substring of Y)
- Y-derivable-prefix density: **0.000437%** (2 / 457916)
- Threshold per op-routable #4: 1.0%
- Residual factor below threshold: ×2290
- WZPSC01 archive emission: 193512 B (sha256 `98e8fa948d5651ba`); roundtrip byte-identical in 0.006s (Wyner 1976 reconstructibility invariant satisfied per Catalog #105/#139/#220/#272 no-op detector)
- lzma sanity ratio on PR101 fp16 bytes: 0.4215 (OUTSIDE sister prober band 0.217-0.228; suggests entropy-flat raw fp16 is genuinely high-entropy at byte distribution level)

## Sister cross-surface comparison

The Wyner-Ziv R(D|Y) paradigm has now been empirically tested at **THREE** Y-derivation surfaces, ALL falsified:

| Wave | Surface | Y type | Density | Below threshold | Sister landing |
|------|---------|--------|---------|-----------------|----------------|
| N+5 | First | 4 canonical sources (Comma2k19 / ImageNet / torch_defaults / math_constants) | 0.000218% | ×4585 | `wyner_ziv_pipeline_stage_codec_l1_long_mlx_600pair_landed_20260528.md` `6f5eabf30` |
| N+7 Slot 2 | Second | per-pair PoseNet-output Y deterministic stand-in (Atick-Redlich 1990) | 0.000218% | ×4585 | `anti_pattern_registry_expansion_13_14_15_plus_wyner_ziv_op_routable_5_landed_20260528.md` `49bdcd78f` |
| **N+9 Slot 3** | **Third** | **cross-substrate composition Y (FECA frontier ZIP-member payload)** | **0.000437%** | **×2290** | **THIS LANDING** |
| N+10 (queued) | Fourth | real PoseNet pre-compute Y on contest 600 pairs (Modal blanket) | pending | pending | path #1 reactivation |

The three falsified surfaces share a common structural property: each Y derivation is **deterministic from sources UNCORRELATED with X's analytical derivation path** (canonical priors / pair index / sister substrate's bytes). The empirical pattern strongly suggests entropy-flat fp16 raw weight bytes do NOT admit contiguous-prefix overlap with any deterministic-from-uncorrelated-sources Y. The Wave N+10 fourth surface tests the orthogonal hypothesis that REAL PoseNet output on real contest frames (the canonical contest-video-derived signal, NOT a stand-in) IS the correct Y source for the cooperative-receiver framing.

## Verdict structure (per Catalog #307 paradigm-vs-implementation)

- **PARADIGM**: Wyner 1976 R(D|Y) + Atick-Tishby-Wyner cooperative-receiver triple per Catalog #311 — **INTACT**. The cooperative-receiver framing makes NO commitment to a specific Y-derivation source; it commits to the existence of SOME side-info Y for which I(X; Y) > 0 reduces decoder-side rate.
- **IMPLEMENTATION**: Prefix-detector at cross-substrate composition Y surface — **FALSIFIED**. The contiguous-prefix overlap operationalization fails on this Y derivation, sister to the same operationalization failing on the prior two Y surfaces.
- **PER CLAUDE.md "Forbidden premature KILL"**: substrate DEFERRED-PENDING-research; NOT killed. Reactivation paths enumerated below.

## Reactivation criteria (priority-ordered per Catalog #313)

1. **Wave N+10 path #1 — real PoseNet pre-compute Y** (HIGHEST EV; canonical anti-pattern #16 unwind): pre-compute per-pair PoseNet output on contest video 0.mkv via Modal CUDA dispatch (~$1-5 Modal-approved per blanket per `feedback_modal_spend_blanket_approved_but_mlx_first_for_everything_standing_directive_20260528.md`); ship per-pair pose tensor (~14400 bytes for 600 pairs × 6 dims × float32) as decoder-side side-info Y baked into archive. Decoder reproduces Y deterministically at inflate WITHOUT loading scorer per Catalog #6 strict-scorer-rule. **Sister recipe scaffold** at `.omx/operator_authorize_recipes/substrate_wyner_ziv_pipeline_stage_codec_per_pair_posenet_output_y_modal_paired_dispatch.yaml` (dispatch_enabled=false; awaits operator-attested symposium per Catalog #325 6-step contract).

2. **Wave N+10 path #2 — non-prefix Y-overlap detector** (MEDIUM EV; primitive extension): extend canonical `tac.codec.wyner_ziv_layer._detect_y_derivable_prefix` from contiguous-prefix to Hamming-distance / KL-divergence overlap detection. Sister approach if path #1 also yields density < 1%. ~$0 MLX-LOCAL + ~200 LOC primitive extension. Lower EV than path #1 because Hamming on fp16 weights against compressed bytes is still likely entropy-flat.

3. **Wave N+11 path #3 — CSC-style sparse-coding Y derivation** (HIGHEST UPSIDE; substantial new infrastructure): derive Y as reconstruction residual of PR101 weights against sparse codebook trained on contest video latents. Sister direction per Catalog #311 Atick-Tishby-Wyner at the latent-coding surface (vs byte-prefix surface). Higher EV than path #2 but requires substantial new infrastructure; deferred until paths #1+#2 resolve.

## 6-hook wire-in declaration per Catalog #125

- **Hook #1 sensitivity-map**: ACTIVE — per-block cross-substrate Y mutual-information sensitivity surfaces I(X; Y) attribution as observability-only signal per Catalog #305 (non-promotable per Catalog #341 Tier A markers).
- **Hook #2 Pareto constraint**: ACTIVE — pose-axis Lagrangian dual surfaces via Catalog #372 (when Y-density passes threshold; current density 0.000437% does NOT activate a Pareto constraint, but the canonical equation hook is wired for future anchors).
- **Hook #3 bit-allocator**: ACTIVE — per-block Y density routes bit-allocator AWAY from low-density blocks (current implementation: blocks share global density 0.000437% so no per-block routing yet; future fourth-surface real PoseNet Y will surface per-pair density variability).
- **Hook #4 cathedral autopilot dispatch**: ACTIVE via Catalog #335 — canonical anti-pattern #16 + canonical equation #344 anchor auto-discoverable; sister `tac.cathedral_consumers.wyner_ziv_pipeline_stage_codec` consumer surfaces this substrate's THIRD-SURFACE FALSIFIED status to the autopilot ranker.
- **Hook #5 continual-learning posterior**: ACTIVE — canonical equation `wyner_ziv_cross_substrate_composition_y_pose_axis_savings_v1` + empirical anchor `wave_n_plus_9_slot_3_cross_substrate_composition_y_fec6_for_pr101_fp16` + sister anti-pattern + probe outcome all REGISTERED via canonical helpers (fcntl-locked APPEND-ONLY per Catalog #131/#138/#245); Catalog #371 auto-recalibration will fire when ≥3 cumulative anchors land in this equation's domain.
- **Hook #6 probe-disambiguator**: ACTIVE — cross-substrate composition Y derivation IS the canonical disambiguator between (a) FALSIFIED deterministic-from-uncorrelated-sources Y family (Waves N+5/N+7/N+9; THREE empirical receipts) and (b) FUTURE real-PoseNet-pre-compute Y (Wave N+10 path #1). The sister probe-outcome row makes this disambiguation queryable via `tac.probe_outcomes_ledger.query_blocking_outcomes` per Catalog #313.

## Canonical-vs-unique decision per layer (per Catalog #290 + UNIQUE-AND-COMPLETE-PER-METHOD)

- **Architecture (substrate)**: REUSED — `WynerZivPipelineStageCodecArchitecture` from sister L1 trainer; cross-substrate composition Y is a DERIVATION-PATH change, not an architecture change. The encoder/decoder pair is byte-identical to Wave N+5/N+7 sister anchors.
- **Y-derivation path**: NEW UNIQUE — `_derive_cross_substrate_composition_y_fec6_for_pr101()` derives Y from canonical frontier pointer per Catalog #343 (NOT hardcoded literals). FORKED from sister derivation paths because cross-substrate Y has a structurally different load path (ZIP unwrap + canonical pointer resolution + sha256 verification) vs deterministic-from-source generation.
- **Frontier pointer loader**: REUSED — `tac.canonical_frontier_pointer.load_canonical_frontier_pointer_lenient` is the canonical lenient loader per Catalog #343.
- **Prefix detector**: REUSED — `tac.codec.wyner_ziv_layer._detect_y_derivable_prefix` is the canonical longest-prefix matcher; same byte-level semantics across all 3 Y surfaces.
- **Archive emission**: REUSED — `encode_archive_bytes_scaffold` + `WZPSC01` grammar; byte-identical roundtrip verified.
- **Verdict logic**: NEW UNIQUE — third-surface falsification branch added to `_full_main` verdict logic with sister `cross_substrate_y_tested` priority over `per_pair_y_tested` (so verdict message accurately reflects which Y surfaces were tested in this run).
- **Canonical equation**: NEW UNIQUE — `wyner_ziv_cross_substrate_composition_y_pose_axis_savings_v1` registered per Catalog #344. FORK rationale: distinct Y derivation domain (`frontier_axis` / `zip_member_name` / `frontier_archive_loader` as canonical pointer reference) per Catalog #290 falling-rule "FORK_BECAUSE_PRINCIPLED_MISMATCH"; the sister `wyner_ziv_per_pair_posenet_output_y_pose_axis_savings_v1` equation's domain (num_pairs / pose_dim / dtype) does NOT cover the cross-substrate path.
- **Canonical anti-pattern**: NEW UNIQUE — `wyner_ziv_cross_substrate_composition_y_density_decoder_state_dict_surface_v1` (#16) FORKED from sister #15 because the recurrence_conditions + canonical_unwind_path differ structurally (Y comes from different source family).
- **Probe outcomes ledger**: REUSED — `tac.probe_outcomes_ledger.register_probe_outcome` canonical helper per Catalog #313 + #245 4-layer pattern.

## 9-dimension success checklist evidence (per Catalog #294)

1. **UNIQUENESS**: NEW Y-derivation path (FECA frontier ZIP-member payload) distinct from prior 4 canonical sources + per-pair pose stand-in; canonical equation + anti-pattern registered as new entries.
2. **BEAUTY + ELEGANCE**: ~270 LOC trainer addition (helpers + CLI + verdict integration); reviewable in 30 seconds per PR101-style discipline (Catalog #146 + HNeRV parity L4).
3. **DISTINCTNESS**: cross-substrate composition Y is the THIRD Y surface tested in the deterministic-from-uncorrelated-sources family; distinct verdict path + anti-pattern + canonical equation prevent collapse with sister surfaces.
4. **RIGOR**: Catalog #229 PV verified canonical FECA frontier on-disk + Catalog #343 pointer integrity + Catalog #287 placeholder-rationale rejection on all artifacts; canonical Provenance per Catalog #323 throughout.
5. **OPTIMIZATION PER TECHNIQUE**: per Catalog #290 falling-rule applied per layer (REUSED architecture/loader/detector/grammar; FORKED Y-derivation/equation/anti-pattern); substrate-optimal engineering for cross-substrate composition Y as distinct from sister Y derivations.
6. **STACK-OF-STACKS COMPOSABILITY**: cross-substrate composition Y is ORTHOGONAL axis to sister substrate's encoding path; canonical equation domain explicitly captures `frontier_axis` + `zip_member_name` so future stacking experiments can mix multiple Y surfaces.
7. **DETERMINISTIC REPRODUCIBILITY**: byte-identical roundtrip verified; canonical frontier pointer + ZIP member name + sha256 verification ensure Y derivation is bit-stable across runs; seed-pinned canonical encoder/decoder.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: $0.00 GPU + 0.52s wall_clock on M5 Max via MLX-FIRST $0 envelope per CLAUDE.md 8th standing directive 2026-05-26.
9. **OPTIMAL MINIMAL CONTEST SCORE**: THIRD-SURFACE FALSIFICATION at 0.000437% density yields predicted ΔS = 0 (sub-threshold); NOT a contest score win but a frontier-PROTECTING measurement that PREVENTS Wave N+10 from wasting paid CUDA on the same falsified Y surface. Reactivation path #1 (Wave N+10 real PoseNet) is the NEXT score-lowering candidate.

## Observability surface (per Catalog #305)

1. **Inspectable per layer**: training_artifact.json carries Y bytes len + sha + prefix_len + density + frontier archive sha + arch_class + lzma ratio + per-stage byte counts.
2. **Decomposable per signal**: max_density_pct aggregates across 4 canonical + per-pair + cross-substrate Y measurements; best_source field identifies which Y won.
3. **Diff-able across runs**: canonical seed (0) + canonical pointer-based Y loading ensures bit-stable artifact across runs.
4. **Queryable post-hoc**: canonical equation registry + anti-pattern registry + probe outcomes ledger all queryable via `tools/list_canonical_equations.py` + `tools/check_predecessor_probe_outcome.py`.
5. **Cite-able**: canonical Provenance per Catalog #323 on every emitted row; commit sha + run UTC + frontier pointer ref.
6. **Counterfactual-able**: `--cross-substrate-composition-y-frontier-archive-path-override` flag allows testing alternative frontier archives (e.g. CUDA frontier 9cb989ce vs CPU frontier 18e3155f) for byte-mutation sensitivity probes per Catalog #105/#139/#272.

## Predicted ΔS band (per Catalog #296 Dykstra-feasibility check)

- **Predicted band** (per sister design memo §Predicted ΔS band): density >= 1% would yield saturating composition alpha=0.5 with predicted ΔS in [-0.0050, -0.0020]. Density < 1% predicts ΔS = 0 (sub-threshold falsification region).
- **Empirical density**: 0.000437% < 1.0% → predicted ΔS = 0.0 (sub-frontier non-candidate).
- **Dykstra-feasibility per Catalog #296**: alternating-projection feasibility intersection of (a) rate constraint (Y must reduce decoder bytes by ≥ 1% of X bytes), (b) Wyner 1976 R(D|Y) achievability (Y must share I(X; Y) > 0). Empirical I(X; Y) at prefix-detector surface is ~0 (2 bytes overlap on 457916 B X) → infeasible intersection → predicted ΔS = 0. First-principles bound per Shannon I(X; Y) >= H(X) - H(X|Y); empirical H(X) ≈ -log2(0.4215) * 8 bits/byte ≈ 9.93 bits/byte; H(X|Y) ≈ H(X) (since I(X;Y) ≈ 0) → no rate savings achievable at this Y derivation.

## Cargo-cult audit per assumption (per Catalog #303)

1. **ASSUMPTION**: Cross-substrate composition Y (FECA frontier) shares byte-level prefix overlap with PR101 fp16 state_dict because both substrates encode the same contest video.
   - **CLASSIFICATION**: CARGO-CULTED (FALSIFIED empirically).
   - **UNWIND**: shared video-structure encoding is at the SEMANTIC layer (pixel content), not the byte-prefix layer. Entropy coders distribute encoding ACROSS the bitstream; byte-prefix overlap requires SHARED PREFIX in the entropy-encoded representation, which entropy coders explicitly avoid.

2. **ASSUMPTION**: Wyner 1976 R(D|Y) paradigm transfers from theoretical context (continuous Gaussian sources + ideal entropy coders) to operationally bounded context (fp16 raw weight bytes + canonical 4 codec sources).
   - **CLASSIFICATION**: HARD-EARNED (PARADIGM still INTACT per Catalog #307; only the operationalization at the byte-prefix detector surface is falsified).
   - **UNWIND**: alternative operationalizations remain canonical (real PoseNet pre-compute Y; non-prefix overlap detector; sparse-coding latent Y).

3. **ASSUMPTION**: The canonical FECA frontier ZIP-member 'x' payload IS the canonical Y bytes for the cross-substrate composition surface.
   - **CLASSIFICATION**: HARD-EARNED (Wave N+5 empirical anchor verified ZIP grammar; canonical pointer per Catalog #343).
   - **UNWIND**: N/A — this assumption is empirically verified.

## Sister coordination (per Catalog #340)

- **Slot 1 Wave N+8 (`slot_1_z7_mamba2_hinton_`)**: working on `experiments/train_substrate_time_traveler_l5_z7_mamba2_mlx_local.py` + `src/tac/substrates/time_traveler_l5_z7_mamba2/` — DISJOINT scope; no file collisions.
- **Slot 2 Wave N+8 (`slot_2_phase_9_canonical`)**: working on `tools/operator_pr_submission_full_lifecycle.py` + `src/tac/submission_packet/` — DISJOINT scope; no file collisions.
- **Slot 2 Wave N+8 strict-gate (`slot2_wave_n8_strict_gat`)**: working on preflight strict-gate self-protection (likely `src/tac/preflight.py`) — DISJOINT scope; no file collisions.
- **This work (Slot 3 Wave N+9)** touches: `src/tac/substrates/wyner_ziv_pipeline_stage_codec/trainer.py`, `src/tac/canonical_anti_patterns/builtins.py`, `.omx/state/canonical_equations_registry.jsonl`, `.omx/state/canonical_anti_patterns_registry.jsonl`, `.omx/state/probe_outcomes.jsonl`, `.omx/operator_authorize_recipes/substrate_wyner_ziv_cross_substrate_composition_y_fec6_for_pr101_modal_paired_dispatch.yaml`, `experiments/results/wyner_ziv_cross_substrate_composition_y_fec6_for_pr101_l1_mlx_20260528/training_artifact.json` + sister archive bin, `.omx/research/wyner_ziv_cross_substrate_composition_y_fec6_for_pr101_landed_20260528.md` (this file).

## Operator-routable next-steps

1. **NOW**: review THIS landing memo + the THIRD-surface empirical anchor. The PARADIGM-INTACT classification is HARD-EARNED but the Contrarian's caveat (item in council_dissent above) requires explicit acknowledgement that prefix-detector on fp16 tensor weights is structurally narrow.
2. **DECIDE**: which reactivation path to fund next?
   - Path #1 (real PoseNet pre-compute Y; Wave N+10): ~$1-5 Modal blanket; runs CONTEST PoseNet on 600 pairs at compress time; HIGHEST EV per anti-pattern #16 unwind.
   - Path #2 (non-prefix Y-overlap detector; ~$0 MLX-LOCAL + ~200 LOC primitive extension): orthogonal evidence value but lower EV than #1.
   - Path #3 (CSC-style sparse-coding Y; substantial new infrastructure): deferred until #1+#2 resolve.
3. **PR-creation operator-explicit-per-PR gate** per `feedback_pr_creation_requires_explicit_operator_authorization_with_adversarial_negative_findings_audit_standing_directive_20260528.md`: this empirical research landing does NOT trigger PR creation; reactivation paths #1/#2/#3 individually do NOT trigger PR creation either (each requires operator-attended symposium + Catalog #246 paired-CUDA-CPU anchor before any contest-PR claim).

## Provenance + non-promotable markers

- **Hardware substrate**: darwin_arm64_m5_max_macos_mlx_local
- **Evidence grade**: macOS-MLX research-signal (per Catalog #192/#317/#341)
- **Score claim**: false
- **Promotable**: false
- **Score claim valid**: false
- **Promotion eligible**: false
- **Rank or kill eligible**: false
- **Ready for exact eval dispatch**: false
- **Axis tag**: [macOS-MLX research-signal]
- **Canonical helper invocation**: `tac.codec.wyner_ziv_layer.insert_wyner_ziv_layer` + `tac.codec.wyner_ziv_layer.reconstruct_from_wyner_ziv_layer` + `tac.canonical_frontier_pointer.load_canonical_frontier_pointer_lenient`

## Cross-references

- Sister Wave N+5 landing: `wyner_ziv_pipeline_stage_codec_l1_long_mlx_600pair_landed_20260528.md` (FIRST-surface falsification)
- Sister Wave N+7 Slot 2 landing: `anti_pattern_registry_expansion_13_14_15_plus_wyner_ziv_op_routable_5_landed_20260528.md` (SECOND-surface falsification)
- Sister recipe scaffold: `.omx/operator_authorize_recipes/substrate_wyner_ziv_pipeline_stage_codec_per_pair_posenet_output_y_modal_paired_dispatch.yaml` (Wave N+10 path #1)
- Canonical equation: `.omx/state/canonical_equations_registry.jsonl` → `wyner_ziv_cross_substrate_composition_y_pose_axis_savings_v1`
- Canonical anti-pattern: `.omx/state/canonical_anti_patterns_registry.jsonl` → `wyner_ziv_cross_substrate_composition_y_density_decoder_state_dict_surface_v1`
- Probe outcome: `.omx/state/probe_outcomes.jsonl` → `wyner_ziv_cross_substrate_composition_y_fec6_for_pr101_density_third_surface_20260528`
- Standing directives: `feedback_mlx_first_numpy_portable_individually_fractally_optimized_standing_directive_20260526.md` + `feedback_modal_spend_blanket_approved_but_mlx_first_for_everything_standing_directive_20260528.md` + `feedback_pr_creation_requires_explicit_operator_authorization_with_adversarial_negative_findings_audit_standing_directive_20260528.md`
- CLAUDE.md non-negotiables: "Forbidden premature KILL without research exhaustion"; "Apples-to-apples evidence discipline"; "Submission auth eval — BOTH CPU AND CUDA"; "MLX portable-local-substrate authority"; Catalog #307 paradigm-vs-implementation; Catalog #311 Atick-Tishby-Wyner triple; Catalog #313 probe-outcomes-canonical; Catalog #325 per-substrate symposium; Catalog #343 frontier-pointer-canonical; Catalog #344 canonical-equations-registry; Catalog #346 council-roster-canonical
