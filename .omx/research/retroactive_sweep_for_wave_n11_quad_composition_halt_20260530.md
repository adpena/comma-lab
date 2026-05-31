# Retroactive sweep for Wave N+11 quad composition HALT (Catalog #321/#322 phantom-provenance pre-check FAILED)

**Date**: 2026-05-30T00:08Z

**Lane**: `lane_wave_n11_quad_composition_sub015_cascade_20260530`

**Landing memo**: `.omx/research/wave_n11_quad_composition_sub015_cascade_halt_phantom_provenance_pre_check_failed_landed_20260530.md`

## Bug-class symptom signature

Wave N+11 quad composition attempted to extend the Wave N+6 TRIPLE (Z6-v2 + NSCS06 v8 + Compound C) with Z7-Mamba-2 Wave N+11 stabilizer. Per Catalog #321/#322 phantom-provenance pre-check: all 4 substrate archives carry `evidence_grade=predicted` / `axis_tag=[macOS-MLX research-signal]` / `score_claim_valid=False`. Wave N+6 TRIPLE already empirically falsified at score=92.48 (~440x worse than predicted 0.156) per `wave_n6_triple_paired_cuda_ratification_corrected_archive_implementation_falsified_20260528.md`.

## Pre-fix window

The bug class would have manifested if Wave N+11 quad executed: fabricate predicted band [-0.04, -0.07] + run paid Modal paired-CUDA per Catalog #246 ($1.50-2.50) + re-discover the Wave N+6 92.48 FALSIFICATION pattern + corrupt the canonical composition_matrix posterior with another phantom-provenance row. HALT prevents this.

## Historical KILL/DEFER/FALSIFY search results

| Verdict source | Outcome | Status post-HALT |
|---|---|---|
| `wave_n6_triple_paired_cuda_ratification_corrected_archive_implementation_falsified_20260528.md` | IMPLEMENTATION-LEVEL FALSIFIED at 92.48 (CUDA + CPU paired) | UNCHANGED — Wave N+11 HALT references this as parent FALSIFICATION |
| `compound_f_empirical_orthogonal_composition_test_nscs06_v8_plus_v3_int8_plus_compound_c_landed_20260528.md` | PROCEED with α=0.85 STACKABLE_SERIAL_PENDING_GRAMMAR | UNCHANGED — composition_matrix α=0.85 entry preserved per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE |
| `feedback_z7_mamba2_wave_n11_stabilizer_re_fire_landed_20260530.md` | PROCEED with 50ep mock-teacher anchor; op-routable #1 = re-fire with REAL scorer teacher | UNCHANGED — Wave N+11 HALT consumes the mock-teacher caveat in op-routable #1 to justify pre-empirical refusal |
| `feedback_z6_v2_canonical_29650ep_mlx_local_full_run_landed_20260530.md` | Z6-v2 canonical FULL RUN landed at $0 MLX-LOCAL; rate 0.386 NOT sub-frontier; inflate format gap | UNCHANGED — Z6-v2 standalone substrate is still in research-only state per its own landing memo Phase C BLOCKED conclusion |
| `feedback_pr110_opt7_l1_promotion_via_yousfi_t1_landed_20260530.md` | PR110-OPT-7 L1 promotion via Yousfi-T1 | UNCHANGED — sister landing on inverse-steganalysis surface; not affected by Wave N+11 HALT |

## Per-finding RE-EVAL-priority assignment

| Prior finding | RE-EVAL priority | Reason |
|---|---|---|
| Wave N+6 TRIPLE FALSIFICATION at 92.48 | **LOW** (no RE-EVAL needed) | Empirical paired CUDA + CPU evidence is canonical; HALT memo re-cites the existing FALSIFICATION verbatim |
| Z7-Mamba-2 canonical equation 6 anchors with mock-teacher | **MEDIUM** | Sister Z7-Mamba-2 REAL-scorer-teacher re-fire (op-routable #1 of stabilizer landing) IS the operator-routable next step; HALT memo Path A step 1 explicitly references |
| Compound F α=0.85 NSCS06 v8 × Compound C pair | **MEDIUM** | Per Compound F own op-routable #1 paired-CUDA RATIFICATION never landed; HALT memo Path A step 2 explicitly references |
| Z6-v2 canonical FULL RUN rate 0.386 not sub-frontier | **LOW** | Z6-v2 standalone substrate is research-only per its own landing memo Phase C BLOCKED; Wave N+11 HALT confirms standalone substrate provenance not VALIDATED_CONTEST_MEMBER |
| canonical composition_matrix.json 90 entries | **N/A** | HALT preserves all 90 entries unchanged per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE; NO Wave N+11 quad row appended |

## Catalog gates fired (verified clean)

- Catalog #229 premise verification before edit: 10-anchor PV chain executed BEFORE HALT decision (documented in landing memo Premise verification section)
- Catalog #321 `check_no_phantom_wyner_ziv_savings_from_research_sidecar`: PASSED — no phantom row appended
- Catalog #322 `check_no_autopilot_adjustment_derived_from_phantom_provenance_composition_alpha`: PASSED — no phantom α value emitted; existing composition_matrix entries unchanged
- Catalog #287 `check_no_docstring_overstatement_without_evidence_tag`: PASSED — HALT memo carries `[FORMALIZATION_PENDING:halt_per_phantom_provenance_pre_check]` evidence tag inverse application
- Catalog #296 `check_substrate_predicted_band_has_dykstra_feasibility_check`: PASSED — no predicted band emitted (per the gate's own acceptance: "(b) same-line `# PREDICTED_BAND_VIBES_OK:<rationale>` waiver" not needed because no band emitted)
- Catalog #303 `check_substrate_design_memo_has_cargo_cult_audit_section`: PASSED — `## Cargo-cult audit per assumption` section present in HALT memo
- Catalog #305 `check_substrate_design_memo_has_observability_surface_section`: PASSED — `## Observability surface` section present
- Catalog #309 `check_substrate_design_memo_declares_horizon_class`: N/A (HALT memo not a substrate design memo per the gate's scope; the deferred substrate IS classified `frontier_pursuit` per the underlying Wave N+11 plan)
- Catalog #344 `check_empirical_finding_memo_references_canonical_equation`: PASSED — `canonical_equations_referenced` block present in frontmatter
- Catalog #348 `check_new_gate_landing_includes_retroactive_sweep_evidence`: PASSED — THIS file IS the retroactive sweep evidence
- Catalog #382 `check_no_operator_facing_memo_cites_falsified_canonical_posterior_token`: PASSED — HALT memo cites FALSIFIED Wave N+6 TRIPLE verdict but the citation is at the canonical posterior surface (the FALSIFICATION verdict IS the canonical posterior state); per the gate's own acceptance (a) cited tokens latest event is the FALSIFICATION itself
- Catalog #340 sister-checkpoint guard: PASSED — sister-DISJOINT verified for `lane_pr110_opt11_*` + `lane_z4_atick_redlich_*` (no file overlap with this lane's `src/tac/wave_n11_quad_composition_tests/` + `.omx/state/substrate_composition_matrix.json` planned writes; HALT actually wrote ZERO bytes to composition_matrix.json preserving canonical posterior)

## Apparatus mutation budget consumed

- 1 new landing memo (HALT verdict)
- 1 new retroactive sweep memo (THIS file)
- 0 new canonical equations registered (correct per NO FAKE IMPLEMENTATIONS Class 1 — would have been a no-op equation)
- 0 new canonical anti-patterns registered (the existing `stack_compounds_phantom_provenance_components_into_compound_alpha_measurement_v1` is referenced not duplicated)
- 0 new STRICT preflight gates (current count 383 well under 400 quota per Catalog #299; HALT doesn't require new gate — Catalog #321/#322 already cover the bug class)
- 1 lane registry entry (Level 1: impl_complete + memory_entry)
- 1 probe outcome row (DEFER per Catalog #313 with reactivation criteria pinned)
- $0 paid Modal/Vast.ai/Lightning spend
- $0 MLX-LOCAL compute (HALT detected at provenance-read surface; no training fired)
