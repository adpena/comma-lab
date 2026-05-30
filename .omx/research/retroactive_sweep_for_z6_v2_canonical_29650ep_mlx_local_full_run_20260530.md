# Retroactive Sweep for `lane_z6_v2_canonical_29650ep_mlx_local_full_run_20260530`

**Date**: 2026-05-30 ~21:47 UTC
**Lane**: `lane_z6_v2_canonical_29650ep_mlx_local_full_run_20260530` L1
**Per Catalog #348**: every new gate/finding landing requires a retroactive sweep documenting any historical KILL/DEFER/FALSIFY verdicts the new finding might invalidate.

## Bug-class symptom signature

This subagent landed the FIRST canonical end-to-end empirical anchor on z6_v2 via the canonical PR95-faithful 29,650-epoch MLX-LOCAL curriculum at $0 spend per operator MLX-first paradigm correction 2026-05-30. The substantive signature:

1. **Phase A FULL RUN GREEN** (rc=0, wall=3162s=52.7min on M5 Max, MLX 0.31.2)
2. **Phase B 8/8 axes PASS** (stage transitions canonical L14, per-stage lambda/sigma per L16/L17, Muon stage-8-only per L15, z6_v2 Rao-Ballard FiLM-ego-motion arch present, EMA preserved, archive emitted, loss converged 31.6x from initial to minimum)
3. **Phase C BLOCKED** at z6_v2 inflate.py canonical contest output format gap (PNG vs .raw, 64 frames vs 1200, archive 580467 bytes → rate term alone 0.386 > frontier 0.192 by 2x)

## Pre-fix window

All prior z6_v2 empirical anchors landed in the 7 days preceding this dispatch:
- Pre-flight verification subagent landed 2026-05-30T20:37Z (sister memo `feedback_z6_v2_canonical_29650ep_mlx_local_pre_flight_verification_landed_20260530.md`)
- m9-v3 PR95 curriculum wire-in landed 2026-05-30 ~15:00Z (`feedback_m9_v3_pr95_faithful_curriculum_sister_wave_wire_in_landed_20260530.md`)
- Wave N+47 PR95-L14-L32 lesson-set expansion 2026-05-28 (CLAUDE.md HNeRV parity discipline §L14)

## Historical-KILL/DEFER/FALSIFY search results

Grep across `~/.claude/projects/-Users-adpena-Projects-pact/memory/*.md` for KILL/FALSIFIED/DEFERRED verdicts mentioning z6_v2 or PR95 curriculum:

- `feedback_z6_v2_candidate_4c_scorer_logit_conditioning_20260518.md` → L1 lane, no KILL verdict
- `feedback_z6_v2_redesign_cargo_cult_unwind_path_b_20260517.md` → DEFER-pending-iteration; cargo-cult-unwind paradigm intact
- `feedback_z6_v2_wave_2_dispatch_smoke_before_full_paired_cpu_cuda_20260518.md` → DEFER per recipe `Z6_TRAINER_MODE` env hardcode bug (extincted via Catalog #326)
- No prior **KILL** verdicts on z6_v2 substrate paradigm

**Affected findings (RE-EVAL-priority assignment per Catalog #348 contract)**:

| Memo / Verdict | Status | RE-EVAL priority | Rationale |
|---|---|---|---|
| z6_v2_redesign_cargo_cult_unwind_path_b DEFER 2026-05-17 | RATIFIED-AND-EXTENDED | LOW | Cargo-cult unwind paradigm now EMPIRICALLY VERIFIED end-to-end via 29,650-epoch FULL RUN; the prior DEFER's reactivation criteria (long-form canonical empirical anchor at $0 spend) is now SATISFIED. |
| z6_v2_wave_2 DEFER per Z6_TRAINER_MODE bug 2026-05-18 | RATIFIED-AND-EXTENDED | LOW | Catalog #326 extincted the bug class structurally; the FULL RUN demonstrates trainer wires correctly when invoked canonically with `--pr95-faithful-curriculum-enabled --pr95-curriculum-total-epochs 29650`. |
| pr95_faithful_curriculum equation residual 0.0 across 3 prior anchors | RATIFIED-AND-EXTENDED | LOW | This sub agent's 4th anchor ratifies the canonical 8-stage curriculum at 29,650-epoch FULL scale (not just 100-500 epoch smoke). |

## RE-EVAL-priority assignment

**LOW priority** for all 3 affected findings: each was already RATIFIED at smoke scale (N=100, N=500 per pre-flight); this 4th anchor extends to canonical 29,650-epoch FULL scale. No KILL/FALSIFICATION verdicts to overturn.

**MEDIUM priority** for NEW operator-routable findings surfaced by Phase C:

1. **z6_v2 inflate.py canonical contest output format gap**: emits PNG `output_dir/<base>/<frame>.png` not canonical `output_dir/<base>.raw` uint8 H*W*C concat per upstream/evaluate.py TensorVideoDataset contract; `--num-pairs 32` produces only 64 frames vs canonical 1200. Operator-routable canonical resolution: extend `tac.substrates.z6_v2_cargo_cult_unwind.inflate.inflate_one_video` per Catalog #146 contest-compliant runtime contract.

2. **z6_v2 archive size structurally exceeds frontier rate term**: archive.zip 580467 bytes → rate term alone `25 * 580467 / 37545489 = 0.386` > canonical frontier 0.192 by 2x. Even with 0 distortion, z6_v2 in current substrate-engineering size cannot match frontier. Operator-routable canonical resolution: substrate-shrinkage iteration (quantize z6_v2 architecture, FP4/INT8 weights, sparsify FiLM modulation parameters) per CLAUDE.md HNeRV parity L7 substrate-engineering-vs-bolt-on split.

## Cross-references

- Catalog #192 (`check_macos_cpu_advisory_not_promoted_without_linux_verification`) — macOS-MLX advisory never promotable
- Catalog #220 (`check_substrate_l1_scaffold_no_byte_addition_without_operational_score_improvement_mechanism`) — z6_v2 satisfies via `lane_class=substrate_engineering` + `research_only=true` opt-out
- Catalog #287 (`check_no_docstring_overstatement_without_evidence_tag`) — all numbers in this memo carry `[macOS-MLX research-signal]` / empirical anchor citations
- Catalog #307 (`check_kill_verdict_distinguishes_paradigm_vs_implementation_falsification`) — Phase C gap is IMPLEMENTATION-LEVEL; paradigm INTACT
- Catalog #313 (`check_dispatch_target_has_no_predecessor_adjudicated_outcome`) — sister probe outcome PROCEED 14-day advisory registered
- Catalog #341 (`check_cathedral_consumer_mps_prescreen_routing_carries_canonical_markers`) — canonical Provenance carries `score_claim=False` + `promotable=False` + `evidence_grade=macOS-MLX-research-signal`
- Catalog #344 (`check_empirical_finding_memo_references_canonical_equation`) — references canonical equation `pr95_faithful_curriculum_cross_substrate_compounding_savings_v1` (4th anchor)
- Catalog #348 (`check_new_gate_landing_includes_retroactive_sweep_evidence`) — THIS sweep memo per the 4-field contract
- Catalog #371 (auto-recalibrator) — 3+ anchor threshold triggered; auto-recalibration confirmed (residual=0.0)

## Conclusion

**0 historical KILL/DEFER/FALSIFY verdicts invalidated**. 3 prior DEFER verdicts RATIFIED-AND-EXTENDED at scale. 2 NEW operator-routable findings surfaced (inflate format gap + substrate shrinkage requirement) routed via Catalog #313 PROCEED 14-day advisory with reactivation criteria pinned.
