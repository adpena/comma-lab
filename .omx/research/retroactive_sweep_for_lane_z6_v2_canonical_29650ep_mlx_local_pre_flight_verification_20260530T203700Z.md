# Retroactive Sweep for lane_z6_v2_canonical_29650ep_mlx_local_pre_flight_verification_20260530

**Generated**: 2026-05-30T20:37:00Z
**Lane**: `lane_z6_v2_canonical_29650ep_mlx_local_pre_flight_verification_20260530` L1
**Per**: CLAUDE.md Catalog #348 EVENT-DRIVEN RETROACTIVE VERDICT-TAINT SWEEP non-negotiable

## Bug-class symptom signature

Pre-fix bug class would have manifested as: a substrate trainer that documents PR95 8-stage 29,650-epoch curriculum support via `--pr95-faithful-curriculum-enabled` CLI flag + claims canonical stage transitions + Muon-final-stage-only routing, but never empirically verified end-to-end on a real run (the sister-wave wire-in 49f41e22c had test-level evidence at 41/41 PASS but not multi-substrate multi-N smoke evidence; the prior PR95 curriculum MLX-LOCAL smoke validation memo 2026-05-30 closed N=100 across 3 substrates but never extended to N=500 scaling verification + wall-clock extrapolation for the 29,650-epoch FULL RUN that operator MLX-first paradigm correction 2026-05-30 unlocks at $0).

THIS lane closes that gap for the z6_v2 substrate specifically via:
- N=100 reproducibility smoke (verifying sister memo 2026-05-30 result reproduces)
- N=500 extended smoke (proving curriculum scaling stability past sister smoke endpoint)
- Two run reproducibility test (revealing MLX non-determinism with seed=0 — DOCUMENTED honestly per CLAUDE.md "NO FAKE IMPLEMENTATIONS")
- Inflate roundtrip GREEN per Catalog #146 contract
- Wall-clock extrapolation to 29,650-epoch FULL RUN (~24.7min on M5 Max at $0)

## Pre-fix window

Subagents/landings from 2026-05-26 (when m9-v3 canonical helper landed via c91481212) through 2026-05-30 (when m9-v3 sister-wave wire-in landed via 49f41e22c + PR95 curriculum MLX-LOCAL smoke validation memo landed); the gap THIS pre-flight closes is empirical evidence at multi-N (N=100 + N=500) for z6_v2 specifically AND wall-clock extrapolation for the 29,650-epoch FULL RUN that 8th MLX-first standing directive + operator MLX-first paradigm correction 2026-05-30 verbatim unlocks.

## Historical KILL / DEFER / FALSIFY search results

Searched for KILL / DEFER / FALSIFY verdicts on z6_v2 substrate OR PR95 curriculum substrate-class shift OR MLX-LOCAL 29,650-epoch FULL RUN paradigm:

**Affected findings: 0**

- `feedback_pr95_curriculum_mlx_local_smoke_validation_landed_20260530.md` — PROCEED 14-day Catalog #313 advisory; NOT INVALIDATED by THIS landing; THIS landing **RATIFIES** that sister memo's N=100 result via reproducibility test + extends to N=500 scaling verification + Phase E operator-routing recommendation.
- `feedback_m9_v3_pr95_faithful_curriculum_sister_wave_wire_in_landed_20260530.md` — PROCEED 14-day; NOT INVALIDATED; THIS landing **RATIFIES** the 3-layer architecture wired (long_training_canonical.notify_global_epoch + harness kwargs + adapter _train_step_pr95_faithful_curriculum) via end-to-end runtime verification.
- Z6-v2 cargo-cult-unwind design memo + L1 LONG-RUN MLX-LOCAL promotion 2026-05-28 — NOT INVALIDATED; THIS landing **RATIFIES** the canonical mlx_score_aware harness + Z6V2 substrate-distinguishing primitives (Rao-Ballard FiLM-ego-motion + FoE + Atick-Redlich) via end-to-end PR95 curriculum composition verification.

## Per-finding RE-EVAL-priority assignment

None — 0 affected findings.

## Catalog #348 compliance verdict

CLEAN — no historical findings invalidated; THIS landing is RATIFYING-AND-EXTENDING (not falsifying); per CLAUDE.md "Forbidden premature KILL without research exhaustion" the canonical PR95 curriculum + z6_v2 substrate paradigm BOTH remain INTACT.

## Sister DISJOINT verification per Catalog #340

Checked `python tools/subagent_checkpoint.py read --latest-incomplete` at session start; in-flight sister subagent `z8_m12a_canonical_29650ep_mlx_local_pre_flight_verification_20260530T202732Z` writes to `experiments/results/z8_m12a_canonical_29650ep_mlx_local_pre_flight_verification_20260530T202732Z/` — DISJOINT from THIS subagent's output dir `experiments/results/z6_v2_canonical_29650ep_mlx_local_pre_flight_verification_20260530T203216Z/`. Different substrate (z6_v2 vs z8) + different trainer file + different output dir.

## Cross-references

- Lane: `lane_z6_v2_canonical_29650ep_mlx_local_pre_flight_verification_20260530`
- Probe outcome: `z6_v2_canonical_29650ep_mlx_local_pre_flight_verification_20260530` PROCEED 14-day advisory expires 2026-06-13T20:36:27Z
- Canonical equation: `pr95_faithful_curriculum_cross_substrate_compounding_savings_v1` anchors 2→3 (triggers Catalog #371 auto-recalibration on 3+ anchor threshold)
- Sister memos: `feedback_pr95_curriculum_mlx_local_smoke_validation_landed_20260530.md` + `feedback_m9_v3_pr95_faithful_curriculum_sister_wave_wire_in_landed_20260530.md`
- Output dir: `experiments/results/z6_v2_canonical_29650ep_mlx_local_pre_flight_verification_20260530T203216Z/`
