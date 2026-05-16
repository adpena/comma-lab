# L5 v2 Architecture-Lock Packet Hardening

- date: `2026-05-16`
- agent: `codex`
- scope: Time-Traveler L5 v2 staircase, architecture-lock authority, side-info effect-curve custody
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`

## Why

The L5 v2 staircase had stricter internal booleans, but no single durable
operator-facing packet answered: "is TT5L architecture lock allowed right now?"
That creates an authority gap. A later operator or subagent could see partial
readiness, skip the side-info curve or first-anchor timing custody, and lock a
staircase architecture from incomplete evidence.

This landing makes the no-lock state explicit and reviewable. It also wires the
side-info effect-curve producer into the TT5L next-action command path, so the
missing blocker has a concrete, byte-closed artifact producer instead of only a
verbal TODO.

## Landed

- `tools/build_l5_v2_architecture_lock_packet.py` writes the current lock/no-lock
  packet JSON and Markdown report.
- `tac.optimization.l5_staircase_v2.l5_v2_architecture_lock_packet()` requires
  all gate evidence, Dykstra score-axis sanity, move-level feasibility,
  side-info gate evidence, probe gate evidence, paired-axis plan evidence,
  paired CPU/CUDA side-info effect curve, first-anchor timing smoke, and
  exact/diagnostic anchor-pair custody.
- `tools/operator_briefing.py` now surfaces the packet artifact path, report
  path, boolean, and blockers.
- `tools/build_l5_v2_sideinfo_effect_curve.py` is referenced by the TT5L
  next-action command template, preventing the side-info curve from remaining a
  hidden missing artifact.

The generated packet is intentionally a no-lock packet:

- `architecture_lock_allowed=false`
- blockers include missing probe gate evidence, paired-axis plan, paired
  CPU/CUDA side-info curve, first-anchor timing smoke, and exact/diagnostic
  anchor-pair custody
- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`

## Current Artifacts

- `.omx/research/l5_v2_architecture_lock_packet_20260516_codex.json`
- `.omx/research/l5_v2_architecture_lock_packet_20260516_codex.md`
- `.omx/research/l5_v2_sideinfo_effect_curve_producer_20260516_codex.md`

## Verification

- `.venv/bin/python tools/build_l5_v2_architecture_lock_packet.py`
- `.venv/bin/python -m pytest src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_architecture_lock_packet_requires_timing_and_anchor src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_architecture_lock_packet_allows_only_after_full_custody src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_architecture_lock_packet_cli_writes_no_lock_packet -q` -> `3 passed`
- `.venv/bin/python -m ruff check src/tac/optimization/l5_staircase_v2.py src/tac/tests/test_l5_staircase_v2.py tools/operator_briefing.py tools/build_l5_v2_architecture_lock_packet.py` -> clean
- `.venv/bin/python -m pytest src/tac/tests/test_l5_v2_sideinfo_effect_curve.py src/tac/tests/test_l5_v2_measurement_schedule.py src/tac/tests/test_l5_staircase_v2.py src/tac/tests/test_operator_briefing.py -q` -> `128 passed`
- `.venv/bin/python -m ruff check src/tac/optimization/l5_staircase_v2.py src/tac/optimization/l5_v2_measurement_schedule.py src/tac/optimization/l5_v2_sideinfo_effect_curve.py src/tac/tests/test_l5_staircase_v2.py src/tac/tests/test_l5_v2_sideinfo_effect_curve.py tools/operator_briefing.py tools/build_l5_v2_architecture_lock_packet.py tools/build_l5_v2_sideinfo_effect_curve.py` -> clean
- `git diff --check` -> clean

## Next Non-PR106 L5 Action

Close the no-lock blockers in order:

1. Populate and audit paired C1/Z5/TT5L probe observations.
2. Build the paired CPU/CUDA side-info effect curve across zero, random_lsb,
   shuffled, trained, and ablated variants.
3. Run the TT5L first-anchor timing smoke with custody.
4. Attach exact or diagnostic anchor-pair evidence.

Only after the packet flips true should TT5L architecture lock be treated as
allowed.
