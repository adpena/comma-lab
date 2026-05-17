# L5-v2 TT5L Custody and No-op Hardening Landed

Date: 2026-05-17
Status: landed locally; pending commit/push
Scope: L5-v2 side-info effect-curve validator + TT5L side-info variant packet builder

## Why this patch exists

The preserved read-only audit
`.omx/research/l5_v2_tt5l_variant_builder_readonly_audit_preserved_20260517_codex.md`
found two concrete false-authority gaps:

1. `validate_l5_v2_sideinfo_effect_curve()` could accept custody-light
   hand-written curves that had pair identity and side-info liveness but no
   exact-eval custody fields.
2. `build_tt5l_sideinfo_variant_packets()` recorded section hashes but did not
   block expected-changing variants when archive/member/side-info-section bytes
   were unchanged from the source.

Both gaps are L5-v2 architecture-lock hazards. They can make a side-info effect
curve look complete before the evidence is byte-closed, axis-labelled, and
contest-custodied.

## Changes

- `src/tac/optimization/l5_v2_measurement_schedule.py`
  - Added per-cell exact-eval custody validation through
    `tac.exact_eval_custody.validate_exact_eval_evidence`.
  - Required artifact path, log path, hardware, inflate/eval device, auth-eval
    command, sample count, component fields, raw-output aggregate SHA, and
    inflated-output manifest custody before a curve can advance the schedule.
  - Threaded repo-root custody context into the validator so referenced
    artifact/log/manifest files must exist under the repository root and the
    manifest SHA/aggregate SHA must match.

- `src/tac/optimization/l5_v2_sideinfo_effect_curve.py`
  - Preserved hardware, inflate device, eval device, and auth-eval command in
    normalized observed cells so builder-produced exact-eval cells still satisfy
    the shared validator.

- `src/tac/optimization/l5_staircase_v2.py`
  - Passed repo-root context into loaded side-info effect-curve validation so
    hand-written public artifacts cannot bypass file custody.

- `src/tac/optimization/tt5l_sideinfo_variant_packets.py`
  - Added generation timestamp and per-variant generation rule/seed/source
    metadata.
  - Added source archive/member/side-info-section SHA fields per variant.
  - Added changed-from-source booleans for archive SHA, archive member SHA,
    side-info section SHA, and decoded side-info array.
  - Added blockers for expected-changing variants whose source-to-variant byte
    changes collapse to no-op archive/member/side-info-section bytes.
  - Expanded the operator-facing markdown report with member/side-section
    changed flags and generation rules.

- `src/tac/tests/test_l5_v2_measurement_schedule.py`
  - Added a negative custody-light side-info effect-curve test.
  - Updated positive fixtures to create real artifact/log/manifest files under
    `tmp_path`, proving the stricter validator is not satisfied by bare strings.

- `src/tac/tests/test_tt5l_sideinfo_variant_packets.py`
  - Extended byte-closure assertions to verify recorded section offsets,
    lengths, and SHAs against the actual inner TT5L member bytes.
  - Added a monkeypatched no-op replacement regression proving expected
    variants block unchanged archive/member/side-info-section bytes.

- `src/tac/tests/test_l5_staircase_v2.py`
  - Updated L5 staircase fixtures so valid side-info effect-curve artifacts
    carry the same custody fields now required by the shared validator.

- `.omx/research/l5_v2_architecture_lock_packet_20260516_codex.json`
  - Regenerated after the stricter validator. The live packet now surfaces the
    missing custody fields in the current side-info effect-curve artifact as
    blockers instead of silently accepting them.

- `tools/build_l5_v2_lattice_measurement_schedule.py`
  - Added `--repo-root` and passes it into schedule construction so CLI-loaded
    side-info effect curves use the same custody rules as in-process callers.

## Verification

- `.venv/bin/python -m pytest -q src/tac/tests/test_l5_v2_measurement_schedule.py src/tac/tests/test_l5_v2_sideinfo_effect_curve.py src/tac/tests/test_l5_staircase_v2.py src/tac/tests/test_tt5l_sideinfo_variant_packets.py`
  - Result: `139 passed`
- Broader L5/TT5L slice:
  - `.venv/bin/python -m pytest -q src/tac/tests/test_l5_v2_sideinfo_effect_curve.py src/tac/tests/test_l5_v2_probe_intake.py src/tac/tests/test_prove_tt5l_move_level_feasibility.py src/tac/tests/test_tt5l_sideinfo_variant_packets.py src/tac/tests/test_build_tt5l_move_level_feasibility_artifact.py src/tac/tests/test_l5_v2_probe_disambiguator.py src/tac/tests/test_smoke_time_traveler_l5_autonomy.py src/tac/tests/test_time_traveler_l5_z6_remote_driver.py src/tac/tests/test_l5_v2_paired_measurement_dispatch_plan.py src/tac/tests/test_train_time_traveler_l5_z6_full_main_lift.py src/tac/tests/test_remote_lane_time_traveler_l5_script.py src/tac/tests/test_l5_v2_measurement_schedule.py src/tac/tests/test_l5_staircase_v2.py src/tac/tests/test_build_tt5l_first_anchor_timing_smoke_artifact.py src/tac/tests/test_train_time_traveler_full_cpu_mode.py src/tac/tests/test_l5_paper_fidelity_claim_hygiene.py`
  - Result: `281 passed, 1 skipped`
- Adversarial review follow-up:
  - Agent `019e33f0-35ab-7052-aeb1-275dd452a333` found that the first patch
    still accepted non-existent path strings because it did not pass
    `artifact_base_dir` into `validate_exact_eval_evidence`.
  - Follow-up patch added repo-root propagation through the public validator,
    schedule builder, CLI, and L5 staircase loader.
- `.venv/bin/python -m ruff check src/tac/optimization/l5_v2_measurement_schedule.py src/tac/optimization/l5_v2_sideinfo_effect_curve.py src/tac/optimization/tt5l_sideinfo_variant_packets.py src/tac/tests/test_l5_v2_measurement_schedule.py src/tac/tests/test_l5_v2_sideinfo_effect_curve.py src/tac/tests/test_l5_staircase_v2.py src/tac/tests/test_tt5l_sideinfo_variant_packets.py`
  - Result: `All checks passed`
- `.venv/bin/python -m py_compile src/tac/optimization/l5_v2_measurement_schedule.py src/tac/optimization/l5_v2_sideinfo_effect_curve.py src/tac/optimization/tt5l_sideinfo_variant_packets.py src/tac/optimization/l5_staircase_v2.py`
  - Result: passed

## Remaining blockers

- This patch does not create new exact-eval cells or score claims.
- The live L5-v2 architecture-lock packet remains blocked until the side-info
  effect-curve artifact is rebuilt from paired CPU/CUDA exact-eval cells with
  the required custody fields.
- Two unrelated lattice-coordinate files appeared in the worktree during this
  pass:
  - `src/tac/lattice_state_ledger.py`
  - `tools/check_lattice_coordinate.py`

They appear to belong to a separate coherence-audit landing
(`lane_coherence_audit_lattice_coordinate_assignment_20260516`). They were not
edited or validated in this L5-v2 hardening patch and should be reviewed as a
separate partner-work landing.
