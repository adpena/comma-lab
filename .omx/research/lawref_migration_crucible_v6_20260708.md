# LawRef migration of crucible_v6 constants (#351 follow-up) — landing memo

STORES CONSULTED: .omx/research/lawref_constant_compiler_351_20260708.md (migration plan) · src/tac/witness_dsl/{lawref,lawref_builtins}.py + src/tac/canonical_equations/evaluators.py (mechanism) · src/tac/witness_autoconfig.py::derive_crucible_v6_config (target, at v6.4) · tools/launch_witness_run.py (hookup) · .omx/research/t5_crucible/{ORCHESTRATION_LEDGER.md (reqs P/Q/T), DRAFT_OPTIMAL_STACK_v6_20260708.md (constants' laws + provenance)} · probe artifact JSONs (tau-confirm, wave-A trace, P-CON).

**Task (T5 CRUCIBLE, LAWREF MIGRATION agent):** migrate `derive_crucible_v6_config`'s rot-prone
constants to LawRefs (the #351 constant-compiler) + hook the launcher so run-1 ships
`constants_manifest.json`. **VALUE-IDENTITY IS THE LAW:** the migration changes ZERO emitted
values — bit-match tests are the acceptance bar. `[no-triality]` md; code commits normal treatment.

Canonical equation ids registered/consumed here (evaluators.py): `tau_end_knee_launch_v1`,
`hosc_beta_fireband_pin_v1`, `lr_control_denominator_v1`, `lr_hold_frac_no_hold_v1`,
`settle_window_v1`, `tail_cycle_floor_v1`, `conley_absolute_bar_v1`,
`adaptive_eps_saturation_alarm_v1` (+ the existing `forfeit_matched_exit_v1` reused for s*).

## Scope decision (stated — the honest boundary)

`derive_crucible_v6_config` **actually emits** exactly `_CRUCIBLE_V6_DELTAS`. Of the migration-plan
laws {τ_end · ν-derived · β_hosc pin · adaptive-ε clamps · s_fit · LR pin}, the ones the variant
**hardcodes-and-emits** are **τ_end 0.31, β-pin 10.0, LR pin (1000 / 1.0)** → these 4 are the
**CONSUMED** set: migrated to LawRefs, resolved at config time, manifest-recorded, and the emitted
launch.sh stays byte-identical. The ν-family window laws (settle 237.09 / tail-cycle 387.09), the
P-CON absolute persistence bar (s_fit 1.7505 / 1.3018 logit), and the adaptive-ε saturation alarm
(0.7) are **NOT emitted by this variant** — they are the schedule-derivation / control-law
machinery on other surfaces. Wiring a non-emitted flag into the argv would BREAK value-identity, so
those are built as **LIBRARY** LawRefs (the req-T single-source-of-truth; bit-match tested against
their real artifact/draft values) but are NEVER wired into the argv. s* reuses the existing
`forfeit_matched_exit_v1` builtin (no rebuild).

## What landed

1. **Evaluators** (`src/tac/canonical_equations/evaluators.py`): 8 new uniform
   `evaluator(inputs)->value`. TYPE DISCIPLINE: passthrough evaluators return the input UNCOERCED
   (`str(1000) != str(1000.0)` would break launch.sh byte-identity); only settle/tail-cycle compute
   a float (matching float anchors).
2. **Built-in LawRefs** (`src/tac/witness_dsl/lawref_builtins.py`): CONSUMED trio + LR-hold +
   LIBRARY ν-family/persistence-bar/adaptive-ε. `CRUCIBLE_V6_CONSUMED_LAWREFS` (keyed by the
   `_CRUCIBLE_V6_DELTAS` key) + `CRUCIBLE_V6_CONSUMED_TARGET_TAGS` ({schedule: mod32cap}).
   τ_end is a `measured_anchor` reading `launch_tau` (0.31) from `tau_knee_ptau2_20260708.json`
   with a `fallback=0.31` (waiver) so a missing artifact NEVER blocks the launch. β/LR are
   `derived_at_config` literal pins (design constants; laws in provenance). τ_end carries
   `config_tags={schedule: mod32cap}` → a mod48 vehicle fails closed (P-CT1 protection).
3. **Variant migration** (`src/tac/witness_autoconfig.py`): `derive_crucible_v6_config` resolves the
   CONSUMED LawRefs (`resolve_flag_dict_constants`) into a manifest, **asserts each resolved value
   bit-matches the sealed literal (value AND type) — fail CLOSED on drift**, overlays them into
   `d6 = {**_CRUCIBLE_V6_DELTAS, **resolved}`, and stores `crucible_v6_deltas=d6` +
   `constants_manifest` on the config (two new provenance-only fields; never emitted). The
   `_sealed_205_flags` crucible trailing block now reads `self.crucible_v6_deltas or
   _CRUCIBLE_V6_DELTAS`, so the LR-pin tokens are genuinely consumed from the resolver.
4. **Launcher hookup** (`tools/launch_witness_run.py`): `write_constants_manifest(cfg, out_dir)`
   writes `constants_manifest.json` beside launch.sh (schema `constants_manifest.v1`; only when the
   config carries compiled constants → no file for non-crucible paths, byte-and-file-identical).

## Value-identity proofs (the acceptance bar)

- **Per-constant bit-match + type** (`test_crucible_v6_consumed_bitmatch_and_type_per_constant`):
  softmax_temp_end 0.31 · hosc_beta_end 10.0 · lr_anneal_epochs 1000 (int) · lr_hold_frac 1.0.
- **launch.sh byte-identity**: `to_command` of the LawRef path == the pure-literal path
  (`test_crucible_v6_lawref_resolved_path_is_byte_identical_to_literal_path`) AND, verified
  interactively, == **git HEAD's** `derive_crucible_v6_config().to_command()` byte-for-byte.
- **manifest schema + completeness** + fallback-with-waiver path + config-conditionality fail-closed
  + value-identity guard raises on drift + non-crucible paths carry empty migration fields.
- Full suite: `test_lawref_constant_compiler.py` + `test_witness_autoconfig.py` = **100 passed**;
  ruff F clean. (Pre-existing, unrelated: `test_check_344_...` fails on a 500-memo repo backlog —
  confirmed identical with my changes stashed.)

## Per-constant bit-match table

| delta key | value | equation_id | ladder | tier |
|---|---|---|---|---|
| softmax_temp_end | 0.31 | tau_end_knee_launch_v1 | measured_anchor (artifact launch_tau) | CONSUMED |
| hosc_beta_end | 10.0 | hosc_beta_fireband_pin_v1 | derived_at_config (linear-replica pin) | CONSUMED |
| lr_anneal_epochs | 1000 (int) | lr_control_denominator_v1 | derived_at_config (control den) | CONSUMED |
| lr_hold_frac | 1.0 | lr_hold_frac_no_hold_v1 | derived_at_config (no hold) | CONSUMED |
| settle (237.09) | 3/ν | settle_window_v1 | measured_anchor | LIBRARY |
| tail-cycle (387.09) | 3/ν+150 | tail_cycle_floor_v1 | measured_anchor | LIBRARY |
| s_fit (1.7505 / 1.3018) | logit bar | conley_absolute_bar_v1 | measured_anchor | LIBRARY |
| adaptive-ε alarm (0.7) | ε_raw clamp | adaptive_eps_saturation_alarm_v1 | derived_at_config | LIBRARY |

means != ends: this is apparatus (a MEANS). Only a byte-closed n600 exact row < 0.19110 moves the pointer.
