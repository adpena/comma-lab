# crucible_v7 LAUNCH-PATH FIX — seal v7 r1 BLOCKER + MAJOR + MINOR closed

Fixes the three findings in `.omx/research/t5_crucible/seal_v7_r1_bugs_20260708.md` (commit cc66ce473).
[no-triality] — this is a launch-path wiring/seam fix; the v7 config itself already lands its lever→DSL
+ finding→equations treatment. Code carries the triality (the config is a requirement-V-native
TypedWitnessConfig; this fix only makes it REACHABLE by the ONE launch path).

means != ends: this makes v7 LAUNCHABLE. The pointer (0.19110) moves ONLY on a byte-closed n600 exact
row < 0.19110. This is a MEANS.

## STORES CONSULTED
- The seal: `.omx/research/t5_crucible/seal_v7_r1_bugs_20260708.md` (BLOCKER #1 / MAJOR #2 / MINOR #3).
- Live source: `tools/launch_witness_run.py` (`derive_named_config` 694-, `config_family` 363-,
  `_identity_header` 384-, `_emitted_flag_names` 546, gate chain b-perf 1154 / b0.5 1182 / b0.6 1226,
  `dsl_config_gate_action` 585-, argparse `--config` choices 895-), `src/tac/witness_autoconfig.py`
  (v7 builders `_build_crucible_v7` 1954- / `compile_crucible_v7_config` 2109- / `CrucibleV7Compiled`
  1788- / `_crucible_v7_schedule_governance` 1856-), `src/tac/witness_dsl/typed_config.py`
  (`TypedWitnessConfig` 365-, `to_command`/`to_trainer_flags` 549-588, `verify_launch_manifest` 626-,
  `build_launch_manifest` 604-), `tools/schedule_provenance_gate.py` (`classify_launch` 279-,
  `validate_governance_entry` 201- requires `isinstance(entry, dict)`; registries `schedule_when_flags`
  = `-start-epoch$`, `event_start_flags` = `-start-event$`).
- CLAUDE.md: NO-FAKE (a config that LOOKS launchable but runs proven_base is a fake), "Operator gates
  must be wired and used", §"Off is a tracked queue" (observability/audit-table truthfulness),
  Recursive-adversarial-review axis-9 (measured-runnability: I ran the REAL dry-run, not reasoned it).

## THE FALL-THROUGH ANATOMY (BLOCKER #1)
`derive_named_config(config, ...)` had explicit branches for sealed_205 / store_nothing_205 /
fresh_seeded / crucible_v6, then a bare `return wac.derive_config(..., all_levers=(config=="all_levers"))`.
That final line is a SILENT CATCH-ALL: it handled the two legitimate names (proven_base, all_levers)
AND every unmapped name identically — an unmapped name got a proven_base `WitnessConfig`, and
`config_family` (which keys off selector fields) then stamped it **proven_base**. `--config crucible_v7`
was ALSO rejected one layer up (argparse `choices` omitted it). So v7 was doubly unreachable, and the
fall-through is a bug CLASS on its own: any future config name typo → a silent proven_base run.

Fix = three parts:
1. **argparse choices** += `crucible_v7` (+ help text) — the token is now accepted.
2. **explicit `crucible_v7` branch** → `wac.compile_crucible_v7_config(...).to_launch_config()`.
3. **fail-LOUD default** — only `proven_base`/`all_levers` ride `derive_config`; ANY OTHER name RAISES
   `ValueError("unknown config name ...")`. The silent catch-all is dead; a new config MUST add a branch.

## THE PROTOCOL-SPLIT (MAJOR #2) + THE ADAPTER
The launcher's duck-typed cfg protocol needs ALL of: `to_command` / `to_trainer_flags` / `name` /
`purpose` / `wall_clock_budget_days` / `dsl_levers` (emit + identity) AND `constants_manifest` (b0.5
manifest_keys + `write_constants_manifest`) / `dsl_program_manifest` (b0.6 `verify_launch_manifest`) /
`schedule_governance` as a DICT (b0.5 `classify_launch` does `isinstance(entry, dict)`). But:
- a bare `TypedWitnessConfig` has the emit adapters + `name` + a `schedule_governance` of PYDANTIC
  objects (not dicts) and NO `dsl_program_manifest`/`constants_manifest` → b0.6 degrades to WARN-no-op
  (the v7 provenance gate INERT on the launched object) and b0.5 sees non-dict governance entries;
- the `CrucibleV7Compiled` has the manifests + governance-dict but NO emit adapters/`name`.

Fix = `CrucibleV7LaunchConfig` (frozen dataclass, `witness_autoconfig.py`) + `CrucibleV7Compiled.
to_launch_config()`. The adapter DELEGATES the emit surface to the typed config (argv SoT, ~17x
perf-env prefix intact) and CARRIES the compiled artifact's three manifests. One object now satisfies
BOTH halves → b0.6 VERIFIES for real (present + argv fingerprint matches), b0.5 gets dict governance,
`write_constants_manifest` emits the v6-inherited LawRef manifest beside launch.sh.
Not wired (by design, documented in the adapter docstring): `--dsl-lever`/`--purpose` CLI overrides —
v7 AUTHORS its lever set + purpose; passing them refuses at the `dataclasses.replace` seam rather than
silently emitting an un-composed lever (loud > silent-lie).

## MINOR #3 — governance table now describes launch.sh reality
`--tau-octave-max-dwell` was declared a FAIL_SAFE_CAP row but is NEVER emitted (the trainer DERIVES it
via `derive_octave_max_dwell(anneal, N)` when unset; it matches neither `-start-epoch$` nor
`-start-event$`, so `classify_launch` never touches it). `ScheduleGovernance.class` is constrained to
{event, cap} — there is no "derived-internal" class to re-tag it into. So the truthful fix (the PREFERRED
goal: "governance table describes launch.sh reality") is to REMOVE the phantom row and fold its
provenance into the `--tau-advance-mode` EVENT rationale (dwell = TRAINER-DERIVED-INTERNAL, provenance
in `derive_octave_max_dwell`). Emitting a hardcoded value instead would defeat the runtime derivation
(value-provenance regression). Now every `schedule_governance` KEY is an emitted launch token.

## DRY-RUN GATE-CHAIN TRANSCRIPT (measured, `--config crucible_v7 --num-pairs 8 --epochs 3000 --dry-run`)
```
# perf-env guard: launch.sh carries ['TAC_MLX_CUSTOM_GROUPED_BACKWARD'] (~17x fast path emitted).   [b-perf]
# schedule-provenance gate: classifying emitted --*-start-epoch triggers ... (0 naked; no REFUSE)    [b0.5]
# dsl-config gate: OK — DSL-authored ('crucible_v7', 134 flags, typed-validated)                     [b0.6 VERIFIED]
# wrote constants_manifest.json  (v6-inherited LawRef constants)                                     [b0]
# tac-config-family: crucible_v7   (in launch.sh)                                                    [config_family]
# DRY-RUN: launch.sh written + flags validated; NOT spawning.                                        [rc=0]
```
(system-admission REFUSED as ADVISORY-only in dry-run because the machine had concurrent jobs; it does
not block a dry-run — rc stayed 0. Wall-clock budget is DERIVED = 7.427 days at epochs=3000, positive,
so the L45 gate has a real budget to check; the gate itself runs only on a real launch after the bench.)

## TESTS — `src/tac/tests/test_launch_witness_crucible_v7_resolution.py` (12, all pass)
name-resolution → `CrucibleV7LaunchConfig` (not proven_base) · config_family label · unknown-name
RAISES · proven_base/all_levers still fall through · real-argparse rejects unknown choice · adapter
full-protocol surface · b0.6 `verify_launch_manifest` VERIFIES (+ gate action=="ok" under enforce) ·
b0.5 0-naked on the adapter · wall-clock budget derived+positive · MINOR: dwell NOT a governance key +
every governance key is emitted · full `main([... --dry-run])` gate chain rc==0 with the transcript above.
Regression suites green: test_crucible_v7_config (27) · test_launch_witness_run (51) ·
test_schedule_provenance_gate (17) · test_tau_advance_self_paced (unchanged).

## SIBLING SCOPE
The working tree carried a concurrent mod-dim/persistence/verdict-reclaim workstream (train_levelset*,
event_wirings, tail_cycles, tau_advance, curriculum_dsl, persistence_topology_loss, + several ?? files).
NOT mine — NOT committed/reverted. `git diff HEAD` on my two edited files (`launch_witness_run.py`,
`witness_autoconfig.py`) shows ONLY my crucible_v7 hunks; committed via serializer, named-files only,
post-edit shas.
