# DSL-only typed config — requirement V (#353) landing

**Operator 2026-07-08 (verbatim):** "The config must be defined in the DSL — no ad hoc or hand
crafting. We may need to use pydantic and possibly verification and validation tools as well and
type checking and formalization and integrate all with apparatus to prevent more dumbass bullshit."
Charter amendment (operator): use Astral **ty**, NOT mypy.

**STORES CONSULTED:** memories `config_must_be_dsl_defined_typed_validated_no_adhoc_20260708`
(the spec) + `value_provenance_ladder_no_bare_constants_20260708` + `elementwise_audits_launder_
structural_cargocult_pr95_skeleton_20260709`; CLAUDE.md triality "DSL HOLDS every designed lever" +
config-orphan confound + NO-FAKE; the LANDED sibling `schedule_provenance_gate` (commit 31e760120 —
launcher b0.5 rc=6, drift leg, `schedule_governance` surface) + coordinator messages; source:
`curriculum_dsl.WitnessProgram` (the emitter SoT), `witness_autoconfig.derive_crucible_v6_config`
(the migrated seam), `confound_gates.py` + `triality_drift_detector.py` (apparatus patterns),
`[tool.ty]` in pyproject.toml.

## Recovery provenance
This landing was **finished by a RECOVERY agent** (Opus) after the original builder died on a session
limit with ~82 tool-uses of UNCOMMITTED work in the tree. Verdict discipline: everything below was
**re-verified against the artifacts** (tests run, argv byte-identity re-proven empirically, ty re-run,
diffs re-read as a hostile reviewer), not assumed from the predecessor's intent.
`recovery-written, fresh-eyes-reviewed(1)`. Sibling files
(`train_levelset_witness_realized_through_R_mlx.py`, `train_witness_realized_through_R_mlx.py`,
`curriculum_dsl.py`'s `SegFormUnifyTau` factory, `test_seg_form_unify_tau.py`) belong to a DIFFERENT
recovery and are LEFT UNTOUCHED — `curriculum_dsl.py` is NOT committed by this landing (typed_config
imports only its pre-existing symbols).

## Completion matrix (charter item → state, verified)
| # | Charter item | State | Verification |
|---|---|---|---|
| 1 | typed pydantic-v2 schema (extra=forbid, Provenanced+waiver, to_program adapter) | **BUILT** | 34 schema tests pass; typed_config.py ty-clean |
| 2 | crucible_v6 migrated under byte-identity | **BUILT** | argv byte-identical with/without manifest (empirical); design-doc test untouched+passing |
| 3 | program manifest + launcher additive check b0.6 rc=7 + escape hatch | **BUILT** | 10 gate-decision tests; crucible dry-run prints `gate: OK`; rc distinct from b0.5 rc=6 |
| 4 | apparatus: drift leg + STRICT preflight #403 + ty script | **BUILT** (ty script FIXED) | 13 apparatus tests; #403 in CONFOUND_GATES warn-only, live-count 4 |
| 5 | schedule_governance modeled first-class | **BUILT** | ScheduleGovernance field; 7 governance tests |

**Inherited-and-verified:** items 1,2,3,5 + the drift leg + #403 gate — all built by the predecessor,
re-verified here. **Fixed in recovery:** the ty script (`typecheck_witness_dsl.sh`) was CWD-fragile —
`ty` discovers `[tool.ty]` by walking up from CWD and resolves include-globs relative to that root, so
the absolute-path invocation intermittently emitted `WARN No python files found` and checked nothing. Now
`cd "$ROOT"` + a relative path; verified it checks reliably from an arbitrary CWD.

## What landed (4 layers)

1. **Typed pydantic v2 schema — `src/tac/witness_dsl/typed_config.py`** (NEW). `TypedWitnessConfig`
   (`extra="forbid"` everywhere = anti-invented-flag at the schema layer) models the WitnessProgram-
   authoring surface: program meta (out_dir/gt_cache/num_pairs>0/epochs>0/mlx_device∈{gpu,cpu}/seed/
   purpose) · schedule (`TypedAnneal`/`TypedStage`{flag+epoch paired, ge=0}/`TypedRegularizer`) ·
   `TypedLever` · the `schedule_governance` block as a FIRST-CLASS typed field (`ScheduleGovernance`
   {class∈{event,cap}, sensor, rationale}; cap/event both require a sensor + non-empty rationale). Every
   VALUE knob is a `Provenanced` wrapper: value(scalar-only) + ladder class ∈ {DERIVED-LIVE, DERIVED-
   AT-CONFIG, MEASURED-ANCHOR, HARDCODED-WITH-WAIVER} + unit + source + waiver — HARDCODED REQUIRES a
   non-empty waiver (fail-closed); a waiver on a non-hardcoded value is refused. Adapter
   `to_program() -> WitnessProgram` (does NOT rewrite the DSL — constructs it; DSL `validate` is the
   never-invent-flags authority). `program_manifest()` + module helpers `build_launch_manifest` /
   `verify_launch_manifest` emit + check the DSL-provenance attestation.

2. **Autoconfig seam migrated (byte-identity preserved) — `witness_autoconfig.py`.** Added the additive,
   argv-INERT field `dsl_program_manifest`. `derive_crucible_v6_config` now routes through
   `_attach_dsl_program_manifest`: it CONSTRUCTS + typed-validates a `TypedWitnessConfig` of the
   crucible's DSL-authorable schedule/curriculum knobs (each with its ladder class) and FAILS CLOSED on
   any `WitnessProgram.validate` violation, then attaches a `build_launch_manifest` whose flag
   fingerprint is taken from the config's ACTUAL emitted argv. **Byte-identity:** emission is UNTOUCHED,
   so the crucible argv is byte-identical — proven by the pre-existing byte-identity gates in
   `test_witness_autoconfig.py` / `test_lawref_constant_compiler.py` (191 pass unchanged) + new argv-
   stability + no-manifest-token-in-argv tests. Non-crucible configs carry an EMPTY manifest (unchanged).

3. **Launcher enforcement — `tools/launch_witness_run.py`** (additive step b0.6, AFTER the sibling's
   schedule-provenance gate b0.5). Pure decision helper `dsl_config_gate_action` (unit-tested): OK when
   a valid DSL manifest's fingerprint matches the emitted argv; REFUSE (rc=7) a manifest present-but-
   TAMPERED; WARN on an ABSENT manifest (migration queue) unless `--enforce-dsl-config-gate`; escape
   hatch `--allow-non-dsl-config "<rationale>"` stamps `dsl_config_override.txt` LOUDLY; advisory on
   `--dry-run`/`--skip-dsl-config-gate`; fail-OPEN on infra error. rc=7 is distinct from the sibling's
   rc=6. Verified: crucible dry-run prints `# dsl-config gate: OK — DSL-authored ('crucible_v6', 106
   flags, typed-validated)`.

4. **Apparatus.** (a) Drift leg in `triality_drift_detector.py` (`dsl_config_bypass_violations`, fail-
   open, window-granular, waiver `# DSL_CONFIG_BYPASS_OK:<rationale>`): flags a NEW `derive_*_config`/
   `*_flags` emitter added OUTSIDE `witness_dsl` in a window that does not cite the typed layer. (b) ty
   target `tools/typecheck_witness_dsl.sh` (POSIX sh, `ty check src/tac/witness_dsl`, extends
   `[tool.ty]`; NO mypy). (c) STRICT preflight gate `check_launch_config_authored_in_dsl` (Catalog #403)
   in `confound_gates.py`, auto-wired via `CONFOUND_GATES` (WARN-ONLY), waiver
   `# DSL_CONFIG_AUTHORING_OK:<rationale>`.

## Byte-identity result
Crucible v6 emitted argv is BYTE-IDENTICAL to the pre-#353 form (emission untouched; the added field +
gate are argv-inert). Guaranteed by the pre-existing crucible byte-identity gates (all pass) + new
stability/inertness tests. `test_crucible_v6_schedule_matches_design_doc` passes untouched.

## Tests: 69 new (34 schema · 12 migration · 10 launcher-gate · 13 apparatus) + confound-registry
tests updated (7→8 gates, live-count bound 4). Directly-affected sweep (typed_config + migration +
launcher-gate + apparatus + confound_gates + schedule_provenance_gate + witness_autoconfig): **241 pass**.
Broader crucible/DSL/sibling-coexistence sweep: **245 pass** (incl. the sibling's seg_form_unify tests,
untouched).

## Migration queue (un-migrated derive_* — the #403 live-count-4, documented, WARN-only)
- `derive_sealed_205_config`, `derive_store_nothing_205_config`, `derive_fresh_seeded_config`,
  `derive_config` — still hand-assemble via `WitnessConfig.to_trainer_flags`. **Full replacement of the
  `WitnessConfig` ordered-tuple emitter with `WitnessProgram.compile_trainer_argv` is a SEPARATE larger
  migration** (the two emitters order flags differently by design; forcing token-order byte-identity
  would require gutting `WitnessProgram`, which the charter forbids). Strict-flip #403 → 0 and launcher
  `--enforce-dsl-config-gate` → default when this queue drains (each gets typed routing or a migration
  waiver). This landing migrates crucible_v6 (the live launch candidate) + builds the typed authoring
  surface + apparatus the rest plug into.

## ty (type-check) debt
`typed_config.py` (this task's surface) = **0** code diagnostics (clean). The scoped
`ty check src/tac/witness_dsl` surfaces **16 pre-existing diagnostics** (21 file-location mentions —
multi-line diagnostics cite >1 file), ALL in sibling modules: lever_registry · lawref · schedule_readback
· campaign (`invalid-argument-type` warnings, mostly `int(x|None)` narrowing) + **2 ERROR-severity
`unsupported-operator` in `curriculum_dsl.py:1413`** (`if None not in (_tau_s,_l7_s) and not (0<_tau_s<_l7_s)`
— ty cannot follow the `None not in (...)` narrowing; a runtime-safe FALSE POSITIVE). Because of those 2
errors `ty` (and therefore the script) exits **rc=1** — EXPECTED and DOCUMENTED; the script is **advisory,
not wired into any hard gate**. Per the "annotate what you touch, don't boil the ocean" charter these are
sibling-owned debt, not fixed here. Ratcheting `curriculum_dsl:1413` (an explicit non-None assert or a
guard ty can follow) + the narrowing warnings is future work.

## Means != ends
A typed config is a MEANS. Only a byte-closed n600 exact row < 0.19110 from `upstream/evaluate.py`
(contest-CPU/CUDA, NEVER MPS) moves the pointer. Pointer UNMOVED. [no-triality]

## Paths
- `src/tac/witness_dsl/typed_config.py` (NEW schema + manifest helpers)
- `src/tac/witness_autoconfig.py` (dsl_program_manifest field + `_attach_dsl_program_manifest` + crucible wire-in)
- `tools/launch_witness_run.py` (b0.6 gate + `dsl_config_gate_action` + 3 flags)
- `tools/triality_drift_detector.py` (DSL-config-bypass leg)
- `tools/typecheck_witness_dsl.sh` (NEW ty target)
- `src/tac/confound_gates.py` (`check_launch_config_authored_in_dsl` #403)
- tests: `test_typed_config_schema.py`, `test_typed_config_migration.py`, `test_launch_dsl_config_gate.py`,
  `test_dsl_config_gate_apparatus.py` (NEW) + `test_confound_gates.py` (registry update)
