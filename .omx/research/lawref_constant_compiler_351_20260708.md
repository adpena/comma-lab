# LawRef constant-compiler (task #351) — landing memo

STORES CONSULTED (retrieval-first, per ROOT CAUSE nexus):
- `tools/corpus_query.py "LawRef constant compiler value provenance ladder"` — no prior LawRef
  module; nearest surfaces = requirement T (ledger), the #43 constant-provenance L2 gate, and
  `src/tac/substrates/_shared/constants_provenance_manifest.py` (a STATIC provenance manifest, a
  DIFFERENT surface — 4-value DERIVED/MEASURED/LEARNED/ARBITRARY taxonomy for substrate constants;
  NOT a resolver that COMPILES equations→values). LawRef is the requirement-T ladder + resolver.
- `.omx/research/t5_crucible/ORCHESTRATION_LEDGER.md` req **T** (VALUE-PROVENANCE LADDER — LawRef is
  its by-construction enforcement), req **P** (signal-completeness / named consumers), req **Q**
  (probes→durable instruments). P-CT1 lesson: ν=0.012653 valid at mod-32/this schedule → re-fit on
  config change = the CONFIG-CONDITIONAL fail-closed this resolver mechanizes.
- `src/tac/canonical_equations/equation.py` (CanonicalEquation / EmpiricalAnchor / the
  `python_callable_module_path` field), `__init__.py` (build_* factories + registry API).
- `src/tac/witness_dsl/curriculum_dsl.py` (WitnessProgram, `base` flag dict, `flag_dict()`,
  `compile_trainer_argv()`, `validate()`, Lever).
- τ/ν/forfeit/σ_eff anchors: `probe_tau_confirm_ep1000_20260708.md`, `probe_waveA_ct_schedule_*`,
  `DRAFT_OPTIMAL_STACK_v6_20260708.md` (§ folds 4/6; B18 r*=0.95·σ_eff), and the artifacts under
  `.omx/research/t5_crucible/artifacts/`.

COLLISION: sibling (v6.3 fixer) owns `src/tac/witness_autoconfig.py` + the v6 draft + its tests —
UNTOUCHED. Scope here = `src/tac/witness_dsl/` + `src/tac/canonical_equations/` + new tests + memo.

## The value-provenance ladder (req T), mechanized

Ladder classes (highest→lowest, verbatim from req T):
1. `derived_live` — law evaluated at runtime from measured state (cannot rot).
2. `derived_at_config` — law + pinned inputs, executable + cited (rots only if an INPUT rots).
3. `measured_anchor` — artifact-cited measurement, staleness-scoped, CONFIG-CONDITIONAL.
4. `hardcoded_waiver` — last resort: recorded reason + owner + re-derivation trigger.

A LawRef IS a class-(2/3) value made executable: the equation_id + typed inputs resolve, at
DSL-compile time (the mx.compile analogy), into an actual value + a provenance manifest. A bare
literal with none of the ladder metadata is the bug class req T extincts.

## What resolves end-to-end (the 3 first laws — bit-matching sealed values)

All 3 resolve through the real T5-crucible artifacts + the evaluator surface, to the
sealed values (bit-match IS the validation; `test_lawref_constant_compiler.py`, 34 tests):

| LawRef | equation_id | law | resolved value | sealed value | ladder |
|---|---|---|---|---|---|
| `S_STAR_FORFEIT_MATCHED_EXIT` | `forfeit_matched_exit_v1` | s* = ν·forfeit | `6.897090095741019e-06` | rounds to **6.8971e-6** | derived_at_config |
| `TAU_STAR_MASLOV_Q90` | `tau_star_maslov_quantile_v1` | τ* = m_q(q90)/ln5 | `0.4619441215759677` | **bit-identical** to the artifact's stored `tau_star.q90` | measured_anchor |
| `R_STAR_CRITICAL_NUCLEUS` | `critical_nucleus_release_v1` | r* = 0.95·σ_eff | `1.4249999999999998` | **1.425** | derived_at_config |

- s*: ν = `0.012653403634932212` read from the wave-A trace-probe artifact
  (`probes/ct1/result/stages/tau_softplus/nu_per_ep`, P-CT1), forfeit =
  `0.0005450778537325913` literal (P-CT3 recovery constant). REAL FINDING surfaced by
  the LawRef discipline: the artifact's *stored* `s_star_nu_times_forfeit_S_per_ep`
  (6.897090681181217e-6) used an **inline forfeit not reproducible from any stored field**
  — the ν·forfeit product from the two CITED inputs is 6.897090095741019e-6 (agrees to the
  sealed 5 sig figs, differs at ~1e-13). LawRef makes both inputs explicit → the value is now
  reproducible; the bare stored constant was not. (This is precisely the req-T rot the ladder extincts.)
- τ*: m_q(q90) = `0.7434703826904296` read from the τ-confirm artifact
  (`results/0/table/m_q/q90`); ln5 defaults to `math.log(5)` (== the artifact's stored ln5).
  Resolver value == artifact's stored `tau_star.q90` bit-for-bit.
- r*: coeff 0.95 + σ_eff 1.5 (dilation-knee probe), both literals; σ_eff carries a
  `config_tags={"sigma_probe":...}` → an UNVERIFIABLE-tag warning when the target omits it.

Evaluator surface (`src/tac/canonical_equations/evaluators.py`): a uniform
`evaluator(resolved_inputs) -> value` registry keyed by `equation_id`, opt-in via
`register_evaluator` / `populate_lawref_evaluators()`. The resolver calls
`resolve_equation_value(equation_id, inputs)`. This is the executable **equations→DSL** link.

## Fail-closed semantics

- **CONFIG-CONDITIONALITY conflict** (the P-CT1 protection mechanized): if a target config tag
  key is also on an input's `config_tags` with a DIFFERENT value → `ConfigConditionalityViolation`
  naming the exact mismatch (`schedule = mod32cap (anchor) != mod48cap (target)`). ALWAYS raises —
  NEVER swallowed by a fallback (silently using a mod-32 ν on a mod-48 run is exactly the bug).
- **sha256 mismatch / staleness exceeded / artifact missing / extract-key missing** →
  `LawResolveError`, UNLESS the LawRef declares a `fallback` (+ `fallback_waiver_reason`), in which
  case the fallback is used and the manifest records `fallback_used=True` + the reason (req-T class-4).
- **Unverifiable config tag** (key on the input but absent from target) → a warning (not a raise) so
  the manifest surfaces "cannot confirm applicability" without blocking.
- **Determinism**: the VALUE depends only on resolved inputs + the pure-math evaluator;
  `resolved_at_utc` is metadata only. Two resolves → identical value (tested).

## File paths

- `src/tac/witness_dsl/lawref.py` — InputRef / LawRef / ResolvedConstant + `resolve()` +
  `resolve_flag_dict_constants()` (the mechanism).
- `src/tac/witness_dsl/lawref_builtins.py` — the 3 first LawRefs (migration seed).
- `src/tac/canonical_equations/evaluators.py` — the evaluator registration surface + 3 evaluators.
- `src/tac/witness_dsl/curriculum_dsl.py` — `WitnessProgram.compile_trainer_argv_with_constants()`
  (additive; existing `compile_trainer_argv` byte-identical — tested).
- `src/tac/{witness_dsl,canonical_equations}/__init__.py` — exports.
- `src/tac/tests/test_lawref_constant_compiler.py` — 34 tests.

## Migration plan (FOLLOW-UP, post-v6.3) + launcher hookup

**crucible_v6's 5 constants → LawRefs** (do NOT do now — sibling owns `witness_autoconfig.py`
+ the v6 draft; this is the named next step):
1. `τ_end` (0.31 knee band) → `tau_star_maslov_quantile_v1` with the SC-3 chosen convention
   (q-hat) as a `config_tags={"convention": ...}` input. (~1 LawRef, reuse existing evaluator.)
2. `ν laws` (settle 237 / cycle 387 / dwell / s*) → new evaluators `settle_window_v1` etc.,
   ν anchored to the trace-probe artifact (config-conditional mod32cap). (~4 evaluators + LawRefs.)
3. `β_hosc` (fire-band 1.7252, the misprinted-1.41 sibling) → `hosc_beta_fireband_v1`. (~1.)
4. `adaptive-ε clamps` (0.3/0.7 — the >90%-epoch-binding class-(3-or-4) anchors) → derived_live
   `c(τ)` form OR measured_anchor LawRefs with staleness. (~2.)
5. `s_fit` absolute persistence bar (1.7504924172 / 1.3017706202) → `conley_absolute_bar_v1`. (~1.)
Estimate: ~9 evaluators + ~9 LawRefs + tests, ~350-450 LOC, folding each hardcoded v6 constant
into `lawref_builtins.py` and swapping the autoconfig literal for the LawRef (the ONLY autoconfig
touch — one line per constant). Each fold ELEVATES a class-(3-or-4) literal to class-(2/3).

**Launcher hookup** (one line, NOT wired here per the collision boundary): in the witness
launch path, after building the program, replace `program.compile_trainer_argv()` with
`argv, manifest = program.compile_trainer_argv_with_constants(target_config_tags=<run tags>)`
and write `manifest` to `constants_manifest.json` beside `launch.sh`.

## Triality
This landing touches BOTH the **DSL leg** (`witness_dsl/lawref*.py` + WitnessProgram method) and
the **equations leg** (`canonical_equations/evaluators.py`) — per-leg drift satisfied by
construction; the LawRef IS the executable equations→DSL edge. DAG FEED-lawref appended.

