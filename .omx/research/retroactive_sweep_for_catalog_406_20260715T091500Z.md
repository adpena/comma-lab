# Retroactive sweep — Catalog #406 DSL compile binding

**Bug-class symptom signature:** witness execution/admission from a raw or
post-compile argv; missing `dsl_provenance.json`; manifest/hash disagreement;
`--skip-dsl-config-gate`, `--allow-non-dsl-config`, dry-run downgrade, or verifier
exception used to proceed without an exact typed-program recompile.

**Pre-fix window:** all witness-launch history before the Catalog #406 landing.
The directly relevant transition begins with requirement V on 2026-07-08, when
the typed manifest gate existed but was explicitly WARN/override/fail-open.

**Historical STAND_DOWN search:** an exact search across
`.omx/research/*stand_down*.md` found no score-family KILL/DEFER/STAND_DOWN whose
premise depended on a non-DSL witness argv. Therefore no historical negative
scientific verdict is automatically invalidated by this apparatus fix.

## Findings and re-evaluation priority

| Finding | Evidence | Disposition | RE-EVAL priority |
|---|---|---|---|
| Requirement-V gate explicitly allowed rationale/skip/dry-run and failed open | `.omx/research/dsl_only_typed_config_requirement_V_20260708.md:72-73` | Apparatus premise superseded by Catalog #406; do not reuse its admission claim | P0 complete in this landing |
| Sealed #205 began as a hand-authored one-off, then was canonicalized by name | `.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md` FEED-205p3 | Historical result bytes are not invalidated; any future resume/launch must compile a fresh binding triple | P1 before reuse |
| V9 provenance DAG already named flags appended after DSL as a config-orphan confound | `.omx/research/v9_frontier_provenance_dsl_recursive_fractal_directives_DAG_FEED_20260714T133500Z.md:26` | Confirming prior; now structurally enforced at launcher and governor | Closed |
| Witness sweep spec forbids hand-authored flags/launch edits | `.omx/research/witness_train_sweep_spec_20260714.md:86,270` | Consistent; future launches gain the rc-8 structural guard | No re-eval |

**Verdict scope:** apparatus/admission provenance only. No score, archive, model,
or contest-axis verdict is changed. Pointer UNMOVED.
