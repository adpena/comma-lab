# CANONICALIZATION UNIT 2 — emission-side hygiene pair (LANDED 2026-07-09)

**Task #388** (operator GO 2026-07-09 "Go on building all"). Subordinate to NO-FAKE + THE GOAL.
Pointer **0.19110 UNMOVED** — this is MEANS (emission apparatus), not an exact-eval row.
`[apparatus · NON-PROMOTABLE · $0 · no GPU]`.

## What shipped

| Surface | File | Contract |
|---|---|---|
| Schema | `src/tac/verdicts/measurement_row.py` | frozen `MeasurementRow` + `Provenance` + `AxisTag` / `ReviewStatus` StrEnums; validated in `__post_init__`; `to_json_dict()` stable-ordered |
| Emission | `src/tac/verdicts/emit.py` | `emit_verdict()` — refuses missing scope / rows-for-measured / composition / constraint_carved; negatives need a reformulation queue; atomic tmp+rename write |
| Package | `src/tac/verdicts/__init__.py` | re-exports the public surface |
| Serializer flag | `tools/subagent_commit_serializer.py` | OPTIONAL `--triality-legs {dag,dsl,equations,none}` + `--triality-reason`; recorded in the JSONL log. **85 insertions, 0 deletions** (byte-identical flag-absent behavior) |
| Drift softening | `tools/triality_drift_detector.py` | reads the latest committed serializer row for the window; a structured disposition SOFTENS the core `classify()` drift to info |

## The load-bearing invariants (why these fields, not others)

- **P2 (noise floor)** — `noise_floor` is nullable (== UNKNOWN, honest) but a NON-None floor MUST carry
  `floor_provenance`. Net: a silent-zero floor cannot be constructed. (`design_philosophies_eightfold_20260709` P2.)
- **P10 (constraint_carved)** — every verdict states what design-space region it removes/pins; required by `emit_verdict`.
- **P12 (composition)** — every verdict carries its interaction sign with the active lever set, OR an explicit
  `deferred_to_ab_protocol`; required by `emit_verdict`.
- **verdict-scope ladder** — INSTANCE < FORMULATION < FAMILY < PARADIGM + a `scoped_to`; FAMILY/PARADIGM need
  `family_evidence` (citation or ≥2 distinct formulations). One failed formulation ≠ dead family.
- **axis authority** — only `[contest-CPU]` / `[contest-CUDA]` are authority; everything else is advisory
  (MEMORY MPS/authority discipline). Surfaced as `AxisTag.is_authority` + `is_authority_axis` in the JSON.
- **review-status (MEMORY L81)** — `reviewed` / `unreviewed_recovery_written` / `provisional`; only `reviewed`
  is load-bearing. Sister of `tac.council_continual_learning.EmpiricalVerificationStatus`.
- **n600** — a subset (`n_samples != 600`) MUST state its reason ("allergic to non-n600/toys").

## Tests

62 new, all green (`.venv/bin/python -m pytest src/tac/tests/test_verdict_measurement_row.py
src/tac/tests/test_verdict_emit.py src/tac/tests/test_serializer_triality_legs.py`):
- `test_verdict_measurement_row.py` (24) — schema validation pos/neg, P2 silent-zero-floor impossibility, n600-reason, axis authority, frozen.
- `test_verdict_emit.py` (17) — every refusal path + deferred-composition + empty-reformulation-with-reason + atomic roundtrip.
- `test_serializer_triality_legs.py` (21) — `_parse_triality_legs` pos/neg; flag recorded in log; **absent-flag identical behavior + no triality keys**; malformed-flag refuses before any git action; `triality_disposition_from_rows` softening; 3 live-hook end-to-end (control blocks / disposition softens / none softens).

Backward-compat proof: serializer diff is purely additive (0 deletions); all 58 existing serializer tests green.
All new code fully ruff-clean; tools pass `ruff --select F` (pre-existing UP017/SIM115/I001/RUF100/C420 debt left untouched per "touch nothing else").

## CONSUMER LIST for #389 (the wire-in that follows)

`tac.verdicts` is deliberately LEAF-CLEAN — it imports NEITHER `tac.through_r` NOR `tac.session_bus`
(sibling units building in parallel). #389 wires the consumption. Each expected consumption site carries a
`# TODO(#389)` marker in the module docstrings of `src/tac/verdicts/__init__.py` + `measurement_row.py` +
`emit.py`. Expected #389 consumers:

1. **`tac.session_bus`** (cross-agent verdict fan-in) — reads emitted verdict JSONs (schema `verdict.v1`)
   from sibling agents; the stable `to_json_dict()` shape + `schema_version` field are the parse contract.
2. **`tac.through_r`** (measurement authority) — cross-checks a `MeasurementRow`'s `axis_tag` / `value` /
   `provenance.inputs_sha256` against the through-R byte-close authority; an `AxisTag.is_authority` row that
   through_r cannot reproduce is a flag.
3. **`tools/triality_drift_detector.py`** — ALREADY consumes the serializer `--triality-legs` disposition
   (this unit). #389 may additionally have it consult emitted verdict JSONs for the verdict-scope leg.
4. **`tools/costate_digest.py` / dashboards** — may surface the duty-to-measure queue keyed on emitted rows
   (review_status == `unreviewed_recovery_written` auto-queues fresh eyes).

Wire-in contract for #389: import `from tac.verdicts import MeasurementRow, emit_verdict, ...`; do NOT
re-implement the schema; extend via the `extra` dict on `emit_verdict` or a new `to_json_dict` key with a
`schema_version` bump.

## Triality disposition
DAG leg = FEED-canon-u2 (in `.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`).
DSL / equations legs = **N/A-with-reason** (apparatus: a typed emission contract, no witness lever /
trainer flag / curriculum surface and no NEW measured physics relation). Committed via the serializer with
`--triality-legs dag`.
