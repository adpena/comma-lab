# Eightfold apparatus build — landing memo (2026-07-09, #387)

**Operator GO 2026-07-09 "Encode all... update apparatus and configure any hooks
or gates as necessary".** Made the 8 design philosophies
(`~/.claude/.../design_philosophies_eightfold_20260709.md`) STRUCTURAL: gates
where automatable (P1, P4), standing SEAL checks where fuzzy (P2/P5/P6/P7/P8).

## STORES CONSULTED (recall-before-build)
- `docs/operating_manual_craft_handoff.md` (§4 re-derive not recognize; §8 don't
  agree-with-the-test — both applied below).
- `design_philosophies_eightfold_20260709.md` (the 8; apparatus-routing section).
- `src/tac/confound_gates.py` (the gate family pattern: `_finish`, `_waiver_present`,
  `_span_source`, warn-only + strict-flip discipline, CONFOUND_GATES registry).
- `src/tac/witness_dsl/activation_ledger.py` (`canonicalize_significance_keys`,
  `_read_significance`, `known_levers`, `_SIGNIFICANCE_LEVER_ALIASES`).
- `src/tac/witness_dsl/lever_registry.py` (`lever_factories` — held factory set).
- `.omx/state/lever_relative_significance.jsonl` (the P1 store; 8 rows).
- `src/tac/witness_control/{sigma_min_plateau,verdict_trend_alarm}.py` (P4 exemplars).
- `tools/auto_push_main.py` (the #259 fmtools ADVISORY firewall — separate venv,
  subprocess, fail-open, never-authority — mirrored for the P4 uncertain path).
- `.omx/research/t5_crucible3/ORCHESTRATION_LEDGER.md` (SEAL standing-check home).

## What landed
**P1 gate — `check_significance_keys_canonical` (warn-only, `EIGHTFOLD_GATES`).**
Every relative-significance store key must resolve, through the actual
`canonicalize_significance_keys` (DRY — the gate uses the apparatus's own
reconciliation, not a re-impl), to a HELD DSL `Lever` factory name. Unresolvable =
the duty-to-measure ORPHAN bug class (a task-#-keyed row that never reconciles →
a built+held lever FALSELY reported unbuilt; the 2026-07-09 receipt). In-row
waiver `# SIGNIFICANCE_KEY_OK:<rationale>` inside the row's `notes` string
(JSONL-safe; placeholder rejected via Catalog #287 sister). **LIVE COUNT = 4**:
`seg_chroma_boundary_276` (notes name held `SegChromaBoundary` → disposition: add
alias), `seg_down_weight_274` (build or waiver), `latent_table_truncate_d18_k90`
+ `mod32_neutrality_19_ab` (intentional non-DSL-factory findings → waiver on next
append). Warn-only; strict-flip when live=0 (sibling activation_ledger work).

**P4 gate — `check_witness_control_meters_have_canaries` (warn-only, `EIGHTFOLD_GATES`).**
Every MEASUREMENT/detector class in `witness_control/*.py` must ship a
canary/positive-control. Deterministic floor: CERTAIN meter = name matches
`*(Detector|Alarm|Trend|Plateau|Monitor|Observer|Meter)`; a certain meter with no
canary token in module or `tests/test_<mod>.py` = violation. Bare `*Gate` is
DELIBERATELY EXCLUDED from name-anchoring (in this codebase Gate = actuator:
`EventBackstopGate` fires treatments, `GateStep` is a frozen value-object — neither
is a measurement surface; verified they are NOT false-flagged). UNCERTAIN = an
`observe/detect/classify` method with an ambiguous name → NOT counted, listed;
opt-in fmtools advisory (`use_fmtools=True`) records a second opinion, NEVER sole
authority (default OFF ⇒ per-session preflight pays ZERO FM cost). **LIVE COUNT = 1**:
`VerdictTrendAlarm`.

## §4 re-derivation correction (don't agree-with-the-test)
The build brief named BOTH `sigma_min_plateau` (canary_suite) AND `verdict_trend`
(backtest fixtures) as PASSING exemplars. Re-derived from the artifacts:
`SigmaMinPlateauDetector` PASSES (real `canary_suite`: synthetic positive
`synthetic_plateau_series` + rising negative `synthetic_rising_series`).
`VerdictTrendAlarm` FAILS — it carries NO canary/positive-control/backtest token
in-module or in any test file. It is a CURRENT VIOLATOR, not a passing exemplar.
Disposition (sibling, warn-only ⇒ non-blocking): add a canary_suite-style
positive+negative control OR a `# METER_CANARY_OK:<rationale>` waiver. I did NOT
edit `verdict_trend_alarm.py` (sibling-contended witness_control; a real canary is
new logic beyond a gate-builder's scope).

## fmtools (operator nudge — recall-before-build, then honest composition)
Checked `fmtools` first (#259; `tools/auto_push_main.py` advisory exemplar): it is
a SEPARATE venv (`~/Projects/fmtools/.venv`) invoked by SUBPROCESS, fail-open,
never-authority. Wired it as an OPT-IN advisory for the P4 uncertain path only
(`_fm_meter_advisory`), recorded in the finding rationale, never sole authority —
the warn-only gate + advisory classifier = the honest composition the nudge asked
for. Default `use_fmtools=False` so `preflight_all` pays zero FM cost (the nudge's
accepted heuristic-only-with-uncertain-list fallback). LIVE evidence (the FM is
present on this box and ran): `BirthCompletionController → not_meter` ("acts as a
controller"); `PolyakTailAverager → meter` ("accumulates and reports ... a reading
for downstream decision-making") — a defensible-but-debatable call, which is
exactly why it is advisory-only and the deterministic name-heuristic stays the
floor (both remain UNCERTAIN, uncounted, by default).

**P2/P5/P6/P7/P8** → SEAL standing checks (fuzzy-by-nature, not static gates):
appended to `ORCHESTRATION_LEDGER.md` STANDING FAILURE-MODE CHECKS +
`.omx/research/crucible_standing_checks_eightfold_20260709.md` (the reusable
template every future convening cites). P2's noise-floor column routed into the
#385 brief spec (noted in the template — the doc, not the task tool).

## NO-FAKE / verification
- Both gates ACTUALLY scan their targets (P1 reads the store + resolves through the
  real canonicalize; P4 AST-parses each module + checks canary tokens in module &
  test). No token-theater.
- 29 tests (`src/tac/tests/test_eightfold_gates.py`, all pass): P4 explicitly
  verifies the heuristic CATCHES a synthetic meter-without-canary AND does NOT
  false-flag an actuator/value-object; P1 verifies resolve/alias/orphan/waiver/
  placeholder-reject/latest-wins/strict-raise; both carry a repo live-count bound
  so a NEW violation is caught.
- 86 existing confound-gate tests still pass; ruff F clean on all edited files.

## Triality legs (same commit)
- **DSL** — N/A: no new lever (these are apparatus gates on existing stores/code),
  so nothing to add to `witness_dsl`. Stated explicitly.
- **DAG** — appended FEED-eightfold-apparatus to
  `.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`.
- **equations** — N/A: these encode PROCESS law (apparatus discipline), not an
  S_τ / score law, so no `canonical_equations` row. Stated explicitly.

## Pointer
0.19110 UNMOVED (this is apparatus/means — makes the campaign harder to fool, does
not itself move the exact score).
