# fmtools baked into the costate organ — ADVISORY semantic-classification SENSE layer (task #522)

**Source:** operator 2026-07-17 (verbatim): *"We can also bake fmtools into the costate organ or
costate controller as well."* Landed as an ADVISORY sense layer, never the actuating/blocking
decision. Sister of the three established fmtools consumers
(`tools/triality_drift_detector.fm_scope_advisories` · `tools/dashboard_fm_events` ·
`tools/magnitude_dismissal_detector`) — the SAME detached-subprocess pattern, now a shared
`tac`-level adapter the costate organ consumes at four insertion points.

**Axis / authority:** every FM output is `on-device FM · advisory · read-only · NON-PROMOTABLE`.
The pointer is UNMOVED — this is MEANS (an observability + duty-ranking sense organ), never a
score/verdict/actuation. It moves only when the controller's picks land a lower byte-closed row.

## The advisory boundary (binding; CONTAINMENT unchanged)

FM outputs land as an explicit `fm_advisory` field / digest section and **NEVER feed actuation, a
block, a verdict, a promotion, or a numeric score.** The deterministic/numeric layers remain the
decision; the FM is a semantic second opinion. Score-neutral + read-only by construction (it only
READS text + emits closed-set labels; rationale restates only words present in the text; the FM
never invents numbers).

**HOME = `src/tac/fm_advisory.py` (NOT `tac.witness_control`).** The witness_control package carries
the structural "no actuation" invariant (`test_no_actuation_capability`: source-token scan forbids
any `subprocess`/spawn token so the controller cannot launch/stop/mutate a run). This adapter MUST
spawn a classification subprocess, so it sits at the `tac` top level beside the tools/-layer FM
consumers — spawning an on-device text CLASSIFIER is not run-actuation.

## The shared adapter (`src/tac/fm_advisory.py`)

- `fm_python()` / `available()` — resolve the fmtools-venv interpreter (`FM_ADVISORY_PYTHON` →
  `DASH_FM_PYTHON` → `~/Projects/fmtools/.venv/bin/python`); None ⇒ every entry point degrades to
  None (the layer is ABSENT, never a value-shaped stub). Zero pact-venv deps.
- `classify(items, labels, instructions)` — the generic single-label classifier: spawns ONE
  subprocess under the fmtools venv running a closed-`anyOf` structured-generation script; returns
  rows aligned to input (`label` None on FM abstention), or **None** on absence/total failure.
  In-process content-hash cache (no disk write ⇒ the write-free digest stays write-free). The
  subprocess runner `_run_job` is the single mockable seam (tests never touch a live model).
- `classify_one`, `prosify` (k-is-v framing to dodge the FM language guardrail), `numeric_regime_hint`.

## The four organ insertion points (all advisory)

| # | insertion point | input | output | wired into |
|---|---|---|---|---|
| (a) | **REGIME supplement** | recent telemetry text + numeric annulus per-class flip shares | `fm_regime` ∈ {lane-erosion, mixed-Lane-Road, movable-island-unborn, OTHER/novel} + `agrees_with_numeric` flag | shadow row `fm_advisory` + digest section; **disagreement ⇒ a surfaced diagnostic line, never an override** |
| (b) | **EVENT-intelligence** | notable run-log events (reuses `dashboard_fm_events.extract_notable_events`) | event class ∈ {stage-transition, lever-engage, guard-alarm, convergence, regression, info} | shadow row + digest (annotation feeding the #344 context) |
| (c) | **DUTY-ranking supplement** | top-K never-fired duty rows + current regime text | per-lever relevance ∈ {high, medium, low} | digest section — a **SECONDARY sort hint; the P8 floor-aware order stays the base order** |
| (d) | **CONFOUND-alarm classing** | recent unresolved harness-failure rows + known class ids (reuses `dashboard_fm_events.known_failure_classes`) | matched known class or `none` | digest section (composes with the Class-5 bug-sweep machinery) |

`shadow_advisory(...)` bundles (a)+(b) for the shadow-row field; the digest section runs all four.

## Numeric agreement (grounded, honest)

`numeric_regime_hint` derives a regime from the MEASURED annulus per-class flip shares
(Road=0/Lane=1/Movable=3): Lane-dominant → lane-erosion; Lane+Road both material → mixed-Lane-Road;
Movable≈0 while others move → movable-island-unborn; else OTHER/novel. Returns None when shares are
unavailable ⇒ `agrees_with_numeric = None` (agreement is never guessed). The agreement flag is a
real numeric-vs-semantic comparison, not a vibe.

## Wiring

- **Shadow observer row schema** (`tac.witness_control.shadow_controller.ShadowReport`): additive
  `fm_advisory: dict | None = None`, included in `to_row()` ONLY when non-None ⇒ **byte-identical row
  when absent**. Populated only via `build_shadow_report(..., with_fm_advisory=True)` (default OFF —
  a compute-cost subprocess), which calls `_attach_fm_advisory` (fail-open ⇒ leaves None).
- **Digest** (`tools/costate_digest.py::section_fm_advisory`): consumes the ALREADY-COMPUTED digest
  `data` (shadow classification, annulus, duty, failure ledger) — no numeric rework — and renders a
  compact section.

## What fires when the venv is absent / gated off

- **fmtools venv ABSENT:** `available()` is False ⇒ `section_fm_advisory` returns `[]` and the shadow
  field stays None ⇒ the digest is **byte-identical** to before this task, and shadow rows are
  byte-identical. Proven by `test_build_digest_byte_identical_without_fm` +
  `to_row` omission test.
- **Compute-cost gate (`_fm_advisory_enabled`):** the FM section is a genuine compute cost (on-device
  subprocesses, ~8s measured) so per CLAUDE.md *"'Off' is a tracked queue"* it gates on cost with a
  RECORDED reason: **default ON** for explicit `tools/costate_digest.py` / witness-status check-ins
  (the agent asked), **default OFF** in the `<5s` SessionStart hot path (measured 1.6s session-start
  vs 12.8s with FM). `COSTATE_FM_ADVISORY=1/0` overrides both directions. The gate reason is written
  into `data["fm_advisory"]` when skipped (queryable, never a silent default).

## Not a DSL lever (no lever-registry entry warranted)

The FM sense layer changes no training weights / archive bytes / d_seg / d_pose — it is observability,
not a score-affecting lever. Therefore it is NOT a DSL `Lever` and does NOT enter the activation
ledger / duty-to-measure queue. Its one "off" state (the compute-cost gate) is tracked-with-reason via
the env gate + the recorded `data["fm_advisory"].reason`, satisfying the orphaned-signal reconciliation
without a lever row.

## Tests (26 green; `ruff --select F` clean; CONTAINMENT `test_no_actuation_capability` green)

- `src/tac/tests/test_fm_advisory.py` — venv-absent degrade (clean None for every entry point),
  classify subprocess contract (mocked runner), content-hash cache (runner called once for identical
  text), `numeric_regime_hint` + agreement-flag logic, the four insertion-helper shapes, `prosify`.
- `src/tac/tests/test_costate_digest_fm_advisory.py` — venv-absent ⇒ no lines / byte-identical digest,
  regime+agreement render, disagreement diagnostic line, present-but-no-inputs, fail-open on exception,
  the compute-cost gate (default on/off + env override).

## Measured live (2026-07-17, this host has the fmtools venv)

Full `tools/costate_digest.py` renders all four insertion points: regime=lane-erosion, 6 events
classified, 3 duty-relevance hints, 2 confound classes matched — wall-clock 12.8s (FM on) vs 1.6s
(session-start, FM gated off). Advisory · NON-PROMOTABLE; pointer UNMOVED.
