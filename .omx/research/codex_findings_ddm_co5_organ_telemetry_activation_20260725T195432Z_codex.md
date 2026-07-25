# Codex findings — DDM CO5 costate-organ telemetry activation

UTC: 2026-07-25T19:54:32Z
Task: `#708`
Lane: `lane_ddm_co5_organ_telemetry_activation_20260725`
Status: `BUILT_FAIL_CLOSED_MAIN_REVIEW_REQUIRED`
Maturity: `_dev`
Research only: `true`
Actuation: `NONE`

## Verdict

The charter premise is falsified by the settled CT1 source of truth. CT1 is
fresh at consumption, but its own findings state
`FLOWING_TYPED_INPUTS_NO_DELTA_S_PER_HOUR_VERDICT`: cadence is measured, the
cumulative objective trace is explicitly `ADVISORY_BATCH_LOCAL` and not N600,
and only one exact campaign endpoint is surfaced. EV1 contributes a complete
600/600 cross-sectional endpoint join, not a temporal trajectory.

The four CO4 enhancements are now typed, hash-lineage-bound, consumption-time
freshness-checked, wired into the existing SENSE/DECIDE surfaces, and visible
through `tools/costate_digest.py`. Their honest state is **0/4 active, 4/4
held**. No launch, scorer invocation, campaign mutation, actuation change,
score claim, or pointer edit occurred.

Receipt:
`.omx/research/ddm_co5_organ_telemetry_activation_20260725T195432Z/ddm_co5_organ_telemetry_activation_receipt.json`.

## Per-enhancement findings

### 1. Pontryagin/Bellman adjacent-transition residual

- **Spec recalled:** compute
  `lambda_t - (dL_t/dx_t + J_t^T lambda_(t+1))` only across ordered adjacent
  campaign states.
- **Wired:** typed SENSE row consuming CT1 exact-endpoint/cadence rows and EV1
  evidence-join lineage.
- **Backtest:** `FAILED_CLOSED_MISSING_MATCHED_AUTHORITY`. CT1 has no adjacent
  state costates or realized transition Jacobians; EV1 pair rows are
  cross-sectional and cannot be reinterpreted as a trajectory.
- **State:** `DESIGNED_NOT_ACTIVE`.

### 2. M34 per-state dual consistency

- **Spec recalled:** compare organ and M34 duals only at the same state,
  dimension, units, and measured uncertainty band.
- **Wired:** typed SENSE row consuming the CT1 exact-endpoint row and EV1
  dimension-byte-home lineage.
- **Backtest:** `FAILED_CLOSED_MISSING_MATCHED_AUTHORITY`. Neither source has a
  same-state organ/M34 dual pair or measured uncertainty band.
- **State:** `DESIGNED_NOT_ACTIVE`.

### 3. Compression progress per effort

- **Spec recalled:** measure receiver-realized campaign score-action reduction
  per wall-clock hour; never promote a batch-local trace to exact N600
  progress.
- **Wired:** typed SENSE row consuming CT1 measured cadence, batch-local trace,
  exact-endpoint row, and EV1 lineage.
- **Backtest:** `FAILED_CLOSED_MISSING_MATCHED_AUTHORITY`. CT1 measures
  `302.9149270083662 s/step`, but the only cumulative objective trace is
  `ADVISORY_BATCH_LOCAL`, explicitly not N600, and the surfaced exact row is
  step 50 only. A legal delta requires at least two exact endpoints with
  matching archive-byte and wall-clock custody.
- **State:** `DESIGNED_NOT_ACTIVE`.

### 4. Regret-bounded duty allocation

- **Spec recalled:** rank advisory duties by measured progress per effort plus
  a preregistered exploration bonus backed by fired-duty history.
- **Wired:** typed DECIDE row consuming CT1 cadence/geometry-event rows and EV1
  lineage.
- **Backtest:** `FAILED_CLOSED_MISSING_MATCHED_AUTHORITY`. Its progress/effort
  dependency is inactive. CT1's 12 geometry-cure receipts are an event count,
  not typed duty identity/outcome history, and no exploration-confidence
  contract exists.
- **State:** `DESIGNED_NOT_ACTIVE`.

Every negative is scoped to stopped-v5 CT1 telemetry × EV1; no enhancement
family is killed.

## Consumption freshness and digest visibility

The CT1 receipt is registered in the organ's existing `SOURCES` tuple. At
every consumption, the organ rehashes the settled source tree, rejects
symlinks/special files, and compares file count, bytes, and tree SHA against
the CT1 receipt. Current observed custody is `[fresh]`: 126 files, 1,075,654
bytes, tree SHA
`18526644423949012b29dc1e043c744ee93e5739338b0e98d97f3f1462d8467c`.
Missing or changed source custody becomes `[stale-advisory]` and cannot
activate a row.

The human digest now emits:

```text
DDM-CO5: active=0/4 held=4 freshness=[fresh] gate=PREMISE_FALSIFIED_CT1_DELTA_S_PER_HOUR_GATE_NOT_SATISFIED actuation=NONE
```

The same typed state appears in the JSON digest and the existing named
campaign consumer views.

## FEED-708-co5

The durable DAG feed is
`.omx/research/ddm_co5_organ_telemetry_activation_20260725T195432Z/DAG_FEED.md`.
It routes CO4 specs + CT1 typed telemetry + EV1 joins through consumption-time
freshness and four backtests into the existing campaign costate state, SENSE,
DECIDE, blockers, and digest surfaces. It adds no parallel registry.

## Triality

- **DSL/typed leg:** CT1 is added to the existing campaign `SOURCES` registry;
  the four rows use existing organ schemas and authority firewalls.
- **DAG leg:** `FEED-708-co5` records the exact producer/backtest/consumer
  trajectory.
- **Equation leg:** the design laws and missing admission evidence are recorded
  in
  `.omx/research/ddm_co5_organ_telemetry_activation_20260725T195432Z/EQUATIONS.md`.
  No canonical equation or empirical anchor is registered because no
  backtest passed.

## Verification

Three consecutive clean passes completed after the final Python edit. Each
pass included:

- Ruff lint over every touched Python file;
- Python byte-compilation;
- `33 passed` across campaign-costate and organ focused suites;
- an exact human-digest smoke for the CO5 line;
- `git diff --check`.

Review-tracker pass IDs `ddm-co5-portable-clean-1`,
`ddm-co5-portable-clean-2`, and `ddm-co5-portable-clean-3` are recorded for
all three touched Python files. Tests model the settled fresh tree without
requiring the operator SSD to be mounted in CI, prove source-tree drift yields
`[stale-advisory]` with zero activation, and prove that removing CT1's explicit
batch-local authority label fails closed.

## STORES CONSULTED

- `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`,
  `PROGRAM.md`, and delegated authority prompt;
- top-10 Claude memory entries and current directive-consumption surfaces;
- canonical lane registry, lane maturity audit, subagent progress, per-arm
  inbox, and broadcast inbox;
- CO4 findings, SHA-pinned receipt, self-check, DECIDE, and bandit rows;
- CT1 findings, R6 receipt, six typed observability rows, and stopped-v5 source
  campaign tree;
- EV1 600/600 campaign evidence-join receipt;
- existing `tac.ddm_campaign_costate`, `tac.ddm_costate_organ`,
  `tools/costate_digest.py`, and focused tests.

## MAIN landing review requirement

MAIN must review the entire branch diff before merge, especially:

1. the CT1 consumption-time tree-fingerprint equivalence and
   `[stale-advisory]` failure path;
2. the exact six-row CT1 schema and the
   `ADVISORY_BATCH_LOCAL`/N600 authority wall;
3. the fact that EV1 cross-sectional rows cannot satisfy Bellman or temporal
   progress requirements;
4. all four `DESIGNED_NOT_ACTIVE` backtest reasons and blockers;
5. the digest's human/JSON visibility and unmodified `actuation=NONE`;
6. the operator broadcast correction that official leaderboard best, not
   local baselines, is the competitive frontier. This landing makes no
   frontier or pointer edit.

Until MAIN review and merge, this landing is local-only, non-promotable, and
cannot authorize execution.
