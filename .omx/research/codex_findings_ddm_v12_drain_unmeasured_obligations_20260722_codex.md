# DDM V12 full obligation drain — resolved fork

**Lane:** `ddm_v12_drain_unmeasured_obligations`  
**Tasks:** #603 / #613 on master #578  
**Axis:** `[macOS-CPU frozen-scorer advisory]` — non-promotable  
**Authority:** `research_only=true`, `score_claim=false`, `d_seg_claim=false`, `d_pose_claim=false`  
**MAIN landing review:** **REQUIRED**

## Outcome

`ADVISORY_FORMULATION_PLATEAU_WITH_200KB_CEILING_NONBINDING_V6_SUCCESSOR_NAMED` fired. The
bounded n600 decision inventory is exhausted and the exact upper budget ladder is flat, but the
receiver-closed state remains 29.31× above the d_seg gate.

| requested added budget | exact total bytes | realized added bytes | d_seg | d_pose |
|---:|---:|---:|---:|---:|
| 0 | 102,105 | 0 | 0.034502249824 | 163.039648911962 |
| 16,384 | 106,106 | 4,001 | 0.034003668891 | 163.034719422881 |
| 49,152 | 106,106 | 4,001 | 0.034003668891 | 163.034719422881 |
| 98,304 | 106,106 | 4,001 | 0.034003668891 | 163.034719422881 |
| 147,456 | 106,106 | 4,001 | 0.034003668891 | 163.034719422881 |

The exact advisory objective improves 43.896380982393→43.848576571009 (Δ = -0.047804411384).
Waterfill admitted 44 bundles / 407 atoms: 377 Lane coefficient symbols, 29 boundary shearlet
atoms, and one xi-transported Movable shape atom. The 200,000-byte envelope was available but never
binding; no proposal was rate-rejected.

## Inventory and rank honesty

The exhaustive conflict-free partition covers all 4,096 bounded atoms in 353 bundles. Exact
dispositions are 338 scorer-measured bundles / 3,994 atoms, 10 strict receiver no-op bundles / 66
atoms, and five lower-EV address-conflict bundles / 36 atoms. Thus `decision_inventory_exhausted`
is true while `measurement_inventory_exhausted` is correctly false. No atom is silently called
scorer-measured.

Predicted EV was weak ordering authority: Pearson = 0.065620210489 and average-tie Spearman =
-0.234655546178 against exact measured objective gain across 338 measured bundles; 44 gains were
positive and 294 nonpositive. Exact canonical-batch replay, never predicted EV, governed admission.

## Final residual and successor

| stratum | base d_seg | final d_seg | disposition |
|---|---:|---:|---|
| Movable | 0.988264941023 | 0.989518086727 | dominant predictor-structural binder; worsened |
| Lane | 0.424611121005 | 0.436911324151 | second binder; worsened |
| Road | 0.071837475662 | 0.070367123841 | small incidental improvement |
| Undrivable | 0.005010949479 | 0.005065354915 | near-clean; slightly worsened |
| MyCar | 0.002402219760 | 0.001334655442 | incidental improvement |
| boundary codim-1 | 0.427522828370 | 0.427441397755 | essentially flat |
| cell interior | 0.024043482406 | 0.023533800583 | small improvement |

The negative is FORMULATION-scoped: post-solve correction of this bound v6
`fixed_ar1_hold24` 0.0345 predictor is dominated. The chart/event/carrier families and the
describe-line paradigm remain open. The named successor is a v6 successor whose PREDICT stage
natively carries Movable island worldsheet birth/death events; correction must not be asked to
create a worldsheet absent from the predictor.

## Law, custody, and review

`ddm_describe_line_rate_distortion_bracket_v1` gained an append-only fourth anchor bound to the n600
receipt. Its old n64→n256 projection remains historical DERIVED evidence; the new n600 row is
`MEASURED_N600_RECEIVER_CLOSED`. Primary receipt SHA-256 is
`eab2ef2478fb07f6a3242781887442c3fc49e9c34e10bd73a93f25d9a0262f0a`; the terminal archive SHA-256
is `623f860e78a105a8a31efc46d92002597d32e3459447cfed642c81243441ed11`.

Twelve resumable commands cover the full pool. The slowest command was 491.129356 s; cumulative
command time was 3,983.912482 s. Each command was under ten minutes, and all candidate/base/budget
checkpoints are preserved. The 62.680853 s `wallclock.total_seconds` in the primary receipt is the
terminal invocation only; cumulative custody is the sum of its bound invocation rows.

Round-1 fixed four issues before sealing: average ranks for ties; exact separation of scorer,
receiver-reject, and conflict dispositions; non-binding-ceiling wording; and exact-budget-rung
flattening instead of a last-admission count. Eighteen optimization tests and seven canonical-law
tests pass; three clean review marks follow the last source edit.

## Bounded re-derivation

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/adpena/Projects/pact/.venv/bin/python tools/run_ddm_v9_carrier_compose.py --config .omx/research/configs/ddm_v12_obligation_n600_20260722.json --output-directory .omx/research/ddm_v12_obligation_n600_20260722T161517Z
```

A completed replay validates and returns the sealed receipt immediately. A fresh drain is resumable
in at most 32-bundle commands; measured maximum command time is 491.129356 s.

## STORES CONSULTED

- `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, and `docs/operating_manual_craft_handoff.md`
- v7.5/v8 operating specs; V7/V8/V9/V10/V11 DDM findings, receipts, DAG rows, and equation history
- V6 predictor archive/receipt; frozen target receipt/cache; SegNet/PoseNet custody
- `reports/latest.md`, lane registry, canonical task status, subagent progress, and canonical equations registry
- per-arm inbox (empty) and fleet inbox through `2026-07-21T13:15:53Z`; both 2026-07-19 EV/Fisher directives consumed

No SSD bytes were written, no paid dispatch ran, and no scorer weights, GT table, or pixel stream
entered an archive. `0.1910828242 [contest-CPU]` remains unchanged.
