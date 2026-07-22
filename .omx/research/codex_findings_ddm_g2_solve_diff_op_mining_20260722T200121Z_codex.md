---
schema: codex_findings.v1
lane_id: ddm_g2_solve_diff_op_mining_20260722
review_round: 1
reviewer: codex:gpt-5.6-sol
evidence_axis: "[macOS-CPU frozen-scorer advisory]"
research_only: true
score_claim: false
main_landing_review_required: true
---

# Round-1 findings — DDM G2 solve-minus-predict operator mining

## Disposition

`PASS_AFTER_FIX`, advisory analysis only.  The reusable instrument completed all
600 pairs in 50 immutable stages on the SSD tier and emitted typed pair,
stratum, costate, tolerance, temporal-feature, start-receipt, and temporal-window
JSONL plus five PNG/HTML chart pairs.  Independent validation rehashed all
163,800 stage rows and 95,770,073 member bytes.  No scorer was loaded, no archive
was emitted, no dispatch ran, and no score or promotion is claimed.

Durable aggregate: `ddm_g2_solve_diff_op_mining_n600_20260722T194000Z/aggregate_ledger.json`.
Full resumable stages remain on
`/Volumes/VertigoDataTier/pact/ddm_g2_solve_diff_op_mining_n600_20260722T194000Z`.

## Finding 1 — the differential is mostly resize-visible

The exact factor-2 projector split of solve-minus-predict camera energy is:

- range: `2.537778030775775e12` (`0.6168064528841017`);
- kernel: `1.576605045000225e12` (`0.38319354711589826`);
- reconstruction max absolute error: `0`; maximum orthogonality residual:
  `4.237517714500427e-08`.

This is materially different from treating the previously settled nullity
dimension as an energy forecast.  The bounded-uint8 primitive reachability count
is reported separately and is not equated with either fraction.

## Finding 2 — active debt is MyCar/Road-heavy in residual energy, not Lane-only

Across the exclusive active target-class partition (persistent plus birth), the
scorer-plane residual energy is:

| class | active energy | share |
|---|---:|---:|
| MyCar | 289,075,446,986 | 56.40% |
| Road | 144,402,946,073 | 28.18% |
| Movable | 30,478,385,479 | 5.95% |
| Lane | 26,307,028,053 | 5.13% |
| Undrivable | 22,249,537,270 | 4.34% |

This is endpoint RGB-plane residual energy, not conditional d-seg.  It therefore
does not contradict V12's measured conditional failure on Movable/Lane and must
not be used as an evaluator ranking by itself.

## Finding 3 — persistence is not held-out xi predictability

Lane and Movable have high one-step compact-chart persistence (`0.9472660` and
`0.9921018` mean explained energy), but the held-out linear xi chart is negative
on average (`-0.0084498` and `-0.0114872`).  Under the declared strict 0.5 support
match, Movable partitions into 65 unmatched birth cells, zero xi-matched
post-birth cells, and 2,008 unmatched post-birth cells.

The oft-repeated `0.989518086727` number is V12's measured Movable
**target-class d-seg** at rungs 1--4.  It is not a predictable-after-birth
fraction.  Treating it as the latter would be a custody error.

## Finding 4 — byte ranking gives candidates, not KKT admission

Measured real-coded payload sizes rank:

1. compact parabolic shearlet: `92,544 B`;
2. xi-transport window residuals: `252,112 B`;
3. rank-four head chart: `351,900 B`;
4. full irreducible residual: `757,769,836 B`.

All four remain `BLOCKED_NO_RECEIVER_DELTA_DSEG`.  The exact resize adjoint and
rank-four head quotient produce reusable coefficient costates, but the frozen
SegNet inner encoder Jacobian is absent.  Recommendation: perturb the measured
compact-shearlet and rank-four coordinates through a receiver-closed path and
rank the resulting realized delta-S/byte; byte rank alone does not license a
carrier.

## Finding 5 — both endpoint sensitivities are now explicit

START: V12's only realized archive change, `102,105 -> 106,106 B`, improves
global d-seg by `0.000498580933` and d-pose by `0.004929489081`, while conditional
Lane and Movable d-seg worsen.  The receipt does not identify which predictor
component caused those column changes, so component attribution remains
`NOT_IDENTIFIABLE_FROM_RECEIPT`.

END: five deterministic retained-energy rungs were priced and labeled
`DERIVED_TOLERANCE_LADDER`; none is an evaluator measurement.  Real coder bytes
are non-monotone under simple amplitude scaling, so these rungs are not a
receiver R-D curve.

## Late Lane/Movable addenda

The 19:16--19:26Z operator hypotheses are preserved as scoped implementation
blockers, not silently answered with proxies:

- Lane phase symbols: no receiver-closed phase perturbation/delta-d-seg;
- Lane 16-channel stride-2 skip band: no internal activation custody in the
  scorer-free path;
- Lane BEV jitter/openpilot grammar: no worldsheet correspondence in the typed
  stage rows;
- Movable projective tracks/templates: no track identity or receiver-closed
  projective reconstruction.

These families remain open.  The measured persistence-versus-xi split is the
available rung-1 signal.

## Round-1 implementation corrections

- Bound production to at most 12 pairs and made each chunk a complete,
  write-once, source-hashed stage.
- Used the V12 receiver's actual scorer-plane geometry and applied the same exact
  realization to both endpoints.
- Replaced dense temporal accumulation with compact stage features and loaded
  cached labels/margins into Fisher costates.
- Replaced an Apple NumPy warning-prone normal-equation matmul with explicit
  finite-checked contractions; the 50 immutable producer stages were not
  rewritten.
- Added a fail-closed finalizer override that binds the prior producer SHA,
  refuses missing/mixed stages, and records separate producer/finalizer hashes.
- Fixed git worktree provenance so `git_sha` resolves through `commondir` instead
  of becoming null.

Verification: Ruff clean; py_compile clean; focused suite `16 passed`; full n600
finalization warning-free.  Producer module SHA
`7290303774d6bfbb0a4f5686b598fa2cea1722490774d107cf91c5ea6facfe1e`;
finalizer module SHA
`2d073d97d512fdf0829441ee2fe328de08fe8ae15b3302e68cd951bec2419213`;
receipt SHA `ada87717b39bc34ad67a3104d652574e544d82938fa3a1ea898acdf624c2bd67`.

## Stores consulted

Authority prompt and its SHA; CLAUDE.md; AGENTS.md; v7.5/v8 operating specs;
PROGRAM; latest Codex/Claude design and council memos; reports/latest.md; lane,
subagent, task, frontier, probe, and posterior state; C1 solved-plane receipt;
V12 archive/receipt; GT cache; G2G2 and describe-line DAG history; #233 jitter
surfaces; exact resize, factorized-adjoint, Fisher/head, costate, and temporal
coding implementations.

Pointer remains `0.1910828242 [contest-CPU]`.  This branch requires MAIN landing
review.
