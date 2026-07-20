# #400 design note — `erm_margin_topk_v1` exact-evaluation scheduler

**Date:** 2026-07-20 UTC  
**Status:** `DESIGN_ONLY / research_only=true / NO_EXECUTION_AUTHORITY`  
**Consumer:** #396/#400 MC-finisher; future #319 `K>1` candidate emission  
**Pointer:** `0.1910828242 [contest-CPU Linux x86_64]` **UNMOVED**

## Decision

Adopt a bounded candidate scheduler that ranks `K=128` proposals with the existing Fisher/margin and
Cole–Hopf target geometry, then exact-scores a guarded `k=8`. This is an exact-evaluation allocation change,
not a new accept objective. `DirectionPinnedPairLocalObjective` and complete archive bytes retain sole
accept/rollback authority.

## Contract

### Inputs

- deterministic proposal IDs and proposal bytes/parameter deltas;
- baseline frozen margins and class-pair/stratum IDs;
- corrected `flip_margin_step_law_v1` realization prediction (first-order + secant + bounded QP);
- candidate-specific exact archive bytes when cheap to obtain;
- ξ-conditioned Pose prediction when present, explicitly non-authoritative;
- fixed `K`, `k`, seed, scorer/source hashes, and candidate generator hash.

### Cheap key

For predicted target/rival margins `mhat_pj`, compute

`E_seg = sum_p tau*log(1 + sum_{j != y*_p} exp(-mhat_pj/tau))`.

Restrict to the edited cells plus the measured collateral neighborhood. Order by
`(E_seg, predicted_pose_debt, exact_archive_bytes, candidate_id)`. Never persist this as `S`, `d_seg`, or
an exact score.

### Selection

- six lowest keys;
- one lowest key from a class-pair/stratum absent from those six;
- one deterministic proposal-engine control;
- deduplicate, then fill any vacancy by the next lowest key;
- exact-score all eight in the canonical batch geometry and apply the existing monotone exact-S gate.

### Calibration/fallback

On a bounded frozen fixture, exact-score all `K` candidates and record `recall_exact_best@k`, exact regret,
rank components, wall split, and all candidate IDs. `k=8` is admissible only if every preregistered pool
contains its exact best in the selected set. Otherwise retry `k=16`; any miss then fails closed to full exact
evaluation. No proxy-only acceptance exists.

## Economics to verify

- Exact calls: `128 -> 8`, saving `120` or `93.75%`; call-count compression `16x`.
- #400 local completed-fixture wall anchor: `46.61348766786978/64 = 0.7283357448104653 s/call`.
- #454 cheap-action anchor: `0.006529812060762197 s` median.
- Mixed-anchor scenario only: all-rank plus top-8 exact `6.662501902251268 s` versus all-128 exact
  `93.22697533573956 s`, a conditional `92.849159%` reduction / `13.99278x` speedup.

The implementation receipt must replace that scenario with one matched end-to-end measurement and report
ranking, rendering, scorer, Pose, archive, and verification time separately.

## Safety and resumability

- Preserve the complete candidate manifest and rank rows before the first exact call.
- Checkpoint after ranking and after every exact batch; resume by content hash and next candidate ID.
- Retain every rejected exact result; no overwrite.
- Any missing/NaN margin, custody mismatch, rank-source drift, or resume mismatch selects full exact fallback.
- No learned energy, no paper-derived temperature/stopping constant, no proxy score claim.

## Acceptance test matrix

| Test | Required result |
|---|---|
| deterministic ranking | same inputs/seed -> byte-identical manifest and selected IDs |
| permutation stability | candidate input order changes neither keys nor selected IDs after ID tie-break |
| diversity guard | selected set includes the declared missing stratum when one exists |
| exact-best recall | full-control exact best appears in top-k on every preregistered calibration pool |
| failure fallback | NaN/missing margin, custody drift, or top-k miss routes to full exact evaluation |
| authority separation | cheap rank cannot call the accept path; only exact objective mutates the ratchet |
| resume | interruption after any exact batch resumes without duplicate/omitted candidate calls |
| accounting | actual exact-call count and complete wall split reconcile to the receipt |

## Triality and routing

- **DSL:** future typed selector config; N/A in this design-only landing.
- **DAG:** consumes the existing ERM -> #396/#400/#319 feed.
- **Equations:** consumes #542 Cole–Hopf target, `segnet_head_rank4_linear_flipdist_v1`,
  `flip_margin_step_law_v1`, and the exact score law; no new equation until measurement.
- **Probe disambiguator:** full-`K` control versus top-8/top-16 decides the selector; do not arbitrate by prose.

## MAIN review boundary

MAIN must confirm that the implementation ranks before any expensive scorer work, that the corrected
inner-Jacobian path—not naive first-order—is used, and that the full exact fallback is impossible to bypass.
