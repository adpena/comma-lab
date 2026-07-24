# Codex findings — DDM OF1 scorer-value oracle facade

Date: 2026-07-24
Lane: `ddm_of1_scorer_value_oracle_facade`
Authority: delegated facade-only task; no producer rebuild and no remeasurement
Verdict: **BUILT_AND_LOCALLY_VERIFIED; MAIN LANDING REVIEW REQUIRED**

## Outcome

`src/tac/scorer_value_oracle.py` is one typed, read-only admission-state API keyed
to the exact 21 row names in the DDM-366 dimension-completeness contract. It
rehashes every selected top-level artifact at consumption time. MS4D reads also
run the existing complete-bundle loader, revalidate the selected component data
artifact, and return the validated data payload with both manifest and component
lineage. External NPZ values stay lazy and are rehashed before a read-only member
memmap is opened.

Freshness behavior is caller-selected:

- `FAIL_CLOSED` raises `StaleProducerError` on missing bytes, SHA drift, schema
  drift, selector drift, or nested MS4D custody failure.
- `STALE_ADVISORY` returns no value and the exact tag `[stale-advisory]`.
- A producer absence is data, not an invented constant: `TYPED-GAP` plus a
  concrete reason and next action.

Verified coverage at consumption: **14 WRAPPED / 7 TYPED-GAP / 0 stale-advisory**.
`WRAPPED` means a live producer surface is exposed; the row's `authority_scope`
states any narrower-than-full-contract limitation and prevents coverage from
silently becoming scientific authority.

## COVERAGE report

| DDM-366 row | Status | Producer or exact gap |
|---|---|---|
| sub-pixel placement (874-res, pre-R) | WRAPPED | M2/#577 unrounded scorer-plane manifest. Scope is the scorer planes only; no machine-readable #149 camera-grid placement law is claimed. |
| stem-lattice phase (stride-2) | WRAPPED | DM4 scorer-recursive support config: sealed stride 2 and construction rule. |
| resize-kernel support / nullity | WRAPPED | #580 exact separable resize support, nullity, projector, and uint8 reachability receipt. |
| ERF neighborhood | WRAPPED | DM4 measured registered `r50=85 px` plus provenance. `r90` remains outside this producer's authority. |
| cell (Laguerre/argmax polytope) | WRAPPED | Frozen n600 `lstars` target-cache descriptor; external cache rehashed on member open. |
| stratum / class hyperplanes | WRAPPED | #583 frozen rank-4 valid-cell head prototypes and source custody. |
| frame roles | WRAPPED | M2/#577 distinct y0/y1 planes plus paired Pose6 source custody. |
| pair (pose screw, 6-of-12) | WRAPPED | MS4D `POSE_METRIC`: n600 six-scalar centers, low-rank quadratics, active tubes. |
| temporal / flicker | WRAPPED | G4 temporal-class and flip-mass decomposition; no receiver acceptance promotion. |
| clip (n600 stationarity) | WRAPPED | G4 n600 stationarity, concentration, and amortization data. |
| amplitude / uint8 deadzone | WRAPPED | MS4D `COMPOSITE_R_SECOND_ORDER`: direct intrinsic Hessians/readbacks. The returned producer explicitly says secants are not applicable without actuator input. |
| gain (SE state-dependence) | WRAPPED | DM1 L4 solved-value demand/margin rows. No complete gain-state response curve is claimed. |
| gain (normalization affine) | TYPED-GAP | The affine is source-inspected but has no dedicated sealed value artifact. |
| frequency / R passband | TYPED-GAP | #580 proves spatial support/nullity, not the frequency response or 3.2x along-tangent deficit. |
| YUV6 luma phases | TYPED-GAP | Pose custody contains scored targets/tubes, not the four named luma phase channels. |
| chroma pose-null | TYPED-GAP | Resize nullity cannot stand in for the joint Seg/Pose chroma-null intersection. |
| margin (Fisher surrogate) | WRAPPED | MS4D `SEG_METRIC`: all 1,200 PF2 margin-Fisher Grams, spectra, normals, and lambda ranges. |
| null/gauge energy | TYPED-GAP | #580 supplies `ker(A)` but no sealed composite artifact also carries the 52 percent head-gauge energy. |
| pose dims 7–12 | TYPED-GAP | MS4D intentionally carries the scored first six outputs, not a source-SHA-bound exclusion receipt for outputs 7–12. |
| rate (archive bytes only) | WRAPPED | MS5/MS6/RG3 1,200-way PF2 assignment table. Exact archive-byte pricing remains realized-rate caller authority. |
| score axes + weights | TYPED-GAP | No sealed, upstream-`evaluate.py`-SHA-bound score-functional artifact is present. |

## Consumer proof

`build_live_ddm_costate` now reads the PF2 assignment surface through
`ScorerValueOracle.bucket_assignments()`, records the observed producer SHA in
its input hashes, exposes the fresh row lineage under source custody, and reports
the facade coverage counts without claiming unverified freshness for unrelated
rows.

## Verification

- `58 passed`:
  `test_scorer_value_oracle.py`, `test_ddm_costate_organ.py`, and
  `test_ddm_campaign_costate.py`.
- Ruff: all four touched Python files clean.
- Live facade verification: 14 wrapped fresh, 7 typed gaps, 0 stale advisory.
- Oracle-specific suite exceeds the delegated minimum of 15 tests.
- No producer was rebuilt or remeasured; no job, GPU, archive, or score dispatch
  occurred.

## Triality and pointer honesty

- DSL: no lever or launch configuration changed; this is a read-only facade.
- DAG: existing producer artifact -> consumption-time hash/schema validation ->
  typed oracle row -> costate source custody.
- Equations: no equation was added or altered; DDM-366 remains the row-name
  authority and each producer retains its original scientific scope.
- Pointer: **UNMOVED**. No score claim and no promotion authority.

STORES CONSULTED: DDM-366 contract; MS4D custody bundle and existing loader;
M2/#577 plane manifest; #580 receipt; #583 prototype bank; DM1; DM4 config; G4;
PF2 assignment table; frozen target-cache receipt; lane registry; live delegation
inbox.

MAIN must adversarially review the branch diff and the seven typed gaps before
merging. In particular, MAIN should confirm that `WRAPPED` is interpreted as
surface availability under `authority_scope`, not full semantic completeness of
the DDM-366 row.
