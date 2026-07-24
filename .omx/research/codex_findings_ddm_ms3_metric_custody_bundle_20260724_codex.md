# Codex findings — DDM MS3 metric custody bundle

## Verdict

`PARTIAL_MISSING_FOUR_EXACT_MEASUREMENT_SURFACES`

Scope: INSTANCE × current landed producer/custody state. This is not a
FORMULATION, FAMILY, or PARADIGM negative. Evidence axis
`[macOS-CPU frozen-scorer advisory]`; `score_claim=false`; pointer
`0.1910828242 [contest-CPU]` unchanged; MAIN landing review required.

Bundle:
`.omx/research/ddm_ms3_metric_custody_bundle_20260724T035249Z/BUNDLE-PARTIAL.json`,
SHA-256
`22262887239846262fcc73001eab79baedd4b6d0b44ea5f1c0f8b8714d161f1b`.

## Findings are first rungs

1. CONFIRMED — PF2 and G3 can be reused exactly without rebuilding. PF2 has
   1,200 unique typed rows and exact SHA
   `85084f7bd3a03dbd1b9f04fe6a9b84df4948a6caf64620beef42da8924345f73`.
   G3 has exact SHA
   `0c9ce6d0ce2b2c0830400f096438355242527d40f682fc1b201f67d8d951a4e4`
   and the required top24 → top64 → control24 → full-n600 order. The loader
   rechecks file bytes, SHA, and JSON content schema at every consumption.
   Next measurement: preserve those exact inputs while measuring the hard
   blocks before easy mass.

2. CONFIRMED PARTIAL — Seg has no custodied full-n600 rank-4 row-Gram or
   lambda-range producer output for the 1,200 PF2 buckets. AT1X is contracted
   Seg energy, not a full row-Gram. Repository search found no other producer
   of the sealed `margin_fisher_gram` surface. Next measurement: emit a 4×4
   PSD margin-Fisher Gram, matching eigenspectrum, lambda interval and
   n600 count for every unchanged PF2 key.

3. CONFIRMED PARTIAL — Pose has no all-600 canonical-batch32 exact Pose6
   quadratic or active-tube convergence table. V16 covers eight pairs at
   batch16 and does not converge every attempted KKT row. Next measurement:
   produce ordered pair IDs 0..599 at batch32 with center, ≤6-rank factors,
   positive contest-budget tube radius, and explicit
   `NON_CONVERGED_*` states rather than interpolation.

4. CONFIRMED PARTIAL — composite-\(R\) has neither an all-bucket exact
   Hessian/adjoint readback nor paired realized secants. G2F is an n64 control
   and cannot be promoted to n600. Next measurement: emit the exact separable
   full-kernel model and equal-amplitude positive/negative receiver-realized
   secants side by side for every PF2 bucket at batch32.

5. CONFIRMED PARTIAL — matched Fisher/Euclidean signed cosine and relative
   norm rows do not exist for the 1,200 PF2 buckets. Euclidean-only evidence
   cannot fill this gap. Next measurement: measure both vectors on identical
   bucket/input custody and retain Euclidean only as
   `LABELED_CONTROL_ONLY`.

6. IMPLEMENTED, NOT MEASURED — the loader now validates all four strict
   COMPLETE schemas, hard-refuses drift, disallows Euclidean as primary Seg
   geometry, requires Pose non-convergence flags, and exposes a single
   completeness gate to `build_minimum_description_headline`. A synthetic
   contract fixture exercises 1,200 Seg/composite/dual rows and 600 Pose rows;
   those values are test data and grant no evidence authority. The registered
   equation `ddm_metric_custody_bundle_completion_v1` captures only the
   structural conjunction. Next measurement: land real component producer
   artifacts, replace PARTIAL with BUNDLE-COMPLETE, and rerun the same loader
   under MAIN review.

7. CONFIRMED — no runnable existing producer emits the four sealed scientific
   data schemas from the available checkpoints/caches; therefore invoking the
   frozen scorer would not, by itself, materialize the required Jacobian,
   Hessian, active-tube, and matched-readback products. No fake n600 run was
   substituted. Next action: MAIN assigns or composes the missing producer
   surfaces, then performs the authorized four-thread n600 batch32 measurement.

8. APPARATUS NOTE — this lane is registered at L1 with `impl_complete`,
   `strict_preflight`, and `three_clean_review` evidence. The global
   `lane_maturity.py validate` command still reports 110 pre-existing missing
   evidence paths in unrelated historical lanes; none names this lane. This arm
   did not rewrite or waive those records. Next maintenance action: the owner
   of the global lane registry reconciles those historical evidence paths.

## Durable apparatus

- Four SHA-bound PARTIAL component receipts, each with exact blockers and a
  next measurement.
- Strict typed loader and COMPLETE scientific validators.
- Read-only deterministic materializer with storage preflight.
- Minimum-description headline integration that suppresses metric authority
  for PARTIAL.
- Registered canonical completion law with import-tested callable.
- DAG FEED and directive-consumption table.

## STORES CONSULTED

- delegated authority, `CLAUDE.md`, `AGENTS.md`,
  `docs/operating_manual_craft_handoff.md`
- scorer-native doctrine points 1–9b
- MS2 receipt/findings/DAG and canonical equations
- PF2, G3, AT1X, V16, G2F, RD1 and canonical registry state
- per-arm and fleet broadcast inboxes

No HNeRV/PR95/110/128 lineage was used as calibration or carrier.
