# Codex Tier-0 session summary — Task #578 predictor R2 miss delta

`lane_id=predictor_r2_missdelta` · `research_only=true` ·
`[macOS-CPU advisory]` · `score_claim=false` · `promotion_eligible=false` ·
`pointer=0.1910828242 [contest-CPU] UNMOVED` · `MAIN_REVIEW_REQUIRED=true`

## Landed

- Commit `2423da2faf8b2b85bc4ac9e3d39ba2f84323ff8a` implements the exclusive
  real-mask D1 decomposition, strict PBD1/PBS1 codecs, n64-only PRF1 predictor
  refinements, resumable n64/n600 measurement stages, and D4 KKT composer.
- The measurement receipt, findings, DAG feed, build spec, and reuse manifest
  are durable under `.omx/research/predictor_r2_missdelta_*_20260721*`.
- Fresh bulk custody is preserved at
  `/Volumes/VertigoDataTier/pact/evidence/predictor_r2_20260721/canonical_r2_20260721/`.
- Two clean review-tracker passes, Ruff, 21 focused/inherited tests, JSON
  parsing, CLI import/help, diff checks, and serializer post-commit hashes pass.

## Terminal verdict and blockers

`BOUNDARY_DELTA_BAR_MISSED`: exact PBD1 costs 2.950087653 bits/miss for
1,772,327 real boundary events, versus 0.365 required. The current dense-anchor
activity stream and offset stream each independently exceed the whole budget.
The negative is formulation-scoped.

The measured description-space KKT knee is 96,078 variable bytes and
d_seg=0.024453667535 under the delegated implied-base projection, but the
declared round-1 base is already 262,498 bytes and therefore conflicts with the
216,222-byte box. The Road refinement is globally d_seg-positive but spills
56,905 Lane cells; MAIN must review that facet before any reuse. No realization,
receiver closure, scorer evaluation, archive proof, or contest-axis score was
performed.

## Recommended next action for MAIN/Claude

Review and merge the commit only as research/advisory evidence. If continuing
the family, replace dense per-anchor activity with sparse contour gaps/runs and
partition coherent shapes surgically by class/stratum/component before another
measurement; do not tune the current probability model as though it could
close the measured 8.08x rate gap. Preserve the Lane spill as a hard full-facet
constraint and keep the pointer unchanged.

## STORES CONSULTED

Task #578 delegated authority; CLAUDE.md/AGENTS.md and v7.5/v8 contracts;
round-1 predictor artifacts; frozen n600 cache; Task #595 Lane packet; #557
range coder; #307 contour prior; canonical breakeven equation; lane registry;
arm and broadcast inboxes.
