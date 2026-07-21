# Predictor R2 miss-delta build specification

`task=578` · `lane_id=predictor_r2_missdelta` · `research_only=true` ·
`axis=[macOS-CPU advisory]` · `score_claim=false` ·
`pointer=0.1910828242 [contest-CPU] UNMOVED` ·
`MAIN_REVIEW_REQUIRED=true`

## Objective

Recompute the merged round-1 predictor on the exact real n64 and n600 mask
cache, partition every predictor miss into an exclusive structural class, and
measure whether predictor-known contour deltas make the boundary-jitter class
affordable. Then measure bounded n64-fit predictor refinements and compose the
description-space rate/distortion curve, including an explicit eaten remainder.
This task does not edit or execute PROJECT realization and does not produce an
`upstream/evaluate.py` score.

## Owned files and collision boundary

- Add `src/tac/optimization/predictor_r2_missdelta.py`: deterministic
  decomposition, strict counted boundary-delta sidecar, n64-fit refinements,
  resumable stages, and composed curve.
- Extend `tools/measure_predictor_upgrade_xi_chart.py` with the round-2 stages;
  preserve all existing round-1 CLI behavior.
- Add focused tests under `src/tac/tests/` and dated receipt, findings, DAG FEED,
  and reuse-manifest artifacts under `.omx/research/`.
- Do not edit `predict_project_receiver.py`, receiver realization, inverse-R,
  score execution, frontier pointers, upstream, main, sibling worktrees, or live
  run directories. No change to `predict_project_schema.py` is planned: the new
  sidecar is independently strict and its decode contract is carried by this
  research-only module.

## D1 structural decomposition

For every recomputed round-1 miss, use only the target mask for measurement and
the predicted field for receiver-known geometry. A miss is classified in this
exclusive order:

1. `boundary_delta_1_2px`: its `(predicted,target)` class pair occurs on a
   predictor boundary, and a deterministic nearest compatible contour anchor
   places the miss at Chebyshev distance one or two from that anchor. This
   definition is chosen because the sidecar can replay the correction from the
   predicted field plus the counted event stream.
2. `coherent_blob`: among remaining misses, the 8-connected same-target-class
   component has at least four cells.
3. `scattered_incoherent`: every remaining miss.

Report exact counts and fractions per class and round-1 stratum, plus per-class
raster run-length histograms, compatible contour-adjacency pairs, component-size
histograms, and Euclidean distance-to-any-predicted-boundary histograms. Empty
rows stay explicit. The n64 row is `MEASURED_DEVELOPMENT_PREFIX`; the n600 row is
`MEASURED [macOS-CPU advisory]`.

## D2 strict boundary-delta sidecar

The receiver deterministically re-enumerates predictor boundary anchors by
class pair, connected component, and #307 straightness-first traversal. The
counted stream records only anchor activity and exact target sites/classes.
Adaptive arithmetic contexts include:

- arc-length phase bin of the deterministic contour traversal;
- canonical predicted/target class pair;
- local curvature bin derived from adjacent traversal directions; and
- prior activity/offset symbol for digital-straightness persistence.

All model state is causal and generic. The sidecar carries a canonical header,
four range-coded streams, lengths, and a checksum; it rejects malformed,
truncated, trailing, noncanonical, or baseline-mismatched input. Decode must
reconstruct the exact class-(a) corrected field and re-encode byte-identically.
Report payload and full-container bits per corrected miss versus the binding
`0.365 bits/miss-pixel` box-fit bar, per class and stratum as well as aggregate.

## D3 bounded predictor refinements

Fit every refinement on n64 only and apply unchanged to n600.

- Lane: decompose false negatives/positives into boundary-delta, phase-bin, and
  nonvisible/interior rows. Evaluate a bounded phase-bin correction table over
  receiver-known Lane-chart contour contexts; count the finite table bytes.
- Road: report misses by predicted class and by `boundary_delta`, lower-field,
  horizon, and Movable-shadow contexts. Evaluate a bounded context table keyed
  by predicted class, row band, and boundary status; count its bytes.

Candidates are nested and measured in order. Stop adding a candidate when its
saved exception bits, priced by the measured D2 coder, do not exceed its counted
table bits. The stop is formulation-scoped and does not kill either predictor
family.

## D4 composed description-space curve

Starting from the round-1 predictor and each admitted D3 refinement, compose:

1. predictor/chart/static counted bytes already bound by round 1;
2. exact decoded class-(a) boundary-delta sidecar bytes;
3. class-(b) shape bytes measured with the reused #307 contour-string coder on
   the exact coherent-blob maps; and
4. zero bytes for the explicitly eaten class-(c) set.

The description `d_seg` is remaining mask disagreements divided by
`600*512*384`; it is not through-R d_seg. Report hard-box prefixes at 216,222
bytes and the score-price KKT comparison using
`realization_breakeven_bytes_v1`, rate price `25/37,545,489`, and flip quantum
`100/(600*512*384)`. No mask pixel is described as a realized scorer recovery.

## Resumability, disk hygiene, and acceptance

- SSD root:
  `/Volumes/VertigoDataTier/pact/evidence/predictor_r2_20260721/`.
- n64 and n600 stages are chunked by independent pair masks. Every chunk is
  atomically written and source/config-hash bound; resume refuses drift.
- Preserve durable chunks and sidecars. True scratch uses context-managed temp
  directories. No bulk deletion or move occurs without a reproducibility
  manifest, so cleanup defaults to keep/certify.
- Cache, round-1 module, G1 receipt, Lane packet, predictor charts, contour coder,
  and canonical-equation source hashes are bound into stage custody.
- Acceptance: focused pytest; exact sidecar parse/decode/re-encode; n64 then n600
  completion; receipt JSON parse and invariants; Ruff; py_compile;
  `git diff --check`; two clean `review_tracker` passes for every changed Python
  file; serializer commit with expected post-edit SHA-256; final inbox read and
  delegation checkpoint.

## STORES CONSULTED

Delegated authority and both live inboxes; `CLAUDE.md`; `AGENTS.md`; `PROGRAM.md`;
craft handoff; v7.5/v8 specifications; round-1 build spec, code, receipt,
findings, DAG FEED, and reuse manifest; #307 contour coder and n600 finding;
#557 context-arithmetic implementations; #595 Lane chart custody; G1 receipt;
`s2_partition_seed`; `realization_breakeven_bytes_v1`; lane/task/subagent state;
frozen n600 cache; SSD evidence state.
