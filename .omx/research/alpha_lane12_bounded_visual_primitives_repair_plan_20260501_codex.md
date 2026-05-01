# Alpha Lane 12 Bounded Visual-Primitives Repair Plan - 2026-05-01

Scope: Lane 12 / Alpha repair planning from the bounded CPU visual-primitives
artifact. No MCP, paid dispatch, retraining, archive rebuild, CUDA auth eval,
score claim, L2 clearance packet, or exact-eval claim was created.

## Evidence Boundary

This plan uses empirical CPU tensor diagnostics only. It can reject the current
`jsonfix40` implementation/config and define the next repair specs, but it
cannot promote, rank, kill a family, or authorize exact-eval spend.

Score truth remains exact archive bytes through:

```text
archive.zip -> inflate.sh -> upstream/evaluate.py
```

## Input Artifact

```text
artifact = experiments/results/lane_12_nerv_20260430_codex_jsonfix40/alpha_geo_0_vs_pfp16_visual_primitives_bounded_20260501.json
diagnostic = alpha_geo_0_nerv_geometry
score_evidence_grade = empirical
device = cpu
baseline = experiments/results/lane_g_v3_pfp16/archive_lane_g_v3_pfp16.zip
baseline_member = masks.mkv
baseline_archive_sha256 = 0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f
baseline_decoded_mask_sha256 = cce3a986341c40df9b9ebca24ff96e16c4b41b40b388dc2af86161ba76e2b4e9
candidate = experiments/results/lane_12_nerv_20260430_codex_jsonfix40/archive_lane_12_nerv.zip
candidate_member = masks.nrv
candidate_archive_sha256 = 864549cc648f0b3a023076c11812ccd0f10b1d013ed3fd6bb24d20bbcde85c97
candidate_decoded_mask_sha256 = 5d8504ac2bb018a123fa238dbcb55615ca50278942c03bf49425df46023389b4
```

The artifact is non-promotable by construction:

```text
promotion_eligible = false
score_claim_eligible = false
exact_eval_claim = false
```

## Gate Failures

The current candidate fails both exploratory retrain review and exact-eval
spend review.

```text
global_disagreement = 0.012303928799099393
  exploratory max = 0.003
  exact-eval spend max = 0.001

boundary_2px_disagreement = 0.14883144511692872
  exploratory max = 0.005
  exact-eval spend max = 0.002

pair_transition_disagreement = 0.009507171571470149
  exploratory max = 0.004
  exact-eval spend max = 0.002

critical_missing_rate = 0.6452857808237408
  exploratory max = 0.02
  exact-eval spend max = 0.001

critical_missing_area_rate = 0.004038536840025019
  exploratory max = 0.01
  exact-eval spend max = 0.0005
```

The critical area rate is inside the loose exploratory threshold but still
blocks exact-eval spend. The missing-component count is the more severe Alpha
repair blocker.

## Failure Shape

Lane geometry is the primary failure mode:

```text
lane_marking_recall = 0.2115568938212039
lane_missing_components = 9665 / 13782
lane_missing_rate = 0.7012770280075461
lane_missing_area_rate = 0.39965308882756284
lane_boundary_baseline_coverage_at_2px = 0.3401439704827463
lane_boundary_bidirectional_chamfer_px = 19.824030938867615
lane_boundary_hausdorff_p95_px = 106.00774718604669
```

Vehicle area is mostly preserved, but component count still fails the strict
critical gate:

```text
vehicle_undrivable_recall = 0.9950934805331972
vehicle_missing_components = 33 / 1247
vehicle_missing_rate = 0.026463512429831595
vehicle_missing_area_rate = 0.0000042465333497772345
```

Temporal failure is mostly missed baseline transitions:

```text
transition_tp_pixels = 117766
transition_fn_pixels = 1936567
transition_fp_pixels = 304587
transition_f1 = 0.095099661402374
worst_pair_indices = 266, 267, 1035, 1044, 1034, 1045, 1039, 1036, 0, 1029
```

The bounded artifact skipped residual-region ranking with
`--residual-region-count 0`, so the next CPU-only diagnostic should materialize
repair boxes before any implementation work.

## Deterministic Repair Specs

1. `critical_component_recall_retrain`
   - Target the decoded baseline `masks.mkv`, not fresh SegNet argmax labels.
   - Oversample frames and boxes from `critical_box_failures`.
   - Weight class `1` and class `2` component interiors, with extra weight for
     pose-sensitive boxes where `box_y1 >= 0.60 * height`.
   - First success gate: `critical_missing_rate <= 0.02`.
   - Exact-eval spend review gate: `critical_missing_rate <= 0.001` and
     `critical_missing_area_rate <= 0.0005`.

2. `boundary_band_retrain`
   - Use decoded-baseline boundary bands at radii `1, 2, 3, 5`.
   - Prioritize lane and road boundary families because lane 2px coverage is
     only `0.3401439704827463`.
   - First success gate: `boundary_2px_disagreement <= 0.005`.
   - Exact-eval spend review gate: `boundary_2px_disagreement <= 0.002`.

3. `temporal_transition_retrain`
   - Train adjacent frame pairs, not just independent coordinates.
   - Oversample the worst transition pairs listed above.
   - Penalize false-negative baseline transitions separately from false
     positives because FN dominates the transition error.
   - First success gate: `pair_transition_disagreement <= 0.004`.
   - Exact-eval spend review gate: `pair_transition_disagreement <= 0.002`.

4. `decoded_baseline_global_overfit`
   - Keep uniform decoded-baseline CE as the base objective.
   - Do not lower component/boundary/temporal weights to chase byte count until
     all geometry gates pass.
   - First success gate: `global_disagreement <= 0.003`.
   - Exact-eval spend review gate: `global_disagreement <= 0.001`.

5. `charged_sparse_critical_residual_after_retrain`
   - Activate only after decoded-baseline retrain reduces global and boundary
     blockers while critical component blockers remain.
   - Source regions must come from `critical_box_failures` and a rerun
     `residual_region_ranking.regions` packet.
   - All residual side information is charged inside `archive.zip`.

## Currently Admissible Commands

CPU-only residual repair region materialization. This is diagnostic evidence
only and should not be used for score, promotion, rank, retirement, or L2
clearance.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/diagnose_nerv_geometry.py \
  --baseline experiments/results/lane_g_v3_pfp16/archive_lane_g_v3_pfp16.zip \
  --baseline-member masks.mkv \
  --candidate experiments/results/lane_12_nerv_20260430_codex_jsonfix40/archive_lane_12_nerv.zip \
  --candidate-member masks.nrv \
  --num-frames 1200 \
  --height 384 \
  --width 512 \
  --threshold-preset none \
  --mask-cache-dir experiments/results/lane_12_nerv_20260430_codex_jsonfix40/predecoded_mask_cache \
  --residual-region-count 200 \
  --residual-region-min-area 4 \
  --residual-region-boundary-radius 2 \
  --visual-component-classes 1,2 \
  --visual-disable-temporal-tracks \
  --visual-boundary-distance-sample-cap 128 \
  --visual-boundary-distance-global-sample-cap 8192 \
  --output-json experiments/results/lane_12_nerv_20260430_codex_jsonfix40/alpha_geo_0_vs_pfp16_residual_regions_20260501.json
```

After a new candidate exists, repeat the same diagnostic with the new archive
path and use `--threshold-preset exploratory` for the first gate. Expected
exit code is `0` only if exploratory geometry passes.

## Blocked Commands

Do not run Lane 12 retraining or exact eval from the current state. The L2
clearance packet is absent and the bounded visual-primitives artifact fails
geometry gates.

Build-only retrain template after a valid external L2 clearance packet exists:

```bash
RUN_AUTH_EVAL=0 \
GT_MASKS_SOURCE=decoded-baseline \
DECODED_BASELINE_PATH=experiments/results/lane_g_v3_pfp16/archive_lane_g_v3_pfp16.zip \
DECODED_BASELINE_MEMBER=masks.mkv \
scripts/remote_lane_nerv.sh
```

Exact eval remains blocked until a successor archive has passing Alpha-Geo
diagnostics, pose-regeneration provenance against the candidate mask stream,
deterministic archive custody, and required Grand Council review. No exact eval
command is admissible now.

## Code Hook Landed

`experiments/diagnose_nerv_geometry.py` now emits a
`visual_primitives.repair_retrain_spec` packet for future diagnostics. The
packet maps exact-eval spend blockers to the deterministic specs above and
separates currently admissible CPU-only diagnostics from commands blocked by
L2 clearance. The packet remains empirical/no-claim:

```text
promotion_eligible = false
score_claim_eligible = false
exact_eval_claim = false
l2_clearance_created = false
```

Focused verification:

```text
.venv/bin/python -m py_compile experiments/diagnose_nerv_geometry.py src/tac/tests/test_lane12_nerv_geometry_diagnostics.py
.venv/bin/python -m pytest src/tac/tests/test_lane12_nerv_geometry_diagnostics.py -q
18 passed
```

## L2 Status

L2 remains blocked. This plan does not create or justify
`.omx/state/lane12_nerv_l2_clearance.json`.
