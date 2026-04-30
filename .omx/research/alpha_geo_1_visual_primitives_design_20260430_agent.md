# Alpha-Geo-1 Visual-Primitives Diagnostic Design - 2026-04-30

Author: Codex agent
Scope: research/specification only. No score claim. No code changes.

## Evidence Boundary

This document defines CPU mask-geometry diagnostics for Alpha-Geo-1. These
diagnostics may reject candidates, select retraining targets, and choose sparse
correction regions. They cannot promote, rank, retire a method family, or prove
a score.

Score truth remains exact CUDA auth eval on exact archive bytes through:

```text
archive.zip -> inflate.sh -> upstream/evaluate.py
```

The diagnostic packet evidence grade is `empirical`. If any diagnostic result
disagrees with exact CUDA auth eval, exact CUDA auth eval wins.

## Source Inputs

Primary local inputs:

- `experiments/diagnose_nerv_geometry.py`
- `experiments/results/lane_12_nerv_20260430_codex_jsonfix40/alpha_geo_0_vs_lane_g_v3.json`
- `experiments/results/lane_12_nerv_20260430_codex_jsonfix40/alpha_geo_0_vs_lane_a_base.json`
- `.omx/research/alpha_pose_preserving_redesign_spec_20260430_codex.md`
- `.omx/research/pufferlib_rl_visual_primitives_shannon_floor_20260430_agent.md`

Relevant existing diagnostic facts for Lane 12 `jsonfix40`:

| Metric | `jsonfix40` value | Exploratory gate below | Interpretation |
|---|---:|---:|---|
| Global mask disagreement | `0.012303928799099393` | `<= 0.003` | Fail |
| 1px boundary-band disagreement | `0.2086177911086304` | `<= 0.005` | Fail |
| 3px boundary-band disagreement | `0.11633853036183021` | `<= 0.005` | Fail |
| 5px boundary-band disagreement | `0.08223161952370056` | `<= 0.005` | Fail |
| Lane-marking recall | `0.2115568938212039` | `>= 0.995` | Fail |
| Vehicle/undrivable recall | `0.9950934805331972` | `>= 0.997` | Fail narrowly |
| Pair-transition disagreement | `0.009507171571470149` | `<= 0.004` | Fail |
| Pair-transition F1 | `0.095099661402374` | `>= 0.75` | Fail |
| Stable false-flip rate | `0.0013034438031416468` | `<= 0.004` | Pass alone |
| Missing component rate | `0.4611606740560512` | `<= 0.02` | Fail |
| P95 matched centroid jump | `155.69285700895662 px` | `<= 2 px` | Fail |
| Max matched centroid jump | `289.6654980546722 px` | `<= 2 px` | Fail |

The same values appear against both Lane G v3 and Lane A/base mask targets.
This is a geometry-preservation failure signal only; it is not a separate score
claim. The exact CUDA result remains the authoritative Lane 12 result.

## Design Goal

Alpha-Geo-1 must train and validate against the exact decoded baseline archive
mask stream, not fresh scorer-side SegNet labels. The visual-primitives layer
turns mask agreement into explicit deterministic state:

- boxes for connected components and object/lane regions,
- centroids for component position stability,
- lane and boundary polylines for road/lane geometry,
- temporal tracks for component and boundary continuity,
- pose-sensitive primitives for likely PoseNet failure modes.

This follows the visual-primitives research note: use explicit coordinates,
rule-based verifiers, and trajectory/coverage rewards as cheap diagnostics. It
does not adopt DeepSeek/PufferLib/RL as score evidence or runtime dependencies.
Bandit/PufferLib-style search may consume these metrics later as a surrogate
state only after exact-eval correlation is measured.

## Diagnostic Packet

Every Alpha-Geo-1 candidate should emit:

```text
alpha_geo_visual_primitives_v1.json
```

Required top-level fields:

```json
{
  "schema_version": 1,
  "diagnostic": "alpha_geo_visual_primitives",
  "score_evidence_grade": "empirical",
  "device": "cpu",
  "scorer_proxy": false,
  "source": {
    "baseline_archive": "...",
    "candidate_archive": "...",
    "baseline_member": "masks.mkv",
    "candidate_member": "masks.nrv or masks.mkv",
    "baseline_mask_sha256": "...",
    "candidate_mask_sha256": "...",
    "diagnose_nerv_geometry_json": "..."
  },
  "shape": {
    "frames": 1200,
    "height": 384,
    "width": 512,
    "num_classes": 5
  },
  "extraction_config": {},
  "existing_geometry_metrics": {},
  "primitive_metrics": {},
  "pass_fail": {},
  "next_action": {}
}
```

`existing_geometry_metrics` should embed or reference the current
`diagnose_nerv_geometry.py` output: global disagreement, class confusion,
boundary bands, temporal transitions, speckles, and centroid jumps.

## Deterministic Extraction Rules

All extraction is from integer class masks in `(T,H,W)` layout after archive
decode. No scorer network is loaded.

Class IDs:

```text
0 road
1 lane_marking
2 vehicle_undrivable
3 sky_or_movable
4 background
```

Coordinate convention:

- Pixel coordinate is `(x, y)`.
- Boxes are half-open `[x0, y0, x1, y1]`, where `x1` and `y1` are one past the
  last included pixel.
- Distances are in mask pixels.
- Component connectivity is 4-connected unless a config explicitly records
  `connectivity=8`.
- Component ordering is deterministic: frame index, class ID, decreasing area,
  then `y0`, `x0`, `y1`, `x1`.
- Ties are broken by the lexicographic tuple above.

Default extraction config:

```json
{
  "component_connectivity": 4,
  "critical_component_min_area": 9,
  "tiny_component_max_area": 8,
  "boundary_radii_px": [1, 2, 3, 5],
  "polyline_sample_step_px": 4,
  "polyline_match_radius_px": 2,
  "track_min_iou": 0.05,
  "track_max_centroid_jump_px": 16,
  "pose_sensitive_y_fraction_min": 0.60,
  "scanline_y_fractions": [0.55, 0.70, 0.85, 0.95],
  "scanline_x_fractions": [0.20, 0.35, 0.50, 0.65, 0.80]
}
```

Config changes are allowed only if recorded. Candidates with different configs
must not be compared as if they were identical.

## Boxes

For every frame and class, label connected components. For each baseline
component with area `>= critical_component_min_area`, record:

```json
{
  "frame": 0,
  "class_id": 1,
  "component_id": 0,
  "area_px": 123,
  "box_xyxy": [10, 20, 18, 85],
  "centroid_xy": [14.3, 53.2],
  "perimeter_px": 98
}
```

Candidate matching:

1. Restrict to same frame and class.
2. Compute overlap area between every baseline and candidate component.
3. Match each baseline component to the candidate component with largest
   overlap.
4. If all overlaps are zero, mark the component missing.
5. If two baseline components choose the same candidate component, record a
   merge; the primary match remains the largest-overlap baseline component.
6. If one baseline component overlaps multiple candidate components, record a
   split count.

Box metrics:

```text
box_missing_rate[class] = missing_baseline_components / baseline_components
box_area_missing_rate[class] = missing_baseline_area / baseline_area
box_iou_p05[class]
box_iou_p50[class]
box_center_shift_p95_px[class]
box_area_ratio_error_p95[class] = p95(abs(candidate_area / baseline_area - 1))
box_split_rate[class]
box_merge_rate[class]
critical_box_failures = sorted worst rows by class priority, then area desc
```

Critical classes are lane marking (`1`), vehicle/undrivable (`2`), and road
boundary-adjacent road components (`0` with boundary contact).

## Centroids

Centroids are computed exactly from component pixels:

```text
centroid_x = sum(x_i) / area
centroid_y = sum(y_i) / area
centroid_jump_px = sqrt((candidate_x - baseline_x)^2 + (candidate_y - baseline_y)^2)
```

Centroid metrics:

```text
centroid_jump_p50_px[class]
centroid_jump_p95_px[class]
centroid_jump_max_px[class]
centroid_over_1px_rate[class]
centroid_over_2px_rate[class]
missing_component_rate[class]
missing_component_area_rate[class]
```

For matched components, also compute normalized drift:

```text
centroid_jump_norm = centroid_jump_px / max(1, sqrt(baseline_area_px))
```

The normalized value prevents tiny components from dominating review, while
absolute pixel gates remain mandatory for lane and near-field pose-sensitive
objects.

## Lane And Boundary Polylines

Boundary pixel set:

```text
B_t = pixels whose 4-neighborhood contains at least one different class
```

Class-specific boundaries:

```text
lane_boundary = B_t intersect pixels where class is 1 or a neighbor is 1
road_boundary = B_t intersect pixels where class is 0 or a neighbor is 0
vehicle_boundary = B_t intersect pixels where class is 2 or a neighbor is 2
```

Polyline extraction:

1. Label connected boundary components.
2. Drop boundary components with fewer than `polyline_sample_step_px` pixels
   unless they touch a critical class.
3. Split graph nodes with degree not equal to 2 into separate simple paths.
4. Order each path from lowest `y` endpoint to highest `y` endpoint; if tied,
   lowest `x` first.
5. Resample every `polyline_sample_step_px` along path length.
6. Simplify only with a deterministic Ramer-Douglas-Peucker epsilon recorded in
   config. Default epsilon is `0`, meaning no simplification.

If full ordered tracing is not implemented yet, a valid first implementation
may emit sampled boundary point sets with `polyline_ordered=false`; metrics must
then be limited to set coverage and bidirectional distance, not endpoint order.

Polyline matching:

- Same frame and primitive family.
- Primary cost is bidirectional Chamfer distance.
- Tie-break by class ID, baseline path length descending, then first point.

Distance metrics:

```text
directed_chamfer(A -> C) = mean_a min_c ||a - c||_2
bidirectional_chamfer = 0.5 * (directed_chamfer(A -> C) + directed_chamfer(C -> A))
hausdorff_p95 = p95 over nearest-neighbor distances in both directions
coverage_at_1px = fraction of baseline polyline samples within 1 px of candidate set
coverage_at_2px = fraction of baseline polyline samples within 2 px of candidate set
endpoint_error_px = max distance between matched start/endpoints
```

Lane metrics:

```text
lane_polyline_missing_rate
lane_polyline_coverage_at_2px
lane_polyline_chamfer_p95_px
lane_endpoint_error_p95_px
lane_lower_third_endpoint_error_p95_px
```

Road/boundary metrics:

```text
road_boundary_coverage_at_2px
road_boundary_chamfer_p95_px
vehicle_boundary_coverage_at_2px
vehicle_boundary_chamfer_p95_px
boundary_band_1px_disagreement
boundary_band_2px_disagreement
boundary_band_3px_disagreement
boundary_band_5px_disagreement
```

Boundary-band disagreement should continue to come from
`diagnose_nerv_geometry.py` for continuity with the current Alpha-Geo-0 output.
The 2px band is added because the Alpha redesign spec names a 2px target.

## Temporal Tracks

Tracks are deterministic component chains, not learned optical flow.

Per adjacent frame pair, match baseline components to baseline components in
the next frame by:

1. same class,
2. highest IoU if IoU `>= track_min_iou`,
3. otherwise nearest centroid if distance `<= track_max_centroid_jump_px`,
4. otherwise terminate the track.

Apply the same rule to the candidate stream. Then compare candidate tracks to
baseline tracks through matched component IDs.

Track metrics:

```text
track_survival_recall[class] = candidate_matched_track_edges / baseline_track_edges
track_fragmentation_rate[class] = extra_candidate_track_breaks / baseline_track_edges
track_centroid_velocity_error_p95_px[class]
track_box_velocity_error_p95_px[class]
track_illegal_jump_rate[class]
track_birth_death_mismatch_rate[class]
```

Existing temporal mask metrics remain mandatory:

```text
pair_transition_disagreement_rate
pair_transition_precision
pair_transition_recall
pair_transition_f1
stable_region_false_flip_rate
worst_frame_pairs
```

The worst-frame-pair list is the entry point for targeted residuals and
training curriculum. Alpha-Geo-1 should always inspect the top 10 worst pairs
before spending retraining or exact-eval budget.

## Pose-Sensitive Primitives

These primitives are intended to catch mask damage likely to move PoseNet even
when global Hamming looks small.

### Near-Field Lane Endpoints

For every lane polyline sample with `y >= 0.60 * H`, record endpoint and local
tangent drift:

```text
near_lane_endpoint_error_p95_px
near_lane_tangent_angle_error_p95_deg
near_lane_coverage_at_1px
near_lane_coverage_at_2px
```

### Road Scanline Intercepts

At configured `scanline_y_fractions`, find road-boundary x-intercepts. At
configured `scanline_x_fractions`, find road-boundary y-intercepts. Record:

```text
road_scanline_x_intercept_error_p95_px
road_scanline_y_intercept_error_p95_px
road_scanline_missing_rate
```

This is a cheap proxy for drivable-region shape and horizon/lane geometry.

### Vanishing-Point Proxy

When at least two long lane polylines are present, fit straight lines to the
upper half of the left and right lane polylines using deterministic least
squares. Their intersection is a proxy, not a camera-calibration claim.

Record:

```text
vanishing_proxy_valid_frames
vanishing_proxy_error_p50_px
vanishing_proxy_error_p95_px
vanishing_proxy_missing_rate
```

This metric is diagnostic only and must not be overinterpreted when lane
polylines are absent or curved.

### Near-Field Vehicle/Undrivable Boxes

For class `2` components whose centroid has `y >= 0.50 * H` or whose box bottom
has `y1 >= 0.60 * H`, record:

```text
near_vehicle_box_missing_rate
near_vehicle_box_iou_p05
near_vehicle_center_shift_p95_px
near_vehicle_bottom_edge_error_p95_px
near_vehicle_area_ratio_error_p95
```

### Renderer Embedding Drift

If the renderer embedding path can be run without scorer-side shortcuts, record
framewise baseline/candidate embedding drift:

```text
renderer_embedding_l2_p50
renderer_embedding_l2_p95
renderer_embedding_delta_l2_p95
```

This is optional for the first packet but required before a large Alpha-Geo-1
training run. It is still diagnostic and non-promotable.

### CUDA Component Sensitivity Overlay

If a valid CUDA-authored `component_sensitivity_v1` packet exists, aggregate
PoseNet, SegNet, and combined sensitivity mass over each primitive. Record
these separately:

```text
pose_sensitivity_missed_mass_rate
seg_sensitivity_missed_mass_rate
combined_sensitivity_missed_mass_rate
sensitivity_weighted_boundary_disagreement
sensitivity_weighted_track_break_rate
```

CPU, MPS, fake, random, or proxy sensitivity maps must not populate these
fields except under an explicit `non_promotable_debug` namespace.

## Pass/Fail Gates

Use two gates: `exploratory_retrain_gate` and `exact_eval_spend_gate`.

Passing `exploratory_retrain_gate` means the candidate is plausible enough to
continue local retraining, residual planning, or pose regeneration. Passing
`exact_eval_spend_gate` means the candidate is plausible enough to build a
deterministic archive and consider exact CUDA eval. Neither gate promotes.

| Metric | Exploratory retrain gate | Exact-eval spend gate |
|---|---:|---:|
| Global disagreement | `<= 0.003` | `<= 0.001` |
| 1px boundary-band disagreement | `<= 0.005` | `<= 0.002` |
| 2px boundary-band disagreement | `<= 0.005` | `<= 0.002` |
| 3px boundary-band disagreement | `<= 0.005` | `<= 0.002` |
| 5px boundary-band disagreement | `<= 0.005` | `<= 0.002` |
| Stable false-flip rate | `<= 0.004` | `<= 0.002` |
| Pair-transition disagreement | `<= 0.004` | `<= 0.002` |
| Pair-transition F1 | `>= 0.75` | `>= 0.90` |
| Lane recall | `>= 0.995` | `>= 0.999` |
| Vehicle/undrivable recall | `>= 0.997` | `>= 0.999` |
| Tiny speckle rate | `<= 0.0005` | `<= 0.0001` |
| Missing component rate, all critical classes | `<= 0.02` | `<= 0.001` |
| Missing component area rate, lane | `<= 0.01` | `<= 0.0005` |
| Box center shift P95, critical classes | `<= 2 px` | `<= 1 px` |
| Box IoU P05, critical classes | `>= 0.75` | `>= 0.90` |
| Centroid jump P95, critical classes | `<= 2 px` | `<= 1 px` |
| Centroid jump max, critical classes | `<= 8 px` | `<= 2 px` |
| Lane polyline coverage at 2px | `>= 0.98` | `>= 0.995` |
| Lane polyline Chamfer P95 | `<= 2 px` | `<= 1 px` |
| Lane lower-third endpoint error P95 | `<= 3 px` | `<= 1.5 px` |
| Road-boundary coverage at 2px | `>= 0.98` | `>= 0.995` |
| Road scanline intercept error P95 | `<= 3 px` | `<= 1.5 px` |
| Track survival recall, critical classes | `>= 0.98` | `>= 0.995` |
| Track fragmentation rate, critical classes | `<= 0.02` | `<= 0.005` |
| Track velocity error P95, critical classes | `<= 2 px` | `<= 1 px` |
| Near-field vehicle missing rate | `<= 0.005` | `0` |
| Near-field vehicle bottom-edge error P95 | `<= 2 px` | `<= 1 px` |
| Renderer embedding L2 drift | tracked and improved vs `jsonfix40` | improved vs `jsonfix40` and monotone across latest retrain |
| CUDA sensitivity missed mass, if available | `<= 0.005` | `<= 0.001` |

Hard blockers for exact eval spend:

- Any score-affecting side information is outside `archive.zip`.
- Archive byte estimate is at or above the PFP16 A++ byte frontier unless a
  reviewed exact distortion-reduction justification exists.
- Candidate uses fresh SegNet targets without decoded-baseline target custody.
- Pose regeneration provenance is missing for candidates that changed masks.
- Diagnostic packet omits source archive SHA, candidate archive SHA, mask SHA,
  or extraction config.
- Any required primitive field is silently skipped. Skips must be explicit with
  a reason and must fail the exact-eval spend gate unless the primitive is
  provably absent in the baseline.

## Before Retraining

Run the primitive extractor on the decoded baseline `masks.mkv` before training
or continuing Alpha-Geo-1:

1. Decode the baseline archive mask stream.
2. Hash the decoded mask tensor or deterministic serialized mask stream.
3. Emit `baseline_visual_primitives_v1.json`.
4. Store extraction config and class histogram.
5. Use this primitive packet as the target contract for training, not fresh
   SegNet labels.

Training objective design should use the packet as supervision:

- standard mask loss against decoded baseline classes,
- extra weight on baseline boundary bands,
- extra weight on lane and near-field vehicle components,
- temporal pair-diff preservation,
- centroid and box consistency terms for matched critical components,
- optional renderer embedding drift term once implemented.

The weights are training hyperparameters, not evidence. The packet must record
them if they influence candidate generation.

## Before Sparse Corrections

If a NeRV/INR candidate fails primitive gates but is byte-plausible, use the
failure table to plan charged residual corrections.

Deterministic correction priority:

1. Missing lane-marking components in the lower two-thirds of the image.
2. Lane polyline endpoint and coverage failures.
3. Road-boundary scanline failures.
4. Near-field vehicle/undrivable missing boxes.
5. Worst temporal track breaks on the top 10 worst frame pairs.
6. High CUDA PoseNet-sensitivity missed mass, only if a valid CUDA sensitivity
   packet exists.
7. Remaining boundary-band disagreements by descending connected error area.

Every correction plan must record:

```text
residual_region_id
frame or frame range
class ID
primitive type
source failure metric
estimated uncompressed pixels
encoded side-info bytes
archive member that carries the correction
deterministic decode rule
```

No residual sidecar can be used for promotion unless the bytes are charged
inside `archive.zip` and the deterministic archive manifest records them.

## Before Pose Regeneration

Mask-changing Alpha candidates should not reuse old optimized poses without a
specific stale-pose isolation test. Before pose regeneration:

1. Require `exploratory_retrain_gate` pass, or require an explicit forensic tag.
2. Record baseline renderer SHA, candidate mask SHA, and primitive packet SHA.
3. Rebuild optimized poses against decoded candidate masks and the intended
   renderer.
4. Re-run primitive diagnostics after pose regeneration only if the mask stream
   changes. Pose files alone do not change mask primitive metrics.

If primitive gates fail badly, pose regeneration is likely wasted; fix geometry
or residuals first.

## Before Exact CUDA Eval

Exact eval spend is allowed only after:

1. `exact_eval_spend_gate` passes.
2. Candidate archive is built deterministically.
3. Archive byte estimate is below the current byte frontier or has reviewed
   exact distortion justification.
4. Payload closure is clean: no hidden files, resource forks, caches, traversal
   paths, absolute paths, or score-affecting sidecars.
5. Manifest records archive bytes, archive SHA-256, member bytes, member hashes,
   mask primitive packet SHA, source manifest, and pose provenance.
6. The planned command is:

```bash
.venv/bin/python experiments/contest_auth_eval.py \
  --archive <candidate archive.zip> \
  --inflate-sh submissions/robust_current/inflate.sh \
  --upstream-dir upstream \
  --device cuda \
  --keep-work-dir \
  --work-dir <evidence dir>
```

After exact eval, recompute score from components and preserve JSON, logs,
archive, manifest, hardware provenance, source/staged-tree manifest, and
adversarial review status.

## Bandit/PufferLib Use

The primitive packet may become a cheap surrogate state for bandit/BO search:

```text
state = bytes + global metrics + boundary metrics + class recall
      + box/centroid metrics + polyline metrics + track metrics
      + pose-sensitive metrics + optional CUDA sensitivity aggregates
```

Allowed actions:

- choose retraining target weights,
- choose boundary/residual budget,
- choose temporal correction density,
- choose pose regeneration versus geometry repair,
- decide whether to build an archive for exact-eval review.

Forbidden:

- direct PPO over exact CUDA eval,
- promotion from surrogate reward,
- broad Alpha/NeRV/INR/mask-compression retirement from diagnostic failures,
- using local model or RL output as an archive sidecar.

Graduation condition for any RL/PufferLib surrogate:

```text
top-20 surrogate ranking has stable positive Spearman correlation with exact
CUDA component deltas on >= 20 independently built candidate archives
```

Until then, use deterministic validators and small bandit/BO sweeps first.

## Implementation Notes

Minimal code path when implementation is approved:

1. Keep `diagnose_nerv_geometry.py` as the base.
2. Add a separate `experiments/extract_alpha_geo_visual_primitives.py` rather
   than enlarging exact-eval code.
3. Reuse its zip-slip-safe archive member read pattern.
4. Emit one JSON packet and no images by default.
5. Add focused synthetic tests for:
   - box missing/split/merge,
   - centroid drift,
   - lane polyline coverage,
   - temporal track fragmentation,
   - exact-eval gate fail-closed behavior.

No implementation should touch upstream scorer files or make score-affecting
runtime changes.

## Alpha-Geo-1 Decision Rules

Use this preflight sequence:

1. Baseline packet missing: stop and extract baseline primitives.
2. Candidate cannot decode deterministically: stop and fix archive/mask decode.
3. Existing `diagnose_nerv_geometry.py` exploratory gates fail by more than 2x:
   do not retrain long or exact-eval; inspect failures and redesign target.
4. Primitive gate failures localize to sparse regions: build charged residual
   plan and rerun diagnostics.
5. Primitive gate failures are broad over lane/road/temporal tracks: retrain
   against decoded baseline masks with primitive-weighted objective.
6. Exploratory gates pass but exact-eval spend gate fails: run short retrain or
   residual iteration; do not exact-eval yet unless explicitly forensic.
7. Exact-eval spend gate passes and byte/custody gates pass: build deterministic
   archive, regenerate poses if masks changed, then run exact CUDA auth eval.

For `jsonfix40`, this design would have stopped before exact-eval spend based
on lane recall, boundary disagreement, temporal transition F1, missing
components, and centroid jumps. That is the intended behavior for future
Alpha-Geo-1 iterations: reject geometry-collapse candidates cheaply, while
reserving exact CUDA eval for byte-plausible candidates that preserve decoded
baseline primitives.
