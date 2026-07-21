# Advected motion base — n64 governed rate verdict

`lane_advected_motion_base_20260721` · `research_only=true` ·
`[macOS-CPU advisory]` · `score_claim=false` · `promotion_eligible=false` ·
`pointer=0.1910828242 [contest-CPU] UNMOVED` · `MAIN review required`

## Verdict

**MEASURED, FORMULATION-SCOPED RATE NEGATIVE.** Replacing the static pair base
with the counted planar-PPCS advected base increased exact solved-target
exception bytes by **180,280 B** over the preregistered n64 prefix:
19,739,340 B advected versus 19,559,060 B static. The strict n64 gate therefore
returned `N64_RATE_GATE_FAIL_STOP_BEFORE_N600`; n600 was not executed.

This rejects only the tested formulation: PPCS planar `(dy, ds, dpsi)`, a
transported-chart-selected ground-plane homography with source persistence
off-ground, and a lossless full-plane correction to the deterministic #549/C1
target. It does **not** kill depth-stratified, per-class, object-residual,
learned-flow, locally gated, or lossy scorer-aware exception families.

The measured failure mode is **ground-plane warp fidelity** against the
solved photometric target, not an absence of motion signal: advection improves
pre-correction d_pose by 2.659%, but worsens d_seg by 5.671%, increases the
mean number of differing RGB values by 4,251.609 per pair (0.834%), and needs
0.922% more correction bytes. It is better on only 11/64 pairs, equal on 3,
and worse on 50. That pattern queues depth/per-class/local gating; it does not
authorize an advection-family verdict.

## Built path and byte boundary

The executable receiver path is

`frame_1_base = W_xi(frame_0_base)`

and the scene chart follows

`chart_1 = first_argmax(W_xi(onehot(chart_0)))`.

The transported chart chooses the RGB branch: the G1 homography is used on
ground strata and the same-pixel source base is persisted off-ground. Thus no
foreground or sky pixel is falsely assigned ground depth.

`xi=(dy,0,ds,0,dpsi,0)` is decoded directly from the already-counted
`PPCS.trajectory` section (raw section SHA-256
`591baea6e83f13a2b99a157318f47cb58a0bdee5f6f310dda868c600d9c22add`).
The actuator adds **zero** video-derived bytes and performs zero decoder-side
scorer calls. The canonical `tac.lie`/G1 homography and existing
predict-project receiver are reused. The hash-pinned G1 receipt supplies only
projection geometry (`pitch_rad=-0.05`); its external nearest-target-pair motion
proxy is not consumed. No ep725, checkpoint, R1-dxi, or old-archive bytes are
consumed.

## N64 result

All values below are **MEASURED** with native CPU-Torch, seed 1234, four CPU
threads, and deterministic algorithms. This is an advisory rate experiment,
not a contest score.

| Surface | d_seg mean | d_pose mean | exact exception B | admitted at lambda | mean delta-S |
|---|---:|---:|---:|---:|---:|
| #549 solved target | 0.000126282374 | 0.000060044317 | 0 | n/a | n/a |
| static base | 0.007238626480 | 187.594818592 | 19,559,060 | 64/64 | -43.794756349 |
| advected base | 0.007649103791 | 182.606285095 | 19,739,340 | 64/64 | -43.253873327 |

The exact exception stream selects the smallest verified Brotli-11 encoding of
modular uint8 delta, xor delta, or target replacement. Each chosen payload is
decoded and checked for byte-exact target recovery. At
`lambda*=6.658589531221714e-7 S/B`, all 64 corrections are individually
admitted for both bases, but advection is still the more expensive base by
180,280 B and has the weaker aggregate delta-S. Pose-before-correction improves,
but pose is already solved by #549 and is not an open objective for this task.

The composed D4 rate rows make the terminal comparison explicit:

| Predict base + exact project correction | final d_seg | final d_pose | bytes added |
|---|---:|---:|---:|
| static + solved-target correction | 0.000126282374 | 0.000060044317 | 19,559,060 |
| advected + solved-target correction | 0.000126282374 | 0.000060044317 | 19,739,340 |

These final distortions are the n64 measured #549 target distortions after
byte-exact correction, not a new pose solve or official score.

## Exception bytes by counted |xi| bucket

The bucket edges are **DERIVED** once from the full n600 counted seed; the n64
prefix is then evaluated without redefining them. Consequently q2 has no prefix
members.

| Full-seed bucket | n64 pairs | static B | advected B | advected - static B |
|---|---:|---:|---:|---:|
| q1 `[0, 0.0428688381)` | 1 | 358,533 | 358,533 | 0 |
| q2 `[0.0428688381, 0.0433195633)` | 0 | 0 | 0 | 0 |
| q3 `[0.0433195633, 0.0465700895)` | 21 | 6,398,594 | 6,462,640 | +64,046 |
| q4 `[0.0465700895, 0.0522600930]` | 42 | 12,801,933 | 12,918,167 | +116,234 |

## Determinism, resumability, and storage

The GT cache is exact SHA-pinned ZIP_STORED mmap; only one pair is resident at
a time. Every pair writes an atomic immutable stage JSON, and every eight pairs
writes a distinct checkpoint. The successful tree contains 64 stage records
and eight checkpoints under
`/Volumes/VertigoDataTier/pact/evidence/advected_base_20260721/advected_base/`.
Receipt SHA-256:
`4a6650f2b1c8f8290a0a099fecf8fa7dd7a1d8826b5bfa72441cf553f1e676d0`.

The first attempt exposed an over-strict harness assertion at pair 7: target
replacement is legitimately base-independent, so two distinct bases may yield
the same compressed target payload. The per-arm byte-exact parseback checks
were already sufficient. The bad assertion was removed, regression-tested, and
all seven prior stage records were preserved without reuse at
`/Volumes/VertigoDataTier/pact/evidence/advected_base_20260721/failed_runs/invalid_target_replacement_assertion_f893065b3554c16edad4c013884fb85ebf05279b13122e80b87848428c558c5a/`.

A later fresh code review found that the first successful receipt applied the
ground homography outside the declared ground strata. That whole-plane result
(+278,252 B) was not deleted or silently promoted: it is preserved with a
machine-readable supersession manifest under
`/Volumes/VertigoDataTier/pact/evidence/advected_base_20260721/superseded_whole_plane_20260721T151700Z/`.
The headline rows in this memo come only from the corrected ground-stratified
rerun.

## Reproduction

```bash
PYTHONPATH=src /Users/adpena/Projects/pact/.venv/bin/python -u \
  tools/measure_predict_project_receiver.py --advected-base \
  --seed /Volumes/VertigoDataTier/pact/evidence/seed_compose_20260721/seeds/seed_compose_b2_loose.ppcs \
  --gt-cache /Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --upstream /Users/adpena/Projects/pact/upstream \
  --output-dir /Volumes/VertigoDataTier/pact/evidence/advected_base_20260721 \
  --pair-start 0 --pair-end 64 --chunk-size 8 --threads 4
```

## Triality and authority

- DSL/schema: `predict_project_counted_planar_xi.v1` and
  `predict_project_advected_base_measurement.v1`, additive and fail-closed.
- DAG: `.omx/research/advected_motion_base_DAG_FEED_20260721.md`.
- Equations/LawRefs: `ego_motion_cumulative_se3_bspline_v1`,
  `lane_band_ego_factorization_source_reparam_v1`,
  `lane_band_source_reparam_measured_resolution_v1`, and the hash-pinned G1
  geometry scalar resolutions.

No archive was built, no official evaluator ran, no contest score is claimed,
and the frontier pointer remains unchanged. Any landing on `main` requires a
fresh MAIN review of the isolated-worktree commit and its receipt custody.

## STORES CONSULTED

Delegated authority and PROGRAM; CLAUDE.md and AGENTS.md; v7.5/v8/v10 specs;
latest joint Seg/Pose #549 memo; PPCS B2 seed and decoder; predict-project
receiver/schema; canonical `tac.lie` and G1 worldsheet warp/receipt; frozen n600
GT cache; native upstream scorer modules and weights; lane registry, subagent
progress, operator directive inbox, current pointer, and SSD evidence tree.
