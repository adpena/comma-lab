# DDM SN1 error-source tensor — DAG / FEED 603

Date: 2026-07-23  
Lane: `ddm_sn1_segnet_telemetry_asymmetry`  
State: advisory research instrument; `research_only=true`,
`execution_allowed=false`, `score_claim=false`

## Dependency DAG

1. SHA-pinned v19c strict n600 replay supplies 2,265,811 exact residual
   errors and per-batch camera/cell digests.
2. The current v19c semantic field and SHA-pinned DV1
   `spline_plus_events` semantic field determine the exclusive three-way
   source partition.
3. SN1 supplies ordered SDWL1 \(D_2\) thresholds and three receiver-realized
   inverse feasibility segments.
4. G3/G4 add pair-tail, scene-event, and historical recurrence covariates.
   G2 adds only a class-level energy/byte marginal with
   `NO_JOINT_PER_CELL_CUSTODY`. v14/e1 are cross-checks only.
5. Frozen SegNet and PoseNet relay hooks consume the same SHA-pinned receiver
   and GT batches. They emit bounded layer × channel × pooled-space × time
   products, while frozen weights independently emit conv/BN/SE/GELU/
   LayerScale analytic factors and exact resize-kernel custody.
6. The counted six-coordinate pose sidecar feeds the existing full-screw
   \(\xi\) mapping and ground-homography proxy for SegNet spatial transport.
   PoseNet keeps `xi_advected=NOT_REQUESTED`; no invented transport is added.
7. The typed n600 tensor reduces into the exact source/stratum budget,
   menu-ready cluster rows, and the amortized vocabulary-gap ranking.
8. The accepted product is finalized from a complete certified checkpoint
   tree and reproduces receipt SHA-256
   `ecf9f015fa6999b9bb7602c93027da713bb278389b92d5d1bf0b95f4ced19faa`.
9. Downstream consumers must remeasure receiver survival, Pose, and exact
   bytes before any candidate can enter a score or promotion decision.

## Deterministic execution and custody

- SegNet stored execution order is `encoder.model.conv_stem` through
  `segmentation_head`; PoseNet is `vision.stem` through
  `hydra.final_layer.pose`. Finalization reconstructs this order from the
  recorded integer `order`, never JSON object key order.
- Every long stage is resumable. Accepted stage checkpoints comprise 153
  files and 5,115,498,029 bytes with tree SHA-256
  `d477c65b1c6f80d5bf18fdf2f61c3c1795398f9b4dc606709c6b0722ce60c746`.
- Checkpoints were written to `/Volumes/VertigoDataTier/pact` and the canonical
  source path is a symlink. The final compact artifacts stay under the
  research output directory; superseded bulk has separate certify-or-block
  manifests.
- Stable rank uses deterministic float64 power iteration with finite checks.
  Final JSON/JSONL traversal refuses non-finite values.
- A finalize-only replay from the complete measurement bundle reproduced the
  accepted receipt byte-for-byte. Measurement implementation and finalizer
  implementation identities remain separate in the receipt.

## Unified-solver feed

- Sensitivity map: consume cluster error mass keyed by ordered pair, target
  stratum, sided \(D_2\) band, curvature band, scale, temporal proxy, G3 tail
  bucket, boundary-distance band, curve availability, and observable
  paint-floor mechanism. Per-cell errors remain localized only to the
  `segmentation_head`; scorer-native relay products are aggregate associations,
  never silently joined as per-cell hidden causes.
- Pareto constraint: no line item is admissible until exact
  paint→R→uint8→frozen-SegNet receiver survival, official Pose output, and
  exact serialized bytes are measured on the same candidate.
- Bit allocator: prefer vocabulary improvement over chart/parameter edits,
  and those over point correction. The first measured priority is the
  738,090-error semantic vocabulary gap at 1,610 shared bytes; the ratio is
  semantic reach, not realized score value.
- Cathedral autopilot: route `NEVER_DESCRIBED` rows to the DV-line round-2
  description design, `DESCRIBED_BUT_REALIZATION_LOST` rows to DR1
  receiver-closed placement/paint/prototype solves. Within DR1, route
  `COARSE_DESCRIPTION`, `PAINT_FUNCTION`, and `TEXTURE_PRIOR_REGION_ERF` to
  distinct actuators. Route only the measured 635,011-error leftover to #366
  descent scope.
- Relay selector: use `decoder.blocks.4` as the top SegNet advisory relay and
  `vision.head` as the top PoseNet advisory relay under the measured
  low-rank/forgiving score. This is a next-probe ranking only; intervention
  still requires receiver-closed Seg/Pose/rate measurement.
- Continual learning: the receipt, source-budget JSON, solve-menu JSONL, and
  vocabulary ranking, mechanism budget, #149 survival-wall record, and
  scorer-native/analytic products are the canonical empirical anchor. Do not
  remeasure source receipts whose SHAs are recorded there.
- Probe disambiguator: the exhaustive precedence rule separates semantic
  coverage, realization loss, and tested-program residual. Competing richer
  vocabularies must be added as explicit semantic fields and compared against
  the same frozen v19c residual, never silently relabel the third bucket.

## No execution authorization

This feed does not authorize training, remote/GPU work, archive mutation,
pointer movement, or promotion. MAIN must review the serializer commit before
landing. The next executable action is a receiver-closed solve probe for the
highest-mass menu entries, beginning with the top Undrivable→Road
coarse-description/no-continuous-Lane-curve/G3-tail cluster, with lane claim
and exact Seg/Pose/rate custody.
