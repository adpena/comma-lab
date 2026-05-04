# Lane12 L2 Unblock Readiness Redesign - 2026-05-02

Scope: fail-closed readiness tooling for Lane 12 / NeRV L2 unblock review. No
remote job was launched, no training was run, no score claim is made, and
`.omx/state/lane12_nerv_l2_clearance.json` was not written.

## Tool Added

`experiments/plan_lane12_l2_unblock.py` audits existing artifacts and emits a
deterministic JSON readiness packet. The default output path is:

```text
experiments/results/lane12_l2_unblock_readiness_20260502/lane12_l2_unblock_readiness.json
```

The tool validates the same launcher-level clearance criteria before reporting
any unblock readiness:

```text
lane_id in {lane_12_nerv_mask_codec, lane_12_nerv}
cleared_for_retraining_unblock = true
lane12_l2 = true
geometry_gate_passed = true
grand_council_clean_passes >= 3
evidence = one or more existing local evidence file paths
```

It then cross-checks that the cited artifact set contains:

- an Alpha-Geo geometry JSON with `diagnostic=alpha_geo_0_nerv_geometry`,
  empirical non-proxy evidence, full `1200x384x512` shape, and
  `pass_fail.overall_pass=true`;
- an `alpha_geo_primitive_contract_v1` packet with non-claim flags false and
  decoded-baseline shape custody;
- optional pose-regeneration provenance for `RUN_AUTH_EVAL=1` readiness.

The tool separates exact CUDA evidence, empirical geometry/contract evidence,
and missing prerequisites. Exact CUDA evidence can record a negative measured
implementation, but it cannot by itself clear Lane12 L2.

## Current Expected Outcome

The current local state is expected to remain blocked. The historical
`jsonfix40` Lane12 artifact is exact CUDA negative evidence for the measured
configuration and has empirical Alpha-Geo geometry failures. It should not be
used to create a clearance packet.

## Redesign Recipe

The next permissible Lane12 run must use decoded-baseline targets:

```text
GT_MASKS_SOURCE=decoded-baseline
DECODED_BASELINE_MEMBER=masks.mkv
ALPHA_PRIMITIVE_CONTRACT=<alpha_geo_primitive_contract_v1.json>
RUN_AUTH_EVAL=0 by default
```

The retired SegNet target path remains blocked by default. It is allowed only
for explicitly documented forensic/debug reruns through the existing
`ALLOW_RETIRED_SEGNET_TARGET=1` remote environment and trainer
`--allow-forensic-segnet-target` flag. Such reruns are no-clearance and
no-promotion evidence.

Geometry must be checked with Alpha-Geo promotion thresholds before any exact
eval spend:

```text
global_disagreement <= 0.001
boundary_band_disagreement <= 0.002 at radii 1, 2, 3, 5
stable_region_false_flip_rate <= 0.002
pair_transition_disagreement <= 0.002
class recall >= 0.999 for classes 1 and 2
tiny_speckle_rate <= 0.0001
max_component_centroid_jump_px <= 1.0
missing_component_rate <= 0.0
```

Exact eval custody after a build-only candidate requires candidate
pose-regeneration provenance, an Alpha-Geo provenance JSON with
`pass_fail.overall_pass=true` for the exact archive, archive SHA matches, and
canonical CUDA auth eval through `experiments/contest_auth_eval.py --device
cuda`.

## Verification Target

Focused verification for this tool is:

```bash
.venv/bin/python -m py_compile experiments/plan_lane12_l2_unblock.py src/tac/tests/test_plan_lane12_l2_unblock.py
.venv/bin/python -m pytest src/tac/tests/test_plan_lane12_l2_unblock.py
```

## 2026-05-02 Decoded-Baseline Contract Preflight

Follow-up infrastructure patch:

- `experiments/preflight_lane12_decoded_baseline_build.py`
- `src/tac/tests/test_preflight_lane12_decoded_baseline_build.py`
- canonical hash fix in `experiments/train_nerv_mask.py`

Bug class fixed: Alpha-Geo diagnostics canonicalized decoded mask hashes as
`torch.uint8`, while the NeRV trainer canonicalized the same decoded masks as
`torch.int64`. The live primitive contract therefore could be reported as
shape-usable by the readiness planner while still failing the trainer's exact
`decoded_mask_sha256_match` gate before training. The trainer now uses the same
uint8 canonical mask-hash contract for class IDs `<=255`; labels are still cast
to `long` at sampling time, so the training loss contract is unchanged.

New preflight artifact:

```text
experiments/results/lane12_l2_unblock_readiness_20260502/decoded_baseline_build_preflight.json
```

Live result:

```text
decoded_baseline_contract_preflight_passed=true
ready_for_build_only_remote_training=false
decoded_mask_sha256=cce3a986341c40df9b9ebca24ff96e16c4b41b40b388dc2af86161ba76e2b4e9
```

The contract, decoded baseline archive member, target mask SHA, and weighted
sampling scaffold are now locally coherent. Remote training remains blocked
because:

- `.omx/state/lane12_nerv_l2_clearance.json` is missing;
- there is no passing Alpha-Geo geometry JSON with
  `diagnostic=alpha_geo_0_nerv_geometry`, empirical non-proxy evidence, full
  `1200x384x512` shape, and `pass_fail.overall_pass=true`.

This tool performs no training, launches no remote job, writes no clearance
state, and makes no score claim. It is the build-only trainer-input preflight
that should run immediately before any future dispatch claim for
`lane_12_nerv_mask_codec`.

## 2026-05-02 Alpha-Geo Promotion-Threshold Evidence Guard

Follow-up infrastructure patch:

- `experiments/plan_lane12_l2_unblock.py`
- `experiments/preflight_lane12_decoded_baseline_build.py`
- `experiments/alpha_geo0_pose_regen.py`
- `experiments/modal_alpha_geo0_pose_regen.py`
- `scripts/remote_lane_12_alpha_geo0_pose_regen.sh`
- `scripts/remote_lane_nerv.sh`

Bug class fixed: Alpha-Geo pose-regeneration runners emitted
`alpha_geo_0_nerv_geometry` with `--threshold-preset exploratory`, while the
L2 unblock contract requires promotion-threshold geometry before any clearance
or exact-eval spend. The readiness planner previously accepted any
`pass_fail.overall_pass=true` geometry JSON, so a loose exploratory pass could
have been mistaken for clearance-grade geometry evidence.

New guardrails:

- L2 readiness now requires `diagnostic_config.threshold_preset=promotion` and
  exact promotion threshold values in the geometry JSON.
- Alpha-Geo local, Modal, and shell pose-regeneration paths now invoke
  `experiments/diagnose_nerv_geometry.py --threshold-preset promotion`.
- `scripts/remote_lane_nerv.sh` rejects `RUN_AUTH_EVAL=1` if
  `ALPHA_GEO_PROVENANCE` lacks promotion-threshold custody, even when
  `pass_fail.overall_pass=true`.
- The decoded-baseline preflight records the launcher guard for promotion
  geometry validation.

This still does not clear Lane12 L2, does not write
`.omx/state/lane12_nerv_l2_clearance.json`, and does not launch a remote job.
It makes the missing Alpha-Geo evidence path deterministic and fail-closed:
future geometry evidence must be produced under promotion thresholds before it
can satisfy readiness or authorize exact-eval dispatch.

## 2026-05-02 Contract/Runtime Closure Hardening Addendum

Scope: narrow Lane 12/NeRV unblock hardening only. No remote GPU job was
dispatched, no retraining was started, no SJ-KL or active Lightning state was
touched, no clearance packet was written, and no score claim is made.

Files changed:

- `experiments/plan_lane12_l2_unblock.py`
- `experiments/preflight_lane12_decoded_baseline_build.py`
- `src/tac/tests/test_plan_lane12_l2_unblock.py`
- `src/tac/tests/test_preflight_lane12_decoded_baseline_build.py`

Bug classes permanently guarded:

- Pose-regeneration provenance can no longer be satisfied by an arbitrary JSON
  placeholder. The readiness planner now discovers existing Alpha-Geo pose
  summaries and requires completion/custody gates before any exact-eval
  dispatch readiness.
- `alpha_geo_primitive_contract_v1` is now checked for decoded-baseline
  SHA-256, hash algorithm, dtype, `masks.mkv` source member, archive SHA, and
  expected `1200x384x512` shape, not only its diagnostic string.
- Decoded-baseline preflight now includes a runtime closure check proving the
  standalone packed-payload path has NeRV/QZS3/QP1 parsing support, masks.nrv
  inflate support, packed-payload summary emission, archive-default mask
  source, and TTO/scorer fallbacks default-off.

Regenerated artifacts:

```text
experiments/results/lane12_l2_unblock_readiness_20260502/lane12_l2_unblock_readiness.json
experiments/results/lane12_l2_unblock_readiness_20260502/decoded_baseline_build_preflight.json
```

Current live readiness:

```text
decoded_baseline_contract_preflight_passed=true
runtime_closure.passed=true
ready_for_build_only_remote_training=false
ready_for_retraining_unblock=false
ready_for_exact_eval_dispatch=false
usable_primitive_contract_count=1
usable_pose_regeneration_provenance_count=0
```

Why Lane 12 is still blocked:

- `.omx/state/lane12_nerv_l2_clearance.json` is missing.
- No Alpha-Geo geometry artifact passes the promotion-threshold contract:
  `diagnostic=alpha_geo_0_nerv_geometry`, empirical non-proxy evidence, full
  `1200x384x512` mask shape, `diagnostic_config.threshold_preset=promotion`,
  exact promotion thresholds, and `pass_fail.overall_pass=true`.
- Four existing pose-regeneration/Alpha-Geo summaries are now enumerated, but
  none is usable: the Lightning summaries failed before pose optimization and
  exact eval; the Modal summaries failed NVDEC/DALI CUDA preflight.

Verification:

```bash
.venv/bin/python -m py_compile \
  experiments/plan_lane12_l2_unblock.py \
  experiments/preflight_lane12_decoded_baseline_build.py \
  src/tac/tests/test_plan_lane12_l2_unblock.py \
  src/tac/tests/test_preflight_lane12_decoded_baseline_build.py \
  submissions/robust_current/unpack_renderer_payload.py

.venv/bin/python -m pytest \
  src/tac/tests/test_plan_lane12_l2_unblock.py \
  src/tac/tests/test_preflight_lane12_decoded_baseline_build.py \
  src/tac/tests/test_unpack_renderer_payload_fixedslice.py \
  src/tac/tests/test_lane12_nerv_dependency_closure.py -q
```

Result: `30 passed in 3.56s`; `git diff --check` on the touched Lane 12 files
and regenerated readiness artifacts passed.
