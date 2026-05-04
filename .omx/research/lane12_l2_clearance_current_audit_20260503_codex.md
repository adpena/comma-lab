# Lane 12 L2 Clearance Current Audit - 2026-05-03

Scope: Lane 12 NeRV / Alpha-Geo / pose-regeneration dispatch-prep only. No
remote GPU job was launched, no training was run, no exact eval was dispatched,
and no score or promotion claim is made.

## Current Evidence Refresh

Generated current local readiness artifacts:

```text
experiments/results/lane12_l2_unblock_readiness_20260503_current/lane12_l2_unblock_readiness.json
experiments/results/lane12_l2_unblock_readiness_20260503_current/decoded_baseline_build_preflight.json
```

Readiness summary:

```text
ready_for_retraining_unblock=false
ready_for_exact_eval_dispatch=false
eligible_to_create_clearance_packet=false
passing_geometry_count=0
usable_primitive_contract_count=1
usable_pose_regeneration_provenance_count=0
matched_alpha_geo_pose_candidate_count=0
decoded_baseline_contract_preflight_passed=true
runtime_closure.passed=true
```

The live `.omx/state/lane12_nerv_l2_clearance.json` packet is absent.

## Minimum Artifact Needed

Real Lane 12 L2 clearance now requires, at minimum:

1. A candidate archive whose Alpha-Geo geometry JSON passes the promotion
   contract:
   `diagnostic=alpha_geo_0_nerv_geometry`,
   `score_evidence_grade=empirical`, `scorer_proxy=false`, full
   `1200x384x512` shape, `diagnostic_config.threshold_preset=promotion`,
   exact promotion thresholds, `pass_fail.overall_pass=true`, baseline and
   candidate archive SHA-256 custody, and candidate member resolving to
   `masks.nrv`.
2. A usable `alpha_geo_primitive_contract_v1` packet. This is currently green
   for decoded-baseline training.
3. Three clean review passes and an existing evidence path before
   `experiments/plan_lane12_l2_unblock.py --write-clearance-packet` may write
   `.omx/state/lane12_nerv_l2_clearance.json`.
4. For exact-eval dispatch readiness, a completed/custody-clean
   pose-regeneration provenance record whose `candidate_archive_sha256` matches
   the passing Alpha-Geo geometry record.

## Existing Candidate Check

Current Lane 12 CDO1 geometry-repair artifacts are not sufficient for L2
clearance. The largest local policy,
`experiments/results/lane12_geometry_gate_repair_candidate_20260503/lane12_geometry_gate_budget_16384b/archive.zip`,
records:

```text
candidate_disagreement_pixels_before=2902861
repaired_disagreement_pixels=20098
residual_disagreement_pixels_after=2882763
global_disagreement_after=0.01221874237060547
dispatch_gate.passed=false
```

This remains far above the promotion `global_disagreement_max=0.001` gate.
It is local empirical geometry evidence only.

## Dispatch Decision

No remote dispatch was made. The existing Alpha-Geo-0 pose-regeneration script
can produce the pose-regeneration exact-eval provenance class, but dispatching
it against the current jsonfix40/repair artifacts would not create L2 clearance
because no current candidate has promotion-grade Alpha-Geo geometry. A GPU job
that can only generate matching pose provenance for a geometry-red candidate
would spend compute without satisfying the fail-closed unblock contract.

Bounded command template, only after a geometry-passing candidate exists and a
fresh claim is accepted:

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id lane_12_alpha_geo0_pose_regen \
  --platform vast.ai \
  --instance-job-id <instance_id>:lane12_alpha_geo0_pose_regen_<timestamp> \
  --agent codex:gpt-5.5 \
  --predicted-eta-utc <utc_eta> \
  --status eval_dispatching \
  --notes "Alpha-Geo pose-regeneration provenance for geometry-passing Lane 12 candidate; no score claim until JSON custody"

.venv/bin/python scripts/launch_lane_on_vastai.py \
  --lane-script scripts/remote_lane_12_alpha_geo0_pose_regen.sh \
  --label lane12_alpha_geo0_pose_regen_<timestamp> \
  --prefer-fast-chip
```

The safer immediate non-GPU next step is to produce or identify a new
mask-changing Lane 12 candidate that can pass Alpha-Geo promotion thresholds
locally, then rerun:

```bash
.venv/bin/python experiments/plan_lane12_l2_unblock.py \
  --geometry-json <passing_alpha_geo_geometry.json> \
  --primitive-contract-json experiments/results/lane_12_nerv_20260430_codex_jsonfix40/alpha_geo_1_vs_pfp16_repair_regions_20260501T080036Z.primitive_contract.json \
  --clearance-evidence .omx/research/lane12_l2_clearance_current_audit_20260503_codex.md \
  --grand-council-clean-passes 3 \
  --write-clearance-packet \
  --output-json experiments/results/lane12_l2_unblock_readiness_20260503_after_green_geometry/lane12_l2_unblock_readiness.json
```

## Verification

Commands run:

```bash
.venv/bin/python experiments/plan_lane12_l2_unblock.py \
  --output-json experiments/results/lane12_l2_unblock_readiness_20260503_current/lane12_l2_unblock_readiness.json

.venv/bin/python experiments/preflight_lane12_decoded_baseline_build.py \
  --output-json experiments/results/lane12_l2_unblock_readiness_20260503_current/decoded_baseline_build_preflight.json

.venv/bin/python -m pytest \
  src/tac/tests/test_plan_lane12_l2_unblock.py \
  src/tac/tests/test_preflight_lane12_decoded_baseline_build.py \
  src/tac/tests/test_lane12_nerv_dependency_closure.py \
  src/tac/tests/test_remote_lane_12_alpha_geo0_pose_regen_script.py \
  src/tac/tests/test_build_lane12_geometry_gate_repair_candidate.py -q

git diff --check -- \
  experiments/plan_lane12_l2_unblock.py \
  experiments/preflight_lane12_decoded_baseline_build.py \
  scripts/remote_lane_12_alpha_geo0_pose_regen.sh \
  src/tac/tests/test_plan_lane12_l2_unblock.py \
  src/tac/tests/test_preflight_lane12_decoded_baseline_build.py \
  src/tac/tests/test_remote_lane_12_alpha_geo0_pose_regen_script.py \
  src/tac/tests/test_build_lane12_geometry_gate_repair_candidate.py
```

Result:

```text
37 passed in 2.37s
git diff --check passed
```
