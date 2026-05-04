# Lane 12 NeRV Unblock Worker - 2026-05-03

Worker scope: local-only Lane 12 unblock audit. No remote GPU, training, eval,
Modal, Lightning, or Vast dispatch was launched.

## Objective

Remove safe local infrastructure blockers for Lane 12 NeRV retraining readiness
without relaxing contest gates. If the remaining blocker is missing scientific
evidence rather than code, preserve a precise fail-closed unblock plan.

## Local Inspection

Inspected surfaces:

- `experiments/plan_lane12_l2_unblock.py`
- `experiments/preflight_lane12_decoded_baseline_build.py`
- `experiments/train_nerv_mask.py`
- `scripts/remote_lane_nerv.sh`
- `submissions/robust_current/unpack_renderer_payload.py`
- `submissions/robust_current/inflate_renderer.py`
- `submissions/robust_current/inflate.sh`
- `src/tac/tests/test_plan_lane12_l2_unblock.py`
- `src/tac/tests/test_preflight_lane12_decoded_baseline_build.py`
- `src/tac/tests/test_unpack_renderer_payload_fixedslice.py`
- `src/tac/tests/test_lane12_nerv_dependency_closure.py`

Findings:

- The reported NERV/QZS3 parser blocker is already removed in this tree.
  `unpack_renderer_payload.py` contains the `NRV1` fixed-slice parser for
  `masks.nrv + Brotli(QZS3 renderer) + Brotli(QP1 poses)` and content-validates
  QZS3/QP1 magic before accepting the split.
- The reported ALPHA_PRIMITIVE_CONTRACT blocker is removed for the current
  default decoded-baseline path. The primitive contract at
  `experiments/results/lane_12_nerv_20260430_codex_jsonfix40/alpha_geo_1_vs_pfp16_repair_regions_20260501T080036Z.primitive_contract.json`
  passes local trainer consumption gates.
- The local trainer/runtime closure is green: decoded-baseline mask loading,
  primitive-contract sampling pool construction, remote wrapper static guards,
  packed payload unpacking, `masks.nrv` inflate support, and strict scorer
  defaults pass preflight.
- The remaining blockers are evidence blockers, not parser/runtime blockers:
  no valid `.omx/state/lane12_nerv_l2_clearance.json`; no passing
  promotion-threshold Alpha-Geo geometry record; no usable completed
  pose-regeneration provenance for exact-eval dispatch.

## Local Implementation

Added a fail-closed deterministic L2 clearance writer to
`experiments/plan_lane12_l2_unblock.py`.

New CLI/API behavior:

- `--write-clearance-packet` requests state creation.
- `--clearance-evidence <path>` cites one or more existing evidence files.
- `--grand-council-clean-passes <n>` must be `>= 3`.
- `--clearance-lane-id` must be `lane_12_nerv_mask_codec` or `lane_12_nerv`.
- The packet is written only when all local prerequisites are true:
  promotion-threshold geometry pass, usable primitive contract, valid evidence
  paths, valid lane id, and sufficient clean-pass count.
- The packet records `score_claim=false`, `promotion_eligible=false`,
  `remote_job_launched=false`, geometry evidence, primitive-contract evidence,
  cited evidence paths, and deterministic command provenance.
- Current repo evidence does not satisfy the writer gates, so no clearance file
  was written.

Added focused tests in `src/tac/tests/test_plan_lane12_l2_unblock.py`:

- green synthetic evidence materializes a valid deterministic clearance packet;
- failed geometry refuses state write and leaves retraining blocked.

## Artifacts

Worker artifacts:

- `experiments/results/lane12_nerv_unblock_worker_20260503/lane12_l2_unblock_readiness.json`
  - SHA-256: `5644b896710c9f8b0d1e21939938e8c897d189e9e2c8c21b595411b431cd234a`
- `experiments/results/lane12_nerv_unblock_worker_20260503/decoded_baseline_build_preflight.json`
  - SHA-256: `65073508eefd8afb3c3a442281b0756436f5e9c941c54a5592b5dd8a6081b812`

Current local readiness summary:

- `decoded_baseline_contract_preflight_passed=true`
- `runtime_closure.passed=true`
- `launcher_guards.passed=true`
- `usable_primitive_contract_count=1`
- `passing_geometry_count=0`
- `usable_pose_regeneration_provenance_count=0`
- `ready_for_retraining_unblock=false`
- `ready_for_exact_eval_dispatch=false`
- `clearance_state_written=false`

Critical measured blocker values from existing Lane 12 geometry:

- global disagreement: `0.012303928799099393` versus promotion max `0.001`
- boundary 2px disagreement: `0.14883144511692872` versus promotion max `0.002`
- pair transition disagreement: `0.009507171571470149` versus promotion max
  `0.002`

Exact CUDA negative retained:

- archive bytes: `296478`
- score: `26.03719330455429`
- PoseNet: `49.7784996`
- SegNet: `0.03528685`
- hardware: RTX 4090 CUDA, diagnostic/non-promotion for current frontier work

## Verification

Passed:

```bash
.venv/bin/python -m py_compile \
  experiments/plan_lane12_l2_unblock.py \
  experiments/preflight_lane12_decoded_baseline_build.py \
  submissions/robust_current/unpack_renderer_payload.py

.venv/bin/python -m pytest \
  src/tac/tests/test_plan_lane12_l2_unblock.py \
  src/tac/tests/test_preflight_lane12_decoded_baseline_build.py \
  src/tac/tests/test_unpack_renderer_payload_fixedslice.py \
  src/tac/tests/test_lane12_nerv_dependency_closure.py \
  -q
# 36 passed in 3.80s

git diff --check -- \
  experiments/plan_lane12_l2_unblock.py \
  src/tac/tests/test_plan_lane12_l2_unblock.py \
  experiments/results/lane12_nerv_unblock_worker_20260503/lane12_l2_unblock_readiness.json \
  experiments/results/lane12_nerv_unblock_worker_20260503/decoded_baseline_build_preflight.json
```

## Remaining Blockers

Remote training is still blocked.

Minimum unblock sequence:

1. Produce or identify a Lane 12/Alpha candidate whose decoded `masks.nrv`
   geometry passes promotion thresholds against the decoded baseline:
   `diagnostic=alpha_geo_0_nerv_geometry`, `score_evidence_grade=empirical`,
   `scorer_proxy=false`, shape `1200x384x512`,
   `diagnostic_config.threshold_preset=promotion`, and
   `pass_fail.overall_pass=true`.
2. Record three clean adversarial/Grand Council passes in a local evidence file.
3. Run the planner with explicit clearance write:

```bash
.venv/bin/python experiments/plan_lane12_l2_unblock.py \
  --geometry-json <passing_alpha_geo_geometry.json> \
  --primitive-contract-json experiments/results/lane_12_nerv_20260430_codex_jsonfix40/alpha_geo_1_vs_pfp16_repair_regions_20260501T080036Z.primitive_contract.json \
  --clearance-evidence <lane12_l2_review.md> \
  --grand-council-clean-passes 3 \
  --write-clearance-packet \
  --output-json experiments/results/lane12_nerv_unblock_worker_20260503/lane12_l2_unblock_readiness_after_green_geometry.json \
  --force
```

4. Only after the clearance packet validates, a training dispatcher may claim
   the lane with `tools/claim_lane_dispatch.py claim ...` and run build-only
   NeRV retraining. Exact eval still requires candidate archive custody plus
   completed pose-regeneration provenance.

## Dispatch Verdict

No remote training is allowed now. The local infrastructure is materially
closer: parser/runtime/contract/preflight and deterministic clearance writing
are in place, but the scientific geometry gate remains red.

## 2026-05-03 Codex Hardening Addendum

Scope: local-only readiness hardening. No training, eval, remote GPU, dispatch
claim, or clearance packet write was performed.

Patch:

- `experiments/plan_lane12_l2_unblock.py` now requires Alpha-Geo geometry
  records to carry baseline/candidate archive SHA-256 custody and candidate
  member `masks.nrv` before they can pass the geometry gate.
- Pose-regeneration summaries now expose the candidate archive SHA used for
  pose regeneration.
- Exact-eval dispatch readiness now requires at least one passing Alpha-Geo
  geometry record and one usable POSE_REGEN provenance record matched by
  `candidate_archive_sha256`. Mismatched green records remain blocked under
  `alpha_geo_pose_regen_candidate_mismatch`.
- The planner CLI now reports the real `clearance_state_written` value.
- `docs/runbooks/alpha_lane12_large_move_next_actions.md` records the exact
  clearance and later exact-dispatch provenance criteria.

Current repo state after the hardening pass:

- `.omx/state/lane12_nerv_l2_clearance.json` is still absent.
- Current Lane 12 geometry remains red against promotion thresholds.
- Usable POSE_REGEN provenance count remains `0`.
- `ready_for_retraining_unblock=false`.
- `ready_for_exact_eval_dispatch=false`.

Operator clearance proof command after genuinely green evidence exists:

```bash
.venv/bin/python experiments/plan_lane12_l2_unblock.py \
  --geometry-json <passing_alpha_geo_geometry_with_source_sha.json> \
  --primitive-contract-json experiments/results/lane_12_nerv_20260430_codex_jsonfix40/alpha_geo_1_vs_pfp16_repair_regions_20260501T080036Z.primitive_contract.json \
  --pose-regeneration-provenance <matching_pose_regen_provenance.json> \
  --clearance-evidence <lane12_l2_review.md> \
  --grand-council-clean-passes 3 \
  --write-clearance-packet \
  --output-json experiments/results/lane12_l2_unblock_readiness_20260503_hardened/lane12_l2_unblock_readiness.json \
  --force
```

Verification commands from this addendum:

```bash
.venv/bin/python -m py_compile experiments/plan_lane12_l2_unblock.py
.venv/bin/python -m pytest src/tac/tests/test_plan_lane12_l2_unblock.py -q
.venv/bin/python experiments/plan_lane12_l2_unblock.py --output-json /tmp/lane12_l2_unblock_readiness_hardened.json --force
```

Full focused verification after the addendum:

```bash
.venv/bin/python -m py_compile \
  experiments/plan_lane12_l2_unblock.py \
  experiments/preflight_lane12_decoded_baseline_build.py \
  submissions/robust_current/unpack_renderer_payload.py

.venv/bin/python -m pytest \
  src/tac/tests/test_plan_lane12_l2_unblock.py \
  src/tac/tests/test_preflight_lane12_decoded_baseline_build.py \
  src/tac/tests/test_unpack_renderer_payload_fixedslice.py \
  src/tac/tests/test_lane12_nerv_dependency_closure.py \
  -q
# 42 passed in 4.69s
```
