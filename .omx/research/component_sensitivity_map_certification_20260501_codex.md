# Component Sensitivity Map Certification - 2026-05-01

Scope: promotion path from CUDA component-sensitivity maps to
`component_sensitivity_v1` manifests. This is an engineering/scientific
hardening ledger, not score evidence.

## Evidence Boundary

Current `profile_component_sensitivity.py` maps are diagnostic Fisher/proxy
artifacts even when authored on CUDA. They may guide prediction, perturbation
planning, and debugging, but they cannot be promoted or consumed by OWV3/NWCS
as authoritative sensitivity.

Promotion requires a distinct certification artifact:

```text
raw eligible CUDA direct finite-difference maps
  + official CUDA archive-response curves
  + stability/sample-plan/baseline custody
  + >=3 clean review passes
  -> certified tac_score_sensitivity_map_v1
  -> component_sensitivity_v1 manifest
```

Do not edit or strip source-map metadata. Source maps remain immutable and are
referenced by SHA-256.

## Code Landed

- `experiments/certify_component_sensitivity_maps.py`
  - certifies only CUDA maps with
    `sensitivity_source="direct_renderer_cuda_finite_difference_component_response"`;
  - rejects Fisher/proxy/debug/smoke/random/non-CUDA maps;
  - checks full absolute 600-pair sample coverage;
  - rechecks stability gates;
  - requires official response curves for PoseNet, SegNet, and combined with
    canonical archive eval path, passed gates, no blockers, same-run zero, and
    prediction-error pass;
  - writes new certified map files without mutating source maps.
- `src/tac/sensitivity_map.py`
  - added certified-map metadata validator and writer helpers.
- `experiments/build_component_sensitivity_manifest.py`
  - now rejects raw diagnostic maps and clean-but-uncertified maps for
    promotion assembly.
- `src/tac/component_sensitivity_artifact.py`
  - now validates `component_maps.*.certification` inside promotion manifests.

## Certification Metadata

Certified maps carry `component_sensitivity_map_certification_v1` metadata:

```json
{
  "format": "component_sensitivity_map_certification_v1",
  "component": "posenet|segnet|combined",
  "device": "cuda",
  "official_component_response": true,
  "canonical_scorer_path": true,
  "promotion_eligible": true,
  "source_map_sha256": "...",
  "official_response_curve_sha256": "...",
  "stability_sha256": "...",
  "sample_plan_sha256": "...",
  "baseline_archive_sha256": "...",
  "baseline_archive_bytes": 686635,
  "contest_auth_eval_json_sha256": "...",
  "review_clean_passes": 3,
  "response_gate_results": {},
  "stability_gate_results": {}
}
```

## L40S R7 Response Forensics

Job:
`official_component_response_pfp16_a_plus_plus_20260501_r7_predicted_from_r2_l40s_codex_lightning_l40s`

Harvest:
`experiments/results/lightning_batch/official_component_response_pfp16_a_plus_plus_20260501_r7_predicted_from_r2_l40s_codex_lightning_l40s`

Classification: `diagnostic_cuda_official_component_response`, not promotable.

Observed blockers:

- `prediction_error_gate_failed` on all components.
- Same-run eps=0 reproduced internally, but the same archive on L40S produced
  component values that drift from the PFP16 A++ T4 baseline JSON. This is a
  runner/scorer calibration hazard and blocks promotion.

Hardening landed after this forensic observation:

- `experiments/profile_component_sensitivity_official.py` now records and
  gates `external_baseline_repro` when an external baseline JSON is supplied
  and same-run eps=0 is used. External-baseline drift now adds
  `external_baseline_reproduction_failed`.
- Certifier rejects official curves with `external_baseline_repro=false`.

## T4 R2 Sensitivity

Job `component_sensitivity_pfp16_a_plus_plus_cuda_fisher_20260501_r2`
completed and was harvested at:

`experiments/results/lightning_batch/component_sensitivity_pfp16_a_plus_plus_cuda_fisher_20260501_r2`

CUDA/T4, 600 pairs, 480/120 calibration/holdout, elapsed
`600.3931622969999s`, baseline SHA
`0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`,
bytes `686635`. Evidence remains diagnostic Fisher/proxy only:
`promotion_eligible=false`, `official_component_response=false`.

## Verification

```bash
.venv/bin/python -m py_compile \
  experiments/certify_component_sensitivity_maps.py \
  experiments/build_component_sensitivity_manifest.py \
  experiments/profile_component_sensitivity_official.py \
  src/tac/sensitivity_map.py \
  src/tac/component_sensitivity_artifact.py

.venv/bin/python -m pytest \
  src/tac/tests/test_sensitivity_map.py \
  src/tac/tests/test_component_sensitivity_artifact.py \
  src/tac/tests/test_build_component_sensitivity_manifest.py \
  src/tac/tests/test_certify_component_sensitivity_maps.py \
  src/tac/tests/test_profile_component_sensitivity_official.py -q
```

Observed focused results:

```text
64 passed
58 passed
```

Alpha-Geo contract worker verification:

```text
src/tac/tests/test_lane12_nerv_geometry_diagnostics.py: 19 passed
```

## Next Gates

1. Harvest the running T4 official response job when terminal.
2. Manually inspect `external_baseline_repro` for the old-code T4 response if
   it was submitted before this hardening landed.
3. If T4 response fails prediction gates, fit an R8 signed/quadratic response
   model from official curves and rebuild fresh prediction deltas.
4. Do not certify maps until the source maps are direct finite-difference CUDA
   maps, not Fisher/proxy diagnostics.
5. Do not unlock OWV3/NWCS promotion until a certified
   `component_sensitivity_v1` exists.

## 2026-05-01T06:55Z Certification Supersession

Supersedes the initial optional-custody language: prediction-delta and
perturbation-basis custody are now mandatory for certification.

Landed code requirements:

- `experiments/certify_component_sensitivity_maps.py` requires
  `--prediction-deltas-json` in
  `official_component_response_prediction_deltas_v1` format.
- It requires `--perturbation-basis-json` in `perturbation_basis_v1` format,
  `basis_kind=archive_byte_additive`, canonical response eval path, and
  baseline archive SHA/byte match when `source_archive` is present.
- It cross-checks prediction atom IDs against basis atom IDs and requires
  matching epsilon ladders.
- It requires each official curve's perturbation metadata to cite the same
  prediction-deltas SHA and perturbation-basis SHA.
- If the official response summary cites an external baseline JSON, each
  curve must have `gate_results.external_baseline_repro=true`.

Current r7 T4/L40S response packets are therefore forensic only. They can
inform R8 model fitting, but cannot certify maps because they fail prediction
gates and show external-baseline drift relative to the PFP16 A++ anchor.

## 2026-05-01T07:10Z Diagnostic Direct-FD Handoff Tightened

Supersedes the earlier Fisher-only harvest assumption:

- Diagnostic component-sensitivity harvest validation now accepts exactly two
  sources: `fisher_proxy` and
  `direct_renderer_cuda_finite_difference_component_response`.
- Both sources remain `promotion_eligible=false` and `score_claim=false`.
  Direct-FD is only `certification_handoff_eligible=true`; Fisher/proxy is
  planning-only.
- Validation now loads every `*_sensitivity_map.pt` artifact and checks map
  metadata for CUDA device, component identity, allowed source,
  `score_claim=false`, `promotion_eligible=false`,
  `official_component_response=false`, and `canonical_scorer_path=false`.
- `experiments/profile_component_sensitivity.py` now writes explicit
  `score_claim=false` fields into diagnostic summaries and response curves to
  make the invariant machine-checkable.

Open implementation requirement:

- Direct-FD map generation must be sharded and merged deterministically before
  serious GPU spend. A valid single job exists, but it is not optimal wall-clock
  because full-channel finite differences are expensive.

## 2026-05-01T07:55Z Direct-FD Shard Implementation And Dispatch State

Landed implementation:

- `experiments/profile_component_sensitivity.py` supports deterministic
  direct-FD channel sharding via `--finite-difference-shard-index` and
  `--finite-difference-shard-count`, records `finite_difference_shard`
  metadata, and emits calibration plus holdout map files.
- Lightning component-sensitivity submit and validation now pass/validate
  shard metadata and holdout maps. Partial shards are explicitly
  non-handoff/non-promotable.
- `experiments/merge_component_sensitivity_shards.py` provides a lightweight
  post-harvest merge path that does not reload renderer/scorer/video inputs.
  Strict mode requires full non-overlapping shard coverage; incomplete mode
  remains planning-only and non-handoff.
- `experiments/modal_component_sensitivity_shards.py` provides a lightweight
  Modal A10G/T4 fallback using the same 16-shard topology and PFP16 archive
  custody. Modal direct-FD remains diagnostic only.

Verification:

```bash
.venv/bin/python -m py_compile \
  experiments/profile_component_sensitivity.py \
  src/tac/deploy/lightning/batch_jobs.py \
  scripts/launch_lightning_batch_job.py \
  scripts/lightning_repro_workspace.py \
  experiments/merge_component_sensitivity_shards.py \
  experiments/modal_component_sensitivity_shards.py \
  src/tac/tests/test_profile_component_sensitivity.py \
  src/tac/tests/test_lightning_batch_jobs.py \
  src/tac/tests/test_lightning_repro_workspace.py \
  src/tac/tests/test_merge_component_sensitivity_shards.py

.venv/bin/python -m pytest \
  src/tac/tests/test_profile_component_sensitivity.py \
  src/tac/tests/test_lightning_batch_jobs.py \
  src/tac/tests/test_lightning_repro_workspace.py \
  src/tac/tests/test_merge_component_sensitivity_shards.py -q
```

Observed focused results:

```text
113 passed
105 passed
```

Dispatch state:

- Lightning waves submitted: 16x L40S, 16x T4, 16x RTX PRO. Latest refresh
  path:
  `.omx/state/lightning_refresh_direct_fd_allwaves_20260501T074759Z.jsonl`.
  All were still `Pending` at that refresh.
- Modal A10G fallback submitted: label
  `pfp16_direct_fd_modal_a10g_20260501`; call IDs in
  `experiments/results/modal_component_sensitivity/pfp16_direct_fd_modal_a10g_20260501/modal_call_ids.json`.
  Initial recovery showed all shards pending.

Certification rule remains unchanged:

- These direct-FD maps can become certification handoff inputs only after full
  deterministic merge, full 600-pair sample-plan validation, prediction-delta
  and perturbation-basis custody, official archive-response curves,
  external-baseline reproduction, and adversarial review. They are not score
  evidence.

## 2026-05-01T08:15Z Execution Backend Hardening And Live Queue Update

Backend custody hardening:

- Lightning SSH is now treated strictly as transport, not proof of GPU
  availability. Interactive CUDA work requires a fresh runtime probe; exact
  sensitivity/eval work remains assigned to Batch Jobs with explicit machines.
- Added `scripts/configure_lightning_ssh.py` and tests so a new operator
  machine can reproduce the hardened SSH alias without using the Lightning UI
  helper's permissive host-key policy.
- Added the locked `cloud` extra for `lightning-sdk`, `modal`, and `vastai`.
  This closes the bug class where `uv sync --locked` could silently remove the
  provider CLIs/imports needed to harvest active experiments.

Live direct-FD telemetry:

- Latest Lightning refresh:
  `.omx/state/lightning_refresh_direct_fd_allwaves_20260501T081337Z.jsonl`.
- Status: L40S shards `00,01,03,04,05,07` running; L40S shards
  `02,06,08,09,10,11,12,13,14,15` pending; all T4 fallback shards pending;
  all RTX PRO fallback shards pending.
- Modal fallback r3:
  `experiments/results/modal_component_sensitivity/pfp16_direct_fd_modal_a10g_20260501_r3/modal_call_ids.json`.
  Recovery shows all 16 pending; no failures and no artifacts harvested yet.

Alpha diagnostic feed into sensitivity/repair planning:

- Alpha-Geo-1 wrote a 1000-region primitive-contract packet:
  `experiments/results/lane_12_nerv_20260430_codex_jsonfix40/alpha_geo_1_vs_pfp16_repair_regions_20260501T080036Z.json`
  and `.primitive_contract.json`.
- SHA-256 values:
  `fcc7fcf9e22518cd95a5af4cb36aff189249c6248b5298f377ecc8ca66991a3e`
  and `e5da815b680ba5c02bf653dae8c77b4f6d12500461e45b06d0cfb0881be5c16e`.
- This packet is CPU empirical and cannot promote/rank/kill. It can inform
  Alpha repair geometry and finite-difference perturbation design once CUDA
  component-sensitivity evidence lands.

## 2026-05-01T09:35Z Direct-FD Merge, Prediction Guard, And Response-Only Plans

Full direct-FD map status:

- Mixed fast merge completed with exactly-once 717/717 channel coverage:
  `experiments/results/lightning_batch/component_sensitivity_pfp16_direct_fd_mixed_l40s_rtxpro_modal_20260501_complete_merge_20260501T091302Z`.
  It combines L40S, RTX PRO, and one Modal shard; it is planning/diagnostic
  only.
- Homogeneous Modal A10G r3 completed all 16 shards and merged exactly-once:
  `experiments/results/modal_component_sensitivity/pfp16_direct_fd_modal_a10g_20260501_r3_complete_merge_20260501T0929Z`.
  This is an independent backend cross-check, still diagnostic/non-score.

Top-channel agreement:

- Mixed combined top channels: `renderer.head.weight` channels 2, 1, 0, then
  `renderer.stem_res.conv1.weight` channel 0 and
  `renderer.stem_res.conv2.weight` channel 4.
- Modal A10G combined top channels: `renderer.head.weight` channels 2, 1, 0,
  then `renderer.fuse_conv.weight` channel 0 and
  `renderer.stem_res.conv1.weight` channel 0.
- This confirms the head/stem/fuse sensitivity geometry is real enough for
  perturbation selection, but not yet byte-unit prediction.

Adversarial correction landed:

- Grand Council review found the generated prediction deltas were
  mathematically unit-inconsistent: direct-FD maps measure channel weight-space
  response per RMS weight perturbation, not response per archive byte delta.
- `experiments/build_component_response_prediction_deltas.py` now refuses to
  produce archive-byte `predicted_delta` artifacts from direct-FD maps unless
  map metadata explicitly records `archive_byte_prediction_eligible=true`.
- `experiments/build_component_response_plan_from_sensitivity_artifacts.py`
  now supports response-only plans with
  `--response-only-no-prediction-deltas`; these plans are for official CUDA
  response calibration only and cannot pass promotion certification by
  prediction gates.
- Explicit perturbation bases are now fail-closed against baseline archive
  custody: source archive SHA/bytes must match and any recorded atom
  `original_byte` must match the baseline member byte.
- Merged-shard validation now cross-checks summary and validation merge
  payloads, source-shard custody, source indices, and assigned-channel SHA
  vectors.

Clean response-only official-response plan packets:

- Mixed response-only plan:
  `experiments/results/official_component_response_pfp16_direct_fd_mixed_complete_response_only_20260501`.
  Plan SHA-256:
  `d0026cf7ec5f22ff3de4e8402d3f3a660767eb64e890d779cb466933ed60c23f`.
- Modal A10G response-only plan:
  `experiments/results/official_component_response_pfp16_direct_fd_modal_a10g_complete_response_only_20260501`.
  Plan SHA-256:
  `a460009d891a38287378164989ca8a417780a11f857ae998f5d724a5eaa310bf`.

Superseded artifacts:

- Earlier plan dirs without the `_response_only_` suffix contain invalid huge
  byte-response predictions derived from weight-space direct-FD maps. Do not
  submit those in `--require-passed` mode and do not use their prediction
  deltas for claims.

Lightning blocker:

- SSH alias `scratch-studio-devbox` is configured and offers key fingerprint
  `SHA256:af6xKc8r7y0WYc4FL6lGrvhHDT0qyo2gSruKhrk/c5Y`, but Lightning rejects
  it with `Permission denied (publickey)`. This is a Lightning-side
  account/Studio key authorization issue after the key is offered, not a repo
  config failure. User action required: add/reauthorize
  `~/.ssh/lightning_rsa.pub` in Lightning UI, then verify
  `ssh -o BatchMode=yes scratch-studio-devbox true`.

Verification:

- `py_compile` passed for touched direct-FD planning, perturbation,
  prediction, selector, and Lightning SSH config files.
- Focused tests passed: `27 passed` across component-response perturbation
  planning, sensitivity-ranked basis selection, and Lightning SSH config.
- MCP preflight clean; Lightning supply-chain scan clean and recorded at
  `.omx/state/lightning_supply_chain_local_codex_20260501T0939Z.json`;
  `git diff --check` clean.

## 2026-05-01T09:50Z Response-Only Calibration Dispatch

The clean mixed direct-FD response-only packet has been staged and dispatched
to Lightning Batch Jobs for official CUDA response measurement.

Staging:

- Manifest:
  `.omx/state/component_response_direct_fd_mixed_response_only_20260501T0950Z_manifest.json`.
- Manifest SHA-256:
  `ca98d9a707ea5be6da5a69c045e89563a0bcd18917eb88242048dc918aebd907`.
- File count/bytes: `1146` / `179556072`.
- Remote manifest verification: OK.
- Remote supply-chain scan: OK.
- Interactive Studio CUDA: still unavailable; Batch Job CUDA preflight remains
  the authority.

Jobs:

- `component_response_pfp16_direct_fd_mixed_response_only_t4_aws_20260501T0950Z`
  (`g4dn.2xlarge`) submitted, first refresh `Pending`.
- `component_response_pfp16_direct_fd_mixed_response_only_t4_aws_small_20260501T0950Z`
  (`g4dn.xlarge`) submitted, first refresh `Pending`.
- GCP `n1-standard-8` duplicate was rejected before creation because this
  Studio submitted against an AWS cluster.

Interpretation rule:

- These jobs intentionally run without `--require-passed` because the nonzero
  points have no archive-byte prediction deltas. The result should be used to
  fit/calibrate byte-basis response, not to claim a certified sensitivity map
  or score frontier by itself.
