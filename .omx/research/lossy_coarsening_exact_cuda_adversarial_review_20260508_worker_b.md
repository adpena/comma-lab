# Lossy Coarsening Exact CUDA Adversarial Review - Worker B - 2026-05-08

## Scope

Review target:
`experiments/results/lightning_batch/lossy-coarsening-cuda-20260508T0312-noproject/auth_eval_work/contest_auth_eval.json`

Proxy-positive sources checked:

- `reports/cathedral_autopilot_evidence.jsonl`
- `reports/raw/pr101_lossy_coarsening_20260508T011743Z/manifest.json`
- `experiments/results/lossy_coarsening_20260508T030750Z/build_manifest.json`

This is not a broad lossy-coarsening family kill. It is an exact negative for
the measured PR101 direct decoder coarsening configuration:
`per_tensor_K_budget=0.05` / `rel_err_actual_int8=0.03855950900557584`.

## Classification

The exact CUDA eval is valid score evidence for this archive, but it is a
score-negative outcome.

- Evidence artifact grade reported by eval tool: `A++`
- Scientific classification: `A-negative` for this measured configuration
- Reason: canonical exact score is `0.351718793322788`, worse than the PR101
  exact replay anchor `0.22635331443973267` by `+0.125365478883055`
- Do not promote, rank as a frontier win, or use the proxy row's predicted band
  for scoring without recalibration
- Do not mark `lossy_coarsening` as family-falsified; only this direct PR101
  `0.05` budget K-coarsening test is falsified as a score improvement

## Exact Fields Checked

From `contest_auth_eval.json`:

- `score_recomputed_from_components=0.351718793322788`
- `canonical_score=0.351718793322788`
- `canonical_score_source=score_recomputed_from_components`
- `final_score=0.35`
- `reported_final_score_display_rounded=0.35`
- `score_rounding_abs_delta=0.0017187933227880148`
- `avg_posenet_dist=0.00037762`
- `avg_segnet_dist=0.00186125`
- `score_pose_contribution=0.061450793322787946`
- `score_seg_contribution=0.186125`
- `score_rate_contribution=0.10414300000000001`
- `archive_size_bytes=156404`
- `n_samples=600`
- `device=cuda`
- `gpu_model=Tesla T4`
- `gpu_t4_match=true`
- `cuda_available=true`
- `inflate_elapsed_seconds=72.82378337599994`
- `evaluate_elapsed_seconds=40.32066902500014`
- `upstream_commit=11ad728f563d8970929e8947a1cf6124ee6303e4`
- `pact_commit=<error:CalledProcessError(128, ['git', 'rev-parse', 'HEAD'])>`

Component recomputation:

- Stored contribution sum:
  `0.061450793322787946 + 0.186125 + 0.10414300000000001 =
  0.351718793322788`
- Delta against `score_recomputed_from_components`: `0`
- Formula recomputation from rounded component distances gives
  `0.351718797026908`, delta `~3.7e-09`; this is only rounding in the displayed
  `avg_*` fields.

## Archive And Runtime Custody

Archive checked:
`experiments/results/lightning_batch/lossy-coarsening-cuda-20260508T0312-noproject/auth_eval_work/archive.zip`

- Size by `stat`: `156404`
- SHA-256 by `shasum`: `ab8a8a13c70b3d3bbf2ce3d8a81a77691b776a6e0fb1cbe9ce504dc3f59c1b28`
- `cmp` against `experiments/results/lossy_coarsening_20260508T030750Z/archive.zip`: identical
- ZIP structure by `zipinfo -v`: one stored member `x`, no comment, no extra
  field, central directory one entry
- Member `x`: `156304` bytes, CRC `733a7865`, SHA-256
  `ae67d4a5ee6f2228975c1385ac46aa9a190fc7dbc10bddda83f751a25f708660`
- Inflated output `0.raw`: `3662409600` bytes, SHA-256
  `8c80b17b54dce97ecc107f32099e1ee096aa43cd5343a15231574dc6a1f749c8`

Build manifest alignment:

- `archive_bytes=156404`
- `archive_sha256=ab8a8a13c70b3d3bbf2ce3d8a81a77691b776a6e0fb1cbe9ce504dc3f59c1b28`
- `rel_err_budget=0.05`
- `rel_err_actual_int8=0.03855950900557584`
- `rel_err_actual_fp32_smoke=0.03481125033136704`
- `max_per_tensor_rel_err_fp32_smoke=0.04973323787800202`
- `section_brotli_payload_bytes=140222`
- `section_total_bytes=140310`
- `n_tensors=28`
- `n_symbols=228958`

Runtime manifest from provenance:

- `runtime_file_count=4`
- `runtime_tree_sha256=d55ed9a31ab76a2498fdce98ddb5852544c504ad166fd68df1323f341ca4b3e7`
- `external_dependency_roots=[]`
- repo-local `tac` import closure: `module_count=0`, `file_count=0`,
  `unresolved_modules=[]`, `parse_errors=[]`
- `upstream/evaluate.py` SHA-256:
  `7da71a84ce24286bc6b583470f9bbd25c998971da301320d0d4e9d6fd40baa4b`
- Runtime files:
  - `inflate.py`: `7154` bytes,
    `bf7d1b4b1327262a65e74f211e40d3d11b7071ed76a51828c0439e072b36a909`
  - `inflate.sh`: `1687` bytes,
    `0f7ff27300780162353109a2afee2b2c6f0bcee25725a6aff3d09303424661f8`
  - `src/codec.py`: `16734` bytes,
    `637fa5e4b47bfb2595358903dfaafbbc7a7ca48d5f7ccd81c06d1397e060bb72`
  - `src/model.py`: `2197` bytes,
    `e63b04ad3df4942b9bc1e31afd8ec84177dfbe83827f67cf7c5a682b05c1b46b`

## Proxy Comparison

Earlier raw proxy manifest:

- `evidence_grade=[MPS-research-signal]`
- `score_claim=false`
- `promotion_eligible=false`
- Best proxy config: `per_tensor_K_budget=0.05`
- Proxy archive bytes: `156344`
- Proxy rel_err: `0.03856566284611934`
- Proxy byte delta vs PR101 brotli baseline: `-21800`

Exact byte-closed build:

- Exact archive bytes: `156404`, not `156344`
- The `+60` bytes are expected from the runtime format adding a `uint32`
  decoder-section prefix and `28 * fp16` scale side-info
- This 60-byte delta is not the failure driver

Harvested `cathedral_autopilot_evidence.jsonl` row:

- `score_contest_cuda=0.351718793322788`
- `empirical_archive_bytes=156404`
- `archive_sha256=ab8a8a13c70b3d3bbf2ce3d8a81a77691b776a6e0fb1cbe9ce504dc3f59c1b28`
- `archive_bytes_match_expected=true`
- `archive_sha256_match_expected=true`
- `predicted_band=[0.18,0.22]`
- `score_claim=false`
- `promotion_eligible=false`

Adversarial verdict: the byte proxy was directionally right about rate savings
but wrong about distortion cost. The exact run saved rate contribution
(`0.104143` vs PR101's `~0.118695`, about `-0.01455`) while worsening SegNet
and PoseNet enough to dominate the savings. Using the rounded PR101 replay
components from the public-frontier ledger, the coarsened candidate has:

- Seg contribution: `0.186125` vs PR101 `0.066304`, about `+0.119821`
- Pose contribution: `0.061450793322787946` vs PR101 `0.041355`, about
  `+0.020096`
- Rate contribution: `0.104143` vs PR101 `0.118695`, about `-0.014552`

The proxy objective used decoder-symbol relative error, not score-aware
distortion. That proxy failed to price SegNet/PoseNet sensitivity.

## Likely Failure Modes

1. Per-tensor int8-symbol L1 relative error is not a reliable scorer proxy.
   A `3.856%` decoder-symbol perturbation is small in the proxy but large
   enough to shift decoded frames into substantially worse SegNet and PoseNet
   regions.

2. The K-search is byte-aware but not component-aware. It treats tensors by
   local relative reconstruction error rather than downstream score gradient,
   layer sensitivity, frame-region sensitivity, or pose/mask sensitivity.

3. The `0.05` budget sits past the trust region for direct PR101 decoder
   coarsening. The `0.02` and `0.03` proxy rows are not tested by CUDA here and
   should remain open, but their byte upside is much smaller.

4. Direct post-hoc coarsening lacks recovery. There is no retraining,
   distillation, layer repair, latent recentering, or score-aware fine-tune
   after the coarsening perturbation.

5. The predicted score band `[0.18, 0.22]` assumed distortion would not blow up
   enough to cancel byte savings. Exact CUDA falsified that predictor for this
   configuration.

## Reactivation Criteria

Do not redispatch the same `budget=0.05` direct PR101 K-coarsening packet.
Reactivate the family only when at least one of these changes is present:

- A budget ladder with byte-closed runtime packets at `0.01`, `0.02`, and/or
  `0.03`, with explicit expected score breakeven and exact CUDA queued only
  when the rate win can plausibly exceed component risk.
- Score-aware per-tensor allocation using CUDA component sensitivity, not
  uniform per-tensor relative-error budgets.
- Layer whitelist/blacklist from exact component sensitivity, especially for
  early/high-leverage decoder layers.
- Fine-tune, QAT, or distillation after coarsening, with the same charged-bit
  proof and exact CUDA auth-eval path.
- A stacked candidate where lossy coarsening is only applied to tensors with
  proven low scorer sensitivity and the packet still records old/new archive
  SHA-256 and charged-byte accounting.
- A recalibrated predictor that uses this exact negative as a calibration
  point and refuses score-band claims from rel_err alone.

## Unresolved Risks

- `contest_auth_eval.json` records `pact_commit` as a git command error. This
  does not change the measured CUDA result, but promotion/paper custody should
  be supplemented with the exact repo commit or rerun in a git-visible
  workspace before any public claim.
- The harvested JSONL `source` string points to
  `experiments/results/lightning_batch/lossy-coarsening-cuda-20260508T0312-noproject/contest_auth_eval.json`,
  but the file actually present is under `auth_eval_work/contest_auth_eval.json`.
  The evidence row is semantically correct but has a stale path.
- The source `submission_dir` currently contains `__pycache__` files on disk.
  The runtime manifest used by exact eval lists only four source files, but a
  release packet should rerun strict compliance and public hygiene before
  upload.
- This review did not rerun CUDA evaluation; it is an adversarial read-only
  review of the landed artifact.
