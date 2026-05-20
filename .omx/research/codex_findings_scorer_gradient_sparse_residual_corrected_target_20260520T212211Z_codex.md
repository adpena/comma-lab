# Codex Findings: Scorer-Gradient Sparse Residual Corrected-Target Smoke

Generated: 2026-05-20T21:22:11Z
Author: Codex
Axis: [macOS-CPU advisory], score_claim=false

## Verdict

Backprop is a valid compress-time optimization tool, but the first hard-pair
residual smoke is not yet a frontier candidate. The corrected upstream-target
run improved the local PoseNet pair objective while still regressing full-video
advisory score, which means a widened lane needs a full-response surrogate and
byte-budget allocator rather than pair-local gradient ranking alone.

## Empirical finding

Baseline:

- Raw: `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/op3v3_decoder_q_inflate_controls_20260520_codex/baseline/inflated/0.raw`
- Advisory score: `0.19206142414659494`

Corrected target decode:

- Tool: `tools/decode_upstream_video_to_raw.py`
- Semantics: `upstream.frame_utils.yuv420_to_rgb` via PyAV, matching `upstream/evaluate.py` CPU `AVVideoDataset`
- Target manifest: `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/scorer_gradient_sparse_residual_smoke_20260520_codex/target/upstream_av_0_manifest.json`
- Target raw SHA-256: `bb9cb031acc7d9898d28618d49b256cc9f2e9cc92a25327acd4a9061f6565907`
- Prior ffmpeg target raw SHA-256: `4f1ca43f44f3a7c83e78162cbe5c82d845416e7b9496b6ba743fdb64ee67b23a`

Corrected smoke result:

- Result JSON: `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/scorer_gradient_sparse_residual_smoke_20260520_codex/scorer_gradient_pose_pair508_k512_d1_upstream_target_20260520_codex.json`
- Component: `pose`
- Pair: `508`
- Changed pixels: `512`
- Changed raw bytes: `1535`
- Changed frames: `2`
- Packed residual bytes: `1997`
- Charge-proxy archive bytes: `180514`
- Charge-proxy archive SHA-256: `8d144704eb597e38cda35713b15ee6ebdeea6a3a9ea600d9082a758a199d90f9`
- Advisory score: `0.19339092414659495`
- Delta vs baseline: `+0.0013295000000000112`
- Local pair 508 before pose: `5.236185825197026e-05`
- Local pair 508 after pose: `4.91187020088546e-05`
- Local pair pose delta: `-3.2431562431156635e-06`
- Local pair SegNet delta: `0.0`

## Interpretation

The corrected run falsifies the naive "pair-local PoseNet gradient is enough"
premise. The perturbation descends the local pair PoseNet objective, yet the
full advisory score regresses. The likely issue is objective mismatch: local
pair loss ignores full-video averaging, rate charge, SegNet boundary fragility,
decode semantics, and small scorer/numerical differences across the complete
evaluation loop.

This supports the LL lane:

- Train or fit a scorer-response surrogate at compress time.
- Use Hinton-style soft targets/logits for differentiable SegNet signal instead
  of relying on hard argmax alone.
- Allocate residual candidates by predicted full-score improvement per byte.
- Require held-out full-advisory correlation before widening candidate spend.
- Keep all scorer/surrogate use compress-time only. The live `inflate.py`
  remains scorer-free unless a separate contest-compliant archive grammar is
  explicitly built and reviewed.

## Code landings

- Added target decode helper: `tools/decode_upstream_video_to_raw.py`.
- Hardened `tools/run_scorer_gradient_sparse_residual_smoke.py` so candidate
  directories include a target-raw SHA-derived slug and do not silently
  overwrite non-empty prior runs without `--overwrite-candidate`.
- Added `select_budgeted_gradient_residuals(...)` in
  `src/tac/optimization/scorer_gradient_sparse_residual.py`; it accepts any
  gradient source, saliency mask, per-pixel byte costs, and a budget limit.
  This is the reusable backprop/water-fill primitive for LL.
- Added tests in `src/tac/tests/test_scorer_gradient_sparse_residual.py`.

## Verification

- `.venv/bin/python -m pytest -q src/tac/tests/test_scorer_gradient_sparse_residual.py src/tac/tests/test_sparse_residual_oracle.py`
  - `8 passed`
- `.venv/bin/python -m py_compile tools/run_scorer_gradient_sparse_residual_smoke.py tools/decode_upstream_video_to_raw.py src/tac/optimization/scorer_gradient_sparse_residual.py src/tac/optimization/sparse_residual_oracle.py`
  - passed

## Next action

Proceed with `lane_ll_hinton_distilled_scorer_saliency_residual_20260520` as a
local `$0` optimization lane, but do not claim score movement from it until:

1. It produces a held-out surrogate-response correlation against full advisory
   candidates.
2. The residual payload is consumed through a real inflate/runtime path or is
   kept explicitly advisory-only.
3. Full-video advisory improves after charging bytes.
4. Contest CUDA exact eval is run before any promotion language.
