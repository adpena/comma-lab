# Renderer Self-Compression Nextwave Worker - 2026-05-03

Score claim: false. Promotion eligible: false. Remote GPU dispatch performed: false. Dispatch claim opened: false.

## Source C-101
- Archive: `experiments/results/lightning_batch/exact_eval_c091_native_pose_manifold_top128_s025_t4_20260503T122013Z/archive.zip`
- Bytes/SHA: `276489` / `1c9be2dd14b2607b9a86ecc071d6a37842896d18d62449885dac58d55afbbd64`
- Exact CUDA score: `0.3151520345392486`; SegNet `0.00060804`; PoseNet `0.00049344`; samples `600`; device `cuda`; GPU `Tesla T4`.
- Runtime tree SHA-256: `29206234a663100221ff47733605063185e17005a768ff0dcfa81e64ef571d92`

## Candidate
- Candidate ID: `f1_0135_f2block006_all003`
- Archive: `/Users/adpena/Projects/pact/experiments/results/renderer_selfcompression_nextwave_worker_20260503/c101_triple_zero_screen/f1_0135_f2block006_all003/archive.zip`
- Bytes/SHA: `275687` / `649e1c51111386e060a1ff20d11b4020f1afa589e97cfe2aac04ae9ad0a014a5`
- Delta vs C-101: `-802` bytes; unchanged-component score projection `0.3146180156588446`.
- Sub-0.314 byte target: needs `1731` saved bytes; this candidate saves `802`, so it is diagnostic exact-eval-ready, not a sub-0.314 dispatch recommendation.
- Pose-safety report: `experiments/results/renderer_selfcompression_nextwave_worker_20260503/c101_triple_zero_screen/f1_0135_f2block006_all003/pose_safety_preflight.json`
- Pose-safety: `safe_for_exact_eval_dispatch=true`; sampled pairs `[0, 150, 300, 449, 599]`; thresholds mean `<=3.0`, RMS `<=8.0`, max `<=80.0`.
- Output deltas: mean `2.963364362716675`, RMS `4.4585012006202955`, max `73.02064514160156`.
- Source/candidate archive SHA pair: `1c9be2dd14b2607b9a86ecc071d6a37842896d18d62449885dac58d55afbbd64` -> `649e1c51111386e060a1ff20d11b4020f1afa589e97cfe2aac04ae9ad0a014a5`.
- No-op checks: renderer changed, archive SHA changed, archive bytes changed, `3588` QZS3 values changed, non-renderer members preserved, runtime unpack verified, `transform_is_noop=false`.
- Charged bytes: archive `275687`; payload `p` `275587`; slices mask `219472`, renderer `54954`, actions `255`, pose `896`.

## Negative Boundaries
- Single-transform best local-safe: `zero_fp4_frame1_head_0.135`, `275784` bytes, `705` saved.
- Frame1 boundary: `0.14` saved `788` bytes but failed pose-safety (`max_abs=85.814453125 > 80`).
- Best combined local-safe before triples: `f1_0135_f2block006`, `275700` bytes, `789` saved.
- Triple attempts above `802` saved bytes failed on mean/max output deltas; best byte-only triple was `831` saved and unsafe.
- Naive QZS3 reblocks remain unsafe by wide margins in the existing C091 screen.

## Modal Surfaces
- Read local mirrored recovery artifacts only; no live Modal mutation, cancel, or new dispatch.
- Local triage lists H100/A100/A10G calls as still running with zero saved artifacts. Log tails only show source copy and lane start lines.

## Exact CUDA Command
Run only on a CUDA host. This is a diagnostic exact-eval-ready command, not a remote dispatch recommendation:

```bash
mkdir -p experiments/results/renderer_selfcompression_nextwave_worker_20260503/exact_eval_c101_f1_0135_f2block006_all003_cuda/work && \
.venv/bin/python experiments/contest_auth_eval.py \
--archive experiments/results/renderer_selfcompression_nextwave_worker_20260503/c101_triple_zero_screen/f1_0135_f2block006_all003/archive.zip \
--inflate-sh submissions/robust_current/inflate.sh \
--upstream-dir upstream \
--device cuda \
--work-dir experiments/results/renderer_selfcompression_nextwave_worker_20260503/exact_eval_c101_f1_0135_f2block006_all003_cuda/work \
--expected-runtime-tree-sha256 29206234a663100221ff47733605063185e17005a768ff0dcfa81e64ef571d92
```

Optional adjudication after the CUDA run:

```bash
.venv/bin/python scripts/adjudicate_contest_auth_eval.py \
--contest-json experiments/results/renderer_selfcompression_nextwave_worker_20260503/exact_eval_c101_f1_0135_f2block006_all003_cuda/work/contest_auth_eval.json \
--provenance experiments/results/renderer_selfcompression_nextwave_worker_20260503/exact_eval_c101_f1_0135_f2block006_all003_cuda/work/provenance.json \
--archive experiments/results/renderer_selfcompression_nextwave_worker_20260503/c101_triple_zero_screen/f1_0135_f2block006_all003/archive.zip \
--result-copy experiments/results/renderer_selfcompression_nextwave_worker_20260503/exact_eval_c101_f1_0135_f2block006_all003_cuda/contest_auth_eval.adjudicated.json \
--baseline-score 0.3151520345392486 \
--baseline-archive-bytes 276489 \
--predicted-band 0.314 0.318 \
--regression-threshold 0.004 \
--delta-key score_delta_vs_c101 \
--max-posenet-dist 0.005 \
--max-segnet-dist 0.005 \
--baseline-posenet-dist 0.00049344 \
--baseline-segnet-dist 0.00060804 \
--required-device cuda \
--required-samples 600 \
--max-sane-score 1.0
```

## Verification
- `.venv/bin/python -m py_compile experiments/results/renderer_selfcompression_nextwave_worker_20260503/c101_combined_zero_search.py` passed.
- `.venv/bin/python -m pytest src/tac/tests/test_search_renderer_parity_shrink_candidate.py src/tac/tests/test_preflight_renderer_transplant_pose_safety.py -q` passed: `10 passed`.
