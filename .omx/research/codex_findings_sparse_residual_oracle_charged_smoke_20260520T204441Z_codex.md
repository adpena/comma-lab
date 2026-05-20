# Codex findings: charged sparse-residual oracle smoke

Date: 2026-05-20T20:44:41Z
Lane: `lane_sparse_residual_oracle_charged_smoke_20260520`
Axis: `[macOS-CPU advisory sparse-residual-oracle]`
Authority: advisory only; not a contest score claim; not promotion eligible.

## What landed

- Reusable module: `src/tac/optimization/sparse_residual_oracle.py`
- Operator tool: `tools/run_sparse_residual_oracle_smoke.py`
- Tests: `src/tac/tests/test_sparse_residual_oracle.py`
- Result packet: `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/sparse_residual_oracle_charged_smoke_20260520_codex/sparse_residual_oracle_k256_d1_20260520_codex.json`
- Correction payload: `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/sparse_residual_oracle_charged_smoke_20260520_codex/k256_d1_all/sparse_residual_corrections.bin`

The implementation reuses the canonical engineered-corrections sparse binary grammar via `tac.engineered_corrections.pack_sparse_corrections`. It does not create a competing sidecar format.

## Smoke command

```bash
.venv/bin/python tools/run_sparse_residual_oracle_smoke.py \
  --baseline-raw experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/op3v3_decoder_q_inflate_controls_20260520_codex/baseline/inflated/0.raw \
  --target-raw experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/sparse_residual_oracle_charged_smoke_20260520_codex/target/0.raw \
  --archive experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/runtime_hfv1/archive.zip \
  --output-root experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/sparse_residual_oracle_charged_smoke_20260520_codex \
  --output experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/sparse_residual_oracle_charged_smoke_20260520_codex/sparse_residual_oracle_k256_d1_20260520_codex.json \
  --top-k-pixels 256 \
  --max-abs-delta 1 \
  --chunk-frames 16 \
  --rate-cap-bytes 4096 \
  --baseline-score 0.19206142414659494 \
  --run-advisory \
  --cleanup-candidate-raw \
  --timeout 1800
```

Target raw was decoded from `upstream/videos/0.mkv` with:

```bash
ffmpeg -hide_banner -loglevel error -y -i upstream/videos/0.mkv -f rawvideo -pix_fmt rgb24 .../target/0.raw
```

The decoded target raw SHA-256 was `4f1ca43f44f3a7c83e78162cbe5c82d845416e7b9496b6ba743fdb64ee67b23a`. The 3.4 GiB raw was deleted after the smoke because it is rebuildable from the pinned upstream video; the compact JSON result retains the hash and decode command.

## Empirical result

Baseline advisory packet:

- Score: `0.19206142414659494`
- PoseNet distortion: `2.943e-05`
- SegNet distortion: `0.00056039`
- Archive bytes: `178517`
- Rate: `0.00475469`

Sparse residual oracle candidate:

- Selected pixels: `256`
- Changed pixels: `256`
- Changed bytes in raw: `768`
- Changed frames: `2`
- Packed correction bytes: `687`
- Charged proxy archive bytes: `179204`
- PoseNet distortion: `2.943e-05`
- SegNet distortion: `0.00056039`
- Rate: `0.00477298`
- Score: `0.19251867414659496`
- Delta vs baseline: `+0.00045725000000002014`

## Interpretation

This k256 raw-error oracle produced a component-null result: PoseNet and SegNet did not move at report precision. The entire score regression is the rate term from 687 charged bytes. This is useful because it prevents an uncharged sparse-pixel optimism trap: sparse pixel nudges can be made visible, packed, and charged, but raw-error top-k selection is not high-signal enough at this byte scale.

## Decision

Do not widen this exact raw-error top-k path by spending exact CUDA eval. Keep the reusable tool, but require one of the following before the next sparse residual smoke:

- scorer/pixel sensitivity targeting, not raw RGB error targeting;
- per-axis target selection using PoseNet-vs-SegNet decomposition;
- a real inflate runtime consumer that proves the correction bytes are consumed by stock inflate;
- a byte-economics threshold showing predicted component savings greater than `25 * packed_bytes / original_uncompressed_bytes`.

Lane registry was updated as research-only with this reactivation criterion:

`widen only with scorer/pixel sensitivity targeting or real inflate consumer; raw-error top-k k256 produced component-null +0.00045725 rate-only regression`

## Verification

- `.venv/bin/python -m pytest -q src/tac/tests/test_sparse_residual_oracle.py src/tac/tests/test_inflate_postprocess_surface.py` -> `8 passed`
- `.venv/bin/python -m py_compile src/tac/optimization/sparse_residual_oracle.py tools/run_sparse_residual_oracle_smoke.py` -> pass
- `.venv/bin/python tools/lane_maturity.py validate` -> `1064 lane(s) validated cleanly`
