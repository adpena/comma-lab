# Codex Findings: MLX Streaming Hash Cache Modal Bridge

UTC: 2026-05-21T21:12:00Z

## Verdict

PROCEED as the next contest-CPU-faithful bridge for local MLX training signal.

The previous MLX cache audit proved that local macOS advisory raw bytes are not
byte-identical to Modal Linux x86_64 contest-CPU raw bytes for the same archive.
This landing adds a compact Linux-side scorer-input identity artifact so Modal
auth-eval can emit exact scorer-input array hashes without returning multi-GB
raw files or NumPy tensors through the function result payload.

## What Landed

- Added streaming hash-only cache generation to
  `tac.local_acceleration.mlx_preprocess`.
- Added `--hash-only --batch-pairs` to
  `tools/build_mlx_scorer_input_cache.py`.
- Added `--scorer-input-cache-hashes-out` and
  `--scorer-input-cache-hash-batch-pairs` to
  `experiments/contest_auth_eval.py`.
- Added opt-in Modal CPU wrapper wiring via
  `experiments/modal_auth_eval_cpu.py`:
  `scorer_input_cache_hashes=True`.
- The Modal CPU artifact collector now harvests
  `scorer_input_cache_hashes.json` when requested.

## Empirical Anchor

Local FEC6/PR101 full-cache parity check:

```bash
env PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -p .venv/bin/python \
  tools/build_mlx_scorer_input_cache.py \
  --raw experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/local_macos_cpu_advisory_smoke_20260519T143700Z_workdir/inflated/0.raw \
  --output-dir experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T211200Z_hash_only_macos_full600 \
  --archive-sha256 6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf \
  --inflated-outputs-aggregate-sha256 dbc67c898ecb158912f86c920f09bf2c68307b77c1cec3c1baa27a845d3850f1 \
  --hash-only \
  --batch-pairs 16
```

Result:

- pair count: `600`
- real time: `3.47s`
- streamed hash-only array hashes matched the existing full-cache manifest
  exactly.

Matched array hashes:

- SegNet last RGB:
  `ea4cf2c4879fcdf4cd177cc4e3c762433aa076b631ce252947372cda4da37536`
- PoseNet YUV6 pair:
  `aae96b7cb270059174d987740a95e9fd0d9f4474142fd77ed1c1fce6a4124ed0`
- Pair indices:
  `b5d8a47e63045d3032bdc9da91c26e221e453a89f13c94049c6f5e850e49ba81`

The full tensor files remain ignored experiment artifacts. The landed contract
is the compact JSON hash manifest and the Modal opt-in hook.

## Authority Boundary

This artifact is not a score path. It carries:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`

Allowed use is identity and transfer calibration: prove that a local MLX cache,
surrogate dataset, or model input pipeline corresponds to the same scorer-input
surface produced by a target auth-eval axis.

## Verification

Commands run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_mlx_preprocess.py \
  src/tac/tests/test_mlx_cache_audit.py \
  src/tac/tests/test_mlx_scorer_fidelity.py -q
```

Result: `16 passed in 2.16s`.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m compileall -q \
  src/tac/local_acceleration/mlx_preprocess.py \
  tools/build_mlx_scorer_input_cache.py \
  experiments/contest_auth_eval.py \
  experiments/modal_auth_eval_cpu.py \
  src/tac/tests/test_mlx_preprocess.py
```

Result: pass.

```bash
git diff --check -- \
  src/tac/local_acceleration/mlx_preprocess.py \
  tools/build_mlx_scorer_input_cache.py \
  src/tac/tests/test_mlx_preprocess.py \
  experiments/contest_auth_eval.py \
  experiments/modal_auth_eval_cpu.py
```

Result: pass.

CLI visibility:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  tools/build_mlx_scorer_input_cache.py --help | rg 'hash-only|batch-pairs'
```

Result: flags visible.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/contest_auth_eval.py --help | rg 'scorer-input-cache'
```

Result: flags visible.

## Recommended Next Action

Run the existing Modal CPU auth-eval wrapper for the FEC6 archive with
`scorer_input_cache_hashes=True`, then audit the returned
`scorer_input_cache_hashes.json` against the contest-CPU auth-eval JSON. That
will produce the first byte-closed Linux scorer-input identity anchor for local
MLX transfer calibration.
