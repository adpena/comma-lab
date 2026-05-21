# Codex Findings: MLX CUDA Scorer-Input Hash Bridge

UTC: 2026-05-21T21:31:00Z

## Verdict

PROCEED. The CUDA Modal auth-eval wrapper now has parity with the CPU wrapper
for emitting scorer-input hash artifacts.

## What Landed

- `experiments/modal_auth_eval.py` now accepts:
  - `scorer_input_cache_hashes: bool = False`
  - `scorer_input_cache_hash_batch_pairs: int = 8`
- When requested, the CUDA wrapper passes these through to
  `experiments/contest_auth_eval.py` as:
  - `--scorer-input-cache-hashes-out <work_dir>/scorer_input_cache_hashes.json`
  - `--scorer-input-cache-hash-batch-pairs <N>`
- The CUDA artifact collector now harvests
  `scorer_input_cache_hashes.json`.
- Local request, spawn metadata, validation, and result JSON surfaces record the
  hash-request flags.

## Authority Boundary

This is not a score path and does not alter the evaluator score. It adds an
optional scorer-input identity side artifact so MLX/local training datasets can
be calibrated against the exact contest-CUDA raw/scorer-input surface before
they are trusted for surrogate training or transfer predictions.

The next paid run should request this artifact on the target PR110/FEC6 archive
and then feed it into `tools/audit_mlx_scorer_input_cache.py` under
`scorer_input_cache_hash_identity_v1`.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_modal_auth_eval.py -q
```

Result: `40 passed in 0.45s`.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m compileall -q \
  experiments/modal_auth_eval.py \
  src/tac/tests/test_modal_auth_eval.py
```

Result: pass.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py \
  policy-check experiments/modal_auth_eval.py src/tac/tests/test_modal_auth_eval.py
```

Result: `0 violations`.

```bash
git diff --check -- \
  experiments/modal_auth_eval.py \
  src/tac/tests/test_modal_auth_eval.py
```

Result: pass.
