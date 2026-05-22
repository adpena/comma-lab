# Codex Findings: MLX Preprocess Parity + Cache-Audit Discovery

timestamp_utc: 2026-05-22T04:46:44Z
lane_id: lane_codex_mlx_auth_scorer_hardening_20260522
author: codex
verdict: PROCEED

## Scope

Hardened the local MLX scorer-input lane against two false-authority risks:

1. Local preprocessing drift from upstream `DistortionNet.preprocess_input`.
2. Operator discoverability gap between the natural command name `tools/mlx_cache_audit.py` and the canonical implementation `tools/audit_mlx_scorer_input_cache.py`.

## Findings

### F-1: Upstream preprocessing parity now has a direct fixture

`src/tac/tests/test_mlx_preprocess.py` now compares `preprocess_scorer_inputs_from_pairs(...)` against the real upstream preprocessing call path:

- `upstream/modules.py::DistortionNet.preprocess_input`
- `upstream/modules.py::PoseNet.preprocess_input`
- `upstream/modules.py::SegNet.preprocess_input`
- `upstream/frame_utils.py::rgb_to_yuv6`

The fixture uses `__new__` lightweight scorer shells so it exercises the upstream preprocessing methods without constructing the heavy scorer backbones or loading scorer weights.

Result: local MLX scorer-input cache generation is now guarded against shape/order/YUV/last-frame drift at the upstream preprocessing boundary.

### F-2: Cache-audit CLI has a stable operator alias

Added `tools/mlx_cache_audit.py` as a compatibility alias for `tools/audit_mlx_scorer_input_cache.py`.

Result: the natural short command path is now discoverable without creating a second implementation surface.

## Verification

Commands:

```bash
.venv/bin/python -m ruff check src/tac/tests/test_mlx_preprocess.py src/tac/tests/test_mlx_cache_audit.py tools/mlx_cache_audit.py
.venv/bin/python -m pytest src/tac/tests/test_mlx_preprocess.py src/tac/tests/test_mlx_cache_audit.py -q
.venv/bin/python -m pytest src/tac/tests/test_mlx_preprocess.py src/tac/tests/test_mlx_cache_audit.py src/tac/tests/test_mlx_scorer_fidelity.py src/tac/tests/test_auth_eval_schema.py -q
```

Observed:

- `ruff`: pass
- focused preprocess/cache-audit tests: `25 passed`
- broader MLX/auth-schema slice: `58 passed`

## Residual Risk

This is still a scorer-input/cache fidelity guard, not an MLX scorer-output parity claim. It does not authorize score claims, rank/kill decisions, promotion, or exact-eval replacement. The next production-hardening step remains end-to-end scorer-output parity on a byte-closed archive/auth-eval pair, with axis labels preserved.
