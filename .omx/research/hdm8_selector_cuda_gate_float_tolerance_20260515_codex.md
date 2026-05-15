# HDM8 Selector CUDA Gate Float Tolerance - 2026-05-15

## Scope

Fix a false blocker in the HDM8 selector CUDA component-risk gate.

## Failure

The full 600-pair CUDA-derived selector archive had strong charged proxy
evidence:

- archive sha256: `34dc94644f5619ea7e6254079e3e4d3bbf0952f8a0ad287f675f7a249f359071`
- archive bytes: `187226`
- selector payload: `829` brotli bytes
- charged CUDA-prefix delta: `-0.005772186805473756`
- pose delta: `-4.723334091825866e-05`
- SegNet delta: `+1.067140516577969e-12`

The gate blocked solely because the SegNet delta was `1.07e-12` above the
zero threshold. This is floating aggregation noise; it is far below scorer
resolution and not a component regression.

## Fix

`src/tac/hdm8_selector_cuda_gate.py` now applies an explicit `1e-10`
floating tolerance when comparing pose, SegNet, and score deltas to their
thresholds. This preserves fail-closed behavior for real regressions while
preventing sub-ulp aggregation noise from blocking exact-eval dispatch.

Regression:

- `src/tac/tests/test_hdm8_selector_cuda_gate.py::test_cuda_prefix_selector_gate_tolerates_float_noise_on_neutral_seg`

Verification:

```bash
.venv/bin/ruff check src/tac/hdm8_selector_cuda_gate.py src/tac/tests/test_hdm8_selector_cuda_gate.py tools/build_hdm8_film_grain_sidecar_packet.py src/tac/tests/test_hdm8_film_grain_sidecar.py
PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest src/tac/tests/test_hdm8_selector_cuda_gate.py src/tac/tests/test_hdm8_film_grain_sidecar.py -q
```

Result:

- ruff: passed
- pytest: `23 passed`

## Effect

The full 600-pair CUDA selector gate passes on the candidate above. It is still
not a score claim; promotion requires exact contest-CUDA auth eval of the
byte-closed archive/runtime packet.
